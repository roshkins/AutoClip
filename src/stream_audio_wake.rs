use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use url::Url;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const SAMPLE_RATE: usize = 16_000;
const CHUNK_MS: usize = 500; // read cadence for responsiveness
const WINDOW_MS: usize = 8_000; // transcription window length
const STEP_MS: usize = 1_000; // inference cadence

/// Common wake loop; caller provides how to spawn an ffmpeg PCM source (stream or mic).
fn run_wake_loop_with_spawn<Spawn>(
    mut spawn_pcm: Spawn,
    model_path: &Path,
    wake_norm: &str,
    log_raw: bool,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
) -> Result<()>
where
    Spawn: FnMut() -> Result<(Child, Box<dyn Read + Send>)> + Send + 'static,
{
    let ctx = WhisperContext::new_with_params(
        model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?,
        WhisperContextParameters::default(),
    )
    .context("loading whisper model")?;
    let mut state = ctx.create_state().context("creating whisper state")?;

    let chunk_samples = SAMPLE_RATE * CHUNK_MS / 1000;
    let chunk_bytes = chunk_samples * 2; // s16le
    let window_samples = SAMPLE_RATE * WINDOW_MS / 1000;
    let step = Duration::from_millis(STEP_MS as u64);

    let mut buf = vec![0i16; chunk_samples];
    let mut pcm: Vec<f32> = Vec::with_capacity(window_samples);

    let mut failures = 0usize;

    while !stop.load(Ordering::Relaxed) {
        let (mut ffmpeg, mut pcm_reader) = match spawn_pcm() {
            Ok(p) => p,
            Err(err) => {
                eprintln!("failed to start ffmpeg for wake audio: {err:#}");
                std::thread::sleep(Duration::from_millis(200));
                failures = failures.saturating_add(1);
                if failures >= 8 {
                    anyhow::bail!("audio capture failed repeatedly; set MIC_DEVICE to your input (Windows: run 'ffmpeg -list_devices true -f dshow -i dummy' and use audio=...)");
                }
                continue;
            }
        };

        pcm.clear();
        let mut last_run = Instant::now();

        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            if let Err(err) = read_exact_i16(&mut *pcm_reader, &mut buf, chunk_bytes) {
                eprintln!("ffmpeg read error or EOF: {err:#}; restarting capture");
                failures = failures.saturating_add(1);
                if failures >= 8 {
                    anyhow::bail!("audio capture failed repeatedly; set MIC_DEVICE to your input (Windows: run 'ffmpeg -list_devices true -f dshow -i dummy' and use audio=...)");
                }
                break;
            }
            failures = 0;

            for s in &buf {
                pcm.push(*s as f32 / 32768.0);
            }
            if pcm.len() > window_samples {
                let drop = pcm.len() - window_samples;
                pcm.drain(0..drop);
            }

            if last_run.elapsed() < step || pcm.len() < SAMPLE_RATE {
                continue;
            }
            last_run = Instant::now();

            if let Some(text) = transcribe_window(&mut state, &pcm)? {
                let norm = normalize(&text);
                if log_raw && !norm.is_empty() {
                    println!("stream raw: {}", text.trim());
                    println!("stream norm: {norm}");
                }
                if norm.contains(wake_norm) {
                    let already = fired.swap(true, Ordering::Relaxed);
                    if !already {
                        eprintln!("wake phrase detected via stream audio: {norm}");
                    }
                }
            }
        }

        let _ = ffmpeg.kill();

        if fired.load(Ordering::Relaxed) || stop.load(Ordering::Relaxed) {
            break;
        }

        std::thread::sleep(Duration::from_millis(200));
    }

    Ok(())
}

/// Listen to stream audio (HLS) via ffmpeg, run Whisper locally, and fire when the wake phrase is detected.
pub fn start_stream_wake_from_hls(
    media_url: &Url,
    model_path: &Path,
    wake_phrase: &str,
    log_raw: bool,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
) -> Result<()> {
    let media_url = media_url.clone();
    let model_path = model_path.to_path_buf();
    let wake = normalize(wake_phrase);
    std::thread::spawn(move || {
        if let Err(err) = run_wake_loop_with_spawn(
            move || spawn_ffmpeg_pcm(&media_url),
            &model_path,
            &wake,
            log_raw,
            stop,
            fired,
        ) {
            eprintln!("stream wake loop error: {err:#}");
        }
    });
    Ok(())
}

/// Listen to microphone audio via ffmpeg, run Whisper locally, and fire when the wake phrase is detected.
pub fn start_mic_wake_with_ffmpeg(
    mic_device: Option<&str>,
    model_path: &Path,
    wake_phrase: &str,
    log_raw: bool,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
) -> Result<()> {
    let mic = mic_device.map(|s| s.to_string());
    let model_path = model_path.to_path_buf();
    let wake = normalize(wake_phrase);
    std::thread::spawn(move || {
        if let Err(err) = run_wake_loop_with_spawn(
            move || spawn_ffmpeg_pcm_mic(mic.as_deref()),
            &model_path,
            &wake,
            log_raw,
            stop,
            fired,
        ) {
            eprintln!("mic wake loop error: {err:#}");
        }
    });
    Ok(())
}

fn transcribe_window(state: &mut whisper_rs::WhisperState, audio: &[f32]) -> Result<Option<String>> {
    let mut params = FullParams::new(SamplingStrategy::default());
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_translate(false);
    params.set_no_timestamps(true);
    params.set_single_segment(true);
    params.set_language(Some("en"));
    params.set_no_speech_thold(0.6);

    state.full(params, audio).context("running whisper")?;
    let num_segments = state.full_n_segments().context("segment count")?;
    for i in 0..num_segments {
        let text = state.full_get_segment_text(i).context("segment text")?;
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return Ok(Some(trimmed.to_string()));
        }
    }
    Ok(None)
}

fn spawn_ffmpeg_pcm(media_url: &Url) -> Result<(Child, Box<dyn Read + Send>)> {
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-nostdin")
        .arg("-i")
        .arg(media_url.as_str())
        .arg("-vn")
        .arg("-f")
        .arg("s16le")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg(format!("{}", SAMPLE_RATE))
        .arg("-")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null());

    let mut child = cmd.spawn().context("spawning ffmpeg for stream audio")?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("ffmpeg stdout missing"))?;
    Ok((child, Box::new(stdout)))
}

fn spawn_ffmpeg_pcm_mic(device: Option<&str>) -> Result<(Child, Box<dyn Read + Send>)> {
    let mut candidates = Vec::new();
    if let Some(d) = device {
        candidates.push(d.to_string());
    } else {
        candidates.extend(default_mic_candidates());
        candidates.extend(list_system_mics());
    }

    let mut last_err: Option<anyhow::Error> = None;
    for input in candidates {
        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-nostdin");
        cmd.arg("-loglevel").arg("warning");
        cmd.arg("-thread_queue_size").arg("4096");

        #[cfg(target_os = "windows")]
        {
            cmd.arg("-f").arg("dshow").arg("-rtbufsize").arg("10M").arg("-i").arg(&input);
        }
        #[cfg(target_os = "macos")]
        {
            cmd.arg("-f").arg("avfoundation").arg("-i").arg(&input);
        }
        #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
        {
            cmd.arg("-f").arg("pulse").arg("-i").arg(&input);
        }

        cmd.arg("-vn")
            .arg("-f")
            .arg("s16le")
            .arg("-ac")
            .arg("1")
            .arg("-ar")
            .arg(format!("{}", SAMPLE_RATE))
            .arg("-")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .stdin(Stdio::null());

        match cmd.spawn().with_context(|| format!("spawning ffmpeg for mic audio using input='{input}' (set MIC_DEVICE to override)")) {
            Ok(mut child) => {
                if let Some(stdout) = child.stdout.take() {
                    return Ok((child, Box::new(stdout)));
                } else {
                    let _ = child.kill();
                    last_err = Some(anyhow::anyhow!("ffmpeg stdout missing for input '{input}'"));
                }
            }
            Err(err) => {
                last_err = Some(err);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no mic candidates succeeded; set MIC_DEVICE to a valid input")))
}

fn default_mic_candidates() -> Vec<String> {
    #[cfg(target_os = "windows")]
    {
        return vec!["audio=default".to_string(), "audio=virtual-audio-capturer".to_string()];
    }
    #[cfg(target_os = "macos")]
    {
        return vec![":0".to_string(), ":1".to_string()]; // first/second input
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        return vec!["default".to_string(), "hw:0".to_string()];
    }
}

pub(crate) fn list_system_mics() -> Vec<String> {
    #[cfg(target_os = "windows")]
    {
        let output = Command::new("ffmpeg")
            .arg("-hide_banner")
            .arg("-list_devices")
            .arg("true")
            .arg("-f")
            .arg("dshow")
            .arg("-i")
            .arg("dummy")
            .output();

        if let Ok(out) = output {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let mut names = Vec::new();
            for line in stderr.lines() {
                if line.contains("(audio)") {
                    if let Some(start) = line.find('"') {
                        if let Some(end) = line[start + 1..].find('"') {
                            let name = &line[start + 1..start + 1 + end];
                            names.push(format!("audio=\"{}\"", name));
                        }
                    }
                }
            }
            return names;
        }
    }
    Vec::new()
}

fn read_exact_i16(reader: &mut dyn Read, buf: &mut [i16], expected_bytes: usize) -> Result<()> {
    let mut raw = vec![0u8; expected_bytes];
    reader.read_exact(&mut raw)?;
    for (chunk, out) in raw.chunks_exact(2).zip(buf.iter_mut()) {
        *out = i16::from_le_bytes([chunk[0], chunk[1]]);
    }
    Ok(())
}

fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_space = false;
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_space = false;
        } else if ch.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_basic() {
        assert_eq!(normalize("  Orange!! now"), "orange now");
    }

    #[test]
    fn normalize_drops_special() {
        assert_eq!(normalize("**ORANGE**"), "orange");
    }
}
