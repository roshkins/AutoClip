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

/// Listen to stream audio (HLS) via ffmpeg, run Whisper locally, and fire when the wake phrase is detected.
pub fn start_stream_wake_from_hls(
    media_url: &Url,
    model_path: &Path,
    wake_phrase: &str,
    log_raw: bool,
    fired: Arc<AtomicBool>,
) -> Result<()> {
    let media_url = media_url.clone();
    let model_path = model_path.to_path_buf();
    let wake = normalize(wake_phrase);
    std::thread::spawn(move || {
        if let Err(err) = run_wake_loop(&media_url, &model_path, &wake, log_raw, fired) {
            eprintln!("stream wake loop error: {err:#}");
        }
    });
    Ok(())
}

fn run_wake_loop(media_url: &Url, model_path: &Path, wake_norm: &str, log_raw: bool, fired: Arc<AtomicBool>) -> Result<()> {
    let ctx = WhisperContext::new_with_params(
        model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?,
        WhisperContextParameters::default(),
    )
    .context("loading whisper model")?;
    let mut state = ctx.create_state().context("creating whisper state")?;

    let mut ffmpeg = spawn_ffmpeg_pcm(media_url)?;
    let mut stdout = ffmpeg
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("ffmpeg stdout missing"))?;

    let chunk_samples = SAMPLE_RATE * CHUNK_MS / 1000;
    let chunk_bytes = chunk_samples * 2; // s16le
    let window_samples = SAMPLE_RATE * WINDOW_MS / 1000;
    let step = Duration::from_millis(STEP_MS as u64);

    let mut buf = vec![0i16; chunk_samples];
    let mut pcm: Vec<f32> = Vec::with_capacity(window_samples);
    let mut last_run = Instant::now();

    loop {
        if fired.load(Ordering::Relaxed) {
            break;
        }

        if let Err(err) = read_exact_i16(&mut stdout, &mut buf, chunk_bytes) {
            eprintln!("ffmpeg read error or EOF: {err:#}");
            break;
        }

        // append new samples and trim to window
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
                fired.store(true, Ordering::Relaxed);
                eprintln!("wake phrase detected via stream audio: {norm}");
                break;
            }
        }
    }

    let _ = ffmpeg.kill();
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

fn spawn_ffmpeg_pcm(media_url: &Url) -> Result<Child> {
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

    let child = cmd.spawn().context("spawning ffmpeg for stream audio")?;
    Ok(child)
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
