use std::env;
use std::ffi::CStr;
use std::io::{ErrorKind, Read};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Once;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use url::Url;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const SAMPLE_RATE: usize = 16_000;
const CHUNK_MS: usize = 500; // read cadence for responsiveness
const WINDOW_MS: usize = 6_000; // transcription window length (shorter to cut through noise)
const STEP_MS: usize = 1_000; // inference cadence
const DEFAULT_RT_TARGET: f32 = 0.9;

const NO_DETECT: u64 = u64::MAX;
const GGML_LOG_LEVEL_DEBUG: c_int = 5;

static WHISPER_LOG_ONCE: Once = Once::new();
static WHISPER_LOG_DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

struct WhisperHandle {
    model_path: PathBuf,
    ctx: &'static WhisperContext,
    state: whisper_rs::WhisperState,
    use_gpu: bool,
}

impl WhisperHandle {
    fn new(model_path: &Path) -> Result<Self> {
        install_whisper_log_filter();
        let model_path = model_path.to_path_buf();
        let prefer_gpu = whisper_prefers_gpu();
        let (ctx, use_gpu) = create_whisper_context(&model_path, prefer_gpu)
            .or_else(|err| {
                if prefer_gpu {
                    eprintln!("whisper GPU init failed: {err:#}; falling back to CPU");
                    create_whisper_context(&model_path, false)
                } else {
                    Err(err)
                }
            })?;
        eprintln!("whisper: creating state (use_gpu={use_gpu})");
        let state = ctx.create_state().context("creating whisper state")?;
        eprintln!("whisper: state created (use_gpu={use_gpu})");
        Ok(Self {
            model_path,
            ctx,
            state,
            use_gpu,
        })
    }

    fn transcribe_with_fallback(&mut self, audio: &[f32]) -> Result<Option<(String, f32, f32)>> {
        match transcribe_window(&mut self.state, audio) {
            Ok(out) => Ok(out),
            Err(err) if self.use_gpu => {
                eprintln!("whisper GPU path failed during inference: {err:#}; retrying on CPU");
                let (ctx, _) = create_whisper_context(&self.model_path, false)?;
                eprintln!("whisper: creating CPU state after GPU failure");
                let state = ctx.create_state().context("creating whisper state")?;
                eprintln!("whisper: CPU state created after GPU failure");
                self.ctx = ctx;
                self.state = state;
                self.use_gpu = false;
                transcribe_window(&mut self.state, audio)
            }
            Err(err) => Err(err),
        }
    }
}

fn install_whisper_log_filter() {
    WHISPER_LOG_ONCE.call_once(|| {
        let enable_debug = env::var("WHISPER_LOG_LEVEL")
            .or_else(|_| env::var("GGML_LOG_LEVEL"))
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .map(|v| matches!(v.as_str(), "debug" | "trace"))
            .unwrap_or(false);
        WHISPER_LOG_DEBUG_ENABLED.store(enable_debug, Ordering::Relaxed);
        unsafe {
            whisper_rs::set_log_callback(Some(whisper_log_callback), std::ptr::null_mut());
        }
    });
}

unsafe extern "C" fn whisper_log_callback(
    level: c_int,
    text: *const c_char,
    _user_data: *mut c_void,
) {
    if text.is_null() {
        return;
    }
    if level == GGML_LOG_LEVEL_DEBUG && !WHISPER_LOG_DEBUG_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let msg = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    eprint!("{}", msg);
}

fn whisper_prefers_gpu() -> bool {
    match env::var("WHISPER_GPU") {
        Ok(v) => {
            let trimmed = v.trim();
            !(trimmed.is_empty()
                || trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false"))
        }
        Err(_) => true,
    }
}

fn create_whisper_context(
    model_path: &Path,
    use_gpu: bool,
) -> Result<(&'static WhisperContext, bool)> {
    let mut wparams = WhisperContextParameters::default();
    wparams.use_gpu = use_gpu;

    eprintln!("whisper: initializing context (use_gpu={use_gpu})");
    let ctx = WhisperContext::new_with_params(
        model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?,
        wparams,
    )
    .context("loading whisper model")?;
    eprintln!("whisper: context initialized (use_gpu={use_gpu})");
    // Leak the context so the state can borrow it for the process lifetime.
    let ctx = Box::leak(Box::new(ctx));
    Ok((ctx, use_gpu))
}

/// Common wake loop; caller provides how to spawn an ffmpeg PCM source (stream or mic).
fn run_wake_loop_with_spawn<Spawn>(
    mut spawn_pcm: Spawn,
    model_path: &Path,
    wake_norm: &str,
    log_raw: bool,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
    _start_instant: Instant,
    detect_ns: Arc<AtomicU64>,
    audio_ns: Arc<AtomicU64>,
) -> Result<()>
where
    Spawn: FnMut() -> Result<(Child, Box<dyn Read + Send>)> + Send + 'static,
{
    let mut whisper = WhisperHandle::new(model_path)?;
    log_whisper_backend();

    let chunk_samples = SAMPLE_RATE * CHUNK_MS / 1000;
    let chunk_bytes = chunk_samples * 2; // s16le
    let window_samples = SAMPLE_RATE * WINDOW_MS / 1000;
    let step = Duration::from_millis(STEP_MS as u64);

    let mut buf = vec![0i16; chunk_samples];
    let mut pcm: Vec<f32> = Vec::with_capacity(window_samples);
    let mut total_samples: u64 = 0;

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
            total_samples = total_samples.saturating_add(chunk_samples as u64);
            let audio_end_secs = total_samples as f64 / SAMPLE_RATE as f64;
            let nanos_f = (audio_end_secs * 1_000_000_000.0).round();
            let nanos = nanos_f
                .max(0.0)
                .min((NO_DETECT - 1) as f64) as u64;
            audio_ns.store(nanos, Ordering::Relaxed);
            if pcm.len() > window_samples {
                let drop = pcm.len() - window_samples;
                pcm.drain(0..drop);
            }

            if last_run.elapsed() < step || pcm.len() < SAMPLE_RATE {
                continue;
            }
            last_run = Instant::now();

            if let Some((text, seg_t0_secs, seg_t1_secs)) =
                whisper.transcribe_with_fallback(&pcm)?
            {
                let window_start_secs = (total_samples.saturating_sub(pcm.len() as u64) as f32) / SAMPLE_RATE as f32;
                let norm = normalize(&text);
                if log_raw && !norm.is_empty() {
                    println!("stream raw: {}", text.trim());
                }
                if norm.contains(wake_norm) {
                    let match_pos = norm.find(wake_norm).unwrap_or(0);
                    let seg_span = (seg_t1_secs - seg_t0_secs).max(0.0);
                    let frac = if !norm.is_empty() { (match_pos as f32 / norm.len() as f32).clamp(0.0, 1.0) } else { 0.0 };
                    let phrase_start_secs = (seg_t0_secs + frac * seg_span).max(0.0);
                    let abs_t0_secs = (window_start_secs + phrase_start_secs).max(0.0);
                    let nanos_f = (abs_t0_secs as f64 * 1_000_000_000.0).round();
                    let nanos = nanos_f
                        .max(0.0)
                        .min((NO_DETECT - 1) as f64) as u64;
                    let _ = detect_ns.compare_exchange(NO_DETECT, nanos, Ordering::Relaxed, Ordering::Relaxed);
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

/// Select the largest Whisper model that meets the realtime target.
/// Honors WHISPER_MODEL as a hard override. Candidates may be provided via
/// WHISPER_MODEL_CANDIDATES (comma-separated filenames); otherwise defaults
/// to medium/small/base/tiny English ggml models under ./models.
pub fn select_best_model_path() -> PathBuf {
    if let Ok(explicit) = env::var("WHISPER_MODEL") {
        let p = PathBuf::from(explicit);
        eprintln!("using WHISPER_MODEL override: {}", p.display());
        return p;
    }

    let target: f32 = env::var("WHISPER_RT_TARGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_RT_TARGET);

    let candidates = env::var("WHISPER_MODEL_CANDIDATES")
        .ok()
        .map(|v| v.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|s| s.to_string()).collect())
        .unwrap_or_else(|| {
            vec![
                "ggml-medium.en.bin".to_string(),
                "ggml-small.en.bin".to_string(),
                "ggml-base.en.bin".to_string(),
                "ggml-tiny.en.bin".to_string(),
            ]
        });

    let mut tested = Vec::new();
    for name in candidates {
        let path = if Path::new(&name).is_absolute() {
            PathBuf::from(&name)
        } else {
            Path::new("models").join(&name)
        };

        if !path.exists() {
            tested.push((path.clone(), None));
            continue;
        }

        let rt_factor = benchmark_model_rt(&path);
        tested.push((path.clone(), rt_factor));
    }

    // Pick the first candidate that meets target; fall back to first existing.
    let mut chosen: Option<PathBuf> = None;
    for (p, rt) in &tested {
        if let Some(rt) = rt {
            eprintln!("model {:?}: rt_factor={:.3}", p.file_name().unwrap_or_default(), rt);
            if *rt <= target {
                chosen = Some(p.clone());
                break;
            }
        } else {
            eprintln!("model {:?}: unavailable or failed to benchmark", p.file_name().unwrap_or_default());
        }
    }

    if chosen.is_none() {
        chosen = tested.iter().find(|(p, rt)| p.exists() && rt.is_some()).map(|(p, _)| p.clone());
    }

    chosen.unwrap_or_else(|| {
        let fallback = PathBuf::from("models/ggml-tiny.en.bin");
        eprintln!("no candidate models benchmarked; falling back to {}", fallback.display());
        fallback
    })
}

fn benchmark_model_rt(path: &Path) -> Option<f32> {
    // 8s of silence to match WINDOW_MS.
    let samples = SAMPLE_RATE * WINDOW_MS / 1000;
    let audio: Vec<f32> = vec![0.0; samples];

    let prefer_gpu = whisper_prefers_gpu();
    let (ctx, _use_gpu) = create_whisper_context(path, prefer_gpu)
        .or_else(|_| {
            if prefer_gpu {
                create_whisper_context(path, false)
            } else {
                Err(anyhow::anyhow!("whisper init failed"))
            }
        })
        .ok()?;
    let mut state = ctx.create_state().ok()?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_translate(false);
    params.set_no_timestamps(true);
    params.set_single_segment(true);
    params.set_language(Some("en"));
    params.set_no_speech_thold(0.45);

    let start = Instant::now();
    if state.full(params, &audio).is_err() {
        return None;
    }
    let elapsed = start.elapsed().as_secs_f32();
    let rt = elapsed / (WINDOW_MS as f32 / 1000.0);
    Some(rt)
}

/// Listen to stream audio (HLS) via ffmpeg, run Whisper locally, and fire when the wake phrase is detected.
pub fn start_stream_wake_from_hls(
    media_url: Arc<Mutex<Url>>,
    model_path: &Path,
    wake_phrase: &str,
    log_raw: bool,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
    start_instant: Instant,
    detect_ns: Arc<AtomicU64>,
    audio_ns: Arc<AtomicU64>,
) -> Result<()> {
    let media_url = media_url.clone();
    let model_path = model_path.to_path_buf();
    let wake = normalize(wake_phrase);
    std::thread::spawn(move || {
        if let Err(err) = run_wake_loop_with_spawn(
            move || {
                let url = {
                    let guard = media_url
                        .lock()
                        .map_err(|_| anyhow::anyhow!("media url lock poisoned"))?;
                    guard.clone()
                };
                spawn_ffmpeg_pcm(&url)
            },
            &model_path,
            &wake,
            log_raw,
            stop,
            fired,
            start_instant,
            detect_ns,
            audio_ns,
        ) {
            eprintln!("stream wake loop error: {err:#}");
        }
    });
    Ok(())
}

/// Detect the wake phrase inside a local media file (TS/MP4/etc) and return its timestamp (seconds).
pub fn detect_wake_in_file(
    input_path: &Path,
    model_path: &Path,
    wake_phrase: &str,
    log_raw: bool,
) -> Result<Option<f32>> {
    let wake = normalize(wake_phrase);
    let mut whisper = WhisperHandle::new(model_path)?;
    log_whisper_backend();

    let (mut ffmpeg, mut pcm_reader) = spawn_ffmpeg_pcm_file(input_path)?;
    let result = run_wake_loop_file(&mut whisper, &mut *pcm_reader, &wake, log_raw);
    let _ = ffmpeg.kill();
    result
}

/// Listen to microphone audio via ffmpeg, run Whisper locally, and fire when the wake phrase is detected.
pub fn start_mic_wake_with_ffmpeg(
    mic_device: Option<&str>,
    model_path: &Path,
    wake_phrase: &str,
    log_raw: bool,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
    start_instant: Instant,
    detect_ns: Arc<AtomicU64>,
    audio_ns: Arc<AtomicU64>,
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
            start_instant,
            detect_ns,
            audio_ns,
        ) {
            eprintln!("mic wake loop error: {err:#}");
        }
    });
    Ok(())
}

fn transcribe_window(state: &mut whisper_rs::WhisperState, audio: &[f32]) -> Result<Option<(String, f32, f32)>> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_translate(false);
    params.set_no_timestamps(true);
    params.set_single_segment(true);
    params.set_language(Some("en"));
    params.set_no_speech_thold(0.6);

    state.full(params, audio).context("running whisper")?;
    let num_segments = state.full_n_segments();
    for i in 0..num_segments {
        let segment = match state.get_segment(i) {
            Some(segment) => segment,
            None => continue,
        };
        let text = segment.to_str().context("segment text")?;
        let t0_secs = segment.start_timestamp() as f32 * 0.01;
        let t1_secs = segment.end_timestamp() as f32 * 0.01;
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return Ok(Some((trimmed.to_string(), t0_secs, t1_secs)));
        }
    }
    Ok(None)
}

fn run_wake_loop_file(
    whisper: &mut WhisperHandle,
    pcm_reader: &mut dyn Read,
    wake_norm: &str,
    log_raw: bool,
) -> Result<Option<f32>> {
    let chunk_samples = SAMPLE_RATE * CHUNK_MS / 1000;
    let chunk_bytes = chunk_samples * 2; // s16le
    let window_samples = SAMPLE_RATE * WINDOW_MS / 1000;
    let step = Duration::from_millis(STEP_MS as u64);

    let mut buf = vec![0i16; chunk_samples];
    let mut pcm: Vec<f32> = Vec::with_capacity(window_samples);
    let mut total_samples: u64 = 0;
    let mut last_run = Instant::now();

    loop {
        if let Err(err) = read_exact_i16(pcm_reader, &mut buf, chunk_bytes) {
            if let Some(io_err) = err.downcast_ref::<std::io::Error>() {
                if io_err.kind() == ErrorKind::UnexpectedEof || io_err.kind() == ErrorKind::BrokenPipe {
                    break;
                }
            }
            eprintln!("file wake: ffmpeg read ended: {err:#}");
            break;
        }

        for s in &buf {
            pcm.push(*s as f32 / 32768.0);
        }
        total_samples = total_samples.saturating_add(chunk_samples as u64);
        if pcm.len() > window_samples {
            let drop = pcm.len() - window_samples;
            pcm.drain(0..drop);
        }

        if last_run.elapsed() < step || pcm.len() < SAMPLE_RATE {
            continue;
        }
        last_run = Instant::now();

        if let Some((text, seg_t0_secs, seg_t1_secs)) =
            whisper.transcribe_with_fallback(&pcm)?
        {
            let window_start_secs = (total_samples.saturating_sub(pcm.len() as u64) as f32)
                / SAMPLE_RATE as f32;
            let norm = normalize(&text);
            if log_raw && !norm.is_empty() {
                println!("file raw: {}", text.trim());
            }
            if norm.contains(wake_norm) {
                let match_pos = norm.find(wake_norm).unwrap_or(0);
                let seg_span = (seg_t1_secs - seg_t0_secs).max(0.0);
                let frac = if !norm.is_empty() {
                    (match_pos as f32 / norm.len() as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let phrase_start_secs = (seg_t0_secs + frac * seg_span).max(0.0);
                let abs_t0_secs = (window_start_secs + phrase_start_secs).max(0.0);
                eprintln!("wake phrase detected in file audio: {norm}");
                return Ok(Some(abs_t0_secs));
            }
        }
    }

    Ok(None)
}

fn spawn_ffmpeg_pcm(media_url: &Url) -> Result<(Child, Box<dyn Read + Send>)> {
    let mut cmd = Command::new(ffmpeg_bin());
    cmd.arg("-nostdin");

    for arg in ffmpeg_hwaccel_flags() {
        cmd.arg(arg);
    }

    cmd.arg("-i")
        .arg(media_url.as_str())
        .args(wake_af_flags())
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

    let mut child = cmd
        .spawn()
        .map_err(ffmpeg_spawn_err)
        .context("spawning ffmpeg for stream audio")?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("ffmpeg stdout missing"))?;
    Ok((child, Box::new(stdout)))
}

fn spawn_ffmpeg_pcm_file(input_path: &Path) -> Result<(Child, Box<dyn Read + Send>)> {
    let mut cmd = Command::new(ffmpeg_bin());
    cmd.arg("-nostdin");
    if std::env::var("CLIP_TS_REALTIME")
        .ok()
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
    {
        cmd.arg("-re");
    }

    cmd.arg("-i")
        .arg(input_path)
        .args(wake_af_flags())
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

    let mut child = cmd
        .spawn()
        .map_err(ffmpeg_spawn_err)
        .context("spawning ffmpeg for file audio")?;
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
        let mut cmd = Command::new(ffmpeg_bin());
        cmd.arg("-nostdin");
        cmd.arg("-loglevel").arg("warning");
        cmd.arg("-thread_queue_size").arg("4096");

        #[cfg(target_os = "windows")]
        {
            cmd.arg("-f").arg("dshow").arg("-rtbufsize").arg("10M");
        }
        #[cfg(target_os = "macos")]
        {
            cmd.arg("-f").arg("avfoundation");
        }
        #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
        {
            cmd.arg("-f").arg("pulse");
        }

        for arg in ffmpeg_hwaccel_flags() {
            cmd.arg(arg);
        }

        #[cfg(target_os = "windows")]
        {
            cmd.arg("-i").arg(&input);
        }
        #[cfg(target_os = "macos")]
        {
            cmd.arg("-i").arg(&input);
        }
        #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
        {
            cmd.arg("-i").arg(&input);
        }

        cmd.args(wake_af_flags());

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

        match cmd
            .spawn()
            .map_err(ffmpeg_spawn_err)
            .with_context(|| format!("spawning ffmpeg for mic audio using input='{input}' (set MIC_DEVICE to override)")) {
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

fn ffmpeg_bin() -> String {
    std::env::var("FFMPEG_BIN").unwrap_or_else(|_| "ffmpeg".to_string())
}

fn ffmpeg_spawn_err(err: std::io::Error) -> anyhow::Error {
    if err.kind() == ErrorKind::NotFound {
        anyhow::anyhow!(
            "ffmpeg executable not found. Install FFmpeg and ensure it's on PATH, or set FFMPEG_BIN to the full path (e.g., C:\\ffmpeg\\bin\\ffmpeg.exe)."
        )
    } else {
        err.into()
    }
}

fn ffmpeg_hwaccel_flags() -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(hw) = env::var("FFMPEG_HWACCEL") {
        if !hw.trim().is_empty() {
            out.push("-hwaccel".to_string());
            out.push(hw);
        }
    }
    if let Ok(dev) = env::var("FFMPEG_HWACCEL_DEVICE") {
        if !dev.trim().is_empty() {
            out.push("-hwaccel_device".to_string());
            out.push(dev);
        }
    }
    out
}

fn wake_af_flags() -> Vec<String> {
    if let Ok(af) = env::var("WAKE_FF_AF") {
        if !af.trim().is_empty() {
            return vec!["-af".to_string(), af];
        }
    }
    Vec::new()
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

fn log_whisper_backend() {
    let cublas = env::var("WHISPER_CUBLAS").unwrap_or_else(|_| "(unset)".to_string());
    let ggml_log = env::var("GGML_LOG_LEVEL").unwrap_or_else(|_| "(unset)".to_string());
    let whisper_gpu_env = env::var("WHISPER_GPU").unwrap_or_else(|_| "(unset)".to_string());
    let nvidia_present = detect_nvidia_gpus_present();
    eprintln!(
        "whisper backend: use_gpu=true (requested); WHISPER_CUBLAS={cublas}; WHISPER_GPU={whisper_gpu_env}; GGML_LOG_LEVEL={ggml_log}; nvidia_detected={nvidia_present}"
    );
    if cublas == "(unset)" {
        eprintln!("whisper backend warning: WHISPER_CUBLAS not set; if the binary wasn't built with CUDA, inference will fall back to CPU");
    }
}

fn detect_nvidia_gpus_present() -> bool {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=index")
        .arg("--format=csv,noheader")
        .output();
    if let Ok(out) = output {
        out.status.success() && !String::from_utf8_lossy(&out.stdout).trim().is_empty()
    } else {
        false
    }
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
