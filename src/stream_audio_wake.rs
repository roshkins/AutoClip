use std::env;
use std::fs;
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

use crate::gpu;
use crate::profile::profile_span;

const SAMPLE_RATE: usize = 16_000;
const CHUNK_MS: usize = 500; // read cadence for responsiveness
const WINDOW_MS: usize = 6_000; // transcription window length (shorter to cut through noise)
const STEP_MS: usize = 1_000; // inference cadence
const DEFAULT_RT_TARGET: f32 = 0.9;
const DEFAULT_EMOTION_WORDS: &str =
    "omg,oh my god,wow,no way,lets go,holy,insane,unbelievable,wtf";

const NO_DETECT: u64 = u64::MAX;
const GGML_LOG_LEVEL_DEBUG: c_int = 5;

static WHISPER_LOG_ONCE: Once = Once::new();
static WHISPER_LOG_DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug)]
struct EmotionConfig {
    audio_enabled: bool,
    threshold: f32,
    audio_rms: f32,
    audio_peak: f32,
    audio_weight: f32,
    text_weight: f32,
    refractory: Duration,
    phrases: Vec<WakePhrase>,
    debug: bool,
}

impl EmotionConfig {
    fn from_env() -> Option<Self> {
        let enabled = env::var("CLIP_EMOTION_ENABLE")
            .ok()
            .and_then(|v| parse_bool_env(&v))
            .unwrap_or(false);
        if !enabled {
            return None;
        }
        let audio_enabled = env::var("CLIP_EMOTION_AUDIO")
            .ok()
            .and_then(|v| parse_bool_env(&v))
            .unwrap_or(true);
        let threshold = env::var("CLIP_EMOTION_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.5);
        let audio_rms = env::var("CLIP_EMOTION_AUDIO_RMS")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.08);
        let audio_peak = env::var("CLIP_EMOTION_AUDIO_PEAK")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.35);
        let audio_weight = env::var("CLIP_EMOTION_AUDIO_WEIGHT")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.0);
        let text_weight = env::var("CLIP_EMOTION_TEXT_WEIGHT")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.0);
        let refractory = env::var("CLIP_EMOTION_REFRACTORY_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(8));
        let debug = env::var("CLIP_EMOTION_DEBUG")
            .ok()
            .and_then(|v| parse_bool_env(&v))
            .unwrap_or(false);
        let raw_words = env::var("CLIP_EMOTION_WORDS")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_EMOTION_WORDS.to_string());
        let phrases = build_wake_phrases(&split_emotion_phrases(&raw_words));

        Some(Self {
            audio_enabled,
            threshold,
            audio_rms,
            audio_peak,
            audio_weight,
            text_weight,
            refractory,
            phrases,
            debug,
        })
    }

    fn text_enabled(&self) -> bool {
        !self.phrases.is_empty()
    }
}

fn parse_bool_env(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => Some(true),
        "0" | "false" | "no" | "n" | "off" => Some(false),
        _ => None,
    }
}

fn split_emotion_phrases(raw: &str) -> Vec<String> {
    raw.split(|c| matches!(c, ',' | '|' | ';' | '\n' | '\r'))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn audio_stats(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut sum_sq = 0.0f32;
    let mut peak = 0.0f32;
    for &s in samples {
        let v = s.abs();
        sum_sq += v * v;
        if v > peak {
            peak = v;
        }
    }
    let rms = (sum_sq / samples.len() as f32).sqrt();
    (rms, peak)
}

#[derive(Clone, Debug)]
pub struct WordTiming {
    pub text: String,
    pub norm: String,
    pub t0: f32,
    pub t1: f32,
}

#[derive(Clone, Debug)]
pub struct TranscriptPayload {
    pub text: String,
    pub words: Vec<WordTiming>,
}

#[derive(Clone, Debug)]
struct TranscriptWindow {
    text: String,
    seg_t0: f32,
    seg_t1: f32,
    words: Vec<WordTiming>,
}

#[derive(Clone, Debug)]
struct WakePhrase {
    raw: String,
    norm: String,
    words: Vec<String>,
}

struct WhisperHandle {
    model_path: PathBuf,
    state: whisper_rs::WhisperState,
    use_gpu: bool,
    cpu_ctx: Option<&'static WhisperContext>,
    cpu_state: Option<whisper_rs::WhisperState>,
}

impl WhisperHandle {
    fn new(model_path: &Path) -> Result<Self> {
        Self::new_with_gpu(model_path, whisper_prefers_gpu())
    }

    fn new_with_gpu(model_path: &Path, prefer_gpu: bool) -> Result<Self> {
        install_whisper_log_filter();
        let model_path = model_path.to_path_buf();
        let mut prefer_gpu = prefer_gpu;
        if prefer_gpu && !whisper_gpu_allowed("whisper") {
            prefer_gpu = false;
        }
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
            state,
            use_gpu,
            cpu_ctx: None,
            cpu_state: None,
        })
    }

    fn ensure_cpu_state(&mut self) -> Result<&mut whisper_rs::WhisperState> {
        if self.cpu_state.is_none() {
            let (ctx, _) = create_whisper_context(&self.model_path, false)?;
            eprintln!("whisper: creating CPU state for fallback");
            let state = ctx.create_state().context("creating whisper CPU state")?;
            self.cpu_ctx = Some(ctx);
            self.cpu_state = Some(state);
        }
        Ok(self
            .cpu_state
            .as_mut()
            .expect("cpu state set when needed"))
    }

    fn transcribe_with_mode(
        &mut self,
        audio: &[f32],
        single_segment: bool,
    ) -> Result<Option<TranscriptWindow>> {
        if self.use_gpu {
            if !whisper_gpu_allowed("whisper") {
                let state = self.ensure_cpu_state()?;
                return transcribe_audio(state, audio, single_segment);
            }
            let _lease = match gpu::try_acquire_gpu_lease("whisper") {
                Some(lease) => lease,
                None => {
                    let state = self.ensure_cpu_state()?;
                    return transcribe_audio(state, audio, single_segment);
                }
            };
            match transcribe_audio(&mut self.state, audio, single_segment) {
                Ok(out) => return Ok(out),
                Err(err) => {
                    eprintln!(
                        "whisper GPU path failed during inference: {err:#}; retrying on CPU"
                    );
                    self.use_gpu = false;
                    let state = self.ensure_cpu_state()?;
                    return transcribe_audio(state, audio, single_segment);
                }
            }
        }
        let state = self.ensure_cpu_state()?;
        transcribe_audio(state, audio, single_segment)
    }

    fn transcribe_with_fallback(&mut self, audio: &[f32]) -> Result<Option<TranscriptWindow>> {
        self.transcribe_with_mode(audio, true)
    }

    fn transcribe_full_with_fallback(&mut self, audio: &[f32]) -> Result<Option<TranscriptWindow>> {
        self.transcribe_with_mode(audio, false)
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

fn whisper_clip_prefers_gpu() -> bool {
    let prefer = match env::var("WHISPER_CLIP_GPU") {
        Ok(v) => {
            let trimmed = v.trim();
            !(trimmed.is_empty()
                || trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false"))
        }
        Err(_) => false,
    };
    if !prefer {
        return false;
    }
    whisper_gpu_allowed("whisper clip")
}

fn whisper_min_free_vram_mb() -> u64 {
    env::var("WHISPER_MIN_FREE_VRAM_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(2048)
}

fn whisper_gpu_allowed(label: &str) -> bool {
    let min_free = whisper_min_free_vram_mb();
    gpu::gpu_vram_allows(min_free, None, label)
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
    wake_phrases: Vec<WakePhrase>,
    log_raw: bool,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
    _start_instant: Instant,
    detect_ns: Arc<AtomicU64>,
    audio_ns: Arc<AtomicU64>,
    last_word_ns: Arc<AtomicU64>,
) -> Result<()>
where
    Spawn: FnMut() -> Result<(Child, Box<dyn Read + Send>)> + Send + 'static,
{
    if wake_phrases.is_empty() {
        anyhow::bail!("no wake phrases provided");
    }
    let mut whisper = WhisperHandle::new(model_path)?;
    log_whisper_backend();
    let emotion_cfg = EmotionConfig::from_env();
    let mut emotion_last_fire: Option<Instant> = None;
    let mut rms_ema: Option<f32> = None;
    let mut peak_ema: Option<f32> = None;

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

            if let Some(window) = whisper.transcribe_with_fallback(&pcm)? {
                let window_start_secs =
                    (total_samples.saturating_sub(pcm.len() as u64) as f32) / SAMPLE_RATE as f32;
                let norm = normalize(&window.text);
                if log_raw && !norm.is_empty() {
                    println!("stream raw: {}", window.text.trim());
                }
                if log_raw && !window.words.is_empty() {
                    for word in &window.words {
                        println!(
                            "stream word: {:.2}-{:.2} {}",
                            word.t0, word.t1, word.text
                        );
                    }
                }
                if !window.words.is_empty() || !norm.is_empty() {
                    let word_secs = window
                        .words
                        .last()
                        .map(|w| w.t1.max(w.t0))
                        .unwrap_or_else(|| window.seg_t1.max(window.seg_t0));
                    let abs_secs = (window_start_secs + word_secs).max(0.0);
                    let nanos_f = (abs_secs as f64 * 1_000_000_000.0).round();
                    let nanos = nanos_f
                        .max(0.0)
                        .min((NO_DETECT - 1) as f64) as u64;
                    last_word_ns.store(nanos, Ordering::Relaxed);
                }
                if let Some(cfg) = emotion_cfg.as_ref() {
                    let (rms, peak) = audio_stats(&pcm);
                    rms_ema = Some(match rms_ema {
                        Some(prev) => prev * 0.9 + rms * 0.1,
                        None => rms,
                    });
                    peak_ema = Some(match peak_ema {
                        Some(prev) => prev * 0.9 + peak * 0.1,
                        None => peak,
                    });

                    let mut text_score = 0.0f32;
                    let mut emotion_start_secs = None;
                    if cfg.text_enabled() {
                        if !window.words.is_empty() {
                            if let Some((idx, _phrase)) =
                                match_wake_words(&window.words, &cfg.phrases)
                            {
                                text_score = 1.0;
                                emotion_start_secs = Some(window.words[idx].t0.max(0.0));
                            }
                        }
                        if text_score == 0.0 && !norm.is_empty() {
                            if let Some((_pos, _phrase)) =
                                match_wake_text(&norm, &cfg.phrases)
                            {
                                text_score = 1.0;
                                emotion_start_secs = Some(window.seg_t0.max(0.0));
                            }
                        }
                    }

                    let mut audio_score = 0.0f32;
                    if cfg.audio_enabled {
                        if rms >= cfg.audio_rms && cfg.audio_rms > 0.0 {
                            audio_score += (rms / cfg.audio_rms).min(2.0);
                        }
                        if peak >= cfg.audio_peak && cfg.audio_peak > 0.0 {
                            audio_score += (peak / cfg.audio_peak).min(2.0);
                        }
                        if let Some(base) = rms_ema {
                            if base > 0.0 && rms > base * 1.8 {
                                audio_score += 0.5;
                            }
                        }
                    }

                    let score = cfg.audio_weight * audio_score + cfg.text_weight * text_score;
                    if cfg.debug {
                        eprintln!(
                            "emotion audio: rms={:.3} peak={:.3} text_score={:.2} audio_score={:.2} total={:.2}",
                            rms,
                            peak,
                            text_score,
                            audio_score,
                            score
                        );
                    }
                    let now = Instant::now();
                    let ready = emotion_last_fire
                        .map(|t| now.duration_since(t) >= cfg.refractory)
                        .unwrap_or(true);
                    if ready && score >= cfg.threshold && !fired.load(Ordering::Relaxed) {
                        emotion_last_fire = Some(now);
                        let base_secs = emotion_start_secs.unwrap_or(window.seg_t0.max(0.0));
                        let abs_t0_secs = (window_start_secs + base_secs).max(0.0);
                        let nanos_f = (abs_t0_secs as f64 * 1_000_000_000.0).round();
                        let nanos = nanos_f
                            .max(0.0)
                            .min((NO_DETECT - 1) as f64) as u64;
                        let _ = detect_ns.compare_exchange(
                            NO_DETECT,
                            nanos,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        );
                        let already = fired.swap(true, Ordering::Relaxed);
                        if !already {
                            eprintln!(
                                "emotion trigger via audio: score={:.2} (rms={:.2}, peak={:.2})",
                                score, rms, peak
                            );
                        }
                    }
                }
                let mut phrase_start_secs = None;
                let mut matched_phrase: Option<&WakePhrase> = None;
                if !window.words.is_empty() {
                    if let Some((idx, phrase)) = match_wake_words(&window.words, &wake_phrases) {
                        phrase_start_secs = Some(window.words[idx].t0.max(0.0));
                        matched_phrase = Some(phrase);
                    }
                }
                if phrase_start_secs.is_none() && !norm.is_empty() {
                    if let Some((pos, phrase)) = match_wake_text(&norm, &wake_phrases) {
                        let seg_span = (window.seg_t1 - window.seg_t0).max(0.0);
                        let frac = (pos as f32 / norm.len() as f32).clamp(0.0, 1.0);
                        phrase_start_secs = Some((window.seg_t0 + frac * seg_span).max(0.0));
                        matched_phrase = Some(phrase);
                    }
                }
                if let Some(phrase_start_secs) = phrase_start_secs {
                    let abs_t0_secs = (window_start_secs + phrase_start_secs).max(0.0);
                    let nanos_f = (abs_t0_secs as f64 * 1_000_000_000.0).round();
                    let nanos = nanos_f
                        .max(0.0)
                        .min((NO_DETECT - 1) as f64) as u64;
                    let _ = detect_ns.compare_exchange(
                        NO_DETECT,
                        nanos,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    );
                    let already = fired.swap(true, Ordering::Relaxed);
                    if !already {
                        let matched = matched_phrase
                            .map(|p| p.raw.as_str())
                            .unwrap_or("unknown");
                        eprintln!("wake phrase detected via stream audio: {matched}");
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
    wake_phrases: &[String],
    log_raw: bool,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
    start_instant: Instant,
    detect_ns: Arc<AtomicU64>,
    audio_ns: Arc<AtomicU64>,
    last_word_ns: Arc<AtomicU64>,
) -> Result<()> {
    let media_url = media_url.clone();
    let model_path = model_path.to_path_buf();
    let wake_phrases = build_wake_phrases(wake_phrases);
    if wake_phrases.is_empty() {
        anyhow::bail!("no wake phrases provided");
    }
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
            wake_phrases,
            log_raw,
            stop,
            fired,
            start_instant,
            detect_ns,
            audio_ns,
            last_word_ns,
        ) {
            eprintln!("stream wake loop error: {err:#}");
        }
    });
    Ok(())
}

pub fn run_wake_worker_stream(
    media_url_path: &Path,
    model_path: &Path,
    wake_phrases: &[String],
    log_raw: bool,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
    start_instant: Instant,
    detect_ns: Arc<AtomicU64>,
    audio_ns: Arc<AtomicU64>,
    last_word_ns: Arc<AtomicU64>,
) -> Result<()> {
    let path = media_url_path.to_path_buf();
    let model_path = model_path.to_path_buf();
    let wake_phrases = build_wake_phrases(wake_phrases);
    if wake_phrases.is_empty() {
        anyhow::bail!("no wake phrases provided");
    }
    run_wake_loop_with_spawn(
        move || {
            let url_raw = fs::read_to_string(&path)
                .with_context(|| format!("reading media url file {}", path.display()))?;
            let url_line = url_raw
                .lines()
                .map(|l| l.trim())
                .find(|l| !l.is_empty())
                .ok_or_else(|| anyhow::anyhow!("media url file is empty"))?;
            let url = Url::parse(url_line).context("parsing media url")?;
            spawn_ffmpeg_pcm(&url)
        },
        &model_path,
        wake_phrases,
        log_raw,
        stop,
        fired,
        start_instant,
        detect_ns,
        audio_ns,
        last_word_ns,
    )
}

/// Detect the wake phrase inside a local media file (TS/MP4/etc) and return its timestamp (seconds).
pub fn detect_wake_in_file(
    input_path: &Path,
    model_path: &Path,
    wake_phrases: &[String],
    log_raw: bool,
) -> Result<Option<f32>> {
    let wake_phrases = build_wake_phrases(wake_phrases);
    if wake_phrases.is_empty() {
        anyhow::bail!("no wake phrases provided");
    }
    let mut whisper = WhisperHandle::new(model_path)?;
    log_whisper_backend();

    let (mut ffmpeg, mut pcm_reader) = spawn_ffmpeg_pcm_file(input_path)?;
    let result = run_wake_loop_file(&mut whisper, &mut *pcm_reader, &wake_phrases, log_raw);
    let _ = ffmpeg.kill();
    result
}

/// Listen to microphone audio via ffmpeg, run Whisper locally, and fire when the wake phrase is detected.
pub fn start_mic_wake_with_ffmpeg(
    mic_device: Option<&str>,
    model_path: &Path,
    wake_phrases: &[String],
    log_raw: bool,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
    start_instant: Instant,
    detect_ns: Arc<AtomicU64>,
    audio_ns: Arc<AtomicU64>,
    last_word_ns: Arc<AtomicU64>,
) -> Result<()> {
    let mic = mic_device.map(|s| s.to_string());
    let model_path = model_path.to_path_buf();
    let wake_phrases = build_wake_phrases(wake_phrases);
    if wake_phrases.is_empty() {
        anyhow::bail!("no wake phrases provided");
    }
    std::thread::spawn(move || {
        if let Err(err) = run_wake_loop_with_spawn(
            move || spawn_ffmpeg_pcm_mic(mic.as_deref()),
            &model_path,
            wake_phrases,
            log_raw,
            stop,
            fired,
            start_instant,
            detect_ns,
            audio_ns,
            last_word_ns,
        ) {
            eprintln!("mic wake loop error: {err:#}");
        }
    });
    Ok(())
}

pub fn run_wake_worker_mic(
    mic_device: Option<&str>,
    model_path: &Path,
    wake_phrases: &[String],
    log_raw: bool,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
    start_instant: Instant,
    detect_ns: Arc<AtomicU64>,
    audio_ns: Arc<AtomicU64>,
    last_word_ns: Arc<AtomicU64>,
) -> Result<()> {
    let mic = mic_device.map(|s| s.to_string());
    let model_path = model_path.to_path_buf();
    let wake_phrases = build_wake_phrases(wake_phrases);
    if wake_phrases.is_empty() {
        anyhow::bail!("no wake phrases provided");
    }
    run_wake_loop_with_spawn(
        move || spawn_ffmpeg_pcm_mic(mic.as_deref()),
        &model_path,
        wake_phrases,
        log_raw,
        stop,
        fired,
        start_instant,
        detect_ns,
        audio_ns,
        last_word_ns,
    )
}

fn transcribe_audio(
    state: &mut whisper_rs::WhisperState,
    audio: &[f32],
    single_segment: bool,
) -> Result<Option<TranscriptWindow>> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_translate(false);
    params.set_no_timestamps(false);
    params.set_token_timestamps(true);
    params.set_split_on_word(true);
    params.set_single_segment(single_segment);
    params.set_language(Some("en"));
    params.set_no_speech_thold(0.6);

    let _span = profile_span("whisper: transcribe window");
    state.full(params, audio).context("running whisper")?;
    let num_segments = state.full_n_segments();
    let mut text = String::new();
    let mut seg_t0 = 0.0f32;
    let mut seg_t1 = 0.0f32;
    let mut seg_set = false;
    let mut words: Vec<WordTiming> = Vec::new();
    for i in 0..num_segments {
        let segment = match state.get_segment(i) {
            Some(segment) => segment,
            None => continue,
        };
        let segment_text = segment.to_str().context("segment text")?;
        let trimmed = segment_text.trim();
        if !trimmed.is_empty() {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(trimmed);
            if !seg_set {
                seg_t0 = segment.start_timestamp() as f32 * 0.01;
                seg_t1 = segment.end_timestamp() as f32 * 0.01;
                seg_set = true;
            }
        }
        let token_count = segment.n_tokens();
        for token_idx in 0..token_count {
            let Some(token) = segment.get_token(token_idx) else {
                continue;
            };
            let token_text = match token.to_str_lossy() {
                Ok(val) => val,
                Err(_) => continue,
            };
            let token_text = token_text.trim();
            let norm = normalize(token_text);
            if norm.is_empty() {
                continue;
            }
            let data = token.token_data();
            let t0 = (data.t0.max(0) as f32) * 0.01;
            let t1 = (data.t1.max(data.t0) as f32) * 0.01;
            words.push(WordTiming {
                text: token_text.to_string(),
                norm,
                t0,
                t1,
            });
        }
    }
    if text.is_empty() && words.is_empty() {
        return Ok(None);
    }
    if !seg_set {
        if let Some(first) = words.first() {
            seg_t0 = first.t0;
        }
        if let Some(last) = words.last() {
            seg_t1 = last.t1;
        }
    }
    Ok(Some(TranscriptWindow {
        text,
        seg_t0,
        seg_t1,
        words,
    }))
}

#[allow(dead_code)]
pub fn transcribe_words_from_input(
    input: &str,
    start_offset_secs: Option<f32>,
    duration_secs: Option<f32>,
    force_ts_input: bool,
) -> Result<Vec<WordTiming>> {
    Ok(
        transcribe_clip_audio(input, start_offset_secs, duration_secs, force_ts_input)?
            .words,
    )
}

pub fn transcribe_clip_audio(
    input: &str,
    start_offset_secs: Option<f32>,
    duration_secs: Option<f32>,
    force_ts_input: bool,
) -> Result<TranscriptPayload> {
    let duration = match duration_secs {
        Some(d) if d.is_finite() && d > 0.0 => Some(d),
        _ => None,
    };
    if duration.is_none() {
        anyhow::bail!("captions require a finite clip duration");
    }

    let model_path = select_best_model_path();
    let mut whisper = WhisperHandle::new_with_gpu(&model_path, whisper_clip_prefers_gpu())?;
    log_whisper_backend();

    let (mut ffmpeg, mut pcm_reader) =
        spawn_ffmpeg_pcm_input(input, start_offset_secs, duration, force_ts_input)?;
    let audio = read_pcm_f32(&mut *pcm_reader)?;
    let _ = ffmpeg.kill();

    if audio.is_empty() {
        return Ok(TranscriptPayload {
            text: String::new(),
            words: Vec::new(),
        });
    }

    let window = whisper.transcribe_full_with_fallback(&audio)?;
    if let Some(window) = window {
        let text = if !window.text.trim().is_empty() {
            window.text.trim().to_string()
        } else {
            window
                .words
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_string()
        };
        return Ok(TranscriptPayload {
            text,
            words: window.words,
        });
    }

    Ok(TranscriptPayload {
        text: String::new(),
        words: Vec::new(),
    })
}

fn run_wake_loop_file(
    whisper: &mut WhisperHandle,
    pcm_reader: &mut dyn Read,
    wake_phrases: &[WakePhrase],
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

        if let Some(window) = whisper.transcribe_with_fallback(&pcm)? {
            let window_start_secs =
                (total_samples.saturating_sub(pcm.len() as u64) as f32) / SAMPLE_RATE as f32;
            let norm = normalize(&window.text);
            if log_raw && !norm.is_empty() {
                println!("file raw: {}", window.text.trim());
            }
            if log_raw && !window.words.is_empty() {
                for word in &window.words {
                    println!(
                        "file word: {:.2}-{:.2} {}",
                        word.t0, word.t1, word.text
                    );
                }
            }
            let mut phrase_start_secs = None;
            let mut matched_phrase: Option<&WakePhrase> = None;
            if !window.words.is_empty() {
                if let Some((idx, phrase)) = match_wake_words(&window.words, wake_phrases) {
                    phrase_start_secs = Some(window.words[idx].t0.max(0.0));
                    matched_phrase = Some(phrase);
                }
            }
            if phrase_start_secs.is_none() && !norm.is_empty() {
                if let Some((pos, phrase)) = match_wake_text(&norm, wake_phrases) {
                    let seg_span = (window.seg_t1 - window.seg_t0).max(0.0);
                    let frac = (pos as f32 / norm.len() as f32).clamp(0.0, 1.0);
                    phrase_start_secs = Some((window.seg_t0 + frac * seg_span).max(0.0));
                    matched_phrase = Some(phrase);
                }
            }
            if let Some(phrase_start_secs) = phrase_start_secs {
                let abs_t0_secs = (window_start_secs + phrase_start_secs).max(0.0);
                let matched = matched_phrase
                    .map(|p| p.raw.as_str())
                    .unwrap_or("unknown");
                eprintln!("wake phrase detected in file audio: {matched}");
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

fn spawn_ffmpeg_pcm_input(
    input: &str,
    start_offset_secs: Option<f32>,
    duration_secs: Option<f32>,
    force_ts_input: bool,
) -> Result<(Child, Box<dyn Read + Send>)> {
    let mut cmd = Command::new(ffmpeg_bin());
    cmd.arg("-nostdin");

    let is_url = Url::parse(input).is_ok();
    if is_url {
        for arg in ffmpeg_hwaccel_flags() {
            cmd.arg(arg);
        }
    } else if force_ts_input {
        cmd.arg("-f").arg("mpegts");
    }

    cmd.arg("-i").arg(input);
    if let Some(ss) = start_offset_secs.filter(|v| v.is_finite() && *v > 0.0) {
        cmd.arg("-ss").arg(format!("{ss:.3}"));
    }
    if let Some(d) = duration_secs.filter(|v| v.is_finite() && *v > 0.0) {
        cmd.arg("-t").arg(format!("{d:.3}"));
    }
    cmd.args(wake_af_flags())
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
        .context("spawning ffmpeg for caption audio")?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("ffmpeg stdout missing"))?;
    Ok((child, Box::new(stdout)))
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

fn read_pcm_f32(reader: &mut dyn Read) -> Result<Vec<f32>> {
    let mut raw = Vec::new();
    reader.read_to_end(&mut raw)?;
    let mut out = Vec::with_capacity(raw.len() / 2);
    for chunk in raw.chunks_exact(2) {
        let val = i16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(val as f32 / 32768.0);
    }
    Ok(out)
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

fn build_wake_phrases(raw: &[String]) -> Vec<WakePhrase> {
    let mut out = Vec::new();
    for phrase in raw {
        let norm = normalize(phrase);
        if norm.is_empty() {
            continue;
        }
        let words: Vec<String> = norm
            .split_whitespace()
            .map(|w| w.to_string())
            .collect();
        if words.is_empty() {
            continue;
        }
        out.push(WakePhrase {
            raw: phrase.clone(),
            norm,
            words,
        });
    }
    out
}

fn match_wake_words<'a>(
    words: &[WordTiming],
    phrases: &'a [WakePhrase],
) -> Option<(usize, &'a WakePhrase)> {
    let mut best: Option<(usize, &WakePhrase)> = None;
    for phrase in phrases {
        let count = phrase.words.len();
        if count == 0 || words.len() < count {
            continue;
        }
        for start in 0..=words.len().saturating_sub(count) {
            if phrase
                .words
                .iter()
                .enumerate()
                .all(|(idx, w)| words[start + idx].norm == *w)
            {
                let replace = best.map(|(best_start, _)| start < best_start).unwrap_or(true);
                if replace {
                    best = Some((start, phrase));
                }
                break;
            }
        }
    }
    best
}

fn match_wake_text<'a>(
    transcript: &str,
    phrases: &'a [WakePhrase],
) -> Option<(usize, &'a WakePhrase)> {
    let mut best: Option<(usize, &WakePhrase)> = None;
    for phrase in phrases {
        if phrase.norm.is_empty() {
            continue;
        }
        if let Some(pos) = transcript.find(&phrase.norm) {
            let replace = best.map(|(best_pos, _)| pos < best_pos).unwrap_or(true);
            if replace {
                best = Some((pos, phrase));
            }
        }
    }
    best
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

    #[test]
    fn audio_stats_reports_rms_and_peak() {
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let (rms, peak) = audio_stats(&samples);
        assert!((peak - 1.0).abs() < 1e-6);
        assert!(rms > 0.6 && rms < 0.8);
    }

    #[test]
    fn split_emotion_phrases_handles_commas() {
        let phrases = split_emotion_phrases("wow, oh my god | wtf");
        assert_eq!(phrases, vec!["wow", "oh my god", "wtf"]);
    }
}
