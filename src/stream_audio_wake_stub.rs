use anyhow::{bail, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use url::Url;

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

pub fn select_best_model_path() -> PathBuf {
    if let Ok(explicit) = std::env::var("WHISPER_MODEL") {
        let p = PathBuf::from(explicit);
        eprintln!("whisper disabled; using WHISPER_MODEL override: {}", p.display());
        return p;
    }
    let fallback = PathBuf::from("models/ggml-tiny.en.bin");
    eprintln!(
        "whisper disabled; using fallback model path: {}",
        fallback.display()
    );
    fallback
}

pub fn start_stream_wake_from_hls(
    _media_url: Arc<Mutex<Url>>,
    _model_path: &Path,
    _wake_phrases: &[String],
    _log_raw: bool,
    _stop: Arc<AtomicBool>,
    _fired: Arc<AtomicBool>,
    _start_instant: Instant,
    _detect_ns: Arc<AtomicU64>,
    _audio_ns: Arc<AtomicU64>,
    _last_word_ns: Arc<AtomicU64>,
) -> Result<()> {
    bail!("whisper support disabled; rebuild with `--features whisper`");
}

pub fn run_wake_worker_stream(
    _media_url_path: &Path,
    _model_path: &Path,
    _wake_phrases: &[String],
    _log_raw: bool,
    _stop: Arc<AtomicBool>,
    _fired: Arc<AtomicBool>,
    _start_instant: Instant,
    _detect_ns: Arc<AtomicU64>,
    _audio_ns: Arc<AtomicU64>,
    _last_word_ns: Arc<AtomicU64>,
) -> Result<()> {
    bail!("whisper support disabled; rebuild with `--features whisper`");
}

pub fn detect_wake_in_file(
    _input_path: &Path,
    _model_path: &Path,
    _wake_phrases: &[String],
    _log_raw: bool,
) -> Result<Option<f32>> {
    bail!("whisper support disabled; rebuild with `--features whisper`");
}

pub fn start_mic_wake_with_ffmpeg(
    _mic_device: Option<&str>,
    _model_path: &Path,
    _wake_phrases: &[String],
    _log_raw: bool,
    _stop: Arc<AtomicBool>,
    _fired: Arc<AtomicBool>,
    _start_instant: Instant,
    _detect_ns: Arc<AtomicU64>,
    _audio_ns: Arc<AtomicU64>,
    _last_word_ns: Arc<AtomicU64>,
) -> Result<()> {
    bail!("whisper support disabled; rebuild with `--features whisper`");
}

pub fn run_wake_worker_mic(
    _mic_device: Option<&str>,
    _model_path: &Path,
    _wake_phrases: &[String],
    _log_raw: bool,
    _stop: Arc<AtomicBool>,
    _fired: Arc<AtomicBool>,
    _start_instant: Instant,
    _detect_ns: Arc<AtomicU64>,
    _audio_ns: Arc<AtomicU64>,
    _last_word_ns: Arc<AtomicU64>,
) -> Result<()> {
    bail!("whisper support disabled; rebuild with `--features whisper`");
}

pub(crate) fn list_system_mics() -> Vec<String> {
    eprintln!("whisper disabled; microphone listing unavailable");
    Vec::new()
}

#[allow(dead_code)]
pub fn transcribe_words_from_input(
    _input: &str,
    _start_offset_secs: Option<f32>,
    _duration_secs: Option<f32>,
    _force_ts_input: bool,
) -> Result<Vec<WordTiming>> {
    bail!("whisper support disabled; rebuild with `--features whisper`");
}

pub fn transcribe_clip_audio(
    _input: &str,
    _start_offset_secs: Option<f32>,
    _duration_secs: Option<f32>,
    _force_ts_input: bool,
) -> Result<TranscriptPayload> {
    bail!("whisper support disabled; rebuild with `--features whisper`");
}
