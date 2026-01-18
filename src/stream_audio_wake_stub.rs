use anyhow::{bail, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use url::Url;

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
) -> Result<()> {
    bail!("whisper support disabled; rebuild with `--features whisper`");
}

pub(crate) fn list_system_mics() -> Vec<String> {
    eprintln!("whisper disabled; microphone listing unavailable");
    Vec::new()
}
