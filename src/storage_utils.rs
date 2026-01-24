//! File-system helpers for runtime logs and worker coordination.
//!
//! This module centralizes paths and small utilities that write/read runtime
//! status files so they stay consistent across the codebase.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use serde_json::Value;
use url::Url;

use crate::WakeWorkerStatus;

pub const NON_VIDEO_SUBDIR: &str = "_non_video";

/// Return the current unix timestamp in milliseconds.
pub(crate) fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Resolve the root output directory for a save path.
pub(crate) fn resolve_save_root(save_root: &str) -> PathBuf {
    if save_root.trim().is_empty() {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    } else {
        PathBuf::from(save_root)
    }
}

/// Return the folder where non-video support files should be placed.
pub(crate) fn non_video_dir(save_root: &str) -> PathBuf {
    let base = resolve_save_root(save_root);
    let dir = if base
        .file_name()
        .and_then(|v| v.to_str())
        .map(|v| v.eq_ignore_ascii_case(NON_VIDEO_SUBDIR))
        .unwrap_or(false)
    {
        base
    } else {
        base.join(NON_VIDEO_SUBDIR)
    };
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Append a line to the run log in the non-video folder.
pub(crate) fn log_run_event(save_root: &str, message: &str) {
    let base = non_video_dir(save_root);
    let path = base.join("autoclip_run.log");
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(file, "[{}] {}", now_unix_ms(), message);
    }
}

/// Append a wake pipeline event to a jsonl file in the debug folder.
pub(crate) fn log_wake_event(save_root: &str, wake_id: u64, event: &str, fields: Value) {
    let base = if save_root.trim().is_empty() {
        Path::new(".")
    } else {
        Path::new(save_root)
    };
    let dir = base.join("_debug");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join(format!("wake_{wake_id:04}.jsonl"));
    let mut obj = match fields {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    obj.insert("wake_id".to_string(), Value::from(wake_id));
    obj.insert("event".to_string(), Value::from(event));
    obj.insert("ts_ms".to_string(), Value::from(now_unix_ms()));
    if let Ok(line) = serde_json::to_string(&Value::Object(obj)) {
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = writeln!(file, "{line}");
        }
    }
}

/// Write a file atomically by using a temp file + rename.
pub(crate) fn write_atomic_file(path: &Path, payload: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let tmp_ext = format!("tmp{}", std::process::id());
    let tmp = path.with_extension(tmp_ext);
    fs::write(&tmp, payload).with_context(|| format!("writing {}", tmp.display()))?;
    match fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            fs::write(path, payload).with_context(|| format!("writing {}", path.display()))?;
            let _ = fs::remove_file(&tmp);
            Ok(())
        }
        Err(err) => {
            let _ = fs::remove_file(&tmp);
            Err(err).with_context(|| format!("renaming {}", path.display()))
        }
    }
}

/// Persist the wake worker status JSON.
pub(crate) fn write_wake_worker_status(path: &Path, status: &WakeWorkerStatus) -> Result<()> {
    let payload = serde_json::to_vec(status).context("serializing wake status")?;
    write_atomic_file(path, &payload)
}

/// Persist the currently used media URL.
pub(crate) fn write_media_url_file(path: &Path, url: &Url) -> Result<()> {
    write_atomic_file(path, url.as_str().as_bytes())
}

/// Resolve the wake worker status path.
pub(crate) fn whisper_worker_status_path(save_root: &str) -> PathBuf {
    std::env::var("WHISPER_WORKER_STATUS_PATH")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let base = non_video_dir(save_root);
            base.join(".whisper_wake_status.json")
        })
}

/// Resolve the wake worker media URL path.
pub(crate) fn whisper_worker_media_url_path(save_root: &str) -> PathBuf {
    std::env::var("WHISPER_WORKER_MEDIA_URL_PATH")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let base = non_video_dir(save_root);
            base.join(".whisper_wake_media_url.txt")
        })
}

/// Poll interval for wake worker status files.
pub(crate) fn whisper_worker_status_poll() -> Duration {
    std::env::var("WHISPER_WORKER_STATUS_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or_else(|| Duration::from_millis(500))
}
