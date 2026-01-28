//! Minimal scoped profiling helpers.
//!
//! When enabled via `CLIP_PROFILE`, spans log their elapsed time on drop.

use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Return whether profiling spans are enabled.
pub fn profile_enabled() -> bool {
    std::env::var("CLIP_PROFILE")
        .ok()
        .map(|v| parse_bool(&v))
        .unwrap_or(false)
}

/// Scoped timer that logs elapsed time on drop when profiling is enabled.
pub struct ProfileSpan {
    label: &'static str,
    start: Instant,
    enabled: bool,
}

fn profile_log_file() -> Option<&'static Mutex<Option<std::fs::File>>> {
    static FILE: OnceLock<Mutex<Option<std::fs::File>>> = OnceLock::new();
    let path = std::env::var("CLIP_PROFILE_LOG").ok();
    if path.is_none() {
        return None;
    }
    Some(FILE.get_or_init(|| {
        let file = path.as_ref().and_then(|p| {
            let log_path = Path::new(p);
            if let Some(parent) = log_path.parent() {
                let _ = create_dir_all(parent);
            }
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)
                .ok()
        });
        Mutex::new(file)
    }))
}

fn write_profile_log(line: &str) {
    let Some(file_lock) = profile_log_file() else {
        return;
    };
    if let Ok(mut guard) = file_lock.lock() {
        if let Some(file) = guard.as_mut() {
            let _ = writeln!(file, "{line}");
        }
    }
}

impl ProfileSpan {
    /// Start a new profiling span with a label.
    pub fn new(label: &'static str) -> Self {
        let enabled = profile_enabled();
        let start = Instant::now();
        Self {
            label,
            start,
            enabled,
        }
    }
}

impl Drop for ProfileSpan {
    fn drop(&mut self) {
        if self.enabled {
            let line = format!(
                "profile: {} took {:.3}s",
                self.label,
                self.start.elapsed().as_secs_f32()
            );
            eprintln!("{line}");
            write_profile_log(&line);
        }
    }
}

/// Convenience helper to create a new profiling span.
pub fn profile_span(label: &'static str) -> ProfileSpan {
    ProfileSpan::new(label)
}
