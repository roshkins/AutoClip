//! Minimal scoped profiling helpers.
//!
//! When enabled via `CLIP_PROFILE`, spans log their elapsed time on drop.

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
            eprintln!(
                "profile: {} took {:.3}s",
                self.label,
                self.start.elapsed().as_secs_f32()
            );
        }
    }
}

/// Convenience helper to create a new profiling span.
pub fn profile_span(label: &'static str) -> ProfileSpan {
    ProfileSpan::new(label)
}
