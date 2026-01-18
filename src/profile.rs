use std::time::Instant;

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

pub fn profile_enabled() -> bool {
    std::env::var("CLIP_PROFILE")
        .ok()
        .map(|v| parse_bool(&v))
        .unwrap_or(false)
}

pub struct ProfileSpan {
    label: &'static str,
    start: Instant,
    enabled: bool,
}

impl ProfileSpan {
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

pub fn profile_span(label: &'static str) -> ProfileSpan {
    ProfileSpan::new(label)
}
