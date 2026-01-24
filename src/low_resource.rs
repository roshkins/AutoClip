//! Low-resource mode configuration and env override handling.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const LOW_RESOURCE_OVERRIDES: &[(&str, &str)] = &[
    ("CLIP_LIVE_FAST", "1"),
    ("CLIP_POSE", "0"),
    ("CLIP_FACE_TRACK", "0"),
    ("CLIP_DETECT", "0"),
    ("CLIP_GAMEPLAY", "0"),
    ("CLIP_REGION_DETECT", "0"),
    ("CLIP_LLM_ENABLE", "0"),
    ("CLIP_EMOTION_ENABLE", "0"),
    ("CLIP_EMOTION_AUDIO", "0"),
    ("CLIP_EMOTION_FACE", "0"),
    ("CLIP_FACE_BUDGET_SECS", "6"),
    ("CLIP_DETECT_BUDGET_SECS", "6"),
    ("CLIP_DETECT_SAMPLES", "1"),
    ("CLIP_DETECT_STEP", "2.5"),
    ("CLIP_LIVE_LAYOUT_TTL_SECS", "300"),
    ("WHISPER_GPU", "0"),
    ("WHISPER_CLIP_GPU", "0"),
    ("WHISPER_RT_TARGET", "0.5"),
    ("WHISPER_MODEL_CANDIDATES", "ggml-tiny.en.bin"),
    ("WHISPER_WORKER_STATUS_MS", "1000"),
];

#[derive(Clone, Copy, Debug)]
pub(crate) struct LowResourceConfig {
    env_flag: &'static str,
    overrides: &'static [(&'static str, &'static str)],
}

impl Default for LowResourceConfig {
    fn default() -> Self {
        Self {
            env_flag: "CLIP_LOW_RESOURCES",
            overrides: LOW_RESOURCE_OVERRIDES,
        }
    }
}

impl LowResourceConfig {
    fn enabled(self) -> bool {
        std::env::var(self.env_flag)
            .ok()
            .and_then(|v| parse_bool(&v))
            .unwrap_or(false)
    }
}

struct LowResourceState {
    config: LowResourceConfig,
    enabled: bool,
    baseline: HashMap<String, Option<String>>,
}

impl LowResourceState {
    fn new(config: LowResourceConfig) -> Self {
        Self {
            config,
            enabled: false,
            baseline: HashMap::new(),
        }
    }

    fn refresh(&mut self) {
        let enabled = self.config.enabled();
        match (self.enabled, enabled) {
            (false, true) => self.apply(),
            (true, false) => self.restore(),
            (true, true) => self.enforce(),
            (false, false) => {}
        }
    }

    fn apply(&mut self) {
        for (key, value) in self.config.overrides {
            if !self.baseline.contains_key(*key) {
                self.baseline.insert((*key).to_string(), std::env::var(*key).ok());
            }
            std::env::set_var(key, value);
        }
        self.enabled = true;
        eprintln!("low resource mode enabled");
    }

    fn enforce(&self) {
        for (key, value) in self.config.overrides {
            if std::env::var(key).ok().as_deref() != Some(*value) {
                std::env::set_var(key, value);
            }
        }
    }

    fn restore(&mut self) {
        let keys: Vec<String> = self.baseline.keys().cloned().collect();
        for key in keys {
            match self.baseline.get(&key).cloned().unwrap_or(None) {
                Some(value) => std::env::set_var(&key, value),
                None => std::env::remove_var(&key),
            }
        }
        self.baseline.clear();
        self.enabled = false;
        eprintln!("low resource mode disabled");
    }
}

/// Whether low-resource overrides are enabled via CLIP_LOW_RESOURCES.
pub(crate) fn low_resource_enabled() -> bool {
    LowResourceConfig::default().enabled()
}

/// Apply or restore environment overrides based on the current low-resource flag.
pub(crate) fn refresh_low_resource_state() {
    if let Ok(mut state) = low_resource_state().lock() {
        state.refresh();
    }
}

/// Ensure clip detection budgets are set for live mode when missing.
pub(crate) fn ensure_live_detect_budgets() {
    let mut face_secs = std::env::var("CLIP_FACE_BUDGET_SECS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0);
    let mut total_secs = std::env::var("CLIP_DETECT_BUDGET_SECS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0);

    let mut set_face = false;
    let mut set_total = false;

    if face_secs.is_none() {
        face_secs = Some(20.0);
        set_face = true;
    }
    if total_secs.is_none() {
        let base = face_secs.unwrap_or(20.0);
        total_secs = Some(base * 2.0);
        set_total = true;
    }

    if set_face {
        if let Some(value) = face_secs {
            std::env::set_var("CLIP_FACE_BUDGET_SECS", format!("{:.0}", value));
        }
    }
    if set_total {
        if let Some(value) = total_secs {
            std::env::set_var("CLIP_DETECT_BUDGET_SECS", format!("{:.0}", value));
        }
    }

    if set_face || set_total {
        let face = face_secs.unwrap_or(20.0);
        let total = total_secs.unwrap_or(face * 2.0);
        let gameplay = (total - face).max(0.0);
        eprintln!(
            "clip detect: live budgets face={:.0}s gameplay={:.0}s (total {:.0}s)",
            face, gameplay, total
        );
    }
}

fn low_resource_state() -> &'static Mutex<LowResourceState> {
    static STATE: OnceLock<Mutex<LowResourceState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(LowResourceState::new(LowResourceConfig::default())))
}

#[cfg(test)]
pub(crate) fn reset_low_resource_state_for_tests() {
    if let Ok(mut state) = low_resource_state().lock() {
        state.enabled = false;
        state.baseline.clear();
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}
