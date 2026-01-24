//! Caption and subtitle helpers for open/closed captions.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::clip_layout::StackedLayoutDims;
use crate::stream_audio_wake::{TranscriptPayload, WordTiming};
use crate::text_utils::normalize_title_whitespace;
use crate::parse_bool;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CaptionPosition {
    Margin,
    Chest,
}

#[derive(Clone, Debug)]
pub(crate) struct CaptionConfig {
    pub(crate) position: CaptionPosition,
    pub(crate) font: Option<String>,
    pub(crate) font_size: f32,
    pub(crate) color: String,
    pub(crate) outline_color: String,
    pub(crate) outline: u32,
    pub(crate) min_word_secs: f32,
    pub(crate) max_words: usize,
    pub(crate) chest_ratio: f32,
    pub(crate) margin_offset_px: f32,
    pub(crate) debug: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct CaptionWord {
    pub(crate) text: String,
    pub(crate) start: f32,
    pub(crate) end: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct SubtitleSpec {
    pub(crate) path: PathBuf,
}

#[derive(Debug)]
pub(crate) struct TempSubtitle {
    path: PathBuf,
}

impl TempSubtitle {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for TempSubtitle {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub(crate) fn captions_enabled() -> bool {
    std::env::var("CLIP_CAPTIONS")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

pub(crate) fn closed_captions_enabled() -> bool {
    std::env::var("CLIP_CLOSED_CAPTIONS")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn parse_caption_position(value: &str) -> Option<CaptionPosition> {
    match value.trim().to_ascii_lowercase().as_str() {
        "margin" | "seam" | "gap" | "between" => Some(CaptionPosition::Margin),
        "chest" | "torso" => Some(CaptionPosition::Chest),
        _ => None,
    }
}

fn resolve_caption_pixels(out_h: u32, value: Option<f32>, default_ratio: f32) -> f32 {
    let raw = value.unwrap_or(default_ratio);
    if raw.is_finite() && raw > 0.0 {
        if raw <= 2.0 {
            (out_h as f32 * raw).max(1.0)
        } else {
            raw
        }
    } else {
        out_h as f32 * default_ratio
    }
}

pub(crate) fn default_caption_font() -> Option<String> {
    #[cfg(windows)]
    {
        let candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\seguisb.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
        ];
        for path in candidates {
            if Path::new(path).exists() {
                return Some(path.to_string());
            }
        }
    }
    None
}

pub(crate) fn read_caption_config(out_h: u32) -> Option<CaptionConfig> {
    if !captions_enabled() {
        return None;
    }
    let position = std::env::var("CLIP_CAPTIONS_POSITION")
        .ok()
        .and_then(|v| parse_caption_position(&v))
        .unwrap_or(CaptionPosition::Margin);
    let mut font = std::env::var("CLIP_CAPTIONS_FONT")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    if font.is_none() {
        font = default_caption_font();
    }
    let font_size = std::env::var("CLIP_CAPTIONS_SIZE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let font_size = resolve_caption_pixels(out_h, font_size, 0.045).clamp(12.0, out_h as f32);
    let color = std::env::var("CLIP_CAPTIONS_COLOR")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "white".to_string());
    let outline_color = std::env::var("CLIP_CAPTIONS_OUTLINE_COLOR")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "black".to_string());
    let outline = std::env::var("CLIP_CAPTIONS_OUTLINE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(3.0)
        .max(0.0)
        .round() as u32;
    let min_word_secs = std::env::var("CLIP_CAPTIONS_MIN_WORD_SECS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.12)
        .clamp(0.04, 1.0);
    let max_words = std::env::var("CLIP_CAPTIONS_MAX_WORDS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(300);
    let chest_ratio = std::env::var("CLIP_CAPTIONS_CHEST_RATIO")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.65)
        .clamp(0.2, 0.9);
    let margin_offset = std::env::var("CLIP_CAPTIONS_MARGIN_OFFSET")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let margin_offset_px = resolve_caption_pixels(out_h, margin_offset, 0.0);
    let debug = std::env::var("CLIP_CAPTIONS_DEBUG")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false);

    Some(CaptionConfig {
        position,
        font,
        font_size,
        color,
        outline_color,
        outline,
        min_word_secs,
        max_words,
        chest_ratio,
        margin_offset_px,
        debug,
    })
}

pub(crate) fn build_caption_words(
    mut words: Vec<WordTiming>,
    cfg: &CaptionConfig,
    duration_secs: Option<f32>,
) -> Vec<CaptionWord> {
    words.sort_by(|a, b| a.t0.partial_cmp(&b.t0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::new();
    let min_word = cfg.min_word_secs.max(0.04);
    for (idx, word) in words.iter().enumerate() {
        if out.len() >= cfg.max_words {
            break;
        }
        if word.norm.is_empty() {
            continue;
        }
        let text = word.text.trim();
        if text.is_empty() {
            continue;
        }
        let start = word.t0.max(0.0);
        let mut end = word.t1.max(start + min_word);
        if let Some(next) = words.get(idx + 1) {
            if next.t0.is_finite() && next.t0 > start {
                let cap = (next.t0 - 0.01).max(start + min_word);
                end = end.min(cap);
            }
        }
        if let Some(limit) = duration_secs {
            if start >= limit {
                continue;
            }
            end = end.min(limit);
        }
        if end <= start {
            continue;
        }
        out.push(CaptionWord {
            text: text.to_string(),
            start,
            end,
        });
    }
    out
}

pub(crate) fn caption_y_for_layout(
    cfg: &CaptionConfig,
    layout_is_stacked: bool,
    stacked_dims: Option<StackedLayoutDims>,
    out_h: u32,
) -> f32 {
    let frame_h = if layout_is_stacked {
        stacked_dims.map(|d| d.face_h).unwrap_or(out_h) as f32
    } else {
        out_h as f32
    };
    let mut y = match cfg.position {
        CaptionPosition::Margin if layout_is_stacked => {
            frame_h - cfg.font_size * 0.6 + cfg.margin_offset_px
        }
        _ => frame_h * cfg.chest_ratio - cfg.font_size * 0.5,
    };
    let max_y = (out_h as f32 - cfg.font_size).max(0.0);
    if !y.is_finite() {
        y = 0.0;
    }
    y.clamp(0.0, max_y)
}

pub(crate) fn build_caption_drawtext_chain(
    words: &[CaptionWord],
    cfg: &CaptionConfig,
    y: f32,
) -> String {
    if words.is_empty() {
        return String::new();
    }
    let mut filters = Vec::with_capacity(words.len());
    let font_arg = cfg.font.as_ref().map(|f| drawtext_font_arg(f));
    let fontsize = cfg.font_size.round().max(8.0) as u32;
    for word in words {
        let text = escape_drawtext_value(&word.text);
        let mut parts = Vec::new();
        parts.push(format!("drawtext=text='{}'", text));
        if let Some(arg) = &font_arg {
            parts.push(arg.clone());
        }
        parts.push(format!("fontcolor={}", cfg.color));
        parts.push(format!("fontsize={}", fontsize));
        if cfg.outline > 0 {
            parts.push(format!("borderw={}", cfg.outline));
            parts.push(format!("bordercolor={}", cfg.outline_color));
        }
        parts.push("x=(w-text_w)/2".to_string());
        parts.push(format!("y={:.1}", y));
        parts.push(format!(
            "enable='between(t,{:.3},{:.3})'",
            word.start, word.end
        ));
        filters.push(parts.join(":"));
    }
    filters.join(",")
}

pub(crate) fn wrap_caption_text(text: &str, max_len: usize) -> String {
    let text = text.trim();
    if text.len() <= max_len {
        return text.to_string();
    }
    let split_at = text
        .char_indices()
        .take_while(|(_, ch)| !ch.is_control())
        .filter_map(|(idx, ch)| if ch == ' ' { Some(idx) } else { None })
        .filter(|idx| *idx <= max_len)
        .last();
    if let Some(idx) = split_at {
        let left = text[..idx].trim();
        let right = text[idx + 1..].trim();
        if !left.is_empty() && !right.is_empty() {
            return format!("{left}\n{right}");
        }
    }
    text.to_string()
}

pub(crate) fn build_srt_from_payload(
    payload: &TranscriptPayload,
    duration_secs: Option<f32>,
) -> Option<String> {
    let duration = duration_secs.filter(|v| v.is_finite() && *v > 0.0)?;
    let mut words = payload.words.clone();
    if words.is_empty() {
        let text = normalize_title_whitespace(&payload.text);
        if text.is_empty() {
            return None;
        }
        let line = wrap_caption_text(&text, 42);
        return Some(format!(
            "1\n{} --> {}\n{}\n",
            format_srt_time(0.0),
            format_srt_time(duration),
            line
        ));
    }

    words.sort_by(|a, b| {
        a.t0.partial_cmp(&b.t0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let max_gap = 0.8;
    let max_span = 4.0;
    let max_words = 8usize;
    let min_word = 0.12;

    let mut cues: Vec<(f32, f32, String)> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut cue_start = 0.0;
    let mut last_end = 0.0;

    for word in words {
        let text = word.text.trim();
        if text.is_empty() {
            continue;
        }
        let start = word.t0.max(0.0);
        let mut end = word.t1.max(start + min_word);
        if end > duration {
            end = duration;
        }
        if end <= start {
            continue;
        }
        let gap = if current.is_empty() { 0.0 } else { start - last_end };
        let span = if current.is_empty() { 0.0 } else { end - cue_start };
        let should_break =
            !current.is_empty() && (gap > max_gap || current.len() >= max_words || span > max_span);
        if should_break {
            let line = current.join(" ").trim().to_string();
            if !line.is_empty() {
                let cue_end = last_end.max(cue_start + min_word);
                cues.push((cue_start, cue_end.min(duration), line));
            }
            current.clear();
        }
        if current.is_empty() {
            cue_start = start;
        }
        current.push(text.to_string());
        last_end = end;
    }

    if !current.is_empty() {
        let line = current.join(" ").trim().to_string();
        if !line.is_empty() {
            let cue_end = last_end.max(cue_start + min_word);
            cues.push((cue_start, cue_end.min(duration), line));
        }
    }

    if cues.is_empty() {
        return None;
    }

    let mut out = String::new();
    for (idx, (start, end, text)) in cues.iter().enumerate() {
        let line = wrap_caption_text(text, 42);
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            idx + 1,
            format_srt_time(*start),
            format_srt_time(*end),
            line
        ));
    }
    Some(out)
}

pub(crate) fn write_temp_srt(contents: &str) -> Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis();
    let pid = std::process::id();
    let filename = format!("autoclip_cc_{pid}_{stamp}.srt");
    let path = std::env::temp_dir().join(filename);
    fs::write(&path, contents).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

pub(crate) fn subtitle_codec_for_output(out_path: &Path) -> &'static str {
    let ext = out_path
        .extension()
        .and_then(|v| v.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "mp4" | "m4v" => "mov_text",
        "mkv" => "srt",
        _ => "mov_text",
    }
}

fn escape_drawtext_value(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            ':' => out.push_str("\\:"),
            ',' => out.push_str("\\,"),
            '%' => out.push_str("\\%"),
            '\n' | '\r' => out.push(' '),
            _ => out.push(ch),
        }
    }
    out
}

fn drawtext_font_arg(font: &str) -> String {
    let value = escape_drawtext_value(font);
    if Path::new(font).exists() || font.contains('/') || font.contains('\\') {
        format!("fontfile='{}'", value)
    } else {
        format!("font='{}'", value)
    }
}

pub(crate) fn format_srt_time(secs: f32) -> String {
    let secs = secs.max(0.0);
    let total_ms = (secs * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let s = total_secs % 60;
    let total_mins = total_secs / 60;
    let m = total_mins % 60;
    let h = total_mins / 60;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}
