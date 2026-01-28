//! Caption and subtitle helpers for open/closed captions.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::clip_layout::StackedLayoutDims;
use crate::stream_audio_wake::TranscriptPayload;
#[cfg(test)]
use crate::stream_audio_wake::WordTiming;
use crate::text_utils::normalize_title_whitespace;
use crate::parse_bool;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CaptionPosition {
    Margin,
    Chest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CaptionRender {
    Drawtext,
    Subtitles,
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

#[cfg(test)]
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

pub(crate) fn caption_render_mode() -> CaptionRender {
    match std::env::var("CLIP_CAPTIONS_RENDER")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("subtitles") | Some("srt") => CaptionRender::Subtitles,
        _ => CaptionRender::Drawtext,
    }
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
        if let Some(fallback) = fallback_caption_font_any() {
            return Some(fallback);
        }
    }
    None
}

#[cfg(windows)]
fn fallback_caption_font_any() -> Option<String> {
    let windows_dir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".to_string());
    let font_dir = Path::new(&windows_dir).join("Fonts");
    let entries = fs::read_dir(&font_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|v| v.to_str()) {
            let ext = ext.to_ascii_lowercase();
            if matches!(ext.as_str(), "ttf" | "otf" | "ttc") {
                return Some(path.to_string_lossy().into_owned());
            }
        }
    }
    None
}

#[cfg(windows)]
fn resolve_caption_font(value: &str) -> Option<String> {
    let font = value.trim();
    if font.is_empty() {
        return None;
    }
    let path = Path::new(font);
    if path.exists() {
        if let Ok(meta) = fs::metadata(path) {
            if meta.len() >= 4096 {
                return Some(font.to_string());
            }
        }
        return None;
    }
    let windows_dir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".to_string());
    let font_dir = Path::new(&windows_dir).join("Fonts");
    if font.contains('/') || font.contains('\\') {
        let candidate = font_dir.join(path.file_name()?);
        if candidate.exists() {
            return Some(candidate.to_string_lossy().into_owned());
        }
    }
    let font_lower = font.to_ascii_lowercase();
    let candidates: &[&str] = match font_lower.as_str() {
        "consolas" => &["consola.ttf", "consolab.ttf", "consolai.ttf", "consolaz.ttf"],
        "segoe ui" | "segoeui" => &[
            "segoeui.ttf",
            "segoeuib.ttf",
            "segoeuii.ttf",
            "segoeuiz.ttf",
            "seguisb.ttf",
        ],
        "arial" => &["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"],
        "calibri" => &["calibri.ttf", "calibrib.ttf", "calibrii.ttf", "calibriz.ttf"],
        "tahoma" => &["tahoma.ttf", "tahomabd.ttf"],
        "verdana" => &["verdana.ttf", "verdanab.ttf", "verdanai.ttf", "verdanaz.ttf"],
        "trebuchet ms" | "trebuchet" => &["trebuc.ttf", "trebucbd.ttf", "trebucit.ttf", "trebucbi.ttf"],
        "times new roman" | "times" => &["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"],
        "courier new" | "courier" => &["cour.ttf", "courbd.ttf", "couri.ttf", "courbi.ttf"],
        _ => &[],
    };
    for file in candidates {
        let candidate = font_dir.join(file);
        if candidate.exists() {
            return Some(candidate.to_string_lossy().into_owned());
        }
    }
    let base = font_lower
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    for ext in ["ttf", "otf", "ttc"] {
        let candidate = font_dir.join(format!("{base}.{ext}"));
        if candidate.exists() {
            return Some(candidate.to_string_lossy().into_owned());
        }
    }
    None
}

#[cfg(not(windows))]
fn resolve_caption_font(_value: &str) -> Option<String> {
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
    if let Some(raw) = font.take() {
        font = match resolve_caption_font(&raw) {
            Some(resolved) => Some(resolved),
            None => {
                if cfg!(windows) {
                    None
                } else {
                    Some(raw)
                }
            }
        };
    }
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

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn build_caption_words(
    mut words: Vec<WordTiming>,
    cfg: &CaptionConfig,
    duration_secs: Option<f32>,
) -> Vec<CaptionWord> {
    words.sort_by(|a, b| a.t0.partial_cmp(&b.t0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::new();
    let min_word = cfg.min_word_secs.max(0.04);
    const SAME_START_EPS: f32 = 0.001;
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

#[cfg(test)]
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

fn wrap_caption_text_by_width(
    text: &str,
    max_width_px: f32,
    font_name: &str,
    font_size: f32,
) -> String {
    if text.trim().is_empty() || max_width_px <= 0.0 {
        return text.trim().to_string();
    }
    let Some(_) = measure_text_width_px(text, font_name, font_size) else {
        return wrap_caption_text(text, 42);
    };
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let candidate = if current.is_empty() {
            word.to_string()
        } else {
            format!("{} {}", current, word)
        };
        if fits_width(&candidate, max_width_px, font_name, font_size) {
            current = candidate;
            continue;
        }
        if current.is_empty() {
            let parts = split_word_by_width(word, max_width_px, font_name, font_size);
            let parts_len = parts.len();
            for (idx, part) in parts.into_iter().enumerate() {
                if idx + 1 == parts_len {
                    current = part;
                } else {
                    lines.push(part);
                }
            }
        } else {
            lines.push(current);
            current = word.to_string();
            if !fits_width(&current, max_width_px, font_name, font_size) {
                let parts = split_word_by_width(&current, max_width_px, font_name, font_size);
                let parts_len = parts.len();
                current.clear();
                for (idx, part) in parts.into_iter().enumerate() {
                    if idx + 1 == parts_len {
                        current = part;
                    } else {
                        lines.push(part);
                    }
                }
            }
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines.join("\n")
}

fn fits_width(text: &str, max_width_px: f32, font_name: &str, font_size: f32) -> bool {
    match measure_text_width_px(text, font_name, font_size) {
        Some(width) => width <= max_width_px,
        None => true,
    }
}

fn split_word_by_width(
    word: &str,
    max_width_px: f32,
    font_name: &str,
    font_size: f32,
) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in word.chars() {
        let candidate = format!("{}{}", current, ch);
        if fits_width(&candidate, max_width_px, font_name, font_size) || current.is_empty() {
            current = candidate;
        } else {
            parts.push(current);
            current = ch.to_string();
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

fn max_line_width_px(text: &str, font_name: &str, font_size: f32) -> Option<f32> {
    let mut max_width = 0.0;
    for line in text.split('\n') {
        let width = measure_text_width_px(line, font_name, font_size)?;
        if width > max_width {
            max_width = width;
        }
    }
    Some(max_width)
}

#[cfg(windows)]
fn measure_text_width_px(text: &str, font_name: &str, font_size: f32) -> Option<f32> {
    use windows_sys::Win32::Foundation::SIZE;
    use windows_sys::Win32::Graphics::Gdi::{
        CreateCompatibleDC, CreateFontW, DeleteDC, DeleteObject, GetTextExtentPoint32W,
        SelectObject, CLIP_DEFAULT_PRECIS, DEFAULT_CHARSET, DEFAULT_PITCH, DEFAULT_QUALITY,
        FF_DONTCARE, FW_NORMAL, OUT_DEFAULT_PRECIS,
    };
    if text.is_empty() {
        return Some(0.0);
    }
    let hdc = unsafe { CreateCompatibleDC(0) };
    if hdc == 0 {
        return None;
    }
    let height = -(font_size.round().max(1.0) as i32);
    let font_w = to_wide(font_name);
    let hfont = unsafe {
        CreateFontW(
            height,
            0,
            0,
            0,
            FW_NORMAL as i32,
            0,
            0,
            0,
            DEFAULT_CHARSET as u32,
            OUT_DEFAULT_PRECIS as u32,
            CLIP_DEFAULT_PRECIS as u32,
            DEFAULT_QUALITY as u32,
            (DEFAULT_PITCH | FF_DONTCARE) as u32,
            font_w.as_ptr(),
        )
    };
    if hfont == 0 {
        unsafe {
            DeleteDC(hdc);
        }
        return None;
    }
    let old = unsafe { SelectObject(hdc, hfont as isize) };
    let text_w = to_wide(text);
    let mut size = SIZE { cx: 0, cy: 0 };
    let ok = unsafe {
        GetTextExtentPoint32W(hdc, text_w.as_ptr(), (text_w.len() - 1) as i32, &mut size)
    };
    unsafe {
        SelectObject(hdc, old);
        DeleteObject(hfont as isize);
        DeleteDC(hdc);
    }
    if ok == 0 {
        None
    } else {
        Some(size.cx as f32)
    }
}

#[cfg(not(windows))]
fn measure_text_width_px(_text: &str, _font_name: &str, _font_size: f32) -> Option<f32> {
    None
}

#[cfg(windows)]
fn to_wide(text: &str) -> Vec<u16> {
    let mut out: Vec<u16> = text.encode_utf16().collect();
    out.push(0);
    out
}

fn is_special_whisper_token(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    (trimmed.starts_with("[_") && trimmed.ends_with(']'))
        || (trimmed.starts_with("<|") && trimmed.ends_with("|>"))
}

fn strip_special_whisper_tokens(text: &str) -> String {
    let mut keep = Vec::new();
    for token in text.split_whitespace() {
        if !is_special_whisper_token(token) {
            keep.push(token);
        }
    }
    keep.join(" ").trim().to_string()
}

fn is_punct_only(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.is_empty() && trimmed.chars().all(|ch| !ch.is_ascii_alphanumeric())
}

fn build_caption_cues(
    payload: &TranscriptPayload,
    duration_secs: Option<f32>,
    min_word_secs: f32,
    max_words: usize,
) -> Option<Vec<(f32, f32, String)>> {
    let duration = duration_secs.filter(|v| v.is_finite() && *v > 0.0)?;
    let mut words = payload.words.clone();
    if words.is_empty() {
        let cleaned = strip_special_whisper_tokens(&payload.text);
        let text = normalize_title_whitespace(&cleaned);
        if text.is_empty() {
            return None;
        }
        return Some(vec![(0.0, duration, text)]);
    }

    words.sort_by(|a, b| {
        a.t0.partial_cmp(&b.t0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let min_word = min_word_secs.max(0.04);
    let bucket_secs = std::env::var("CLIP_CAPTIONS_BUCKET_SECS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0);
    let max_words = max_words.max(1);

    let mut cues: Vec<(f32, f32, String)> = Vec::new();
    let mut added = 0usize;
    let mut current_bucket: Option<u64> = None;
    const SAME_START_EPS: f32 = 0.001;

    for (idx, word) in words.iter().enumerate() {
        if added >= max_words {
            break;
        }
        let raw = word.text.trim();
        if raw.is_empty() || is_special_whisper_token(raw) {
            continue;
        }
        let cleaned = strip_special_whisper_tokens(raw);
        if cleaned.is_empty() {
            continue;
        }
        if let Some(bucket_size) = bucket_secs {
            let bucket = ((word.t0.max(0.0) / bucket_size).floor() as u64).max(0);
            if current_bucket != Some(bucket) {
                current_bucket = Some(bucket);
                let start = (bucket as f32 * bucket_size).min(duration);
                let mut end = word.t1.max(start + min_word);
                if end > duration {
                    end = duration;
                }
                if end > start {
                    cues.push((start, end, String::new()));
                }
            }
            if let Some(last) = cues.last_mut() {
                if is_punct_only(&cleaned) {
                    last.2.push_str(&cleaned);
                } else {
                    if !last.2.is_empty() {
                        last.2.push(' ');
                    }
                    last.2.push_str(&cleaned);
                }
                last.1 = last.1.max(word.t1).max(last.0 + min_word).min(duration);
                if !is_punct_only(&cleaned) {
                    added = added.saturating_add(1);
                }
            }
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
        if end > duration {
            end = duration;
        }
        if end <= start {
            continue;
        }
        if is_punct_only(&cleaned) {
            if let Some(last) = cues.last_mut() {
                last.1 = last.1.max(end);
                last.2.push_str(&cleaned);
            }
            continue;
        }
        if let Some(last) = cues.last_mut() {
            if (last.0 - start).abs() <= SAME_START_EPS {
                if !last.2.is_empty() {
                    last.2.push(' ');
                }
                last.2.push_str(&cleaned);
                last.1 = last.1.max(end);
                added = added.saturating_add(1);
                continue;
            }
        }
        cues.push((start, end, cleaned));
        added = added.saturating_add(1);
    }

    if cues.is_empty() {
        None
    } else {
        Some(cues)
    }
}

#[cfg(test)]
pub(crate) fn build_srt_from_payload(
    payload: &TranscriptPayload,
    duration_secs: Option<f32>,
) -> Option<String> {
    build_srt_from_payload_with_limits(payload, duration_secs, 0.12, usize::MAX)
}

pub(crate) fn build_srt_from_payload_with_limits(
    payload: &TranscriptPayload,
    duration_secs: Option<f32>,
    min_word_secs: f32,
    max_words: usize,
) -> Option<String> {
    let cues = build_caption_cues(payload, duration_secs, min_word_secs, max_words)?;
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

pub(crate) fn build_ass_from_payload_with_limits(
    payload: &TranscriptPayload,
    duration_secs: Option<f32>,
    cfg: &CaptionConfig,
    y: f32,
    out_w: u32,
    out_h: u32,
) -> Option<String> {
    let cues = build_caption_cues(payload, duration_secs, cfg.min_word_secs, cfg.max_words)?;
    if cues.is_empty() {
        return None;
    }

    let width_ratio = std::env::var("CLIP_CAPTIONS_WIDTH_RATIO")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.clamp(0.5, 1.0))
        .unwrap_or(0.85);
    let glyph_ratio = std::env::var("CLIP_CAPTIONS_GLYPH_RATIO")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.clamp(0.4, 1.2))
        .unwrap_or(0.7);
    let scale_lock = std::env::var("CLIP_CAPTIONS_SCALE_LOCK")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .map(|v| !matches!(v.as_str(), "0" | "false" | "off" | "none"))
        .unwrap_or(false);
    let min_scale = std::env::var("CLIP_CAPTIONS_SCALE_MIN")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.clamp(0.2, 1.0))
        .unwrap_or(0.5);
    let max_scale = std::env::var("CLIP_CAPTIONS_SCALE_MAX")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.clamp(1.0, 6.0))
        .unwrap_or(2.5);

    let base_size = cfg.font_size.round().max(8.0);
    let mut max_w = out_w as f32 * width_ratio;
    if cfg.outline > 0 {
        max_w = (max_w - (cfg.outline as f32 * 2.0)).max(8.0);
    }

    let font_name = cfg
        .font
        .as_ref()
        .and_then(|font| {
            let font_path = Path::new(font);
            if font_path.exists() {
                font_name_from_path(font_path)
            } else {
                Some(font.clone())
            }
        })
        .unwrap_or_else(|| "Arial".to_string());

    let primary = ass_color_from_value(&cfg.color).unwrap_or_else(|| "&H00FFFFFF".to_string());
    let outline = ass_color_from_value(&cfg.outline_color).unwrap_or_else(|| "&H00000000".to_string());
    let alignment = 8;
    let margin_v = y.round().clamp(0.0, out_h as f32) as u32;

    let mut out = String::new();
    out.push_str("[Script Info]\n");
    out.push_str("ScriptType: v4.00+\n");
    out.push_str(&format!("PlayResX: {out_w}\n"));
    out.push_str(&format!("PlayResY: {out_h}\n"));
    out.push_str("ScaledBorderAndShadow: yes\n\n");
    out.push_str("[V4+ Styles]\n");
    out.push_str("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n");
    out.push_str(&format!(
        "Style: Default,{font_name},{base_size:.0},{primary},{primary},{outline},&H00000000,0,0,0,0,100,100,0,0,1,{outline_width},0,{alignment},10,10,{margin_v},1\n\n",
        outline_width = cfg.outline
    ));
    out.push_str("[Events]\n");
    out.push_str("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n");

    let fixed_size = if scale_lock {
        let max_width = cues
            .iter()
            .map(|(_, _, text)| {
                let line = wrap_caption_text_by_width(text, max_w, &font_name, base_size);
                if let Some(width) = max_line_width_px(&line, &font_name, base_size) {
                    width
                } else {
                    line.split('\n')
                        .map(|segment| segment.chars().map(char_width_units).sum::<f32>())
                        .fold(0.0, f32::max)
                        * glyph_ratio
                }
            })
            .fold(0.0, f32::max);
        if max_width > 0.0 {
            let scaled = max_w / max_width * base_size;
            scaled.clamp(base_size * min_scale, base_size * max_scale)
        } else {
            base_size
        }
    } else {
        base_size
    };

    for (start, end, text) in cues {
        let line = wrap_caption_text_by_width(&text, max_w, &font_name, base_size);
        let mut escaped = escape_ass_text(&line);
        escaped = escaped.replace('\n', "\\N");
        let size = if scale_lock {
            fixed_size
        } else {
            if let Some(width) = max_line_width_px(&line, &font_name, base_size) {
                let scaled = max_w / width * base_size;
                scaled.clamp(base_size * min_scale, base_size * max_scale)
            } else {
                let max_units = line
                    .split('\n')
                    .map(|segment| segment.chars().map(char_width_units).sum::<f32>())
                    .fold(0.0, f32::max);
                if max_units > 0.0 {
                    let scaled = max_w / (max_units * glyph_ratio);
                    scaled.clamp(base_size * min_scale, base_size * max_scale)
                } else {
                    base_size
                }
            }
        };
        let x_center = out_w as f32 / 2.0;
        let override_tag = format!("{{\\an8\\pos({:.0},{:.0})\\fs{:.0}}}", x_center, y, size);
        out.push_str(&format!(
            "Dialogue: 0,{},{},Default,,0,0,0,,{}{}\n",
            format_ass_time(start),
            format_ass_time(end),
            override_tag,
            escaped
        ));
        if cfg.debug {
            let width = max_line_width_px(&line, &font_name, size)
                .unwrap_or_else(|| line.chars().map(char_width_units).sum::<f32>() * size * glyph_ratio);
            eprintln!(
                "captions: ass cue {:.2}-{:.2} size {:.1}px width {:.1}/{:.1} text '{}'",
                start,
                end,
                size,
                width,
                max_w,
                line.replace('\n', " | ")
            );
        }
    }

    Some(out)
}

#[allow(dead_code)]
pub(crate) fn build_drawtext_caption_filter(
    payload: &TranscriptPayload,
    duration_secs: Option<f32>,
    cfg: &CaptionConfig,
    y: f32,
    out_w: u32,
    out_h: u32,
) -> Option<String> {
    let result = build_drawtext_caption_filter_with_limit(payload, duration_secs, cfg, y, out_w, out_h);
    match result {
        DrawtextBuildResult::Chain { chain, .. } => Some(chain),
        DrawtextBuildResult::TooManyCues { .. } => None,
    }
}

pub(crate) enum DrawtextBuildResult {
    Chain { chain: String, cue_count: usize },
    TooManyCues { cue_count: usize, max_cues: usize },
}

pub(crate) fn build_drawtext_caption_filter_with_limit(
    payload: &TranscriptPayload,
    duration_secs: Option<f32>,
    cfg: &CaptionConfig,
    y: f32,
    out_w: u32,
    out_h: u32,
) -> DrawtextBuildResult {
    let Some(cues) = build_caption_cues(payload, duration_secs, cfg.min_word_secs, cfg.max_words) else {
        return DrawtextBuildResult::Chain {
            chain: String::new(),
            cue_count: 0,
        };
    };
    if cues.is_empty() {
        return DrawtextBuildResult::Chain {
            chain: String::new(),
            cue_count: 0,
        };
    }

    let max_drawtext_cues = std::env::var("CLIP_CAPTIONS_DRAWTEXT_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(120);
    if cues.len() > max_drawtext_cues {
        return DrawtextBuildResult::TooManyCues {
            cue_count: cues.len(),
            max_cues: max_drawtext_cues,
        };
    }

    let width_ratio = std::env::var("CLIP_CAPTIONS_WIDTH_RATIO")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.clamp(0.5, 1.0))
        .unwrap_or(0.85);
    let glyph_ratio = std::env::var("CLIP_CAPTIONS_GLYPH_RATIO")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.clamp(0.4, 1.2))
        .unwrap_or(0.7);
    let scale_lock = std::env::var("CLIP_CAPTIONS_SCALE_LOCK")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .map(|v| !matches!(v.as_str(), "0" | "false" | "off" | "none"))
        .unwrap_or(false);
    let min_scale = std::env::var("CLIP_CAPTIONS_SCALE_MIN")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.clamp(0.2, 1.0))
        .unwrap_or(0.5);
    let max_scale = std::env::var("CLIP_CAPTIONS_SCALE_MAX")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.clamp(1.0, 6.0))
        .unwrap_or(2.5);
    let max_w = (out_w as f32 * width_ratio - (cfg.outline as f32 * 2.0)).max(8.0);
    let base_size = cfg.font_size.round().max(8.0);
    let font_name = cfg
        .font
        .as_ref()
        .and_then(|font| {
            let font_path = Path::new(font);
            if font_path.exists() {
                font_name_from_path(font_path)
            } else {
                Some(font.clone())
            }
        })
        .unwrap_or_else(|| "Arial".to_string());
    let y = y.round().clamp(0.0, out_h as f32);
    let fixed_size = if scale_lock {
        let max_width = cues
            .iter()
            .map(|(_, _, text)| {
                let line = wrap_caption_text_by_width(text, max_w, &font_name, base_size);
                if let Some(width) = max_line_width_px(&line, &font_name, base_size) {
                    width
                } else {
                    line.split('\n')
                        .map(|segment| segment.chars().map(char_width_units).sum::<f32>())
                        .fold(0.0, f32::max)
                        * glyph_ratio
                }
            })
            .fold(0.0, f32::max);
        if max_width > 0.0 {
            let scaled = max_w / max_width * base_size;
            scaled.clamp(base_size * min_scale, base_size * max_scale)
        } else {
            base_size
        }
    } else {
        base_size
    };

    let cue_count = cues.len();
    let mut chain: Vec<String> = Vec::new();
    for (start, end, text) in cues {
        let line = wrap_caption_text_by_width(&text, max_w, &font_name, base_size);
        let mut text_value = escape_drawtext_value(&line);
        text_value = text_value.replace('\n', "\\n");

        let mut parts: Vec<String> = Vec::new();
        parts.push(format!("text='{text_value}'"));
        parts.push("x=(w-text_w)/2".to_string());
        parts.push(format!("y={}", y as u32));
        if scale_lock {
            parts.push(format!("fontsize={fixed_size:.2}"));
        } else {
            if let Some(width) = max_line_width_px(&line, &font_name, base_size) {
                let scaled = max_w / width * base_size;
                let size = scaled.clamp(base_size * min_scale, base_size * max_scale);
                parts.push(format!("fontsize={size:.2}"));
            } else {
                parts.push(format!(
                    "fontsize=(max({base_size:.2}*{min_scale:.2}\\,min({base_size:.2}*{max_scale:.2}\\,{max_w:.2}/max(text_w\\,1)*{base_size:.2})))"
                ));
            }
        }
        if let Some(font) = cfg.font.as_ref() {
            parts.push(drawtext_font_arg(font));
        }
        parts.push(format!("fontcolor={}", cfg.color));
        if cfg.outline > 0 {
            parts.push(format!("borderw={}", cfg.outline));
        }
        parts.push(format!("bordercolor={}", cfg.outline_color));
        parts.push("shadowcolor=black@0.0".to_string());
        parts.push("shadowx=0".to_string());
        parts.push("shadowy=0".to_string());
        parts.push(format!("enable='between(t,{start:.3},{end:.3})'"));

        chain.push(format!("drawtext={}", parts.join(":")));
        if cfg.debug {
            let width = max_line_width_px(&line, &font_name, base_size)
                .unwrap_or_else(|| line.chars().map(char_width_units).sum::<f32>() * base_size * glyph_ratio);
            eprintln!(
                "captions: drawtext cue {:.2}-{:.2} width {:.1}/{:.1} text '{}'",
                start,
                end,
                width,
                max_w,
                line.replace('\n', " | ")
            );
        }
    }

    DrawtextBuildResult::Chain {
        chain: chain.join(","),
        cue_count,
    }
}

pub(crate) fn adjust_caption_font_size(
    cfg: &CaptionConfig,
    payload: &TranscriptPayload,
    out_w: u32,
) -> CaptionConfig {
    let mut adjusted = cfg.clone();
    let width_ratio = std::env::var("CLIP_CAPTIONS_WIDTH_RATIO")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.clamp(0.5, 1.0))
        .unwrap_or(0.85);
    let glyph_ratio = std::env::var("CLIP_CAPTIONS_GLYPH_RATIO")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.clamp(0.4, 1.2))
        .unwrap_or(0.7);
    let max_units = payload
        .words
        .iter()
        .map(|word| word.text.trim())
        .filter(|text| !text.is_empty())
        .map(|text| text.chars().map(char_width_units).sum::<f32>())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);
    if max_units <= 0.0 || out_w == 0 {
        return adjusted;
    }
    let mut width_budget = out_w as f32 * width_ratio;
    if adjusted.outline > 0 {
        width_budget = (width_budget - (adjusted.outline as f32 * 2.0)).max(8.0);
    }
    let estimated = max_units * adjusted.font_size * glyph_ratio;
    if estimated > width_budget {
        let new_size = width_budget / (max_units * glyph_ratio);
        adjusted.font_size = new_size.clamp(10.0, adjusted.font_size);
    }
    adjusted
}

fn char_width_units(ch: char) -> f32 {
    if ch.is_ascii_uppercase() {
        1.1
    } else if ch.is_ascii_lowercase() {
        1.0
    } else if ch.is_ascii_digit() {
        0.9
    } else if ch.is_whitespace() {
        0.4
    } else {
        0.8
    }
}

fn ass_color_from_rgb(r: u8, g: u8, b: u8) -> String {
    format!("&H00{:02X}{:02X}{:02X}", b, g, r)
}

fn parse_hex_color(value: &str) -> Option<(u8, u8, u8)> {
    let trimmed = value.trim().trim_start_matches('#');
    if trimmed.len() == 6 {
        let r = u8::from_str_radix(&trimmed[0..2], 16).ok()?;
        let g = u8::from_str_radix(&trimmed[2..4], 16).ok()?;
        let b = u8::from_str_radix(&trimmed[4..6], 16).ok()?;
        return Some((r, g, b));
    }
    None
}

fn parse_named_color(value: &str) -> Option<(u8, u8, u8)> {
    match value.trim().to_ascii_lowercase().as_str() {
        "white" => Some((255, 255, 255)),
        "black" => Some((0, 0, 0)),
        "red" => Some((255, 0, 0)),
        "green" => Some((0, 255, 0)),
        "blue" => Some((0, 0, 255)),
        "yellow" => Some((255, 255, 0)),
        "cyan" => Some((0, 255, 255)),
        "magenta" => Some((255, 0, 255)),
        "gray" | "grey" => Some((128, 128, 128)),
        _ => None,
    }
}

fn ass_color_from_value(value: &str) -> Option<String> {
    let (r, g, b) = parse_hex_color(value).or_else(|| parse_named_color(value))?;
    Some(ass_color_from_rgb(r, g, b))
}

fn escape_filter_value(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            ':' => out.push_str("\\:"),
            ',' => out.push_str("\\,"),
            _ => out.push(ch),
        }
    }
    out
}

fn font_name_from_path(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_string_lossy();
    let cleaned = stem.replace('_', " ").replace('-', " ").trim().to_string();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

pub(crate) fn build_caption_subtitles_filter(
    srt_path: &Path,
    cfg: &CaptionConfig,
    y: f32,
    out_w: u32,
    out_h: u32,
) -> String {
    let mut parts = Vec::new();
    let path = srt_path.to_string_lossy().replace('\\', "/");
    parts.push(format!("subtitles='{}'", escape_filter_value(&path)));
    if out_w > 0 && out_h > 0 {
        parts.push(format!("original_size={}x{}", out_w, out_h));
    }

    let mut style_parts: Vec<String> = Vec::new();
    let font_size = cfg.font_size.round().max(8.0) as u32;
    style_parts.push(format!("FontSize={font_size}"));
    style_parts.push("BorderStyle=1".to_string());
    if cfg.outline > 0 {
        style_parts.push(format!("Outline={}", cfg.outline));
    }
    style_parts.push("Shadow=0".to_string());

    let alignment = 8;
    style_parts.push(format!("Alignment={alignment}"));
    let margin_v = y.round().clamp(0.0, out_h as f32) as u32;
    style_parts.push(format!("MarginV={margin_v}"));

    if let Some(color) = ass_color_from_value(&cfg.color) {
        style_parts.push(format!("PrimaryColour={color}"));
    }
    if let Some(color) = ass_color_from_value(&cfg.outline_color) {
        style_parts.push(format!("OutlineColour={color}"));
    }

    if let Some(font) = cfg.font.as_ref() {
        let font_path = Path::new(font);
        if font_path.exists() {
            if let Some(parent) = font_path.parent() {
                let dir = parent.to_string_lossy().replace('\\', "/");
                parts.push(format!("fontsdir='{}'", escape_filter_value(&dir)));
            }
            if let Some(name) = font_name_from_path(font_path) {
                style_parts.push(format!("FontName={}", escape_filter_value(&name)));
            }
        } else {
            style_parts.push(format!("FontName={}", escape_filter_value(font)));
        }
    }

    if !style_parts.is_empty() {
        let style = style_parts.join(",");
        parts.push(format!("force_style='{}'", escape_filter_value(&style)));
    }

    parts.join(":")
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

pub(crate) fn write_temp_ass(contents: &str) -> Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis();
    let pid = std::process::id();
    let filename = format!("autoclip_cc_{pid}_{stamp}.ass");
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
            '\n' => out.push('\n'),
            '\r' => out.push(' '),
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

fn format_ass_time(secs: f32) -> String {
    let secs = secs.max(0.0);
    let total_cs = (secs * 100.0).round() as u64;
    let cs = total_cs % 100;
    let total_secs = total_cs / 100;
    let s = total_secs % 60;
    let total_mins = total_secs / 60;
    let m = total_mins % 60;
    let h = total_mins / 60;
    format!("{h}:{m:02}:{s:02}.{cs:02}")
}

fn escape_ass_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            _ => out.push(ch),
        }
    }
    out
}
