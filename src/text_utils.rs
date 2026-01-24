//! Text helpers for transcript cleanup and title normalization.
//!
//! These are used by the metadata and captioning pipeline to keep strings
//! short, readable, and safe for filenames.

use crate::truncate_str;

/// Trim whitespace and cap the transcript length, preserving a human-readable
/// ellipsis for truncation.
pub fn trim_transcript(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.len() <= max_chars {
        return trimmed.to_string();
    }
    truncate_str(trimmed, max_chars)
}

/// Collapse repeated whitespace into single spaces and trim ends.
pub fn normalize_title_whitespace(title: &str) -> String {
    title
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

/// Sanitize a title into a filesystem-safe slug.
///
/// Only ASCII letters/digits and `-` are preserved. Other runs collapse into
/// underscores.
pub fn sanitize_title_for_filename(title: &str) -> String {
    let mut out = String::new();
    let mut last_sep = false;
    for ch in title.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_sep = false;
        } else if ch == '-' {
            out.push('-');
            last_sep = false;
        } else if !last_sep {
            out.push('_');
            last_sep = true;
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        String::new()
    } else {
        trimmed.to_string()
    }
}

/// Generate a fallback title from transcript text, honoring `max_chars`.
///
/// This keeps whitespace tidy and applies the same truncation rules used in
/// other metadata paths.
pub fn fallback_title_from_transcript(text: &str, max_chars: usize) -> String {
    let mut title = normalize_title_whitespace(text);
    if title.is_empty() {
        return title;
    }
    if title.len() > max_chars {
        title = truncate_str(&title, max_chars);
    }
    title
}
