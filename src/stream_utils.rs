//! Stream list parsing and persistence helpers.
//!
//! These utilities support both CLI parsing and the hot-reloaded `streams.txt`
//! file used in multi-stream mode.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::url_utils::normalize_page_url;

/// Split a raw wake phrase list into normalized strings.
///
/// Accepts commas, pipes, semicolons, or newlines as delimiters.
pub fn split_wake_phrases(raw: &str) -> Vec<String> {
    raw.split(|c| matches!(c, ',' | '|' | ';' | '\n' | '\r'))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Split a raw stream list into individual entries.
///
/// Accepts commas, pipes, semicolons, or newlines as delimiters.
pub fn split_stream_list(raw: &str) -> Vec<String> {
    raw.split(|c| matches!(c, ',' | '|' | ';' | '\n' | '\r'))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Normalize a stream URL, adding a scheme when missing.
///
/// Returns `None` if the input is empty after trimming.
pub fn normalize_stream_url(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(normalize_page_url(trimmed))
}

/// Parse the contents of a `streams.txt` file into normalized URLs.
///
/// Comments (`#` or `//`) and blank lines are ignored. Duplicates are removed
/// while preserving the first-seen order.
pub fn parse_streams_file(contents: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
            continue;
        }
        let Some(url) = normalize_stream_url(trimmed) else { continue };
        if seen.insert(url.clone()) {
            out.push(url);
        }
    }
    out
}

/// Default path for the hot-reload streams list (`streams.txt` in CWD).
pub fn default_streams_file_path() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("streams.txt")
}

/// Load and parse a streams file from disk.
pub fn read_streams_file(path: &Path) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("reading streams file {}", path.display()))?;
    Ok(parse_streams_file(&contents))
}

/// Write a streams file if it differs from the desired contents.
///
/// This keeps the hot-reload file stable and avoids unnecessary disk writes.
pub fn sync_streams_file(path: &Path, streams: &[String]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating streams dir {}", parent.display()))?;
    }
    let mut lines = Vec::new();
    lines.push("# AutoClip streams list (hot-reload).".to_string());
    lines.push("# One stream URL per line.".to_string());
    lines.push(String::new());
    lines.extend(streams.iter().cloned());
    let next = lines.join("\n");
    let write = match fs::read_to_string(path) {
        Ok(current) => current != next,
        Err(_) => true,
    };
    if write {
        fs::write(path, next)
            .with_context(|| format!("writing streams file {}", path.display()))?;
    }
    Ok(())
}
