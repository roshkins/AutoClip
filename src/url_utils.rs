//! URL normalization and metadata extraction helpers.
//!
//! These functions centralize heuristics used across stream discovery and
//! metadata parsing to keep URLs consistent and safe.

use std::time::{Duration, SystemTime};

use url::Url;

/// Normalize a page URL, ensuring a scheme and trimming whitespace.
///
/// If the input is missing a scheme, `https://` is assumed.
pub fn normalize_page_url(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return trimmed.to_string();
    }
    if Url::parse(trimmed).is_ok() {
        return trimmed.to_string();
    }
    if trimmed.contains("://") || looks_like_path(trimmed) {
        return trimmed.to_string();
    }
    format!("https://{trimmed}")
}

fn looks_like_path(value: &str) -> bool {
    let value = value.trim();
    if value.is_empty() {
        return false;
    }
    if value.starts_with("./")
        || value.starts_with("../")
        || value.starts_with('/')
        || value.starts_with('\\')
    {
        return true;
    }
    let bytes = value.as_bytes();
    bytes.len() >= 3 && bytes[1] == b':' && (bytes[2] == b'\\' || bytes[2] == b'/')
}

/// Sanitize an m3u8 URL extracted from HTML/JSON blobs.
///
/// Strips surrounding quotes/whitespace and trailing slashes or backslashes
/// that can break HLS requests.
pub fn sanitize_m3u8_url(raw: &str) -> String {
    let mut out = raw.trim();
    out = out.trim_end_matches(|c| c == '\\' || c == '/');
    out = out.trim_matches(|c: char| c == '"' || c == '\'' || c.is_whitespace());
    out.to_string()
}

/// Extract the earliest expiry time from a signed URL, if present.
///
/// The function inspects common query params like `exp`, `expires`, and token
/// variants. Returns `None` if no usable expiry is found.
pub fn signed_url_expiry(url: &Url) -> Option<SystemTime> {
    let mut best: Option<SystemTime> = None;
    for (key, value) in url.query_pairs() {
        let key = key.to_ascii_lowercase();
        let key = key.as_str();
        let is_exp = matches!(
            key,
            "exp"
                | "expires"
                | "expire"
                | "expiry"
                | "token_exp"
                | "hdntl_exp"
                | "hls_exp"
                | "sig_exp"
        ) || key.ends_with("_exp");
        if !is_exp {
            continue;
        }
        let raw = value.trim();
        if raw.is_empty() || !raw.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let mut secs = match raw.parse::<u64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if secs > 1_000_000_000_000 {
            secs /= 1000;
        }
        if secs > 10_000_000_000 {
            secs /= 1000;
        }
        let ts = SystemTime::UNIX_EPOCH + Duration::from_secs(secs);
        best = Some(best.map_or(ts, |prev| prev.min(ts)));
    }
    best
}

/// Build a safe `Origin` header value for a page URL.
///
/// Returns `None` if the input is not a valid URL.
pub fn origin_for_page(page_url: &str) -> Option<String> {
    if let Ok(u) = Url::parse(page_url) {
        if let Some(host) = u.host_str() {
            return Some(format!("{}://{}", u.scheme(), host));
        }
    }
    None
}

fn sanitize_stream_id(raw: &str) -> String {
    let mut out = String::new();
    let mut last_underscore = false;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_underscore = false;
        } else if ch == '-' {
            out.push('-');
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "stream".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Convert a stream URL into a stable, filesystem-safe stream id.
///
/// This lowercases the host/path and replaces unsafe characters with `_`.
pub fn stream_id_from_url(url: &str) -> String {
    let normalized = normalize_page_url(url);
    if let Ok(parsed) = Url::parse(&normalized) {
        let host = parsed.host_str().unwrap_or("stream");
        let path = parsed.path().trim_matches('/');
        let base = if path.is_empty() {
            host.to_string()
        } else {
            format!("{host}_{path}")
        };
        return sanitize_stream_id(&base);
    }
    sanitize_stream_id(&normalized)
}

/// Extract a meta image URL (`og:image`, `twitter:image`, etc.) from HTML.
///
/// `key` should be a snippet like `property="og:image"` used to locate the
/// appropriate `<meta>` tag.
pub fn extract_meta_image(html: &str, key: &str) -> Option<String> {
    let needle = format!(r#"{key}""#);
    let pos = html.find(&needle)?;
    let tail = &html[pos..];
    let content_pos = tail.find("content=")?;
    let mut rest = &tail[content_pos + "content=".len()..];
    rest = rest.trim_start();
    let quote = rest.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    rest = &rest[1..];
    let end = rest.find(quote)?;
    let value = rest[..end].trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}
