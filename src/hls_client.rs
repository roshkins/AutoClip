//! HLS discovery and HTTP helper utilities.
//!
//! This module fetches HLS playlists from stream pages (Kick/TikTok/Twitch),
//! handles headless discovery, and provides helpers for media/segment access.

use anyhow::{Context, Result};
use m3u8_rs::{MasterPlaylist, MediaPlaylist, VariantStream};
use reqwest::{header, Client, StatusCode};
use serde_json::Value;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::sleep;
use url::Url;

use crate::truncate_str;
use crate::url_utils::{origin_for_page, sanitize_m3u8_url};

#[derive(Debug)]
struct HttpStatusError {
    status: StatusCode,
    url: String,
}

impl HttpStatusError {
    fn new(status: StatusCode, url: &str) -> Self {
        Self {
            status,
            url: url.to_string(),
        }
    }
}

impl std::fmt::Display for HttpStatusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "non-success status {} for {}", self.status, self.url)
    }
}

impl std::error::Error for HttpStatusError {}

/// Check whether an error is a specific HTTP status produced by this module.
pub(crate) fn is_http_status(err: &anyhow::Error, status: StatusCode) -> bool {
    err.downcast_ref::<HttpStatusError>()
        .map(|e| e.status == status)
        .unwrap_or(false)
}

/// HTTP header hints used for HLS playlist requests.
#[derive(Debug, Clone, Default)]
pub struct StreamHeaders {
    /// Optional Referer header (typically the page URL).
    pub referer: Option<String>,
    /// Optional Origin header derived from the page URL.
    pub origin: Option<String>,
    /// Optional Cookie header for authenticated streams.
    pub cookie: Option<String>,
}

impl StreamHeaders {
    fn for_page(page_url: &str, origin: Option<String>, cookie: Option<String>) -> Self {
        Self {
            referer: Some(page_url.to_string()),
            origin,
            cookie,
        }
    }
}

fn collect_env_cookies() -> Option<String> {
    let mut cookie_parts: Vec<String> = Vec::new();
    for key in ["COOKIE_HEADER", "KICK_COOKIE", "TIKTOK_COOKIE", "TWITCH_COOKIE"] {
        if let Ok(val) = std::env::var(key) {
            if !val.trim().is_empty() {
                cookie_parts.push(val);
            }
        }
    }
    if cookie_parts.is_empty() {
        None
    } else {
        Some(cookie_parts.join("; "))
    }
}

fn headless_script_for_page(page_url: &str) -> String {
    if page_url.contains("tiktok.com") {
        std::env::var("HEADLESS_M3U8_SCRIPT_TIKTOK")
            .unwrap_or_else(|_| "scripts/capture_m3u8_tiktok.js".to_string())
    } else {
        std::env::var("HEADLESS_M3U8_SCRIPT")
            .unwrap_or_else(|_| "scripts/capture_m3u8.js".to_string())
    }
}

/// Minimal client to fetch and parse HLS playlists.
#[derive(Clone)]
pub struct HlsClient {
    client: Client,
}

impl HlsClient {
    /// Create a new HLS client with a browser-like header set.
    pub fn new() -> Result<Self> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::USER_AGENT,
            header::HeaderValue::from_static(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            ),
        );
        headers.insert(
            header::ACCEPT,
            header::HeaderValue::from_static(
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            ),
        );
        headers.insert(
            header::ACCEPT_LANGUAGE,
            header::HeaderValue::from_static("en-US,en;q=0.9"),
        );
        let client = Client::builder()
            .default_headers(headers)
            .redirect(reqwest::redirect::Policy::limited(5))
            .timeout(Duration::from_secs(15))
            .build()
            .context("building reqwest client")?;
        Ok(Self { client })
    }

    /// Fetch and parse a master playlist with optional headers.
    pub async fn fetch_master_with_headers(
        &self,
        url: &str,
        referer: Option<&str>,
        origin: Option<&str>,
        cookie: Option<&str>,
    ) -> Result<MasterPlaylist> {
        let body = self
            .fetch_bytes_with_headers(url, referer, origin, cookie)
            .await?;
        self.parse_master_or_media(url, &body)
    }

    /// Fetch a page via headless Playwright, extract the first m3u8 URL, and parse
    /// it as a master playlist. Adds Referer/Origin tied to the page URL. Forwards
    /// COOKIE_HEADER / KICK_COOKIE / TIKTOK_COOKIE / TWITCH_COOKIE if present. If
    /// M3U8_URL_OVERRIDE is set, use
    /// that master URL directly instead of headless extraction.
    pub async fn fetch_master_from_page_with_headers(
        &self,
        page_url: &str,
    ) -> Result<(String, MasterPlaylist, StreamHeaders)> {
        let env_cookie = collect_env_cookies();
        let origin = origin_for_page(page_url).unwrap_or_else(|| "https://kick.com".to_string());
        let base_headers =
            StreamHeaders::for_page(page_url, Some(origin.clone()), env_cookie.clone());

        let is_tiktok = page_url.contains("tiktok.com");
        let is_twitch = page_url.contains("twitch.tv");
        let headless_script = headless_script_for_page(page_url);

        if let Ok(override_url) = std::env::var("M3U8_URL_OVERRIDE") {
            let master = self
                .fetch_master_with_headers(
                    &override_url,
                    base_headers.referer.as_deref(),
                    base_headers.origin.as_deref(),
                    base_headers.cookie.as_deref(),
                )
                .await?;
            return Ok((override_url, master, base_headers));
        }

        if is_tiktok {
            if let Some((url, master)) = self
                .try_fetch_tiktok_master(page_url, env_cookie.as_deref())
                .await?
            {
                return Ok((url, master, base_headers));
            }
            eprintln!("TikTok HTTP discovery failed or stream offline; falling back to headless");
        }
        if is_twitch {
            if let Some((url, master)) = self
                .try_fetch_twitch_master(page_url, env_cookie.as_deref())
                .await?
            {
                return Ok((url, master, base_headers));
            }
            eprintln!("Twitch HTTP discovery failed or stream offline; falling back to headless");
        }

        self.fetch_master_with_headless_with_headers(
            page_url,
            env_cookie.as_deref(),
            &headless_script,
        )
        .await
    }

    /// Fetch a master playlist from a page URL using the default discovery path.
    pub async fn fetch_master_from_page(&self, page_url: &str) -> Result<(String, MasterPlaylist)> {
        let (master_url, master, _headers) =
            self.fetch_master_from_page_with_headers(page_url).await?;
        Ok((master_url, master))
    }

    /// Force a headless discovery pass to refresh the master playlist + headers.
    pub async fn fetch_master_from_page_headless_with_headers(
        &self,
        page_url: &str,
    ) -> Result<(String, MasterPlaylist, StreamHeaders)> {
        let env_cookie = collect_env_cookies();
        let headless_script = headless_script_for_page(page_url);
        self.fetch_master_with_headless_with_headers(
            page_url,
            env_cookie.as_deref(),
            &headless_script,
        )
        .await
    }

    async fn fetch_master_with_headless_with_headers(
        &self,
        page_url: &str,
        cookie_env: Option<&str>,
        script_path: &str,
    ) -> Result<(String, MasterPlaylist, StreamHeaders)> {
        let output = Command::new("node")
            .arg(script_path)
            .arg(page_url)
            .output()
            .await
            .with_context(|| format!("running headless script {script_path}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if !output.status.success() {
            anyhow::bail!(
                "headless script exited with {}: {}",
                output.status,
                stderr.trim()
            );
        }

        let mut m3u8_url_line: Option<String> = None;
        let mut cookie_from_headless: Option<String> = None;
        for line in stdout.lines() {
            let trimmed = line.trim();
            if m3u8_url_line.is_none() && trimmed.to_lowercase().contains(".m3u8") {
                m3u8_url_line = Some(trimmed.to_string());
            }
            if let Some(rest) = trimmed.strip_prefix("COOKIES:") {
                if !rest.trim().is_empty() {
                    cookie_from_headless = Some(rest.trim().to_string());
                }
            }
        }

        let m3u8_url_raw = m3u8_url_line.ok_or_else(|| {
            anyhow::anyhow!(
                "headless script did not emit an m3u8 url; stderr: {}",
                stderr.trim()
            )
        })?;
        let m3u8_url_raw = sanitize_m3u8_url(&m3u8_url_raw);
        // Use the raw URL from headless to avoid invalidating any signed token.
        let m3u8_url = m3u8_url_raw;
        eprintln!("headless extracted m3u8 url: {}", m3u8_url);

        let combined_cookie = match (cookie_env, cookie_from_headless.as_deref()) {
            (Some(env_c), Some(headless_c)) => Some(format!("{env_c}; {headless_c}")),
            (Some(env_c), None) => Some(env_c.to_string()),
            (None, Some(headless_c)) => Some(headless_c.to_string()),
            (None, None) => None,
        };

        let origin = origin_for_page(page_url);

        let body = self
            .fetch_bytes_with_headers(
                &m3u8_url,
                Some(page_url),
                origin.as_deref(),
                combined_cookie.as_deref(),
            )
            .await?;

        let parsed = self.parse_master_or_media(&m3u8_url, &body)?;
        let headers = StreamHeaders::for_page(page_url, origin, combined_cookie);
        Ok((m3u8_url, parsed, headers))
    }

    async fn try_fetch_tiktok_master(
        &self,
        page_url: &str,
        cookie_env: Option<&str>,
    ) -> Result<Option<(String, MasterPlaylist)>> {
        let origin =
            origin_for_page(page_url).unwrap_or_else(|| "https://www.tiktok.com".to_string());
        let cookie_header = cookie_env
            .filter(|c| !c.trim().is_empty())
            .map(|c| c.to_string());

        let html = match self
            .fetch_text_with_headers(
                page_url,
                Some(page_url),
                Some(&origin),
                cookie_header.as_deref(),
            )
            .await
        {
            Ok(h) => h,
            Err(err) => {
                eprintln!("TikTok: failed to fetch page HTML: {err:#}");
                return Ok(None);
            }
        };

        let room_id = match extract_tiktok_room_id(&html) {
            Some(id) => id,
            None => {
                eprintln!("TikTok: no room_id found; stream may be offline");
                return Ok(None);
            }
        };

        let mut live_info: Option<Value> = None;
        let mut attempts = 0;
        while attempts < 3 {
            match self
                .fetch_tiktok_room_info(
                    &room_id,
                    page_url,
                    Some(&origin),
                    cookie_header.as_deref(),
                )
                .await
            {
                Ok(v) => {
                    live_info = Some(v);
                    break;
                }
                Err(err) => {
                    attempts += 1;
                    if attempts >= 3 {
                        eprintln!("TikTok: room info fetch failed: {err:#}");
                        return Ok(None);
                    }
                    sleep(Duration::from_millis(300)).await;
                }
            }
        }

        let live_info = live_info.unwrap_or(Value::Null);
        let mut candidates = collect_tiktok_hls_candidates(&live_info);

        if candidates.is_empty() {
            if let Some(fallback_url) = self
                .fetch_tiktok_live_detail_url(
                    &room_id,
                    page_url,
                    Some(&origin),
                    cookie_header.as_deref(),
                )
                .await?
            {
                candidates.push(("live_detail".to_string(), fallback_url));
            }
        }

        if candidates.is_empty() {
            eprintln!("TikTok: no HLS candidates from room info or detail API");
            return Ok(None);
        }

        let selected = pick_best_tiktok_hls(&candidates)
            .unwrap_or_else(|| candidates[0].1.clone());
        let master = self
            .fetch_master_with_headers(
                &selected,
                Some(page_url),
                Some(&origin),
                cookie_header.as_deref(),
            )
            .await?;
        Ok(Some((selected, master)))
    }

    async fn try_fetch_twitch_master(
        &self,
        page_url: &str,
        cookie_env: Option<&str>,
    ) -> Result<Option<(String, MasterPlaylist)>> {
        let login = match extract_twitch_login(page_url) {
            Some(name) => name,
            None => {
                eprintln!("Twitch: could not determine channel login from URL");
                return Ok(None);
            }
        };

        let origin =
            origin_for_page(page_url).unwrap_or_else(|| "https://www.twitch.tv".to_string());
        let cookie_header = cookie_env.filter(|c| !c.trim().is_empty());

        let (sig, token) = match self
            .fetch_twitch_playback_token(&login, page_url, Some(&origin), cookie_header)
            .await
        {
            Ok(t) => t,
            Err(err) => {
                eprintln!("Twitch: playback token fetch failed: {err:#}");
                return Ok(None);
            }
        };

        let hls_url = build_twitch_hls_url(&login, &sig, &token)?;
        let master = self
            .fetch_master_with_headers(&hls_url, Some(page_url), Some(&origin), cookie_header)
            .await?;
        Ok(Some((hls_url, master)))
    }

    async fn fetch_twitch_playback_token(
        &self,
        login: &str,
        referer: &str,
        origin: Option<&str>,
        cookie: Option<&str>,
    ) -> Result<(String, String)> {
        const DEFAULT_TWITCH_CLIENT_ID: &str = "kimne78kx3ncx6brgo4mv6wki5h1ko";
        const TWITCH_PLAYBACK_HASH: &str =
            "0828119ded2d05bcfcf8e4da97d91aa47a1ec89be12f0161a2367c6a6fc1ce2c";

        let client_id = std::env::var("TWITCH_CLIENT_ID")
            .unwrap_or_else(|_| DEFAULT_TWITCH_CLIENT_ID.to_string());
        let auth_token = std::env::var("TWITCH_OAUTH_TOKEN")
            .or_else(|_| std::env::var("TWITCH_AUTH_TOKEN"))
            .ok()
            .and_then(|v| twitch_auth_header(&v));

        let payload = serde_json::json!({
            "operationName": "PlaybackAccessToken",
            "variables": {
                "isLive": true,
                "login": login,
                "isVod": false,
                "vodID": "",
                "playerType": "site"
            },
            "extensions": {
                "persistedQuery": {
                    "version": 1,
                    "sha256Hash": TWITCH_PLAYBACK_HASH
                }
            }
        });

        let mut req = self
            .client
            .post("https://gql.twitch.tv/gql")
            .header("Client-ID", client_id)
            .header(header::ACCEPT, "application/json")
            .header(header::CONTENT_TYPE, "application/json")
            .header("Referer", referer);
        if let Some(o) = origin {
            req = req.header("Origin", o);
        }
        if let Some(c) = cookie {
            req = req.header(header::COOKIE, c);
        }
        if let Some(auth) = auth_token {
            req = req.header(header::AUTHORIZATION, auth);
        }

        let resp = req
            .json(&payload)
            .send()
            .await
            .context("requesting Twitch playback token")?;
        let status = resp.status();
        let body = resp.text().await.context("reading Twitch playback token body")?;
        if !status.is_success() {
            eprintln!(
                "Twitch token status {} body: {}",
                status,
                truncate_str(&body, 500)
            );
            anyhow::bail!("Twitch token request returned status {status}");
        }

        let json: Value = serde_json::from_str(&body).context("parsing Twitch token JSON")?;
        if let Some((sig, value)) = extract_twitch_playback_token(&json) {
            return Ok((sig, value));
        }
        anyhow::bail!("Twitch token response missing playback access token");
    }

    async fn fetch_text_with_headers(
        &self,
        url: &str,
        referer: Option<&str>,
        origin: Option<&str>,
        cookie: Option<&str>,
    ) -> Result<String> {
        let mut req = self.client.get(url);
        if let Some(r) = referer {
            req = req.header("Referer", r);
        }
        if let Some(o) = origin {
            req = req.header("Origin", o);
        }
        if let Some(c) = cookie {
            req = req.header(header::COOKIE, c);
        }
        let resp = req
            .send()
            .await
            .with_context(|| format!("request failed for {url}"))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .with_context(|| format!("reading body for {url}"))?;
        if !status.is_success() {
            eprintln!(
                "fetch {} -> status {} body preview: {}",
                url,
                status,
                truncate_str(&text, 500)
            );
            anyhow::bail!("non-success status {} for {}", status, url);
        }
        Ok(text)
    }

    async fn fetch_tiktok_room_info(
        &self,
        room_id: &str,
        referer: &str,
        origin: Option<&str>,
        cookie: Option<&str>,
    ) -> Result<Value> {
        let url = "https://webcast.tiktok.com/webcast/room/info";
        let mut req = self.client.get(url).query(&[
            ("room_id", room_id),
            ("aid", "1988"),
            ("device_platform", "web"),
            ("app_name", "tiktok_web"),
            ("language", "en"),
        ]);
        req = req.header("Referer", referer);
        if let Some(o) = origin {
            req = req.header("Origin", o);
        }
        if let Some(c) = cookie {
            req = req.header(header::COOKIE, c);
        }
        let resp = req
            .send()
            .await
            .with_context(|| "requesting TikTok room info")?;
        let status = resp.status();
        let body = resp.text().await.context("reading TikTok room info body")?;
        if !status.is_success() {
            eprintln!("TikTok room info status {} body: {}", status, truncate_str(&body, 500));
            anyhow::bail!("TikTok room info request returned status {status}");
        }
        let json: Value = serde_json::from_str(&body).context("parsing TikTok room info JSON")?;
        let data = json.get("data").cloned().unwrap_or(Value::Null);
        Ok(data)
    }

    async fn fetch_tiktok_live_detail_url(
        &self,
        room_id: &str,
        referer: &str,
        origin: Option<&str>,
        cookie: Option<&str>,
    ) -> Result<Option<String>> {
        let url = format!("https://www.tiktok.com/api/live/detail/?roomID={room_id}");
        let mut req = self.client.get(&url);
        req = req.header("Referer", referer);
        if let Some(o) = origin {
            req = req.header("Origin", o);
        }
        if let Some(c) = cookie {
            req = req.header(header::COOKIE, c);
        }
        let resp = req
            .send()
            .await
            .with_context(|| "requesting TikTok live detail")?;
        let status = resp.status();
        let body = resp.text().await.context("reading TikTok live detail body")?;
        if !status.is_success() {
            return Ok(None);
        }
        let json: Value = serde_json::from_str(&body).context("parsing TikTok live detail JSON")?;
        Ok(json
            .get("liveUrl")
            .and_then(|v| v.as_str())
            .map(|v| v.to_string()))
    }

    fn fetch_media_playlist_url(
        &self,
        master_url: &str,
        master: &MasterPlaylist,
    ) -> Result<Url> {
        let base = Url::parse(master_url).context("invalid master playlist url")?;

        let variant = master
            .variants
            .iter()
            .max_by(|a, b| compare_variant_quality(a, b))
            .context("master playlist has no variants")?;

        let uri = &variant.uri;
        let resolved = base.join(uri).context("joining variant uri")?;
        Ok(resolved)
    }

    /// Fetch and parse a media playlist (variant) from the provided URL.
    pub async fn fetch_media(&self, url: &str) -> Result<MediaPlaylist> {
        let body = self.fetch_bytes(url).await?;
        let parsed = m3u8_rs::parse_media_playlist_res(&body)
            .map_err(|e| anyhow::anyhow!("failed to parse media playlist: {e}"))?;
        Ok(parsed)
    }

    pub async fn fetch_media_with_headers(
        &self,
        url: &str,
        headers: &StreamHeaders,
    ) -> Result<MediaPlaylist> {
        let body = self
            .fetch_bytes_with_headers(
                url,
                headers.referer.as_deref(),
                headers.origin.as_deref(),
                headers.cookie.as_deref(),
            )
            .await?;
        let parsed = m3u8_rs::parse_media_playlist_res(&body)
            .map_err(|e| anyhow::anyhow!("failed to parse media playlist: {e}"))?;
        Ok(parsed)
    }

    pub async fn refresh_media_url_from_page_headless_with_headers(
        &self,
        page_url: &str,
    ) -> Result<(Url, StreamHeaders)> {
        let (master_url, master, headers) =
            self.fetch_master_from_page_headless_with_headers(page_url).await?;
        let media_url = self.highest_variant_url(&master_url, &master)?;
        Ok((media_url, headers))
    }

    /// Fetch an arbitrary segment (by URI) relative to a media playlist URL.
    pub async fn fetch_segment_from_playlist(
        &self,
        playlist_url: &Url,
        segment_uri: &str,
    ) -> Result<Vec<u8>> {
        let segment_url = playlist_url
            .join(segment_uri)
            .context("joining segment url")?;
        self.fetch_bytes(segment_url.as_str()).await
    }

    pub async fn fetch_segment_from_playlist_with_headers(
        &self,
        playlist_url: &Url,
        segment_uri: &str,
        headers: &StreamHeaders,
    ) -> Result<Vec<u8>> {
        let segment_url = playlist_url
            .join(segment_uri)
            .context("joining segment url")?;
        self.fetch_bytes_with_headers(
            segment_url.as_str(),
            headers.referer.as_deref(),
            headers.origin.as_deref(),
            headers.cookie.as_deref(),
        )
        .await
    }

    async fn fetch_bytes(&self, url: &str) -> Result<Vec<u8>> {
        self.fetch_bytes_with_headers(url, None, None, None).await
    }

    async fn fetch_bytes_with_headers(
        &self,
        url: &str,
        referer: Option<&str>,
        origin: Option<&str>,
        cookie: Option<&str>,
    ) -> Result<Vec<u8>> {
        let mut req = self.client.get(url);
        if let Some(r) = referer {
            req = req.header("Referer", r);
        }
        if let Some(o) = origin {
            req = req.header("Origin", o);
        }
        if let Some(c) = cookie {
            req = req.header(header::COOKIE, c);
        }
        req = req.header(
            header::ACCEPT,
            "application/vnd.apple.mpegurl,application/x-mpegURL,application/octet-stream",
        );

        // Log the effective headers for troubleshooting signed/authorized endpoints.
        if std::env::var("LOG_M3U8_HEADERS").is_ok() {
            let mut dbg_headers = Vec::new();
            if let Some(r) = referer {
                dbg_headers.push(format!("Referer={r}"));
            }
            if let Some(o) = origin {
                dbg_headers.push(format!("Origin={o}"));
            }
            if let Some(c) = cookie {
                dbg_headers.push(format!("Cookie={}...", c.chars().take(80).collect::<String>()));
            }
            eprintln!("m3u8 request headers: {}", dbg_headers.join(" | "));
        }

        let resp = req
            .send()
            .await
            .with_context(|| format!("request failed for {url}"))?;
        let status = resp.status();
        let bytes = resp
            .bytes()
            .await
            .with_context(|| format!("reading body for {url}"))?;
        if !status.is_success() {
            let preview = String::from_utf8_lossy(&bytes);
            eprintln!(
                "fetch {} -> status {} body preview: {}",
                url,
                status,
                preview.chars().take(500).collect::<String>()
            );
            return Err(HttpStatusError::new(status, url).into());
        }
        Ok(bytes.to_vec())
    }
}

impl HlsClient {
    fn parse_master_or_media(&self, m3u8_url: &str, body: &[u8]) -> Result<MasterPlaylist> {
        match m3u8_rs::parse_master_playlist_res(body) {
            Ok(master) if !master.variants.is_empty() => return Ok(master),
            Ok(_master) => {
                // Empty variants; try media parse and wrap as single-variant master.
                if let Ok(_media) = m3u8_rs::parse_media_playlist_res(body) {
                    let variant = VariantStream {
                        uri: m3u8_url.to_string(),
                        ..Default::default()
                    };
                    let mut out = MasterPlaylist::default();
                    out.variants.push(variant);
                    return Ok(out);
                }
                // Fall through to error below if media parse fails.
            }
            Err(e) => {
                // Try media parse before bailing.
                if let Ok(_media) = m3u8_rs::parse_media_playlist_res(body) {
                    let variant = VariantStream {
                        uri: m3u8_url.to_string(),
                        ..Default::default()
                    };
                    let mut out = MasterPlaylist::default();
                    out.variants.push(variant);
                    return Ok(out);
                }
                return Err(anyhow::anyhow!("failed to parse master playlist: {e}"));
            }
        }

        Err(anyhow::anyhow!("failed to parse playlist at {m3u8_url}"))
    }

    /// Resolve the highest-quality variant URL from a master playlist.
    pub fn highest_variant_url(&self, master_url: &str, master: &MasterPlaylist) -> Result<Url> {
        self.fetch_media_playlist_url(master_url, master)
    }
}

fn extract_tiktok_room_id(html: &str) -> Option<String> {
    for marker in [
        "id=\"SIGI_STATE\"",
        "id=\"sigi-persisted-data\"",
        "id=\"__UNIVERSAL_DATA_FOR_REHYDRATION__\"",
    ] {
        if let Some(block) = extract_json_script_block(html, marker) {
            if let Ok(json) = serde_json::from_str::<Value>(&block) {
                if let Some(id) = find_room_id_value(&json) {
                    return Some(id);
                }
                // Some pages nest under __DEFAULT_SCOPE__ for the universal data script.
                if let Some(default_scope) = json.get("__DEFAULT_SCOPE__") {
                    if let Some(id) = find_room_id_value(default_scope) {
                        return Some(id);
                    }
                }
            }
        }
    }

    // Fallback: scan for roomId in the HTML.
    if let Some(idx) = html.find("roomId\":\"") {
        let start = idx + "roomId\":\"".len();
        let rest = &html[start..];
        let end = rest.find('"').unwrap_or(rest.len());
        let candidate = &rest[..end];
        if !candidate.is_empty() && candidate.chars().all(|c| c.is_ascii_digit()) {
            return Some(candidate.to_string());
        }
    }

    None
}

fn extract_json_script_block(html: &str, marker: &str) -> Option<String> {
    let tag_start = html.find(marker)?;
    let after_tag = html[tag_start..].find('>')?;
    let script_start = tag_start + after_tag + 1;
    let script_end_rel = html[script_start..].find("</script>")?;
    let script_end = script_start + script_end_rel;
    Some(html[script_start..script_end].to_string())
}

fn find_room_id_value(value: &Value) -> Option<String> {
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                if k.eq_ignore_ascii_case("roomid") || k.eq_ignore_ascii_case("room_id") {
                    if let Some(s) = v.as_str() {
                        if !s.is_empty() {
                            return Some(s.to_string());
                        }
                    } else if let Some(n) = v.as_i64() {
                        return Some(n.to_string());
                    }
                }
                if let Some(found) = find_room_id_value(v) {
                    return Some(found);
                }
            }
            None
        }
        Value::Array(arr) => {
            for v in arr {
                if let Some(found) = find_room_id_value(v) {
                    return Some(found);
                }
            }
            None
        }
        _ => None,
    }
}

fn collect_tiktok_hls_candidates(live_info: &Value) -> Vec<(String, String)> {
    let mut out = Vec::new();

    if let Some(stream_data) = live_info
        .get("stream_url")
        .and_then(|v| v.get("live_core_sdk_data"))
        .and_then(|v| v.get("pull_data"))
        .and_then(|v| v.get("stream_data"))
    {
        if let Some(map) = stream_data.as_object() {
            for (quality, entry) in map {
                let parsed = if let Some(s) = entry.as_str() {
                    serde_json::from_str::<Value>(s).ok()
                } else {
                    Some(entry.clone())
                };

                if let Some(val) = parsed {
                    if let Some(hls) = extract_hls_from_stream_entry(&val) {
                        out.push((quality.clone(), hls));
                    }
                }
            }
        }
    }

    if let Some(map) = live_info
        .get("stream_url")
        .and_then(|v| v.get("hls_pull_url_map"))
        .and_then(|v| v.as_object())
    {
        for (quality, url) in map {
            if let Some(u) = url.as_str() {
                out.push((quality.clone(), u.to_string()));
            }
        }
    }

    if let Some(url) = live_info
        .get("stream_url")
        .and_then(|v| v.get("hls_pull_url"))
        .and_then(|v| v.as_str())
    {
        out.push(("hls_pull".to_string(), url.to_string()));
    }

    out
}

fn extract_hls_from_stream_entry(entry: &Value) -> Option<String> {
    let main = entry.get("main");
    let candidates = [
        main.and_then(|v| v.get("https_hls")),
        main.and_then(|v| v.get("hls")),
        main.and_then(|v| v.get("hls_pull_url")),
    ];

    for candidate in candidates.into_iter().flatten() {
        if let Some(url) = candidate.as_str() {
            if url.to_ascii_lowercase().contains("m3u8") {
                return Some(url.to_string());
            }
        }
    }

    None
}

fn pick_best_tiktok_hls(candidates: &[(String, String)]) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }

    let preferred = [
        "origion",
        "origin",
        "full_hd1",
        "uhd",
        "hd1",
        "hd",
        "sd2",
        "sd1",
        "sd",
        "ld",
    ];

    for pref in preferred {
        if let Some((_, url)) = candidates
            .iter()
            .find(|(q, _)| q.to_ascii_lowercase() == pref)
        {
            return Some(url.clone());
        }
    }

    candidates.first().map(|(_, url)| url.clone())
}

pub(crate) fn extract_twitch_login(page_url: &str) -> Option<String> {
    let url = Url::parse(page_url).ok()?;
    let host = url.host_str()?.to_ascii_lowercase();
    if !host.ends_with("twitch.tv") {
        return None;
    }

    for (k, v) in url.query_pairs() {
        if k.eq_ignore_ascii_case("channel") || k.eq_ignore_ascii_case("login") {
            if is_valid_twitch_login(&v) {
                return Some(v.to_string());
            }
        }
    }

    let mut segments = url.path_segments()?.filter(|s| !s.is_empty());
    let first = segments.next()?;
    let first_lc = first.to_ascii_lowercase();
    if first_lc == "popout" || first_lc == "embed" {
        if let Some(next) = segments.next() {
            if is_valid_twitch_login(next) {
                return Some(next.to_string());
            }
        }
    }

    if is_valid_twitch_login(first) {
        return Some(first.to_string());
    }

    None
}

fn is_valid_twitch_login(login: &str) -> bool {
    let lower = login.to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }
    let reserved = [
        "videos",
        "directory",
        "p",
        "settings",
        "downloads",
        "friends",
        "inventory",
        "jobs",
        "store",
        "login",
        "signup",
        "search",
        "prime",
        "bits",
    ];
    if reserved.contains(&lower.as_str()) {
        return false;
    }
    login.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn twitch_auth_header(token: &str) -> Option<String> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("oauth ") {
        return Some(trimmed.to_string());
    }
    if lower.starts_with("oauth:") {
        return Some(format!("OAuth {}", trimmed[6..].trim()));
    }
    if lower.starts_with("bearer ") {
        return Some(format!("OAuth {}", trimmed[7..].trim()));
    }
    Some(format!("OAuth {}", trimmed))
}

fn extract_twitch_playback_token(value: &Value) -> Option<(String, String)> {
    match value {
        Value::Array(items) => {
            for item in items {
                if let Some(tok) = extract_twitch_playback_token(item) {
                    return Some(tok);
                }
            }
            None
        }
        Value::Object(_) => {
            let data = value.get("data")?;
            let token = data
                .get("streamPlaybackAccessToken")
                .or_else(|| data.get("playbackAccessToken"))?;
            let sig = token.get("signature")?.as_str()?;
            let value = token.get("value")?.as_str()?;
            Some((sig.to_string(), value.to_string()))
        }
        _ => None,
    }
}

fn build_twitch_hls_url(login: &str, sig: &str, token: &str) -> Result<String> {
    let mut url =
        Url::parse(&format!("https://usher.ttvnw.net/api/channel/hls/{login}.m3u8"))
            .context("parsing Twitch usher URL")?;
    {
        let mut pairs = url.query_pairs_mut();
        pairs.append_pair("sig", sig);
        pairs.append_pair("token", token);
        pairs.append_pair("allow_source", "true");
        pairs.append_pair("allow_audio_only", "true");
        pairs.append_pair("allow_spectre", "true");
        pairs.append_pair("player", "twitchweb");
        pairs.append_pair("playlist_include_framerate", "true");
        pairs.append_pair("fast_bread", "true");
    }
    Ok(url.to_string())
}

/// Compare variants by average bandwidth, then bandwidth, then resolution pixels, then fallback to order.
fn compare_variant_quality(a: &VariantStream, b: &VariantStream) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let a_band: Option<u64> = a
        .average_bandwidth
        .map(|v| v as u64)
        .or(Some(a.bandwidth as u64));
    let b_band: Option<u64> = b
        .average_bandwidth
        .map(|v| v as u64)
        .or(Some(b.bandwidth as u64));

    match (a_band, b_band) {
        (Some(a_bw), Some(b_bw)) if a_bw != b_bw => return a_bw.cmp(&b_bw),
        _ => {}
    }

    let a_pixels = a.resolution.map(|r| r.width as u64 * r.height as u64);
    let b_pixels = b.resolution.map(|r| r.width as u64 * r.height as u64);

    match (a_pixels, b_pixels) {
        (Some(a_px), Some(b_px)) if a_px != b_px => return a_px.cmp(&b_px),
        _ => {}
    }

    Ordering::Equal
}
