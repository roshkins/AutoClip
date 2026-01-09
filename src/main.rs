//! AutoClip MVP stub.
//! Current behavior: headless-grab an m3u8 from a page (Kick-style), pick the
//! top variant, and save a 30s vertical clip via FFmpeg. Everything else (rolling
//! buffering, wake-word detection, VRAM budgeting) is logged-only scaffolding.

use anyhow::{Context, Result};
use m3u8_rs::{MasterPlaylist, MediaPlaylist, VariantStream};
use reqwest::{header, Client};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::process::Command;
use url::Url;

#[derive(Debug, Clone)]
pub struct Config {
    /// The URL to the streamer you want to AutoClip.
    pub kick_url: String,
    /// Phrase to trigger the automatic clip.
    pub activation_phrase: String,
    /// Amount of video (seconds) to keep on a rolling buffer before the trigger.
    pub before_buffer_length: u32,
    /// Amount of video (seconds) to keep after the trigger.
    pub after_buffer_length: u32,
    /// Output video resolution target (e.g., "1080x1920").
    pub resolution: String,
    /// GPU VRAM budget in MB before offloading to system RAM.
    pub vram_allocation: u32,
    /// Directory where clips are saved.
    pub save_path: String,
    /// Filename prefix to prepend to the incrementing counter.
    pub file_name_stub: String,
}

impl Config {
    pub fn example() -> Self {
        Self {
            kick_url: "https://example.com/stream".to_string(),
            activation_phrase: "clip that".to_string(),
            before_buffer_length: 5,
            after_buffer_length: 10,
            resolution: "1080x1920".to_string(),
            vram_allocation: 512,
            save_path: "./clips".to_string(),
            file_name_stub: "clip".to_string(),
        }
    }
}

/// Core application placeholder.
pub struct AutoClip {
    pub config: Config,
}

impl AutoClip {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Convenience entry point that defers to `run_with_page` without overriding page URL.
    pub async fn run(&self) -> Result<()> {
        self.run_with_page(None).await
    }

    /// Orchestrates the current flow:
    /// - When given a page URL (arg or CLIP_PAGE_URL), grab its m3u8 via headless
    ///   script and save a 30s vertical clip with FFmpeg using the best variant.
    /// - Otherwise, run stubbed logging for buffering / wake-word / post-process.
    pub async fn run_with_page(&self, page_url_override: Option<&str>) -> Result<()> {
        // Take an explicit page URL if provided; otherwise fall back to CLIP_PAGE_URL env if set.
        if let Some(page_url) = page_url_override
            .map(|s| s.to_string())
            .or_else(|| std::env::var("CLIP_PAGE_URL").ok())
            .filter(|s| !s.is_empty())
        {
            let output = self.clip_30s_from_page(&page_url).await?;
            println!("saved 30s clip to {}", output.display());
            return Ok(());
        }

        // Fallback to stubbed flow when no page provided.
        self.buffer_stream().await?;
        self.detect_wake_word().await?;
        self.post_process_clip().await
    }

    async fn buffer_stream(&self) -> Result<()> {
        println!(
            "[stub] buffering stream from {} with before={}s after={}s", // minimal log
            self.config.kick_url, self.config.before_buffer_length, self.config.after_buffer_length
        );
        Ok(())
    }

    async fn detect_wake_word(&self) -> Result<()> {
        println!(
            "[stub] listening for activation phrase '{}'", // stubbed wake-word detection
            self.config.activation_phrase
        );
        Ok(())
    }

    async fn post_process_clip(&self) -> Result<()> {
        println!(
            "[stub] post-processing to {} and saving under {} with prefix {}",
            self.config.resolution, self.config.save_path, self.config.file_name_stub
        );
        Ok(())
    }

    /// Fetch master from a page (headless Playwright), pick best variant, and run
    /// ffmpeg to save a 30s vertical clip.
    pub async fn clip_30s_from_page(&self, page_url: &str) -> Result<PathBuf> {
        let hls = HlsClient::new()?;
        let (master_url, master) = hls.fetch_master_from_page(page_url).await?;
        let media_url = hls.highest_variant_url(&master_url, &master)?;

        let output_path = next_output_path(&self.config.save_path, &self.config.file_name_stub)?;
        let (out_w, out_h) = parse_resolution(&self.config.resolution).unwrap_or((1080, 1920));

        run_ffmpeg_30s(&media_url, &output_path, out_w, out_h).await?;
        Ok(output_path)
    }
}
/// Minimal client to fetch and parse HLS playlists.
#[derive(Clone)]
pub struct HlsClient {
    client: Client,
}

impl HlsClient {
    pub fn new() -> Result<Self> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::USER_AGENT,
            header::HeaderValue::from_static(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            ),
        );
        headers.insert(header::ACCEPT, header::HeaderValue::from_static("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"));
        headers.insert(header::ACCEPT_LANGUAGE, header::HeaderValue::from_static("en-US,en;q=0.9"));
        let client = Client::builder()
            .default_headers(headers)
            .redirect(reqwest::redirect::Policy::limited(5))
            .timeout(Duration::from_secs(15))
            .build()
            .context("building reqwest client")?;
        Ok(Self { client })
    }

    /// Fetch and parse a master playlist from the provided URL.
    pub async fn fetch_master(&self, url: &str) -> Result<MasterPlaylist> {
        let body = self.fetch_bytes(url).await?;
        let parsed = m3u8_rs::parse_master_playlist_res(&body)
            .map_err(|e| anyhow::anyhow!("failed to parse master playlist: {e}"))?;
        Ok(parsed)
    }

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
        let parsed = m3u8_rs::parse_master_playlist_res(&body)
            .map_err(|e| anyhow::anyhow!("failed to parse master playlist: {e}"))?;
        Ok(parsed)
    }

    /// Fetch a page via headless Playwright, extract the first m3u8 URL, and parse
    /// it as a master playlist. Adds Referer/Origin tied to the page URL. Forwards
    /// COOKIE_HEADER or KICK_COOKIE if present. If M3U8_URL_OVERRIDE is set, use
    /// that master URL directly instead of headless extraction.
    pub async fn fetch_master_from_page(&self, page_url: &str) -> Result<(String, MasterPlaylist)> {
        let env_cookie = std::env::var("COOKIE_HEADER")
            .ok()
            .or_else(|| std::env::var("KICK_COOKIE").ok());

        let headless_script = std::env::var("HEADLESS_M3U8_SCRIPT")
            .unwrap_or_else(|_| "scripts/capture_m3u8.js".to_string());

        if let Ok(override_url) = std::env::var("M3U8_URL_OVERRIDE") {
            let master = self
                .fetch_master_with_headers(&override_url, Some(page_url), Some("https://kick.com"), env_cookie.as_deref())
                .await?;
            return Ok((override_url, master));
        }

        self.fetch_master_with_headless(page_url, env_cookie.as_deref(), &headless_script)
            .await
    }

    /// Use a headless browser (Node + Playwright script) to capture an m3u8 URL.
    async fn fetch_master_with_headless(
        &self,
        page_url: &str,
        cookie_env: Option<&str>,
        script_path: &str,
    ) -> Result<(String, MasterPlaylist)> {
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

        let m3u8_url_raw = m3u8_url_line
            .ok_or_else(|| anyhow::anyhow!("headless script did not emit an m3u8 url; stderr: {}", stderr.trim()))?;
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

        let body = self
            .fetch_bytes_with_headers(
                &m3u8_url,
                Some(page_url),
                Some("https://kick.com"),
                combined_cookie.as_deref(),
            )
            .await?;

        let parsed = m3u8_rs::parse_master_playlist_res(&body).map_err(|e| {
            let preview: String = String::from_utf8_lossy(&body)
                .chars()
                .take(500)
                .collect();
            eprintln!(
                "failed to parse master playlist at {} (headless): {}; body preview: {}",
                m3u8_url, e, preview
            );
            anyhow::anyhow!("failed to parse master playlist at {m3u8_url}: {e}")
        })?;

        Ok((m3u8_url, parsed))
    }

    /// Pick the highest-quality variant from a master playlist and return its joined URL.
    pub fn highest_variant_url(&self, master_url: &str, master: &MasterPlaylist) -> Result<Url> {
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

    /// Fetch the first media segment bytes from a media playlist URL.
    pub async fn fetch_first_segment(&self, playlist_url: &str) -> Result<Vec<u8>> {
        let media = self.fetch_media(playlist_url).await?;
        let first = media
            .segments
            .first()
            .map(|s| &s.uri)
            .context("media playlist has no segments")?;

        let playlist_url = Url::parse(playlist_url).context("invalid playlist url")?;
        let segment_url = playlist_url
            .join(first)
            .context("joining segment url")?;

        self.fetch_bytes(segment_url.as_str()).await
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
        req = req.header(header::ACCEPT, "application/vnd.apple.mpegurl,application/x-mpegURL,application/octet-stream");

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
            eprintln!("fetch {} -> status {} body preview: {}", url, status, preview.chars().take(500).collect::<String>());
            anyhow::bail!("non-success status {} for {}", status, url);
        }
        Ok(bytes.to_vec())
    }

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

fn parse_resolution(res: &str) -> Option<(u32, u32)> {
    let parts: Vec<_> = res.split('x').collect();
    if parts.len() != 2 {
        return None;
    }
    let w = parts[0].parse().ok()?;
    let h = parts[1].parse().ok()?;
    Some((w, h))
}

fn sanitize_m3u8_url(raw: &str) -> String {
    raw.trim()
        .trim_matches('\'')
        .trim_matches('"')
        .trim_end_matches('\\')
        .to_string()
}

fn next_output_path(save_dir: &str, stub: &str) -> Result<PathBuf> {
    let dir = Path::new(save_dir);
    fs::create_dir_all(dir).with_context(|| format!("creating save dir {save_dir}"))?;

    let mut idx = 1;
    loop {
        let candidate = dir.join(format!("{stub}_{idx:03}.mp4"));
        if !candidate.exists() {
            return Ok(candidate);
        }
        idx += 1;
    }
}

async fn run_ffmpeg_30s(input_hls: &Url, out_path: &Path, out_w: u32, out_h: u32) -> Result<()> {
    let vf = format!(
        "scale={}:{}:force_original_aspect_ratio=decrease,pad={}:{}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        out_w, out_h, out_w, out_h
    );

    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(input_hls.as_str())
        .arg("-map")
        .arg("0:v:0")
        .arg("-map")
        .arg("0:a:0?")
        .arg("-t")
        .arg("30")
        .arg("-vf")
        .arg(&vf)
        .arg("-c:v")
        .arg("libx264")
        .arg("-preset")
        .arg("veryfast")
        .arg("-crf")
        .arg("23")
        .arg("-c:a")
        .arg("copy") // keep source codec/bitrate to preserve highest audio quality
        .arg("-shortest")
        .arg(out_path.as_os_str())
        .status()
        .await
        .context("failed to run ffmpeg")?;

    if !status.success() {
        anyhow::bail!("ffmpeg exited with status {status}");
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let page_url_arg = args.get(1).map(|s| s.as_str());

    let mut config = Config::example();
    if let Some(url) = page_url_arg {
        config.kick_url = url.to_string();
    }

    if page_url_arg.is_none() && std::env::var("CLIP_PAGE_URL").is_err() {
        eprintln!("usage: autoclip <page_url>  (or set CLIP_PAGE_URL)");
        return Ok(());
    }

    let app = AutoClip::new(config);
    app.run_with_page(page_url_arg).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn run_stub_succeeds() {
        // Ensure no network-dependent env vars force the main path.
        std::env::remove_var("CLIP_PAGE_URL");
        std::env::remove_var("M3U8_URL_OVERRIDE");
        let app = AutoClip::new(Config::example());
        assert!(app.run().await.is_ok());
    }

    /// Optional network test: set M3U8_TEST_URL to a live master playlist URL (e.g., Kick channel).
    /// The test fetches the master playlist, picks the highest-quality variant, then downloads the first segment.
    #[tokio::test]
    async fn fetch_first_segment_when_url_set() -> Result<()> {
        let Some(master_url) = std::env::var("M3U8_TEST_URL").ok().filter(|v| !v.is_empty()) else {
            eprintln!("skipping: set M3U8_TEST_URL to a live master playlist url (e.g., Kick channel)");
            return Ok(());
        };

        let client = HlsClient::new()?;
        let master = client.fetch_master(&master_url).await?;
        let media_url = client.highest_variant_url(&master_url, &master)?;

        let segment_bytes = client.fetch_first_segment(media_url.as_str()).await?;
        assert!(!segment_bytes.is_empty(), "segment should contain data");
        Ok(())
    }

    /// Optional network test: set M3U8_PAGE_URL to a live page (e.g., https://kick.com/kyootbot when live).
    /// It scrapes the page for the m3u8, picks the best variant, and fetches the first segment.
    #[tokio::test]
    async fn fetch_master_from_page_when_set() -> Result<()> {
        let Some(page_url) = std::env::var("M3U8_PAGE_URL").ok().filter(|v| !v.is_empty()) else {
            eprintln!("skipping: set M3U8_PAGE_URL to a live page containing m3u8");
            return Ok(());
        };

        let client = HlsClient::new()?;
        let (master_url, master) = client.fetch_master_from_page(&page_url).await?;
        let media_url = client.highest_variant_url(&master_url, &master)?;

        let segment_bytes = client.fetch_first_segment(media_url.as_str()).await?;
        assert!(!segment_bytes.is_empty(), "segment should contain data");
        Ok(())
    }
}
