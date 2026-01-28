//! Face ID enrollment helpers.
//!
//! These functions handle optional face ID enrollment from profile images or
//! live stream frames when enabled via environment variables.

use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use reqwest::Client;
use serde_json::Value;

use crate::clip_detect;
use crate::url_utils::{extract_meta_image, kick_slug_from_url, stream_id_from_url};
use crate::parse_bool;
use crate::hls_client::HlsClient;

/// Return whether face ID enrollment/lookup is enabled.
pub fn face_id_enabled() -> bool {
    std::env::var("CLIP_FACE_ID")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

/// Resolve the face-id enrollment file for a given stream URL.
///
/// Respects `CLIP_FACE_ID_FILE` when set; otherwise uses `face_id/<stream>.json`.
pub fn face_id_file_for_stream(page_url: &str) -> PathBuf {
    if let Ok(path) = std::env::var("CLIP_FACE_ID_FILE") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    let stream_id = stream_id_from_url(page_url);
    Path::new("face_id").join(format!("{stream_id}.json"))
}

pub(crate) async fn fetch_profile_image_url(page_url: &str) -> Result<Option<String>> {
    if let Some(url) = fetch_kick_profile_image_url_api(page_url).await? {
        return Ok(Some(url));
    }
    let client = Client::new();
    let resp = client.get(page_url).send().await?;
    if !resp.status().is_success() {
        return Ok(None);
    }
    let html = resp.text().await?;
    if let Some(url) = extract_meta_image(&html, r#"property="og:image"#) {
        return Ok(Some(url));
    }
    if let Some(url) = extract_meta_image(&html, r#"name="twitter:image"#) {
        return Ok(Some(url));
    }
    Ok(None)
}

async fn fetch_kick_profile_image_url_api(page_url: &str) -> Result<Option<String>> {
    ensure_kick_oauth_env();
    let Some(slug) = kick_slug_from_url(page_url) else {
        return Ok(None);
    };
    let client_id = std::env::var("KICK_CLIENT_ID").ok();
    let client_secret = std::env::var("KICK_CLIENT_SECRET").ok();
    let Some(client_id) = client_id.filter(|v| !v.trim().is_empty()) else {
        return Ok(None);
    };
    let Some(client_secret) = client_secret.filter(|v| !v.trim().is_empty()) else {
        return Ok(None);
    };
    let client = Client::new();
    let token = match fetch_kick_app_token(&client, &client_id, &client_secret).await? {
        Some(token) => token,
        None => return Ok(None),
    };
    let channel = match fetch_kick_channel_by_slug(&client, &token, &slug).await? {
        Some(chan) => chan,
        None => return Ok(None),
    };
    let user_id = channel
        .broadcaster_user_id
        .or(channel.user_id)
        .or(channel.id);
    let Some(user_id) = user_id else {
        return Ok(None);
    };
    let user = fetch_kick_user_by_id(&client, &token, user_id).await?;
    Ok(user.and_then(|u| u.profile_picture.or(u.profile_pic)))
}

fn ensure_kick_oauth_env() {
    static PROMPTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    if PROMPTED.set(()).is_err() {
        return;
    }
    if std::env::var("KICK_CLIENT_ID").is_ok()
        && std::env::var("KICK_CLIENT_SECRET").is_ok()
    {
        return;
    }
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return;
    }
    let mut stdout = io::stdout();
    if std::env::var("KICK_CLIENT_ID").is_err() {
        let _ = write!(stdout, "Kick client_id (leave blank to skip): ");
        let _ = stdout.flush();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_ok() {
            let value = input.trim().to_string();
            if !value.is_empty() {
                std::env::set_var("KICK_CLIENT_ID", value);
            }
        }
    }
    if std::env::var("KICK_CLIENT_SECRET").is_err() {
        let _ = write!(stdout, "Kick client_secret (input visible, leave blank to skip): ");
        let _ = stdout.flush();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_ok() {
            let value = input.trim().to_string();
            if !value.is_empty() {
                std::env::set_var("KICK_CLIENT_SECRET", value);
            }
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct KickTokenResponse {
    access_token: String,
    expires_in: Option<u64>,
    #[allow(dead_code)]
    token_type: Option<String>,
    #[allow(dead_code)]
    scope: Option<String>,
}

#[derive(Clone, Debug)]
struct KickAppToken {
    access_token: String,
    expires_at: std::time::Instant,
}

static KICK_TOKEN_CACHE: std::sync::OnceLock<std::sync::Mutex<Option<KickAppToken>>> =
    std::sync::OnceLock::new();

async fn fetch_kick_app_token(
    client: &Client,
    client_id: &str,
    client_secret: &str,
) -> Result<Option<KickAppToken>> {
    let cache = KICK_TOKEN_CACHE.get_or_init(|| std::sync::Mutex::new(None));
    if let Ok(guard) = cache.lock() {
        if let Some(token) = guard.as_ref() {
            if token.expires_at > std::time::Instant::now() {
                return Ok(Some(token.clone()));
            }
        }
    }
    let mut form = vec![
        ("client_id", client_id.to_string()),
        ("client_secret", client_secret.to_string()),
        ("grant_type", "client_credentials".to_string()),
    ];
    if let Ok(scope) = std::env::var("KICK_OAUTH_SCOPE") {
        let trimmed = scope.trim();
        if !trimmed.is_empty() {
            form.push(("scope", trimmed.to_string()));
        }
    }
    let resp = client
        .post("https://id.kick.com/oauth/token")
        .form(&form)
        .send()
        .await?;
    if !resp.status().is_success() {
        return Ok(None);
    }
    let body: KickTokenResponse = resp.json().await?;
    let expires_in = body.expires_in.unwrap_or(0);
    if body.access_token.trim().is_empty() || expires_in == 0 {
        return Ok(None);
    }
    let expires_at = std::time::Instant::now()
        .checked_add(std::time::Duration::from_secs(expires_in.saturating_sub(30)))
        .unwrap_or_else(std::time::Instant::now);
    let token = KickAppToken {
        access_token: body.access_token,
        expires_at,
    };
    if let Ok(mut guard) = cache.lock() {
        *guard = Some(token.clone());
    }
    Ok(Some(token))
}

#[derive(Debug, serde::Deserialize)]
struct KickChannel {
    id: Option<u64>,
    broadcaster_user_id: Option<u64>,
    user_id: Option<u64>,
}

#[derive(Debug, serde::Deserialize)]
struct KickUser {
    profile_picture: Option<String>,
    profile_pic: Option<String>,
}

async fn fetch_kick_channel_by_slug(
    client: &Client,
    token: &KickAppToken,
    slug: &str,
) -> Result<Option<KickChannel>> {
    let resp = client
        .get("https://api.kick.com/public/v1/channels")
        .bearer_auth(&token.access_token)
        .query(&[("slug", slug)])
        .send()
        .await?;
    if !resp.status().is_success() {
        return Ok(None);
    }
    let json: Value = resp.json().await?;
    let Some(data) = json.get("data").and_then(|v| v.as_array()) else {
        return Ok(None);
    };
    let Some(first) = data.first() else {
        return Ok(None);
    };
    let channel: KickChannel = serde_json::from_value(first.clone())?;
    Ok(Some(channel))
}

async fn fetch_kick_user_by_id(
    client: &Client,
    token: &KickAppToken,
    user_id: u64,
) -> Result<Option<KickUser>> {
    let resp = client
        .get("https://api.kick.com/public/v1/users")
        .bearer_auth(&token.access_token)
        .query(&[("id", user_id.to_string())])
        .send()
        .await?;
    if !resp.status().is_success() {
        return Ok(None);
    }
    let json: Value = resp.json().await?;
    let Some(data) = json.get("data").and_then(|v| v.as_array()) else {
        return Ok(None);
    };
    let Some(first) = data.first() else {
        return Ok(None);
    };
    let user: KickUser = serde_json::from_value(first.clone())?;
    Ok(Some(user))
}

async fn enroll_face_id_from_stream_frame(page_url: &str, file_path: &Path) -> Result<bool> {
    let hls = HlsClient::new()?;
    let (master_url, master) = hls.fetch_master_from_page(page_url).await?;
    let media_url = hls.highest_variant_url(&master_url, &master)?;
    clip_detect::enroll_face_id_from_media_url(media_url.as_str(), file_path).await
}

/// Attempt auto-enrollment of face ID data when enabled.
///
/// This first tries the profile image (`og:image`/`twitter:image`), then falls
/// back to a frame from the live stream if needed.
pub async fn maybe_auto_enroll_face_id(page_url: &str) {
    if !face_id_enabled() {
        return;
    }
    let file_path = face_id_file_for_stream(page_url);
    let _ = std::env::set_var(
        "CLIP_FACE_ID_FILE",
        file_path.to_string_lossy().as_ref(),
    );
    if file_path.exists() {
        return;
    }
    let image_url = match fetch_profile_image_url(page_url).await {
        Ok(Some(url)) => url,
        Ok(None) => {
            eprintln!("face id: no profile image found on {}", page_url);
            return;
        }
        Err(err) => {
            eprintln!("face id: failed to fetch profile image: {err:#}");
            return;
        }
    };
    if let Some(saved) = save_profile_image_for_stream(page_url, &image_url).await {
        eprintln!("face id: saved profile image -> {}", saved.display());
    }
    match clip_detect::enroll_face_id_from_image_url(&image_url, &file_path).await {
        Ok(true) => {
            eprintln!("face id: enrolled from profile image -> {}", file_path.display());
        }
        Ok(false) => {
            eprintln!("face id: profile image did not yield a face; trying stream frame");
            match enroll_face_id_from_stream_frame(page_url, &file_path).await {
                Ok(true) => {
                    eprintln!("face id: enrolled from stream frame -> {}", file_path.display());
                }
                Ok(false) => {
                    eprintln!("face id: stream frame did not yield a face");
                }
                Err(err) => {
                    eprintln!("face id: stream frame enrollment failed: {err:#}");
                }
            }
        }
        Err(err) => {
            eprintln!("face id: enrollment failed: {err:#}");
        }
    }
}

async fn save_profile_image_for_stream(page_url: &str, image_url: &str) -> Option<PathBuf> {
    let stream_id = stream_id_from_url(page_url);
    let dir = Path::new("face_id").join("profile_images");
    let _ = fs::create_dir_all(&dir);
    let ext = image_url
        .split('?')
        .next()
        .and_then(|s| Path::new(s).extension().and_then(|v| v.to_str()))
        .map(|s| s.to_ascii_lowercase())
        .filter(|s| matches!(s.as_str(), "jpg" | "jpeg" | "png" | "webp"))
        .unwrap_or_else(|| "jpg".to_string());
    let path = dir.join(format!("{stream_id}.{ext}"));
    let client = Client::new();
    let resp = client.get(image_url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let bytes = resp.bytes().await.ok()?;
    if bytes.is_empty() {
        return None;
    }
    if fs::write(&path, &bytes).is_ok() {
        Some(path)
    } else {
        None
    }
}
