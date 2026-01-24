//! Face ID enrollment helpers.
//!
//! These functions handle optional face ID enrollment from profile images or
//! live stream frames when enabled via environment variables.

use std::path::{Path, PathBuf};

use anyhow::Result;
use reqwest::Client;

use crate::clip_detect;
use crate::url_utils::{extract_meta_image, stream_id_from_url};
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

async fn fetch_profile_image_url(page_url: &str) -> Result<Option<String>> {
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
    if file_path.exists() {
        return;
    }
    let _ = std::env::set_var(
        "CLIP_FACE_ID_FILE",
        file_path.to_string_lossy().as_ref(),
    );
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
