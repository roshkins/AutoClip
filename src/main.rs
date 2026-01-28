//! AutoClip: headless stream discovery + rolling buffer + wake-word clipping.
//! This binary handles stream discovery, buffering, wake detection, layout selection,
//! and FFmpeg-based rendering (stacked or full-frame). Configuration lives in config.env.

use anyhow::{Context, Result};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashSet, hash_map::DefaultHasher};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command as StdCommand, Stdio};
use std::time::{Duration, Instant, SystemTime};
use tokio::process::Command;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::sync::Semaphore;
use tokio::time::sleep;
use url::Url;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::collections::HashMap;

mod clip_detect;
mod clip_gameplay;
mod clip_layout;
mod layout_utils;
mod hls_client;
mod low_resource;
mod captions;
mod gpu;
mod loading;
mod face_id;
mod profile;
mod rolling_buffer;
mod stream_utils;
mod text_utils;
mod url_utils;
mod storage_utils;
#[cfg(test)]
pub mod test_util;
use clip_detect::{detect_layout_hints, read_clip_detect_config, run_face_threshold_sweep};
use clip_gameplay::{read_clip_gameplay_config, ClipGameplayDetector};
use clip_layout::{
    build_face_only_filter_graph, build_full_frame_fill_filter_graph,
    build_stacked_filter_graph, build_tracked_full_frame_fill_filter_graph, read_clip_layout_config,
    read_clip_layout_hints, resolve_stacked_layout_dims, ClipLayoutConfig, ClipLayoutMode,
    ClipLayoutHints, FilterGraph, NormalizedPoint,
};
use layout_utils::{
    face_rect_for_gameplay_guess, face_rect_from_hints, guess_gameplay_low_resource,
    parse_resolution, rect_area, rect_contains_rect,
};
use hls_client::{HlsClient, KickVodInfo, StreamHeaders, is_http_status};
use low_resource::{ensure_live_detect_budgets, low_resource_enabled, refresh_low_resource_state};
use captions::{
    CaptionRender, SubtitleSpec, TempSubtitle, adjust_caption_font_size,
    build_ass_from_payload_with_limits, build_caption_subtitles_filter,
    build_drawtext_caption_filter_with_limit, DrawtextBuildResult,
    build_srt_from_payload_with_limits, caption_render_mode, caption_y_for_layout,
    captions_enabled, closed_captions_enabled, read_caption_config, subtitle_codec_for_output,
    write_temp_ass, write_temp_srt,
};
use profile::profile_span;
use rolling_buffer::RollingBuffer;
use face_id::{face_id_enabled, maybe_auto_enroll_face_id};
use storage_utils::{
    log_run_event, log_wake_event, non_video_dir, now_unix_ms, write_atomic_file,
    write_media_url_file, write_wake_worker_status, whisper_worker_media_url_path,
    whisper_worker_status_path, whisper_worker_status_poll,
};
use stream_utils::{
    default_streams_file_path, normalize_stream_url, read_streams_file, split_stream_list,
    split_wake_phrases, sync_streams_file,
};
use text_utils::{
    fallback_title_from_transcript, normalize_title_whitespace, sanitize_title_for_filename,
    trim_transcript,
};
use url_utils::{
    kick_slug_from_url, normalize_page_url, origin_for_page, sanitize_m3u8_url,
    signed_url_expiry, stream_id_from_url,
};
#[cfg(feature = "whisper")]
mod stream_audio_wake;
#[cfg(not(feature = "whisper"))]
#[path = "stream_audio_wake_stub.rs"]
mod stream_audio_wake;
use stream_audio_wake::{
    detect_wake_in_file, run_wake_worker_mic, run_wake_worker_stream,
    start_mic_wake_with_ffmpeg, start_stream_wake_from_hls, transcribe_clip_audio,
    TranscriptPayload,
};

#[derive(Debug, Clone)]
pub struct Config {
    /// The URL to the streamer you want to AutoClip.
    pub kick_url: String,
    /// Phrase to trigger the automatic clip.
    pub activation_phrase: String,
    /// Optional explicit wake phrases (overrides env/default when set).
    pub wake_phrases: Option<Vec<String>>,
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
    /// Whether to print raw/normalized transcripts from the stream listener.
    pub log_raw_wake: bool,
    /// When true, listen to microphone for wakeword instead of stream audio.
    pub use_mic_for_wake: bool,
    /// Optional explicit microphone device for ffmpeg capture (e.g., audio="Microphone (XYZ)").
    pub mic_device: Option<String>,
}

impl Config {
    /// Build a minimal example configuration for tests and demos.
    pub fn example() -> Self {
        Self {
            kick_url: "https://example.com/stream".to_string(),
            activation_phrase: "orange".to_string(),
            wake_phrases: None,
            before_buffer_length: 50,
            after_buffer_length: 10,
            resolution: "1080x1920".to_string(),
            vram_allocation: 512,
            save_path: "./clips".to_string(),
            file_name_stub: "clip".to_string(),
            log_raw_wake: false,
            use_mic_for_wake: false,
            mic_device: None,
        }
    }
}

/// Main application wrapper that wires configuration into the runtime.
pub struct AutoClip {
    pub config: Config,
}

impl AutoClip {
    /// Construct an AutoClip instance from a config.
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
            .map(|s| normalize_page_url(&s))
        {
            self.run_until_wake_and_clip(&page_url).await?;
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

    /// Continuously buffer the HLS stream, wait for the wake phrase, then save the
    /// previous `before_buffer_length` seconds plus `after_buffer_length` seconds to a vertical MP4.
    pub async fn run_until_wake_and_clip(&self, page_url: &str) -> Result<()> {
        // Requirement: wake phrase appears 50s into the clip and 10s from the end.
        let before = Duration::from_secs(50);
        let after_tail = Duration::from_secs(10);
        let clip_window = before + after_tail;
        ensure_live_detect_budgets();
        let latency_headroom = std::env::var("WAKE_BUFFER_HEADROOM_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(20));
        let mut buffer_window = clip_window + latency_headroom;
        let mut buffer_window_ns = duration_to_ns(buffer_window);
        let mut buffer = RollingBuffer::new(buffer_window);
        let buffer_target_ns = Arc::new(std::sync::atomic::AtomicU64::new(buffer_window_ns));
        let max_latency_ns = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let whisper_worker = Arc::new(Mutex::new(None));
        spawn_ctrl_c_handler(stop.clone(), Some(whisper_worker.clone()));
        let mut last_detect_instant: Option<Instant> = None;
        let wake_start = Arc::new(Mutex::new(Instant::now()));
        let _start_instant = *lock_or_recover(&wake_start, "wake start");
        let detect_ns = Arc::new(std::sync::atomic::AtomicU64::new(u64::MAX));
        let audio_ns = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let last_word_ns = Arc::new(AtomicU64::new(u64::MAX));
        let monitor_audio_ns = audio_ns.clone();
        let monitor_stop = stop.clone();
        let monitor_target = buffer_target_ns.clone();
        let monitor_latency = max_latency_ns.clone();
        let monitor_headroom = latency_headroom;
        let monitor_clip = clip_window;
        let monitor_start = wake_start.clone();
        std::thread::spawn(move || {
            let mut max_latency = Duration::ZERO;
            let mut warned = false;
            let poll = Duration::from_millis(250);
            while !monitor_stop.load(Ordering::Relaxed) {
                let audio_ns_now = monitor_audio_ns.load(Ordering::Relaxed);
                if audio_ns_now > 0 {
                    let start_instant = *lock_or_recover(&monitor_start, "wake start");
                    let audio_instant = start_instant + Duration::from_nanos(audio_ns_now);
                    let now = Instant::now();
                    if now > audio_instant {
                        let latency = now - audio_instant;
                        if latency > max_latency {
                            max_latency = latency;
                            if !warned && max_latency > monitor_headroom {
                                warned = true;
                                eprintln!(
                                    "processing latency ~{:.1}s exceeds headroom ~{:.1}s; wake timing may drift (set WAKE_BUFFER_HEADROOM_SECS)",
                                    max_latency.as_secs_f32(),
                                    monitor_headroom.as_secs_f32()
                                );
                            }
                            let target = monitor_clip + monitor_headroom + max_latency;
                            monitor_target.store(duration_to_ns(target), Ordering::Relaxed);
                            monitor_latency.store(duration_to_ns(max_latency), Ordering::Relaxed);
                        }
                    }
                }
                std::thread::sleep(poll);
            }
        });

        let skip_clip_save = std::env::var("SKIP_CLIP_SAVE")
            .ok()
            .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
            .unwrap_or(false);
        let save_dir = Path::new(&self.config.save_path);
        if !skip_clip_save {
            fs::create_dir_all(save_dir).context("creating save dir for wakeword clip")?;
        }

        let save_root = self.config.save_path.clone();
        let file_stub = self.config.file_name_stub.clone();
        let resolution = self.config.resolution.clone();
        let stop_for_audio = stop.clone();

        let fired = Arc::new(AtomicBool::new(false));
        let mut pending_saves: Vec<tokio::task::JoinHandle<()>> = Vec::new();
        let clip_gap_watchdog = read_clip_gap_watchdog_secs();
        let clip_gap_warn_cooldown = Duration::from_secs(60);
        let last_clip_saved_ms = Arc::new(AtomicU64::new(now_unix_ms()));
        let mut last_clip_gap_warned_at: Option<Instant> = None;
        let min_wake_clip_gap = read_wake_min_clip_gap_secs();
        let mut live_config = LiveConfigState::from_env();
        if let Some(state) = live_config.as_mut() {
            state.maybe_refresh();
        }
        refresh_low_resource_state();

        let hls = HlsClient::new()?;
        let (master_url, master, headers) =
            hls.fetch_master_from_page_with_headers(page_url).await?;
        let initial_media_url = hls.highest_variant_url(&master_url, &master)?;
        println!("tracking variant: {}", initial_media_url);
        let media_url = Arc::new(Mutex::new(initial_media_url));
        let media_headers = Arc::new(Mutex::new(headers));
        let mut media_url_last_refresh = Instant::now();
        let mut media_url_expires_at = {
            let guard = lock_or_recover(&media_url, "media url");
            signed_url_expiry(&guard)
        };
        let m3u8_refresh_interval = read_m3u8_refresh_secs();
        let m3u8_refresh_margin = Duration::from_secs(30);

        let mut seen: HashSet<String> = HashSet::new();
        let mut stream_time = Duration::ZERO;
        let mut detect_stream_time: Option<Duration> = None;
        let mut after_remaining: Option<Duration> = None;
        let mut wake_counter: u64 = 0;
        let mut active_wake_id: Option<u64> = None;
        let mut last_audio_ns_seen: u64 = 0;
        let mut last_audio_seen_at = Instant::now();
        let offline_timeout = read_stream_offline_secs();
        let mut last_progress_at = Instant::now();
        let mut wake_stall_triggered = false;
        let mut wake_stall_logged = false;
        let refractory = std::env::var("WAKE_REFRACTORY_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(12));
        let mut refractory_until: Option<Instant> = None;
        let latency_refresh_threshold = Duration::from_secs(240);
        let mut last_latency_refresh: Option<Instant> = None;
        let mut restart_triggered = false;
        let mut last_warmup_log: Option<Instant> = None;
        let warmup_log_interval = Duration::from_secs(5);
        let mut restart_request: Option<String> = None;
        let mut last_restart_defer: Option<Instant> = None;
        let restart_defer_interval = Duration::from_secs(5);
        let whisper_isolate = whisper_isolate_enabled();
        let status_poll = whisper_worker_status_poll();
        let worker_status_path = whisper_worker_status_path(&self.config.save_path);
        let worker_media_url_path = whisper_worker_media_url_path(&self.config.save_path);
        let last_status_ms = Arc::new(AtomicU64::new(now_unix_ms()));
        if whisper_isolate && !self.config.use_mic_for_wake {
            let guard = lock_or_recover(&media_url, "media url");
            if let Err(err) = write_media_url_file(&worker_media_url_path, &guard) {
                eprintln!("whisper worker: failed to write media url: {err:#}");
            }
        }

        // Start listening for the wake phrase, either from microphone or stream audio.
        let model_path = stream_audio_wake::select_best_model_path();
        let wake_phrases =
            resolve_wake_phrases(self.config.wake_phrases.as_deref(), &self.config.activation_phrase);
        let wake_label = if wake_phrases.is_empty() {
            self.config.activation_phrase.clone()
        } else {
            wake_phrases.join(", ")
        };
        if whisper_isolate {
            let worker_start = Instant::now();
            *lock_or_recover(&wake_start, "wake start") = worker_start;
            let reader_start_ms = now_unix_ms();
            let _reader = spawn_wake_status_reader(
                worker_status_path.clone(),
                status_poll,
                reader_start_ms,
                last_status_ms.clone(),
                stop.clone(),
                fired.clone(),
                detect_ns.clone(),
                audio_ns.clone(),
                last_word_ns.clone(),
            );
            let mode = if self.config.use_mic_for_wake { "mic" } else { "stream" };
            let media_path = if self.config.use_mic_for_wake {
                None
            } else {
                Some(worker_media_url_path.as_path())
            };
            let worker = spawn_whisper_worker(
                mode,
                &worker_status_path,
                media_path,
                &wake_phrases,
                self.config.log_raw_wake,
            )?;
            *lock_or_recover(&whisper_worker, "whisper worker") = Some(worker);
            println!(
                "whisper worker: isolating wake detection for '{}'",
                wake_label
            );
        } else if self.config.use_mic_for_wake {
            let mic_device = self
                .config
                .mic_device
                .clone()
                .or_else(|| std::env::var("MIC_DEVICE").ok());
            let start_instant = *lock_or_recover(&wake_start, "wake start");
            start_mic_wake_with_ffmpeg(
                mic_device.as_deref(),
                Path::new(&model_path),
                &wake_phrases,
                self.config.log_raw_wake,
                stop_for_audio,
                fired.clone(),
                start_instant,
                detect_ns.clone(),
                audio_ns.clone(),
                last_word_ns.clone(),
            )?;
            println!(
                "listening to microphone for wake phrase(s) '{}' (model: {})",
                wake_label,
                model_path.display(),
            );
        } else {
            let start_instant = *lock_or_recover(&wake_start, "wake start");
            start_stream_wake_from_hls(
                media_url.clone(),
                Path::new(&model_path),
                &wake_phrases,
                self.config.log_raw_wake,
                stop_for_audio,
                fired.clone(),
                start_instant,
                detect_ns.clone(),
                audio_ns.clone(),
                last_word_ns.clone(),
            )?;
            println!(
                "listening to stream audio for wake phrase(s) '{}' (model: {})",
                wake_label,
                model_path.display()
            );
        }

        let poll_interval = Duration::from_millis(500);
        let mut error_backoff = Duration::ZERO;

        'stream_loop: loop {
            if stop.load(Ordering::Relaxed)
                && !fired.load(Ordering::Relaxed)
                && after_remaining.is_none()
            {
                break;
            }
            if let Some(timeout) = offline_timeout {
                if !fired.load(Ordering::Relaxed) && after_remaining.is_none() {
                    let now = Instant::now();
                    let stale = now.duration_since(last_progress_at);
                    if stale >= timeout {
                        let msg = format!(
                            "stream offline or stalled for {:.1}s; shutting down",
                            stale.as_secs_f32()
                        );
                        eprintln!("{msg}");
                        log_run_event(&save_root, &msg);
                        stop.store(true, Ordering::Relaxed);
                        break 'stream_loop;
                    }
                }
            }
            if let Some(state) = live_config.as_mut() {
                state.maybe_refresh();
            }
            refresh_low_resource_state();

            let latency = Duration::from_nanos(max_latency_ns.load(Ordering::Relaxed));
            let mut refreshed = false;
            if latency >= latency_refresh_threshold {
                let now = Instant::now();
                let can_refresh = last_latency_refresh
                    .map(|t| now.duration_since(t) > Duration::from_secs(30))
                    .unwrap_or(true);
                if can_refresh {
                    last_latency_refresh = Some(now);
                    let reason = format!(
                        "processing latency {:.1}s exceeded {:.1}s; refreshing m3u8 via headless",
                        latency.as_secs_f32(),
                        latency_refresh_threshold.as_secs_f32()
                    );
                    match refresh_media_source(
                        &hls,
                        page_url,
                        &media_url,
                        &media_headers,
                        &mut seen,
                        whisper_isolate,
                        self.config.use_mic_for_wake,
                        &worker_media_url_path,
                        &reason,
                    )
                    .await
                    {
                        Ok(new_url) => {
                            media_url_last_refresh = Instant::now();
                            media_url_expires_at = signed_url_expiry(&new_url);
                            refreshed = true;
                        }
                        Err(err) => {
                            eprintln!("headless refresh failed: {err:#}");
                        }
                    }
                }
            }
            if !refreshed {
                if let Some(interval) = m3u8_refresh_interval {
                    let now = Instant::now();
                    if now.duration_since(media_url_last_refresh) >= interval {
                        let reason = format!(
                            "m3u8 refresh interval {:.1}s reached; refreshing via headless",
                            interval.as_secs_f32()
                        );
                        match refresh_media_source(
                            &hls,
                            page_url,
                            &media_url,
                            &media_headers,
                            &mut seen,
                            whisper_isolate,
                            self.config.use_mic_for_wake,
                            &worker_media_url_path,
                            &reason,
                        )
                        .await
                        {
                            Ok(new_url) => {
                                media_url_last_refresh = now;
                                media_url_expires_at = signed_url_expiry(&new_url);
                                refreshed = true;
                            }
                            Err(err) => {
                                eprintln!("headless refresh failed: {err:#}");
                            }
                        }
                    }
                }
            }
            if !refreshed {
                if let Some(expires_at) = media_url_expires_at {
                    let now = SystemTime::now();
                    let remaining = expires_at
                        .duration_since(now)
                        .unwrap_or_else(|_| Duration::ZERO);
                    if remaining <= m3u8_refresh_margin {
                        let reason = format!(
                            "m3u8 expires in {:.1}s; refreshing via headless",
                            remaining.as_secs_f32()
                        );
                        match refresh_media_source(
                            &hls,
                            page_url,
                            &media_url,
                            &media_headers,
                            &mut seen,
                            whisper_isolate,
                            self.config.use_mic_for_wake,
                            &worker_media_url_path,
                            &reason,
                        )
                        .await
                        {
                            Ok(new_url) => {
                                media_url_last_refresh = Instant::now();
                                media_url_expires_at = signed_url_expiry(&new_url);
                            }
                            Err(err) => {
                                eprintln!("headless refresh failed: {err:#}");
                            }
                        }
                    }
                }
            }
            let audio_ns_now = audio_ns.load(Ordering::Relaxed);
            if audio_ns_now != last_audio_ns_seen {
                last_audio_ns_seen = audio_ns_now;
                last_audio_seen_at = Instant::now();
            }

            if let Some(stall_threshold) = read_wake_no_words_secs() {
                if !self.config.use_mic_for_wake
                    && !fired.load(Ordering::Relaxed)
                    && after_remaining.is_none()
                {
                    let now = Instant::now();
                    let progress_recent =
                        now.duration_since(last_progress_at) <= Duration::from_secs(15);
                    if progress_recent {
                        let audio_stale =
                            now.duration_since(last_audio_seen_at) >= stall_threshold;
                        if whisper_isolate {
                            let stall_ms =
                                stall_threshold.as_millis().min(u64::MAX as u128) as u64;
                            let now_ms = now_unix_ms();
                            let last_ms = last_status_ms.load(Ordering::Relaxed);
                            let status_stale = now_ms.saturating_sub(last_ms) >= stall_ms;
                            if status_stale && !wake_stall_triggered {
                                let buffer_ready = buffer.total_duration() >= clip_window;
                                if !buffer_ready {
                                    let now = Instant::now();
                                    if last_warmup_log
                                        .map(|t| now.duration_since(t) >= warmup_log_interval)
                                        .unwrap_or(true)
                                    {
                                        last_warmup_log = Some(now);
                                        let msg = format!(
                                            "wake worker stalled but buffer warming: {:.1}/{:.1}s",
                                            buffer.total_duration().as_secs_f32(),
                                            clip_window.as_secs_f32()
                                        );
                                        eprintln!("{msg}");
                                        log_run_event(&save_root, &msg);
                                    }
                                } else {
                                    wake_stall_triggered = true;
                                    let status_age =
                                        now_ms.saturating_sub(last_ms) as f32 / 1000.0;
                                    let msg = format!(
                                        "wake worker status stale for {:.1}s; restarting process",
                                        status_age
                                    );
                                    if restart_request.is_none() {
                                        restart_request = Some(msg.clone());
                                        log_run_event(
                                            &save_root,
                                            &format!("restart requested: {msg}"),
                                        );
                                    }
                                }
                            } else if audio_stale {
                                if !wake_stall_logged {
                                    wake_stall_logged = true;
                                    let msg = format!(
                                        "wake audio stalled for {:.1}s; stream may be silent (worker alive)",
                                        now.duration_since(last_audio_seen_at).as_secs_f32()
                                    );
                                    eprintln!("{msg}");
                                    log_run_event(&save_root, &msg);
                                }
                            } else {
                                wake_stall_logged = false;
                            }
                        } else if audio_stale && !wake_stall_triggered {
                            let buffer_ready = buffer.total_duration() >= clip_window;
                            if !buffer_ready {
                                let now = Instant::now();
                                if last_warmup_log
                                    .map(|t| now.duration_since(t) >= warmup_log_interval)
                                    .unwrap_or(true)
                                {
                                    last_warmup_log = Some(now);
                                    let msg = format!(
                                        "wake audio stalled but buffer warming: {:.1}/{:.1}s",
                                        buffer.total_duration().as_secs_f32(),
                                        clip_window.as_secs_f32()
                                    );
                                    eprintln!("{msg}");
                                    log_run_event(&save_root, &msg);
                                }
                            } else {
                                wake_stall_triggered = true;
                                let msg = format!(
                                    "wake audio stalled for {:.1}s; restarting process",
                                    now.duration_since(last_audio_seen_at).as_secs_f32()
                                );
                                if restart_request.is_none() {
                                    restart_request = Some(msg.clone());
                                    log_run_event(
                                        &save_root,
                                        &format!("restart requested: {msg}"),
                                    );
                                }
                            }
                        } else if !audio_stale {
                            wake_stall_logged = false;
                        }
                    } else {
                        wake_stall_logged = false;
                    }
                }
            }

            if let Some(limit) = clip_gap_watchdog {
                if !skip_clip_save {
                    let last_ms = last_clip_saved_ms.load(Ordering::Relaxed);
                    if last_ms > 0 {
                        let now_ms = now_unix_ms();
                        let gap = now_ms.saturating_sub(last_ms) as f32 / 1000.0;
                        if gap >= limit.as_secs_f32() {
                            let now = Instant::now();
                            let should_warn = last_clip_gap_warned_at
                                .map(|t| now.duration_since(t) >= clip_gap_warn_cooldown)
                                .unwrap_or(true);
                            if should_warn {
                                last_clip_gap_warned_at = Some(now);
                                let msg = format!(
                                    "clip watchdog: no clip saved for {:.1}s (threshold {:.1}s)",
                                    gap,
                                    limit.as_secs_f32()
                                );
                                eprintln!("{msg}");
                                log_run_event(&save_root, &msg);
                            }
                        } else {
                            last_clip_gap_warned_at = None;
                        }
                    }
                }
            }

            let target_ns = buffer_target_ns.load(Ordering::Relaxed);
            if target_ns > buffer_window_ns {
                buffer_window_ns = target_ns;
                buffer_window = Duration::from_nanos(target_ns);
                buffer.set_capacity(buffer_window);
                let latency = Duration::from_nanos(max_latency_ns.load(Ordering::Relaxed));
                eprintln!(
                    "processing latency ~{:.1}s; expanding buffer to ~{:.1}s",
                    latency.as_secs_f32(),
                    buffer_window.as_secs_f32()
                );
            }
            if !restart_triggered {
                if let Some(limit) = read_wake_buffer_restart_secs() {
                    if buffer_window > limit {
                        restart_triggered = true;
                        let msg = format!(
                            "buffer window {:.1}s exceeded restart limit {:.1}s; restarting",
                            buffer_window.as_secs_f32(),
                            limit.as_secs_f32()
                        );
                        if restart_request.is_none() {
                            restart_request = Some(msg.clone());
                            log_run_event(&save_root, &format!("restart requested: {msg}"));
                        }
                    }
                }
            }

            pending_saves.retain(|handle| !handle.is_finished());
            if let Some(reason) = restart_request.as_ref() {
                let buffer_ready = buffer.total_duration() >= clip_window;
                let wake_pending =
                    fired.load(Ordering::Relaxed) || after_remaining.is_some();
                let in_flight = pending_saves.len();
                if buffer_ready && !wake_pending && in_flight == 0 {
                    let msg = format!("restart executing: {reason}");
                    eprintln!("{msg}");
                    log_run_event(&save_root, &msg);
                    spawn_self_restart(reason.clone());
                } else {
                    let now = Instant::now();
                    if last_restart_defer
                        .map(|t| now.duration_since(t) >= restart_defer_interval)
                        .unwrap_or(true)
                    {
                        last_restart_defer = Some(now);
                        let msg = format!(
                            "restart deferred: buffer_ready={} wake_pending={} in_flight={}",
                            buffer_ready, wake_pending, in_flight
                        );
                        eprintln!("{msg}");
                        log_run_event(&save_root, &msg);
                    }
                }
            }

                let media_url_snapshot = {
                    let guard = lock_or_recover(&media_url, "media url fetch");
                    guard.clone()
                };
                let headers_snapshot = {
                    let guard = lock_or_recover(&media_headers, "media headers fetch");
                    guard.clone()
                };
            let playlist = match hls
                .fetch_media_with_headers(media_url_snapshot.as_str(), &headers_snapshot)
                .await
            {
                Ok(p) => p,
                Err(err) => {
                    if is_http_status(&err, StatusCode::FORBIDDEN) {
                        let reason = "media playlist returned 403; refreshing via headless";
                        match refresh_media_source(
                            &hls,
                            page_url,
                            &media_url,
                            &media_headers,
                            &mut seen,
                            whisper_isolate,
                            self.config.use_mic_for_wake,
                            &worker_media_url_path,
                            reason,
                        )
                        .await
                        {
                            Ok(new_url) => {
                                media_url_last_refresh = Instant::now();
                                media_url_expires_at = signed_url_expiry(&new_url);
                                continue 'stream_loop;
                            }
                            Err(refresh_err) => {
                                eprintln!("headless refresh failed: {refresh_err:#}");
                            }
                        }
                    }
                    eprintln!("failed to fetch media playlist: {err:#}; retrying");
                    error_backoff = bump_backoff(error_backoff);
                    sleep(error_backoff).await;
                    continue;
                }
            };

            let mut made_progress = false;

            for seg in &playlist.segments {
                let uri = seg.uri.clone();
                if !seen.insert(uri.clone()) {
                    continue;
                }

                match hls
                    .fetch_segment_from_playlist_with_headers(
                        &media_url_snapshot,
                        &uri,
                        &headers_snapshot,
                    )
                    .await
                {
                    Ok(bytes) => {
                        let seg_dur_playlist = Duration::from_secs_f32(seg.duration as f32);
                        let seg_dur = choose_segment_duration(
                            pts_duration_from_ts(&bytes),
                            seg_dur_playlist,
                        );
                        buffer.push(bytes, seg_dur);
                        last_progress_at = Instant::now();
                        stream_time = stream_time.saturating_add(seg_dur);
                        made_progress = true;
                    }
                    Err(err) => {
                        if is_http_status(&err, StatusCode::FORBIDDEN) {
                            let reason = "segment fetch returned 403; refreshing via headless";
                            match refresh_media_source(
                                &hls,
                                page_url,
                                &media_url,
                                &media_headers,
                                &mut seen,
                                whisper_isolate,
                                self.config.use_mic_for_wake,
                                &worker_media_url_path,
                                reason,
                            )
                            .await
                            {
                                Ok(new_url) => {
                                    media_url_last_refresh = Instant::now();
                                    media_url_expires_at = signed_url_expiry(&new_url);
                                    continue 'stream_loop;
                                }
                                Err(refresh_err) => {
                                    eprintln!("headless refresh failed: {refresh_err:#}");
                                }
                            }
                        }
                        eprintln!("failed to fetch segment {}: {err:#}", uri);
                        error_backoff = bump_backoff(error_backoff);
                        sleep(error_backoff).await;
                        continue;
                    }
                }

                if fired.load(Ordering::Relaxed) && after_remaining.is_none() {
                    let buffer_ready = buffer.total_duration() >= clip_window;
                    if !buffer_ready {
                        let now = Instant::now();
                        if last_warmup_log
                            .map(|t| now.duration_since(t) >= warmup_log_interval)
                            .unwrap_or(true)
                        {
                            last_warmup_log = Some(now);
                            let msg = format!(
                                "wake detected while buffer warming: {:.1}/{:.1}s; clip will be shorter",
                                buffer.total_duration().as_secs_f32(),
                                clip_window.as_secs_f32()
                            );
                            eprintln!("{msg}");
                            log_run_event(&save_root, &msg);
                        }
                    }
                if refractory_until.map(|t| Instant::now() < t).unwrap_or(false) {
                    // Ignore rapid re-triggers until cooldown expires.
                    let force_gap = min_wake_clip_gap.and_then(|gap| {
                        let now_ms = now_unix_ms();
                        let last_ms = last_clip_saved_ms.load(Ordering::Relaxed);
                        let elapsed_ms = now_ms.saturating_sub(last_ms);
                        if elapsed_ms >= gap.as_millis() as u64 {
                            Some(Duration::from_millis(elapsed_ms))
                        } else {
                            None
                        }
                    });
                    if force_gap.is_none() {
                        log_run_event(&save_root, "wake detected during refractory; ignoring");
                        if let Some(wake_id) = active_wake_id {
                            log_wake_event(&save_root, wake_id, "wake_refractory", json!({}));
                        }
                        continue;
                    }
                    let elapsed = force_gap.unwrap_or_else(|| Duration::from_secs(0));
                    let msg = format!(
                        "wake detected during refractory; overriding to satisfy min clip gap ({:.1}s)",
                        elapsed.as_secs_f32()
                    );
                    log_run_event(&save_root, &msg);
                    if let Some(wake_id) = active_wake_id {
                        log_wake_event(
                            &save_root,
                            wake_id,
                            "wake_refractory_override",
                            json!({ "elapsed_secs": elapsed.as_secs_f32() }),
                        );
                    }
                }
                    let detect_ns_val = detect_ns.load(std::sync::atomic::Ordering::Relaxed);
                    let audio_ns_now = audio_ns.load(std::sync::atomic::Ordering::Relaxed);
                    let age_audio = if detect_ns_val != u64::MAX && audio_ns_now >= detect_ns_val {
                        Duration::from_nanos(audio_ns_now - detect_ns_val)
                    } else {
                        Duration::ZERO
                    };

                    // Wait exactly 10s of post-wake audio to place wake at 50s and end at +10s.
                    refractory_until = Some(Instant::now() + refractory);
                    let ns = detect_ns.load(std::sync::atomic::Ordering::Relaxed);
                    let base_start = *lock_or_recover(&wake_start, "wake start");
                    let detected_instant = if ns != u64::MAX {
                        base_start + Duration::from_nanos(ns)
                    } else {
                        Instant::now()
                    };
                    last_detect_instant = Some(detected_instant);
                    let latency = Instant::now().saturating_duration_since(detected_instant);
                    detect_stream_time = Some(stream_time.saturating_sub(age_audio));
                    after_remaining = Some(after_tail);
                    let wake_id = match active_wake_id {
                        Some(id) => id,
                        None => {
                            wake_counter += 1;
                            active_wake_id = Some(wake_counter);
                            wake_counter
                        }
                    };
                    println!(
                        "wake detected; capturing tail to place wake at 50s into clip (latency ~{:.1}s, buffer ~{:.1}s, cooldown {:?})",
                        latency.as_secs_f32(),
                        buffer_window.as_secs_f32(),
                        refractory
                    );
                    log_run_event(
                        &save_root,
                        &format!(
                            "wake detected: latency={:.1}s buffer={:.1}s cooldown_secs={:.1}",
                            latency.as_secs_f32(),
                            buffer_window.as_secs_f32(),
                            refractory.as_secs_f32()
                        ),
                    );
                    log_wake_event(
                        &save_root,
                        wake_id,
                        "wake_detected",
                        json!({
                            "latency_secs": latency.as_secs_f32(),
                            "buffer_secs": buffer_window.as_secs_f32(),
                            "cooldown_secs": refractory.as_secs_f32(),
                        }),
                    );
                }

            }

            if let Some(rem) = after_remaining.as_mut() {
                if let Some(detect_stream_time) = detect_stream_time {
                    let age = stream_time.saturating_sub(detect_stream_time);
                    let target = after_tail.saturating_sub(age);
                    if target < *rem {
                        *rem = target;
                    }
                }
            }

            if let Some(rem_mut) = after_remaining.as_mut() {
                if *rem_mut <= Duration::ZERO {
                    let output_path = next_output_path(&save_root, &file_stub)?;
                    let ts_path = output_path.with_extension("ts");
                    let snapshot = buffer.snapshot_bytes();
                    let snap_len = buffer.total_duration();
                    let detect_stream_time_val = detect_stream_time.unwrap_or(stream_time);
                    let age = stream_time.saturating_sub(detect_stream_time_val);
                    let detect_offset = snap_len.saturating_sub(age);
                    let warn_pre_roll = detect_offset < before;
                    let start_offset = detect_offset.saturating_sub(before);
                    let available = snap_len.saturating_sub(start_offset);
                    let warn_short = available < clip_window;
                    let clip_len = if available < clip_window {
                        available
                    } else {
                        clip_window
                    };
                    if warn_pre_roll || warn_short {
                        eprintln!(
                            "wake clip off-target; pre-roll ~{:.1}s, buffered ~{:.1}s, available ~{:.1}s (request {:.1}s).",
                            detect_offset.as_secs_f32(),
                            snap_len.as_secs_f32(),
                            available.as_secs_f32(),
                            clip_window.as_secs_f32()
                        );
                    }
                    eprintln!(
                        "wake clip timing: output={}, detect_offset_secs={:.3}, start_offset_secs={:.3}, snap_len_secs={:.3}, clip_len_secs={:.3}",
                        output_path.display(),
                        detect_offset.as_secs_f32(),
                        start_offset.as_secs_f32(),
                        snap_len.as_secs_f32(),
                        clip_len.as_secs_f32()
                    );
                    log_run_event(
                        &save_root,
                        &format!(
                            "wake clip timing: output={} detect_offset_secs={:.3} start_offset_secs={:.3} snap_len_secs={:.3} clip_len_secs={:.3}",
                            output_path.display(),
                            detect_offset.as_secs_f32(),
                            start_offset.as_secs_f32(),
                            snap_len.as_secs_f32(),
                            clip_len.as_secs_f32()
                        ),
                    );
                    let wake_id = active_wake_id.unwrap_or(0);
                    if wake_id > 0 {
                        log_wake_event(
                            &save_root,
                            wake_id,
                            "wake_clip_timing",
                            json!({
                                "output": output_path.display().to_string(),
                                "detect_offset_secs": detect_offset.as_secs_f32(),
                                "start_offset_secs": start_offset.as_secs_f32(),
                                "snap_len_secs": snap_len.as_secs_f32(),
                                "clip_len_secs": clip_len.as_secs_f32(),
                            }),
                        );
                    }
                    let (out_w, out_h) = parse_resolution(&resolution).unwrap_or((1080, 1920));
                    let clip_start = if start_offset > Duration::ZERO {
                        Some(start_offset.as_secs_f32())
                    } else {
                        None
                    };

                    if skip_clip_save {
                        println!(
                            "wake detected; skipping clip save (SKIP_CLIP_SAVE=1) duration ~{:.1}s",
                            clip_len.as_secs_f32()
                        );
                        log_run_event(
                            &save_root,
                            &format!(
                                "wake clip skipped (SKIP_CLIP_SAVE=1) duration_secs={:.1}",
                                clip_len.as_secs_f32()
                            ),
                        );
                        if wake_id > 0 {
                            log_wake_event(
                                &save_root,
                                wake_id,
                                "wake_clip_skipped",
                                json!({ "duration_secs": clip_len.as_secs_f32() }),
                            );
                        }
                    } else {
                        let save_root_for_log = save_root.clone();
                        let save_root_for_handle = save_root_for_log.clone();
                        let wake_id_for_log = wake_id;
                        let last_clip_saved_ms = last_clip_saved_ms.clone();
                        let render_duration = clip_len.as_secs_f32();
                        let render_start = clip_start;
                        let save_future = async move {
                            fs::write(&ts_path, &snapshot).context("writing buffered TS snapshot")?;
                            log_run_event(
                                &save_root_for_log,
                                &format!(
                                    "wake clip: wrote ts snapshot {} (secs={:.1})",
                                    ts_path.display(),
                                    snap_len.as_secs_f32()
                                ),
                            );
                            if wake_id_for_log > 0 {
                                log_wake_event(
                                    &save_root_for_log,
                                    wake_id_for_log,
                                    "wake_ts_written",
                                    json!({
                                        "path": ts_path.display().to_string(),
                                        "secs": snap_len.as_secs_f32(),
                                    }),
                                );
                            }
                            log_run_event(
                                &save_root_for_log,
                                &format!(
                                    "wake clip render start: output={} duration_secs={:.1} start_offset={:.3}",
                                    output_path.display(),
                                    render_duration,
                                    render_start.unwrap_or(0.0)
                                ),
                            );
                            run_ffmpeg_from_file(
                                &ts_path,
                                &output_path,
                                out_w,
                                out_h,
                                clip_len,
                                clip_start,
                                Some(ClipTimingHints {
                                    estimated_total_secs: snap_len.as_secs_f32(),
                                    detect_offset_secs: detect_offset.as_secs_f32(),
                                    before_secs: before.as_secs_f32(),
                                }),
                            )
                            .await?;
                            log_run_event(
                                &save_root_for_log,
                                &format!(
                                    "wake clip render done: output={} exit_code=0",
                                    output_path.display()
                                ),
                            );
                            let detected_at = last_detect_instant.map(|t| t.elapsed().as_secs_f32());
                            println!(
                                "wrote wakeword clip: {} (duration ~{:.1}s) | wake at ~50.0s into clip | detect_elapsed_since_save_start={:?}",
                                output_path.display(),
                                clip_len.as_secs_f32(),
                                detected_at
                            );
                            log_run_event(
                                &save_root_for_log,
                                &format!(
                                    "wake clip saved: {} duration_secs={:.1}",
                                    output_path.display(),
                                    clip_len.as_secs_f32()
                                ),
                            );
                            last_clip_saved_ms.store(now_unix_ms(), Ordering::Relaxed);
                            if wake_id_for_log > 0 {
                                log_wake_event(
                                    &save_root_for_log,
                                    wake_id_for_log,
                                    "wake_clip_saved",
                                    json!({
                                        "path": output_path.display().to_string(),
                                        "duration_secs": clip_len.as_secs_f32(),
                                    }),
                                );
                            }
                            Ok::<(), anyhow::Error>(())
                        };

                        let handle = tokio::spawn(async move {
                            if let Err(err) = save_future.await {
                                eprintln!("failed to persist wakeword clip: {err:#}");
                                log_run_event(
                                    &save_root_for_handle,
                                    &format!("wake clip failed: {err:#}"),
                                );
                                if wake_id_for_log > 0 {
                                    log_wake_event(
                                        &save_root_for_handle,
                                        wake_id_for_log,
                                        "wake_clip_failed",
                                        json!({ "error": format!("{err:#}") }),
                                    );
                                }
                            }
                        });
                        pending_saves.push(handle);
                    }

                    after_remaining = None;
                    detect_stream_time = None;
                    fired.store(false, Ordering::Relaxed);
                    detect_ns.store(u64::MAX, Ordering::Relaxed);
                    log_run_event(&save_root, "wake pipeline reset");
                    if wake_id > 0 {
                        log_wake_event(&save_root, wake_id, "wake_pipeline_reset", json!({}));
                    }
                    active_wake_id = None;
                    continue;
                }
                if !made_progress {
                    // If the playlist is stale, still count down so we don't spin forever.
                    *rem_mut = rem_mut.saturating_sub(poll_interval);
                }
            }

            if fired.load(Ordering::Relaxed) && after_remaining.is_none() {
                let buffer_ready = buffer.total_duration() >= clip_window;
                if !buffer_ready {
                    let now = Instant::now();
                    if last_warmup_log
                        .map(|t| now.duration_since(t) >= warmup_log_interval)
                        .unwrap_or(true)
                    {
                        last_warmup_log = Some(now);
                        let msg = format!(
                            "wake detected while buffer warming: {:.1}/{:.1}s; clip will be shorter",
                            buffer.total_duration().as_secs_f32(),
                            clip_window.as_secs_f32()
                        );
                        eprintln!("{msg}");
                        log_run_event(&save_root, &msg);
                    }
                }
                if refractory_until.map(|t| Instant::now() < t).unwrap_or(false) {
                    let force_gap = min_wake_clip_gap.and_then(|gap| {
                        let now_ms = now_unix_ms();
                        let last_ms = last_clip_saved_ms.load(Ordering::Relaxed);
                        let elapsed_ms = now_ms.saturating_sub(last_ms);
                        if elapsed_ms >= gap.as_millis() as u64 {
                            Some(Duration::from_millis(elapsed_ms))
                        } else {
                            None
                        }
                    });
                    if force_gap.is_none() {
                        log_run_event(&save_root, "wake detected during refractory; ignoring");
                        if let Some(wake_id) = active_wake_id {
                            log_wake_event(&save_root, wake_id, "wake_refractory", json!({}));
                        }
                        continue;
                    }
                    let elapsed = force_gap.unwrap_or_else(|| Duration::from_secs(0));
                    let msg = format!(
                        "wake detected during refractory; overriding to satisfy min clip gap ({:.1}s)",
                        elapsed.as_secs_f32()
                    );
                    log_run_event(&save_root, &msg);
                    if let Some(wake_id) = active_wake_id {
                        log_wake_event(
                            &save_root,
                            wake_id,
                            "wake_refractory_override",
                            json!({ "elapsed_secs": elapsed.as_secs_f32() }),
                        );
                    }
                }
                let detect_ns_val = detect_ns.load(std::sync::atomic::Ordering::Relaxed);
                let audio_ns_now = audio_ns.load(std::sync::atomic::Ordering::Relaxed);
                let age_audio = if detect_ns_val != u64::MAX && audio_ns_now >= detect_ns_val {
                    Duration::from_nanos(audio_ns_now - detect_ns_val)
                } else {
                    Duration::ZERO
                };
                refractory_until = Some(Instant::now() + refractory);
                detect_stream_time = Some(stream_time.saturating_sub(age_audio));
                let ns = detect_ns.load(std::sync::atomic::Ordering::Relaxed);
                let base_start = *lock_or_recover(&wake_start, "wake start");
                let detected_instant = if ns != u64::MAX {
                    base_start + Duration::from_nanos(ns)
                } else {
                    Instant::now()
                };
                last_detect_instant = Some(detected_instant);
                let latency = Instant::now().saturating_duration_since(detected_instant);
                let wake_id = match active_wake_id {
                    Some(id) => id,
                    None => {
                        wake_counter += 1;
                        active_wake_id = Some(wake_counter);
                        wake_counter
                    }
                };
                println!(
                    "wake detected; capturing tail to place wake at 50s into clip (latency ~{:.1}s, buffer ~{:.1}s, cooldown {:?})",
                    latency.as_secs_f32(),
                    buffer_window.as_secs_f32(),
                    refractory
                );
                log_run_event(
                    &save_root,
                    &format!(
                        "wake detected: latency={:.1}s buffer={:.1}s cooldown_secs={:.1}",
                        latency.as_secs_f32(),
                        buffer_window.as_secs_f32(),
                        refractory.as_secs_f32()
                    ),
                );
                log_wake_event(
                    &save_root,
                    wake_id,
                    "wake_detected",
                    json!({
                        "latency_secs": latency.as_secs_f32(),
                        "buffer_secs": buffer_window.as_secs_f32(),
                        "cooldown_secs": refractory.as_secs_f32(),
                    }),
                );
                // Wake fired but we have not yet started counting; ensure we do.
                after_remaining = Some(after_tail);
            }

            if made_progress {
                error_backoff = Duration::ZERO;
            }

            sleep(poll_interval).await;
        }

        kill_child_and_wait(&whisper_worker);

        if !pending_saves.is_empty() {
            eprintln!(
                "waiting for {} in-flight clip(s) before shutdown",
                pending_saves.len()
            );
            for handle in pending_saves {
                if let Err(err) = handle.await {
                    eprintln!("clip task failed: {err}");
                }
            }
        }

        // continuous loop
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WakeWorkerStatus {
    audio_ns: u64,
    detect_ns: u64,
    last_word_ns: u64,
    fired: bool,
    updated_unix_ms: u64,
}

fn whisper_isolate_enabled() -> bool {
    std::env::var("WHISPER_ISOLATE")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn spawn_whisper_worker(
    mode: &str,
    status_path: &Path,
    media_url_path: Option<&Path>,
    wake_phrases: &[String],
    log_raw: bool,
) -> Result<std::process::Child> {
    let exe = std::env::current_exe().context("locating autoclip binary")?;
    let mut cmd = StdCommand::new(exe);
    cmd.arg("whisper-worker");
    cmd.env("WHISPER_WORKER_MODE", mode);
    cmd.env("WHISPER_WORKER_STATUS_PATH", status_path);
    cmd.env(
        "WHISPER_WORKER_LOG_RAW",
        if log_raw { "1" } else { "0" },
    );
    cmd.env(
        "WHISPER_WORKER_STATUS_MS",
        std::env::var("WHISPER_WORKER_STATUS_MS").unwrap_or_else(|_| "500".to_string()),
    );
    cmd.env("CLIP_WAKE_WORDS", wake_phrases.join(","));
    cmd.env("WHISPER_ISOLATE", "0");
    if let Some(path) = media_url_path {
        cmd.env("WHISPER_WORKER_MEDIA_URL_PATH", path);
    }
    if let Ok(mic) = std::env::var("MIC_DEVICE") {
        if !mic.trim().is_empty() {
            cmd.env("MIC_DEVICE", mic);
        }
    }
    cmd.spawn().context("spawning whisper worker")
}

fn spawn_wake_status_writer(
    path: PathBuf,
    poll: Duration,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
    detect_ns: Arc<AtomicU64>,
    audio_ns: Arc<AtomicU64>,
    last_word_ns: Arc<AtomicU64>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut last_emit = WakeWorkerStatus {
            audio_ns: 0,
            detect_ns: u64::MAX,
            last_word_ns: u64::MAX,
            fired: false,
            updated_unix_ms: 0,
        };
        while !stop.load(Ordering::Relaxed) {
            let current = WakeWorkerStatus {
                audio_ns: audio_ns.load(Ordering::Relaxed),
                detect_ns: detect_ns.load(Ordering::Relaxed),
                last_word_ns: last_word_ns.load(Ordering::Relaxed),
                fired: fired.load(Ordering::Relaxed),
                updated_unix_ms: now_unix_ms(),
            };
            if current.audio_ns != last_emit.audio_ns
                || current.detect_ns != last_emit.detect_ns
                || current.last_word_ns != last_emit.last_word_ns
                || current.fired != last_emit.fired
                || current.updated_unix_ms.saturating_sub(last_emit.updated_unix_ms) > poll.as_millis() as u64
            {
                if let Err(err) = write_wake_worker_status(&path, &current) {
                    eprintln!("whisper worker: failed to write status: {err:#}");
                } else {
                    last_emit = current;
                }
            }
            std::thread::sleep(poll);
        }
    })
}

fn spawn_wake_status_reader(
    path: PathBuf,
    poll: Duration,
    min_updated_unix_ms: u64,
    last_status_ms: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
    detect_ns: Arc<AtomicU64>,
    audio_ns: Arc<AtomicU64>,
    last_word_ns: Arc<AtomicU64>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut last_seen_ms = min_updated_unix_ms;
        while !stop.load(Ordering::Relaxed) {
            if fired.load(Ordering::Relaxed) {
                break;
            }
            match fs::read_to_string(&path) {
                Ok(contents) => {
                    if let Ok(status) = serde_json::from_str::<WakeWorkerStatus>(&contents) {
                        if status.updated_unix_ms < min_updated_unix_ms {
                            std::thread::sleep(poll);
                            continue;
                        }
                        if status.updated_unix_ms >= last_seen_ms {
                            last_seen_ms = status.updated_unix_ms;
                            last_status_ms.store(status.updated_unix_ms, Ordering::Relaxed);
                            detect_ns.store(status.detect_ns, Ordering::Relaxed);
                            audio_ns.store(status.audio_ns, Ordering::Relaxed);
                            last_word_ns.store(status.last_word_ns, Ordering::Relaxed);
                            if status.fired {
                                fired.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                    }
                }
                Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                Err(err) => {
                    eprintln!("wake status read failed: {err}");
                }
            }
            std::thread::sleep(poll);
        }
    })
}

const TS_PACKET_SIZE: usize = 188;
const PTS_HZ: f64 = 90_000.0;
const PTS_WRAP: u64 = 1 << 33;
const PCR_HZ: f64 = 27_000_000.0;
const PCR_WRAP: u64 = (1 << 33) * 300;

fn duration_to_ns(d: Duration) -> u64 {
    let nanos = d.as_nanos();
    if nanos > u64::MAX as u128 {
        u64::MAX
    } else {
        nanos as u64
    }
}

fn choose_segment_duration(pts: Option<Duration>, playlist: Duration) -> Duration {
    let playlist_secs = playlist.as_secs_f32();
    let playlist_valid = playlist_secs.is_finite() && playlist_secs > 0.0;
    if let Some(pts_dur) = pts {
        let pts_secs = pts_dur.as_secs_f32();
        if pts_secs.is_finite() && pts_secs > 0.0 {
            if playlist_valid {
                let min_ok = playlist_secs * 0.5;
                let max_ok = playlist_secs * 4.0;
                if pts_secs < min_ok || pts_secs > max_ok {
                    return playlist;
                }
            }
            return pts_dur;
        }
    }
    if playlist_valid {
        playlist
    } else {
        pts.unwrap_or(Duration::ZERO)
    }
}

// Extract a best-effort PTS span from a TS segment to align clip timing to real stream time.

fn pts_duration_from_ts(data: &[u8]) -> Option<Duration> {
    if let Some((start, end)) = pts_span_from_ts(data) {
        if let Some(dur) = duration_from_span(start, end, PTS_WRAP, PTS_HZ) {
            return Some(dur);
        }
    }
    pcr_duration_from_ts(data)
}


fn pts_span_from_ts(data: &[u8]) -> Option<(u64, u64)> {
    let sync = find_ts_sync(data)?;
    let mut audio_first = None;
    let mut audio_last = None;
    let mut video_first = None;
    let mut video_last = None;

    let mut idx = sync;
    while idx + TS_PACKET_SIZE <= data.len() {
        let packet = &data[idx..idx + TS_PACKET_SIZE];
        idx += TS_PACKET_SIZE;

        if packet[0] != 0x47 {
            continue;
        }

        let payload_unit_start = (packet[1] & 0x40) != 0;
        let adaptation_control = (packet[3] >> 4) & 0x03;
        if adaptation_control == 0 || adaptation_control == 2 {
            continue;
        }

        let mut payload_idx = 4usize;
        if adaptation_control == 3 {
            let adapt_len = packet[4] as usize;
            payload_idx = payload_idx.saturating_add(1 + adapt_len);
        }
        if payload_idx >= TS_PACKET_SIZE {
            continue;
        }
        if !payload_unit_start {
            continue;
        }

        let payload = &packet[payload_idx..];
        if payload.len() < 9 {
            continue;
        }
        if payload[0] != 0x00 || payload[1] != 0x00 || payload[2] != 0x01 {
            continue;
        }

        let stream_id = payload[3];
        let is_audio = is_audio_stream_id(stream_id);
        let is_video = is_video_stream_id(stream_id);
        if !(is_audio || is_video) {
            continue;
        }

        let flags = payload[7];
        let pts_dts = (flags >> 6) & 0x03;
        if pts_dts < 2 {
            continue;
        }

        let pts_start = 9;
        if payload.len() < pts_start + 5 {
            continue;
        }
        let Some(pts) = parse_pts(&payload[pts_start..pts_start + 5]) else {
            continue;
        };

        if is_audio {
            if audio_first.is_none() {
                audio_first = Some(pts);
            }
            audio_last = Some(pts);
        } else if is_video {
            if video_first.is_none() {
                video_first = Some(pts);
            }
            video_last = Some(pts);
        }
    }

    if let (Some(first), Some(last)) = (audio_first, audio_last) {
        if last != first {
            return Some((first, last));
        }
    }
    if let (Some(first), Some(last)) = (video_first, video_last) {
        if last != first {
            return Some((first, last));
        }
    }
    None
}


fn pcr_duration_from_ts(data: &[u8]) -> Option<Duration> {
    let (start, end) = pcr_span_from_ts(data)?;
    duration_from_span(start, end, PCR_WRAP, PCR_HZ)
}

fn pcr_span_from_ts(data: &[u8]) -> Option<(u64, u64)> {
    let sync = find_ts_sync(data)?;
    let mut first = None;
    let mut last = None;

    let mut idx = sync;
    while idx + TS_PACKET_SIZE <= data.len() {
        let packet = &data[idx..idx + TS_PACKET_SIZE];
        idx += TS_PACKET_SIZE;

        if packet[0] != 0x47 {
            continue;
        }
        if let Some(pcr) = parse_pcr(packet) {
            if first.is_none() {
                first = Some(pcr);
            }
            last = Some(pcr);
        }
    }

    match (first, last) {
        (Some(f), Some(l)) if l != f => Some((f, l)),
        _ => None,
    }
}

fn parse_pcr(packet: &[u8]) -> Option<u64> {
    if packet.len() < TS_PACKET_SIZE {
        return None;
    }
    let adaptation_control = (packet[3] >> 4) & 0x03;
    if adaptation_control == 0 || adaptation_control == 1 {
        return None;
    }
    let adapt_len = packet[4] as usize;
    if adapt_len < 7 || 5 + adapt_len > packet.len() {
        return None;
    }
    let flags = packet[5];
    if (flags & 0x10) == 0 {
        return None;
    }
    let pcr = &packet[6..12];
    let base = ((pcr[0] as u64) << 25)
        | ((pcr[1] as u64) << 17)
        | ((pcr[2] as u64) << 9)
        | ((pcr[3] as u64) << 1)
        | ((pcr[4] as u64) >> 7);
    let ext = (((pcr[4] & 0x01) as u64) << 8) | (pcr[5] as u64);
    Some(base * 300 + ext)
}

fn duration_from_span(start: u64, end: u64, wrap: u64, hz: f64) -> Option<Duration> {
    let delta = if end >= start {
        end - start
    } else {
        (end + wrap) - start
    };
    if delta == 0 {
        return None;
    }
    let secs = (delta as f64) / hz;
    if !secs.is_finite() || secs <= 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(secs))
}


fn find_ts_sync(data: &[u8]) -> Option<usize> {
    let max_scan = usize::min(data.len(), TS_PACKET_SIZE * 4);
    for start in 0..max_scan {
        if data[start] != 0x47 {
            continue;
        }
        let next = start + TS_PACKET_SIZE;
        if next < data.len() && data[next] == 0x47 {
            return Some(start);
        }
    }
    None
}

fn is_audio_stream_id(id: u8) -> bool {
    (0xC0..=0xDF).contains(&id)
}

fn is_video_stream_id(id: u8) -> bool {
    (0xE0..=0xEF).contains(&id)
}

fn parse_pts(data: &[u8]) -> Option<u64> {
    if data.len() < 5 {
        return None;
    }
    let b0 = data[0];
    if (b0 & 0xF0) != 0x20 && (b0 & 0xF0) != 0x30 {
        return None;
    }
    let b1 = data[1];
    let b2 = data[2];
    let b3 = data[3];
    let b4 = data[4];

    let pts = (((b0 >> 1) & 0x07) as u64) << 30
        | (b1 as u64) << 22
        | (((b2 >> 1) & 0x7F) as u64) << 15
        | (b3 as u64) << 7
        | (((b4 >> 1) & 0x7F) as u64);
    Some(pts)
}

fn lock_or_recover<'a, T>(mutex: &'a Mutex<T>, label: &str) -> MutexGuard<'a, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            eprintln!("warn: {label} lock poisoned; continuing with recovered state");
            poisoned.into_inner()
        }
    }
}

pub(crate) fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut out = s[..max.saturating_sub(3)].to_string();
    out.push_str("...");
    out
}

fn auto_assign_gpus_for_tools() {
    let user_hwaccel = std::env::var("FFMPEG_HWACCEL")
        .ok()
        .filter(|v| !v.trim().is_empty());
    let user_hwaccel_device = std::env::var("FFMPEG_HWACCEL_DEVICE")
        .ok()
        .filter(|v| !v.trim().is_empty());
    let user_encoder = std::env::var("FFMPEG_ENCODER")
        .ok()
        .filter(|v| !v.trim().is_empty());
    let user_whisper_flag = std::env::var("WHISPER_GPU").ok();
    let user_whisper = user_whisper_flag
        .as_deref()
        .filter(|v| !v.trim().is_empty());
    let user_whisper_device = std::env::var("WHISPER_GPU_DEVICE")
        .ok()
        .filter(|v| !v.trim().is_empty());
    #[cfg(feature = "ort")]
    let user_face_backend = std::env::var("CLIP_FACE_BACKEND")
        .ok()
        .filter(|v| !v.trim().is_empty());
    #[cfg(feature = "ort")]
    let user_pose_backend = std::env::var("CLIP_POSE_BACKEND")
        .ok()
        .filter(|v| !v.trim().is_empty());

    let gpus = detect_nvidia_gpus();
    if !gpus.is_empty() {
        let whisper_disabled = user_whisper_flag
            .as_deref()
            .and_then(|v| parse_bool(v))
            .map(|v| !v)
            .unwrap_or(false);
        let whisper_min_free = std::env::var("WHISPER_MIN_FREE_VRAM_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(2048);
        let mut whisper_device = user_whisper_device
            .as_deref()
            .and_then(parse_device_index);
        if whisper_device.is_none() && !whisper_disabled {
            whisper_device = gpu::pick_best_nvidia_device(whisper_min_free, "whisper");
        }
        if !whisper_disabled {
            if user_whisper.is_none() {
                if whisper_device.is_some() {
                    std::env::set_var("WHISPER_GPU", "1");
                }
            }
            if user_whisper_device.is_none() {
                if let Some(device) = whisper_device {
                    std::env::set_var("WHISPER_GPU_DEVICE", device.to_string());
                    eprintln!("auto GPU assign for whisper: {device}");
                }
            }
        }

        let mut ffmpeg_device = user_hwaccel_device
            .as_deref()
            .and_then(parse_device_index);
        if user_hwaccel.is_none() && ffmpeg_device.is_none() {
            let min_free = ffmpeg_min_free_vram_mb();
            ffmpeg_device = gpu::pick_best_nvidia_device_excluding(min_free, whisper_device, "ffmpeg")
                .or_else(|| gpu::pick_best_nvidia_device(min_free, "ffmpeg"));
            if let Some(device) = ffmpeg_device {
                std::env::set_var("FFMPEG_HWACCEL", "cuda");
                std::env::set_var("FFMPEG_HWACCEL_DEVICE", device.to_string());
                eprintln!("auto GPU assign for ffmpeg: {device}");
            }
        }
        if user_encoder.is_none() && ffmpeg_device.is_some() && ffmpeg_has_encoder("h264_nvenc") {
            std::env::set_var("FFMPEG_ENCODER", "h264_nvenc");
        }
        #[cfg(feature = "ort")]
        if user_face_backend.is_none() {
            std::env::set_var("CLIP_FACE_BACKEND", "auto");
            eprintln!("auto GPU assign: face backend auto");
        }
        #[cfg(feature = "ort")]
        if user_pose_backend.is_none() {
            std::env::set_var("CLIP_POSE_BACKEND", "auto");
            eprintln!("auto GPU assign: pose backend auto");
        }
        return;
    }

    if user_hwaccel.is_none() {
        if let Some(hw) = detect_best_hwaccel() {
            std::env::set_var("FFMPEG_HWACCEL", &hw);
            std::env::remove_var("FFMPEG_HWACCEL_DEVICE");
            eprintln!("auto GPU assign: using {hw} (no NVIDIA detected)");
        }
    }

    if user_encoder.is_none() {
        if let Some(enc) = detect_best_encoder() {
            std::env::set_var("FFMPEG_ENCODER", &enc);
            eprintln!("auto GPU assign: using ffmpeg encoder {enc}");
        }
    }
}

fn log_gpu_assignments() {
    let whisper_gpu_env = std::env::var("WHISPER_GPU").unwrap_or_else(|_| "(unset)".to_string());
    let whisper_clip_gpu_env =
        std::env::var("WHISPER_CLIP_GPU").unwrap_or_else(|_| "(unset)".to_string());
    let whisper_device = std::env::var("WHISPER_GPU_DEVICE")
        .ok()
        .and_then(|v| parse_device_index(&v));
    let whisper_clip_device = std::env::var("WHISPER_CLIP_GPU_DEVICE")
        .ok()
        .and_then(|v| parse_device_index(&v));
    let ffmpeg_hwaccel = std::env::var("FFMPEG_HWACCEL").unwrap_or_else(|_| "(unset)".to_string());
    let ffmpeg_device = ffmpeg_hwaccel_device_index();
    let ffmpeg_encoder = std::env::var("FFMPEG_ENCODER").unwrap_or_else(|_| "(unset)".to_string());

    eprintln!(
        "gpu assign: whisper_gpu={whisper_gpu_env} device={:?} clip_gpu={whisper_clip_gpu_env} clip_device={:?}",
        whisper_device, whisper_clip_device
    );
    eprintln!(
        "gpu assign: ffmpeg_hwaccel={ffmpeg_hwaccel} device={:?} encoder={ffmpeg_encoder}",
        ffmpeg_device
    );
    if ffmpeg_hwaccel.eq_ignore_ascii_case("cuda") {
        if let (Some(w), Some(f)) = (whisper_device, ffmpeg_device) {
            if w == f {
                eprintln!(
                    "gpu assign warning: whisper and ffmpeg share GPU {w}; VRAM contention possible"
                );
            }
        }
        if let (Some(w), Some(f)) = (whisper_clip_device, ffmpeg_device) {
            if w == f {
                eprintln!(
                    "gpu assign warning: clip whisper and ffmpeg share GPU {w}; VRAM contention possible"
                );
            }
        }
    }
    gpu::log_nvidia_snapshot("startup");
}

fn ensure_cuda_device_order() {
    let order = std::env::var("CUDA_DEVICE_ORDER")
        .ok()
        .map(|v| v.trim().to_ascii_uppercase())
        .unwrap_or_default();
    if order.is_empty() {
        std::env::set_var("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
        eprintln!("cuda: defaulting CUDA_DEVICE_ORDER=PCI_BUS_ID");
    }
}

fn ffmpeg_bin() -> String {
    std::env::var("FFMPEG_BIN").unwrap_or_else(|_| "ffmpeg".to_string())
}

fn collect_env_cookies_for_download() -> Option<String> {
    let mut cookie_parts: Vec<String> = Vec::new();
    for key in ["COOKIE_HEADER", "KICK_COOKIE"] {
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

fn ffmpeg_headers_for_page(page_url: &str) -> Option<String> {
    let mut headers: Vec<String> = Vec::new();
    headers.push(format!("Referer: {page_url}"));
    if let Some(origin) = origin_for_page(page_url) {
        headers.push(format!("Origin: {origin}"));
    }
    if let Some(cookie) = collect_env_cookies_for_download() {
        headers.push(format!("Cookie: {cookie}"));
    }
    if headers.is_empty() {
        None
    } else {
        Some(headers.join("\r\n") + "\r\n")
    }
}

async fn download_m3u8_to_mp4(source_url: &str, output: &Path, page_url: &str) -> Result<()> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).context("creating VOD download directory")?;
    }
    let mut cmd = Command::new(ffmpeg_bin());
    cmd.arg("-y").arg("-loglevel").arg("warning").arg("-stats");
    if let Some(headers) = ffmpeg_headers_for_page(page_url) {
        cmd.arg("-headers").arg(headers);
    }
    cmd.arg("-i").arg(source_url);
    cmd.arg("-c").arg("copy");
    cmd.arg("-bsf:a").arg("aac_adtstoasc");
    cmd.arg("-movflags").arg("+faststart");
    cmd.arg(output.as_os_str());
    let output_res = cmd.output().await.context("running ffmpeg VOD download")?;
    if !output_res.status.success() {
        let stderr = String::from_utf8_lossy(&output_res.stderr);
        anyhow::bail!(
            "ffmpeg VOD download failed (status {}): {}",
            output_res.status,
            stderr.trim()
        );
    }
    Ok(())
}

async fn maybe_enroll_face_id_from_media(page_url: &str, input_path: &Path) {
    if !face_id_enabled() {
        return;
    }
    let file_path = face_id::face_id_file_for_stream(page_url);
    if file_path.exists() {
        return;
    }
    let Some(input) = input_path.to_str() else {
        eprintln!(
            "face id: skipped VOD enrollment (non-utf8 path {})",
            input_path.display()
        );
        return;
    };
    let _ = std::env::set_var(
        "CLIP_FACE_ID_FILE",
        file_path.to_string_lossy().as_ref(),
    );
    match clip_detect::enroll_face_id_from_media_url(input, &file_path).await {
        Ok(true) => {
            eprintln!(
                "face id: enrolled from VOD media -> {}",
                file_path.display()
            );
        }
        Ok(false) => {
            eprintln!("face id: VOD media did not yield a face");
        }
        Err(err) => {
            eprintln!("face id: VOD media enrollment failed: {err:#}");
        }
    }
}

async fn run_kick_latest_vod(page_url: &str) -> Result<()> {
    let page_url = normalize_page_url(page_url);
    let Some(slug) = kick_slug_from_url(&page_url) else {
        anyhow::bail!("expected a Kick channel URL like https://kick.com/<channel>");
    };
    if std::env::var("CLIP_REPROCESS_CONTINUE_ON_ERROR").ok().is_none() {
        std::env::set_var("CLIP_REPROCESS_CONTINUE_ON_ERROR", "1");
    }
    if std::env::var("CLIP_LLM_ENABLE").ok().is_none() {
        std::env::set_var("CLIP_LLM_ENABLE", "1");
    }
    if face_id_enabled() {
        maybe_auto_enroll_face_id(&page_url).await;
    }
    let hls = HlsClient::new()?;
    let vod: KickVodInfo = hls.fetch_kick_latest_vod(&page_url, &slug).await?;
    let source = sanitize_m3u8_url(&vod.source);
    let inputs_dir = Path::new("inputs");
    let safe_slug = sanitize_title_for_filename(&slug);
    let output_path = inputs_dir.join(format!("kick_vod_{safe_slug}_{}.mp4", vod.id));
    if output_path.exists() {
        let size = output_path.metadata().map(|m| m.len()).unwrap_or(0);
        if size > 0 {
            eprintln!("kick vod: using existing download {}", output_path.display());
            maybe_enroll_face_id_from_media(&page_url, &output_path).await;
            return run_reprocess_ts(output_path.to_str().unwrap()).await;
        }
    }
    eprintln!(
        "kick vod: downloading latest VOD {} (duration {:?}s)",
        vod.id,
        vod.duration
    );
    download_m3u8_to_mp4(&source, &output_path, &page_url).await?;
    eprintln!("kick vod: downloaded to {}", output_path.display());
    maybe_enroll_face_id_from_media(&page_url, &output_path).await;
    run_reprocess_ts(output_path.to_str().unwrap()).await
}

fn ffmpeg_has_encoder(name: &str) -> bool {
    let encoders = detect_ffmpeg_encoders();
    let needle = name.to_ascii_lowercase();
    encoders.iter().any(|enc| enc == &needle)
}

fn detect_ffmpeg_encoders() -> Vec<String> {
    let output = std::process::Command::new(ffmpeg_bin())
        .arg("-hide_banner")
        .arg("-encoders")
        .output();

    let Ok(out) = output else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut encoders = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('V') {
            let mut parts = trimmed.split_whitespace();
            let _flags = parts.next();
            if let Some(name) = parts.next() {
                encoders.push(name.to_ascii_lowercase());
            }
        }
    }
    encoders
}

fn detect_ffmpeg_hwaccels() -> Vec<String> {
    let output = std::process::Command::new(ffmpeg_bin())
        .arg("-hide_banner")
        .arg("-hwaccels")
        .output();

    let Ok(out) = output else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut hwaccels = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.ends_with(':') {
            continue;
        }
        hwaccels.push(trimmed.to_ascii_lowercase());
    }
    hwaccels
}

fn detect_best_hwaccel() -> Option<String> {
    let hwaccels = detect_ffmpeg_hwaccels();
    if hwaccels.is_empty() {
        return None;
    }

    #[cfg(target_os = "windows")]
    let preferred = ["d3d11va", "qsv", "dxva2"];
    #[cfg(target_os = "macos")]
    let preferred = ["videotoolbox"];
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    let preferred = ["qsv", "vaapi"];

    for hw in preferred {
        if hwaccels.iter().any(|v| v == hw) {
            return Some(hw.to_string());
        }
    }
    None
}

fn detect_best_encoder() -> Option<String> {
    let encoders = detect_ffmpeg_encoders();
    if encoders.is_empty() {
        return None;
    }

    let preferred = ["h264_qsv", "h264_amf", "h264_videotoolbox"];
    for enc in preferred {
        if encoders.iter().any(|v| v == enc) {
            return Some(enc.to_string());
        }
    }
    None
}

fn detect_nvidia_gpus() -> Vec<(u32, u64)> {
    let mut gpus = gpu::query_nvidia_free_vram_all().unwrap_or_default();
    gpus.sort_by(|a, b| b.1.cmp(&a.1));
    gpus
}

fn build_hwaccel_flags(hw: &str, device: Option<&str>) -> Vec<String> {
    let hw = hw.trim();
    if hw.is_empty() {
        return Vec::new();
    }
    let mut out = vec!["-hwaccel".to_string(), hw.to_string()];
    if let Some(dev) = device.map(str::trim).filter(|d| !d.is_empty()) {
        out.push("-hwaccel_device".to_string());
        out.push(dev.to_string());
    }
    out
}

fn ffmpeg_hwaccel_flags() -> Vec<String> {
    let hw = std::env::var("FFMPEG_HWACCEL").unwrap_or_default();
    let dev = std::env::var("FFMPEG_HWACCEL_DEVICE").ok();
    build_hwaccel_flags(&hw, dev.as_deref())
}

fn ffmpeg_hwaccel_flags_for_encode(use_hw_encode: bool) -> Vec<String> {
    if use_hw_encode {
        return ffmpeg_hwaccel_flags();
    }
    if let Ok(fallback) = std::env::var("FFMPEG_HWACCEL_FALLBACK") {
        let trimmed = fallback.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
            return Vec::new();
        }
        let dev = if trimmed.eq_ignore_ascii_case("cuda") {
            std::env::var("FFMPEG_HWACCEL_DEVICE").ok()
        } else {
            None
        };
        return build_hwaccel_flags(trimmed, dev.as_deref());
    }
    Vec::new()
}

fn ffmpeg_scale_cuda_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let output = std::process::Command::new("ffmpeg")
            .arg("-hide_banner")
            .arg("-filters")
            .output();
        let Ok(out) = output else { return false; };
        if !out.status.success() {
            return false;
        }
        let stdout = String::from_utf8_lossy(&out.stdout).to_ascii_lowercase();
        stdout.lines().any(|line| line.contains("scale_cuda"))
    })
}

fn unstable_streams() -> &'static Mutex<HashSet<String>> {
    static UNSTABLE: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    UNSTABLE.get_or_init(|| Mutex::new(HashSet::new()))
}

fn mark_stream_unstable(input: &str) {
    if let Ok(mut set) = unstable_streams().lock() {
        set.insert(input.to_string());
    }
}

fn is_stream_unstable(input: &str) -> bool {
    unstable_streams()
        .lock()
        .map(|set| set.contains(input))
        .unwrap_or(false)
}

fn stderr_indicates_filter_issue(stderr: &str) -> bool {
    let s = stderr.to_ascii_lowercase();
    s.contains("error reinitializing filters")
        || s.contains("failed to inject frame into filter network")
        || s.contains("impossible to convert between the formats supported")
        || s.contains("error while processing the decoded data for stream")
}

fn ffmpeg_status_indicates_crash(status: &std::process::ExitStatus, stderr: &str) -> bool {
    let stderr_lower = stderr.to_ascii_lowercase();
    if stderr_lower.contains("0xc0000005")
        || stderr_lower.contains("status_access_violation")
        || stderr_lower.contains("access violation")
    {
        return true;
    }
    if let Some(code) = status.code() {
        if code == 0xC0000005_u32 as i32 || code == -1073741819 {
            return true;
        }
    }
    false
}

fn ffmpeg_video_encoder() -> (String, bool, bool, bool) {
    if let Ok(enc) = std::env::var("FFMPEG_ENCODER") {
        if !enc.trim().is_empty() {
            let is_nvenc = enc.to_ascii_lowercase().contains("nvenc");
            let is_hw = is_hw_encoder(&enc);
            return (enc, is_nvenc, is_hw, true);
        }
    }

    let hw = std::env::var("FFMPEG_HWACCEL").unwrap_or_default();
    if hw.eq_ignore_ascii_case("cuda") {
        return ("h264_nvenc".to_string(), true, true, false);
    }

    ("libx264".to_string(), false, false, false)
}

fn temp_output_path(out_path: &Path) -> PathBuf {
    let parent = out_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = out_path
        .file_stem()
        .and_then(|v| v.to_str())
        .unwrap_or("output");
    let ext = out_path.extension().and_then(|v| v.to_str()).unwrap_or("");
    let mut name = String::from(stem);
    name.push_str(".partial");
    if !ext.is_empty() {
        name.push('.');
        name.push_str(ext);
    }
    parent.join(name)
}

fn temp_output_path_with_suffix(out_path: &Path, suffix: &str) -> PathBuf {
    let parent = out_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = out_path
        .file_stem()
        .and_then(|v| v.to_str())
        .unwrap_or("output");
    let ext = out_path.extension().and_then(|v| v.to_str()).unwrap_or("");
    let mut name = format!("{stem}.{suffix}");
    if !ext.is_empty() {
        name.push('.');
        name.push_str(ext);
    }
    parent.join(name)
}

fn finalize_output_path(temp_path: &Path, out_path: &Path) -> Result<()> {
    if temp_path == out_path {
        return Ok(());
    }
    if !temp_path.exists() {
        return Ok(());
    }
    if out_path.exists() {
        fs::remove_file(out_path).with_context(|| {
            format!("removing existing output {}", out_path.display())
        })?;
    }
    fs::rename(temp_path, out_path)
        .with_context(|| format!("renaming {} -> {}", temp_path.display(), out_path.display()))?;
    Ok(())
}

fn cleanup_temp_output(temp_path: &Path) {
    let _ = fs::remove_file(temp_path);
}

fn is_hw_encoder(encoder: &str) -> bool {
    let enc = encoder.to_ascii_lowercase();
    enc.contains("nvenc")
        || enc.contains("qsv")
        || enc.contains("amf")
        || enc.contains("videotoolbox")
        || enc.contains("vaapi")
}

fn extract_output_index(stem: &str, stub: &str) -> Option<u32> {
    let prefix = format!("{stub}_");
    let rest = stem.strip_prefix(&prefix)?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<u32>().ok()
}

fn next_output_path(save_dir: &str, stub: &str) -> Result<PathBuf> {
    let dir = Path::new(save_dir);
    fs::create_dir_all(dir).with_context(|| format!("creating save dir {save_dir}"))?;

    let counter_dir = non_video_dir(save_dir);
    let counter_path = counter_dir.join(format!(".{stub}_counter"));
    let counter_idx = fs::read_to_string(&counter_path)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(0);
    let mut max_idx = counter_idx;
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|v| v.to_str()).unwrap_or("") != "mp4" {
                continue;
            }
            let stem = path.file_stem().and_then(|v| v.to_str()).unwrap_or("");
            if let Some(idx) = extract_output_index(stem, stub) {
                if idx > max_idx {
                    max_idx = idx;
                }
            }
        }
    }
    let next_idx = max_idx.saturating_add(1).max(1);
    let payload = format!("{next_idx}\n");
    write_atomic_file(&counter_path, payload.as_bytes())
        .with_context(|| format!("writing {}", counter_path.display()))?;
    Ok(dir.join(format!("{stub}_{next_idx:03}.mp4")))
}

async fn run_ffmpeg_30s(input_hls: &Url, out_path: &Path, out_w: u32, out_h: u32) -> Result<()> {
    run_ffmpeg_internal(
        FfmpegRenderSpec::new(input_hls.as_str(), out_path, out_w, out_h)
            .with_duration(Some(30.0))
            .with_regen_pts(false),
    )
    .await
}

#[derive(Clone, Copy, Debug)]
struct ClipTimingHints {
    estimated_total_secs: f32,
    detect_offset_secs: f32,
    before_secs: f32,
}

#[derive(Debug, Clone)]
struct FfmpegRenderSpec {
    input: String,
    out_path: PathBuf,
    out_w: u32,
    out_h: u32,
    duration_secs: Option<f32>,
    start_offset_secs: Option<f32>,
    regen_pts: bool,
    force_ts_input: bool,
    progress: Option<ProgressSpec>,
    live_fast: bool,
    fast_preset: bool,
    fast_seek: bool,
}

impl FfmpegRenderSpec {
    fn new(input: &str, out_path: &Path, out_w: u32, out_h: u32) -> Self {
        Self {
            input: input.to_string(),
            out_path: out_path.to_path_buf(),
            out_w,
            out_h,
            duration_secs: None,
            start_offset_secs: None,
            regen_pts: true,
            force_ts_input: false,
            progress: None,
            live_fast: false,
            fast_preset: false,
            fast_seek: false,
        }
    }

    fn with_duration(mut self, duration_secs: Option<f32>) -> Self {
        self.duration_secs = duration_secs;
        self
    }

    fn with_start_offset(mut self, start_offset_secs: Option<f32>) -> Self {
        self.start_offset_secs = start_offset_secs;
        self
    }

    fn with_regen_pts(mut self, regen_pts: bool) -> Self {
        self.regen_pts = regen_pts;
        self
    }

    fn with_force_ts_input(mut self, force_ts_input: bool) -> Self {
        self.force_ts_input = force_ts_input;
        self
    }

    fn with_progress(mut self, progress: Option<ProgressSpec>) -> Self {
        self.progress = progress;
        self
    }

    fn with_live_fast(mut self, live_fast: bool) -> Self {
        self.live_fast = live_fast;
        self
    }

    fn with_fast_preset(mut self, fast_preset: bool) -> Self {
        self.fast_preset = fast_preset;
        self
    }

    fn with_fast_seek(mut self, fast_seek: bool) -> Self {
        self.fast_seek = fast_seek;
        self
    }
}

struct LayoutDecision {
    layout_is_stacked: bool,
    face_only: bool,
    fullscreen_fill: bool,
    fullscreen_track: Option<clip_layout::FaceTrack>,
    fullscreen_center_override: Option<NormalizedPoint>,
}

fn decide_stacked_layout(
    layout: &ClipLayoutConfig,
    hints: &ClipLayoutHints,
    face_focus: Option<NormalizedPoint>,
) -> LayoutDecision {
    let mut decision = LayoutDecision {
        layout_is_stacked: true,
        face_only: false,
        fullscreen_fill: false,
        fullscreen_track: None,
        fullscreen_center_override: None,
    };
    let face_only_area_threshold = 0.40;
    let face_center = |rect: crate::clip_layout::NormalizedRect| NormalizedPoint {
        x: (rect.x + rect.w / 2.0).clamp(0.0, 1.0),
        y: (rect.y + rect.h / 2.0).clamp(0.0, 1.0),
    };
    let fallback_enabled = clip_layout::face_fallback_enabled();
    let face_found = layout.face_crop.is_some() || hints.face_box.is_some() || hints.face_track.is_some();
    let mut gameplay_found = gameplay_enabled() && hints.game_center.is_some();
    let mut gameplay_guess: Option<bool> = None;
    if !gameplay_found {
        if let Some((guess, area, edge)) =
            guess_gameplay_low_resource(hints, face_only_area_threshold)
        {
            gameplay_guess = Some(guess);
            if guess {
                gameplay_found = true;
                if low_resource_enabled() {
                    eprintln!(
                        "clip layout: low-resource guess -> gameplay present (face area {:.3}, edge {:.3})",
                        area, edge
                    );
                } else {
                    eprintln!(
                        "clip layout: gameplay not detected; face suggests gameplay (area {:.3}, edge {:.3})",
                        area, edge
                    );
                }
            } else if low_resource_enabled() {
                eprintln!(
                    "clip layout: low-resource guess -> no gameplay (face area {:.3}, edge {:.3})",
                    area, edge
                );
            }
        }
    }

    let face_rect = if fallback_enabled || hints.face_track.is_some() {
        face_rect_from_hints(hints)
    } else {
        None
    };
    let face_inside_game = gameplay_found
        && hints
            .game_region
            .and_then(|game| face_rect.map(|face| rect_contains_rect(game, face, 0.01)))
            .unwrap_or(false);
    if face_inside_game {
        decision.layout_is_stacked = false;
        decision.fullscreen_fill = true;
        eprintln!(
            "clip layout: face region inside gameplay region; using single-frame layout"
        );
        return decision;
    }

    if face_found && !gameplay_found {
        if gameplay_guess == Some(false) {
            eprintln!("clip layout: gameplay guess suggests no gameplay; using single-frame");
        }
        decision.layout_is_stacked = false;
        if layout.face_crop.is_some() {
            decision.face_only = true;
            eprintln!("clip layout: no gameplay detected; using face-only layout");
        } else {
            decision.fullscreen_fill = true;
            decision.fullscreen_track = hints.face_track.clone();
            if decision.fullscreen_track.is_none() && fallback_enabled {
                if let Some(face_box) = hints.face_box {
                    let rect = if let Some(focus) = face_focus {
                        clip_layout::recenter_rect_x(face_box, focus.x)
                    } else {
                        face_box
                    };
                    decision.fullscreen_track = Some(clip_layout::synthesize_face_track(rect));
                }
            }
            decision.fullscreen_center_override = if fallback_enabled {
                face_focus
                    .or_else(|| hints.face_box.map(face_center))
                    .or_else(|| {
                        hints
                            .face_track
                            .as_ref()
                            .and_then(|track| track.points.first().map(|p| face_center(p.rect)))
                    })
            } else {
                face_focus.or_else(|| {
                    hints
                        .face_track
                        .as_ref()
                        .and_then(|track| track.points.first().map(|p| face_center(p.rect)))
                })
            };
            eprintln!("clip layout: no gameplay detected; using single-frame face layout");
        }
        return decision;
    }

    if !face_found {
        decision.layout_is_stacked = false;
        decision.fullscreen_fill = true;
        eprintln!("clip layout: no streamer cam detected; using full-frame gameplay fill");
    }

    decision
}

async fn run_ffmpeg_from_file(
    input_path: &Path,
    out_path: &Path,
    out_w: u32,
    out_h: u32,
    duration: Duration,
    start_offset: Option<f32>,
    timing: Option<ClipTimingHints>,
) -> Result<()> {
    let mut duration_secs = duration.as_secs_f32();
    let mut start_secs = start_offset.unwrap_or(0.0).max(0.0);
    let actual_secs = probe_media_duration_secs(input_path).await;
    let mut telemetry_scale = 1.0f32;
    if let Some(timing) = timing {
        if let Some(actual_secs) = actual_secs {
            if timing.estimated_total_secs > 0.0 {
                let scale = actual_secs / timing.estimated_total_secs;
                if scale.is_finite() && scale > 0.0 {
                    telemetry_scale = scale;
                    let detect_actual = timing.detect_offset_secs * scale;
                    let scaled_start = (detect_actual - timing.before_secs).max(0.0);
                    if (scaled_start - start_secs).abs() > 0.05 {
                        eprintln!(
                            "ffprobe: scaling start offset from {:.3}s -> {:.3}s (actual {:.3}s vs est {:.3}s)",
                            start_secs,
                            scaled_start,
                            actual_secs,
                            timing.estimated_total_secs
                        );
                    }
                    start_secs = scaled_start;
                }
            }
        }
        let detect_actual = timing.detect_offset_secs * telemetry_scale;
        let wake_in_clip = (detect_actual - start_secs).max(0.0);
        let delta = wake_in_clip - timing.before_secs;
        eprintln!(
            "wake telemetry: target={:.3}s actual={:.3}s delta={:+.3}s (scale={:.6})",
            timing.before_secs,
            wake_in_clip,
            delta,
            telemetry_scale
        );
    }

    if let Some(actual_secs) = actual_secs {
        let remaining = (actual_secs - start_secs).max(0.0);
        if (actual_secs - duration_secs).abs() > 1.0 {
            eprintln!(
                "ffprobe: input duration ~{:.1}s (start {:.1}s, requested {:.1}s)",
                actual_secs,
                start_secs,
                duration_secs
            );
        }
        if remaining > 0.0 && remaining + 0.5 < duration_secs {
            eprintln!(
                "ffprobe: clamping clip length to ~{:.1}s based on input duration",
                remaining
            );
            duration_secs = remaining;
        }
    }
    let start_offset = if start_secs > 0.0 { Some(start_secs) } else { None };
    let live_fast = timing.is_some() && live_fast_enabled();
    let force_ts_input = is_ts_input(input_path);
    run_ffmpeg_internal(
        FfmpegRenderSpec::new(
            input_path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("non-utf8 input path"))?,
            out_path,
            out_w,
            out_h,
        )
        .with_duration(Some(duration_secs))
        .with_start_offset(start_offset)
        .with_force_ts_input(force_ts_input)
        .with_live_fast(live_fast),
    )
    .await
}

async fn probe_media_duration_secs(path: &Path) -> Option<f32> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=1")
        .arg(path.as_os_str())
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let val = stdout.lines().next()?.trim().parse::<f32>().ok()?;
    if val.is_finite() && val > 0.0 {
        Some(val)
    } else {
        None
    }
}

fn build_ffmpeg_filters(out_w: u32, out_h: u32) -> (String, String, String) {
    let vf_cpu = format!(
        "scale={}:{}:force_original_aspect_ratio=decrease,pad={}:{}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        out_w, out_h, out_w, out_h
    );

    // GPU path when decoding in software: normalize -> upload -> scale on CUDA -> download -> pad in software.
    // The explicit format right after hwdownload avoids auto-inserted scale/format filters
    // that can error with "Impossible to convert between the formats supported by ... auto_scale".
    let vf_gpu_sw = format!(
        "format=yuv420p,hwupload_cuda,scale_cuda=w={}:h={}:force_original_aspect_ratio=decrease:format=nv12:interp_algo=lanczos,hwdownload,format=yuv420p,pad={}:{}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        out_w, out_h, out_w, out_h
    );
    // GPU path when decoding to CUDA frames: scale directly on GPU, then download for padding.
    let vf_gpu_hw = format!(
        "scale_cuda=w={}:h={}:force_original_aspect_ratio=decrease:format=nv12:interp_algo=lanczos,hwdownload,format=yuv420p,pad={}:{}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        out_w, out_h, out_w, out_h
    );

    (vf_cpu, vf_gpu_sw, vf_gpu_hw)
}

fn axis_center_expr(axis: &str, crop_dim: u32, center: f32) -> String {
    let center = center.clamp(0.0, 1.0);
    let half = crop_dim as f32 / 2.0;
    let half_str = format!("{half:.1}");
    format!(
        "max(min({axis}*{center:.4}-{half_str}\\, {axis}-{crop_dim})\\, 0)"
    )
}

fn build_ffmpeg_fill_filters(
    out_w: u32,
    out_h: u32,
    center: NormalizedPoint,
) -> (String, String, String) {
    let x = axis_center_expr("iw", out_w, center.x);
    let y = axis_center_expr("ih", out_h, center.y);
    let vf_cpu = format!(
        "scale={}:{}:force_original_aspect_ratio=increase,crop={}:{}:{x}:{y},format=yuv420p",
        out_w, out_h, out_w, out_h
    );
    let vf_gpu_sw = format!(
        "format=yuv420p,hwupload_cuda,scale_cuda=w={}:h={}:force_original_aspect_ratio=increase:format=nv12:interp_algo=lanczos,hwdownload,format=yuv420p,crop={}:{}:{x}:{y},format=yuv420p",
        out_w, out_h, out_w, out_h
    );
    let vf_gpu_hw = format!(
        "scale_cuda=w={}:h={}:force_original_aspect_ratio=increase:format=nv12:interp_algo=lanczos,hwdownload,format=yuv420p,crop={}:{}:{x}:{y},format=yuv420p",
        out_w, out_h, out_w, out_h
    );

    (vf_cpu, vf_gpu_sw, vf_gpu_hw)
}

fn append_filter_graph(base: &FilterGraph, extra_chain: &str) -> FilterGraph {
    if extra_chain.trim().is_empty() {
        return base.clone();
    }
    match base {
        FilterGraph::Vf(chain) => FilterGraph::Vf(format!("{chain},{extra_chain}")),
        FilterGraph::Complex { graph, output } => {
            let next = format!("{output}_cap");
            let graph = format!("{graph};[{output}]{extra_chain}[{next}]");
            FilterGraph::Complex {
                graph,
                output: next,
            }
        }
    }
}

fn prepend_hwdownload_filter(base: &FilterGraph) -> FilterGraph {
    match base {
        FilterGraph::Vf(chain) => {
            FilterGraph::Vf(format!("hwdownload,format=yuv420p,{chain}"))
        }
        FilterGraph::Complex { graph, output } => {
            if !graph.contains("[0:v]") {
                return base.clone();
            }
            let replaced = graph.replacen("[0:v]", "[hw_src]", 1);
            let graph = format!("[0:v]hwdownload,format=yuv420p[hw_src];{replaced}");
            FilterGraph::Complex {
                graph,
                output: output.clone(),
            }
        }
    }
}

fn tail_trunc(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return chars.into_iter().collect();
    }
    chars[chars.len() - max..].iter().collect::<String>()
}

#[derive(Clone, Copy, Debug)]
struct ProgressSpec {
    total_secs: Option<f32>,
}

fn parse_ffmpeg_timecode(value: &str) -> Option<f32> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("N/A") {
        return None;
    }
    let mut parts = value.split(':');
    let hours: f32 = parts.next()?.parse().ok()?;
    let minutes: f32 = parts.next()?.parse().ok()?;
    let seconds: f32 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    let total = hours * 3600.0 + minutes * 60.0 + seconds;
    if total.is_finite() && total >= 0.0 {
        Some(total)
    } else {
        None
    }
}

fn format_progress_time(secs: f32) -> String {
    let secs = if secs.is_finite() && secs > 0.0 { secs } else { 0.0 };
    let total = secs.floor() as u64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

fn render_progress_line(current: f32, total: Option<f32>) -> String {
    if let Some(total) = total.filter(|v| v.is_finite() && *v > 0.0) {
        let pct = (current / total * 100.0).clamp(0.0, 100.0);
        return format!(
            "ffmpeg progress: {pct:5.1}% ({}/{})",
            format_progress_time(current),
            format_progress_time(total)
        );
    }
    format!("ffmpeg progress: {}", format_progress_time(current))
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn single_instance_enabled(single_mode: bool) -> bool {
    std::env::var("CLIP_SINGLE_INSTANCE")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(single_mode)
}

fn single_mode_from_env() -> bool {
    std::env::var("CLIP_STREAMS_FILE")
        .ok()
        .map(|v| v.trim().is_empty())
        .unwrap_or(true)
}

#[cfg(windows)]
mod single_instance {
    use anyhow::{anyhow, Result};
    use std::ffi::OsStr;
    use std::iter;
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ALREADY_EXISTS};
    use windows_sys::Win32::System::Threading::CreateMutexW;

    pub struct SingleInstanceGuard {
        handle: isize,
    }

    impl Drop for SingleInstanceGuard {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.handle);
            }
        }
    }

    pub fn acquire(name: &str) -> Result<SingleInstanceGuard> {
        let wide: Vec<u16> = OsStr::new(name)
            .encode_wide()
            .chain(iter::once(0))
            .collect();
        let handle = unsafe { CreateMutexW(std::ptr::null_mut(), 0, wide.as_ptr()) };
        if handle == 0 {
            return Err(anyhow!("failed to create mutex"));
        }
        let err = unsafe { GetLastError() };
        if err == ERROR_ALREADY_EXISTS {
            unsafe {
                CloseHandle(handle);
            }
            return Err(anyhow!("another AutoClip instance is already running"));
        }
        Ok(SingleInstanceGuard { handle })
    }
}

#[cfg(not(windows))]
mod single_instance {
    use anyhow::{anyhow, Result};
    use fs2::FileExt;
    use std::fs::OpenOptions;

    pub struct SingleInstanceGuard {
        _file: std::fs::File,
    }

    pub fn acquire(name: &str) -> Result<SingleInstanceGuard> {
        let mut lock_path = std::env::temp_dir();
        let sanitized: String = name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        lock_path.push(format!("{sanitized}.lock"));
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&lock_path)?;
        if let Err(_) = file.try_lock_exclusive() {
            return Err(anyhow!("another AutoClip instance is already running"));
        }
        Ok(SingleInstanceGuard { _file: file })
    }
}

struct LiveLayoutCache {
    updated_at: Instant,
    hints: ClipLayoutHints,
}

fn live_fast_enabled() -> bool {
    std::env::var("CLIP_LIVE_FAST")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(true)
}

fn live_layout_cache_ttl() -> Duration {
    let secs = std::env::var("CLIP_LIVE_LAYOUT_TTL_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(120);
    Duration::from_secs(secs)
}

fn live_layout_cache_state() -> &'static Mutex<Option<LiveLayoutCache>> {
    static CACHE: OnceLock<Mutex<Option<LiveLayoutCache>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

fn take_live_layout_cache() -> Option<ClipLayoutHints> {
    let ttl = live_layout_cache_ttl();
    if ttl == Duration::ZERO {
        return None;
    }
    let mut guard = live_layout_cache_state().lock().ok()?;
    if let Some(cache) = guard.as_ref() {
        if cache.updated_at.elapsed() <= ttl {
            return Some(cache.hints.clone());
        }
    }
    *guard = None;
    None
}

fn update_live_layout_cache(hints: &ClipLayoutHints) {
    let ttl = live_layout_cache_ttl();
    if ttl == Duration::ZERO {
        return;
    }
    if let Ok(mut guard) = live_layout_cache_state().lock() {
        *guard = Some(LiveLayoutCache {
            updated_at: Instant::now(),
            hints: hints.clone(),
        });
    }
}

fn merge_layout_hints(target: &mut ClipLayoutHints, cached: &ClipLayoutHints) {
    if target.face_box.is_none() {
        target.face_box = cached.face_box;
    }
    if target.face_track.is_none() {
        target.face_track = cached.face_track.clone();
    }
    if target.face_region.is_none() {
        target.face_region = cached.face_region;
    }
    if target.face_frame_spec.is_none() {
        target.face_frame_spec = cached.face_frame_spec;
    }
    if target.face_focus.is_none() {
        target.face_focus = cached.face_focus;
    }
    if target.game_center.is_none() {
        target.game_center = cached.game_center;
    }
    if target.game_region.is_none() {
        target.game_region = cached.game_region;
    }
}

struct EnvSnapshot {
    key: &'static str,
    value: Option<String>,
}

fn set_env_for_scope(key: &'static str, value: &str, snapshot: &mut Vec<EnvSnapshot>) {
    if !snapshot.iter().any(|entry| entry.key == key) {
        snapshot.push(EnvSnapshot {
            key,
            value: std::env::var(key).ok(),
        });
    }
    std::env::set_var(key, value);
}

fn restore_env_snapshot(snapshot: Vec<EnvSnapshot>) {
    for entry in snapshot {
        match entry.value {
            Some(value) => std::env::set_var(entry.key, value),
            None => std::env::remove_var(entry.key),
        }
    }
}

async fn refine_face_hints_low_resource(input: &str, hints: &mut ClipLayoutHints) {
    if !low_resource_enabled() {
        return;
    }
    let baseline_area = if let Some(face_rect) = face_rect_for_gameplay_guess(hints) {
        let area = rect_area(face_rect);
        if area >= 0.02 {
            return;
        }
        area
    } else {
        0.0
    };

    let mut cfg = read_clip_detect_config();
    cfg.enabled = true;
    cfg.sample_count = 1;
    cfg.sample_start_secs = cfg.sample_start_secs.max(1.0);
    cfg.sample_step_secs = cfg.sample_step_secs.max(1.0);
    cfg.scan_full_clip = false;
    cfg.track_face = false;
    cfg.face_track_step_secs = None;
    cfg.face_budget_override = Some(Duration::from_secs(2));
    cfg.analysis_budget = Some(Duration::from_secs(2));

    let mut snapshot = Vec::new();
    set_env_for_scope("CLIP_REGION_DETECT", "0", &mut snapshot);
    set_env_for_scope("CLIP_GAMEPLAY_DETECT", "0", &mut snapshot);
    set_env_for_scope("CLIP_GAMEPLAY", "0", &mut snapshot);

    let refined = detect_layout_hints(input, &cfg).await;
    restore_env_snapshot(snapshot);

    let Ok(refined) = refined else {
        return;
    };
    let Some(refined_box) = refined.face_box else {
        return;
    };
    let refined_area = rect_area(refined_box);
    if refined_area <= baseline_area {
        return;
    }
    hints.face_box = Some(refined_box);
    if hints.face_frame_spec.is_none() {
        hints.face_frame_spec = refined.face_frame_spec;
    }
    if hints.face_region.is_none() {
        hints.face_region = refined.face_region;
    }
    eprintln!(
        "clip layout: low-resource refined face box x={:.3} y={:.3} w={:.3} h={:.3}",
        refined_box.x, refined_box.y, refined_box.w, refined_box.h
    );
}

fn min_duration(current: Option<Duration>, max: Duration) -> Duration {
    match current {
        Some(val) if val < max => val,
        _ => max,
    }
}

fn gameplay_enabled() -> bool {
    std::env::var("CLIP_GAMEPLAY")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(true)
}

fn audio_norm_enabled() -> bool {
    std::env::var("CLIP_AUDIO_NORM")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(true)
}

#[derive(Clone, Debug)]
struct ClipMetadata {
    title: String,
    description: String,
}

#[derive(Clone, Debug)]
struct LlmConfig {
    endpoint: String,
    model: String,
    temperature: f32,
    timeout: Duration,
    max_transcript_chars: usize,
    title_max_chars: usize,
    description_max_chars: usize,
    gpu_layers: Option<i32>,
    rename_files: bool,
    debug: bool,
}

static LLM_BACKOFF_UNTIL_MS: AtomicU64 = AtomicU64::new(0);
static LLM_DISABLED_OFFLINE: AtomicBool = AtomicBool::new(false);
static OPEN_CAPTIONS_DISABLED: AtomicBool = AtomicBool::new(false);
static CLOSED_CAPTIONS_DISABLED: AtomicBool = AtomicBool::new(false);

fn llm_backoff_secs() -> u64 {
    std::env::var("CLIP_LLM_BACKOFF_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(120)
}

fn llm_backoff_active() -> bool {
    let until = LLM_BACKOFF_UNTIL_MS.load(Ordering::Relaxed);
    until > now_unix_ms()
}

fn llm_offline_disabled() -> bool {
    if LLM_DISABLED_OFFLINE.load(Ordering::Relaxed) && !llm_backoff_active() {
        LLM_DISABLED_OFFLINE.store(false, Ordering::Relaxed);
        return false;
    }
    LLM_DISABLED_OFFLINE.load(Ordering::Relaxed)
}

fn note_llm_failure() {
    let secs = llm_backoff_secs();
    if secs == 0 {
        return;
    }
    let until = now_unix_ms().saturating_add(secs.saturating_mul(1000));
    LLM_BACKOFF_UNTIL_MS.store(until, Ordering::Relaxed);
}

fn disable_llm_for_run(reason: &str) {
    if !LLM_DISABLED_OFFLINE.swap(true, Ordering::Relaxed) {
        eprintln!("llm: disabling for this run ({reason})");
    }
}

fn llm_enabled() -> bool {
    std::env::var("CLIP_LLM_ENABLE")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn read_llm_config() -> Option<LlmConfig> {
    if !llm_enabled() {
        return None;
    }
    if llm_offline_disabled() {
        return None;
    }
    if llm_backoff_active() {
        return None;
    }
    let endpoint = std::env::var("CLIP_LLM_ENDPOINT")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "http://localhost:1234/v1/chat/completions".to_string());
    let model = std::env::var("CLIP_LLM_MODEL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| {
            "gemma-2-2b-it"
                .to_string()
        });
    let temperature = std::env::var("CLIP_LLM_TEMPERATURE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.2)
        .clamp(0.0, 1.0);
    let timeout_secs = std::env::var("CLIP_LLM_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(20.0)
        .max(1.0);
    let max_transcript_chars = std::env::var("CLIP_LLM_MAX_TRANSCRIPT_CHARS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4000)
        .max(200);
    let title_max_chars = std::env::var("CLIP_LLM_TITLE_MAX_CHARS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(80)
        .max(20);
    let description_max_chars = std::env::var("CLIP_LLM_DESCRIPTION_MAX_CHARS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(280)
        .max(60);
    let gpu_layers = std::env::var("CLIP_LLM_GPU_LAYERS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok());
    let rename_files = std::env::var("CLIP_LLM_FILE_RENAME")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(true);
    let debug = std::env::var("CLIP_LLM_DEBUG")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false);

    Some(LlmConfig {
        endpoint,
        model,
        temperature,
        timeout: Duration::from_secs_f32(timeout_secs),
        max_transcript_chars,
        title_max_chars,
        description_max_chars,
        gpu_layers,
        rename_files,
        debug,
    })
}

async fn generate_clip_metadata(
    transcript: &str,
    cfg: &LlmConfig,
) -> Result<Option<ClipMetadata>> {
    if transcript.trim().is_empty() {
        return Ok(None);
    }
    let transcript = trim_transcript(transcript, cfg.max_transcript_chars);
    let system = format!(
        "You are a clip metadata assistant. Return ONLY valid JSON with keys \
\"title\" and \"description\". Title <= {} chars. Description <= {} chars. \
Description must be a concise 1-2 sentence summary of the clip highlight, \
based only on the transcript.",
        cfg.title_max_chars, cfg.description_max_chars
    );
    let user = format!(
        "Transcript:\n{}\n\nReturn JSON only.",
        transcript
    );

    let mut payload = json!({
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": 256
    });
    if let Some(layers) = cfg.gpu_layers {
        if let Value::Object(obj) = &mut payload {
            obj.insert("n_gpu_layers".to_string(), Value::from(layers));
        }
    }

    let client = Client::builder()
        .timeout(cfg.timeout)
        .build()
        .context("building LLM client")?;
    let resp = match client
        .post(&cfg.endpoint)
        .json(&payload)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(err) => {
            if llm_error_indicates_offline(&err.to_string()) {
                disable_llm_for_run("endpoint unreachable");
            }
            note_llm_failure();
            return Err(err).context("LLM request failed");
        }
    };
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        note_llm_failure();
        anyhow::bail!("LLM error {}: {}", status, truncate_str(&body, 400));
    }
    let content = match parse_llm_content(&body) {
        Some(val) => val,
        None => {
            if cfg.debug {
                eprintln!("llm: failed to parse response: {}", truncate_str(&body, 400));
            }
            return Ok(None);
        }
    };
    let meta = match parse_llm_json(&content, cfg) {
        Some(meta) => meta,
        None => {
            if cfg.debug {
                eprintln!("llm: failed to parse JSON content: {}", truncate_str(&content, 200));
            }
            return Ok(None);
        }
    };
    Ok(Some(meta))
}

fn parse_llm_content(body: &str) -> Option<String> {
    let json: Value = serde_json::from_str(body).ok()?;
    let content = json
        .get("choices")
        .and_then(|v| v.get(0))
        .and_then(|v| v.get("message"))
        .and_then(|v| v.get("content"))
        .and_then(|v| v.as_str())?;
    Some(content.to_string())
}

fn parse_llm_json(content: &str, cfg: &LlmConfig) -> Option<ClipMetadata> {
    let trimmed = content.trim();
    let json_str = if trimmed.starts_with('{') {
        trimmed.to_string()
    } else {
        let start = trimmed.find('{')?;
        let end = trimmed.rfind('}')?;
        trimmed[start..=end].to_string()
    };
    let json: Value = serde_json::from_str(&json_str).ok()?;
    let mut title = json.get("title")?.as_str()?.to_string();
    let mut description = json.get("description")?.as_str()?.to_string();
    title = normalize_title_whitespace(&title);
    description = normalize_title_whitespace(&description);
    if title.is_empty() || description.is_empty() {
        return None;
    }
    if title.len() > cfg.title_max_chars {
        title = truncate_str(&title, cfg.title_max_chars);
    }
    if description.len() > cfg.description_max_chars {
        description = truncate_str(&description, cfg.description_max_chars);
    }
    Some(ClipMetadata { title, description })
}

fn rename_output_with_title(out_path: &Path, title: &str) -> Result<Option<PathBuf>> {
    let slug = sanitize_title_for_filename(title);
    if slug.is_empty() {
        return Ok(None);
    }
    let parent = out_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = out_path
        .file_stem()
        .and_then(|v| v.to_str())
        .unwrap_or("clip");
    let ext = out_path.extension().and_then(|v| v.to_str()).unwrap_or("mp4");
    let mut suffix = slug;
    if suffix.len() > 60 {
        suffix = truncate_str(&suffix, 60);
    }
    let new_name = format!("{stem}__{suffix}.{ext}");
    let new_path = parent.join(new_name);
    if new_path == out_path || new_path.exists() {
        return Ok(None);
    }
    fs::rename(out_path, &new_path)
        .with_context(|| format!("renaming {} -> {}", out_path.display(), new_path.display()))?;
    Ok(Some(new_path))
}

fn ffmpeg_encode_timeout() -> Option<Duration> {
    if ffmpeg_no_limits_enabled() {
        return None;
    }
    std::env::var("FFMPEG_ENCODE_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(Duration::from_secs_f32)
}

fn ffmpeg_min_free_vram_mb() -> u64 {
    std::env::var("FFMPEG_MIN_FREE_VRAM_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(512)
}

fn parse_device_index(value: &str) -> Option<u32> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<u32>().ok()
}

fn ffmpeg_hwaccel_device_index() -> Option<u32> {
    std::env::var("FFMPEG_HWACCEL_DEVICE")
        .ok()
        .and_then(|v| parse_device_index(&v))
}

fn llm_error_indicates_offline(message: &str) -> bool {
    let msg = message.to_ascii_lowercase();
    msg.contains("connection refused")
        || msg.contains("actively refused")
        || msg.contains("os error 10061")
        || msg.contains("error trying to connect")
}

async fn check_llm_health() {
    if !llm_enabled() || llm_offline_disabled() {
        return;
    }
    let endpoint = std::env::var("CLIP_LLM_ENDPOINT")
        .unwrap_or_else(|_| "".to_string())
        .trim()
        .to_string();
    if endpoint.is_empty() {
        return;
    }
    let timeout = Duration::from_secs(2);
    let client = match Client::builder().timeout(timeout).build() {
        Ok(client) => client,
        Err(err) => {
            eprintln!("llm: failed to build health-check client: {err}");
            return;
        }
    };
    match client.get(&endpoint).send().await {
        Ok(resp) => {
            eprintln!("llm: endpoint reachable (status {})", resp.status());
        }
        Err(err) => {
            let msg = err.to_string();
            if llm_error_indicates_offline(&msg) {
                disable_llm_for_run("endpoint unreachable at startup");
            }
            eprintln!("llm: endpoint check failed: {msg}");
        }
    }
}

fn open_captions_allowed() -> bool {
    !OPEN_CAPTIONS_DISABLED.load(Ordering::Relaxed)
}

fn closed_captions_allowed() -> bool {
    !CLOSED_CAPTIONS_DISABLED.load(Ordering::Relaxed)
}

fn disable_closed_captions_for_run(reason: &str) {
    if !CLOSED_CAPTIONS_DISABLED.swap(true, Ordering::Relaxed) {
        eprintln!("captions: disabling closed captions for this run ({reason})");
    }
}

fn disable_open_captions_for_run(reason: &str) {
    if !OPEN_CAPTIONS_DISABLED.swap(true, Ordering::Relaxed) {
        eprintln!("captions: disabling open captions for this run ({reason})");
    }
    disable_closed_captions_for_run(reason);
}

fn captions_persist_failures() -> bool {
    std::env::var("CLIP_CAPTIONS_PERSIST_FAILURE")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn reset_caption_disable_for_segment() {
    if !captions_persist_failures() {
        if captions_enabled() {
            OPEN_CAPTIONS_DISABLED.store(false, Ordering::Relaxed);
        }
        if closed_captions_enabled() {
            CLOSED_CAPTIONS_DISABLED.store(false, Ordering::Relaxed);
        }
    }
}

struct TempFilterScript {
    path: PathBuf,
}

impl TempFilterScript {
    fn new(kind: &str, graph: &str) -> Result<Self> {
        let stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_millis();
        let filename = format!("autoclip_ffmpeg_{kind}_{stamp}.txt");
        let path = std::env::temp_dir().join(filename);
        fs::write(&path, graph).context("writing ffmpeg filter script")?;
        Ok(Self { path })
    }
}

impl Drop for TempFilterScript {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn should_use_filter_script(graph: &str) -> bool {
    if !cfg!(windows) {
        return false;
    }
    if graph.contains("drawtext=") || graph.contains("subtitles=") {
        return true;
    }
    graph.len() > 8000
}

#[cfg(windows)]
#[derive(Clone, Debug)]
struct FontconfigPaths {
    config_file: PathBuf,
    config_dir: PathBuf,
}

#[cfg(windows)]
fn fontconfig_paths_for_ffmpeg() -> Option<&'static FontconfigPaths> {
    static FONTCONFIG_PATHS: OnceLock<Option<FontconfigPaths>> = OnceLock::new();
    FONTCONFIG_PATHS
        .get_or_init(|| {
            let windows_dir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".to_string());
            let fonts_dir = Path::new(&windows_dir).join("Fonts");
            if !fonts_dir.exists() {
                return None;
            }
            let config_dir = std::env::temp_dir().join("autoclip_fontconfig");
            if fs::create_dir_all(&config_dir).is_err() {
                return None;
            }
            let config_file = config_dir.join("fonts.conf");
            if !config_file.exists() {
                let fonts_dir_str = fonts_dir
                    .to_string_lossy()
                    .replace('\\', "/");
                let config = format!(
                    "<?xml version=\"1.0\"?>\n\
<!DOCTYPE fontconfig SYSTEM \"fonts.dtd\">\n\
<fontconfig>\n  <dir>{}</dir>\n</fontconfig>\n",
                    fonts_dir_str
                );
                if fs::write(&config_file, config).is_err() {
                    return None;
                }
            }
            Some(FontconfigPaths { config_file, config_dir })
        })
        .as_ref()
}

fn apply_fontconfig_env_for_ffmpeg(cmd: &mut Command) {
    if !cfg!(windows) {
        return;
    }
    if std::env::var_os("FONTCONFIG_FILE").is_some()
        || std::env::var_os("FONTCONFIG_PATH").is_some()
    {
        return;
    }
    if let Some(paths) = fontconfig_paths_for_ffmpeg() {
        cmd.env("FONTCONFIG_FILE", paths.config_file.as_os_str());
        cmd.env("FONTCONFIG_PATH", paths.config_dir.as_os_str());
    }
}

async fn run_ffmpeg_encode(
    input: &str,
    out_path: &Path,
    filters: &FilterGraph,
    encoder: &str,
    use_hw_encode: bool,
    use_nvenc: bool,
    fast_preset: bool,
    allow_hwaccel: bool,
    use_hw_frames: bool,
    regen_pts: bool,
    force_ts_input: bool,
    fast_seek: bool,
    start_offset_secs: Option<f32>,
    duration_secs: Option<f32>,
    progress: Option<ProgressSpec>,
    metadata: Option<&ClipMetadata>,
    subtitles: Option<&SubtitleSpec>,
) -> Result<std::process::Output> {
    let mut cmd = Command::new("ffmpeg");
    let mut _filter_script: Option<TempFilterScript> = None;
    apply_fontconfig_env_for_ffmpeg(&mut cmd);
    let hwaccel_env = std::env::var("FFMPEG_HWACCEL").unwrap_or_default();
    let hwaccel_label = if hwaccel_env.trim().is_empty() {
        "(unset)".to_string()
    } else {
        hwaccel_env.clone()
    };
    let hw_device = ffmpeg_hwaccel_device_index();
    if allow_hwaccel {
        eprintln!(
            "ffmpeg: encode start (encoder={encoder}, hwaccel={hwaccel_label}, device={hw_device:?}, subtitles={})",
            subtitles.is_some()
        );
    } else {
        eprintln!(
            "ffmpeg: encode start (encoder={encoder}, hwaccel=disabled, subtitles={})",
            subtitles.is_some()
        );
    }
    gpu::log_nvidia_snapshot_throttled("ffmpeg", 30);
    cmd.arg("-y");
    // Elevate logging when using NVENC to capture filter negotiation issues.
    if use_nvenc {
        cmd.arg("-loglevel").arg("verbose");
    } else {
        cmd.arg("-loglevel").arg("warning");
    }
    if regen_pts {
        cmd.arg("-fflags").arg("+genpts");
    }
    if allow_hwaccel {
        for arg in ffmpeg_hwaccel_flags_for_encode(use_hw_encode) {
            cmd.arg(arg);
        }
    }
    if use_nvenc && allow_hwaccel && use_hw_frames {
        cmd.arg("-hwaccel_output_format").arg("cuda");
    }
    if force_ts_input {
        cmd.arg("-f").arg("mpegts");
    }
    if progress.is_some() {
        cmd.arg("-progress").arg("pipe:1");
        cmd.arg("-nostats");
    }
    if let Some(ss) = start_offset_secs {
        if fast_seek && ss > 0.0 {
            cmd.arg("-ss").arg(format!("{ss:.3}"));
        }
    }
    cmd.arg("-i").arg(input);
    if let Some(subs) = subtitles {
        cmd.arg("-i").arg(subs.path.as_os_str());
    }
    if let Some(ss) = start_offset_secs {
        if !fast_seek && ss > 0.0 {
            cmd.arg("-ss").arg(format!("{ss:.3}"));
        }
    }
    if let Some(d) = duration_secs {
        cmd.arg("-t").arg(format!("{d:.3}"));
    }
    match filters {
        FilterGraph::Vf(vf) => {
            cmd.arg("-map").arg("0:v:0");
            if should_use_filter_script(vf) {
                let script = TempFilterScript::new("vf", vf)?;
                cmd.arg("-filter_script:v").arg(script.path.as_os_str());
                _filter_script = Some(script);
            } else {
                cmd.arg("-vf").arg(vf);
            }
        }
        FilterGraph::Complex { graph, output } => {
            if should_use_filter_script(graph) {
                let script = TempFilterScript::new("complex", graph)?;
                cmd.arg("-filter_complex_script").arg(script.path.as_os_str());
                _filter_script = Some(script);
            } else {
                cmd.arg("-filter_complex").arg(graph);
            }
            cmd.arg("-map").arg(format!("[{output}]"));
        }
    }
    cmd.arg("-map")
        .arg("0:a:0?")
        .arg("-fps_mode")
        .arg("vfr")
        .arg("-c:v")
        .arg(encoder);
    if subtitles.is_some() {
        cmd.arg("-map").arg("1:0");
        cmd.arg("-c:s").arg(subtitle_codec_for_output(out_path));
        cmd.arg("-metadata:s:s:0").arg("language=eng");
        cmd.arg("-disposition:s:0").arg("default");
    }

    if use_nvenc {
        let preset = if fast_preset { "p1" } else { "p4" };
        cmd.arg("-preset").arg(preset);
        cmd.arg("-tune").arg("hq");
        cmd.arg("-b:v").arg("0");
        cmd.arg("-cq").arg("23");
    } else {
        let preset = if fast_preset { "ultrafast" } else { "veryfast" };
        cmd.arg("-preset").arg(preset);
        cmd.arg("-crf").arg("23");
    }

    let audio_norm = audio_norm_enabled();
    let mut audio_filters: Vec<String> = Vec::new();
    if regen_pts {
        if let Some(d) = duration_secs {
            let trim_end = start_offset_secs.unwrap_or(0.0).max(0.0) + d;
            // Keep full audio duration when seeking so -ss doesn't shorten the tail.
            audio_filters.push(format!("atrim=end={trim_end:.3},asetpts=N/SR/TB"));
        }
    }
    if audio_norm {
        audio_filters.push("loudnorm=I=-16:LRA=11:TP=-1.5".to_string());
    }
    if !audio_filters.is_empty() {
        cmd.arg("-af").arg(audio_filters.join(","));
    }
    if regen_pts || audio_norm {
        cmd.arg("-c:a").arg("aac").arg("-b:a").arg("160k");
    } else {
        cmd.arg("-c:a").arg("copy");
    }

    if duration_secs.is_none() {
        cmd.arg("-shortest");
    }
    if let Some(meta) = metadata {
        cmd.arg("-metadata").arg(format!("title={}", meta.title));
        cmd.arg("-metadata")
            .arg(format!("description={}", meta.description));
        cmd.arg("-metadata")
            .arg(format!("comment={}", meta.description));
    }
    cmd.arg(out_path.as_os_str());
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn ffmpeg")?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("missing ffmpeg stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("missing ffmpeg stderr"))?;

    const MAX_FFMPEG_STDERR_BYTES: usize = 1_000_000;
    const MAX_FFMPEG_STDOUT_BYTES: usize = 256_000;

    let stderr_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        let mut truncated = false;
        loop {
            let n = reader.read(&mut tmp).await.context("reading ffmpeg stderr")?;
            if n == 0 {
                break;
            }
            if buf.len() < MAX_FFMPEG_STDERR_BYTES {
                let remaining = MAX_FFMPEG_STDERR_BYTES - buf.len();
                let take = remaining.min(n);
                buf.extend_from_slice(&tmp[..take]);
                if take < n {
                    truncated = true;
                }
            } else {
                truncated = true;
            }
        }
        if truncated {
            buf.extend_from_slice(b"\n[ffmpeg stderr truncated]\n");
        }
        Ok::<Vec<u8>, anyhow::Error>(buf)
    });

    let mut stdout_lines = BufReader::new(stdout).lines();
    let mut stdout_buf = Vec::new();
    let total_secs = progress
        .as_ref()
        .and_then(|spec| spec.total_secs)
        .filter(|v| v.is_finite() && *v > 0.0);
    let mut last_time: Option<f32> = None;
    let mut last_print = Instant::now();
    let mut last_pct = -1.0f32;
    let mut last_reported_time: Option<f32> = None;
    let mut stall_timed_out = false;
    let mut printed = false;
    let timeout = ffmpeg_encode_timeout();
    let mut timed_out = false;
    let stall_timeout = if progress.is_some() {
        ffmpeg_progress_stall_timeout()
    } else {
        None
    };

    let read_stdout = async {
        loop {
            let next_line = if let Some(limit) = stall_timeout {
                match tokio::time::timeout(limit, stdout_lines.next_line()).await {
                    Ok(res) => res.context("reading ffmpeg progress")?,
                    Err(_) => {
                        stall_timed_out = true;
                        break;
                    }
                }
            } else {
                stdout_lines
                    .next_line()
                    .await
                    .context("reading ffmpeg progress")?
            };
            let Some(line) = next_line else { break };
            if stdout_buf.len() < MAX_FFMPEG_STDOUT_BYTES {
                let remaining = MAX_FFMPEG_STDOUT_BYTES - stdout_buf.len();
                let line_bytes = line.as_bytes();
                let take = remaining.min(line_bytes.len());
                stdout_buf.extend_from_slice(&line_bytes[..take]);
                if take < line_bytes.len() {
                    stdout_buf.extend_from_slice(b"\n[ffmpeg stdout truncated]\n");
                } else {
                    stdout_buf.push(b'\n');
                }
            }

            if progress.is_some() {
                if let Some(value) = line.strip_prefix("out_time=") {
                    last_time = parse_ffmpeg_timecode(value);
                } else if let Some(value) = line.strip_prefix("out_time_us=") {
                    if let Ok(us) = value.trim().parse::<f32>() {
                        last_time = Some(us / 1_000_000.0);
                    }
                } else if let Some(value) = line.strip_prefix("out_time_ms=") {
                    if let Ok(raw) = value.trim().parse::<f32>() {
                        last_time = Some(raw / 1_000_000.0);
                    }
                } else if line.trim() == "progress=end" {
                    if let Some(total) = total_secs {
                        last_time = Some(total);
                    }
                }
            }

            if let Some(current) = last_time {
                let should_print = if let Some(total) = total_secs {
                    let pct = (current / total * 100.0).clamp(0.0, 100.0);
                    if pct - last_pct >= 0.5 {
                        last_pct = pct;
                        true
                    } else {
                        false
                    }
                } else {
                    let advance = last_reported_time
                        .map(|prev| (current - prev).abs() >= 0.5)
                        .unwrap_or(true);
                    if advance {
                        last_reported_time = Some(current);
                    }
                    advance
                } || last_print.elapsed() > Duration::from_millis(500);
                if should_print {
                    printed = true;
                    last_print = Instant::now();
                    eprint!("\r{}", render_progress_line(current, total_secs));
                    let _ = io::stderr().flush();
                }
            }
        }
        Ok::<(), anyhow::Error>(())
    };

    if let Some(limit) = timeout {
        match tokio::time::timeout(limit, read_stdout).await {
            Ok(res) => {
                res?;
            }
            Err(_) => {
                timed_out = true;
            }
        }
    } else {
        read_stdout.await?;
    }

    if stall_timed_out {
        if let Some(limit) = stall_timeout {
            eprintln!(
                "ffmpeg: progress stalled for {:.1}s; terminating",
                limit.as_secs_f32()
            );
        } else {
            eprintln!("ffmpeg: progress stalled; terminating");
        }
        let _ = child.kill().await;
    }

    if timed_out {
        if let Some(limit) = timeout {
            eprintln!(
                "ffmpeg: encode timed out after {:.1}s; terminating",
                limit.as_secs_f32()
            );
        } else {
            eprintln!("ffmpeg: encode timed out; terminating");
        }
        let _ = child.kill().await;
    }

    let status = child.wait().await.context("waiting for ffmpeg")?;
    let mut stderr_buf = stderr_task
        .await
        .context("joining ffmpeg stderr task")??;
    if timed_out {
        let msg = if let Some(limit) = timeout {
            format!("\nffmpeg encode timed out after {:.1}s\n", limit.as_secs_f32())
        } else {
            "\nffmpeg encode timed out\n".to_string()
        };
        stderr_buf.extend_from_slice(msg.as_bytes());
    }
    if stall_timed_out {
        let msg = if let Some(limit) = stall_timeout {
            format!(
                "\nffmpeg progress stalled for {:.1}s; terminating\n",
                limit.as_secs_f32()
            )
        } else {
            "\nffmpeg progress stalled; terminating\n".to_string()
        };
        stderr_buf.extend_from_slice(msg.as_bytes());
    }
    if printed {
        eprintln!();
    }
    Ok(std::process::Output {
        status,
        stdout: stdout_buf,
        stderr: stderr_buf,
    })
}

async fn run_ffmpeg_internal(spec: FfmpegRenderSpec) -> Result<()> {
    let _span = profile_span("ffmpeg: encode");
    let input = spec.input.as_str();
    let out_path = &spec.out_path;
    let out_w = spec.out_w;
    let out_h = spec.out_h;
    let duration_secs = spec.duration_secs;
    let start_offset_secs = spec.start_offset_secs;
    let regen_pts = spec.regen_pts;
    let force_ts_input = spec.force_ts_input;
    let mut progress = spec.progress;
    let live_fast = spec.live_fast;
    let fast_preset = spec.fast_preset;
    let fast_seek = spec.fast_seek;
    if progress.is_none() && ffmpeg_progress_enabled() {
        progress = Some(ProgressSpec {
            total_secs: duration_secs,
        });
    }
    if let Some(mut current) = progress {
        if current.total_secs.is_none() {
            current.total_secs = duration_secs;
        }
        progress = Some(current);
    }
    let (mut video_encoder, mut is_nvenc, mut is_hw, _encoder_forced) = ffmpeg_video_encoder();
    if is_hw || is_nvenc {
        let min_free = ffmpeg_min_free_vram_mb();
        let device = ffmpeg_hwaccel_device_index();
        if !gpu::gpu_vram_allows(min_free, device, "ffmpeg") {
            eprintln!("ffmpeg: disabling GPU encode due to low VRAM");
            video_encoder = "libx264".to_string();
            is_nvenc = false;
            is_hw = false;
        }
    }
    let scale_cuda_available = ffmpeg_scale_cuda_available();
    if is_hw && !scale_cuda_available {
        eprintln!("ffmpeg: scale_cuda filter missing; forcing CPU encode");
        video_encoder = "libx264".to_string();
        is_nvenc = false;
        is_hw = false;
    }
    let layout = read_clip_layout_config();
    let mut layout_hints = read_clip_layout_hints();
    if live_fast {
        if let Some(cached) = take_live_layout_cache() {
            merge_layout_hints(&mut layout_hints, &cached);
            eprintln!("live render: using cached layout hints");
        }
    }
    if !live_fast {
        refine_face_hints_low_resource(input, &mut layout_hints).await;
    }
    let mut layout_is_stacked = matches!(layout.mode, ClipLayoutMode::Stacked);
    let mut face_only = false;
    let mut fullscreen_fill = false;
    let mut fullscreen_track = None;
    let mut fullscreen_center_override = None;
    let face_focus = layout_hints.face_focus;
    if layout_is_stacked {
        let mut detect_cfg = read_clip_detect_config();
        let user_game_center = layout_hints.game_center;
        let user_game_region = layout_hints.game_region;
        let face_track_override = std::env::var("CLIP_FACE_TRACK")
            .ok()
            .and_then(|v| parse_bool(&v));
        if live_fast && user_game_center.is_none() {
            layout_hints.game_center = Some(NormalizedPoint { x: 0.5, y: 0.5 });
        }
        if live_fast {
            detect_cfg.sample_count = detect_cfg.sample_count.min(2);
            if face_track_override != Some(true) {
                detect_cfg.scan_full_clip = false;
                detect_cfg.track_face = false;
                detect_cfg.face_track_step_secs = None;
                eprintln!(
                    "live render: face tracking disabled (set CLIP_FACE_TRACK=1 to enable)"
                );
            }
            detect_cfg.face_budget_override =
                Some(min_duration(detect_cfg.face_budget_override, Duration::from_secs(6)));
            detect_cfg.analysis_budget =
                Some(min_duration(detect_cfg.analysis_budget, Duration::from_secs(8)));
        }
        let mut need_face = layout.face_crop.is_none() && layout_hints.face_box.is_none();
        if detect_cfg.track_face && layout_hints.face_track.is_none() {
            need_face = true;
        }
        let need_gameplay = user_game_center.is_none() && !live_fast;
        if detect_cfg.enabled && (need_face || need_gameplay) {
            match detect_layout_hints(input, &detect_cfg).await {
                Ok(detected) => {
                    if live_fast {
                        update_live_layout_cache(&detected);
                    }
                    if need_face {
                        if let Some(face) = detected.face_box {
                            layout_hints.face_box = Some(face);
                            eprintln!(
                                "clip detect: face box x={:.3} y={:.3} w={:.3} h={:.3}",
                                face.x, face.y, face.w, face.h
                            );
                        }
                        if layout_hints.face_track.is_none() {
                            layout_hints.face_track = detected.face_track;
                        }
                        if layout_hints.face_region.is_none() {
                            layout_hints.face_region = detected.face_region;
                        }
                        if layout_hints.face_frame_spec.is_none() {
                            layout_hints.face_frame_spec = detected.face_frame_spec;
                        }
                    }
                    if need_gameplay && user_game_center.is_none() {
                        layout_hints.game_center = detected.game_center;
                    } else {
                        layout_hints.game_center = user_game_center;
                    }
                    if user_game_region.is_none() {
                        layout_hints.game_region = detected.game_region;
                    } else {
                        layout_hints.game_region = user_game_region;
                    }
                }
                Err(err) => {
                    eprintln!("clip detect: detection failed: {err:#}");
                }
            }
        }
        let decision = decide_stacked_layout(&layout, &layout_hints, face_focus);
        layout_is_stacked = decision.layout_is_stacked;
        face_only = decision.face_only;
        fullscreen_fill = decision.fullscreen_fill;
        fullscreen_track = decision.fullscreen_track;
        fullscreen_center_override = decision.fullscreen_center_override;
    }
    let stacked_dims = if layout_is_stacked {
        Some(resolve_stacked_layout_dims(out_w, out_h, &layout, &layout_hints))
    } else {
        None
    };
    let mut caption_cfg = if live_fast {
        None
    } else {
        read_caption_config(out_h)
    };
    if caption_cfg.is_some() && !open_captions_allowed() {
        caption_cfg = None;
    }
    let llm_cfg = if live_fast { None } else { read_llm_config() };
    let mut closed_captions = if live_fast { false } else { closed_captions_enabled() };
    if closed_captions && !closed_captions_allowed() {
        closed_captions = false;
    }
    if live_fast {
        eprintln!("live render: skipping captions/LLM for faster clip output");
    }
    let mut transcript: Option<TranscriptPayload> = None;
    if caption_cfg.is_some() || llm_cfg.is_some() || closed_captions {
        if duration_secs.is_none() {
            eprintln!("captions: skipping (clip duration unknown)");
        } else {
            let input_owned = input.to_string();
            let start_offset = start_offset_secs;
            let duration = duration_secs;
            let force_ts = force_ts_input;
            {
                let _span = profile_span("captions: transcribe audio");
                transcript = match tokio::task::spawn_blocking(move || {
                    transcribe_clip_audio(&input_owned, start_offset, duration, force_ts)
                })
                .await
                {
                    Ok(Ok(payload)) => Some(payload),
                    Ok(Err(err)) => {
                        eprintln!("captions: transcription failed: {err:#}");
                        None
                    }
                    Err(err) => {
                        eprintln!("captions: transcription task failed: {err}");
                        None
                    }
                };
            }
        }
    }
    reset_caption_disable_for_segment();
    let mut caption_chain: Option<String> = None;
    let mut _open_caption_temp: Option<TempSubtitle> = None;
    if let (Some(cfg), Some(payload)) = (caption_cfg.as_ref(), transcript.as_ref()) {
        let adjusted_cfg = adjust_caption_font_size(cfg, payload, out_w);
        let y = caption_y_for_layout(&adjusted_cfg, layout_is_stacked, stacked_dims, out_h);
        let mut try_subtitles = |payload: &TranscriptPayload| {
            if let Some(ass) = build_ass_from_payload_with_limits(
                payload,
                duration_secs,
                &adjusted_cfg,
                y,
                out_w,
                out_h,
            ) {
                match write_temp_ass(&ass) {
                    Ok(path) => {
                        _open_caption_temp = Some(TempSubtitle::new(path.clone()));
                        let chain =
                            build_caption_subtitles_filter(&path, &adjusted_cfg, y, out_w, out_h);
                        if !chain.is_empty() {
                            caption_chain = Some(chain);
                        }
                        if adjusted_cfg.debug {
                            let cue_count = ass.lines().filter(|line| line.starts_with("Dialogue:")).count();
                            eprintln!(
                                "open captions: generated {cue_count} cues via subtitles filter"
                            );
                        }
                    }
                    Err(err) => {
                        eprintln!("open captions: failed to write ass: {err:#}");
                    }
                }
            }
        };

        match caption_render_mode() {
            CaptionRender::Drawtext => {
                match build_drawtext_caption_filter_with_limit(
                    payload,
                    duration_secs,
                    &adjusted_cfg,
                    y,
                    out_w,
                    out_h,
                ) {
                    DrawtextBuildResult::Chain { chain, cue_count } => {
                        if !chain.trim().is_empty() {
                            caption_chain = Some(chain);
                        }
                        if adjusted_cfg.debug {
                            eprintln!(
                                "open captions: generated {cue_count} cues via drawtext"
                            );
                        }
                    }
                    DrawtextBuildResult::TooManyCues { cue_count, max_cues } => {
                        eprintln!(
                            "open captions: drawtext cue count {cue_count} exceeds limit {max_cues}; falling back to subtitles renderer"
                        );
                        try_subtitles(payload);
                    }
                }
            }
            CaptionRender::Subtitles => {
                try_subtitles(payload);
            }
        }
    }
    let mut subtitle_spec: Option<SubtitleSpec> = None;
    let mut _subtitle_temp: Option<TempSubtitle> = None;
    if closed_captions {
        if let Some(payload) = transcript.as_ref() {
            if let Some(srt) = build_srt_from_payload_with_limits(
                payload,
                duration_secs,
                caption_cfg
                    .as_ref()
                    .map(|c| c.min_word_secs)
                    .unwrap_or(0.12),
                caption_cfg
                    .as_ref()
                    .map(|c| c.max_words)
                    .unwrap_or(usize::MAX),
            ) {
                match write_temp_srt(&srt) {
                    Ok(path) => {
                        subtitle_spec = Some(SubtitleSpec { path: path.clone() });
                        _subtitle_temp = Some(TempSubtitle::new(path));
                    }
                    Err(err) => {
                        eprintln!("closed captions: failed to write srt: {err:#}");
                    }
                }
            } else {
                eprintln!("closed captions: no transcript to embed");
            }
        } else {
            eprintln!("closed captions: transcription unavailable");
        }
    }
    let has_open_captions = caption_chain.is_some();
    let has_closed_captions = subtitle_spec.is_some();
    let has_captions = has_open_captions || has_closed_captions;
    let subtitle_encode_spec: Option<SubtitleSpec> = None;
    let mut clip_metadata: Option<ClipMetadata> = None;
    if let (Some(cfg), Some(payload)) = (llm_cfg.as_ref(), transcript.as_ref()) {
        let mut text = payload.text.trim().to_string();
        if text.is_empty() {
            text = payload
                .words
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_string();
        }
        if !text.is_empty() {
            let _span = profile_span("llm: generate metadata");
            match generate_clip_metadata(&text, cfg).await {
                Ok(Some(meta)) => {
                    clip_metadata = Some(meta);
                }
                Ok(None) => {
                    let title = fallback_title_from_transcript(&text, cfg.title_max_chars);
                    let description = truncate_str(&text, cfg.description_max_chars);
                    if !title.is_empty() && !description.is_empty() {
                        clip_metadata = Some(ClipMetadata { title, description });
                    }
                }
                Err(err) => {
                    eprintln!("llm: metadata generation failed: {err:#}");
                    let title = fallback_title_from_transcript(&text, cfg.title_max_chars);
                    let description = truncate_str(&text, cfg.description_max_chars);
                    if !title.is_empty() && !description.is_empty() {
                        clip_metadata = Some(ClipMetadata { title, description });
                    }
                }
            }
        }
    }
    let hwaccel = std::env::var("FFMPEG_HWACCEL").unwrap_or_default();
    let hw_decode_cuda = is_nvenc && hwaccel.eq_ignore_ascii_case("cuda");
    let mut fullscreen_center = layout_hints
        .game_center
        .unwrap_or(NormalizedPoint { x: 0.5, y: 0.5 });
    if let Some(center) = fullscreen_center_override {
        fullscreen_center = center;
    }
    let (vf_cpu, vf_gpu_sw, vf_gpu_hw) = build_ffmpeg_filters(out_w, out_h);
    let (_vf_fill_cpu, vf_fill_gpu_sw, vf_fill_gpu_hw) =
        build_ffmpeg_fill_filters(out_w, out_h, fullscreen_center);
    let vf_gpu = if hw_decode_cuda { vf_gpu_hw.clone() } else { vf_gpu_sw.clone() };
    let vf_fill_gpu = if hw_decode_cuda {
        vf_fill_gpu_hw.clone()
    } else {
        vf_fill_gpu_sw.clone()
    };
    let tracked_fullscreen = fullscreen_fill && fullscreen_track.is_some();
    let mut filter_cpu = if layout_is_stacked {
        build_stacked_filter_graph(out_w, out_h, &layout, &layout_hints)
    } else if face_only {
        build_face_only_filter_graph(out_w, out_h, &layout, &layout_hints)
    } else if fullscreen_fill {
        if let Some(track) = fullscreen_track.as_ref() {
            build_tracked_full_frame_fill_filter_graph(out_w, out_h, track)
                .unwrap_or_else(|| build_full_frame_fill_filter_graph(out_w, out_h, fullscreen_center))
        } else {
            build_full_frame_fill_filter_graph(out_w, out_h, fullscreen_center)
        }
    } else {
        FilterGraph::Vf(vf_cpu)
    };
    let mut filter_gpu = if layout_is_stacked || face_only {
        filter_cpu.clone()
    } else if fullscreen_fill {
        if tracked_fullscreen {
            filter_cpu.clone()
        } else {
            FilterGraph::Vf(vf_fill_gpu)
        }
    } else {
        FilterGraph::Vf(vf_gpu)
    };
    let captions_active = caption_chain.is_some() || subtitle_spec.is_some();
    let filter_cpu_base = filter_cpu.clone();
    if let Some(chain) = caption_chain.as_deref() {
        filter_cpu = append_filter_graph(&filter_cpu, chain);
        filter_gpu = append_filter_graph(&filter_gpu, chain);
    }
    let force_cpu_filters =
        (is_stream_unstable(input) && is_nvenc) || layout_is_stacked || face_only || tracked_fullscreen;
    let mut allow_hwaccel = !is_stream_unstable(input);
    let use_cpu_filters = force_cpu_filters || !is_nvenc;
    if captions_active && use_cpu_filters {
        if allow_hwaccel {
            eprintln!("ffmpeg: captions with CPU filters; disabling hwaccel decode");
        }
        allow_hwaccel = false;
    }
    let use_hw_frames = allow_hwaccel && hw_decode_cuda && !use_cpu_filters;
    let cpu_hwdownload = use_hw_frames;
    if force_cpu_filters {
        if layout_is_stacked {
            if is_hw {
                if allow_hwaccel && hw_decode_cuda {
                    if use_hw_frames {
                        eprintln!("ffmpeg: stacked layout uses CPU filters; keeping hwaccel decode");
                    } else {
                        eprintln!("ffmpeg: stacked layout uses CPU filters; decoding to system memory");
                    }
                }
            }
        } else if face_only {
            if is_hw {
                if allow_hwaccel && hw_decode_cuda {
                    if use_hw_frames {
                        eprintln!("ffmpeg: face-only layout uses CPU filters; keeping hwaccel decode");
                    } else {
                        eprintln!("ffmpeg: face-only layout uses CPU filters; decoding to system memory");
                    }
                }
            }
        } else if tracked_fullscreen {
            if is_hw {
                if allow_hwaccel && hw_decode_cuda {
                    if use_hw_frames {
                        eprintln!("ffmpeg: tracked full-frame uses CPU filters; keeping hwaccel decode");
                    } else {
                        eprintln!("ffmpeg: tracked full-frame uses CPU filters; decoding to system memory");
                    }
                }
            }
        } else {
            eprintln!("ffmpeg: stream flagged as unstable; using CPU filters with NVENC");
        }
    }

    let filter_cpu_hw = if cpu_hwdownload {
        Some(prepend_hwdownload_filter(&filter_cpu))
    } else {
        None
    };
    let selected_filters = if use_cpu_filters {
        filter_cpu_hw.as_ref().unwrap_or(&filter_cpu)
    } else {
        &filter_gpu
    };
    let temp_out = temp_output_path(out_path);
    cleanup_temp_output(&temp_out);

    let mut output = run_ffmpeg_encode(
        input,
        &temp_out,
        selected_filters,
        &video_encoder,
        is_hw,
        is_nvenc,
        fast_preset,
        allow_hwaccel,
        use_hw_frames,
        regen_pts,
        force_ts_input,
        fast_seek,
        start_offset_secs,
        duration_secs,
        progress,
        clip_metadata.as_ref(),
        subtitle_encode_spec.as_ref(),
    )
    .await?;

    if !output.status.success() {
        cleanup_temp_output(&temp_out);
        let stderr_first = String::from_utf8_lossy(&output.stderr).into_owned();
        let stdout_first = String::from_utf8_lossy(&output.stdout).into_owned();
        let mut retried_cpu = false;
        let mut retried_no_subtitles = false;
        let mut retried_no_captions = false;
        let mut retried_no_subtitles_reason: Option<&'static str> = None;

        if is_hw {
            let filter_failure = stderr_indicates_filter_issue(&stderr_first);
            if filter_failure && !layout_is_stacked {
                mark_stream_unstable(input);
                allow_hwaccel = false;
            }
            if ffmpeg_status_indicates_crash(&output.status, &stderr_first) {
                mark_stream_unstable(input);
                allow_hwaccel = false;
            }
            eprintln!(
                "ffmpeg hardware path failed (status {}); falling back to CPU/libx264. stderr (truncated): {}",
                output.status,
                tail_trunc(&stderr_first, 400)
            );
            if is_nvenc && filter_failure && !use_cpu_filters {
                eprintln!("ffmpeg: retrying NVENC with CPU filters (software decode)");
                cleanup_temp_output(&temp_out);
                output = run_ffmpeg_encode(
                    input,
                    &temp_out,
                    &filter_cpu,
                    &video_encoder,
                    true,
                    true,
                    fast_preset,
                    false,
                    false,
                    regen_pts,
                    force_ts_input,
                    fast_seek,
                    start_offset_secs,
                    duration_secs,
                    progress,
                    clip_metadata.as_ref(),
                    subtitle_encode_spec.as_ref(),
                )
                .await?;
            }
            if !output.status.success() {
                cleanup_temp_output(&temp_out);
                output = run_ffmpeg_encode(
                    input,
                    &temp_out,
                    &filter_cpu,
                    "libx264",
                    false,
                    false,
                    fast_preset,
                    allow_hwaccel,
                    false,
                    regen_pts,
                    force_ts_input,
                    fast_seek,
                    start_offset_secs,
                    duration_secs,
                    progress,
                    clip_metadata.as_ref(),
                    subtitle_encode_spec.as_ref(),
                )
                .await?;
                retried_cpu = true;
            }
        }

        if !output.status.success() {
            let stderr2 = String::from_utf8_lossy(&output.stderr).into_owned();
            if ffmpeg_status_indicates_crash(&output.status, &stderr2) {
                mark_stream_unstable(input);
                allow_hwaccel = false;
                let drop_subtitles = subtitle_encode_spec.is_some();
                if drop_subtitles {
                    eprintln!(
                        "ffmpeg: crash detected; retrying without closed captions (keeping open captions)"
                    );
                } else {
                    eprintln!("ffmpeg: crash detected; retrying with software decode/encode");
                }
                cleanup_temp_output(&temp_out);
                output = run_ffmpeg_encode(
                    input,
                    &temp_out,
                    &filter_cpu,
                    "libx264",
                    false,
                    false,
                    fast_preset,
                    false,
                    false,
                    regen_pts,
                    force_ts_input,
                    fast_seek,
                    start_offset_secs,
                    duration_secs,
                    progress,
                    clip_metadata.as_ref(),
                    if drop_subtitles {
                        None
                    } else {
                        subtitle_encode_spec.as_ref()
                    },
                )
                .await?;
                if output.status.success() && drop_subtitles {
                    retried_no_subtitles = true;
                    retried_no_subtitles_reason = Some("ffmpeg crash");
                }
            }
        }

        if !output.status.success() && has_open_captions && !retried_no_captions {
            let stderr2 = String::from_utf8_lossy(&output.stderr).into_owned();
            if caption_render_mode() == CaptionRender::Drawtext
                && (ffmpeg_status_indicates_crash(&output.status, &stderr2)
                    || ffmpeg_error_indicates_caption_issue(&stderr2))
            {
                if let (Some(cfg), Some(payload)) = (caption_cfg.as_ref(), transcript.as_ref()) {
                    let adjusted_cfg = adjust_caption_font_size(cfg, payload, out_w);
                    let y = caption_y_for_layout(&adjusted_cfg, layout_is_stacked, stacked_dims, out_h);
                    if let Some(ass) = build_ass_from_payload_with_limits(
                        payload,
                        duration_secs,
                        &adjusted_cfg,
                        y,
                        out_w,
                        out_h,
                    ) {
                        match write_temp_ass(&ass) {
                            Ok(path) => {
                                let chain =
                                    build_caption_subtitles_filter(&path, &adjusted_cfg, y, out_w, out_h);
                                _open_caption_temp = Some(TempSubtitle::new(path.clone()));
                                if !chain.is_empty() {
                                    let fallback_filter = append_filter_graph(&filter_cpu_base, &chain);
                                    eprintln!(
                                        "ffmpeg: retrying with subtitles renderer for open captions"
                                    );
                                    cleanup_temp_output(&temp_out);
                                    output = run_ffmpeg_encode(
                                        input,
                                        &temp_out,
                                        &fallback_filter,
                                        "libx264",
                                        false,
                                        false,
                                        fast_preset,
                                        false,
                                        false,
                                        regen_pts,
                                        force_ts_input,
                                        fast_seek,
                                        start_offset_secs,
                                        duration_secs,
                                        progress,
                                        clip_metadata.as_ref(),
                                        None,
                                    )
                                    .await?;
                                    // on success we keep subtitles renderer result
                                }
                            }
                            Err(err) => {
                                eprintln!("open captions: failed to write ass: {err:#}");
                            }
                        }
                    }
                }
            }

            if output.status.success() {
                // Successfully recovered with subtitles renderer.
            } else if ffmpeg_status_indicates_crash(&output.status, &stderr2) {
                eprintln!("ffmpeg: crash detected; retrying without open captions");
                cleanup_temp_output(&temp_out);
                output = run_ffmpeg_encode(
                    input,
                    &temp_out,
                    &filter_cpu_base,
                    "libx264",
                    false,
                    false,
                    fast_preset,
                    false,
                    false,
                    regen_pts,
                    force_ts_input,
                    fast_seek,
                    start_offset_secs,
                    duration_secs,
                    progress,
                    clip_metadata.as_ref(),
                    None,
                )
                .await?;
                if output.status.success() {
                    retried_no_captions = true;
                }
            }
        }

        if !output.status.success() {
            cleanup_temp_output(&temp_out);
            let stderr2 = String::from_utf8_lossy(&output.stderr).into_owned();
            if has_captions && ffmpeg_error_indicates_caption_issue(&stderr2) {
                if has_open_captions && subtitle_encode_spec.is_some() {
                    eprintln!("ffmpeg: captions failed; retrying without closed captions");
                    cleanup_temp_output(&temp_out);
                    output = run_ffmpeg_encode(
                        input,
                        &temp_out,
                        &filter_cpu,
                        "libx264",
                        false,
                        false,
                        fast_preset,
                        allow_hwaccel,
                        false,
                        regen_pts,
                        force_ts_input,
                        fast_seek,
                        start_offset_secs,
                        duration_secs,
                        progress,
                        clip_metadata.as_ref(),
                        None,
                    )
                    .await?;
                    retried_no_subtitles = output.status.success();
                    if retried_no_subtitles && retried_no_subtitles_reason.is_none() {
                        retried_no_subtitles_reason = Some("font/render failure");
                    }
                }
                if !output.status.success() {
                    eprintln!("ffmpeg: captions failed; retrying without captions");
                    cleanup_temp_output(&temp_out);
                    output = run_ffmpeg_encode(
                        input,
                        &temp_out,
                        &filter_cpu_base,
                        "libx264",
                        false,
                        false,
                        fast_preset,
                        allow_hwaccel,
                        false,
                        regen_pts,
                        force_ts_input,
                        fast_seek,
                        start_offset_secs,
                        duration_secs,
                        progress,
                        clip_metadata.as_ref(),
                        None,
                    )
                    .await?;
                    retried_no_captions = true;
                }
            }
        }

        if !output.status.success() {
            cleanup_temp_output(&temp_out);
            let stderr2 = String::from_utf8_lossy(&output.stderr);
            let stdout2 = String::from_utf8_lossy(&output.stdout);
            anyhow::bail!(
                "ffmpeg exited with status {}\nstdout: {}\nstderr: {}",
                output.status,
                stdout2.trim(),
                stderr2.trim()
            );
        }
        if retried_cpu {
            eprintln!(
                "ffmpeg fallback succeeded with CPU/libx264 after GPU pipeline failure. Previous stderr (truncated): {} | stdout (truncated): {}",
                tail_trunc(&stderr_first, 200),
                tail_trunc(&stdout_first, 200)
            );
        }
        if retried_no_subtitles {
            let reason = retried_no_subtitles_reason.unwrap_or("font/render failure");
            if captions_persist_failures() {
                disable_closed_captions_for_run(reason);
            }
            eprintln!(
                "ffmpeg retry succeeded without closed captions after {reason}. Previous stderr (truncated): {}",
                tail_trunc(&stderr_first, 200)
            );
        }
        if retried_no_captions {
            if captions_persist_failures() {
                disable_open_captions_for_run("ffmpeg crash");
            }
            eprintln!(
                "ffmpeg retry succeeded without captions after crash. Previous stderr (truncated): {}",
                tail_trunc(&stderr_first, 200)
            );
        }
    }

    if let Some(subs) = subtitle_spec.as_ref() {
        let muxed_out = temp_output_path_with_suffix(&temp_out, "muxed");
        cleanup_temp_output(&muxed_out);
        let output = run_ffmpeg_mux_subtitles(&temp_out, &muxed_out, subs).await?;
        if !output.status.success() {
            cleanup_temp_output(&muxed_out);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            anyhow::bail!(
                "ffmpeg subtitle mux failed with status {}\nstdout: {}\nstderr: {}",
                output.status,
                stdout.trim(),
                stderr.trim()
            );
        }
        cleanup_temp_output(&temp_out);
        fs::rename(&muxed_out, &temp_out).with_context(|| {
            format!(
                "renaming subtitle mux output {} -> {}",
                muxed_out.display(),
                temp_out.display()
            )
        })?;
    }

    finalize_output_path(&temp_out, out_path)?;
    if let (Some(meta), Some(cfg)) = (clip_metadata.as_ref(), llm_cfg.as_ref()) {
        if cfg.rename_files {
            if let Ok(Some(new_path)) = rename_output_with_title(out_path, &meta.title) {
                eprintln!("clip metadata: renamed output -> {}", new_path.display());
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum EnvValueMode {
    Required,
    Optional,
    Flag,
}

#[derive(Clone, Copy, Debug)]
struct EnvSpec {
    env: &'static str,
    mode: EnvValueMode,
}

const ENV_SPECS: &[EnvSpec] = &[
    EnvSpec { env: "CLIP_PAGE_URL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LAYOUT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_RATIO", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_CROP", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_CONTEXT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ZOOM", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_CENTER", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_BOX", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_REGION", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ANCHOR", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAME_CENTER", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAME_REGION", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_BACKEND", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_ORT_DYLIB", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_ORT_DEVICE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_ORT_GPU_MEM_LIMIT_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_ORT_MIN_FREE_VRAM_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_DUMP_DIR", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_DUMP_RAW", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_PICK_RAW", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_SCORE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_TRACK_STEP", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_REFRAME_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_TILE_MIN_SCORE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_TILE_MAX_DEPTH", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_HEAD_TOP", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_HEAD_TOP_MIN", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_HEAD_TOP_MAX", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_EYE_TOP_RATIO", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_EYE_CHIN_RATIO", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_SHOULDER_SCALE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_MESH_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_MODEL_MIN_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_MODEL_MAX_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_LOAD_TIMEOUT_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_BACKEND", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_MESH_TRACT_OPT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_INPUT_SIZE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_INPUT_MAX", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_INPUT_SCALE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_REGION_SCALE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_HEADROOM", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_POSE", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_POSE_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_MODEL_MIN_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_MODEL_MAX_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_BACKEND", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_POSE_TRACT_OPT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_SCORE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_INPUT_SCALE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_HEAD_RATIO", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_SHOULDER_MARGIN", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_ACTIVE_MOTION", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ACTIVE_AREA_RATIO", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ID", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_ID_FILE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ID_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ID_THRESHOLD", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ID_REQUIRE_MOTION", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_ID_MOTION", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ID_BGR", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_ID_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_TRACK", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_LOCK_CENTER", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_FALLBACK", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FFMPEG_PROGRESS", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FFMPEG_NO_LIMITS", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FFMPEG_PROGRESS_STALL_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_AUDIO_NORM", mode: EnvValueMode::Optional },
    EnvSpec { env: "GPU_VRAM_RESERVE_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "GPU_PICK_STRATEGY", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_CAPTIONS", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_CAPTIONS_POSITION", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_FONT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_SIZE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_COLOR", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_OUTLINE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_OUTLINE_COLOR", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_MIN_WORD_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_MAX_WORDS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_CHEST_RATIO", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_MARGIN_OFFSET", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_RENDER", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_SCALE_MIN", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_SCALE_MAX", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_SCALE_LOCK", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_DRAWTEXT_MAX", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_CAPTIONS_PERSIST_FAILURE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_BUCKET_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_CLOSED_CAPTIONS", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_LLM_ENABLE", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_LLM_ENDPOINT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_TEMPERATURE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_TIMEOUT_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_BACKOFF_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_MAX_TRANSCRIPT_CHARS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_TITLE_MAX_CHARS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_DESCRIPTION_MAX_CHARS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_GPU_LAYERS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_FILE_RENAME", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_LLM_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_BUDGET_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_DETECT", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_DETECT_SIZE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_DETECT_SAMPLES", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_DETECT_START", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_DETECT_STEP", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_DETECT_FULL", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_DETECT_BUDGET_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_REPROCESS_CHUNK_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_REPROCESS_SUBCHUNK_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_REPROCESS_FAST", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_REPROCESS_CONTINUE_ON_ERROR", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_TS_REALTIME", mode: EnvValueMode::Flag },
    EnvSpec { env: "CLIP_GAMEPLAY", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_REGION_DETECT", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_LIVE_CONFIG", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LIVE_CONFIG_POLL_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LIVE_FAST", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_LIVE_LAYOUT_TTL_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LOW_RESOURCES", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_SINGLE_INSTANCE", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_STREAMS_FILE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_STREAMS_POLL_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_STREAMS_MAX_CONCURRENT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_M3U8_REFRESH_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_STREAM_OFFLINE_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_WAKE_WORDS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_PROFILE", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_WATCHDOG_GAP_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_ENABLE", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_EMOTION_AUDIO", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_EMOTION_FACE", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_EMOTION_THRESHOLD", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_WORDS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_AUDIO_RMS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_AUDIO_PEAK", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_AUDIO_WEIGHT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_TEXT_WEIGHT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_REFRACTORY_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_FACE_MOTION", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_GAMEPLAY_MODEL_DIR", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_TEXT_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_VISION_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_TOKENIZER", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_STRIDE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_TOPK", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_SCORE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_SIZE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_LABELS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_NEG_LABELS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_BUDGET_SECS", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_GAMEPLAY_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_CAM_LABELS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAM_NEG_LABELS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAM_SCORE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAM_TOPK", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAM_REGION_SCALE", mode: EnvValueMode::Required },
    EnvSpec { env: "M3U8_URL_OVERRIDE", mode: EnvValueMode::Required },
    EnvSpec { env: "M3U8_TEST_URL", mode: EnvValueMode::Required },
    EnvSpec { env: "M3U8_PAGE_URL", mode: EnvValueMode::Required },
    EnvSpec { env: "COOKIE_HEADER", mode: EnvValueMode::Required },
    EnvSpec { env: "KICK_COOKIE", mode: EnvValueMode::Required },
    EnvSpec { env: "TIKTOK_COOKIE", mode: EnvValueMode::Required },
    EnvSpec { env: "TWITCH_COOKIE", mode: EnvValueMode::Required },
    EnvSpec { env: "HEADLESS_M3U8_SCRIPT", mode: EnvValueMode::Required },
    EnvSpec { env: "HEADLESS_M3U8_SCRIPT_TIKTOK", mode: EnvValueMode::Required },
    EnvSpec { env: "LOG_M3U8_HEADERS", mode: EnvValueMode::Flag },
    EnvSpec { env: "TWITCH_CLIENT_ID", mode: EnvValueMode::Required },
    EnvSpec { env: "TWITCH_OAUTH_TOKEN", mode: EnvValueMode::Required },
    EnvSpec { env: "TWITCH_AUTH_TOKEN", mode: EnvValueMode::Required },
    EnvSpec { env: "WAKE_REFRACTORY_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "WAKE_BUFFER_HEADROOM_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "WAKE_BUFFER_RESTART_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "WAKE_NO_WORDS_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "WAKE_FF_AF", mode: EnvValueMode::Required },
    EnvSpec { env: "SKIP_CLIP_SAVE", mode: EnvValueMode::Optional },
    EnvSpec { env: "MIC_DEVICE", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_RT_TARGET", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_MODEL_CANDIDATES", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_MODEL_BENCHMARK", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_GPU", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_CLIP_GPU", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_MIN_FREE_VRAM_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_CLIP_MIN_FREE_VRAM_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_CLIP_VRAM_FACTOR", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_CLIP_VRAM_OVERHEAD_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_CLIP_VRAM_EXTRA_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_GPU_DEVICE", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_CLIP_GPU_DEVICE", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_LOG_LEVEL", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_ISOLATE", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_WORKER_MODE", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_WORKER_STATUS_PATH", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_WORKER_MEDIA_URL_PATH", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_WORKER_LOG_RAW", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_WORKER_STATUS_MS", mode: EnvValueMode::Required },
    EnvSpec { env: "GGML_LOG_LEVEL", mode: EnvValueMode::Required },
    EnvSpec { env: "FFMPEG_BIN", mode: EnvValueMode::Required },
    EnvSpec { env: "FFMPEG_ENCODER", mode: EnvValueMode::Required },
    EnvSpec { env: "FFMPEG_HWACCEL", mode: EnvValueMode::Required },
    EnvSpec { env: "FFMPEG_HWACCEL_DEVICE", mode: EnvValueMode::Required },
    EnvSpec { env: "FFMPEG_HWACCEL_FALLBACK", mode: EnvValueMode::Required },
    EnvSpec { env: "FFMPEG_MIN_FREE_VRAM_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "FFMPEG_ENCODE_TIMEOUT_SECS", mode: EnvValueMode::Required },
];

struct ParsedCli {
    positionals: Vec<String>,
    override_phrase: Option<String>,
    log_raw_wake: bool,
    log_raw_wake_set: bool,
    mic_device: Option<String>,
    stream_urls: Vec<String>,
    streams_file: Option<String>,
    env_overrides: Vec<(String, String)>,
}

fn env_to_flag(env: &str) -> String {
    env.to_ascii_lowercase().replace('_', "-")
}

fn parse_cli_args(args: &[String]) -> Result<ParsedCli> {
    let mut positionals = Vec::new();
    let mut env_overrides = Vec::new();
    let mut override_phrase = None;
    let mut log_raw_wake = true;
    let mut log_raw_wake_set = false;
    let mut low_resources_override: Option<bool> = None;
    let mut mic_device = None;
    let mut stream_urls: Vec<String> = Vec::new();
    let mut streams_file: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--" {
            positionals.extend(args[i + 1..].iter().cloned());
            break;
        }
        if let Some(rest) = arg.strip_prefix("--phrase=") {
            override_phrase = Some(rest.to_string());
            i += 1;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--stream=") {
            stream_urls.extend(split_stream_list(rest));
            i += 1;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--streams=") {
            streams_file = Some(rest.to_string());
            i += 1;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--phrases=") {
            override_phrase = Some(rest.to_string());
            i += 1;
            continue;
        }
        if arg == "--phrase" || arg == "--phrases" {
            i += 1;
            let Some(val) = args.get(i) else {
                anyhow::bail!("--phrase expects a value");
            };
            override_phrase = Some(val.clone());
            i += 1;
            continue;
        }
        if arg == "--stream" {
            i += 1;
            let Some(val) = args.get(i) else {
                anyhow::bail!("--stream expects a value");
            };
            stream_urls.extend(split_stream_list(val));
            i += 1;
            continue;
        }
        if arg == "--streams" {
            i += 1;
            let Some(val) = args.get(i) else {
                anyhow::bail!("--streams expects a value");
            };
            streams_file = Some(val.clone());
            i += 1;
            continue;
        }
        if arg == "--log-raw-wake" {
            log_raw_wake = true;
            log_raw_wake_set = true;
            i += 1;
            continue;
        }
        if arg == "--no-log-raw-wake" {
            log_raw_wake = false;
            log_raw_wake_set = true;
            i += 1;
            continue;
        }
        if arg == "--low-resources" {
            low_resources_override = Some(true);
            i += 1;
            continue;
        }
        if arg == "--no-low-resources" {
            low_resources_override = Some(false);
            i += 1;
            continue;
        }

        let mut handled_env = false;
        if let Some(raw_flag) = arg.strip_prefix("--") {
            let (flag, inline_value) = match raw_flag.split_once('=') {
                Some((f, v)) => (f.to_ascii_lowercase(), Some(v.to_string())),
                None => (raw_flag.to_ascii_lowercase(), None),
            };
            for spec in ENV_SPECS {
                let spec_flag = env_to_flag(spec.env);
                if spec_flag == flag {
                    let value = match spec.mode {
                        EnvValueMode::Required => {
                            if let Some(val) = inline_value {
                                val
                            } else {
                                let next = args.get(i + 1).ok_or_else(|| {
                                    anyhow::anyhow!("--{flag} expects a value")
                                })?;
                                if next.starts_with("--") {
                                    anyhow::bail!("--{flag} expects a value");
                                }
                                i += 1;
                                next.clone()
                            }
                        }
                        EnvValueMode::Optional => inline_value.unwrap_or_else(|| "1".to_string()),
                        EnvValueMode::Flag => inline_value.unwrap_or_else(|| "1".to_string()),
                    };
                    if spec.env == "MIC_DEVICE" {
                        mic_device = Some(value.clone());
                    }
                    env_overrides.push((spec.env.to_string(), value));
                    handled_env = true;
                    break;
                }
            }
        }

        if handled_env {
            i += 1;
            continue;
        }

        if arg.starts_with('-') {
            i += 1;
            continue;
        }

        positionals.push(arg.clone());
        i += 1;
    }

    if let Some(value) = low_resources_override {
        env_overrides.push((
            "CLIP_LOW_RESOURCES".to_string(),
            if value { "1".to_string() } else { "0".to_string() },
        ));
    }

    Ok(ParsedCli {
        positionals,
        override_phrase,
        log_raw_wake,
        log_raw_wake_set,
        mic_device,
        stream_urls,
        streams_file,
        env_overrides,
    })
}

fn apply_env_overrides(overrides: &[(String, String)]) {
    for (key, value) in overrides {
        std::env::set_var(key, value);
    }
}

fn read_streams_poll_secs() -> Duration {
    std::env::var("CLIP_STREAMS_POLL_SECS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| Duration::from_secs_f32(v.clamp(0.2, 60.0)))
        .unwrap_or_else(|| Duration::from_secs(5))
}

fn read_streams_max_concurrent() -> usize {
    std::env::var("CLIP_STREAMS_MAX_CONCURRENT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.clamp(1, 16))
        .unwrap_or(1)
}

fn read_wake_buffer_restart_secs() -> Option<Duration> {
    let raw = std::env::var("WAKE_BUFFER_RESTART_SECS").ok();
    let parsed = raw.as_deref().and_then(|v| v.parse::<f32>().ok());
    let secs = match parsed {
        Some(v) if v.is_finite() && v > 0.0 => v,
        Some(_) => return None,
        None => 300.0,
    };
    Some(Duration::from_secs_f32(secs))
}

async fn run_ffmpeg_mux_subtitles(
    input: &Path,
    out_path: &Path,
    subtitles: &SubtitleSpec,
) -> Result<std::process::Output> {
    let mut cmd = Command::new("ffmpeg");
    apply_fontconfig_env_for_ffmpeg(&mut cmd);
    cmd.arg("-y");
    cmd.arg("-i").arg(input.as_os_str());
    cmd.arg("-i").arg(subtitles.path.as_os_str());
    cmd.arg("-map").arg("0");
    cmd.arg("-map").arg("1:0");
    cmd.arg("-c").arg("copy");
    cmd.arg("-c:s").arg(subtitle_codec_for_output(out_path));
    cmd.arg("-metadata:s:s:0").arg("language=eng");
    cmd.arg("-disposition:s:s:0").arg("default");
    cmd.arg(out_path.as_os_str());
    let output = cmd
        .output()
        .await
        .context("failed to run ffmpeg subtitle mux")?;
    Ok(output)
}

fn read_wake_no_words_secs() -> Option<Duration> {
    let raw = std::env::var("WAKE_NO_WORDS_SECS").ok();
    let parsed = raw.as_deref().and_then(|v| v.parse::<f32>().ok());
    let secs = match parsed {
        Some(v) if v.is_finite() && v > 0.0 => v,
        Some(_) => return None,
        None => 180.0,
    };
    Some(Duration::from_secs_f32(secs))
}

fn read_clip_gap_watchdog_secs() -> Option<Duration> {
    let raw = std::env::var("CLIP_WATCHDOG_GAP_SECS").ok();
    let parsed = raw.as_deref().and_then(|v| v.parse::<f32>().ok());
    let secs = match parsed {
        Some(v) if v.is_finite() && v > 0.0 => v,
        Some(_) => return None,
        None => 600.0,
    };
    Some(Duration::from_secs_f32(secs))
}

fn read_wake_min_clip_gap_secs() -> Option<Duration> {
    let raw = std::env::var("CLIP_WAKE_MIN_CLIP_SECS").ok();
    let parsed = raw.as_deref().and_then(|v| v.parse::<f32>().ok());
    let secs = match parsed {
        Some(v) if v.is_finite() && v > 0.0 => v,
        Some(v) if v.is_finite() && v <= 0.0 => return None,
        Some(_) => return None,
        None => 300.0,
    };
    Some(Duration::from_secs_f32(secs.clamp(5.0, 3600.0)))
}

fn read_reprocess_chunk_secs() -> Option<f32> {
    let raw = std::env::var("CLIP_REPROCESS_CHUNK_SECS").ok();
    let parsed = raw.as_deref().and_then(|v| v.trim().parse::<f32>().ok());
    let secs = match parsed {
        Some(v) if v.is_finite() && v > 0.0 => v,
        Some(_) => return None,
        None => 60.0,
    };
    Some(secs.clamp(5.0, 3600.0))
}

fn read_reprocess_subchunk_secs() -> Option<f32> {
    let raw = std::env::var("CLIP_REPROCESS_SUBCHUNK_SECS").ok();
    let parsed = raw.as_deref().and_then(|v| v.trim().parse::<f32>().ok());
    let secs = match parsed {
        Some(v) if v.is_finite() && v > 0.0 => v,
        Some(_) => return None,
        None => return None,
    };
    Some(secs.clamp(1.0, 3600.0))
}

fn reprocess_fast_enabled() -> bool {
    std::env::var("CLIP_REPROCESS_FAST")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn reprocess_continue_on_error() -> bool {
    std::env::var("CLIP_REPROCESS_CONTINUE_ON_ERROR")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn ffmpeg_progress_enabled() -> bool {
    std::env::var("CLIP_FFMPEG_PROGRESS")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn ffmpeg_progress_stall_timeout() -> Option<Duration> {
    if ffmpeg_no_limits_enabled() {
        return None;
    }
    std::env::var("CLIP_FFMPEG_PROGRESS_STALL_SECS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(Duration::from_secs_f32)
}

fn ffmpeg_no_limits_enabled() -> bool {
    std::env::var("CLIP_FFMPEG_NO_LIMITS")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReprocessResumeState {
    input_path: String,
    input_size: u64,
    input_mtime_ms: u64,
    chunk_secs: f32,
    subchunk_secs: Option<f32>,
    total_secs: Option<f32>,
    next_start_secs: f32,
    next_index: u32,
    segment_index: Option<u32>,
    segment_start_secs: Option<f32>,
    segment_duration_secs: Option<f32>,
    segment_output_path: Option<String>,
    next_part_index: Option<u32>,
    updated_ms: u64,
}

fn input_fingerprint(path: &Path) -> Option<(u64, u64)> {
    let meta = fs::metadata(path).ok()?;
    let size = meta.len();
    let mtime_ms = meta
        .modified()
        .ok()
        .and_then(|ts| ts.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    Some((size, mtime_ms))
}

fn reprocess_resume_path(save_root: &str, input_path: &Path, chunk_secs: f32) -> PathBuf {
    let base = non_video_dir(save_root);
    let (safe, hash) = reprocess_resume_tag(input_path, chunk_secs);
    base.join(format!("reprocess_{safe}_{hash:016x}.json"))
}

fn reprocess_work_dir(save_root: &str, input_path: &Path, chunk_secs: f32) -> PathBuf {
    let base = non_video_dir(save_root);
    let (safe, hash) = reprocess_resume_tag(input_path, chunk_secs);
    base.join(format!("reprocess_{safe}_{hash:016x}"))
}

fn reprocess_resume_tag(input_path: &Path, chunk_secs: f32) -> (String, u64) {
    let stem = input_path
        .file_stem()
        .and_then(|v| v.to_str())
        .unwrap_or("input");
    let mut safe = String::new();
    for ch in stem.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            safe.push(ch);
        } else if ch.is_ascii_whitespace() {
            safe.push('_');
        }
        if safe.len() >= 32 {
            break;
        }
    }
    if safe.is_empty() {
        safe.push_str("input");
    }
    let mut hasher = DefaultHasher::new();
    input_path.to_string_lossy().hash(&mut hasher);
    chunk_secs.to_bits().hash(&mut hasher);
    if let Some((size, mtime_ms)) = input_fingerprint(input_path) {
        size.hash(&mut hasher);
        mtime_ms.hash(&mut hasher);
    }
    let hash = hasher.finish();
    (safe, hash)
}

fn reprocess_segment_dir(
    save_root: &str,
    input_path: &Path,
    chunk_secs: f32,
    segment_index: u32,
    output_path: &Path,
) -> PathBuf {
    let base = reprocess_work_dir(save_root, input_path, chunk_secs);
    let mut stem = output_path
        .file_stem()
        .and_then(|v| v.to_str())
        .unwrap_or("segment")
        .to_string();
    stem.retain(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if stem.is_empty() {
        stem.push_str("segment");
    }
    base.join(format!("seg_{segment_index:04}_{stem}"))
}

fn collect_part_files(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|v| v.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("mp4"))
                .unwrap_or(false)
        })
        .collect();
    entries.sort();
    Ok(entries)
}

async fn concat_media_parts(parts: &[PathBuf], output_path: &Path) -> Result<()> {
    if parts.is_empty() {
        anyhow::bail!("no media parts to concatenate");
    }
    let cwd = std::env::current_dir().context("concat: resolving cwd")?;
    let output_abs = if output_path.is_absolute() {
        output_path.to_path_buf()
    } else {
        cwd.join(output_path)
    };
    let parent = output_abs.parent().unwrap_or_else(|| Path::new("."));
    let list_path = parent.join("concat_parts.txt");
    let mut list_body = String::new();
    for part in parts {
        let part_abs = if part.is_absolute() {
            part.to_path_buf()
        } else {
            cwd.join(part)
        };
        if !part_abs.exists() {
            anyhow::bail!("concat missing part {}", part_abs.display());
        }
        let line = part_abs
            .to_string_lossy()
            .replace('\\', "/")
            .replace('\'', "'\\''");
        list_body.push_str("file '");
        list_body.push_str(&line);
        list_body.push_str("'\n");
    }
    fs::write(&list_path, list_body)?;

    let temp_out = temp_output_path(&output_abs);
    cleanup_temp_output(&temp_out);
    let status = Command::new(ffmpeg_bin())
        .args([
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path.to_string_lossy().as_ref(),
            "-fflags",
            "+genpts",
            "-avoid_negative_ts",
            "make_zero",
            "-max_interleave_delta",
            "0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-af",
            "aresample=async=1:first_pts=0",
            "-c:s",
            "copy",
            temp_out.to_string_lossy().as_ref(),
        ])
        .status()
        .await
        .context("running ffmpeg concat")?;
    let _ = fs::remove_file(&list_path);
    if !status.success() {
        anyhow::bail!("ffmpeg concat failed with status {status}");
    }
    finalize_output_path(&temp_out, &output_abs)?;
    Ok(())
}

fn read_reprocess_resume_state(path: &Path) -> Option<ReprocessResumeState> {
    let payload = fs::read_to_string(path).ok()?;
    serde_json::from_str(&payload).ok()
}

fn write_reprocess_resume_state(path: &Path, state: &ReprocessResumeState) -> Result<()> {
    let payload = serde_json::to_vec(state).context("serializing reprocess resume state")?;
    write_atomic_file(path, &payload)
}

fn clear_reprocess_resume_state(path: &Path) {
    let _ = fs::remove_file(path);
}

fn read_m3u8_refresh_secs() -> Option<Duration> {
    let raw = std::env::var("CLIP_M3U8_REFRESH_SECS").ok();
    let parsed = raw.as_deref().and_then(|v| v.parse::<f32>().ok());
    let secs = match parsed {
        Some(v) if v.is_finite() && v > 0.0 => v,
        Some(_) => return None,
        None => 240.0,
    };
    Some(Duration::from_secs_f32(secs))
}

fn read_stream_offline_secs() -> Option<Duration> {
    let raw = std::env::var("CLIP_STREAM_OFFLINE_SECS").ok();
    let parsed = raw.as_deref().and_then(|v| v.parse::<f32>().ok());
    let secs = match parsed {
        Some(v) if v.is_finite() && v > 0.0 => v,
        Some(_) => return None,
        None => 120.0,
    };
    Some(Duration::from_secs_f32(secs))
}

fn bump_backoff(current: Duration) -> Duration {
    let next_ms = if current.is_zero() {
        500_u64
    } else {
        let doubled = current.as_millis().saturating_mul(2);
        if doubled > u128::from(u64::MAX) {
            u64::MAX
        } else {
            doubled as u64
        }
    };
    let jitter_ms = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| (d.subsec_millis() as u64) % 250)
        .unwrap_or(0);
    Duration::from_millis(next_ms.saturating_add(jitter_ms).min(5000))
}

async fn refresh_media_source(
    hls: &HlsClient,
    page_url: &str,
    media_url: &Arc<Mutex<Url>>,
    media_headers: &Arc<Mutex<StreamHeaders>>,
    seen: &mut HashSet<String>,
    whisper_isolate: bool,
    use_mic_for_wake: bool,
    worker_media_url_path: &Path,
    reason: &str,
) -> Result<Url> {
    eprintln!("{reason}");
    let (new_url, headers) = hls
        .refresh_media_url_from_page_headless_with_headers(page_url)
        .await?;
    {
        let mut guard = lock_or_recover(&media_url, "media url refresh");
        *guard = new_url.clone();
    }
    {
        let mut guard = lock_or_recover(&media_headers, "media headers refresh");
        *guard = headers;
    }
    seen.clear();
    if whisper_isolate && !use_mic_for_wake {
        if let Err(err) = write_media_url_file(worker_media_url_path, &new_url) {
            eprintln!("whisper worker: failed to write media url: {err:#}");
        }
    }
    eprintln!("refreshed variant: {}", new_url);
    Ok(new_url)
}

fn spawn_self_restart(reason: String) {
    static RESTARTING: AtomicBool = AtomicBool::new(false);
    if RESTARTING.swap(true, Ordering::SeqCst) {
        return;
    }
    eprintln!("{reason}");
    let exe = match std::env::current_exe() {
        Ok(path) => path,
        Err(err) => {
            eprintln!("restart failed: unable to resolve executable: {err:#}");
            return;
        }
    };
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    match StdCommand::new(&exe)
        .args(&args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(_) => {
            eprintln!("restarting autoclip (buffer watchdog).");
            std::process::exit(0);
        }
        Err(err) => {
            eprintln!("restart failed: {err:#}");
        }
    }
}

fn kill_child_now(child: &Arc<Mutex<Option<Child>>>) {
    if let Ok(mut guard) = child.lock() {
        if let Some(proc) = guard.as_mut() {
            kill_child_with_timeout(proc, "child", Duration::from_secs(2));
        }
    }
}

fn kill_child_and_wait(child: &Arc<Mutex<Option<Child>>>) {
    if let Ok(mut guard) = child.lock() {
        if let Some(mut proc) = guard.take() {
            kill_child_with_timeout(&mut proc, "child", Duration::from_secs(10));
        }
    }
}

fn kill_child_with_timeout(child: &mut Child, label: &str, timeout: Duration) {
    let _ = child.kill();
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) => {
                if start.elapsed() >= timeout {
                    eprintln!("{label}: child did not exit after kill within {:?}", timeout);
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(err) => {
                eprintln!("{label}: failed to reap child: {err:#}");
                return;
            }
        }
    }
}

fn spawn_ctrl_c_handler(stop: Arc<AtomicBool>, worker: Option<Arc<Mutex<Option<Child>>>>) {
    let counter = Arc::new(AtomicUsize::new(0));
    tokio::spawn(async move {
        loop {
            if tokio::signal::ctrl_c().await.is_err() {
                return;
            }
            let count = counter.fetch_add(1, Ordering::SeqCst) + 1;
            if count == 1 {
                eprintln!(
                    "Ctrl+C received; finishing in-flight work. Press Ctrl+C again to quit immediately."
                );
                stop.store(true, Ordering::Relaxed);
                if let Some(child) = worker.as_ref() {
                    kill_child_now(child);
                }
            } else {
                eprintln!("Ctrl+C received again; forcing exit.");
                std::process::exit(130);
            }
        }
    });
}

fn build_stream_save_path(base: &str, stream_id: &str) -> String {
    Path::new(base)
        .join(stream_id)
        .to_string_lossy()
        .to_string()
}

struct StreamTask {
    url: String,
    handle: tokio::task::JoinHandle<()>,
}

fn spawn_stream_task(
    base_config: &Config,
    stream_id: &str,
    url: &str,
    semaphore: Arc<Semaphore>,
) -> StreamTask {
    let mut cfg = base_config.clone();
    cfg.kick_url = url.to_string();
    cfg.save_path = build_stream_save_path(&cfg.save_path, stream_id);
    let app = AutoClip::new(cfg);
    let stream_id = stream_id.to_string();
    let url = url.to_string();
    let url_for_task = url.clone();
    let handle = tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => return,
        };
        eprintln!("streams: starting {stream_id} -> {url_for_task}");
        if let Err(err) = app.run_with_page(Some(&url_for_task)).await {
            eprintln!("streams: {stream_id} exited: {err:#}");
        }
    });
    StreamTask { url, handle }
}

async fn run_streams_supervisor(
    base_config: Config,
    streams_path: PathBuf,
) -> Result<()> {
    let poll = read_streams_poll_secs();
    let max_concurrent = read_streams_max_concurrent();
    let semaphore = Arc::new(Semaphore::new(max_concurrent.max(1)));
    let stop = Arc::new(AtomicBool::new(false));
    spawn_ctrl_c_handler(stop.clone(), None);
    eprintln!(
        "streams: watching {} (poll {:.1}s, max concurrent {})",
        streams_path.display(),
        poll.as_secs_f32(),
        max_concurrent
    );

    let mut active: HashMap<String, StreamTask> = HashMap::new();

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let streams = match read_streams_file(&streams_path) {
            Ok(list) => list,
            Err(err) => {
                eprintln!(
                    "streams: failed to read {}: {err}",
                    streams_path.display()
                );
                let _ = sync_streams_file(&streams_path, &[]);
                sleep(poll).await;
                continue;
            }
        };
        let mut desired: HashMap<String, String> = HashMap::new();
        for url in streams {
            let id = stream_id_from_url(&url);
            desired.insert(id, url);
        }

        let finished: Vec<String> = active
            .iter()
            .filter(|(_, task)| task.handle.is_finished())
            .map(|(id, _)| id.clone())
            .collect();
        for id in finished {
            active.remove(&id);
        }

        let mut remove: Vec<String> = Vec::new();
        for (id, task) in active.iter() {
            match desired.get(id) {
                Some(url) if url == &task.url => {}
                _ => {
                    eprintln!("streams: stopping {id}");
                    task.handle.abort();
                    remove.push(id.clone());
                }
            }
        }
        for id in remove {
            active.remove(&id);
        }

        for (id, url) in desired.iter() {
            if !active.contains_key(id) {
                let task = spawn_stream_task(&base_config, id, url, semaphore.clone());
                active.insert(id.clone(), task);
            }
        }

        sleep(poll).await;
    }

    for (id, task) in active {
        eprintln!("streams: stopping {id}");
        task.handle.abort();
    }
    Ok(())
}

fn resolve_wake_phrases(
    override_phrases: Option<&[String]>,
    fallback_phrase: &str,
) -> Vec<String> {
    if let Some(list) = override_phrases {
        if !list.is_empty() {
            return list.to_vec();
        }
    }
    if let Ok(env) = std::env::var("CLIP_WAKE_WORDS") {
        let parsed = split_wake_phrases(&env);
        if !parsed.is_empty() {
            return parsed;
        }
    }
    let fallback = split_wake_phrases(fallback_phrase);
    if fallback.is_empty() {
        vec![fallback_phrase.to_string()]
    } else {
        fallback
    }
}

const LIVE_CONFIG_IGNORE_KEYS: &[&str] = &["CLIP_LIVE_CONFIG", "CLIP_LIVE_CONFIG_POLL_SECS"];
const LIVE_CONFIG_PREFILL: &[(&str, &str)] = &[
    ("CLIP_FACE_FRAME_HEAD_TOP", "-0.28"),
    ("CLIP_FACE_FRAME_HEAD_TOP_MIN", "-0.8"),
    ("CLIP_FACE_FRAME_HEAD_TOP_MAX", "0.2"),
    ("CLIP_FACE_FRAME_EYE_TOP_RATIO", "0.45"),
    ("CLIP_FACE_FRAME_EYE_CHIN_RATIO", "0.55"),
    ("CLIP_FACE_FRAME_SHOULDER_SCALE", "3.2"),
    ("CLIP_FACE_CENTER", "0"),
    ("CLIP_FACE_MESH", "1"),
    ("CLIP_FACE_MESH_MODEL", "models/face_mesh/face_mesh.onnx"),
    ("CLIP_FACE_MESH_MODEL_MIN_MB", "1"),
    ("CLIP_FACE_MESH_MODEL_MAX_MB", "64"),
    ("CLIP_FACE_MESH_LOAD_TIMEOUT_SECS", "60"),
    ("CLIP_FACE_MESH_BACKEND", ""),
    ("CLIP_FACE_MESH_TRACT_OPT", "0"),
    ("CLIP_FACE_MESH_INPUT_SIZE", "192"),
    ("CLIP_FACE_MESH_INPUT_MAX", "512"),
    ("CLIP_FACE_MESH_INPUT_SCALE", "0.003921569"),
    ("CLIP_FACE_MESH_REGION_SCALE", "1.35"),
    ("CLIP_FACE_MESH_HEADROOM", "0.12"),
    ("CLIP_FACE_MESH_DEBUG", "0"),
    ("CLIP_FACE_ZOOM", "1.0"),
    ("CLIP_POSE", "1"),
    ("CLIP_POSE_MODEL", "models/pose/movenet_singlepose_thunder.onnx"),
    ("CLIP_POSE_MODEL_MIN_MB", "1"),
    ("CLIP_POSE_MODEL_MAX_MB", "64"),
    ("CLIP_POSE_LOAD_TIMEOUT_SECS", "60"),
    ("CLIP_POSE_BACKEND", ""),
    ("CLIP_POSE_TRACT_OPT", "0"),
    ("CLIP_POSE_SCORE", "0.30"),
    ("CLIP_POSE_INPUT_SIZE", "256"),
    ("CLIP_POSE_INPUT_MAX", "512"),
    ("CLIP_POSE_INPUT_SCALE", "0.003921569"),
    ("CLIP_POSE_HEAD_RATIO", "0.60"),
    ("CLIP_POSE_SHOULDER_MARGIN", "1.10"),
    ("CLIP_POSE_DEBUG", "0"),
    ("CLIP_FACE_ACTIVE_MOTION", "0.015"),
    ("CLIP_FACE_ACTIVE_AREA_RATIO", "0.35"),
    ("CLIP_FACE_ID", "0"),
    ("CLIP_FACE_ID_FILE", ""),
    ("CLIP_FACE_ID_MODEL", "models/face_id/arcface.onnx"),
    ("CLIP_FACE_ID_THRESHOLD", "0.35"),
    ("CLIP_FACE_ID_REQUIRE_MOTION", "1"),
    ("CLIP_FACE_ID_MOTION", "0.015"),
    ("CLIP_FACE_ID_BGR", "1"),
    ("CLIP_FACE_ID_DEBUG", "0"),
    ("CLIP_ORT_DYLIB", ""),
    ("CLIP_ORT_DEVICE", "auto"),
    ("CLIP_ORT_GPU_MEM_LIMIT_MB", "0"),
    ("CLIP_ORT_MIN_FREE_VRAM_MB", "512"),
    ("CLIP_CAPTIONS", "0"),
    ("CLIP_CAPTIONS_POSITION", "margin"),
    ("CLIP_CAPTIONS_FONT", ""),
    ("CLIP_CAPTIONS_SIZE", "0.045"),
    ("CLIP_CAPTIONS_COLOR", "white"),
    ("CLIP_CAPTIONS_OUTLINE", "3"),
    ("CLIP_CAPTIONS_OUTLINE_COLOR", "black"),
    ("CLIP_CAPTIONS_SCALE_MIN", "0.5"),
    ("CLIP_CAPTIONS_SCALE_MAX", "2.5"),
    ("CLIP_CAPTIONS_SCALE_LOCK", "0"),
    ("CLIP_CAPTIONS_PERSIST_FAILURE", "0"),
    ("CLIP_CAPTIONS_BUCKET_SECS", "0"),
    ("CLIP_CAPTIONS_MIN_WORD_SECS", "0.12"),
    ("CLIP_CAPTIONS_MAX_WORDS", "300"),
    ("CLIP_CAPTIONS_CHEST_RATIO", "0.65"),
    ("CLIP_CAPTIONS_MARGIN_OFFSET", "0"),
    ("CLIP_CAPTIONS_DEBUG", "0"),
    ("CLIP_CLOSED_CAPTIONS", "1"),
    ("CLIP_LLM_ENABLE", "1"),
    ("CLIP_LLM_ENDPOINT", "http://localhost:1234/v1/chat/completions"),
    ("CLIP_LLM_MODEL", "gemma-2-2b-it"),
    ("CLIP_LLM_TEMPERATURE", "0.2"),
    ("CLIP_LLM_TIMEOUT_SECS", "20"),
    ("CLIP_LLM_BACKOFF_SECS", "120"),
    ("CLIP_LLM_MAX_TRANSCRIPT_CHARS", "4000"),
    ("CLIP_LLM_TITLE_MAX_CHARS", "80"),
    ("CLIP_LLM_DESCRIPTION_MAX_CHARS", "280"),
    ("CLIP_LLM_GPU_LAYERS", "0"),
    ("CLIP_LLM_FILE_RENAME", "1"),
    ("CLIP_LLM_DEBUG", "0"),
    ("CLIP_FFMPEG_PROGRESS", "1"),
    ("CLIP_FFMPEG_PROGRESS_STALL_SECS", "120"),
    ("CLIP_REPROCESS_CHUNK_SECS", "60"),
    ("CLIP_REPROCESS_SUBCHUNK_SECS", "0"),
    ("CLIP_REPROCESS_FAST", "0"),
    ("CLIP_REPROCESS_CONTINUE_ON_ERROR", "0"),
    ("CLIP_LIVE_FAST", "1"),
    ("CLIP_LIVE_LAYOUT_TTL_SECS", "120"),
    ("CLIP_LOW_RESOURCES", "0"),
    ("CLIP_WAKE_WORDS", "orange"),
    ("CLIP_PROFILE", "1"),
    ("CLIP_WATCHDOG_GAP_SECS", "600"),
    ("WAKE_BUFFER_RESTART_SECS", "300"),
    ("WAKE_NO_WORDS_SECS", "180"),
    ("CLIP_STREAMS_POLL_SECS", "5"),
    ("CLIP_STREAMS_MAX_CONCURRENT", "1"),
    ("CLIP_M3U8_REFRESH_SECS", "240"),
    ("CLIP_STREAM_OFFLINE_SECS", "120"),
    ("CLIP_EMOTION_ENABLE", "0"),
    ("CLIP_EMOTION_AUDIO", "1"),
    ("CLIP_EMOTION_FACE", "0"),
    ("CLIP_EMOTION_THRESHOLD", "1.5"),
    ("CLIP_EMOTION_WORDS", "omg,oh my god,wow,no way,lets go,holy,insane,unbelievable,wtf"),
    ("CLIP_EMOTION_AUDIO_RMS", "0.08"),
    ("CLIP_EMOTION_AUDIO_PEAK", "0.35"),
    ("CLIP_EMOTION_AUDIO_WEIGHT", "1.0"),
    ("CLIP_EMOTION_TEXT_WEIGHT", "1.0"),
    ("CLIP_EMOTION_REFRACTORY_SECS", "8"),
    ("CLIP_EMOTION_FACE_MOTION", "0.035"),
    ("CLIP_EMOTION_DEBUG", "0"),
    ("FFMPEG_MIN_FREE_VRAM_MB", "512"),
    ("FFMPEG_ENCODE_TIMEOUT_SECS", "300"),
    ("GPU_VRAM_RESERVE_MB", "512"),
    ("GPU_PICK_STRATEGY", "free"),
    ("WHISPER_MIN_FREE_VRAM_MB", "2048"),
    ("WHISPER_CLIP_MIN_FREE_VRAM_MB", "4096"),
    ("WHISPER_CLIP_VRAM_FACTOR", "1.35"),
    ("WHISPER_CLIP_VRAM_OVERHEAD_MB", "512"),
    ("WHISPER_CLIP_VRAM_EXTRA_MB", "1024"),
    ("WHISPER_GPU_DEVICE", ""),
    ("WHISPER_CLIP_GPU_DEVICE", ""),
    ("WHISPER_CLIP_GPU", "0"),
    ("WHISPER_ISOLATE", "0"),
    ("WHISPER_WORKER_MODE", "stream"),
    ("WHISPER_WORKER_STATUS_PATH", ""),
    ("WHISPER_WORKER_MEDIA_URL_PATH", ""),
    ("WHISPER_WORKER_LOG_RAW", "0"),
    ("WHISPER_WORKER_STATUS_MS", "500"),
];

struct LiveConfigState {
    path: PathBuf,
    poll_interval: Duration,
    last_checked: Instant,
    last_modified: Option<SystemTime>,
    applied: HashMap<String, String>,
    baseline: HashMap<String, Option<String>>,
    missing_logged: bool,
}

impl LiveConfigState {
    fn from_env() -> Option<Self> {
        let path = std::env::var("CLIP_LIVE_CONFIG")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())?;
        let poll_secs = std::env::var("CLIP_LIVE_CONFIG_POLL_SECS")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(2.0);
        let poll_interval = Duration::from_secs_f32(poll_secs);
        let now = Instant::now();
        let last_checked = now.checked_sub(poll_interval).unwrap_or(now);
        Some(Self {
            path: PathBuf::from(path),
            poll_interval,
            last_checked,
            last_modified: None,
            applied: HashMap::new(),
            baseline: HashMap::new(),
            missing_logged: false,
        })
    }

    fn maybe_refresh(&mut self) {
        if self.last_checked.elapsed() < self.poll_interval {
            return;
        }
        self.last_checked = Instant::now();

        let metadata = match fs::metadata(&self.path) {
            Ok(meta) => meta,
            Err(_) => {
                if !self.missing_logged {
                    eprintln!(
                        "live config: file not found at {}; skipping updates",
                        self.path.display()
                    );
                }
                self.missing_logged = true;
                self.clear_applied();
                return;
            }
        };

        let modified = metadata.modified().ok();
        if modified.is_some() && modified == self.last_modified {
            return;
        }
        self.last_modified = modified;
        self.missing_logged = false;

        let contents = match fs::read_to_string(&self.path) {
            Ok(contents) => contents,
            Err(err) => {
                eprintln!(
                    "live config: failed to read {}: {err}",
                    self.path.display()
                );
                return;
            }
        };

        let next = parse_live_config(&contents);
        self.apply_settings(next);
    }

    fn apply_settings(&mut self, next: HashMap<String, String>) {
        if next == self.applied {
            return;
        }

        let mut updated = Vec::new();
        for (key, value) in &next {
            if self.applied.get(key) == Some(value) {
                continue;
            }
            if !self.baseline.contains_key(key) {
                self.baseline.insert(key.clone(), std::env::var(key).ok());
            }
            std::env::set_var(key, value);
            updated.push(key.clone());
        }

        let removed: Vec<String> = self
            .applied
            .keys()
            .filter(|key| !next.contains_key(*key))
            .cloned()
            .collect();
        for key in &removed {
            self.restore_baseline(key);
        }

        let ort_dylib_changed = updated.iter().any(|key| key == "CLIP_ORT_DYLIB")
            || removed.iter().any(|key| key == "CLIP_ORT_DYLIB");
        if !updated.is_empty() || !removed.is_empty() {
            if !updated.is_empty() {
                eprintln!("live config: applied {}", updated.join(", "));
            }
            if !removed.is_empty() {
                eprintln!("live config: cleared {}", removed.join(", "));
            }
        }
        if ort_dylib_changed {
            sync_ort_dylib_env(true);
        }

        self.applied = next;
    }

    fn clear_applied(&mut self) {
        if self.applied.is_empty() {
            return;
        }
        let ort_dylib_cleared = self.applied.contains_key("CLIP_ORT_DYLIB");
        let keys: Vec<String> = self.applied.keys().cloned().collect();
        for key in keys {
            self.restore_baseline(&key);
        }
        self.applied.clear();
        if ort_dylib_cleared {
            sync_ort_dylib_env(true);
        }
    }

    fn restore_baseline(&mut self, key: &str) {
        match self.baseline.get(key).cloned().unwrap_or(None) {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
}

fn parse_live_config(contents: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for (idx, raw_line) in contents.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }

        let (raw_key, raw_value) = split_live_config_line(line);
        let Some(raw_key) = raw_key else { continue };
        let value = raw_value.unwrap_or_else(|| "1".to_string());
        if value.trim().is_empty() {
            continue;
        }
        let Some(env_key) = normalize_live_config_key(raw_key) else {
            eprintln!(
                "live config: unknown key '{}' on line {}",
                raw_key.trim(),
                idx + 1
            );
            continue;
        };
        if LIVE_CONFIG_IGNORE_KEYS.contains(&env_key.as_str()) {
            continue;
        }
        out.insert(env_key, value);
    }
    out
}

fn apply_live_config_once(path: &Path) {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(_) => return,
    };
    let parsed = parse_live_config(&contents);
    for (key, value) in parsed {
        let existing = std::env::var(&key).ok();
        if existing
            .as_deref()
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
        {
            continue;
        }
        std::env::set_var(key, value);
    }
    sync_ort_dylib_env(false);
}

fn normalize_ort_dylib_value(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let path = PathBuf::from(trimmed);
    let resolved = if path.is_dir() {
        path.join("onnxruntime.dll")
    } else {
        path
    };
    Some(resolved.to_string_lossy().to_string())
}

fn sync_ort_dylib_env(force_clear: bool) {
    let raw = match std::env::var("CLIP_ORT_DYLIB") {
        Ok(value) => value,
        Err(_) => {
            if force_clear {
                std::env::remove_var("ORT_DYLIB_PATH");
            }
            return;
        }
    };
    if let Some(path) = normalize_ort_dylib_value(&raw) {
        std::env::set_var("ORT_DYLIB_PATH", path);
    } else if force_clear {
        std::env::remove_var("ORT_DYLIB_PATH");
    }
}

fn seed_ort_dylib_from_live_config(path: &Path) {
    let current = std::env::var("CLIP_ORT_DYLIB")
        .ok()
        .filter(|value| !value.trim().is_empty());
    if current.is_none() {
        if let Ok(contents) = fs::read_to_string(path) {
            let parsed = parse_live_config(&contents);
            if let Some(value) = parsed.get("CLIP_ORT_DYLIB") {
                if !value.trim().is_empty() {
                    std::env::set_var("CLIP_ORT_DYLIB", value);
                }
            }
        }
    }
    sync_ort_dylib_env(false);
    if let Ok(value) = std::env::var("ORT_DYLIB_PATH") {
        if !value.trim().is_empty() {
            eprintln!("ort: using dylib {}", value.trim());
        }
    }
}

fn ensure_live_config_file(path: &Path, overrides: &[(String, String)]) {
    let overrides: HashMap<String, String> = overrides
        .iter()
        .filter(|(key, _)| !LIVE_CONFIG_IGNORE_KEYS.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let defaults: HashMap<String, String> = LIVE_CONFIG_PREFILL
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();

    if let Some(parent) = path.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            eprintln!(
                "live config: failed to create {}: {err}",
                parent.display()
            );
            return;
        }
    }

    let mut original = String::new();
    let mut existing_values: HashMap<String, String> = HashMap::new();
    if path.exists() {
        match fs::read_to_string(path) {
            Ok(contents) => {
                original = contents.clone();
                for line in contents.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty()
                        || trimmed.starts_with('#')
                        || trimmed.starts_with("//")
                    {
                        continue;
                    }
                    let (raw_key, raw_value) = split_live_config_line(trimmed);
                    let Some(raw_key) = raw_key else { continue };
                    let Some(env_key) = normalize_live_config_key(raw_key) else { continue };
                    if !is_known_env_key(&env_key) {
                        continue;
                    }
                    let value = raw_value.unwrap_or_else(|| "".to_string());
                    existing_values.insert(env_key, value);
                }
            }
            Err(err) => {
                eprintln!(
                    "live config: failed to read {}: {err}",
                    path.display()
                );
            }
        }
    }

    let mut values: HashMap<String, String> = HashMap::new();
    for spec in ENV_SPECS {
        let key = spec.env.to_string();
        if let Some(value) = defaults.get(&key) {
            values.insert(key.clone(), value.clone());
        } else if let Ok(value) = std::env::var(&key) {
            if !value.trim().is_empty() {
                values.insert(key.clone(), value);
            }
        } else {
            values.insert(key.clone(), String::new());
        }
    }
    for (key, value) in existing_values {
        if overrides.contains_key(&key) {
            continue;
        }
        values.insert(key, value);
    }
    for (key, value) in overrides {
        values.insert(key, value);
    }

    let mut keys: Vec<String> = ENV_SPECS.iter().map(|spec| spec.env.to_string()).collect();
    keys.sort_by(|a, b| {
        let (ga, la) = live_config_group(a);
        let (gb, lb) = live_config_group(b);
        ga.cmp(&gb).then_with(|| la.cmp(&lb))
    });

    let mut lines: Vec<String> = Vec::new();
    lines.push("# AutoClip live config (hot-reload).".to_string());
    lines.push("# Edit values and save; changes apply while running.".to_string());
    lines.push("# Blank values are placeholders and are ignored until set.".to_string());

    let mut last_group: Option<usize> = None;
    for key in keys {
        let (group, label) = live_config_group(&key);
        if last_group != Some(group) {
            lines.push(String::new());
            lines.push(format!("# {label}"));
            last_group = Some(group);
        }
        let value = values.get(&key).cloned().unwrap_or_default();
        lines.push(format!("{key}={value}"));
    }

    let next_contents = lines.join("\n");
    if !path.exists() || next_contents != original {
        if let Err(err) = fs::write(path, next_contents) {
            eprintln!(
                "live config: failed to write {}: {err}",
                path.display()
            );
        } else {
            eprintln!("live config: synced {}", path.display());
        }
    }
}

fn split_live_config_line(line: &str) -> (Option<&str>, Option<String>) {
    if let Some((left, right)) = line.split_once('=') {
        return (Some(left.trim()), Some(normalize_live_config_value(right)));
    }
    if let Some((left, right)) = line.split_once(':') {
        return (Some(left.trim()), Some(normalize_live_config_value(right)));
    }

    let mut parts = line.split_whitespace();
    let key = parts.next();
    let rest: Vec<&str> = parts.collect();
    if rest.is_empty() {
        (key, None)
    } else {
        (key, Some(normalize_live_config_value(&rest.join(" "))))
    }
}

fn normalize_live_config_value(raw: &str) -> String {
    let trimmed = raw.trim();
    let stripped = trimmed
        .strip_prefix('\u{feff}')
        .unwrap_or(trimmed)
        .trim();
    if stripped.len() >= 2 {
        let bytes = stripped.as_bytes();
        if (bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\'')
        {
            return stripped[1..stripped.len() - 1].to_string();
        }
    }
    stripped.to_string()
}

fn normalize_live_config_key(raw: &str) -> Option<String> {
    let mut key = raw.trim();
    if key.is_empty() {
        return None;
    }
    if let Some(stripped) = key.strip_prefix("--") {
        key = stripped;
    }
    let candidate = key
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .replace(['-', '.', ' '], "_")
        .to_ascii_uppercase();
    if candidate.is_empty() {
        return None;
    }
    if is_known_env_key(&candidate) || candidate.starts_with("CLIP_") {
        return Some(candidate);
    }
    let prefixed = format!("CLIP_{candidate}");
    if is_known_env_key(&prefixed) {
        return Some(prefixed);
    }
    if candidate
        .chars()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
    {
        return Some(candidate);
    }
    None
}

fn is_known_env_key(key: &str) -> bool {
    ENV_SPECS.iter().any(|spec| spec.env == key)
}

fn live_config_group(key: &str) -> (usize, &'static str) {
    if key.starts_with("CLIP_") {
        return (0, "Clip");
    }
    if key.starts_with("WAKE_") {
        return (1, "Wake");
    }
    if key.starts_with("WHISPER_") {
        return (2, "Whisper");
    }
    if key.starts_with("GGML_") {
        return (3, "GGML");
    }
    if key.starts_with("FFMPEG_") {
        return (4, "FFmpeg");
    }
    if key.starts_with("M3U8_") {
        return (5, "M3U8");
    }
    if key.starts_with("HEADLESS_") {
        return (6, "Headless");
    }
    if key.starts_with("TWITCH_") {
        return (7, "Twitch");
    }
    if key.starts_with("KICK_") {
        return (8, "Kick");
    }
    if key.starts_with("TIKTOK_") {
        return (9, "TikTok");
    }
    if key.starts_with("COOKIE_") {
        return (10, "Cookies");
    }
    if key.starts_with("MIC_") {
        return (11, "Mic");
    }
    (12, "Other")
}

fn resolve_mic_opts(
    page_url: String,
    phrase: Option<String>,
    log_raw_wake: bool,
    mic_device: Option<String>,
) -> Result<MicOpts> {
    let mut mic_device = mic_device.or_else(|| std::env::var("MIC_DEVICE").ok());
    if mic_device.is_none() {
        mic_device = prompt_for_mic_device();
    }
    Ok(MicOpts {
        page_url,
        phrase,
        log_raw_wake,
        mic_device,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help(args.get(0).map(String::as_str).unwrap_or("autoclip"));
        return Ok(());
    }

    let ParsedCli {
        positionals,
        override_phrase,
        log_raw_wake,
        log_raw_wake_set,
        mic_device,
        stream_urls,
        streams_file,
        env_overrides,
    } = parse_cli_args(&args[1..])?;
    let stream_urls = stream_urls
        .into_iter()
        .filter_map(|url| normalize_stream_url(&url))
        .collect::<Vec<String>>();
    let single_mode_hint = streams_file.is_none() && stream_urls.len() <= 1;
    let mut env_overrides = env_overrides;
    if let Some(phrase_list) = override_phrase.as_deref().filter(|v| !v.trim().is_empty()) {
        env_overrides.retain(|(key, _)| key != "CLIP_WAKE_WORDS");
        env_overrides.push(("CLIP_WAKE_WORDS".to_string(), phrase_list.to_string()));
    }
    let mut streams_path = streams_file.map(PathBuf::from);
    if let Some(path) = streams_path.as_ref() {
        let path_str = path.to_string_lossy().to_string();
        env_overrides.retain(|(key, _)| key != "CLIP_STREAMS_FILE");
        env_overrides.push(("CLIP_STREAMS_FILE".to_string(), path_str));
    }
    let wake_phrases_override = override_phrase
        .as_deref()
        .map(split_wake_phrases)
        .filter(|v| !v.is_empty());
    let live_config_path = match std::env::var("CLIP_LIVE_CONFIG") {
        Ok(path) if !path.trim().is_empty() => PathBuf::from(path.trim()),
        _ => {
            let cwd = std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."));
            let path = cwd.join("config.env");
            std::env::set_var("CLIP_LIVE_CONFIG", path.to_string_lossy().as_ref());
            path
        }
    };
    ensure_live_config_file(&live_config_path, &env_overrides);
    apply_live_config_once(&live_config_path);
    let single_mode = single_mode_hint && single_mode_from_env();
    let single_instance_requested = single_instance_enabled(single_mode);
    if streams_path.is_none() {
        if !stream_urls.is_empty() {
            streams_path = Some(default_streams_file_path());
        } else if let Ok(path) = std::env::var("CLIP_STREAMS_FILE") {
            let trimmed = path.trim();
            if !trimmed.is_empty() {
                streams_path = Some(PathBuf::from(trimmed));
            }
        }
    }
    if let Some(path) = streams_path.as_ref() {
        let path_str = path.to_string_lossy().to_string();
        env_overrides.retain(|(key, _)| key != "CLIP_STREAMS_FILE");
        env_overrides.push(("CLIP_STREAMS_FILE".to_string(), path_str));
    }
    apply_env_overrides(&env_overrides);
    seed_ort_dylib_from_live_config(&live_config_path);
    refresh_low_resource_state();
    ensure_cuda_device_order();

    // Ensure GPU backend is preferred when available; avoid falling back to CPU due to missing env.
    if std::env::var("WHISPER_GPU").is_err() {
        std::env::set_var("WHISPER_GPU", "1");
    }

    auto_assign_gpus_for_tools();
    log_gpu_assignments();
    check_llm_health().await;
    let page_url_arg = positionals.get(0).map(|s| s.as_str());

    if let Some(cmd) = page_url_arg {
        if cmd.eq_ignore_ascii_case("demo-buffer") {
            return run_buffer_demo().await;
        }
        if cmd.eq_ignore_ascii_case("demo-hls-buffer") {
            let page = positionals
                .get(1)
                .cloned()
                .or_else(|| std::env::var("CLIP_PAGE_URL").ok())
                .unwrap_or_default();
            if page.is_empty() {
                eprintln!("usage: autoclip demo-hls-buffer <page_url>  (or set CLIP_PAGE_URL)");
                return Ok(());
            }
            let page = normalize_page_url(&page);
            return run_hls_buffer_demo(&page).await;
        }
        if cmd.eq_ignore_ascii_case("demo-ts") || cmd.eq_ignore_ascii_case("demo-file") {
            let Some(path) = positionals.get(1) else {
                eprintln!("usage: autoclip demo-ts <ts_path> [--phrase WORDS] [--no-log-raw-wake]");
                return Ok(());
            };
            return run_ts_wake_demo(path, override_phrase.clone(), log_raw_wake).await;
        }
        if cmd.eq_ignore_ascii_case("reprocess-ts") {
            let Some(path) = positionals.get(1) else {
                eprintln!("usage: autoclip reprocess-ts <ts_path>");
                return Ok(());
            };
            return run_reprocess_ts(path).await;
        }
        if cmd.eq_ignore_ascii_case("kick-vod") || cmd.eq_ignore_ascii_case("kick-latest-vod") {
            let Some(page_url) = positionals.get(1) else {
                eprintln!("usage: autoclip kick-vod <kick_channel_url>");
                return Ok(());
            };
            let page_url = normalize_page_url(page_url);
            return run_kick_latest_vod(&page_url).await;
        }
        if cmd.eq_ignore_ascii_case("check-gameplay-model") {
            let model_dir = positionals.get(1).map(|s| s.as_str());
            return run_check_gameplay_model(model_dir);
        }
        if cmd.eq_ignore_ascii_case("demo-detect") || cmd.eq_ignore_ascii_case("demo-face") {
            let Some(path) = positionals.get(1) else {
                eprintln!("usage: autoclip demo-detect <media_path>");
                return Ok(());
            };
            return run_clip_detect_demo(path).await;
        }
        if cmd.eq_ignore_ascii_case("face-sweep") {
            let Some(pos_dir) = positionals.get(1) else {
                eprintln!("usage: autoclip face-sweep <positives_dir> <negatives_dir> [score_start score_end score_step] [out_csv]");
                return Ok(());
            };
            let Some(neg_dir) = positionals.get(2) else {
                eprintln!("usage: autoclip face-sweep <positives_dir> <negatives_dir> [score_start score_end score_step] [out_csv]");
                return Ok(());
            };
            let extra = if positionals.len() > 3 {
                &positionals[3..]
            } else {
                &[]
            };
            return run_face_sweep(pos_dir, neg_dir, extra).await;
        }
        if cmd.eq_ignore_ascii_case("demo-wakeword-mic") {
            let Some(page_url) = positionals.get(1) else {
                eprintln!("usage: autoclip demo-wakeword-mic <page_url> [--phrase WORDS] [--log-raw-wake] [--mic-device DEVICE]");
                return Ok(());
            };
            let page_url = normalize_page_url(page_url);
            let demo_log_raw = if log_raw_wake_set { log_raw_wake } else { false };
            let opts = resolve_mic_opts(
                page_url,
                override_phrase.clone(),
                demo_log_raw,
                mic_device.clone(),
            )?;
            return run_wakeword_mic_demo(opts).await;
        }
        if cmd.eq_ignore_ascii_case("whisper-worker") {
            return run_whisper_worker().await;
        }
    }

    let _single_instance_guard = if single_instance_requested {
        let name = "Local\\AutoClipSingleInstance";
        let guard = single_instance::acquire(name)?;
        eprintln!("single instance lock acquired ({name})");
        Some(guard)
    } else {
        None
    };

    let face_id_page = page_url_arg
        .map(|s| s.to_string())
        .or_else(|| std::env::var("CLIP_PAGE_URL").ok())
        .filter(|s| !s.trim().is_empty())
        .map(|s| normalize_page_url(&s));
    if let Some(page_url) = face_id_page.as_deref() {
        if streams_path.is_none() {
            maybe_auto_enroll_face_id(page_url).await;
        } else if face_id_enabled() {
            eprintln!("face id: auto-enroll disabled for multi-stream runs");
        }
    }

    let mut config = Config::example();
    if let Some(url) = page_url_arg {
        config.kick_url = normalize_page_url(url);
    }
    if let Some(p) = override_phrase {
        config.activation_phrase = p;
    }
    config.wake_phrases = wake_phrases_override;
    config.log_raw_wake = log_raw_wake;

    if let Some(path) = streams_path {
        if !stream_urls.is_empty() {
            sync_streams_file(&path, &stream_urls)?;
        }
        return run_streams_supervisor(config, path).await;
    }

    if page_url_arg.is_none() && std::env::var("CLIP_PAGE_URL").is_err() {
        eprintln!("usage: autoclip <page_url>  (or set CLIP_PAGE_URL) | autoclip demo-buffer | autoclip demo-hls-buffer <page_url> | autoclip demo-ts <ts_path> | autoclip reprocess-ts <ts_path> | autoclip kick-vod <kick_channel_url> | autoclip check-gameplay-model [model_dir] | autoclip demo-wakeword-mic <page_url> [--phrase WORDS] [--no-log-raw-wake] | autoclip face-sweep <positives_dir> <negatives_dir> [score_start score_end score_step] [out_csv]");
        return Ok(());
    }

    let app = AutoClip::new(config);
    app.run_with_page(page_url_arg).await
}

/// Demonstrate the rolling buffer by pushing synthetic 400ms chunks and printing state.
async fn run_buffer_demo() -> Result<()> {
    use tokio::time::interval;

    let mut buf = RollingBuffer::new(Duration::from_secs(3));
    let mut ticker = interval(Duration::from_millis(400));

    println!("demo: 3s rolling buffer; pushing 400ms chunks every 400ms");
    for i in 0..12 {
        ticker.tick().await;
        let label = format!("f{i}");
        buf.push(label.as_bytes().to_vec(), Duration::from_millis(400));

        let secs = buf.total_duration().as_secs_f32();
        println!(
            "push {:<3} | chunks: {:2} | duration: {:4.1}s | bytes: {:3}",
            label,
            buf.chunk_count(),
            secs,
            buf.total_bytes()
        );
    }

    let snapshot = buf.snapshot_bytes();
    println!(
        "snapshot ({} bytes): {}",
        snapshot.len(),
        String::from_utf8_lossy(&snapshot)
    );

    Ok(())
}

/// Demonstrate HLS ingestion into the rolling buffer: fetch page -> master -> best variant,
/// pull a handful of segments, push into a duration-capped buffer, and log eviction behavior.
async fn run_hls_buffer_demo(page_url: &str) -> Result<()> {
    let config = Config::example();
    let mut buffer = RollingBuffer::new(Duration::from_secs(config.before_buffer_length as u64));

    let hls = HlsClient::new()?;
    println!("discovering m3u8 via headless for {}", page_url);
    let (master_url, master) = hls.fetch_master_from_page(page_url).await?;
    let media_url = hls.highest_variant_url(&master_url, &master)?;
    let media_url = media_url; // keep owned Url

    println!("best variant: {}", media_url);
    let playlist = hls.fetch_media(media_url.as_str()).await?;
    println!("loaded media playlist with {} segments", playlist.segments.len());

    // Limit to a handful of segments to keep the demo quick.
    let take_n = usize::min(8, playlist.segments.len());
    for (idx, seg) in playlist.segments.iter().take(take_n).enumerate() {
        let seg_dur_playlist = Duration::from_secs_f32(seg.duration as f32);
        let bytes = hls.fetch_segment_from_playlist(&media_url, &seg.uri).await?;
        let seg_dur = choose_segment_duration(
            pts_duration_from_ts(&bytes),
            seg_dur_playlist,
        );
        buffer.push(bytes, seg_dur);

        println!(
            "seg {:02} | uri: {:30} | dur: {:4.2}s | chunks: {:2} | buffered: {:4.2}s | bytes: {}",
            idx,
            truncate_str(&seg.uri, 30),
            seg_dur.as_secs_f32(),
            buffer.chunk_count(),
            buffer.total_duration().as_secs_f32(),
            buffer.total_bytes()
        );
    }

    println!(
        "demo complete: retained {} chunks covering {:4.2}s ({} bytes)",
        buffer.chunk_count(),
        buffer.total_duration().as_secs_f32(),
        buffer.total_bytes()
    );

    // Persist snapshot to a TS file and transcode to vertical MP4 via FFmpeg to mirror the main flow.
    let snapshot = buffer.snapshot_bytes();
    let save_dir = Path::new(&config.save_path);
    fs::create_dir_all(save_dir).context("creating save dir for buffer snapshot")?;

    let ts_path = save_dir.join("buffer_snapshot.ts");
    fs::write(&ts_path, &snapshot).context("writing buffer snapshot to TS file")?;

    let output_path = next_output_path(&config.save_path, &config.file_name_stub)?;
    let (out_w, out_h) = parse_resolution(&config.resolution).unwrap_or((1080, 1920));
    let clip_len = buffer.total_duration();

    println!(
        "ffmpeg: encoding {}s from {} -> {}",
        clip_len.as_secs_f32(),
        ts_path.display(),
        output_path.display()
    );

    run_ffmpeg_from_file(&ts_path, &output_path, out_w, out_h, clip_len, None, None).await?;
    println!("wrote clipped video to {}", output_path.display());

    Ok(())
}

/// Listen on the microphone via SAPI; when the phrase is recognized, clip 30s from the streamer page.
async fn run_wakeword_mic_demo(opts: MicOpts) -> Result<()> {
    let mut cfg = Config::example();
    cfg.kick_url = opts.page_url.clone();
    if let Some(p) = opts.phrase.clone() {
        cfg.activation_phrase = p;
        let parsed = split_wake_phrases(&cfg.activation_phrase);
        if !parsed.is_empty() {
            cfg.wake_phrases = Some(parsed);
        }
    }
    cfg.use_mic_for_wake = true;
    cfg.log_raw_wake = opts.log_raw_wake;
    cfg.mic_device = opts.mic_device.clone();
    let app = AutoClip::new(cfg);
    app.run_until_wake_and_clip(&opts.page_url).await
}

async fn run_whisper_worker() -> Result<()> {
    let mode = std::env::var("WHISPER_WORKER_MODE")
        .unwrap_or_else(|_| "stream".to_string())
        .to_ascii_lowercase();
    let status_path = whisper_worker_status_path("");
    let poll = whisper_worker_status_poll();
    let log_raw = std::env::var("WHISPER_WORKER_LOG_RAW")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false);
    let phrase_env = std::env::var("CLIP_WAKE_WORDS").unwrap_or_else(|_| "orange".to_string());
    let mut wake_phrases = split_wake_phrases(&phrase_env);
    if wake_phrases.is_empty() {
        wake_phrases.push("orange".to_string());
    }
    let model_path = stream_audio_wake::select_best_model_path();

    let stop = Arc::new(AtomicBool::new(false));
    spawn_ctrl_c_handler(stop.clone(), None);
    let fired = Arc::new(AtomicBool::new(false));
    let detect_ns = Arc::new(AtomicU64::new(u64::MAX));
    let audio_ns = Arc::new(AtomicU64::new(0));
    let last_word_ns = Arc::new(AtomicU64::new(u64::MAX));
    let _writer = spawn_wake_status_writer(
        status_path.clone(),
        poll,
        stop.clone(),
        fired.clone(),
        detect_ns.clone(),
        audio_ns.clone(),
        last_word_ns.clone(),
    );

    let mode_for_worker = mode.clone();
    let model_path_for_worker = model_path.clone();
    let wake_phrases_for_worker = wake_phrases.clone();
    let stop_for_worker = stop.clone();
    let fired_for_worker = fired.clone();
    let detect_ns_for_worker = detect_ns.clone();
    let audio_ns_for_worker = audio_ns.clone();
    let last_word_ns_for_worker = last_word_ns.clone();
    let worker_handle = tokio::task::spawn_blocking(move || {
        let start_instant = Instant::now();
        if mode_for_worker == "mic" {
            let mic_device = std::env::var("MIC_DEVICE").ok();
            run_wake_worker_mic(
                mic_device.as_deref(),
                Path::new(&model_path_for_worker),
                &wake_phrases_for_worker,
                log_raw,
                stop_for_worker,
                fired_for_worker,
                start_instant,
                detect_ns_for_worker,
                audio_ns_for_worker,
                last_word_ns_for_worker,
            )?;
        } else {
            let media_url_path = whisper_worker_media_url_path("");
            run_wake_worker_stream(
                &media_url_path,
                Path::new(&model_path_for_worker),
                &wake_phrases_for_worker,
                log_raw,
                stop_for_worker,
                fired_for_worker,
                start_instant,
                detect_ns_for_worker,
                audio_ns_for_worker,
                last_word_ns_for_worker,
            )?;
        }
        Ok(())
    });

    let result = worker_handle
        .await
        .context("whisper worker task join failed")?;
    stop.store(true, Ordering::Relaxed);
    result
}

/// Replay a local TS file to detect a wake phrase and cut a stacked clip.
async fn run_ts_wake_demo(
    ts_path: &str,
    phrase: Option<String>,
    log_raw_wake: bool,
) -> Result<()> {
    let path = Path::new(ts_path);
    if !path.exists() {
        anyhow::bail!("TS file not found: {}", path.display());
    }

    let mut cfg = Config::example();
    let override_phrases = phrase.clone().map(|p| {
        cfg.activation_phrase = p.clone();
        split_wake_phrases(&p)
    });
    cfg.log_raw_wake = log_raw_wake;

    let model_path = stream_audio_wake::select_best_model_path();
    let ts_path_buf = path.to_path_buf();
    let wake_phrases = resolve_wake_phrases(override_phrases.as_deref(), &cfg.activation_phrase);
    let wake_phrases_for_detect = wake_phrases.clone();
    let detect = tokio::task::spawn_blocking(move || {
        detect_wake_in_file(
            &ts_path_buf,
            Path::new(&model_path),
            &wake_phrases_for_detect,
            log_raw_wake,
        )
    })
    .await??;

    let Some(detect_secs) = detect else {
        let label = if wake_phrases.is_empty() {
            cfg.activation_phrase.clone()
        } else {
            wake_phrases.join(", ")
        };
        eprintln!("wake phrase(s) '{}' not detected in file", label);
        return Ok(());
    };

    if std::env::var("CLIP_LAYOUT").is_err() {
        std::env::set_var("CLIP_LAYOUT", "stacked");
    }

    let before = Duration::from_secs(cfg.before_buffer_length as u64);
    let after = Duration::from_secs(cfg.after_buffer_length as u64);
    let clip_len = before + after;
    let start_secs = detect_secs - before.as_secs_f32();
    let start_offset = if start_secs > 0.0 {
        Some(start_secs)
    } else {
        None
    };

    let output_path = next_output_path(&cfg.save_path, &cfg.file_name_stub)?;
    let (out_w, out_h) = parse_resolution(&cfg.resolution).unwrap_or((1080, 1920));
    run_ffmpeg_from_file(path, &output_path, out_w, out_h, clip_len, start_offset, None).await?;
    println!("wrote clipped video to {}", output_path.display());

    Ok(())
}

/// Reprocess a TS snapshot into a new MP4 using the current layout settings.
async fn run_reprocess_ts(ts_path: &str) -> Result<()> {
    let cfg = Config::example();
    run_reprocess_ts_with_config(ts_path, &cfg).await
}

async fn run_reprocess_ts_with_config(ts_path: &str, cfg: &Config) -> Result<()> {
    let path = Path::new(ts_path);
    if !path.exists() {
        anyhow::bail!("media file not found: {}", path.display());
    }

    if face_id_enabled() {
        let face_id_page = std::env::var("CLIP_PAGE_URL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .map(|v| normalize_page_url(&v));
        if let Some(page_url) = face_id_page.as_deref() {
            maybe_auto_enroll_face_id(page_url).await;
            maybe_enroll_face_id_from_media(page_url, path).await;
        }
    }

    if let Some(mut live) = LiveConfigState::from_env() {
        live.maybe_refresh();
    }
    refresh_low_resource_state();

    let reprocess_fast = reprocess_fast_enabled();
    if reprocess_fast {
        std::env::set_var("CLIP_LLM_ENABLE", "0");
        std::env::set_var("CLIP_AUDIO_NORM", "0");
        std::env::set_var("CLIP_EMOTION_ENABLE", "0");
        std::env::set_var("CLIP_EMOTION_AUDIO", "0");
        std::env::set_var("CLIP_EMOTION_FACE", "0");
        eprintln!("reprocess: fast mode enabled (LLM/audio norm disabled)");
    }
    // Always enable captions during reprocess runs.
    std::env::set_var("CLIP_CAPTIONS", "1");
    std::env::set_var("CLIP_CLOSED_CAPTIONS", "1");

    let (out_w, out_h) = parse_resolution(&cfg.resolution).unwrap_or((1080, 1920));
    let progress_total_secs = probe_media_duration_secs(path).await;
    if std::env::var("CLIP_DETECT_STEP").is_err() {
        std::env::set_var("CLIP_DETECT_STEP", "10");
    }
    if std::env::var("CLIP_DETECT_START").is_err() {
        std::env::set_var("CLIP_DETECT_START", "5");
    }
    if std::env::var("CLIP_DETECT_BUDGET_SECS").is_err() {
        std::env::set_var("CLIP_DETECT_BUDGET_SECS", "30");
    }
    if std::env::var("CLIP_FACE_TRACK").is_err() {
        std::env::set_var("CLIP_FACE_TRACK", "1");
    }
    let chunk_secs = read_reprocess_chunk_secs();
    let force_ts_input = is_ts_input(path);
    let continue_on_error = reprocess_continue_on_error();
    let input = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-utf8 input path"))?;

    if let (Some(total_secs), Some(chunk_secs)) = (progress_total_secs, chunk_secs) {
        if total_secs > chunk_secs {
            let resume_path = reprocess_resume_path(&cfg.save_path, path, chunk_secs);
            let subchunk_secs = read_reprocess_subchunk_secs();
            let mut start = 0.0f32;
            let mut idx = 1u32;
            let mut resume_used = false;
            let mut resume_segment: Option<(u32, f32, f32, PathBuf, u32)> = None;
            if let Some(state) = read_reprocess_resume_state(&resume_path) {
                let (size, mtime_ms) = input_fingerprint(path).unwrap_or((0, 0));
                let chunk_match =
                    (state.chunk_secs - chunk_secs).abs() <= 0.001 || state.chunk_secs == 0.0;
                let total_match = match (state.total_secs, Some(total_secs)) {
                    (Some(a), Some(b)) => (a - b).abs() <= 0.5,
                    (None, _) => true,
                    _ => false,
                };
                let subchunk_match = match (state.subchunk_secs, subchunk_secs) {
                    (Some(a), Some(b)) => (a - b).abs() <= 0.01,
                    (None, None) => true,
                    _ => false,
                };
                if state.input_path == input
                    && chunk_match
                    && total_match
                    && subchunk_match
                    && state.input_size == size
                    && state.input_mtime_ms == mtime_ms
                    && state.next_start_secs >= 0.0
                {
                    start = state.next_start_secs;
                    idx = state.next_index.max(1);
                    resume_used = true;
                    if let (Some(seg_idx), Some(seg_start), Some(seg_dur), Some(out), Some(part)) = (
                        state.segment_index,
                        state.segment_start_secs,
                        state.segment_duration_secs,
                        state.segment_output_path.as_ref(),
                        state.next_part_index,
                    ) {
                        if seg_idx == idx
                            && start >= seg_start
                            && start < seg_start + seg_dur - 0.01
                        {
                            resume_segment =
                                Some((seg_idx, seg_start, seg_dur, PathBuf::from(out), part));
                        }
                    }
                }
            }
            let segments = (total_secs / chunk_secs).ceil().max(1.0) as u32;
            if resume_used {
                eprintln!(
                    "reprocess: resuming at segment {}/{} start {:.1}s (chunk {:.1}s)",
                    idx, segments, start, chunk_secs
                );
                if start >= total_secs - 0.01 {
                    clear_reprocess_resume_state(&resume_path);
                    eprintln!("reprocess: resume indicates completion; nothing to do");
                    return Ok(());
                }
            }
            while start < total_secs {
                let remaining = (total_secs - start).max(0.0);
                if remaining <= 0.01 {
                    break;
                }
                let (segment_start, duration, output_path, mut part_index) =
                    if let Some((_seg_idx, seg_start, seg_dur, out, part)) =
                        resume_segment.take()
                    {
                        (
                            seg_start,
                            seg_dur,
                            out,
                            part.max(0).min(10_000),
                        )
                    } else {
                        let output = next_output_path(&cfg.save_path, &cfg.file_name_stub)?;
                        (start, remaining.min(chunk_secs), output, 0)
                    };
                let segment_end = segment_start + duration;
                let mut part_start = start.max(segment_start);
                let use_subchunk = subchunk_secs
                    .filter(|v| *v > 0.0 && *v + 0.01 < duration)
                    .unwrap_or(0.0);
                let (size, mtime_ms) = input_fingerprint(path).unwrap_or((0, 0));
                let mut segment_failed = false;
                if use_subchunk > 0.0 {
                    let seg_dir = reprocess_segment_dir(
                        &cfg.save_path,
                        path,
                        chunk_secs,
                        idx,
                        &output_path,
                    );
                    if part_index == 0 && seg_dir.exists() {
                        let _ = fs::remove_dir_all(&seg_dir);
                    }
                    fs::create_dir_all(&seg_dir)?;
                    if part_index == 0 && (part_start - segment_start).abs() > 0.01 {
                        part_index =
                            ((part_start - segment_start) / use_subchunk).floor().max(0.0) as u32;
                    }
                    while part_start < segment_end - 0.01 {
                        let part_duration = (segment_end - part_start).min(use_subchunk);
                        let part_path = seg_dir.join(format!("part_{part_index:03}.mp4"));
                        let resume_state = ReprocessResumeState {
                            input_path: input.to_string(),
                            input_size: size,
                            input_mtime_ms: mtime_ms,
                            chunk_secs,
                            subchunk_secs: Some(use_subchunk),
                            total_secs: Some(total_secs),
                            next_start_secs: part_start,
                            next_index: idx,
                            segment_index: Some(idx),
                            segment_start_secs: Some(segment_start),
                            segment_duration_secs: Some(duration),
                            segment_output_path: Some(output_path.to_string_lossy().to_string()),
                            next_part_index: Some(part_index),
                            updated_ms: now_unix_ms(),
                        };
                        let _ = write_reprocess_resume_state(&resume_path, &resume_state);
                        eprintln!(
                            "reprocess: segment {}/{} part {} start {:.1}s duration {:.1}s -> {}",
                            idx,
                            segments,
                            part_index + 1,
                            part_start,
                            part_duration,
                            part_path.display()
                        );
                        let mut stall_attempts = 0u32;
                        let max_stall_retries = 1u32;
                        let mut error_attempts = 0u32;
                        let max_error_retries = 1u32;
                        loop {
                        let result = run_ffmpeg_internal(
                            FfmpegRenderSpec::new(input, &part_path, out_w, out_h)
                                .with_duration(Some(part_duration))
                                .with_start_offset(Some(part_start))
                                .with_force_ts_input(force_ts_input)
                                .with_fast_seek(true)
                                .with_progress(Some(ProgressSpec {
                                    total_secs: Some(part_duration),
                                }))
                                .with_live_fast(false)
                                .with_fast_preset(reprocess_fast),
                            )
                            .await;
                            match result {
                                Ok(()) => break,
                                Err(err) => {
                                    let msg = err.to_string();
                                    if ffmpeg_error_indicates_stall(&msg) {
                                        if stall_attempts < max_stall_retries {
                                            stall_attempts += 1;
                                            eprintln!(
                                                "reprocess: ffmpeg stalled; retrying segment {}/{} part {} (retry {}/{})",
                                                idx,
                                                segments,
                                                part_index + 1,
                                                stall_attempts,
                                                max_stall_retries
                                            );
                                            continue;
                                        }
                                        eprintln!(
                                            "reprocess: ffmpeg stalled; leaving resume at segment {}/{} start {:.1}s (rerun to continue)",
                                            idx,
                                            segments,
                                            part_start
                                        );
                                        return Ok(());
                                    }
                                    if error_attempts < max_error_retries {
                                        error_attempts += 1;
                                        eprintln!(
                                            "reprocess: ffmpeg failed; retrying segment {}/{} part {} (retry {}/{})",
                                            idx,
                                            segments,
                                            part_index + 1,
                                            error_attempts,
                                            max_error_retries
                                        );
                                        continue;
                                    }
                                    if continue_on_error {
                                        eprintln!(
                                            "reprocess: ffmpeg failed for segment {}/{} part {}; skipping segment: {err:#}",
                                            idx,
                                            segments,
                                            part_index + 1
                                        );
                                        segment_failed = true;
                                        break;
                                    }
                                    return Err(err);
                                }
                            }
                        }
                        if segment_failed {
                            break;
                        }
                        part_index += 1;
                        part_start += part_duration;
                        let resume_state = ReprocessResumeState {
                            input_path: input.to_string(),
                            input_size: size,
                            input_mtime_ms: mtime_ms,
                            chunk_secs,
                            subchunk_secs: Some(use_subchunk),
                            total_secs: Some(total_secs),
                            next_start_secs: part_start,
                            next_index: idx,
                            segment_index: Some(idx),
                            segment_start_secs: Some(segment_start),
                            segment_duration_secs: Some(duration),
                            segment_output_path: Some(output_path.to_string_lossy().to_string()),
                            next_part_index: Some(part_index),
                            updated_ms: now_unix_ms(),
                        };
                        let _ = write_reprocess_resume_state(&resume_path, &resume_state);
                    }
                    if segment_failed {
                        let next_start = segment_end;
                        let resume_state = ReprocessResumeState {
                            input_path: input.to_string(),
                            input_size: size,
                            input_mtime_ms: mtime_ms,
                            chunk_secs,
                            subchunk_secs: Some(use_subchunk),
                            total_secs: Some(total_secs),
                            next_start_secs: next_start,
                            next_index: idx + 1,
                            segment_index: None,
                            segment_start_secs: None,
                            segment_duration_secs: None,
                            segment_output_path: None,
                            next_part_index: None,
                            updated_ms: now_unix_ms(),
                        };
                        let _ = write_reprocess_resume_state(&resume_path, &resume_state);
                        let _ = fs::remove_dir_all(&seg_dir);
                        eprintln!(
                            "reprocess: skipping segment {}/{} after ffmpeg error",
                            idx, segments
                        );
                        start = next_start;
                        idx += 1;
                        continue;
                    }
                    let parts = collect_part_files(&seg_dir)?;
                    match concat_media_parts(&parts, &output_path).await {
                        Ok(()) => {}
                        Err(err) => {
                            if continue_on_error {
                                let next_start = segment_end;
                                let resume_state = ReprocessResumeState {
                                    input_path: input.to_string(),
                                    input_size: size,
                                    input_mtime_ms: mtime_ms,
                                    chunk_secs,
                                    subchunk_secs: Some(use_subchunk),
                                    total_secs: Some(total_secs),
                                    next_start_secs: next_start,
                                    next_index: idx + 1,
                                    segment_index: None,
                                    segment_start_secs: None,
                                    segment_duration_secs: None,
                                    segment_output_path: None,
                                    next_part_index: None,
                                    updated_ms: now_unix_ms(),
                                };
                                let _ = write_reprocess_resume_state(&resume_path, &resume_state);
                                let _ = fs::remove_dir_all(&seg_dir);
                                eprintln!(
                                    "reprocess: concat failed for segment {}/{}; skipping: {err:#}",
                                    idx, segments
                                );
                                start = next_start;
                                idx += 1;
                                continue;
                            }
                            return Err(err);
                        }
                    }
                    let _ = fs::remove_dir_all(&seg_dir);
                } else {
                    let resume_state = ReprocessResumeState {
                        input_path: input.to_string(),
                        input_size: size,
                        input_mtime_ms: mtime_ms,
                        chunk_secs,
                        subchunk_secs,
                        total_secs: Some(total_secs),
                        next_start_secs: segment_start,
                        next_index: idx,
                        segment_index: None,
                        segment_start_secs: None,
                        segment_duration_secs: None,
                        segment_output_path: None,
                        next_part_index: None,
                        updated_ms: now_unix_ms(),
                    };
                    if let Err(err) = write_reprocess_resume_state(&resume_path, &resume_state) {
                        eprintln!(
                            "reprocess: failed to persist resume state {}: {err:#}",
                            resume_path.display()
                        );
                    }
                    eprintln!(
                        "reprocess: segment {}/{} start {:.1}s duration {:.1}s -> {}",
                        idx,
                        segments,
                        segment_start,
                        duration,
                        output_path.display()
                    );
                    let mut stall_attempts = 0u32;
                    let max_stall_retries = 1u32;
                    let mut error_attempts = 0u32;
                    let max_error_retries = 1u32;
                    loop {
                        let result = run_ffmpeg_internal(
                            FfmpegRenderSpec::new(input, &output_path, out_w, out_h)
                                .with_duration(Some(duration))
                                .with_start_offset(Some(segment_start))
                                .with_force_ts_input(force_ts_input)
                                .with_fast_seek(true)
                                .with_progress(Some(ProgressSpec {
                                    total_secs: Some(duration),
                                }))
                                .with_live_fast(false)
                                .with_fast_preset(reprocess_fast),
                        )
                        .await;
                        match result {
                            Ok(()) => break,
                            Err(err) => {
                                let msg = err.to_string();
                                if ffmpeg_error_indicates_stall(&msg) {
                                    if stall_attempts < max_stall_retries {
                                        stall_attempts += 1;
                                        eprintln!(
                                            "reprocess: ffmpeg stalled; retrying segment {}/{} (retry {}/{})",
                                            idx,
                                            segments,
                                            stall_attempts,
                                            max_stall_retries
                                        );
                                        continue;
                                    }
                                    eprintln!(
                                        "reprocess: ffmpeg stalled; leaving resume at segment {}/{} start {:.1}s (rerun to continue)",
                                        idx,
                                        segments,
                                        segment_start
                                    );
                                    return Ok(());
                                }
                                if error_attempts < max_error_retries {
                                    error_attempts += 1;
                                    eprintln!(
                                        "reprocess: ffmpeg failed; retrying segment {}/{} (retry {}/{})",
                                        idx,
                                        segments,
                                        error_attempts,
                                        max_error_retries
                                    );
                                    continue;
                                }
                                if continue_on_error {
                                    eprintln!(
                                        "reprocess: ffmpeg failed for segment {}/{}; skipping: {err:#}",
                                        idx, segments
                                    );
                                    segment_failed = true;
                                    break;
                                }
                                return Err(err);
                            }
                        }
                    }
                    if segment_failed {
                        let next_start = segment_end;
                        let resume_state = ReprocessResumeState {
                            input_path: input.to_string(),
                            input_size: size,
                            input_mtime_ms: mtime_ms,
                            chunk_secs,
                            subchunk_secs,
                            total_secs: Some(total_secs),
                            next_start_secs: next_start,
                            next_index: idx + 1,
                            segment_index: None,
                            segment_start_secs: None,
                            segment_duration_secs: None,
                            segment_output_path: None,
                            next_part_index: None,
                            updated_ms: now_unix_ms(),
                        };
                        let _ = write_reprocess_resume_state(&resume_path, &resume_state);
                        start = next_start;
                        idx += 1;
                        continue;
                    }
                }
                let next_start = segment_start + duration;
                let resume_state = ReprocessResumeState {
                    input_path: input.to_string(),
                    input_size: size,
                    input_mtime_ms: mtime_ms,
                    chunk_secs,
                    subchunk_secs,
                    total_secs: Some(total_secs),
                    next_start_secs: next_start,
                    next_index: idx + 1,
                    segment_index: None,
                    segment_start_secs: None,
                    segment_duration_secs: None,
                    segment_output_path: None,
                    next_part_index: None,
                    updated_ms: now_unix_ms(),
                };
                let _ = write_reprocess_resume_state(&resume_path, &resume_state);
                println!("reprocessed segment {idx} -> {}", output_path.display());
                start = next_start;
                idx += 1;
            }
            clear_reprocess_resume_state(&resume_path);
            return Ok(());
        }
    }

    let output_path = next_output_path(&cfg.save_path, &cfg.file_name_stub)?;
    if let Err(err) = run_ffmpeg_internal(
        FfmpegRenderSpec::new(input, &output_path, out_w, out_h)
            .with_duration(progress_total_secs)
            .with_force_ts_input(force_ts_input)
            .with_fast_seek(true)
            .with_progress(Some(ProgressSpec {
                total_secs: progress_total_secs,
            }))
            .with_live_fast(false)
            .with_fast_preset(reprocess_fast),
    )
    .await
    {
        if continue_on_error {
            eprintln!("reprocess: ffmpeg failed; skipping output: {err:#}");
            return Ok(());
        }
        return Err(err);
    }
    println!("reprocessed media clip -> {}", output_path.display());
    Ok(())
}

fn is_ts_input(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "ts" | "m2ts"))
        .unwrap_or(false)
}

fn ffmpeg_error_indicates_stall(message: &str) -> bool {
    message.contains("ffmpeg progress stalled") || message.contains("ffmpeg encode timed out")
}

fn ffmpeg_error_indicates_caption_issue(message: &str) -> bool {
    let msg = message.to_ascii_lowercase();
    msg.contains("fontconfig error")
        || msg.contains("cannot load default config file")
        || msg.contains("libass")
        || msg.contains("drawtext")
}

fn run_check_gameplay_model(model_dir: Option<&str>) -> Result<()> {
    if let Some(dir) = model_dir {
        std::env::set_var("CLIP_GAMEPLAY_MODEL_DIR", dir);
    }

    let config = read_clip_gameplay_config(960, 540);
    if !config.enabled {
        anyhow::bail!("gameplay model check: CLIP_GAMEPLAY is disabled");
    }
    if !config.text_model_path.exists() {
        anyhow::bail!(
            "gameplay model check: text model not found at {}",
            config.text_model_path.display()
        );
    }
    if !config.vision_model_path.exists() {
        anyhow::bail!(
            "gameplay model check: vision model not found at {}",
            config.vision_model_path.display()
        );
    }
    if !config.tokenizer_path.exists() {
        anyhow::bail!(
            "gameplay model check: tokenizer not found at {}",
            config.tokenizer_path.display()
        );
    }

    let detector = ClipGameplayDetector::new(&config)?;
    if detector.is_none() {
        anyhow::bail!("gameplay model check: detector not initialized");
    }

    println!("gameplay model check: OK");
    println!("  text: {}", config.text_model_path.display());
    println!("  vision: {}", config.vision_model_path.display());
    println!("  tokenizer: {}", config.tokenizer_path.display());
    Ok(())
}

/// Run face/reticle detection on a local media file and print the hints.
async fn run_clip_detect_demo(path: &str) -> Result<()> {
    let input = Path::new(path);
    if !input.exists() {
        anyhow::bail!("media file not found: {}", input.display());
    }

    let cfg = read_clip_detect_config();
    let hints = detect_layout_hints(path, &cfg).await?;
    if let Some(face) = hints.face_box {
        println!(
            "demo-detect: face x={:.3} y={:.3} w={:.3} h={:.3}",
            face.x, face.y, face.w, face.h
        );
    } else {
        println!("demo-detect: no face detected");
    }
    if let Some(center) = hints.game_center {
        println!("demo-detect: game center x={:.3} y={:.3}", center.x, center.y);
    } else {
        println!("demo-detect: no reticle detected");
    }

    Ok(())
}

fn is_face_eval_media(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|v| v.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "png" | "jpg" | "jpeg" | "bmp" | "gif" | "webp" | "tiff" | "tif" | "mp4" | "mkv"
            | "mov" | "m4v" | "ts"
    )
}

fn collect_face_eval_inputs(root: &Path) -> Result<Vec<String>> {
    if !root.exists() {
        anyhow::bail!("path not found: {}", root.display());
    }
    if !root.is_dir() {
        anyhow::bail!("not a directory: {}", root.display());
    }
    let mut stack = vec![root.to_path_buf()];
    let mut out = Vec::new();
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if is_face_eval_media(&path) {
                out.push(path.to_string_lossy().to_string());
            }
        }
    }
    out.sort();
    Ok(out)
}

async fn run_face_sweep(pos_dir: &str, neg_dir: &str, extra: &[String]) -> Result<()> {
    let pos_root = Path::new(pos_dir);
    let neg_root = Path::new(neg_dir);
    let positives = collect_face_eval_inputs(pos_root)?;
    let negatives = collect_face_eval_inputs(neg_root)?;
    if positives.is_empty() {
        anyhow::bail!("no positives found under {}", pos_root.display());
    }
    if negatives.is_empty() {
        anyhow::bail!("no negatives found under {}", neg_root.display());
    }

    let mut score_start = 0.2f32;
    let mut score_end = 0.7f32;
    let mut score_step = 0.05f32;
    let mut out_path: Option<PathBuf> = None;

    if extra.len() >= 3 {
        let parsed = (
            extra[0].parse::<f32>().ok(),
            extra[1].parse::<f32>().ok(),
            extra[2].parse::<f32>().ok(),
        );
        if let (Some(start), Some(end), Some(step)) = parsed {
            score_start = start;
            score_end = end;
            score_step = step;
            if extra.len() >= 4 {
                out_path = Some(PathBuf::from(&extra[3]));
            }
        } else if !extra.is_empty() {
            out_path = Some(PathBuf::from(&extra[0]));
        }
    } else if extra.len() == 1 {
        out_path = Some(PathBuf::from(&extra[0]));
    }

    if !(score_start.is_finite() && score_end.is_finite() && score_step.is_finite()) {
        anyhow::bail!("invalid score range");
    }
    if score_step <= 0.0 {
        anyhow::bail!("score step must be > 0");
    }
    if score_end < score_start {
        std::mem::swap(&mut score_start, &mut score_end);
    }

    let mut scores = Vec::new();
    let mut current = score_start;
    while current <= score_end + 1e-6 {
        scores.push(current);
        current += score_step;
    }
    if scores.is_empty() {
        anyhow::bail!("no scores generated for sweep");
    }

    let cfg = read_clip_detect_config();
    let stats = run_face_threshold_sweep(&positives, &negatives, &scores, &cfg).await?;

    let out_path = out_path.unwrap_or_else(|| PathBuf::from("face_eval/threshold_sweep.csv"));
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut csv = String::new();
    csv.push_str("score,pos_total,pos_found,pos_rate,neg_total,neg_found,neg_rate,precision,score_metric\n");
    let mut best: Option<(f32, f32, f32)> = None;
    for stat in &stats {
        let pos_rate = if stat.pos_total > 0 {
            stat.pos_found as f32 / stat.pos_total as f32
        } else {
            0.0
        };
        let neg_rate = if stat.neg_total > 0 {
            stat.neg_found as f32 / stat.neg_total as f32
        } else {
            0.0
        };
        let precision = if stat.pos_found + stat.neg_found > 0 {
            stat.pos_found as f32 / (stat.pos_found + stat.neg_found) as f32
        } else {
            0.0
        };
        let score_metric = pos_rate - neg_rate;
        csv.push_str(&format!(
            "{:.4},{},{},{:.4},{},{},{:.4},{:.4},{:.4}\n",
            stat.score,
            stat.pos_total,
            stat.pos_found,
            pos_rate,
            stat.neg_total,
            stat.neg_found,
            neg_rate,
            precision,
            score_metric
        ));
        if best
            .as_ref()
            .map(|(_, metric, _)| score_metric > *metric)
            .unwrap_or(true)
        {
            best = Some((stat.score, score_metric, neg_rate));
        }
    }
    fs::write(&out_path, csv)?;
    println!("face sweep: wrote {}", out_path.display());
    if let Some((score, metric, neg_rate)) = best {
        println!(
            "face sweep: recommended CLIP_FACE_SCORE={:.2} (metric={:.3}, neg_rate={:.2})",
            score, metric, neg_rate
        );
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct MicOpts {
    page_url: String,
    phrase: Option<String>,
    log_raw_wake: bool,
    mic_device: Option<String>,
}

fn prompt_for_mic_device() -> Option<String> {
    println!("No mic device provided. Attempting to list inputs.");

    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("ffmpeg")
            .arg("-list_devices")
            .arg("true")
            .arg("-f")
            .arg("dshow")
            .arg("-i")
            .arg("dummy")
            .status();
        println!("Format example: audio=\"Microphone (Realtek(R) Audio)\"");
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("ffmpeg")
            .arg("-f")
            .arg("avfoundation")
            .arg("-list_devices")
            .arg("true")
            .arg("-i")
            .arg("")
            .status();
        println!("Format example: :0");
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        let _ = Command::new("pactl").arg("list").arg("short").arg("sources").status();
        println!("Format example: default or hw:0");
    }

    #[cfg(target_os = "windows")]
    {
        let devices = crate::stream_audio_wake::list_system_mics();
        if !devices.is_empty() {
            if !io::stdin().is_terminal() {
                println!("Non-interactive shell detected; using first device: {}", devices[0]);
                return Some(devices[0].clone());
            }
            println!("Detected devices:");
            for (idx, d) in devices.iter().enumerate() {
                println!("  [{}] {}", idx + 1, d);
            }
            print!("Enter mic device (or press Enter to use [{}]): ", devices[0]);
            let _ = io::stdout().flush();
            let mut line = String::new();
            if io::stdin().read_line(&mut line).is_ok() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    return Some(devices[0].clone());
                }
                if let Ok(num) = trimmed.parse::<usize>() {
                    if num > 0 && num <= devices.len() {
                        return Some(devices[num - 1].clone());
                    }
                }
                return Some(trimmed.to_string());
            }
        }
    }

    if !io::stdin().is_terminal() {
        return None;
    }

    print!("Enter mic device (or press Enter to try defaults): ");
    let _ = io::stdout().flush();
    let mut line = String::new();
    if io::stdin().read_line(&mut line).is_ok() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    None
}

fn print_help(bin: &str) {
    println!("Usage:");
    println!("  {bin} <page_url> [options]  (scheme optional, e.g. kick.com/user)");
    println!("  {bin} demo-buffer");
    println!("  {bin} demo-hls-buffer <page_url>");
    println!("  {bin} demo-ts <path_to_ts> [--phrase WORDS] [--no-log-raw-wake]");
    println!("  {bin} reprocess-ts <path_to_ts>");
    println!("  {bin} kick-vod <kick_channel_url>");
    println!("  {bin} check-gameplay-model [model_dir]");
    println!("  {bin} demo-detect <media_path>");
    println!("  {bin} face-sweep <positives_dir> <negatives_dir> [score_start score_end score_step] [out_csv]");
    println!("  {bin} demo-wakeword-mic <page_url> [--phrase WORDS] [--log-raw-wake] [--mic-device NAME]");
    println!("");
    println!("Options:");
    println!("  --phrase WORDS          Override wake phrase(s), comma/pipe separated (default: 'orange')");
    println!("  --stream URL            Add a stream URL to watch (repeat or comma/pipe separated)");
    println!("  --streams PATH          Streams file to sync/watch for multi-stream runs");
    println!("  --log-raw-wake         Log raw/normalized transcripts (default on)");
    println!("  --no-log-raw-wake      Disable transcript logging");
    println!("  --low-resources        Enable low-resource mode overrides");
    println!("  --no-low-resources     Disable low-resource mode overrides");
    println!("  --mic-device NAME      Microphone device for mic wake mode");
    println!("  -h, --help             Show this help");
    println!("");
    println!("CLI overrides:");
    println!("  Any environment variable below can be passed as a CLI flag by lowercasing");
    println!("  and replacing '_' with '-' (e.g., CLIP_LAYOUT -> --clip-layout=stacked).");
    println!("  For boolean env vars, use the '=true'/'=false' form to avoid ambiguity.");
    println!("");
    println!("Environment (selected):");
    println!("  CLIP_PAGE_URL            Default page when none is passed");
    println!("  CLIP_LAYOUT              Layout mode: stacked (default) or full");
    println!("  CLIP_FACE_RATIO          Height ratio reserved for face panel (default 0.40)");
    println!("  CLIP_FACE_MIN_STACKED_RATIO Minimum face crop size for stacked layout (default 0.25)");
    println!("  CLIP_FACE_CROP           Face crop expr w:h:x:y (optional, overrides detection/anchor)");
    println!("  CLIP_FACE_CONTEXT        Face crop expansion scale for detected face (default 6.0)");
    println!("  CLIP_FACE_ZOOM           Face crop zoom factor (>1 zooms out, <1 zooms in; default 1.0)");
    println!("  CLIP_FACE_BOX            Normalized face box x:y:w:h (0..1) for auto-crop");
    println!("  CLIP_FACE_REGION         Normalized face bounds x:y:w:h (0..1) for mid-shot framing");
    println!("  CLIP_FACE_ANCHOR         Anchor for default face crop (top-left default)");
    println!("  CLIP_GAME_CENTER         Normalized gameplay center x:y (0..1) for reticle centering");
    println!("  CLIP_GAME_REGION         Normalized gameplay bounds x:y:w:h (0..1) for layout checks");
    println!("  CLIP_FACE_MODEL          Face detector model path (default models/face_detection_yunet_2023mar.onnx)");
    println!("  CLIP_FACE_BACKEND        Face detector backend: auto (default), ort, or tract");
    println!("  CLIP_ORT_DYLIB           Path to onnxruntime.dll (optional)");
    println!("  CLIP_ORT_DEVICE          ORT CUDA device id, auto, or cpu (default auto)");
    println!("  CLIP_ORT_GPU_MEM_LIMIT_MB ORT CUDA arena limit in MB (default 0 = unlimited)");
    println!("  CLIP_ORT_MIN_FREE_VRAM_MB Min free VRAM before using ORT CUDA (default 512)");
    println!("  CLIP_FACE_DUMP_DIR       Write face debug images with rectangles to this folder");
    println!("  CLIP_FACE_DUMP_RAW       Dump raw face candidates (no score filtering) when enabled");
    println!("  CLIP_FACE_PICK_RAW       Pick faces using raw detector score (default true)");
    println!("  CLIP_FACE_SCORE          Face detection confidence threshold (default 0.6)");
    println!("  CLIP_FACE_TRACK_STEP     Seconds between face tracking samples (default 2.0)");
    println!("  CLIP_FACE_REFRAME_SECS   Minimum seconds between tracked reframes (default 6.0)");
    println!("  CLIP_FACE_LOCK_CENTER    Lock crop center to detected face positions (default false)");
    println!("  CLIP_FACE_FALLBACK       Allow face-box fallback when tracking missing (default false)");
    println!("  CLIP_AUDIO_NORM          Normalize clip audio loudness (default true)");
    println!("  CLIP_CAPTIONS            Enable word-by-word open captions (default false)");
    println!("  CLIP_CAPTIONS_POSITION   Caption placement: margin (default) or chest");
    println!("  CLIP_CAPTIONS_RENDER     Open caption renderer: drawtext (default) or subtitles");
    println!("  CLIP_CAPTIONS_FONT       Caption font name or TTF path (optional)");
    println!("  CLIP_CAPTIONS_SIZE       Caption font size px or ratio (<=2 treated as ratio)");
    println!("  CLIP_CAPTIONS_COLOR      Caption text color (default white)");
    println!("  CLIP_CAPTIONS_OUTLINE    Caption outline width (default 3)");
    println!("  CLIP_CAPTIONS_OUTLINE_COLOR Caption outline color (default black)");
    println!("  CLIP_CAPTIONS_WIDTH_RATIO Max width fraction for captions (default 0.85)");
    println!("  CLIP_CAPTIONS_GLYPH_RATIO Glyph width multiplier for caption sizing (default 0.7)");
    println!("  CLIP_CAPTIONS_SCALE_MIN  Drawtext min scale vs base size (default 0.5)");
    println!("  CLIP_CAPTIONS_SCALE_MAX  Drawtext max scale vs base size (default 2.5)");
    println!("  CLIP_CAPTIONS_SCALE_LOCK Lock drawtext font size per segment (default false)");
    println!("  CLIP_CAPTIONS_DRAWTEXT_MAX Max drawtext cues before fallback (default 120)");
    println!("  CLIP_CAPTIONS_PERSIST_FAILURE Keep captions disabled after failure (default false)");
    println!("  CLIP_CAPTIONS_BUCKET_SECS Bucket size to merge words (0 disables, default 0)");
    println!("  CLIP_CAPTIONS_MIN_WORD_SECS Minimum per-word on-screen time (default 0.12s)");
    println!("  CLIP_CAPTIONS_MAX_WORDS  Cap on rendered words (default 300)");
    println!("  CLIP_CAPTIONS_CHEST_RATIO Caption Y ratio when placed on chest (default 0.65)");
    println!("  CLIP_CAPTIONS_MARGIN_OFFSET Extra Y offset for margin captions (default 0)");
    println!("  CLIP_CAPTIONS_DEBUG      Log caption timings (default false)");
    println!("  CLIP_CLOSED_CAPTIONS     Embed closed captions track when available (default true)");
    println!("  CLIP_LLM_ENABLE          Enable LLM metadata (default false)");
    println!("  CLIP_LLM_ENDPOINT        LLM chat completions endpoint (default localhost:1234)");
    println!("  CLIP_LLM_MODEL           LLM model name or path");
    println!("  CLIP_LLM_TEMPERATURE     LLM temperature (default 0.2)");
    println!("  CLIP_LLM_TIMEOUT_SECS    LLM request timeout seconds (default 20)");
    println!("  CLIP_LLM_BACKOFF_SECS    LLM backoff seconds after failure (default 120)");
    println!("  CLIP_LLM_MAX_TRANSCRIPT_CHARS Transcript truncation limit (default 4000)");
    println!("  CLIP_LLM_TITLE_MAX_CHARS LLM title max length (default 80)");
    println!("  CLIP_LLM_DESCRIPTION_MAX_CHARS LLM description max length (default 280)");
    println!("  CLIP_LLM_GPU_LAYERS      LLM GPU layers (0 = CPU, empty = model default)");
    println!("  CLIP_LLM_FILE_RENAME     Rename clip files using LLM title (default true)");
    println!("  CLIP_LLM_DEBUG           Log LLM parse details (default false)");
    println!("  CLIP_FFMPEG_PROGRESS     Print ffmpeg progress/percent during encoding (default false)");
    println!("  CLIP_FFMPEG_PROGRESS_STALL_SECS Fail if ffmpeg makes no progress for N seconds (default 120)");
    println!("  CLIP_REPROCESS_CHUNK_SECS Chunk duration for reprocess-ts media (default 60, 0 disables)");
    println!("  CLIP_REPROCESS_SUBCHUNK_SECS Sub-chunk duration for reprocess resume (default unset)");
    println!("  CLIP_REPROCESS_FAST      Speed-focused reprocess (skip captions/LLM/audio norm, faster encode)");
    println!("  CLIP_REPROCESS_CONTINUE_ON_ERROR Keep reprocess running after ffmpeg errors (default false)");
    println!("  CLIP_FACE_BUDGET_SECS    Override face detection time budget in seconds");
    println!("  CLIP_FACE_TILE_MIN_SCORE Tile search min score (default 0.60; set <= 0 to disable)");
    println!("  CLIP_FACE_TILE_MAX_DEPTH Max bisection depth for tile search (default 3)");
    println!("  CLIP_FACE_FRAME_HEAD_TOP      Head top offset vs face box (default -0.28)");
    println!("  CLIP_FACE_FRAME_HEAD_TOP_MIN  Min head top offset (default -0.8)");
    println!("  CLIP_FACE_FRAME_HEAD_TOP_MAX  Max head top offset (default 0.2)");
    println!("  CLIP_FACE_CENTER           Center face in frame using kalman-smoothed tracking (default false)");
    println!("  CLIP_FACE_FRAME_EYE_TOP_RATIO Eye->top ratio for head estimate (default 0.45)");
    println!("  CLIP_FACE_FRAME_EYE_CHIN_RATIO Eye->chin ratio for head estimate (default 0.55)");
    println!("  CLIP_FACE_FRAME_SHOULDER_SCALE Shoulder width scale vs face (default 3.2)");
    println!("  CLIP_FACE_MESH          Enable face mesh framing (default true)");
    println!("  CLIP_FACE_MESH_MODEL    Face mesh ONNX model path");
    println!("  CLIP_FACE_MESH_MODEL_MIN_MB Min face mesh model size in MB (default 1)");
    println!("  CLIP_FACE_MESH_MODEL_MAX_MB Max face mesh model size in MB (default 64)");
    println!("  CLIP_FACE_MESH_LOAD_TIMEOUT_SECS Max seconds to load face mesh model (default 60)");
    println!("  CLIP_FACE_MESH_BACKEND  Face mesh backend: auto (default), ort, or tract");
    println!("  CLIP_FACE_MESH_TRACT_OPT Enable tract optimizations for face mesh (default false)");
    println!("  CLIP_FACE_MESH_INPUT_SIZE Fallback face mesh input size (default 192)");
    println!("  CLIP_FACE_MESH_INPUT_MAX Max face mesh input side length before clamping (default 512)");
    println!("  CLIP_FACE_MESH_INPUT_SCALE Input scale for face mesh model (default 1/255)");
    println!("  CLIP_FACE_MESH_REGION_SCALE Face mesh crop expansion scale (default 1.35)");
    println!("  CLIP_FACE_MESH_HEADROOM Extra headroom ratio above mesh top (default 0.12)");
    println!("  CLIP_FACE_MESH_DEBUG    Log face mesh bounds (default false)");
    println!("  CLIP_POSE               Enable MoveNet pose framing (default true)");
    println!("  CLIP_POSE_MODEL         MoveNet Thunder ONNX model path");
    println!("  CLIP_POSE_MODEL_MIN_MB  Min pose model size in MB (default 1)");
    println!("  CLIP_POSE_MODEL_MAX_MB  Max pose model size in MB (default 64)");
    println!("  CLIP_POSE_LOAD_TIMEOUT_SECS Max seconds to load pose model (default 60)");
    println!("  CLIP_POSE_BACKEND       Pose backend: auto (default), ort, or tract");
    println!("  CLIP_POSE_TRACT_OPT     Enable tract optimizations for pose (default false)");
    println!("  CLIP_POSE_SCORE         Min keypoint score for pose framing (default 0.30)");
    println!("  CLIP_POSE_INPUT_SIZE    Fallback pose input size when model is dynamic (default 256)");
    println!("  CLIP_POSE_INPUT_MAX     Max pose input side length before clamping (default 512)");
    println!("  CLIP_POSE_INPUT_SCALE   Input scale for pose model (default 1/255)");
    println!("  CLIP_POSE_HEAD_RATIO    Head-top margin ratio from pose (default 0.60)");
    println!("  CLIP_POSE_SHOULDER_MARGIN Shoulder width margin multiplier (default 1.10)");
    println!("  CLIP_POSE_DEBUG         Log pose keypoints and frame spec (default false)");
    println!("  CLIP_FACE_ID            Enable face-ID matching (default false)");
    println!("  CLIP_FACE_ID_FILE       Face-ID embedding file (json)");
    println!("  CLIP_FACE_ID_MODEL      Face-ID ONNX model path");
    println!("  CLIP_FACE_ID_THRESHOLD  Face-ID cosine threshold (default 0.35)");
    println!("  CLIP_FACE_ID_REQUIRE_MOTION Require motion for face-ID (default true)");
    println!("  CLIP_FACE_ID_MOTION     Motion threshold for face-ID (default 0.015)");
    println!("  CLIP_FACE_ID_BGR        Use BGR input for face-ID model (default true)");
    println!("  CLIP_FACE_ID_DEBUG      Log face-ID scores (default false)");
    println!("  CLIP_FACE_DEBUG          Log face detector outputs and best score");
    println!("  CLIP_FACE_TRACK          Track face across the full clip (default true)");
    println!("  CLIP_DETECT              Enable auto-detection for stacked layout (default true)");
    println!("  CLIP_DETECT_SIZE         Reticle detection frame size WxH (default 960x540)");
    println!("  CLIP_DETECT_SAMPLES      Number of detection frames to sample (default 3)");
    println!("  CLIP_DETECT_START        Detection sample start time in seconds (default 1.0)");
    println!("  CLIP_DETECT_STEP         Seconds between detection samples (default 1.5)");
    println!("  CLIP_DETECT_FULL         Sample detection frames across the full clip (local files only)");
    println!("  CLIP_DETECT_BUDGET_SECS  Max seconds to spend analyzing detection samples");
    println!("  CLIP_GAMEPLAY            Enable CLIP model usage (default true)");
    println!("  CLIP_REGION_DETECT       Enable CLIP region detection (default true)");
    println!("  CLIP_LIVE_CONFIG         Path to live config file for hot-reload overrides");
    println!("  CLIP_LIVE_CONFIG_POLL_SECS   Live config poll interval in seconds (default 2)");
    println!("  CLIP_LIVE_FAST           Skip heavy detection/captions for faster live renders (default true)");
    println!("  CLIP_LIVE_LAYOUT_TTL_SECS Reuse detected layout hints for N seconds (default 120)");
    println!("  CLIP_LOW_RESOURCES       Force low-resource overrides (default false)");
    println!("  CLIP_SINGLE_INSTANCE     Require single instance in single-mode runs (default true)");
    println!("  CLIP_STREAMS_FILE        Path to streams file for multi-stream runs");
    println!("  CLIP_STREAMS_POLL_SECS   Streams file poll interval in seconds (default 5)");
    println!("  CLIP_STREAMS_MAX_CONCURRENT Max number of concurrent streams (default 1)");
    println!("  CLIP_M3U8_REFRESH_SECS   Refresh signed m3u8 URL every N seconds (default 240)");
    println!("  CLIP_STREAM_OFFLINE_SECS Exit if no new segments for N seconds (default 120)");
    println!("  CLIP_WAKE_WORDS          Wake phrase list (comma/pipe separated) for clip trigger");
    println!("  CLIP_PROFILE             Enable timing logs for hotspots (default false)");
    println!("  CLIP_EMOTION_ENABLE      Enable emotion triggers from audio/face (default false)");
    println!("  CLIP_EMOTION_WORDS       Emotion keyword list (comma/pipe separated)");
    println!("  CLIP_EMOTION_THRESHOLD   Emotion score threshold (default 1.5)");
    println!("  CLIP_EMOTION_AUDIO_RMS   RMS loudness threshold (default 0.08)");
    println!("  CLIP_EMOTION_AUDIO_PEAK  Peak loudness threshold (default 0.35)");
    println!("  CLIP_EMOTION_FACE_MOTION Face motion threshold (default 0.035)");
    println!("  CLIP_GAMEPLAY_LABELS     CLIP gameplay positive labels");
    println!("  CLIP_GAMEPLAY_NEG_LABELS CLIP gameplay negative labels");
    println!("  CLIP_GAMEPLAY_SCORE      CLIP gameplay score threshold (default 0.12)");
    println!("  CLIP_GAMEPLAY_TOPK       CLIP gameplay top-k patches (default 6)");
    println!("  CLIP_GAMEPLAY_BUDGET_SECS Override gameplay sampling time budget in seconds");
    println!("  CLIP_CAM_LABELS          CLIP cam positive labels");
    println!("  CLIP_CAM_NEG_LABELS      CLIP cam negative labels");
    println!("  CLIP_CAM_SCORE           CLIP cam score threshold (default CLIP_GAMEPLAY_SCORE)");
    println!("  CLIP_CAM_TOPK            CLIP cam top-k patches (default CLIP_GAMEPLAY_TOPK)");
    println!("  CLIP_CAM_REGION_SCALE    Expand CLIP cam region bounds (default 1.6)");
    println!("  CLIP_TS_REALTIME         When set, read local TS files at realtime speed");
    println!("  M3U8_URL_OVERRIDE        Skip discovery; use this master URL directly");
    println!("  COOKIE_HEADER / KICK_COOKIE / TIKTOK_COOKIE / TWITCH_COOKIE   Cookies to send on discovery");
    println!("  HEADLESS_M3U8_SCRIPT / HEADLESS_M3U8_SCRIPT_TIKTOK   Override Playwright scripts");
    println!("  WAKE_REFRACTORY_SECS     Cooldown between wake detections (default 12)");
    println!("  WAKE_BUFFER_HEADROOM_SECS   Extra buffer headroom for wake timing (default 20)");
    println!("  WAKE_BUFFER_RESTART_SECS    Restart if buffer exceeds seconds (default 300, 0 disables)");
    println!("  WAKE_NO_WORDS_SECS       Restart if wake worker stalls for seconds (default 180, 0 disables)");
    println!("  CLIP_WATCHDOG_GAP_SECS   Alert if no clip saved for seconds (default 600, 0 disables)");
    println!("  CLIP_WAKE_MIN_CLIP_SECS  Force a wake clip if last saved clip exceeds this gap (default 300, 0 disables)");
    println!("  SKIP_CLIP_SAVE           If set to 1/true, skip writing clips");
    println!("  WHISPER_MODEL            Path to whisper model (default auto)");
    println!("  WHISPER_GPU              Enable GPU for live wake (default true)");
    println!("  WHISPER_CLIP_GPU         Enable GPU for clip transcription (default false)");
    println!("  WHISPER_MODEL_BENCHMARK  Benchmark multiple models to meet RT target (default false)");
    println!("  WHISPER_MIN_FREE_VRAM_MB Min free VRAM before using whisper GPU (default 2048)");
    println!("  WHISPER_CLIP_MIN_FREE_VRAM_MB Min free VRAM before using whisper GPU for clips (default 4096)");
    println!("  WHISPER_CLIP_VRAM_FACTOR Scale factor for estimating clip GPU VRAM use (default 1.35)");
    println!("  WHISPER_CLIP_VRAM_OVERHEAD_MB Extra VRAM cushion for clip GPU use (default 512)");
    println!("  WHISPER_CLIP_VRAM_EXTRA_MB Extra safety headroom before using clip GPU (default 1024)");
    println!("  WHISPER_GPU_DEVICE       Force NVIDIA device id for live wake (default auto)");
    println!("  WHISPER_CLIP_GPU_DEVICE  Force NVIDIA device id for clip transcription (default auto)");
    println!("  WHISPER_ISOLATE          Run live wake in a helper process (default false)");
    println!("  WHISPER_WORKER_STATUS_MS Poll interval for wake worker status (default 500ms)");
    println!("  FFMPEG_ENCODER / FFMPEG_HWACCEL / FFMPEG_HWACCEL_DEVICE   Encoder/accel knobs");
    println!("  FFMPEG_HWACCEL_FALLBACK  Fallback hwaccel (e.g. d3d11va, cuda, none; default none)");
    println!("  FFMPEG_MIN_FREE_VRAM_MB  Min free VRAM before using GPU encode (default 512)");
    println!("  FFMPEG_ENCODE_TIMEOUT_SECS   Hard cap for ffmpeg encode wall time (seconds)");
    println!("  CLIP_FFMPEG_NO_LIMITS   Disable ffmpeg stall + encode timeouts");
    println!("  GPU_VRAM_RESERVE_MB      Always keep this much VRAM free (default 512)");
    println!("  GPU_PICK_STRATEGY        GPU selection: free (default) or total");
    println!("  LOG_M3U8_HEADERS         Log request headers when fetching playlists");
    println!("  TWITCH_CLIENT_ID         Twitch Client-ID for playback token (default web client)");
    println!("  TWITCH_OAUTH_TOKEN / TWITCH_AUTH_TOKEN   Twitch OAuth token for gated streams (optional)");
    println!("");
    println!("Notes:");
    println!("  - Main path: page URL -> HLS discovery (TikTok HTTP first, headless fallback) -> wake detection -> clip.");
    println!("  - Demo modes: buffer-only, HLS buffer demo, or mic wake demo for quick sanity checks.");
    println!("  - Live config file supports KEY=VALUE, KEY: VALUE, or flag lines like clip-face-debug.");
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::captions::{
        CaptionConfig, CaptionPosition, CaptionWord, build_caption_drawtext_chain,
        build_ass_from_payload_with_limits, build_srt_from_payload, default_caption_font,
        format_srt_time, wrap_caption_text,
    };
    use crate::clip_layout::FaceAnchor;
    use crate::clip_layout::NormalizedRect;
    use crate::stream_audio_wake::WordTiming;
    use crate::test_util::EnvGuard;
    use crate::url_utils::sanitize_m3u8_url;
    use std::process::Command as SysCommand;

    struct CaptionFlagGuard {
        open_disabled: bool,
        closed_disabled: bool,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl CaptionFlagGuard {
        fn new() -> Self {
            static CAPTION_FLAG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
            let lock = CAPTION_FLAG_LOCK.lock().expect("caption flag lock");
            let open_disabled = OPEN_CAPTIONS_DISABLED.load(Ordering::Relaxed);
            let closed_disabled = CLOSED_CAPTIONS_DISABLED.load(Ordering::Relaxed);
            OPEN_CAPTIONS_DISABLED.store(false, Ordering::Relaxed);
            CLOSED_CAPTIONS_DISABLED.store(false, Ordering::Relaxed);
            Self {
                open_disabled,
                closed_disabled,
                _lock: lock,
            }
        }
    }

    impl Drop for CaptionFlagGuard {
        fn drop(&mut self) {
            OPEN_CAPTIONS_DISABLED.store(self.open_disabled, Ordering::Relaxed);
            CLOSED_CAPTIONS_DISABLED.store(self.closed_disabled, Ordering::Relaxed);
        }
    }

    fn tool_available(tool: &str) -> bool {
        SysCommand::new(tool)
            .arg("-version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn llm_error_indicates_offline_matches_connection_refused() {
        let msg = "error trying to connect: tcp connect error: No connection could be made because the target machine actively refused it. (os error 10061)";
        assert!(llm_error_indicates_offline(msg));
        let other = "timeout while waiting for response";
        assert!(!llm_error_indicates_offline(other));
    }

    #[test]
    fn disable_closed_captions_for_run_disables_only_closed() {
        let _guard = CaptionFlagGuard::new();
        disable_closed_captions_for_run("test");
        assert!(open_captions_allowed());
        assert!(!closed_captions_allowed());
    }

    #[test]
    fn disable_open_captions_for_run_disables_both() {
        let _guard = CaptionFlagGuard::new();
        disable_open_captions_for_run("test");
        assert!(!open_captions_allowed());
        assert!(!closed_captions_allowed());
    }

    fn generate_test_video(path: &Path) -> Result<()> {
        let status = SysCommand::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=320x180:d=1",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-shortest",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                path.to_string_lossy().as_ref(),
            ])
            .status()?;
        if !status.success() {
            anyhow::bail!("ffmpeg failed generating test video");
        }
        Ok(())
    }

    fn output_has_subtitles(path: &Path) -> Result<bool> {
        let output = SysCommand::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "s",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                path.to_string_lossy().as_ref(),
            ])
            .output()?;
        if !output.status.success() {
            return Ok(false);
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.lines().any(|line| line.trim() == "subtitle"))
    }

    fn parse_signalstats_yavg(output: &str) -> Option<f64> {
        let mut best: Option<f64> = None;
        for line in output.lines() {
            let Some(idx) = line.find("lavfi.signalstats.YAVG=") else {
                continue;
            };
            let tail = &line[idx + "lavfi.signalstats.YAVG=".len()..];
            let val_str = tail
                .split(|c: char| c.is_whitespace() || c == ',')
                .next()
                .unwrap_or("");
            if let Ok(val) = val_str.parse::<f64>() {
                best = Some(best.map_or(val, |prev| prev.max(val)));
            }
        }
        best
    }

    fn sample_frame_yavg(path: &Path, time_secs: f32) -> Result<f64> {
        let output = SysCommand::new("ffmpeg")
            .args([
                "-hide_banner",
                "-v",
                "info",
                "-ss",
                &format!("{time_secs:.3}"),
                "-i",
                path.to_string_lossy().as_ref(),
                "-frames:v",
                "1",
                "-vf",
                "signalstats,metadata=print",
                "-f",
                "null",
                "-",
            ])
            .output()?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("ffmpeg signalstats failed: {}", stderr.trim());
        }
        let mut merged = String::new();
        merged.push_str(&String::from_utf8_lossy(&output.stdout));
        merged.push_str(&String::from_utf8_lossy(&output.stderr));
        parse_signalstats_yavg(&merged)
            .ok_or_else(|| anyhow::anyhow!("failed to parse signalstats YAVG"))
    }

    #[test]
    fn ffmpeg_caption_error_detection_catches_fontconfig() {
        assert!(ffmpeg_error_indicates_caption_issue(
            "Fontconfig error: Cannot load default config file: No such file: (null)"
        ));
        assert!(ffmpeg_error_indicates_caption_issue(
            "libass: cannot find system fonts"
        ));
        assert!(!ffmpeg_error_indicates_caption_issue(
            "ffmpeg progress stalled for 120.0s; terminating"
        ));
    }

    fn temp_face_id_path(label: &str) -> PathBuf {
        let stamp = now_unix_ms();
        std::env::temp_dir().join(format!("autoclip_face_id_{label}_{stamp}.json"))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn face_id_enroll_skips_when_embedding_exists() {
        let mut env = EnvGuard::new();
        env.set("CLIP_FACE_ID", "1");
        let out_path = temp_face_id_path("existing");
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&out_path, "{}").unwrap();
        env.set("CLIP_FACE_ID_FILE", out_path.to_string_lossy().as_ref());

        maybe_enroll_face_id_from_media("https://kick.com/kingbushcamp", Path::new("C:\\fake")).await;

        assert!(out_path.exists());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn face_id_enrolls_from_network_image() -> Result<()> {
        let mut env = EnvGuard::new();
        env.set("CLIP_FACE_ID", "1");
        env.remove("CLIP_FACE_ID_FILE");

        let model_path = std::env::var("CLIP_FACE_ID_MODEL")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "models/face_id/arcface.onnx".to_string());
        assert!(
            Path::new(&model_path).exists(),
            "face id model missing at {}",
            model_path
        );
        let face_model = std::env::var("CLIP_FACE_MODEL")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "models/face_detection_yunet_2023mar.onnx".to_string());
        assert!(
            Path::new(&face_model).exists(),
            "face detector model missing at {}",
            face_model
        );

        let out_path = temp_face_id_path("network");
        let image_url = "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg";
        let ok = clip_detect::enroll_face_id_from_image_url(image_url, &out_path).await?;
        assert!(ok, "expected face id enrollment from image");
        assert!(out_path.exists(), "expected embedding output to exist");
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn run_stub_succeeds() {
        // Ensure no network-dependent env vars force the main path.
        let mut env = EnvGuard::new();
        env.remove("CLIP_PAGE_URL");
        env.remove("M3U8_URL_OVERRIDE");
        let app = AutoClip::new(Config::example());
        assert!(app.run().await.is_ok());
    }

    #[test]
    fn parse_streams_file_ignores_comments_and_dedupes() {
        let input = "# streams\nkick.com/queenbushcamp\n\n// comment\nkick.com/queenbushcamp\nhttps://twitch.tv/someuser\n";
        let streams = stream_utils::parse_streams_file(input);
        assert_eq!(
            streams,
            vec![
                "https://kick.com/queenbushcamp".to_string(),
                "https://twitch.tv/someuser".to_string()
            ]
        );
    }

    #[test]
    fn split_wake_phrases_handles_delimiters() {
        let input = " hello, world | ok; test\nnext\rline ";
        assert_eq!(
            split_wake_phrases(input),
            vec![
                "hello".to_string(),
                "world".to_string(),
                "ok".to_string(),
                "test".to_string(),
                "next".to_string(),
                "line".to_string()
            ]
        );
    }

    #[test]
    fn stream_id_from_url_sanitizes() {
        assert_eq!(
            stream_id_from_url("https://kick.com/QueenBushCamp"),
            "kick_com_queenbushcamp"
        );
        assert_eq!(
            stream_id_from_url("twitch.tv/SomeUser?src=live"),
            "twitch_tv_someuser"
        );
    }

    #[test]
    fn normalize_page_url_adds_scheme_for_domains() {
        assert_eq!(
            normalize_page_url("kick.com/queenbushcamp"),
            "https://kick.com/queenbushcamp"
        );
        assert_eq!(
            normalize_page_url("https://kick.com/queenbushcamp"),
            "https://kick.com/queenbushcamp"
        );
        assert_eq!(
            normalize_page_url("C:\\videos\\clip.ts"),
            "C:\\videos\\clip.ts"
        );
        assert_eq!(normalize_page_url("/var/tmp/clip.ts"), "/var/tmp/clip.ts");
    }

    #[test]
    fn wake_min_clip_gap_defaults_to_five_minutes() {
        let mut env = EnvGuard::new();
        env.remove("CLIP_WAKE_MIN_CLIP_SECS");
        let gap = read_wake_min_clip_gap_secs().expect("expected default wake min clip gap");
        assert_eq!(gap.as_secs(), 300);
    }

    #[test]
    fn wake_min_clip_gap_can_disable() {
        let mut env = EnvGuard::new();
        env.set("CLIP_WAKE_MIN_CLIP_SECS", "0");
        assert!(read_wake_min_clip_gap_secs().is_none());
    }

    #[test]
    fn sanitize_m3u8_url_strips_quotes_and_trailing_slash() {
        assert_eq!(
            sanitize_m3u8_url("\"https://example.com/stream.m3u8\""),
            "https://example.com/stream.m3u8"
        );
        assert_eq!(
            sanitize_m3u8_url(" 'https://example.com/stream.m3u8'\\ "),
            "https://example.com/stream.m3u8"
        );
    }

    #[test]
    fn signed_url_expiry_picks_earliest_expiry() {
        let url = Url::parse(
            "https://example.com/stream.m3u8?exp=1700000000&token_exp=1800000000",
        )
        .expect("url parse");
        let expiry = signed_url_expiry(&url).expect("expiry");
        assert_eq!(
            expiry,
            SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000)
        );
    }

    #[test]
    fn lock_or_recover_returns_guard_after_poison() {
        let mutex = Mutex::new(42u32);
        let _ = std::panic::catch_unwind(|| {
            let _guard = mutex.lock().unwrap();
            panic!("poison");
        });
        let guard = lock_or_recover(&mutex, "test");
        assert_eq!(*guard, 42);
    }

    #[test]
    fn extract_output_index_parses_variants() {
        assert_eq!(extract_output_index("clip_001", "clip"), Some(1));
        assert_eq!(extract_output_index("clip_010__title", "clip"), Some(10));
        assert_eq!(extract_output_index("clip__oops", "clip"), None);
        assert_eq!(extract_output_index("other_001", "clip"), None);
    }

    #[test]
    fn is_ts_input_matches_only_ts_variants() {
        assert!(is_ts_input(Path::new("clip.ts")));
        assert!(is_ts_input(Path::new("clip.m2ts")));
        assert!(!is_ts_input(Path::new("clip.mp4")));
        assert!(!is_ts_input(Path::new("clip.mkv")));
    }

    #[test]
    fn next_output_path_skips_existing_and_counter() -> Result<()> {
        let stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_millis();
        let dir = std::env::temp_dir().join(format!("autoclip_test_{stamp}"));
        fs::create_dir_all(&dir)?;

        let renamed = dir.join("clip_001__title.mp4");
        fs::write(&renamed, "stub")?;
        let counter_dir = non_video_dir(dir.to_str().unwrap_or("."));
        let counter = counter_dir.join(".clip_counter");
        fs::write(&counter, "2\n")?;

        let next = next_output_path(dir.to_str().unwrap_or("."), "clip")?;
        assert_eq!(next.file_name().unwrap_or_default(), "clip_003.mp4");

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn extract_meta_image_reads_content() {
        let html = r#"<html><head><meta property="og:image" content="https://example.com/a.jpg"></head></html>"#;
        assert_eq!(
            url_utils::extract_meta_image(html, r#"property="og:image"#),
            Some("https://example.com/a.jpg".to_string())
        );
        let html = r#"<meta name="twitter:image" content="https://example.com/b.png"/>"#;
        assert_eq!(
            url_utils::extract_meta_image(html, r#"name="twitter:image"#),
            Some("https://example.com/b.png".to_string())
        );
    }

    #[test]
    fn face_id_file_for_stream_prefers_env_override() {
        let key = "CLIP_FACE_ID_FILE";
        let mut env = EnvGuard::new();
        env.set(key, "face_id/custom.json");
        let path = face_id::face_id_file_for_stream("https://kick.com/QueenBushCamp");
        assert_eq!(path, PathBuf::from("face_id/custom.json"));
    }

    #[test]
    fn face_area_from_hints_uses_box_or_track() {
        let mut hints = clip_layout::ClipLayoutHints::default();
        assert!(layout_utils::face_area_from_hints(&hints).is_none());

        hints.face_box = Some(NormalizedRect {
            x: 0.0,
            y: 0.0,
            w: 0.2,
            h: 0.5,
        });
        let area = layout_utils::face_area_from_hints(&hints).unwrap();
        assert!((area - 0.1).abs() < 1e-6);

        hints.face_box = None;
        hints.face_track = Some(clip_layout::FaceTrack {
            points: vec![clip_layout::FaceTrackPoint {
                time: 0.0,
                rect: NormalizedRect {
                    x: 0.0,
                    y: 0.0,
                    w: 0.4,
                    h: 0.25,
                },
            }],
        });
        let area = layout_utils::face_area_from_hints(&hints).unwrap();
        assert!((area - 0.1).abs() < 1e-6);
    }

    #[test]
    fn captions_drawtext_chain_includes_enable_window() {
        let cfg = CaptionConfig {
            position: CaptionPosition::Chest,
            font: None,
            font_size: 42.0,
            color: "white".to_string(),
            outline_color: "black".to_string(),
            outline: 3,
            min_word_secs: 0.12,
            max_words: 10,
            chest_ratio: 0.65,
            margin_offset_px: 0.0,
            debug: false,
        };
        let words = vec![CaptionWord {
            text: "hello".to_string(),
            start: 0.5,
            end: 0.9,
        }];
        let chain = build_caption_drawtext_chain(&words, &cfg, 120.0);
        assert!(chain.contains("drawtext="));
        assert!(chain.contains("between(t,0.500,0.900)"));
    }

    #[test]
    fn format_srt_time_formats_hours_minutes_seconds() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(3661.234), "01:01:01,234");
    }

    #[test]
    fn wrap_caption_text_splits_on_space() {
        let text = "hello world again";
        assert_eq!(wrap_caption_text(text, 10), "hello\nworld again");
    }

    #[test]
    fn build_srt_from_payload_falls_back_to_text() {
        let payload = TranscriptPayload {
            text: "Hello [_TT_150] world".to_string(),
            words: Vec::new(),
        };
        let srt = build_srt_from_payload(&payload, Some(2.5)).unwrap();
        assert!(srt.contains("00:00:00,000 --> 00:00:02,500"));
        assert!(srt.contains("Hello world"));
        assert!(!srt.contains("[_TT_"));
    }

    #[test]
    fn build_srt_from_payload_uses_word_timings() {
        let payload = TranscriptPayload {
            text: "Hello world".to_string(),
            words: vec![
                WordTiming {
                    text: "Hello".to_string(),
                    norm: "hello".to_string(),
                    t0: 0.0,
                    t1: 0.4,
                },
                WordTiming {
                    text: "world".to_string(),
                    norm: "world".to_string(),
                    t0: 0.5,
                    t1: 1.0,
                },
            ],
        };
        let srt = build_srt_from_payload(&payload, Some(2.0)).unwrap();
        assert!(srt.contains("00:00:00,000 --> 00:00:00,400"));
        assert!(srt.contains("Hello"));
        assert!(srt.contains("00:00:00,500 --> 00:00:01,000"));
        assert!(srt.contains("world"));
    }

    #[test]
    fn build_srt_from_payload_avoids_early_word_start() {
        let mut env = EnvGuard::new();
        env.set("CLIP_CAPTIONS_BUCKET_SECS", "0");
        let payload = TranscriptPayload {
            text: "Hello world".to_string(),
            words: vec![
                WordTiming {
                    text: "Hello".to_string(),
                    norm: "hello".to_string(),
                    t0: 0.0,
                    t1: 0.6,
                },
                WordTiming {
                    text: "world".to_string(),
                    norm: "world".to_string(),
                    t0: 0.4,
                    t1: 0.8,
                },
            ],
        };
        let srt = build_srt_from_payload(&payload, Some(2.0)).unwrap();
        assert!(srt.contains("00:00:00,400 --> 00:00:00,800"));
    }

    #[test]
    fn captions_merge_words_with_whisper_bucket() {
        let mut env = EnvGuard::new();
        env.set("CLIP_CAPTIONS_BUCKET_SECS", "0.10");
        let payload = TranscriptPayload {
            text: "Hello world".to_string(),
            words: vec![
                WordTiming {
                    text: "Hello".to_string(),
                    norm: "hello".to_string(),
                    t0: 0.051,
                    t1: 0.080,
                },
                WordTiming {
                    text: "world".to_string(),
                    norm: "world".to_string(),
                    t0: 0.052,
                    t1: 0.090,
                },
            ],
        };
        let srt = build_srt_from_payload(&payload, Some(1.0)).unwrap();
        assert!(srt.contains("Hello world"));
    }

    #[test]
    fn choose_segment_duration_uses_playlist_when_pts_is_suspicious() {
        let playlist = Duration::from_secs_f32(6.0);
        let too_short = Duration::from_secs_f32(0.8);
        let too_long = Duration::from_secs_f32(30.0);
        assert_eq!(
            choose_segment_duration(Some(too_short), playlist),
            playlist
        );
        assert_eq!(choose_segment_duration(Some(too_long), playlist), playlist);
        let ok = Duration::from_secs_f32(6.5);
        assert_eq!(choose_segment_duration(Some(ok), playlist), ok);
    }

    #[test]
    fn buffer_retains_full_60s_even_with_bad_pts() {
        let clip_window = Duration::from_secs(60);
        let mut buffer = RollingBuffer::new(clip_window);
        let playlist = Duration::from_secs_f32(6.0);
        let bad_pts = Duration::from_secs_f32(0.6);
        for _ in 0..10 {
            let seg = choose_segment_duration(Some(bad_pts), playlist);
            buffer.push(vec![0u8; 16], seg);
        }
        assert_eq!(buffer.total_duration(), clip_window);
    }

    #[test]
    fn stacked_layout_requires_face_and_game() {
        let mut env = EnvGuard::new();
        env.set("CLIP_GAMEPLAY", "1");
        env.set("CLIP_LOW_RESOURCES", "0");
        env.set("CLIP_FACE_FALLBACK", "1");

        let layout = ClipLayoutConfig {
            mode: ClipLayoutMode::Stacked,
            face_ratio: 0.4,
            face_crop: None,
            face_anchor: FaceAnchor::TopLeft,
            face_context_scale: 1.5,
            face_zoom: 1.0,
        };
        let mut both = ClipLayoutHints::default();
        both.face_box = Some(NormalizedRect {
            x: 0.1,
            y: 0.1,
            w: 0.2,
            h: 0.2,
        });
        both.game_center = Some(NormalizedPoint { x: 0.5, y: 0.5 });
        let decision = decide_stacked_layout(&layout, &both, None);
        assert!(decision.layout_is_stacked);
        assert!(!decision.face_only);
        assert!(!decision.fullscreen_fill);

        let mut face_only = both.clone();
        face_only.face_box = Some(NormalizedRect {
            x: 0.35,
            y: 0.35,
            w: 0.3,
            h: 0.3,
        });
        face_only.game_center = None;
        let decision = decide_stacked_layout(&layout, &face_only, None);
        assert!(!decision.layout_is_stacked);
        assert!(decision.fullscreen_fill);
        assert!(!decision.face_only);

        let mut game_only = ClipLayoutHints::default();
        game_only.game_center = Some(NormalizedPoint { x: 0.5, y: 0.5 });
        let decision = decide_stacked_layout(&layout, &game_only, None);
        assert!(!decision.layout_is_stacked);
        assert!(decision.fullscreen_fill);
        assert!(!decision.face_only);
    }

    #[test]
    fn stacked_layout_uses_face_box_without_fallback() {
        let mut env = EnvGuard::new();
        env.set("CLIP_GAMEPLAY", "1");
        env.set("CLIP_LOW_RESOURCES", "0");
        env.set("CLIP_FACE_FALLBACK", "0");

        let layout = ClipLayoutConfig {
            mode: ClipLayoutMode::Stacked,
            face_ratio: 0.4,
            face_crop: None,
            face_anchor: FaceAnchor::TopLeft,
            face_context_scale: 1.5,
            face_zoom: 1.0,
        };
        let mut hints = ClipLayoutHints::default();
        hints.face_box = Some(NormalizedRect {
            x: 0.1,
            y: 0.1,
            w: 0.2,
            h: 0.2,
        });
        hints.game_center = Some(NormalizedPoint { x: 0.5, y: 0.5 });
        let decision = decide_stacked_layout(&layout, &hints, None);
        assert!(decision.layout_is_stacked);
        assert!(!decision.fullscreen_fill);
    }

    #[test]
    fn stacked_layout_uses_low_resource_gameplay_guess() {
        let mut env = EnvGuard::new();
        env.set("CLIP_GAMEPLAY", "0");
        env.set("CLIP_LOW_RESOURCES", "1");
        env.set("CLIP_FACE_FALLBACK", "1");

        let layout = ClipLayoutConfig {
            mode: ClipLayoutMode::Stacked,
            face_ratio: 0.4,
            face_crop: None,
            face_anchor: FaceAnchor::TopLeft,
            face_context_scale: 1.5,
            face_zoom: 1.0,
        };

        let mut edge_face = ClipLayoutHints::default();
        edge_face.face_box = Some(NormalizedRect {
            x: 0.02,
            y: 0.05,
            w: 0.12,
            h: 0.12,
        });
        let decision = decide_stacked_layout(&layout, &edge_face, None);
        assert!(decision.layout_is_stacked);

        let mut centered_face = ClipLayoutHints::default();
        centered_face.face_box = Some(NormalizedRect {
            x: 0.35,
            y: 0.35,
            w: 0.2,
            h: 0.2,
        });
        let decision = decide_stacked_layout(&layout, &centered_face, None);
        assert!(!decision.layout_is_stacked);
        assert!(decision.fullscreen_fill);
    }

    #[test]
    fn low_resource_overrides_apply_and_restore() {
        low_resource::reset_low_resource_state_for_tests();
        let mut env = EnvGuard::new();
        env.set("CLIP_LOW_RESOURCES", "1");
        env.set("CLIP_FACE_TRACK", "1");

        refresh_low_resource_state();
        assert_eq!(std::env::var("CLIP_FACE_TRACK").ok().as_deref(), Some("0"));

        env.set("CLIP_LOW_RESOURCES", "0");
        refresh_low_resource_state();
        assert_eq!(std::env::var("CLIP_FACE_TRACK").ok().as_deref(), Some("1"));
    }

    #[test]
    fn cuda_device_order_defaults_to_pci_bus_id() {
        let mut env = EnvGuard::new();
        env.remove("CUDA_DEVICE_ORDER");
        ensure_cuda_device_order();
        assert_eq!(
            std::env::var("CUDA_DEVICE_ORDER").ok().as_deref(),
            Some("PCI_BUS_ID")
        );
    }

    #[test]
    fn auto_assigns_distinct_gpus_for_whisper_and_ffmpeg() {
        let mut env = EnvGuard::new();
        env.set("GPU_VRAM_OVERRIDE_LIST", "0:6000,1:4000");
        env.set("GPU_VRAM_RESERVE_MB", "0");
        env.set("WHISPER_MIN_FREE_VRAM_MB", "1000");
        env.set("FFMPEG_MIN_FREE_VRAM_MB", "500");
        env.set("FFMPEG_ENCODER", "libx264");
        env.remove("WHISPER_GPU");
        env.remove("WHISPER_GPU_DEVICE");
        env.remove("FFMPEG_HWACCEL");
        env.remove("FFMPEG_HWACCEL_DEVICE");

        auto_assign_gpus_for_tools();

        assert_eq!(std::env::var("WHISPER_GPU").ok().as_deref(), Some("1"));
        assert_eq!(
            std::env::var("WHISPER_GPU_DEVICE").ok().as_deref(),
            Some("0")
        );
        assert_eq!(std::env::var("FFMPEG_HWACCEL").ok().as_deref(), Some("cuda"));
        assert_eq!(
            std::env::var("FFMPEG_HWACCEL_DEVICE").ok().as_deref(),
            Some("1")
        );
    }

    #[test]
    fn non_video_files_use_subdir() -> Result<()> {
        let stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_millis();
        let dir = std::env::temp_dir().join(format!("autoclip_nv_{stamp}"));
        fs::create_dir_all(&dir)?;
        let dir_str = dir.to_string_lossy().to_string();

        let mut env = EnvGuard::new();
        env.set("WHISPER_WORKER_STATUS_PATH", "");
        env.set("WHISPER_WORKER_MEDIA_URL_PATH", "");

        let status_path = whisper_worker_status_path(&dir_str);
        let media_path = whisper_worker_media_url_path(&dir_str);
        let expected_dir = dir.join(storage_utils::NON_VIDEO_SUBDIR);
        assert_eq!(
            status_path,
            expected_dir.join(".whisper_wake_status.json")
        );
        assert_eq!(
            media_path,
            expected_dir.join(".whisper_wake_media_url.txt")
        );

        log_run_event(&dir_str, "test");
        assert!(expected_dir.join("autoclip_run.log").exists());

        let _ = next_output_path(&dir_str, "clip")?;
        assert!(expected_dir.join(".clip_counter").exists());

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn low_resource_face_box_still_uses_kalman_for_fullscreen_track() {
        let mut env = EnvGuard::new();
        env.set("CLIP_LOW_RESOURCES", "1");
        refresh_low_resource_state();
        env.set("CLIP_FACE_BOX", "0.1,0.1,0.8,0.8");

        let hints = read_clip_layout_hints();
        let face_box = hints.face_box.expect("expected face box");
        let track = clip_layout::synthesize_face_track(face_box);
        let FilterGraph::Vf(chain) =
            build_tracked_full_frame_fill_filter_graph(1080, 1920, &track)
                .expect("expected tracked full-frame graph")
        else {
            panic!("expected Vf filter graph");
        };
        assert!(
            chain.contains("between(t"),
            "expected kalman-driven tracked crop expression"
        );

        env.set("CLIP_LOW_RESOURCES", "0");
        refresh_low_resource_state();
    }

    #[test]
    fn low_resource_guess_rejects_small_center_face() {
        let hints = ClipLayoutHints {
            face_box: Some(NormalizedRect {
                x: 0.42,
                y: 0.24,
                w: 0.03,
                h: 0.05,
            }),
            ..ClipLayoutHints::default()
        };
        let guess = guess_gameplay_low_resource(&hints, 0.40);
        assert_eq!(guess.map(|(value, _, _)| value), Some(false));
    }

    #[test]
    fn low_resource_guess_accepts_small_edge_face() {
        let hints = ClipLayoutHints {
            face_box: Some(NormalizedRect {
                x: 0.02,
                y: 0.02,
                w: 0.04,
                h: 0.05,
            }),
            ..ClipLayoutHints::default()
        };
        let guess = guess_gameplay_low_resource(&hints, 0.40);
        assert_eq!(guess.map(|(value, _, _)| value), Some(true));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn captions_embed_subtitles_in_low_resource_mode() -> Result<()> {
        if !tool_available("ffmpeg") || !tool_available("ffprobe") {
            eprintln!("skipping: ffmpeg/ffprobe not available");
            return Ok(());
        }
        let _guard = CaptionFlagGuard::new();
        let stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_millis();
        let dir = std::env::temp_dir().join(format!("autoclip_caption_test_{stamp}"));
        fs::create_dir_all(&dir)?;
        let input = dir.join("input.mp4");
        let output = dir.join("output.mp4");
        generate_test_video(&input)?;

        let mut env = EnvGuard::new();
        env.set("CLIP_LOW_RESOURCES", "1");
        env.set("CLIP_CAPTIONS", "1");
        env.set("CLIP_CLOSED_CAPTIONS", "1");
        env.set("CLIP_TEST_TRANSCRIPT", "hello world from test");
        env.set("CLIP_LAYOUT", "full");
        env.set("CLIP_DETECT", "0");
        env.set("CLIP_FACE_TRACK", "0");
        env.set("CLIP_LIVE_FAST", "0");
        env.set("FFMPEG_ENCODER", "libx264");
        env.set("FFMPEG_HWACCEL", "none");
        if let Some(font) = default_caption_font() {
            env.set("CLIP_CAPTIONS_FONT", &font);
        }

        run_ffmpeg_internal(
            FfmpegRenderSpec::new(input.to_string_lossy().as_ref(), &output, 320, 180)
                .with_duration(Some(1.0))
                .with_regen_pts(true),
        )
        .await?;

        assert!(
            output_has_subtitles(&output)?,
            "expected subtitle stream in output"
        );
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn open_captions_are_visibly_rendered() -> Result<()> {
        if !tool_available("ffmpeg") {
            eprintln!("skipping: ffmpeg not available");
            return Ok(());
        }
        let _guard = CaptionFlagGuard::new();
        let stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_millis();
        let dir = std::env::temp_dir().join(format!("autoclip_open_caps_{stamp}"));
        fs::create_dir_all(&dir)?;
        let input = dir.join("input.mp4");
        let baseline = dir.join("baseline.mp4");
        let output = dir.join("captions.mp4");
        generate_test_video(&input)?;

        let mut env = EnvGuard::new();
        env.set("CLIP_LAYOUT", "full");
        env.set("CLIP_DETECT", "0");
        env.set("CLIP_FACE_TRACK", "0");
        env.set("CLIP_LIVE_FAST", "0");
        env.set("FFMPEG_ENCODER", "libx264");
        env.set("FFMPEG_HWACCEL", "none");
        env.set("CLIP_CAPTIONS_COLOR", "white");
        env.set("CLIP_CAPTIONS_OUTLINE", "0");
        env.set("CLIP_CAPTIONS_POSITION", "chest");
        env.set("CLIP_CAPTIONS_CHEST_RATIO", "0.5");
        env.set("CLIP_CAPTIONS_SIZE", "0.25");
        env.set("CLIP_TEST_TRANSCRIPT", "hello world");
        if let Some(font) = default_caption_font() {
            env.set("CLIP_CAPTIONS_FONT", &font);
        }

        env.set("CLIP_CAPTIONS", "0");
        env.set("CLIP_CLOSED_CAPTIONS", "0");
        run_ffmpeg_internal(
            FfmpegRenderSpec::new(input.to_string_lossy().as_ref(), &baseline, 320, 180)
                .with_duration(Some(1.0))
                .with_regen_pts(true),
        )
        .await?;

        env.set("CLIP_CAPTIONS", "1");
        run_ffmpeg_internal(
            FfmpegRenderSpec::new(input.to_string_lossy().as_ref(), &output, 320, 180)
                .with_duration(Some(1.0))
                .with_regen_pts(true),
        )
        .await?;

        let base_y = sample_frame_yavg(&baseline, 0.25)?;
        let cap_y = sample_frame_yavg(&output, 0.25)?;
        assert!(
            cap_y > base_y + 1.0,
            "expected captions to raise average luma (baseline {base_y:.2}, captions {cap_y:.2})"
        );
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn captions_font_size_scales_down_for_long_words() {
        let cfg = CaptionConfig {
            position: CaptionPosition::Margin,
            font: None,
            font_size: 80.0,
            color: "white".to_string(),
            outline_color: "black".to_string(),
            outline: 2,
            min_word_secs: 0.12,
            max_words: 300,
            chest_ratio: 0.65,
            margin_offset_px: 0.0,
            debug: false,
        };
        let payload = TranscriptPayload {
            text: "supercalifragilisticexpialidocious".to_string(),
            words: vec![WordTiming {
                text: "supercalifragilisticexpialidocious".to_string(),
                norm: "supercalifragilisticexpialidocious".to_string(),
                t0: 0.0,
                t1: 1.0,
            }],
        };
        let adjusted = adjust_caption_font_size(&cfg, &payload, 1080);
        assert!(
            adjusted.font_size < cfg.font_size,
            "expected font size to shrink for long words"
        );
    }

    #[test]
    fn captions_font_size_accounts_for_width_ratio() {
        let mut env = EnvGuard::new();
        env.set("CLIP_CAPTIONS_WIDTH_RATIO", "0.6");
        env.set("CLIP_CAPTIONS_BUCKET_SECS", "0");
        let cfg = CaptionConfig {
            position: CaptionPosition::Margin,
            font: None,
            font_size: 80.0,
            color: "white".to_string(),
            outline_color: "black".to_string(),
            outline: 2,
            min_word_secs: 0.12,
            max_words: 300,
            chest_ratio: 0.65,
            margin_offset_px: 0.0,
            debug: false,
        };
        let payload = TranscriptPayload {
            text: "SUPERCALIFRAGILISTICEXPIALIDOCIOUS".to_string(),
            words: vec![WordTiming {
                text: "SUPERCALIFRAGILISTICEXPIALIDOCIOUS".to_string(),
                norm: "supercalifragilisticexpialidocious".to_string(),
                t0: 0.0,
                t1: 1.0,
            }],
        };
        let adjusted = adjust_caption_font_size(&cfg, &payload, 1080);
        assert!(
            adjusted.font_size < cfg.font_size,
            "expected width ratio to reduce font size"
        );
    }

    #[test]
    fn captions_ass_scales_per_cue_when_unlocked() {
        let mut env = EnvGuard::new();
        env.set("CLIP_CAPTIONS_SCALE_LOCK", "0");
        env.set("CLIP_CAPTIONS_BUCKET_SECS", "0");
        let cfg = CaptionConfig {
            position: CaptionPosition::Margin,
            font: None,
            font_size: 80.0,
            color: "white".to_string(),
            outline_color: "black".to_string(),
            outline: 2,
            min_word_secs: 0.12,
            max_words: 300,
            chest_ratio: 0.65,
            margin_offset_px: 0.0,
            debug: false,
        };
        let payload = TranscriptPayload {
            text: "hi supercalifragilisticexpialidocious".to_string(),
            words: vec![
                WordTiming {
                    text: "hi".to_string(),
                    norm: "hi".to_string(),
                    t0: 0.0,
                    t1: 0.4,
                },
                WordTiming {
                    text: "supercalifragilisticexpialidocious".to_string(),
                    norm: "supercalifragilisticexpialidocious".to_string(),
                    t0: 0.6,
                    t1: 1.4,
                },
            ],
        };
        let ass = build_ass_from_payload_with_limits(&payload, Some(2.0), &cfg, 120.0, 1080, 1920)
            .expect("expected ass output");
        let mut sizes: Vec<u32> = Vec::new();
        let mut rest = ass.as_str();
        while let Some(idx) = rest.find("\\fs") {
            let after = &rest[idx + 3..];
            let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(value) = digits.parse::<u32>() {
                sizes.push(value);
            }
            rest = after;
        }
        assert!(
            ass.contains("\\pos(") && ass.contains("\\an8"),
            "expected ASS overrides to center captions"
        );
        sizes.sort_unstable();
        sizes.dedup();
        assert!(
            sizes.len() > 1,
            "expected per-cue font sizes to differ when unlocked"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn captions_fallback_when_font_invalid() -> Result<()> {
        if !tool_available("ffmpeg") {
            eprintln!("skipping: ffmpeg not available");
            return Ok(());
        }
        let _guard = CaptionFlagGuard::new();
        let stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_millis();
        let dir = std::env::temp_dir().join(format!("autoclip_bad_font_{stamp}"));
        fs::create_dir_all(&dir)?;
        let input = dir.join("input.mp4");
        let output = dir.join("output.mp4");
        let fake_font = dir.join("fake_font.ttf");
        fs::write(&fake_font, b"not a font")?;
        generate_test_video(&input)?;

        let mut env = EnvGuard::new();
        env.set("CLIP_LAYOUT", "full");
        env.set("CLIP_DETECT", "0");
        env.set("CLIP_FACE_TRACK", "0");
        env.set("CLIP_LIVE_FAST", "0");
        env.set("FFMPEG_ENCODER", "libx264");
        env.set("FFMPEG_HWACCEL", "none");
        env.set("CLIP_CAPTIONS", "1");
        env.set("CLIP_CLOSED_CAPTIONS", "0");
        env.set("CLIP_CAPTIONS_FONT", fake_font.to_string_lossy().as_ref());
        env.set("CLIP_TEST_TRANSCRIPT", "hello world");

        run_ffmpeg_internal(
            FfmpegRenderSpec::new(input.to_string_lossy().as_ref(), &output, 320, 180)
                .with_duration(Some(1.0))
                .with_regen_pts(true),
        )
        .await?;

        assert!(output.exists(), "expected output even with bad font");
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn reprocess_continues_on_invalid_media_when_enabled() -> Result<()> {
        if !tool_available("ffmpeg") {
            eprintln!("skipping: ffmpeg not available");
            return Ok(());
        }
        let stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_millis();
        let dir = std::env::temp_dir().join(format!("autoclip_reprocess_bad_{stamp}"));
        fs::create_dir_all(&dir)?;
        let input = dir.join("broken.mp4");
        fs::write(&input, b"not a real mp4")?;

        let mut cfg = Config::example();
        cfg.save_path = dir.join("out").to_string_lossy().to_string();
        cfg.file_name_stub = "clip".to_string();

        let mut env = EnvGuard::new();
        env.set("CLIP_REPROCESS_CONTINUE_ON_ERROR", "1");
        env.set("CLIP_REPROCESS_CHUNK_SECS", "0");
        env.set("CLIP_REPROCESS_FAST", "1");
        env.set("CLIP_TEST_TRANSCRIPT", "hello world");
        env.set("CLIP_LAYOUT", "full");
        env.set("CLIP_DETECT", "0");
        env.set("CLIP_FACE_TRACK", "0");
        env.set("CLIP_LIVE_FAST", "0");
        env.set("FFMPEG_ENCODER", "libx264");
        env.set("FFMPEG_HWACCEL", "none");

        let result = run_reprocess_ts_with_config(input.to_string_lossy().as_ref(), &cfg).await;
        assert!(result.is_ok(), "expected reprocess to continue on error");
        let _ = fs::remove_dir_all(&dir);
        Ok(())
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
        let master = client
            .fetch_master_with_headers(&master_url, None, None, None)
            .await?;
        let media_url = client.highest_variant_url(&master_url, &master)?;

        let media = client.fetch_media(media_url.as_str()).await?;
        let first = media
            .segments
            .first()
            .map(|s| &s.uri)
            .context("media playlist has no segments")?;
        let playlist_url = Url::parse(media_url.as_str()).context("invalid playlist url")?;
        let segment_bytes = client
            .fetch_segment_from_playlist(&playlist_url, first)
            .await?;
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

        let media = client.fetch_media(media_url.as_str()).await?;
        let first = media
            .segments
            .first()
            .map(|s| &s.uri)
            .context("media playlist has no segments")?;
        let playlist_url = Url::parse(media_url.as_str()).context("invalid playlist url")?;
        let segment_bytes = client
            .fetch_segment_from_playlist(&playlist_url, first)
            .await?;
        assert!(!segment_bytes.is_empty(), "segment should contain data");
        Ok(())
    }

    #[test]
    fn ffmpeg_gpu_filter_keeps_explicit_format_after_hwdownload() {
        let (vf_cpu, vf_gpu_sw, vf_gpu_hw) = build_ffmpeg_filters(1080, 1920);

        assert_eq!(
            vf_cpu,
            "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
        );
        assert_eq!(
            vf_gpu_sw,
            "format=yuv420p,hwupload_cuda,scale_cuda=w=1080:h=1920:force_original_aspect_ratio=decrease:format=nv12:interp_algo=lanczos,hwdownload,format=yuv420p,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
        );
        assert!(
            vf_gpu_sw.contains("hwdownload,format=yuv420p"),
            "explicit format after hwdownload prevents auto-inserted auto_scale filter errors"
        );
        assert!(
            !vf_gpu_hw.contains("hwupload_cuda"),
            "cuda decode path should not attempt an extra upload"
        );
    }

    #[test]
    fn truncate_str_adds_ellipsis() {
        assert_eq!(truncate_str("short", 10), "short");
        assert_eq!(truncate_str("0123456789", 10), "0123456789");
        assert_eq!(truncate_str("0123456789A", 10), "0123456...");
    }

    #[test]
    fn normalize_title_whitespace_collapses() {
        let input = "  hello   world \n\t good   bye  ";
        assert_eq!(normalize_title_whitespace(input), "hello world good bye");
    }

    #[test]
    fn sanitize_title_for_filename_basic() {
        assert_eq!(sanitize_title_for_filename("Hello, World!"), "hello_world");
        assert_eq!(sanitize_title_for_filename("Hello-World"), "hello-world");
        assert_eq!(sanitize_title_for_filename("!!!"), "");
    }

    #[test]
    fn trim_transcript_trims_and_truncates() {
        assert_eq!(trim_transcript("  ok  ", 10), "ok");
        assert_eq!(trim_transcript(" 0123456789A ", 10), "0123456...");
    }

    #[test]
    fn fallback_title_from_transcript_respects_limits() {
        assert_eq!(fallback_title_from_transcript("   ", 10), "");
        assert_eq!(fallback_title_from_transcript("hello world", 20), "hello world");
        assert_eq!(fallback_title_from_transcript("hello world", 8), "hello...");
    }

    #[test]
    fn extract_twitch_login_from_url() {
        assert_eq!(
            crate::hls_client::extract_twitch_login("https://www.twitch.tv/rynn?twitch5=0"),
            Some("rynn".to_string())
        );
        assert_eq!(
            crate::hls_client::extract_twitch_login("https://player.twitch.tv/?channel=rynn"),
            Some("rynn".to_string())
        );
        assert_eq!(
            crate::hls_client::extract_twitch_login("https://www.twitch.tv/directory"),
            None
        );
    }
}
