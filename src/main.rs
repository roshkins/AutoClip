//! AutoClip MVP stub.
//! Current behavior: headless-grab an m3u8 from a page (Kick-style), pick the
//! top variant, and save a 30s vertical clip via FFmpeg. Everything else (rolling
//! buffering, wake-word detection, VRAM budgeting) is logged-only scaffolding.

use anyhow::{Context, Result};
use m3u8_rs::{MasterPlaylist, MediaPlaylist, VariantStream};
use reqwest::{header, Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Stdio};
use std::time::{Duration, Instant, SystemTime};
use tokio::process::Command;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::sync::Semaphore;
use tokio::time::sleep;
use url::Url;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::collections::HashMap;

mod clip_detect;
mod clip_gameplay;
mod clip_layout;
mod gpu;
mod loading;
mod profile;
mod rolling_buffer;
use clip_detect::{detect_layout_hints, read_clip_detect_config, run_face_threshold_sweep};
use clip_gameplay::{read_clip_gameplay_config, ClipGameplayDetector};
use clip_layout::{
    build_face_only_filter_graph, build_full_frame_fill_filter_graph,
    build_stacked_filter_graph, build_tracked_full_frame_fill_filter_graph,
    read_clip_layout_config, read_clip_layout_hints, resolve_stacked_layout_dims, ClipLayoutMode,
    FilterGraph, NormalizedPoint, NormalizedRect, StackedLayoutDims,
};
use profile::profile_span;
use rolling_buffer::RollingBuffer;
#[cfg(feature = "whisper")]
mod stream_audio_wake;
#[cfg(not(feature = "whisper"))]
#[path = "stream_audio_wake_stub.rs"]
mod stream_audio_wake;
use stream_audio_wake::{
    detect_wake_in_file, run_wake_worker_mic, run_wake_worker_stream,
    start_mic_wake_with_ffmpeg, start_stream_wake_from_hls, transcribe_clip_audio,
    TranscriptPayload, WordTiming,
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
        spawn_ctrl_c_handler(stop.clone());
        let mut last_detect_instant: Option<Instant> = None;
        let wake_start = Arc::new(Mutex::new(Instant::now()));
        let _start_instant = *wake_start.lock().expect("wake start lock poisoned");
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
                    let start_instant = *monitor_start.lock().expect("wake start lock poisoned");
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
        let mut live_config = LiveConfigState::from_env();
        if let Some(state) = live_config.as_mut() {
            state.maybe_refresh();
        }

        let hls = HlsClient::new()?;
        let (master_url, master) = hls.fetch_master_from_page(page_url).await?;
        let initial_media_url = hls.highest_variant_url(&master_url, &master)?;
        println!("tracking variant: {}", initial_media_url);
        let media_url = Arc::new(Mutex::new(initial_media_url));

        let mut seen: HashSet<String> = HashSet::new();
        let mut stream_time = Duration::ZERO;
        let mut detect_stream_time: Option<Duration> = None;
        let mut after_remaining: Option<Duration> = None;
        let refractory = std::env::var("WAKE_REFRACTORY_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(12));
        let mut refractory_until: Option<Instant> = None;
        let latency_refresh_threshold = Duration::from_secs(240);
        let mut last_latency_refresh: Option<Instant> = None;
        let word_refresh_threshold = Duration::from_secs(30);
        let word_refresh_cooldown = Duration::from_secs(30);
        let mut last_word_refresh: Option<Instant> = None;
        let whisper_isolate = whisper_isolate_enabled();
        let status_poll = whisper_worker_status_poll();
        let worker_status_path = whisper_worker_status_path();
        let worker_media_url_path = whisper_worker_media_url_path();
        if whisper_isolate && !self.config.use_mic_for_wake {
            let guard = media_url.lock().expect("media url lock poisoned");
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
        let mut _whisper_worker: Option<std::process::Child> = None;
        if whisper_isolate {
            let worker_start = Instant::now();
            *wake_start.lock().expect("wake start lock poisoned") = worker_start;
            let _reader = spawn_wake_status_reader(
                worker_status_path.clone(),
                status_poll,
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
            _whisper_worker = Some(spawn_whisper_worker(
                mode,
                &worker_status_path,
                media_path,
                &wake_phrases,
                self.config.log_raw_wake,
            )?);
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
            let start_instant = *wake_start.lock().expect("wake start lock poisoned");
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
            let start_instant = *wake_start.lock().expect("wake start lock poisoned");
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

        'stream_loop: loop {
            if stop.load(Ordering::Relaxed)
                && !fired.load(Ordering::Relaxed)
                && after_remaining.is_none()
            {
                break;
            }
            if let Some(state) = live_config.as_mut() {
                state.maybe_refresh();
            }

            let latency = Duration::from_nanos(max_latency_ns.load(Ordering::Relaxed));
            if latency >= latency_refresh_threshold {
                let now = Instant::now();
                let can_refresh = last_latency_refresh
                    .map(|t| now.duration_since(t) > Duration::from_secs(30))
                    .unwrap_or(true);
                if can_refresh {
                    last_latency_refresh = Some(now);
                    eprintln!(
                        "processing latency {:.1}s exceeded {:.1}s; refreshing m3u8 via headless",
                        latency.as_secs_f32(),
                        latency_refresh_threshold.as_secs_f32()
                    );
                    match hls.refresh_media_url_from_page_headless(page_url).await {
                        Ok(new_url) => {
                            let mut guard = media_url
                                .lock()
                                .expect("media url lock poisoned while refreshing");
                            *guard = new_url.clone();
                            seen.clear();
                            if whisper_isolate && !self.config.use_mic_for_wake {
                                if let Err(err) =
                                    write_media_url_file(&worker_media_url_path, &new_url)
                                {
                                    eprintln!(
                                        "whisper worker: failed to write media url: {err:#}"
                                    );
                                }
                            }
                            eprintln!("refreshed variant: {}", new_url);
                        }
                        Err(err) => {
                            eprintln!("headless refresh failed: {err:#}");
                        }
                    }
                }
            }
            if !self.config.use_mic_for_wake
                && !fired.load(Ordering::Relaxed)
                && after_remaining.is_none()
            {
                let now = Instant::now();
                let last_word_val = last_word_ns.load(Ordering::Relaxed);
                let base_start = *wake_start.lock().expect("wake start lock poisoned");
                let last_word_at = if last_word_val == u64::MAX {
                    base_start
                } else {
                    base_start + Duration::from_nanos(last_word_val)
                };
                let since_word = now.saturating_duration_since(last_word_at);
                if since_word >= word_refresh_threshold {
                    let can_refresh = last_word_refresh
                        .map(|t| now.duration_since(t) >= word_refresh_cooldown)
                        .unwrap_or(true);
                    if can_refresh {
                        last_word_refresh = Some(now);
                        eprintln!(
                            "no words detected for {:.1}s; refreshing m3u8 via headless",
                            since_word.as_secs_f32()
                        );
                        match hls.refresh_media_url_from_page_headless(page_url).await {
                            Ok(new_url) => {
                                let mut guard = media_url
                                    .lock()
                                    .expect("media url lock poisoned while refreshing");
                                *guard = new_url.clone();
                                seen.clear();
                                eprintln!("refreshed variant: {}", new_url);
                            }
                            Err(err) => {
                                eprintln!("headless refresh failed: {err:#}");
                            }
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

            let media_url_snapshot = {
                let guard = media_url
                    .lock()
                    .expect("media url lock poisoned while fetching playlist");
                guard.clone()
            };
            let playlist = match hls.fetch_media(media_url_snapshot.as_str()).await {
                Ok(p) => p,
                Err(err) => {
                    if is_http_status(&err, StatusCode::FORBIDDEN) {
                        eprintln!("media playlist returned 403; refreshing via headless");
                        match hls.refresh_media_url_from_page_headless(page_url).await {
                            Ok(new_url) => {
                            let mut guard = media_url
                                .lock()
                                .expect("media url lock poisoned while refreshing");
                            *guard = new_url.clone();
                            seen.clear();
                            if whisper_isolate && !self.config.use_mic_for_wake {
                                if let Err(err) =
                                    write_media_url_file(&worker_media_url_path, &new_url)
                                {
                                    eprintln!(
                                        "whisper worker: failed to write media url: {err:#}"
                                    );
                                }
                            }
                            eprintln!("refreshed variant: {}", new_url);
                            continue 'stream_loop;
                        }
                            Err(refresh_err) => {
                                eprintln!("headless refresh failed: {refresh_err:#}");
                            }
                        }
                    }
                    eprintln!("failed to fetch media playlist: {err:#}; retrying");
                    sleep(Duration::from_millis(800)).await;
                    continue;
                }
            };

            let mut made_progress = false;

            for seg in &playlist.segments {
                let uri = seg.uri.clone();
                if !seen.insert(uri.clone()) {
                    continue;
                }

                match hls.fetch_segment_from_playlist(&media_url_snapshot, &uri).await {
                    Ok(bytes) => {
                        let seg_dur_playlist = Duration::from_secs_f32(seg.duration as f32);
                        let seg_dur = choose_segment_duration(
                            pts_duration_from_ts(&bytes),
                            seg_dur_playlist,
                        );
                        buffer.push(bytes, seg_dur);
                        stream_time = stream_time.saturating_add(seg_dur);
                        made_progress = true;
                    }
                    Err(err) => {
                        if is_http_status(&err, StatusCode::FORBIDDEN) {
                            eprintln!("segment fetch returned 403; refreshing via headless");
                            match hls.refresh_media_url_from_page_headless(page_url).await {
                                Ok(new_url) => {
                                    let mut guard = media_url
                                        .lock()
                                        .expect("media url lock poisoned while refreshing");
                                    *guard = new_url;
                                    seen.clear();
                                    if whisper_isolate && !self.config.use_mic_for_wake {
                                        if let Err(err) = write_media_url_file(
                                            &worker_media_url_path,
                                            &*guard,
                                        ) {
                                            eprintln!(
                                                "whisper worker: failed to write media url: {err:#}"
                                            );
                                        }
                                    }
                                    continue 'stream_loop;
                                }
                                Err(refresh_err) => {
                                    eprintln!("headless refresh failed: {refresh_err:#}");
                                }
                            }
                        }
                        eprintln!("failed to fetch segment {}: {err:#}", uri);
                        continue;
                    }
                }

                if fired.load(Ordering::Relaxed) && after_remaining.is_none() {
                    if refractory_until.map(|t| Instant::now() < t).unwrap_or(false) {
                        // Ignore rapid re-triggers until cooldown expires.
                        continue;
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
                    let base_start = *wake_start.lock().expect("wake start lock poisoned");
                    let detected_instant = if ns != u64::MAX {
                        base_start + Duration::from_nanos(ns)
                    } else {
                        Instant::now()
                    };
                    last_detect_instant = Some(detected_instant);
                    let latency = Instant::now().saturating_duration_since(detected_instant);
                    detect_stream_time = Some(stream_time.saturating_sub(age_audio));
                    after_remaining = Some(after_tail);
                    println!(
                        "wake detected; capturing tail to place wake at 50s into clip (latency ~{:.1}s, buffer ~{:.1}s, cooldown {:?})",
                        latency.as_secs_f32(),
                        buffer_window.as_secs_f32(),
                        refractory
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
                    } else {
                        let save_future = async move {
                            fs::write(&ts_path, &snapshot).context("writing buffered TS snapshot")?;
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
                            let detected_at = last_detect_instant.map(|t| t.elapsed().as_secs_f32());
                            println!(
                                "wrote wakeword clip: {} (duration ~{:.1}s) | wake at ~50.0s into clip | detect_elapsed_since_save_start={:?}",
                                output_path.display(),
                                clip_len.as_secs_f32(),
                                detected_at
                            );
                            Ok::<(), anyhow::Error>(())
                        };

                        let handle = tokio::spawn(async move {
                            if let Err(err) = save_future.await {
                                eprintln!("failed to persist wakeword clip: {err:#}");
                            }
                        });
                        pending_saves.push(handle);
                    }

                    after_remaining = None;
                    detect_stream_time = None;
                    fired.store(false, Ordering::Relaxed);
                    detect_ns.store(u64::MAX, Ordering::Relaxed);
                    continue;
                }
                if !made_progress {
                    // If the playlist is stale, still count down so we don't spin forever.
                    *rem_mut = rem_mut.saturating_sub(poll_interval);
                }
            }

            if fired.load(Ordering::Relaxed) && after_remaining.is_none() {
                if refractory_until.map(|t| Instant::now() < t).unwrap_or(false) {
                    continue;
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
                let base_start = *wake_start.lock().expect("wake start lock poisoned");
                let detected_instant = if ns != u64::MAX {
                    base_start + Duration::from_nanos(ns)
                } else {
                    Instant::now()
                };
                last_detect_instant = Some(detected_instant);
                let latency = Instant::now().saturating_duration_since(detected_instant);
                println!(
                    "wake detected; capturing tail to place wake at 50s into clip (latency ~{:.1}s, buffer ~{:.1}s, cooldown {:?})",
                    latency.as_secs_f32(),
                    buffer_window.as_secs_f32(),
                    refractory
                );
                // Wake fired but we have not yet started counting; ensure we do.
                after_remaining = Some(after_tail);
            }

            sleep(poll_interval).await;
        }

        if let Some(mut worker) = _whisper_worker {
            let _ = worker.kill();
        }

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

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn write_wake_worker_status(path: &Path, status: &WakeWorkerStatus) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let payload = serde_json::to_string(status).context("serializing wake status")?;
    fs::write(&tmp, payload).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("renaming {}", path.display()))?;
    Ok(())
}

fn write_media_url_file(path: &Path, url: &Url) -> Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, url.as_str()).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("renaming {}", path.display()))?;
    Ok(())
}

fn whisper_isolate_enabled() -> bool {
    std::env::var("WHISPER_ISOLATE")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn whisper_worker_status_path() -> PathBuf {
    std::env::var("WHISPER_WORKER_STATUS_PATH")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(".whisper_wake_status.json")
        })
}

fn whisper_worker_media_url_path() -> PathBuf {
    std::env::var("WHISPER_WORKER_MEDIA_URL_PATH")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(".whisper_wake_media_url.txt")
        })
}

fn whisper_worker_status_poll() -> Duration {
    std::env::var("WHISPER_WORKER_STATUS_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or_else(|| Duration::from_millis(500))
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
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
    detect_ns: Arc<AtomicU64>,
    audio_ns: Arc<AtomicU64>,
    last_word_ns: Arc<AtomicU64>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut last_seen_ms = 0u64;
        while !stop.load(Ordering::Relaxed) {
            if fired.load(Ordering::Relaxed) {
                break;
            }
            match fs::read_to_string(&path) {
                Ok(contents) => {
                    if let Ok(status) = serde_json::from_str::<WakeWorkerStatus>(&contents) {
                        if status.updated_unix_ms >= last_seen_ms {
                            last_seen_ms = status.updated_unix_ms;
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
    if let Some(pts_dur) = pts {
        let secs = pts_dur.as_secs_f32();
        if secs.is_finite() && secs > 0.0 {
            return pts_dur;
        }
    }
    playlist
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

fn is_http_status(err: &anyhow::Error, status: StatusCode) -> bool {
    err.downcast_ref::<HttpStatusError>()
        .map(|e| e.status == status)
        .unwrap_or(false)
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
        self.parse_master_or_media(url, &body)
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
        self.parse_master_or_media(url, &body)
    }

    /// Fetch a page via headless Playwright, extract the first m3u8 URL, and parse
    /// it as a master playlist. Adds Referer/Origin tied to the page URL. Forwards
    /// COOKIE_HEADER / KICK_COOKIE / TIKTOK_COOKIE / TWITCH_COOKIE if present. If
    /// M3U8_URL_OVERRIDE is set, use
    /// that master URL directly instead of headless extraction.
    pub async fn fetch_master_from_page(&self, page_url: &str) -> Result<(String, MasterPlaylist)> {
        let env_cookie = collect_env_cookies();
        let origin = origin_for_page(page_url).unwrap_or_else(|| "https://kick.com".to_string());

        let is_tiktok = page_url.contains("tiktok.com");
        let is_twitch = page_url.contains("twitch.tv");
        let headless_script = headless_script_for_page(page_url);

        if let Ok(override_url) = std::env::var("M3U8_URL_OVERRIDE") {
            let master = self
                .fetch_master_with_headers(&override_url, Some(page_url), Some(&origin), env_cookie.as_deref())
                .await?;
            return Ok((override_url, master));
        }

        if is_tiktok {
            if let Some(res) = self
                .try_fetch_tiktok_master(page_url, env_cookie.as_deref())
                .await?
            {
                return Ok(res);
            }
            eprintln!("TikTok HTTP discovery failed or stream offline; falling back to headless");
        }
        if is_twitch {
            if let Some(res) = self
                .try_fetch_twitch_master(page_url, env_cookie.as_deref())
                .await?
            {
                return Ok(res);
            }
            eprintln!("Twitch HTTP discovery failed or stream offline; falling back to headless");
        }

        self.fetch_master_with_headless(page_url, env_cookie.as_deref(), &headless_script)
            .await
    }

    /// Force a headless discovery pass to refresh the master playlist.
    pub async fn fetch_master_from_page_headless(
        &self,
        page_url: &str,
    ) -> Result<(String, MasterPlaylist)> {
        let env_cookie = collect_env_cookies();
        let headless_script = headless_script_for_page(page_url);
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
        Ok((m3u8_url, parsed))
    }

    async fn try_fetch_tiktok_master(
        &self,
        page_url: &str,
        cookie_env: Option<&str>,
    ) -> Result<Option<(String, MasterPlaylist)>> {
        let origin = origin_for_page(page_url).unwrap_or_else(|| "https://www.tiktok.com".to_string());
        let cookie_header = cookie_env
            .filter(|c| !c.trim().is_empty())
            .map(|c| c.to_string());

        let html = match self
            .fetch_text_with_headers(page_url, Some(page_url), Some(&origin), cookie_header.as_deref())
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
                .fetch_tiktok_room_info(&room_id, page_url, Some(&origin), cookie_header.as_deref())
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
                .fetch_tiktok_live_detail_url(&room_id, page_url, Some(&origin), cookie_header.as_deref())
                .await?
            {
                candidates.push(("live_detail".to_string(), fallback_url));
            }
        }

        if candidates.is_empty() {
            eprintln!("TikTok: no HLS candidates from room info or detail API");
            return Ok(None);
        }

        let selected = pick_best_tiktok_hls(&candidates).unwrap_or_else(|| candidates[0].1.clone());
        let master = self
            .fetch_master_with_headers(&selected, Some(page_url), Some(&origin), cookie_header.as_deref())
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

        let origin = origin_for_page(page_url).unwrap_or_else(|| "https://www.twitch.tv".to_string());
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

        let client_id = std::env::var("TWITCH_CLIENT_ID").unwrap_or_else(|_| DEFAULT_TWITCH_CLIENT_ID.to_string());
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
            eprintln!("Twitch token status {} body: {}", status, truncate_str(&body, 500));
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
            eprintln!("fetch {} -> status {} body preview: {}", url, status, truncate_str(&text, 500));
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
            anyhow::bail!("TikTok room info returned status {status}");
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
            .map(|s| s.to_string()))
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

    /// Refresh the best variant URL from a page using headless discovery.
    pub async fn refresh_media_url_from_page_headless(&self, page_url: &str) -> Result<Url> {
        let (master_url, master) = self.fetch_master_from_page_headless(page_url).await?;
        self.highest_variant_url(&master_url, &master)
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
}

fn extract_tiktok_room_id(html: &str) -> Option<String> {
    for marker in ["id=\"SIGI_STATE\"", "id=\"sigi-persisted-data\"", "id=\"__UNIVERSAL_DATA_FOR_REHYDRATION__\""] {
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

fn extract_twitch_login(page_url: &str) -> Option<String> {
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
    let mut url = Url::parse(&format!("https://usher.ttvnw.net/api/channel/hls/{login}.m3u8"))
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

fn parse_resolution(res: &str) -> Option<(u32, u32)> {
    let parts: Vec<_> = res.split('x').collect();
    if parts.len() != 2 {
        return None;
    }
    let w = parts[0].parse().ok()?;
    let h = parts[1].parse().ok()?;
    Some((w, h))
}

fn rect_area(rect: NormalizedRect) -> f32 {
    let area = rect.w * rect.h;
    if !area.is_finite() {
        return 0.0;
    }
    area.max(0.0)
}

fn face_area_from_hints(hints: &clip_layout::ClipLayoutHints) -> Option<f32> {
    let rect = hints.face_box.or_else(|| {
        hints
            .face_track
            .as_ref()
            .and_then(|track| track.points.first().map(|p| p.rect))
    })?;
    Some(rect_area(rect))
}

fn face_rect_from_hints(hints: &clip_layout::ClipLayoutHints) -> Option<NormalizedRect> {
    hints
        .face_region
        .or(hints.face_box)
        .or_else(|| {
            hints
                .face_track
                .as_ref()
                .and_then(|track| track.points.first().map(|p| p.rect))
        })
}

fn rect_contains_rect(outer: NormalizedRect, inner: NormalizedRect, margin: f32) -> bool {
    let margin = margin.max(0.0);
    let left = outer.x - margin;
    let top = outer.y - margin;
    let right = outer.x + outer.w + margin;
    let bottom = outer.y + outer.h + margin;
    inner.x >= left
        && inner.y >= top
        && (inner.x + inner.w) <= right
        && (inner.y + inner.h) <= bottom
}

fn normalize_page_url(raw: &str) -> String {
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

fn sanitize_m3u8_url(raw: &str) -> String {
    raw.trim()
        .trim_matches('\'')
        .trim_matches('"')
        .trim_end_matches('\\')
        .to_string()
}

fn origin_for_page(page_url: &str) -> Option<String> {
    if let Ok(u) = Url::parse(page_url) {
        if let Some(host) = u.host_str() {
            return Some(format!("{}://{}", u.scheme(), host));
        }
    }
    None
}

fn truncate_str(s: &str, max: usize) -> String {
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
    let user_encoder = std::env::var("FFMPEG_ENCODER")
        .ok()
        .filter(|v| !v.trim().is_empty());
    let user_whisper = std::env::var("WHISPER_GPU")
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
        // Sort by memory already done in detect; pick biggest for whisper, second for ffmpeg if present.
        let whisper_gpu = gpus[0].0;
        let ffmpeg_gpu = if gpus.len() > 1 { gpus[1].0 } else { whisper_gpu };

        if user_whisper.is_none() {
            std::env::set_var("WHISPER_GPU", whisper_gpu.to_string());
            eprintln!("auto GPU assign for whisper: {}", whisper_gpu);
        }
        if user_hwaccel.is_none() {
            std::env::set_var("FFMPEG_HWACCEL", "cuda");
            std::env::set_var("FFMPEG_HWACCEL_DEVICE", ffmpeg_gpu.to_string());
        }
        if user_encoder.is_none() && ffmpeg_has_encoder("h264_nvenc") {
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
        if user_hwaccel.is_none() {
            eprintln!("auto GPU assign for ffmpeg: {}", ffmpeg_gpu);
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

fn ffmpeg_bin() -> String {
    std::env::var("FFMPEG_BIN").unwrap_or_else(|_| "ffmpeg".to_string())
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
    let output = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=index,memory.total")
        .arg("--format=csv,noheader,nounits")
        .output();

    let Ok(out) = output else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut gpus = Vec::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 2 {
            if let (Ok(idx), Ok(mem)) = (parts[0].parse::<u32>(), parts[1].parse::<u64>()) {
                gpus.push((idx, mem));
            }
        }
    }
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
    run_ffmpeg_internal(
        input_hls.as_str(),
        out_path,
        out_w,
        out_h,
        Some(30.0),
        None,
        false,
        false,
        None,
    )
    .await
}

#[derive(Clone, Copy, Debug)]
struct ClipTimingHints {
    estimated_total_secs: f32,
    detect_offset_secs: f32,
    before_secs: f32,
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
    run_ffmpeg_internal(
        input_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-utf8 input path"))?,
        out_path,
        out_w,
        out_h,
        Some(duration_secs),
        start_offset,
        true,
        true,
        None,
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
            "ts progress: {pct:5.1}% ({}/{})",
            format_progress_time(current),
            format_progress_time(total)
        );
    }
    format!("ts progress: {}", format_progress_time(current))
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn ensure_live_detect_budgets() {
    let mut face_secs = std::env::var("CLIP_FACE_BUDGET_SECS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0);
    let mut total_secs = std::env::var("CLIP_DETECT_BUDGET_SECS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0);

    let mut set_face = false;
    let mut set_total = false;

    if face_secs.is_none() {
        face_secs = Some(20.0);
        set_face = true;
    }
    if total_secs.is_none() {
        let base = face_secs.unwrap_or(20.0);
        total_secs = Some(base * 2.0);
        set_total = true;
    }

    if set_face {
        std::env::set_var("CLIP_FACE_BUDGET_SECS", format!("{:.0}", face_secs.unwrap()));
    }
    if set_total {
        std::env::set_var("CLIP_DETECT_BUDGET_SECS", format!("{:.0}", total_secs.unwrap()));
    }

    if set_face || set_total {
        let face = face_secs.unwrap_or(20.0);
        let total = total_secs.unwrap_or(face * 2.0);
        let gameplay = (total - face).max(0.0);
        eprintln!(
            "clip detect: live budgets face={:.0}s gameplay={:.0}s (total {:.0}s)",
            face, gameplay, total
        );
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaptionPosition {
    Margin,
    Chest,
}

#[derive(Clone, Debug)]
struct CaptionConfig {
    position: CaptionPosition,
    font: Option<String>,
    font_size: f32,
    color: String,
    outline_color: String,
    outline: u32,
    min_word_secs: f32,
    max_words: usize,
    chest_ratio: f32,
    margin_offset_px: f32,
    debug: bool,
}

#[derive(Clone, Debug)]
struct CaptionWord {
    text: String,
    start: f32,
    end: f32,
}

#[derive(Clone, Debug)]
struct ClipMetadata {
    title: String,
    description: String,
}

#[derive(Clone, Debug)]
struct SubtitleSpec {
    path: PathBuf,
}

#[derive(Debug)]
struct TempSubtitle {
    path: PathBuf,
}

impl Drop for TempSubtitle {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
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

fn captions_enabled() -> bool {
    std::env::var("CLIP_CAPTIONS")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn closed_captions_enabled() -> bool {
    std::env::var("CLIP_CLOSED_CAPTIONS")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn llm_enabled() -> bool {
    std::env::var("CLIP_LLM_ENABLE")
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

fn read_caption_config(out_h: u32) -> Option<CaptionConfig> {
    if !captions_enabled() {
        return None;
    }
    let position = std::env::var("CLIP_CAPTIONS_POSITION")
        .ok()
        .and_then(|v| parse_caption_position(&v))
        .unwrap_or(CaptionPosition::Margin);
    let font = std::env::var("CLIP_CAPTIONS_FONT")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
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

fn read_llm_config() -> Option<LlmConfig> {
    if !llm_enabled() {
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
            "C:\\Users\\Rashi\\.cache\\lm-studio\\models\\bartowski\\gemma-2-9b-it-GGUF\\gemma-2-9b-it-Q8_0_L.gguf"
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

fn build_caption_words(
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

fn trim_transcript(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.len() <= max_chars {
        return trimmed.to_string();
    }
    truncate_str(trimmed, max_chars)
}

fn normalize_title_whitespace(title: &str) -> String {
    title
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

fn sanitize_title_for_filename(title: &str) -> String {
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

fn fallback_title_from_transcript(text: &str, max_chars: usize) -> String {
    let mut title = normalize_title_whitespace(text);
    if title.is_empty() {
        return title;
    }
    if title.len() > max_chars {
        title = truncate_str(&title, max_chars);
    }
    title
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
    let resp = client
        .post(&cfg.endpoint)
        .json(&payload)
        .send()
        .await
        .context("LLM request failed")?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
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

fn caption_y_for_layout(
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

fn build_caption_drawtext_chain(words: &[CaptionWord], cfg: &CaptionConfig, y: f32) -> String {
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

fn format_srt_time(secs: f32) -> String {
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

fn wrap_caption_text(text: &str, max_len: usize) -> String {
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

fn build_srt_from_payload(
    payload: &TranscriptPayload,
    duration_secs: Option<f32>,
) -> Option<String> {
    let duration = duration_secs
        .filter(|v| v.is_finite() && *v > 0.0)?;
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

fn write_temp_srt(contents: &str) -> Result<PathBuf> {
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

fn subtitle_codec_for_output(out_path: &Path) -> &'static str {
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

async fn run_ffmpeg_encode(
    input: &str,
    out_path: &Path,
    filters: &FilterGraph,
    encoder: &str,
    use_hw_encode: bool,
    use_nvenc: bool,
    allow_hwaccel: bool,
    use_hw_frames: bool,
    regen_pts: bool,
    force_ts_input: bool,
    start_offset_secs: Option<f32>,
    duration_secs: Option<f32>,
    progress: Option<ProgressSpec>,
    metadata: Option<&ClipMetadata>,
    subtitles: Option<&SubtitleSpec>,
) -> Result<std::process::Output> {
    let mut cmd = Command::new("ffmpeg");
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
    cmd.arg("-i").arg(input);
    if let Some(subs) = subtitles {
        cmd.arg("-i").arg(subs.path.as_os_str());
    }
    if let Some(ss) = start_offset_secs {
        if ss > 0.0 {
            cmd.arg("-ss").arg(format!("{ss:.3}"));
        }
    }
    if let Some(d) = duration_secs {
        cmd.arg("-t").arg(format!("{d:.3}"));
    }
    match filters {
        FilterGraph::Vf(vf) => {
            cmd.arg("-map").arg("0:v:0");
            cmd.arg("-vf").arg(vf);
        }
        FilterGraph::Complex { graph, output } => {
            cmd.arg("-filter_complex").arg(graph);
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
        cmd.arg("-preset").arg("p4");
        cmd.arg("-tune").arg("hq");
        cmd.arg("-b:v").arg("0");
        cmd.arg("-cq").arg("23");
    } else {
        cmd.arg("-preset").arg("veryfast");
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

    let stderr_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut buf = Vec::new();
        reader
            .read_to_end(&mut buf)
            .await
            .context("reading ffmpeg stderr")?;
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
    let mut printed = false;
    let timeout = ffmpeg_encode_timeout();
    let mut timed_out = false;

    let read_stdout = async {
        while let Some(line) = stdout_lines
            .next_line()
            .await
            .context("reading ffmpeg progress")?
        {
            stdout_buf.extend_from_slice(line.as_bytes());
            stdout_buf.push(b'\n');

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
    if printed {
        eprintln!();
    }
    Ok(std::process::Output {
        status,
        stdout: stdout_buf,
        stderr: stderr_buf,
    })
}

async fn run_ffmpeg_internal(
    input: &str,
    out_path: &Path,
    out_w: u32,
    out_h: u32,
    duration_secs: Option<f32>,
    start_offset_secs: Option<f32>,
    regen_pts: bool,
    force_ts_input: bool,
    progress: Option<ProgressSpec>,
) -> Result<()> {
    let _span = profile_span("ffmpeg: encode");
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
    let mut layout_is_stacked = matches!(layout.mode, ClipLayoutMode::Stacked);
    let mut face_only = false;
    let mut fullscreen_fill = false;
    let mut fullscreen_track = None;
    let mut fullscreen_center_override = None;
    let face_only_area_threshold = 0.40;
    let face_center = |rect: crate::clip_layout::NormalizedRect| NormalizedPoint {
        x: (rect.x + rect.w / 2.0).clamp(0.0, 1.0),
        y: (rect.y + rect.h / 2.0).clamp(0.0, 1.0),
    };
    if layout_is_stacked {
        let detect_cfg = read_clip_detect_config();
        let need_face = layout.face_crop.is_none() && layout_hints.face_box.is_none();
        let user_game_center = layout_hints.game_center;
        let user_game_region = layout_hints.game_region;
        let need_gameplay = user_game_center.is_none();
        if detect_cfg.enabled && (need_face || need_gameplay) {
            match detect_layout_hints(input, &detect_cfg).await {
                Ok(detected) => {
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
        let face_found = layout.face_crop.is_some()
            || layout_hints.face_box.is_some()
            || layout_hints.face_track.is_some();
        let mut gameplay_found = gameplay_enabled() && layout_hints.game_center.is_some();
        let face_area = face_area_from_hints(&layout_hints).unwrap_or(0.0);
        let face_large = face_area >= face_only_area_threshold;
        if face_found && !gameplay_found && !face_large {
            gameplay_found = true;
            eprintln!(
                "clip layout: gameplay not detected; face area {:.3} below {:.2}; assuming gameplay present for stacked layout",
                face_area,
                face_only_area_threshold
            );
        }
        let face_rect = face_rect_from_hints(&layout_hints);
        let face_inside_game = gameplay_found
            && layout_hints
                .game_region
                .and_then(|game| face_rect.map(|face| rect_contains_rect(game, face, 0.01)))
                .unwrap_or(false);
        if face_inside_game {
            layout_is_stacked = false;
            fullscreen_fill = true;
            eprintln!(
                "clip layout: face region inside gameplay region; using single-frame layout"
            );
        } else if face_found && !gameplay_found {
            layout_is_stacked = false;
            if layout.face_crop.is_some() {
                face_only = true;
                eprintln!("clip layout: no gameplay detected; using face-only layout");
            } else {
                fullscreen_fill = true;
                fullscreen_track = layout_hints.face_track.clone();
                fullscreen_center_override = layout_hints
                    .face_box
                    .map(face_center)
                    .or_else(|| {
                        layout_hints
                            .face_track
                            .as_ref()
                            .and_then(|track| track.points.first().map(|p| face_center(p.rect)))
                    });
                eprintln!("clip layout: no gameplay detected; using single-frame face layout");
            }
        } else if !face_found {
            layout_is_stacked = false;
            fullscreen_fill = true;
            eprintln!("clip layout: no streamer cam detected; using full-frame gameplay fill");
        }
    }
    let stacked_dims = if layout_is_stacked {
        Some(resolve_stacked_layout_dims(out_w, out_h, &layout, &layout_hints))
    } else {
        None
    };
    let caption_cfg = read_caption_config(out_h);
    let llm_cfg = read_llm_config();
    let closed_captions = closed_captions_enabled();
    let mut transcript: Option<TranscriptPayload> = None;
    if caption_cfg.is_some() || llm_cfg.is_some() || closed_captions {
        if duration_secs.is_none() {
            eprintln!("captions: skipping (clip duration unknown)");
        } else {
            let input_owned = input.to_string();
            let start_offset = start_offset_secs;
            let duration = duration_secs;
            let force_ts = force_ts_input;
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
    let mut caption_chain: Option<String> = None;
    if let (Some(cfg), Some(payload)) = (caption_cfg.as_ref(), transcript.as_ref()) {
        let caption_words = build_caption_words(payload.words.clone(), cfg, duration_secs);
        if cfg.debug && !caption_words.is_empty() {
            for word in &caption_words {
                eprintln!(
                    "captions word: {:.2}-{:.2} {}",
                    word.start, word.end, word.text
                );
            }
        }
        if !caption_words.is_empty() {
            let y = caption_y_for_layout(cfg, layout_is_stacked, stacked_dims, out_h);
            let chain = build_caption_drawtext_chain(&caption_words, cfg, y);
            if !chain.is_empty() {
                caption_chain = Some(chain);
            }
        }
    }
    let mut subtitle_spec: Option<SubtitleSpec> = None;
    let mut _subtitle_temp: Option<TempSubtitle> = None;
    if closed_captions {
        if let Some(payload) = transcript.as_ref() {
            if let Some(srt) = build_srt_from_payload(payload, duration_secs) {
                match write_temp_srt(&srt) {
                    Ok(path) => {
                        subtitle_spec = Some(SubtitleSpec { path: path.clone() });
                        _subtitle_temp = Some(TempSubtitle { path });
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
    if let Some(chain) = caption_chain.as_deref() {
        filter_cpu = append_filter_graph(&filter_cpu, chain);
        filter_gpu = append_filter_graph(&filter_gpu, chain);
    }
    let force_cpu_filters =
        (is_stream_unstable(input) && is_nvenc) || layout_is_stacked || face_only || tracked_fullscreen;
    let allow_hwaccel = !is_stream_unstable(input);
    let use_cpu_filters = force_cpu_filters || !is_nvenc;
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
        allow_hwaccel,
        use_hw_frames,
        regen_pts,
        force_ts_input,
        start_offset_secs,
        duration_secs,
        progress,
        clip_metadata.as_ref(),
        subtitle_spec.as_ref(),
    )
    .await?;

    if !output.status.success() {
        cleanup_temp_output(&temp_out);
        let stderr_first = String::from_utf8_lossy(&output.stderr).into_owned();
        let stdout_first = String::from_utf8_lossy(&output.stdout).into_owned();
        let mut retried_cpu = false;

        if is_hw {
            let filter_failure = stderr_indicates_filter_issue(&stderr_first);
            if filter_failure && !layout_is_stacked {
                mark_stream_unstable(input);
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
                    false,
                    false,
                    regen_pts,
                    force_ts_input,
                    start_offset_secs,
                    duration_secs,
                    progress,
                    clip_metadata.as_ref(),
                    subtitle_spec.as_ref(),
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
                    allow_hwaccel,
                    false,
                    regen_pts,
                    force_ts_input,
                    start_offset_secs,
                    duration_secs,
                    progress,
                    clip_metadata.as_ref(),
                    subtitle_spec.as_ref(),
                )
                .await?;
                retried_cpu = true;
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
        } else if retried_cpu {
            eprintln!(
                "ffmpeg fallback succeeded with CPU/libx264 after GPU pipeline failure. Previous stderr (truncated): {} | stdout (truncated): {}",
                tail_trunc(&stderr_first, 200),
                tail_trunc(&stdout_first, 200)
            );
        }
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
    EnvSpec { env: "CLIP_FACE_TILE_MIN_SCORE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_TILE_MAX_DEPTH", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_HEAD_TOP", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_HEAD_TOP_MIN", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_HEAD_TOP_MAX", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_EYE_TOP_RATIO", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_EYE_CHIN_RATIO", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_SHOULDER_SCALE", mode: EnvValueMode::Required },
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
    EnvSpec { env: "CLIP_AUDIO_NORM", mode: EnvValueMode::Optional },
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
    EnvSpec { env: "CLIP_CAPTIONS_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_CLOSED_CAPTIONS", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_LLM_ENABLE", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_LLM_ENDPOINT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_TEMPERATURE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_TIMEOUT_SECS", mode: EnvValueMode::Required },
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
    EnvSpec { env: "CLIP_TS_REALTIME", mode: EnvValueMode::Flag },
    EnvSpec { env: "CLIP_GAMEPLAY", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_REGION_DETECT", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_LIVE_CONFIG", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LIVE_CONFIG_POLL_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_STREAMS_FILE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_STREAMS_POLL_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_STREAMS_MAX_CONCURRENT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_WAKE_WORDS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_PROFILE", mode: EnvValueMode::Optional },
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
    EnvSpec { env: "WAKE_FF_AF", mode: EnvValueMode::Required },
    EnvSpec { env: "SKIP_CLIP_SAVE", mode: EnvValueMode::Optional },
    EnvSpec { env: "MIC_DEVICE", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_RT_TARGET", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_MODEL_CANDIDATES", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_GPU", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_CLIP_GPU", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_MIN_FREE_VRAM_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_CUBLAS", mode: EnvValueMode::Optional },
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

fn split_wake_phrases(raw: &str) -> Vec<String> {
    raw.split(|c| matches!(c, ',' | '|' | ';' | '\n' | '\r'))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn split_stream_list(raw: &str) -> Vec<String> {
    raw.split(|c| matches!(c, ',' | '|' | ';' | '\n' | '\r'))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn normalize_stream_url(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(normalize_page_url(trimmed))
}

fn parse_streams_file(contents: &str) -> Vec<String> {
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

fn default_streams_file_path() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("streams.txt")
}

fn read_streams_file(path: &Path) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("reading streams file {}", path.display()))?;
    Ok(parse_streams_file(&contents))
}

fn sync_streams_file(path: &Path, streams: &[String]) -> Result<()> {
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

fn spawn_ctrl_c_handler(stop: Arc<AtomicBool>) {
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
            } else {
                eprintln!("Ctrl+C received again; forcing exit.");
                std::process::exit(130);
            }
        }
    });
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
        } else {
            if !last_underscore {
                out.push('_');
                last_underscore = true;
            }
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "stream".to_string()
    } else {
        trimmed.to_string()
    }
}

fn stream_id_from_url(url: &str) -> String {
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

fn face_id_enabled() -> bool {
    std::env::var("CLIP_FACE_ID")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn face_id_file_for_stream(page_url: &str) -> PathBuf {
    if let Ok(path) = std::env::var("CLIP_FACE_ID_FILE") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    let stream_id = stream_id_from_url(page_url);
    Path::new("face_id").join(format!("{stream_id}.json"))
}

fn extract_meta_image(html: &str, key: &str) -> Option<String> {
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

async fn enroll_face_id_from_stream_frame(
    page_url: &str,
    file_path: &Path,
) -> Result<bool> {
    let hls = HlsClient::new()?;
    let (master_url, master) = hls.fetch_master_from_page(page_url).await?;
    let media_url = hls.highest_variant_url(&master_url, &master)?;
    clip_detect::enroll_face_id_from_media_url(media_url.as_str(), file_path).await
}

async fn maybe_auto_enroll_face_id(page_url: &str) {
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
    spawn_ctrl_c_handler(stop.clone());
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
    ("CLIP_CAPTIONS_MIN_WORD_SECS", "0.12"),
    ("CLIP_CAPTIONS_MAX_WORDS", "300"),
    ("CLIP_CAPTIONS_CHEST_RATIO", "0.65"),
    ("CLIP_CAPTIONS_MARGIN_OFFSET", "0"),
    ("CLIP_CAPTIONS_DEBUG", "0"),
    ("CLIP_CLOSED_CAPTIONS", "1"),
    ("CLIP_LLM_ENABLE", "1"),
    ("CLIP_LLM_ENDPOINT", "http://localhost:1234/v1/chat/completions"),
    ("CLIP_LLM_MODEL", "C:\\Users\\Rashi\\.cache\\lm-studio\\models\\bartowski\\gemma-2-9b-it-GGUF\\gemma-2-9b-it-Q8_0_L.gguf"),
    ("CLIP_LLM_TEMPERATURE", "0.2"),
    ("CLIP_LLM_TIMEOUT_SECS", "20"),
    ("CLIP_LLM_MAX_TRANSCRIPT_CHARS", "4000"),
    ("CLIP_LLM_TITLE_MAX_CHARS", "80"),
    ("CLIP_LLM_DESCRIPTION_MAX_CHARS", "280"),
    ("CLIP_LLM_GPU_LAYERS", "0"),
    ("CLIP_LLM_FILE_RENAME", "1"),
    ("CLIP_LLM_DEBUG", "0"),
    ("CLIP_WAKE_WORDS", "orange"),
    ("CLIP_PROFILE", "1"),
    ("CLIP_STREAMS_POLL_SECS", "5"),
    ("CLIP_STREAMS_MAX_CONCURRENT", "1"),
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
    ("WHISPER_MIN_FREE_VRAM_MB", "2048"),
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
    let mut env_overrides = env_overrides;
    if let Some(phrase_list) = override_phrase.as_deref().filter(|v| !v.trim().is_empty()) {
        env_overrides.retain(|(key, _)| key != "CLIP_WAKE_WORDS");
        env_overrides.push(("CLIP_WAKE_WORDS".to_string(), phrase_list.to_string()));
    }
    let mut streams_path = streams_file.map(PathBuf::from);
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
    seed_ort_dylib_from_live_config(&live_config_path);

    // Ensure CUDA backend is preferred when available; avoid falling back to CPU due to missing env.
    if std::env::var("WHISPER_CUBLAS").is_err() {
        std::env::set_var("WHISPER_CUBLAS", "1");
    }
    if std::env::var("WHISPER_GPU").is_err() {
        std::env::set_var("WHISPER_GPU", "1");
    }

    auto_assign_gpus_for_tools();
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
            return run_whisper_worker();
        }
    }

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
        eprintln!("usage: autoclip <page_url>  (or set CLIP_PAGE_URL) | autoclip demo-buffer | autoclip demo-hls-buffer <page_url> | autoclip demo-ts <ts_path> | autoclip reprocess-ts <ts_path> | autoclip check-gameplay-model [model_dir] | autoclip demo-wakeword-mic <page_url> [--phrase WORDS] [--no-log-raw-wake] | autoclip face-sweep <positives_dir> <negatives_dir> [score_start score_end score_step] [out_csv]");
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

fn run_whisper_worker() -> Result<()> {
    let mode = std::env::var("WHISPER_WORKER_MODE")
        .unwrap_or_else(|_| "stream".to_string())
        .to_ascii_lowercase();
    let status_path = whisper_worker_status_path();
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

    let start_instant = Instant::now();
    if mode == "mic" {
        let mic_device = std::env::var("MIC_DEVICE").ok();
        run_wake_worker_mic(
            mic_device.as_deref(),
            Path::new(&model_path),
            &wake_phrases,
            log_raw,
            stop.clone(),
            fired.clone(),
            start_instant,
            detect_ns,
            audio_ns,
            last_word_ns,
        )?;
    } else {
        let media_url_path = whisper_worker_media_url_path();
        run_wake_worker_stream(
            &media_url_path,
            Path::new(&model_path),
            &wake_phrases,
            log_raw,
            stop.clone(),
            fired.clone(),
            start_instant,
            detect_ns,
            audio_ns,
            last_word_ns,
        )?;
    }

    stop.store(true, Ordering::Relaxed);
    Ok(())
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
    let path = Path::new(ts_path);
    if !path.exists() {
        anyhow::bail!("TS file not found: {}", path.display());
    }

    let cfg = Config::example();
    let output_path = next_output_path(&cfg.save_path, &cfg.file_name_stub)?;
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
    run_ffmpeg_internal(
        path.to_str()
            .ok_or_else(|| anyhow::anyhow!("non-utf8 input path"))?,
        &output_path,
        out_w,
        out_h,
        None,
        None,
        true,
        true,
        Some(ProgressSpec {
            total_secs: progress_total_secs,
        }),
    )
    .await?;
    println!("reprocessed TS clip -> {}", output_path.display());
    Ok(())
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
    println!("  CLIP_FACE_CROP           Face crop expr w:h:x:y (optional, overrides detection/anchor)");
    println!("  CLIP_FACE_CONTEXT        Face crop expansion scale for detected face (default 6.0)");
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
    println!("  CLIP_AUDIO_NORM          Normalize clip audio loudness (default true)");
    println!("  CLIP_CAPTIONS            Enable word-by-word open captions (default false)");
    println!("  CLIP_CAPTIONS_POSITION   Caption placement: margin (default) or chest");
    println!("  CLIP_CAPTIONS_FONT       Caption font name or TTF path (optional)");
    println!("  CLIP_CAPTIONS_SIZE       Caption font size px or ratio (<=2 treated as ratio)");
    println!("  CLIP_CAPTIONS_COLOR      Caption text color (default white)");
    println!("  CLIP_CAPTIONS_OUTLINE    Caption outline width (default 3)");
    println!("  CLIP_CAPTIONS_OUTLINE_COLOR Caption outline color (default black)");
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
    println!("  CLIP_LLM_MAX_TRANSCRIPT_CHARS Transcript truncation limit (default 4000)");
    println!("  CLIP_LLM_TITLE_MAX_CHARS LLM title max length (default 80)");
    println!("  CLIP_LLM_DESCRIPTION_MAX_CHARS LLM description max length (default 280)");
    println!("  CLIP_LLM_GPU_LAYERS      LLM GPU layers (0 = CPU, empty = model default)");
    println!("  CLIP_LLM_FILE_RENAME     Rename clip files using LLM title (default true)");
    println!("  CLIP_LLM_DEBUG           Log LLM parse details (default false)");
    println!("  CLIP_FACE_BUDGET_SECS    Override face detection time budget in seconds");
    println!("  CLIP_FACE_TILE_MIN_SCORE Tile search min score (default 0.60; set <= 0 to disable)");
    println!("  CLIP_FACE_TILE_MAX_DEPTH Max bisection depth for tile search (default 3)");
    println!("  CLIP_FACE_FRAME_HEAD_TOP      Head top offset vs face box (default -0.28)");
    println!("  CLIP_FACE_FRAME_HEAD_TOP_MIN  Min head top offset (default -0.8)");
    println!("  CLIP_FACE_FRAME_HEAD_TOP_MAX  Max head top offset (default 0.2)");
    println!("  CLIP_FACE_FRAME_EYE_TOP_RATIO Eye->top ratio for head estimate (default 0.45)");
    println!("  CLIP_FACE_FRAME_EYE_CHIN_RATIO Eye->chin ratio for head estimate (default 0.55)");
    println!("  CLIP_FACE_FRAME_SHOULDER_SCALE Shoulder width scale vs face (default 3.2)");
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
    println!("  CLIP_STREAMS_FILE        Path to streams file for multi-stream runs");
    println!("  CLIP_STREAMS_POLL_SECS   Streams file poll interval in seconds (default 5)");
    println!("  CLIP_STREAMS_MAX_CONCURRENT Max number of concurrent streams (default 1)");
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
    println!("  SKIP_CLIP_SAVE           If set to 1/true, skip writing clips");
    println!("  WHISPER_MODEL            Path to whisper model (default auto)");
    println!("  WHISPER_GPU              Enable GPU for live wake (default true)");
    println!("  WHISPER_CLIP_GPU         Enable GPU for clip transcription (default false)");
    println!("  WHISPER_MIN_FREE_VRAM_MB Min free VRAM before using whisper GPU (default 2048)");
    println!("  WHISPER_ISOLATE          Run live wake in a helper process (default false)");
    println!("  WHISPER_WORKER_STATUS_MS Poll interval for wake worker status (default 500ms)");
    println!("  FFMPEG_ENCODER / FFMPEG_HWACCEL / FFMPEG_HWACCEL_DEVICE   Encoder/accel knobs");
    println!("  FFMPEG_HWACCEL_FALLBACK  Fallback hwaccel (e.g. d3d11va, cuda, none; default none)");
    println!("  FFMPEG_MIN_FREE_VRAM_MB  Min free VRAM before using GPU encode (default 512)");
    println!("  FFMPEG_ENCODE_TIMEOUT_SECS   Hard cap for ffmpeg encode wall time (seconds)");
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

    #[tokio::test]
    async fn run_stub_succeeds() {
        // Ensure no network-dependent env vars force the main path.
        std::env::remove_var("CLIP_PAGE_URL");
        std::env::remove_var("M3U8_URL_OVERRIDE");
        let app = AutoClip::new(Config::example());
        assert!(app.run().await.is_ok());
    }

    #[test]
    fn parse_streams_file_ignores_comments_and_dedupes() {
        let input = "# streams\nkick.com/queenbushcamp\n\n// comment\nkick.com/queenbushcamp\nhttps://twitch.tv/someuser\n";
        let streams = parse_streams_file(input);
        assert_eq!(
            streams,
            vec![
                "https://kick.com/queenbushcamp".to_string(),
                "https://twitch.tv/someuser".to_string()
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
    fn extract_meta_image_reads_content() {
        let html = r#"<html><head><meta property="og:image" content="https://example.com/a.jpg"></head></html>"#;
        assert_eq!(
            extract_meta_image(html, r#"property="og:image"#),
            Some("https://example.com/a.jpg".to_string())
        );
        let html = r#"<meta name="twitter:image" content="https://example.com/b.png"/>"#;
        assert_eq!(
            extract_meta_image(html, r#"name="twitter:image"#),
            Some("https://example.com/b.png".to_string())
        );
    }

    #[test]
    fn face_id_file_for_stream_prefers_env_override() {
        let key = "CLIP_FACE_ID_FILE";
        let prev = std::env::var(key).ok();
        std::env::set_var(key, "face_id/custom.json");
        let path = face_id_file_for_stream("https://kick.com/QueenBushCamp");
        assert_eq!(path, PathBuf::from("face_id/custom.json"));
        if let Some(value) = prev {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn face_area_from_hints_uses_box_or_track() {
        let mut hints = clip_layout::ClipLayoutHints::default();
        assert!(face_area_from_hints(&hints).is_none());

        hints.face_box = Some(NormalizedRect {
            x: 0.0,
            y: 0.0,
            w: 0.2,
            h: 0.5,
        });
        let area = face_area_from_hints(&hints).unwrap();
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
        let area = face_area_from_hints(&hints).unwrap();
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
            text: "Hello world".to_string(),
            words: Vec::new(),
        };
        let srt = build_srt_from_payload(&payload, Some(2.5)).unwrap();
        assert!(srt.contains("00:00:00,000 --> 00:00:02,500"));
        assert!(srt.contains("Hello world"));
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
        assert!(srt.contains("00:00:00,000 --> 00:00:01,000"));
        assert!(srt.contains("Hello world"));
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
    fn extract_twitch_login_from_url() {
        assert_eq!(
            extract_twitch_login("https://www.twitch.tv/rynn?twitch5=0"),
            Some("rynn".to_string())
        );
        assert_eq!(
            extract_twitch_login("https://player.twitch.tv/?channel=rynn"),
            Some("rynn".to_string())
        );
        assert_eq!(extract_twitch_login("https://www.twitch.tv/directory"), None);
    }
}
