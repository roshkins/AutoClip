//! AutoClip MVP stub.
//! Current behavior: headless-grab an m3u8 from a page (Kick-style), pick the
//! top variant, and save a 30s vertical clip via FFmpeg. Everything else (rolling
//! buffering, wake-word detection, VRAM budgeting) is logged-only scaffolding.

use anyhow::{Context, Result};
use m3u8_rs::{MasterPlaylist, MediaPlaylist, VariantStream};
use reqwest::{header, Client};
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::process::Command;
use tokio::time::sleep;
use url::Url;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

mod rolling_buffer;
use rolling_buffer::RollingBuffer;
mod stream_audio_wake;
use stream_audio_wake::{start_mic_wake_with_ffmpeg, start_stream_wake_from_hls};

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
        let mut last_detect_instant: Option<Instant> = None;
        let start_instant = Instant::now();
        let detect_ns = Arc::new(std::sync::atomic::AtomicU64::new(u64::MAX));
        let audio_ns = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let monitor_audio_ns = audio_ns.clone();
        let monitor_stop = stop.clone();
        let monitor_target = buffer_target_ns.clone();
        let monitor_latency = max_latency_ns.clone();
        let monitor_headroom = latency_headroom;
        let monitor_clip = clip_window;
        let monitor_start = start_instant;
        std::thread::spawn(move || {
            let mut max_latency = Duration::ZERO;
            let mut warned = false;
            let poll = Duration::from_millis(250);
            while !monitor_stop.load(Ordering::Relaxed) {
                let audio_ns_now = monitor_audio_ns.load(Ordering::Relaxed);
                if audio_ns_now > 0 {
                    let audio_instant = monitor_start + Duration::from_nanos(audio_ns_now);
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

        let hls = HlsClient::new()?;
        let (master_url, master) = hls.fetch_master_from_page(page_url).await?;
        let media_url = hls.highest_variant_url(&master_url, &master)?;
        println!("tracking variant: {}", media_url);

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

        // Start listening for the wake phrase, either from microphone or stream audio.
        let model_path = stream_audio_wake::select_best_model_path();
        if self.config.use_mic_for_wake {
            let mic_device = self
                .config
                .mic_device
                .clone()
                .or_else(|| std::env::var("MIC_DEVICE").ok());
            start_mic_wake_with_ffmpeg(
                mic_device.as_deref(),
                Path::new(&model_path),
                &self.config.activation_phrase,
                self.config.log_raw_wake,
                stop_for_audio,
                fired.clone(),
                start_instant,
                detect_ns.clone(),
                audio_ns.clone(),
            )?;
            println!(
                "listening to microphone for wake phrase '{}' (model: {})",
                self.config.activation_phrase,
                model_path.display(),
            );
        } else {
            start_stream_wake_from_hls(
                &media_url,
                Path::new(&model_path),
                &self.config.activation_phrase,
                self.config.log_raw_wake,
                stop_for_audio,
                fired.clone(),
                start_instant,
                detect_ns.clone(),
                audio_ns.clone(),
            )?;
            println!(
                "listening to stream audio for wake phrase '{}' (model: {})",
                self.config.activation_phrase,
                model_path.display()
            );
        }

        let stop_signal = stop.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                eprintln!("Ctrl+C received, shutting down...");
                stop_signal.store(true, Ordering::Relaxed);
            }
        });

        let poll_interval = Duration::from_millis(500);

        loop {
            if stop.load(Ordering::Relaxed) {
                break;
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

            let playlist = match hls.fetch_media(media_url.as_str()).await {
                Ok(p) => p,
                Err(err) => {
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

                match hls.fetch_segment_from_playlist(&media_url, &uri).await {
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
                    let detected_instant = if ns != u64::MAX {
                        start_instant + Duration::from_nanos(ns)
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
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
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
                    let clip_len = clip_window;
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

                        tokio::spawn(async move {
                            if let Err(err) = save_future.await {
                                eprintln!("failed to persist wakeword clip: {err:#}");
                            }
                        });
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
                let detected_instant = if ns != u64::MAX {
                    start_instant + Duration::from_nanos(ns)
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

        // continuous loop
        Ok(())
    }
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
    /// COOKIE_HEADER or KICK_COOKIE if present. If M3U8_URL_OVERRIDE is set, use
    /// that master URL directly instead of headless extraction.
    pub async fn fetch_master_from_page(&self, page_url: &str) -> Result<(String, MasterPlaylist)> {
        let mut cookie_parts: Vec<String> = Vec::new();
        for key in ["COOKIE_HEADER", "KICK_COOKIE", "TIKTOK_COOKIE"] {
            if let Ok(val) = std::env::var(key) {
                if !val.trim().is_empty() {
                    cookie_parts.push(val);
                }
            }
        }
        let env_cookie = if cookie_parts.is_empty() {
            None
        } else {
            Some(cookie_parts.join("; "))
        };

        let origin = origin_for_page(page_url).unwrap_or_else(|| "https://kick.com".to_string());

        let is_tiktok = page_url.contains("tiktok.com");
        let headless_script = if is_tiktok {
            std::env::var("HEADLESS_M3U8_SCRIPT_TIKTOK")
                .unwrap_or_else(|_| "scripts/capture_m3u8_tiktok.js".to_string())
        } else {
            std::env::var("HEADLESS_M3U8_SCRIPT")
                .unwrap_or_else(|_| "scripts/capture_m3u8.js".to_string())
        };

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
            anyhow::bail!("non-success status {} for {}", status, url);
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

    let gpus = detect_nvidia_gpus();
    if !gpus.is_empty() {
        // Sort by memory already done in detect; pick biggest for whisper, second for ffmpeg if present.
        let whisper_gpu = gpus[0].0;
        let ffmpeg_gpu = if gpus.len() > 1 { gpus[1].0 } else { whisper_gpu };

        std::env::set_var("WHISPER_GPU", whisper_gpu.to_string());
        if user_hwaccel.is_none() {
            std::env::set_var("FFMPEG_HWACCEL", "cuda");
            std::env::set_var("FFMPEG_HWACCEL_DEVICE", ffmpeg_gpu.to_string());
        }
        if user_encoder.is_none() && ffmpeg_has_encoder("h264_nvenc") {
            std::env::set_var("FFMPEG_ENCODER", "h264_nvenc");
        }
        eprintln!("auto GPU assign for whisper: {}", whisper_gpu);
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClipLayoutMode {
    Full,
    Stacked,
}

#[derive(Clone, Copy, Debug)]
enum FaceAnchor {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Center,
}

#[derive(Clone, Debug)]
struct ClipLayoutConfig {
    mode: ClipLayoutMode,
    face_ratio: f32,
    face_crop: Option<String>,
    face_anchor: FaceAnchor,
}

#[derive(Clone, Debug)]
enum FilterGraph {
    Vf(String),
    Complex { graph: String, output: String },
}

fn read_clip_layout_config() -> ClipLayoutConfig {
    let mode = std::env::var("CLIP_LAYOUT")
        .ok()
        .map(|v| v.to_ascii_lowercase())
        .and_then(|v| match v.trim() {
            "stacked" | "tiktok" | "stack" | "split" => Some(ClipLayoutMode::Stacked),
            "full" | "default" | "" => Some(ClipLayoutMode::Full),
            _ => None,
        })
        .unwrap_or(ClipLayoutMode::Full);

    let face_ratio = std::env::var("CLIP_FACE_RATIO")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(0.2, 0.8))
        .unwrap_or(0.40);

    let face_crop = std::env::var("CLIP_FACE_CROP")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty() && !v.eq_ignore_ascii_case("none"))
        .map(|v| v.strip_prefix("crop=").unwrap_or(&v).to_string());

    let face_anchor = std::env::var("CLIP_FACE_ANCHOR")
        .ok()
        .map(|v| v.to_ascii_lowercase())
        .and_then(|v| match v.trim() {
            "top-right" | "right-top" | "tr" => Some(FaceAnchor::TopRight),
            "bottom-left" | "left-bottom" | "bl" => Some(FaceAnchor::BottomLeft),
            "bottom-right" | "right-bottom" | "br" => Some(FaceAnchor::BottomRight),
            "center" | "centre" | "middle" => Some(FaceAnchor::Center),
            "top-left" | "left-top" | "tl" | "" => Some(FaceAnchor::TopLeft),
            _ => None,
        })
        .unwrap_or(FaceAnchor::TopLeft);

    ClipLayoutConfig {
        mode,
        face_ratio,
        face_crop,
        face_anchor,
    }
}

fn resolve_layout_heights(out_h: u32, face_ratio: f32) -> (u32, u32) {
    let mut face_h = (out_h as f32 * face_ratio).round() as u32;
    if face_h < 2 {
        face_h = 2;
    }
    if face_h >= out_h {
        face_h = out_h.saturating_sub(2);
    }
    if face_h % 2 != 0 {
        face_h = face_h.saturating_sub(1);
    }
    let mut game_h = out_h.saturating_sub(face_h);
    if game_h % 2 != 0 {
        game_h = game_h.saturating_sub(1);
        face_h = out_h.saturating_sub(game_h);
    }
    (face_h.max(2), game_h.max(2))
}

fn default_face_crop_expr(anchor: FaceAnchor) -> String {
    const FACE_CROP_W_RATIO: f32 = 0.6;
    const FACE_CROP_H_RATIO: f32 = 0.6;
    let w = format!("iw*{:.3}", FACE_CROP_W_RATIO);
    let h = format!("ih*{:.3}", FACE_CROP_H_RATIO);
    let x = match anchor {
        FaceAnchor::TopLeft | FaceAnchor::BottomLeft => "0".to_string(),
        FaceAnchor::TopRight | FaceAnchor::BottomRight => format!("iw-({w})"),
        FaceAnchor::Center => format!("(iw-({w}))/2"),
    };
    let y = match anchor {
        FaceAnchor::TopLeft | FaceAnchor::TopRight => "0".to_string(),
        FaceAnchor::BottomLeft | FaceAnchor::BottomRight => format!("ih-({h})"),
        FaceAnchor::Center => format!("(ih-({h}))/2"),
    };
    format!("{w}:{h}:{x}:{y}")
}

fn build_stacked_filter_graph(out_w: u32, out_h: u32, layout: &ClipLayoutConfig) -> FilterGraph {
    let (face_h, game_h) = resolve_layout_heights(out_h, layout.face_ratio);
    let face_crop = layout
        .face_crop
        .clone()
        .unwrap_or_else(|| default_face_crop_expr(layout.face_anchor));
    let face_chain = format!(
        "crop={face_crop},scale={out_w}:{face_h}:force_original_aspect_ratio=increase,crop={out_w}:{face_h}"
    );
    let game_chain = format!(
        "scale={out_w}:{game_h}:force_original_aspect_ratio=increase,crop={out_w}:{game_h}"
    );
    let graph = format!(
        "[0:v]split=2[face_src][game_src];[face_src]{face_chain}[face];[game_src]{game_chain}[game];[face][game]vstack=inputs=2,format=yuv420p[v]"
    );
    FilterGraph::Complex {
        graph,
        output: "v".to_string(),
    }
}

fn tail_trunc(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return chars.into_iter().collect();
    }
    chars[chars.len() - max..].iter().collect::<String>()
}

async fn run_ffmpeg_encode(
    input: &str,
    out_path: &Path,
    filters: &FilterGraph,
    encoder: &str,
    use_hw_encode: bool,
    use_nvenc: bool,
    allow_hwaccel: bool,
    regen_pts: bool,
    force_ts_input: bool,
    start_offset_secs: Option<f32>,
    duration_secs: Option<f32>,
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
    if use_nvenc && allow_hwaccel {
        cmd.arg("-hwaccel_output_format").arg("cuda");
    }
    if force_ts_input {
        cmd.arg("-f").arg("mpegts");
    }
    cmd.arg("-i").arg(input);
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

    if use_nvenc {
        cmd.arg("-preset").arg("p4");
        cmd.arg("-tune").arg("hq");
        cmd.arg("-b:v").arg("0");
        cmd.arg("-cq").arg("23");
    } else {
        cmd.arg("-preset").arg("veryfast");
        cmd.arg("-crf").arg("23");
    }

    if regen_pts {
        if let Some(d) = duration_secs {
            let trim_end = start_offset_secs.unwrap_or(0.0).max(0.0) + d;
            // Keep full audio duration when seeking so -ss doesn't shorten the tail.
            cmd.arg("-af")
                .arg(format!("atrim=end={trim_end:.3},asetpts=N/SR/TB"));
        }
        cmd.arg("-c:a")
            .arg("aac")
            .arg("-b:a")
            .arg("160k");
    } else {
        cmd.arg("-c:a").arg("copy");
    }

    if duration_secs.is_none() {
        cmd.arg("-shortest");
    }
    cmd.arg(out_path.as_os_str());
    cmd.output().await.context("failed to run ffmpeg")
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
) -> Result<()> {
    let (mut video_encoder, mut is_nvenc, mut is_hw, _encoder_forced) = ffmpeg_video_encoder();
    let scale_cuda_available = ffmpeg_scale_cuda_available();
    if is_hw && !scale_cuda_available {
        eprintln!("ffmpeg: scale_cuda filter missing; forcing CPU encode");
        video_encoder = "libx264".to_string();
        is_nvenc = false;
        is_hw = false;
    }
    let layout = read_clip_layout_config();
    let layout_is_stacked = matches!(layout.mode, ClipLayoutMode::Stacked);
    let hwaccel = std::env::var("FFMPEG_HWACCEL").unwrap_or_default();
    let hw_decode_cuda = is_nvenc && hwaccel.eq_ignore_ascii_case("cuda");
    let (vf_cpu, vf_gpu_sw, vf_gpu_hw) = build_ffmpeg_filters(out_w, out_h);
    let vf_gpu = if hw_decode_cuda { vf_gpu_hw.clone() } else { vf_gpu_sw.clone() };
    let filter_cpu = if layout_is_stacked {
        build_stacked_filter_graph(out_w, out_h, &layout)
    } else {
        FilterGraph::Vf(vf_cpu)
    };
    let filter_gpu = if layout_is_stacked {
        filter_cpu.clone()
    } else {
        FilterGraph::Vf(vf_gpu)
    };
    let force_cpu_filters = (is_stream_unstable(input) && is_nvenc) || layout_is_stacked;
    if force_cpu_filters {
        if layout_is_stacked {
            if is_hw {
                eprintln!("ffmpeg: stacked layout uses CPU filters; disabling hwaccel decode");
            }
        } else {
            eprintln!("ffmpeg: stream flagged as unstable; using CPU filters with NVENC");
        }
    }

    let use_cpu_filters = force_cpu_filters || !is_nvenc;
    let selected_filters = if use_cpu_filters { &filter_cpu } else { &filter_gpu };
    let allow_hwaccel = !force_cpu_filters;

    let mut output = run_ffmpeg_encode(
        input,
        out_path,
        selected_filters,
        &video_encoder,
        is_hw,
        is_nvenc,
        allow_hwaccel,
        regen_pts,
        force_ts_input,
        start_offset_secs,
        duration_secs,
    )
    .await?;

    if !output.status.success() {
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
                output = run_ffmpeg_encode(
                    input,
                    out_path,
                    &filter_cpu,
                    &video_encoder,
                    true,
                    true,
                    false,
                    regen_pts,
                    force_ts_input,
                    start_offset_secs,
                    duration_secs,
                )
                .await?;
            }
            if !output.status.success() {
                output = run_ffmpeg_encode(
                    input,
                    out_path,
                    &filter_cpu,
                    "libx264",
                    false,
                    false,
                    !force_cpu_filters,
                    regen_pts,
                    force_ts_input,
                    start_offset_secs,
                    duration_secs,
                )
                .await?;
                retried_cpu = true;
            }
        }

        if !output.status.success() {
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

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Ensure CUDA backend is preferred when available; avoid falling back to CPU due to missing env.
    if std::env::var("WHISPER_CUBLAS").is_err() {
        std::env::set_var("WHISPER_CUBLAS", "1");
    }
    if std::env::var("WHISPER_GPU").is_err() {
        std::env::set_var("WHISPER_GPU", "1");
    }

    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help(args.get(0).map(String::as_str).unwrap_or("autoclip"));
        return Ok(());
    }

    auto_assign_gpus_for_tools();
    let page_url_arg = args.get(1).map(|s| s.as_str());
    let mut override_phrase: Option<String> = None;
    let mut log_raw_wake = true; // default on; can be suppressed

    // Parse optional flags for wakeword control.
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--phrase" => {
                if let Some(val) = args.get(i + 1) {
                    override_phrase = Some(val.clone());
                }
                i += 2;
            }
            "--log-raw-wake" => {
                log_raw_wake = true;
                i += 1;
            }
            "--no-log-raw-wake" => {
                log_raw_wake = false;
                i += 1;
            }
            _ => i += 1,
        }
    }

    if let Some(cmd) = page_url_arg {
        if cmd.eq_ignore_ascii_case("demo-buffer") {
            return run_buffer_demo().await;
        }
        if cmd.eq_ignore_ascii_case("demo-hls-buffer") {
            let page = args
                .get(2)
                .cloned()
                .or_else(|| std::env::var("CLIP_PAGE_URL").ok())
                .unwrap_or_default();
            if page.is_empty() {
                eprintln!("usage: autoclip demo-hls-buffer <page_url>  (or set CLIP_PAGE_URL)");
                return Ok(());
            }
            return run_hls_buffer_demo(&page).await;
        }
        if cmd.eq_ignore_ascii_case("demo-wakeword-mic") {
            let opts = parse_mic_args(&args[2..])?;
            return run_wakeword_mic_demo(opts).await;
        }
    }

    let mut config = Config::example();
    if let Some(url) = page_url_arg {
        config.kick_url = url.to_string();
    }
    if let Some(p) = override_phrase {
        config.activation_phrase = p;
    }
    config.log_raw_wake = log_raw_wake;

    if page_url_arg.is_none() && std::env::var("CLIP_PAGE_URL").is_err() {
        eprintln!("usage: autoclip <page_url>  (or set CLIP_PAGE_URL) | autoclip demo-buffer | autoclip demo-hls-buffer <page_url> | autoclip demo-wakeword-mic <page_url> [--phrase NAME] [--no-log-raw-wake]");
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
    }
    cfg.use_mic_for_wake = true;
    cfg.log_raw_wake = opts.log_raw_wake;
    cfg.mic_device = opts.mic_device.clone();
    let app = AutoClip::new(cfg);
    app.run_until_wake_and_clip(&opts.page_url).await
}

#[derive(Debug, Clone)]
struct MicOpts {
    page_url: String,
    phrase: Option<String>,
    log_raw_wake: bool,
    mic_device: Option<String>,
}

fn parse_mic_args(args: &[String]) -> Result<MicOpts> {
    if args.is_empty() {
        anyhow::bail!("usage: autoclip demo-wakeword-mic <page_url> [--phrase NAME] [--log-raw-wake] [--mic-device DEVICE]");
    }

    let page_url = args[0].clone();
    let mut phrase = None;
    let mut log_raw_wake = false;
    let mut mic_device: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--phrase" => {
                i += 1;
                phrase = args.get(i).cloned();
            }
            "--log-raw-wake" => {
                log_raw_wake = true;
            }
            "--mic-device" => {
                i += 1;
                mic_device = args.get(i).cloned();
            }
            other => {
                anyhow::bail!("unknown flag {other}");
            }
        }
        i += 1;
    }

    if mic_device.is_none() {
        if let Ok(env_dev) = std::env::var("MIC_DEVICE") {
            if !env_dev.trim().is_empty() {
                mic_device = Some(env_dev);
            }
        }
    }

    if mic_device.is_none() {
        mic_device = prompt_for_mic_device();
    }

    Ok(MicOpts { page_url, phrase, log_raw_wake, mic_device })
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
    println!("  {bin} <page_url> [options]");
    println!("  {bin} demo-buffer");
    println!("  {bin} demo-hls-buffer <page_url>");
    println!("  {bin} demo-wakeword-mic <page_url> [--phrase WORD] [--log-raw-wake] [--mic-device NAME]");
    println!("");
    println!("Options:");
    println!("  --phrase WORD           Override wake phrase (default: 'orange')");
    println!("  --log-raw-wake         Log raw/normalized transcripts (default on)");
    println!("  --no-log-raw-wake      Disable transcript logging");
    println!("  --mic-device NAME      Microphone device for mic wake mode");
    println!("  -h, --help             Show this help");
    println!("");
    println!("Environment (selected):");
    println!("  CLIP_PAGE_URL            Default page when none is passed");
    println!("  CLIP_LAYOUT              Layout mode: full (default) or stacked/tiktok");
    println!("  CLIP_FACE_RATIO          Height ratio reserved for face panel (default 0.40)");
    println!("  CLIP_FACE_CROP           Face crop expr w:h:x:y (optional, overrides anchor)");
    println!("  CLIP_FACE_ANCHOR         Anchor for default face crop (top-left default)");
    println!("  M3U8_URL_OVERRIDE        Skip discovery; use this master URL directly");
    println!("  COOKIE_HEADER / KICK_COOKIE / TIKTOK_COOKIE   Cookies to send on discovery");
    println!("  HEADLESS_M3U8_SCRIPT / HEADLESS_M3U8_SCRIPT_TIKTOK   Override Playwright scripts");
    println!("  WAKE_REFRACTORY_SECS     Cooldown between wake detections (default 12)");
    println!("  WAKE_BUFFER_HEADROOM_SECS   Extra buffer headroom for wake timing (default 20)");
    println!("  SKIP_CLIP_SAVE           If set to 1/true, skip writing clips");
    println!("  WHISPER_MODEL            Path to whisper model (default auto)");
    println!("  FFMPEG_ENCODER / FFMPEG_HWACCEL / FFMPEG_HWACCEL_DEVICE   Encoder/accel knobs");
    println!("  FFMPEG_HWACCEL_FALLBACK  Fallback hwaccel (e.g. d3d11va, cuda, none; default none)");
    println!("  LOG_M3U8_HEADERS         Log request headers when fetching playlists");
    println!("");
    println!("Notes:");
    println!("  - Main path: page URL -> HLS discovery (TikTok HTTP first, headless fallback) -> wake detection -> clip.");
    println!("  - Demo modes: buffer-only, HLS buffer demo, or mic wake demo for quick sanity checks.");
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
}
