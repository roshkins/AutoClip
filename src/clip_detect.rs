use anyhow::{Context, Result};
use std::cmp::{max, min};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::Factoid;
use tract_onnx::tract_hir::internal::DimLike;

use crate::clip_gameplay::{ClipGameplayDetector, GameplayObservation, read_clip_gameplay_config, log_gameplay_debug};
use crate::clip_layout::{ClipLayoutHints, NormalizedPoint, NormalizedRect};
use crate::loading::LoadingTicker;

#[cfg(feature = "ort")]
use ort::ep;
#[cfg(feature = "ort")]
use ort::session::Session;
#[cfg(feature = "ort")]
use ort::value::TensorRef;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceBackend {
    Auto,
    Tract,
    Ort,
}

#[derive(Clone, Debug)]
pub struct ClipDetectConfig {
    pub enabled: bool,
    pub sample_count: usize,
    pub sample_start_secs: f32,
    pub sample_step_secs: f32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub face_model_path: Option<String>,
    pub face_backend: FaceBackend,
    pub face_score_threshold: f32,
    pub scan_full_clip: bool,
    pub track_face: bool,
    pub analysis_budget: Option<Duration>,
}

const DEFAULT_SAMPLE_COUNT: usize = 3;
const DEFAULT_SAMPLE_START: f32 = 1.0;
const DEFAULT_SAMPLE_STEP: f32 = 1.5;
const DEFAULT_FRAME_WIDTH: u32 = 960;
const DEFAULT_FRAME_HEIGHT: u32 = 540;
const DEFAULT_FACE_MODEL_PATH: &str = "models/face_detection_yunet_2023mar.onnx";
const DEFAULT_FACE_SCORE: f32 = 0.5;
const DEFAULT_FACE_TILE_MIN_SCORE: f32 = 0.60;
const DEFAULT_FACE_TILE_MAX_DEPTH: usize = 3;
const MAX_FULL_SAMPLES: usize = 60;
const MAX_FACE_CANDIDATES: usize = 24;
const FACE_EDGE_MARGIN: f32 = 0.02;
const MAX_FACE_DRIFT: f32 = 0.18;
const FACE_MIN_PX: f32 = 10.0;
const FACE_MAX_PX: f32 = 300.0;
const FACE_LANDMARK_REQUIRED: bool = true;
const FACE_LANDMARK_MAX_EYE_TILT: f32 = 0.35;
const FACE_LANDMARK_MAX_MOUTH_TILT: f32 = 0.45;
const FACE_LANDMARK_NOSE_CENTER_MAX: f32 = 0.25;
const FACE_LANDMARK_MOUTH_CENTER_MAX: f32 = 0.35;
const FACE_LANDMARK_MIN_EYE_RATIO: f32 = 0.18;
const FACE_LANDMARK_EYE_NOSE_MIN: f32 = 0.05;
const FACE_LANDMARK_NOSE_MOUTH_MIN: f32 = 0.06;
const FACE_SCAN_REGION_SIZES: [f32; 2] = [0.5, 0.35];
const FACE_REGION_MARGIN: f32 = 0.02;

fn parse_face_backend(value: &str) -> Option<FaceBackend> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Some(FaceBackend::Auto),
        "tract" | "cpu" => Some(FaceBackend::Tract),
        "ort" | "onnxruntime" | "cuda" | "gpu" => Some(FaceBackend::Ort),
        _ => None,
    }
}

pub fn read_clip_detect_config() -> ClipDetectConfig {
    let enabled = std::env::var("CLIP_DETECT")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(true);

    let sample_count = std::env::var("CLIP_DETECT_SAMPLES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.clamp(1, 6))
        .unwrap_or(DEFAULT_SAMPLE_COUNT);

    let sample_start_secs = std::env::var("CLIP_DETECT_START")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(DEFAULT_SAMPLE_START);

    let sample_step_secs = std::env::var("CLIP_DETECT_STEP")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_SAMPLE_STEP);

    let (frame_width, frame_height) = std::env::var("CLIP_DETECT_SIZE")
        .ok()
        .and_then(|v| parse_size(&v))
        .unwrap_or((DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT));

    let face_score_threshold = std::env::var("CLIP_FACE_SCORE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(0.05, 0.99))
        .unwrap_or(DEFAULT_FACE_SCORE);

    let face_backend = match std::env::var("CLIP_FACE_BACKEND") {
        Ok(value) => match parse_face_backend(&value) {
            Some(backend) => backend,
            None => {
                eprintln!(
                    "clip detect: unknown CLIP_FACE_BACKEND={value}; using auto"
                );
                FaceBackend::Auto
            }
        },
        Err(_) => FaceBackend::Auto,
    };

    let track_face = std::env::var("CLIP_FACE_TRACK")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(true);

    let scan_full_clip = std::env::var("CLIP_DETECT_FULL")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false);

    let analysis_budget = std::env::var("CLIP_DETECT_BUDGET_SECS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(Duration::from_secs_f32);

    let face_model_path = std::env::var("CLIP_FACE_MODEL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_FACE_MODEL_PATH.to_string());
    let face_model_path = if Path::new(&face_model_path).exists() {
        Some(face_model_path)
    } else {
        None
    };

    ClipDetectConfig {
        enabled,
        sample_count,
        sample_start_secs,
        sample_step_secs,
        frame_width: frame_width.max(64).min(1920),
        frame_height: frame_height.max(64).min(1080),
        face_model_path,
        face_backend,
        face_score_threshold,
        scan_full_clip: scan_full_clip || track_face,
        track_face,
        analysis_budget,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FaceSweepStats {
    pub score: f32,
    pub pos_found: usize,
    pub neg_found: usize,
    pub pos_total: usize,
    pub neg_total: usize,
}

#[derive(Clone, Debug)]
struct FaceSweepInput {
    samples: Vec<FaceSample>,
    total_samples: usize,
}

fn split_analysis_budget(
    budget: Option<Duration>,
) -> (Option<Duration>, Option<Duration>) {
    let Some(budget) = budget else {
        return (None, None);
    };
    let total_secs = budget.as_secs_f32();
    if !total_secs.is_finite() || total_secs <= 0.0 {
        return (None, None);
    }
    let face_secs = (total_secs * 0.5)
        .max(8.0)
        .min(20.0)
        .min(total_secs);
    let face_budget = Duration::from_secs_f32(face_secs);
    let gameplay_budget = budget
        .checked_sub(face_budget)
        .filter(|remaining| remaining.as_secs_f32() > 0.05);
    (Some(face_budget), gameplay_budget)
}

pub async fn run_face_threshold_sweep(
    positives: &[String],
    negatives: &[String],
    scores: &[f32],
    config: &ClipDetectConfig,
) -> Result<Vec<FaceSweepStats>> {
    if scores.is_empty() {
        return Ok(Vec::new());
    }
    let min_score = scores
        .iter()
        .copied()
        .fold(f32::INFINITY, |acc, v| acc.min(v));
    let mut sweep_config = config.clone();
    if min_score.is_finite() {
        sweep_config.face_score_threshold = min_score;
    }
    sweep_config.scan_full_clip = false;
    sweep_config.track_face = false;
    sweep_config.analysis_budget = None;

    let detector = match YunetDetector::new(&sweep_config)? {
        Some(detector) => detector,
        None => anyhow::bail!("face model not available for sweep"),
    };

    eprintln!(
        "face sweep: loading samples (positives={}, negatives={})",
        positives.len(),
        negatives.len()
    );
    let pos_inputs =
        build_face_sweep_inputs(positives, &sweep_config, &detector, "positives").await?;
    let neg_inputs =
        build_face_sweep_inputs(negatives, &sweep_config, &detector, "negatives").await?;
    eprintln!(
        "face sweep: sample load complete (positives={}, negatives={})",
        pos_inputs.len(),
        neg_inputs.len()
    );

    let mut stats: Vec<FaceSweepStats> = scores
        .iter()
        .map(|score| FaceSweepStats {
            score: *score,
            pos_found: 0,
            neg_found: 0,
            pos_total: pos_inputs.len(),
            neg_total: neg_inputs.len(),
        })
        .collect();

    let total = (pos_inputs.len() + neg_inputs.len()) * scores.len();
    let mut processed = 0usize;
    let sweep_start = Instant::now();
    let mut last_log = Instant::now();
    log_face_sweep_progress(total, processed, sweep_start, false);

    for input in &pos_inputs {
        for (idx, score) in scores.iter().enumerate() {
            if select_face_with_relaxation(
                &input.samples,
                input.total_samples,
                &detector,
                *score,
            )
            .is_some()
            {
                stats[idx].pos_found += 1;
            }
            processed += 1;
            maybe_log_face_sweep_progress(total, processed, sweep_start, &mut last_log);
        }
    }

    for input in &neg_inputs {
        for (idx, score) in scores.iter().enumerate() {
            if select_face_with_relaxation(
                &input.samples,
                input.total_samples,
                &detector,
                *score,
            )
            .is_some()
            {
                stats[idx].neg_found += 1;
            }
            processed += 1;
            maybe_log_face_sweep_progress(total, processed, sweep_start, &mut last_log);
        }
    }

    log_face_sweep_progress(total, processed, sweep_start, true);
    Ok(stats)
}

fn maybe_log_face_sweep_progress(
    total: usize,
    processed: usize,
    start: Instant,
    last_log: &mut Instant,
) {
    if last_log.elapsed() >= Duration::from_secs(5) {
        log_face_sweep_progress(total, processed, start, false);
        *last_log = Instant::now();
    }
}

fn log_face_sweep_progress(
    total: usize,
    processed: usize,
    start: Instant,
    done: bool,
) {
    if total == 0 {
        return;
    }
    let elapsed = start.elapsed().as_secs_f32().max(0.001);
    let rate = processed as f32 / elapsed;
    let remaining = if rate > 0.0 {
        ((total - processed) as f32 / rate).max(0.0)
    } else {
        0.0
    };
    let eta = format_duration_secs(remaining);
    let pct = processed as f32 / total as f32;
    let bar = progress_bar(pct, 30);
    let line = format!(
        "face sweep: {} {}/{} ({:>5.1}%) eta {}",
        bar,
        processed,
        total,
        pct * 100.0,
        eta
    );
    if done {
        eprintln!("{line}");
    } else {
        use std::io::Write;
        eprint!("\r{line}");
        let _ = std::io::stderr().flush();
    }
}

fn progress_bar(pct: f32, width: usize) -> String {
    let clamped = pct.clamp(0.0, 1.0);
    let filled = (clamped * width as f32).round() as usize;
    let filled = filled.min(width);
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "#".repeat(filled), "-".repeat(empty))
}

fn format_duration_secs(secs: f32) -> String {
    let total = secs.max(0.0).round() as u64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    } else {
        format!("{:02}:{:02}", minutes, seconds)
    }
}

async fn build_face_sweep_inputs(
    inputs: &[String],
    config: &ClipDetectConfig,
    detector: &YunetDetector,
    label: &str,
) -> Result<Vec<FaceSweepInput>> {
    let mut out = Vec::new();
    let total = inputs.len().max(1);
    let start = Instant::now();
    let mut last_log = Instant::now();
    for input in inputs {
        let path = Path::new(input);
        if !path.exists() {
            eprintln!("face sweep: missing input {}", path.display());
            continue;
        }
        let source_dims = probe_media_dimensions(input).await;
        let sample_times = build_sample_times(config, true, None);
        let total_samples = sample_times.len();
        if total_samples == 0 {
            continue;
        }
        let mut face_samples: Vec<FaceSample> = Vec::new();
        for seek in sample_times {
            let seek_arg = if seek > 0.0 { Some(seek) } else { None };
            let candidates =
                detect_face_candidates(input, seek_arg, detector, source_dims).await;
            if !candidates.is_empty() {
                face_samples.push(FaceSample {
                    time: seek,
                    candidates,
                });
            }
        }
        out.push(FaceSweepInput {
            samples: face_samples,
            total_samples,
        });
        if last_log.elapsed() >= Duration::from_secs(5) {
            eprintln!(
                "face sweep: loading {} samples {}/{} ({:.1}%)",
                label,
                out.len(),
                total,
                (out.len() as f32 / total as f32) * 100.0
            );
            last_log = Instant::now();
        }
    }
    let elapsed = start.elapsed().as_secs_f32();
    eprintln!(
        "face sweep: loaded {} {} sample(s) in {:.1}s",
        out.len(),
        label,
        elapsed
    );
    Ok(out)
}

pub async fn detect_layout_hints(
    input: &str,
    config: &ClipDetectConfig,
) -> Result<ClipLayoutHints> {
    let mut hints = ClipLayoutHints::default();
    if !config.enabled {
        return Ok(hints);
    }

    let is_local = Path::new(input).exists();
    let duration_secs = if is_local && config.scan_full_clip {
        probe_media_duration_secs(Path::new(input)).await
    } else {
        None
    };
    let source_dims = if is_local {
        probe_media_dimensions(input).await
    } else {
        Some((config.frame_width, config.frame_height))
    };
    let sample_times = build_sample_times(config, is_local, duration_secs);
    let total_samples = sample_times.len();
    if total_samples > 0 {
        let first = sample_times.first().copied().unwrap_or(0.0);
        let last = sample_times.last().copied().unwrap_or(first);
        eprintln!(
            "clip detect: sampling {} frame(s) from {:.2}s to {:.2}s (step {:.2}s)",
            total_samples,
            first,
            last,
            config.sample_step_secs.max(0.0)
        );
    }
    let (face_budget, gameplay_budget) = split_analysis_budget(config.analysis_budget);
    let has_budget = config.analysis_budget.is_some();

    let mut face_samples: Vec<FaceSample> = Vec::new();
    let mut reticle_weight = 0.0f32;
    let mut reticle_sum_x = 0.0f32;
    let mut reticle_sum_y = 0.0f32;
    let mut gameplay_weight = 0.0f32;
    let mut gameplay_sum_x = 0.0f32;
    let mut gameplay_sum_y = 0.0f32;
    let mut gameplay_best: Option<GameplayObservation> = None;

    let face_detector = match YunetDetector::new(config) {
        Ok(detector) => detector,
        Err(err) => {
            eprintln!("clip detect: failed to load face model: {err:#}");
            None
        }
    };

    if total_samples > 0 {
        if let Some(detector) = face_detector.as_ref() {
            let face_start = Instant::now();
            let face_tick = Some(LoadingTicker::start(
                "clip detect: analyzing face samples",
                Duration::from_secs(5),
            ));
            let mut face_samples_attempted = 0usize;
            for (idx, seek) in sample_times.iter().copied().enumerate() {
                if let Some(budget) = face_budget {
                    let elapsed = face_start.elapsed();
                    if elapsed >= budget {
                        eprintln!(
                            "clip detect: face time budget {:.1}s hit after {}/{} sample(s); stopping early",
                            budget.as_secs_f32(),
                            face_samples_attempted,
                            total_samples
                        );
                        break;
                    }
                }
                eprintln!(
                    "clip detect: face sample {}/{} at {:.2}s",
                    idx + 1,
                    total_samples.max(1),
                    seek
                );
                let seek_arg = if seek > 0.0 { Some(seek) } else { None };
                let candidates = detect_face_candidates(input, seek_arg, detector, source_dims).await;
                face_samples_attempted += 1;
                if !candidates.is_empty() {
                    face_samples.push(FaceSample {
                        time: seek,
                        candidates,
                    });
                }
            }
            drop(face_tick);
            if face_samples_attempted == 0 {
                eprintln!("clip detect: no face samples attempted; skipping face detection");
            }
            eprintln!(
                "clip detect: face sampling complete in {:.1}s",
                face_start.elapsed().as_secs_f32()
            );
        } else {
            eprintln!("clip detect: face detection disabled; skipping face samples");
        }
    }

    if total_samples > 0 {
        if has_budget && gameplay_budget.is_none() {
            eprintln!("clip detect: gameplay time budget 0.0s; skipping gameplay samples");
        } else {
            let gameplay_config = read_clip_gameplay_config(config.frame_width, config.frame_height);
            let gameplay_detector = match ClipGameplayDetector::new(&gameplay_config) {
                Ok(detector) => detector,
                Err(err) => {
                    eprintln!("clip detect: failed to load gameplay model: {err:#}");
                    None
                }
            };

            let gameplay_start = Instant::now();
            let gameplay_tick = Some(LoadingTicker::start(
                "clip detect: analyzing gameplay samples",
                Duration::from_secs(5),
            ));
            let mut gameplay_samples_attempted = 0usize;
            for (idx, seek) in sample_times.iter().copied().enumerate() {
                if let Some(budget) = gameplay_budget {
                    let elapsed = gameplay_start.elapsed();
                    if elapsed >= budget {
                        eprintln!(
                            "clip detect: gameplay time budget {:.1}s hit after {}/{} sample(s); stopping early",
                            budget.as_secs_f32(),
                            gameplay_samples_attempted,
                            total_samples
                        );
                        break;
                    }
                }
                eprintln!(
                    "clip detect: gameplay sample {}/{} at {:.2}s",
                    idx + 1,
                    total_samples.max(1),
                    seek
                );
                let seek_arg = if seek > 0.0 { Some(seek) } else { None };
                let frame = match extract_frame_rgb(
                    input,
                    seek_arg,
                    config.frame_width,
                    config.frame_height,
                )
                .await
                {
                    Ok(frame) => frame,
                    Err(err) => {
                        eprintln!("clip detect: frame extraction failed ({seek:.2}s): {err:#}");
                        continue;
                    }
                };
                gameplay_samples_attempted += 1;

                if let Some(reticle) =
                    detect_reticle_candidate(&frame, config.frame_width, config.frame_height)
                {
                    reticle_sum_x += reticle.center.x * reticle.score;
                    reticle_sum_y += reticle.center.y * reticle.score;
                    reticle_weight += reticle.score;
                }

                if let Some(detector) = gameplay_detector.as_ref() {
                    let gameplay_frame = if gameplay_config.frame_width == config.frame_width
                        && gameplay_config.frame_height == config.frame_height
                    {
                        None
                    } else {
                        match extract_frame_rgb(
                            input,
                            seek_arg,
                            gameplay_config.frame_width,
                            gameplay_config.frame_height,
                        )
                        .await
                        {
                            Ok(data) => Some(data),
                            Err(err) => {
                                eprintln!(
                                    "clip detect: gameplay frame extraction failed ({seek:.2}s): {err:#}"
                                );
                                continue;
                            }
                        }
                    };
                    let (rgb, gw, gh) = if let Some(ref data) = gameplay_frame {
                        (
                            data.as_slice(),
                            gameplay_config.frame_width,
                            gameplay_config.frame_height,
                        )
                    } else {
                        (frame.as_slice(), config.frame_width, config.frame_height)
                    };

                    if let Some(obs) = detector.detect(rgb, gw, gh) {
                        let weight = obs.score.max(0.0);
                        if weight > 0.0 {
                            gameplay_sum_x += obs.center.x * weight;
                            gameplay_sum_y += obs.center.y * weight;
                            gameplay_weight += weight;
                        }
                        let replace = gameplay_best
                            .as_ref()
                            .map(|best| obs.score > best.score)
                            .unwrap_or(true);
                        if replace {
                            gameplay_best = Some(obs);
                        }
                    }
                }
            }
            drop(gameplay_tick);
            if gameplay_samples_attempted == 0 {
                eprintln!("clip detect: no gameplay frames sampled; skipping gameplay detection");
            }
            eprintln!(
                "clip detect: gameplay sampling complete in {:.1}s",
                gameplay_start.elapsed().as_secs_f32()
            );
        }
    }

    let mut face_best: Option<FaceConsensus> = None;
    let mut face_observations: Vec<FaceObservation> = Vec::new();
    if let Some(detector) = face_detector.as_ref() {
        if let Some(result) = select_face_with_relaxation(
            &face_samples,
            total_samples,
            detector,
            config.face_score_threshold,
        ) {
            if result.pass_label != "strict" {
                eprintln!(
                    "clip detect: relaxing face filters -> {}",
                    result.pass_label
                );
            }
            face_best = Some(result.best);
            face_observations = result.observations;
        }
    }
    hints.face_box = face_best.map(|c| c.rect);

    if config.track_face {
        if let Some(best) = &face_best {
            let best_center = rect_center(best.rect);
            let mut points: Vec<crate::clip_layout::FaceTrackPoint> = face_observations
                .iter()
                .filter_map(|obs| {
                    let center = rect_center(obs.rect);
                    let dist = (center.x - best_center.x)
                        .abs()
                        .max((center.y - best_center.y).abs());
                    if dist <= 0.12 {
                        Some(crate::clip_layout::FaceTrackPoint {
                            time: obs.time,
                            rect: obs.rect,
                        })
                    } else {
                        None
                    }
                })
                .collect();
            points.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
            if points.len() >= 2 {
                hints.face_track = Some(crate::clip_layout::FaceTrack { points });
            }
        }
    }

    if let Some(best) = face_best {
        eprintln!(
            "clip detect: face pick t={:.2}s score={:.4}",
            best.time, best.score
        );
    }
    if face_debug_enabled() {
        if let Some(rect) = hints.face_box {
            eprintln!(
                "clip detect: face box x={:.3} y={:.3} w={:.3} h={:.3}",
                rect.x, rect.y, rect.w, rect.h
            );
        } else {
            eprintln!("clip detect: face box not detected");
        }
    }
    if let Some(best) = gameplay_best {
        log_gameplay_debug(&format!(
            "clip detect: gameplay pick score={:.4} center x={:.3} y={:.3}",
            best.score, best.center.x, best.center.y
        ));
    }

    if gameplay_weight > 0.0 {
        hints.game_center = Some(NormalizedPoint {
            x: clamp_unit(gameplay_sum_x / gameplay_weight),
            y: clamp_unit(gameplay_sum_y / gameplay_weight),
        });
    } else if reticle_weight > 0.0 {
        hints.game_center = Some(NormalizedPoint {
            x: clamp_unit(reticle_sum_x / reticle_weight),
            y: clamp_unit(reticle_sum_y / reticle_weight),
        });
    }

    Ok(hints)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FaceSizeStatus {
    Ok,
    TooSmall,
    TooLarge,
}

fn build_face_scan_regions() -> Vec<FaceScanRegion> {
    let mut regions = Vec::new();
    regions.push(FaceScanRegion {
        name: "full",
        rect: NormalizedRect {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        },
    });

    for (idx, raw_size) in FACE_SCAN_REGION_SIZES.iter().enumerate() {
        let size = (*raw_size).clamp(0.25, 1.0);
        if size >= 0.99 {
            continue;
        }
        let right = 1.0 - size;
        let bottom = 1.0 - size;
        let label = |corner: &str| -> &'static str {
            match (idx, corner) {
                (0, "tl") => "top_left",
                (0, "tr") => "top_right",
                (0, "bl") => "bottom_left",
                (0, "br") => "bottom_right",
                (1, "tl") => "top_left_zoom",
                (1, "tr") => "top_right_zoom",
                (1, "bl") => "bottom_left_zoom",
                (1, "br") => "bottom_right_zoom",
                _ => "region",
            }
        };
        regions.push(FaceScanRegion {
            name: label("tl"),
            rect: NormalizedRect {
                x: 0.0,
                y: 0.0,
                w: size,
                h: size,
            },
        });
        regions.push(FaceScanRegion {
            name: label("tr"),
            rect: NormalizedRect {
                x: right,
                y: 0.0,
                w: size,
                h: size,
            },
        });
        regions.push(FaceScanRegion {
            name: label("bl"),
            rect: NormalizedRect {
                x: 0.0,
                y: bottom,
                w: size,
                h: size,
            },
        });
        regions.push(FaceScanRegion {
            name: label("br"),
            rect: NormalizedRect {
                x: right,
                y: bottom,
                w: size,
                h: size,
            },
        });
    }

    regions
}

fn centered_face_region(center: NormalizedPoint, size: f32) -> NormalizedRect {
    let size = size.clamp(0.1, 1.0);
    let half = size / 2.0;
    let x = (center.x - half).clamp(0.0, 1.0 - size);
    let y = (center.y - half).clamp(0.0, 1.0 - size);
    NormalizedRect {
        x,
        y,
        w: size,
        h: size,
    }
}

fn region_is_full(region: NormalizedRect) -> bool {
    region.x <= 0.001 && region.y <= 0.001 && region.w >= 0.999 && region.h >= 0.999
}

fn best_raw_score(candidates: &[FaceCandidate]) -> f32 {
    candidates
        .iter()
        .map(|c| c.raw_score)
        .fold(f32::NEG_INFINITY, |acc, v| acc.max(v))
}

fn build_face_tile_regions(depth: usize) -> Vec<NormalizedRect> {
    let tiles = 1usize << depth;
    let tiles_f = tiles as f32;
    let size = (1.0 / tiles_f).clamp(0.01, 1.0);
    let mut regions = Vec::with_capacity(tiles * tiles);
    for row in 0..tiles {
        for col in 0..tiles {
            let x = (col as f32 * size).clamp(0.0, 1.0 - size);
            let y = (row as f32 * size).clamp(0.0, 1.0 - size);
            regions.push(NormalizedRect {
                x,
                y,
                w: size,
                h: size,
            });
        }
    }
    regions
}

fn rect_inside_region(rect: NormalizedRect, region: NormalizedRect, margin: f32) -> bool {
    if region_is_full(region) {
        return true;
    }
    let mut margin = margin.max(0.0);
    let max_margin = (region.w.min(region.h) * 0.2).min(0.08);
    if margin > max_margin {
        margin = max_margin;
    }
    let left = region.x + margin;
    let right = region.x + region.w - margin;
    let top = region.y + margin;
    let bottom = region.y + region.h - margin;
    if right <= left || bottom <= top {
        return false;
    }
    rect.x >= left
        && rect.y >= top
        && rect.x + rect.w <= right
        && rect.y + rect.h <= bottom
}

async fn detect_face_candidates(
    input: &str,
    seek: Option<f32>,
    detector: &YunetDetector,
    source_dims: Option<(u32, u32)>,
) -> Vec<FaceCandidate> {
    let regions = build_face_scan_regions();
    let full_region = regions
        .iter()
        .find(|region| region.name == "full")
        .map(|region| region.rect)
        .unwrap_or(NormalizedRect {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        });

    let mut all_candidates = Vec::new();
    let dump_raw = face_dump_raw_enabled();
    let mut full_candidates = match detect_faces_in_region(
        input,
        seek,
        detector,
        full_region,
        source_dims,
    )
    .await
    {
        Ok(candidates) => candidates,
        Err(err) => {
            eprintln!("clip detect: face frame extraction failed (full): {err:#}");
            Vec::new()
        }
    };
    if face_debug_enabled() {
        eprintln!(
            "clip detect: face scan full -> {} candidates",
            full_candidates.len()
        );
    }
    if dump_raw {
        match detect_faces_in_region_raw(
            input,
            seek,
            detector,
            full_region,
            source_dims,
        )
        .await
        {
            Ok(raw_candidates) => {
                maybe_dump_face_candidates(input, seek, &raw_candidates, "faces_raw").await;
            }
            Err(err) => {
                eprintln!("clip detect: face raw dump failed (full): {err:#}");
            }
        }
    }

    let mut best_full: Option<FaceCandidate> = None;
    let mut has_ok_size = false;
    let mut seed_centers: Vec<NormalizedPoint> = Vec::new();
    for candidate in &full_candidates {
        let status = face_size_status(candidate.model_rect, detector.input_w, detector.input_h);
        if status == FaceSizeStatus::Ok {
            has_ok_size = true;
        }
        let replace = best_full
            .as_ref()
            .map(|best| candidate.score > best.score)
            .unwrap_or(true);
        if replace {
            best_full = Some(*candidate);
        }
    }
    if !has_ok_size && !full_candidates.is_empty() {
        let mut ranked = full_candidates.clone();
        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for candidate in ranked {
            let center = rect_center(candidate.rect);
            let separated = seed_centers.iter().all(|p| {
                (center.x - p.x).abs().max((center.y - p.y).abs()) > 0.12
            });
            if separated {
                seed_centers.push(center);
            }
            if seed_centers.len() >= 2 {
                break;
            }
        }
    }
    all_candidates.append(&mut full_candidates);

    if has_ok_size || seed_centers.is_empty() {
        all_candidates = apply_face_tile_search(
            input,
            seek,
            detector,
            source_dims,
            all_candidates,
        )
        .await;
        maybe_dump_face_candidates(input, seek, &all_candidates, "faces").await;
        return all_candidates;
    }

    let zoom_size = FACE_SCAN_REGION_SIZES
        .last()
        .copied()
        .unwrap_or(0.35)
        .clamp(0.2, 1.0);
    let mut extra_regions: Vec<(&'static str, NormalizedRect)> = Vec::new();
    for (idx, center) in seed_centers.iter().enumerate() {
        let label = if idx == 0 { "zoom" } else { "zoom_alt" };
        let zoom_region = centered_face_region(*center, zoom_size);
        extra_regions.push((label, zoom_region));
    }

    for (label, region) in extra_regions {
        match detect_faces_in_region(input, seek, detector, region, source_dims).await {
            Ok(mut candidates) => {
                if face_debug_enabled() {
                    eprintln!(
                        "clip detect: face scan {} -> {} candidates",
                        label,
                        candidates.len()
                    );
                }
                all_candidates.append(&mut candidates);
            }
            Err(err) => {
                eprintln!(
                    "clip detect: face frame extraction failed ({}): {err:#}",
                    label
                );
            }
        }
    }

    all_candidates = apply_face_tile_search(
        input,
        seek,
        detector,
        source_dims,
        all_candidates,
    )
    .await;
    maybe_dump_face_candidates(input, seek, &all_candidates, "faces").await;
    all_candidates
}

async fn apply_face_tile_search(
    input: &str,
    seek: Option<f32>,
    detector: &YunetDetector,
    source_dims: Option<(u32, u32)>,
    mut candidates: Vec<FaceCandidate>,
) -> Vec<FaceCandidate> {
    let Some(tile_config) = face_tile_config() else {
        return candidates;
    };
    let best = best_raw_score(&candidates);
    if best.is_finite() && best >= tile_config.min_score {
        return candidates;
    }
    let Some(source_dims) = source_dims else {
        eprintln!("clip detect: tile search skipped (unknown source dims)");
        return candidates;
    };
    eprintln!(
        "clip detect: tile search min_score={:.2} max_depth={}",
        tile_config.min_score, tile_config.max_depth
    );
    for depth in 1..=tile_config.max_depth {
        let mut found = Vec::new();
        let regions = build_face_tile_regions(depth);
        for region in regions {
            let raw = match detect_faces_in_region_raw(
                input,
                seek,
                detector,
                region,
                Some(source_dims),
            )
            .await
            {
                Ok(candidates) => candidates,
                Err(_) => continue,
            };
            for candidate in raw {
                if candidate.raw_score >= tile_config.min_score {
                    found.push(candidate);
                }
            }
        }
        if !found.is_empty() {
            eprintln!(
                "clip detect: tile search depth {} -> {} candidates",
                depth,
                found.len()
            );
            candidates.extend(found);
            return candidates;
        }
        eprintln!(
            "clip detect: tile search depth {} -> no candidates",
            depth
        );
    }
    candidates
}

async fn maybe_dump_face_candidates(
    input: &str,
    seek: Option<f32>,
    candidates: &[FaceCandidate],
    suffix: &str,
) {
    if face_dump_dir().is_none() {
        return;
    }
    if candidates.is_empty() {
        return;
    }
    if let Err(err) = dump_face_candidates(input, seek, candidates, suffix).await {
        eprintln!("clip detect: face dump failed: {err:#}");
    }
}

async fn dump_face_candidates(
    input: &str,
    seek: Option<f32>,
    candidates: &[FaceCandidate],
    suffix: &str,
) -> Result<()> {
    let Some(dump_dir) = face_dump_dir() else {
        return Ok(());
    };
    std::fs::create_dir_all(&dump_dir)?;

    let stem = Path::new(input)
        .file_stem()
        .and_then(|v| v.to_str())
        .unwrap_or("clip");
    let mut safe_stem = sanitize_filename_component(stem);
    if safe_stem.is_empty() {
        safe_stem = "clip".to_string();
    }
    let time_tag = seek.unwrap_or(0.0);
    let time_tag = format!("{time_tag:.2}").replace('.', "_");
    let out_path = dump_dir.join(format!("{safe_stem}_t{time_tag}_{suffix}.png"));

    let mut filters = Vec::new();
    for candidate in candidates {
        let rect = candidate.rect;
        filters.push(format!(
            "drawbox=x=iw*{:.6}:y=ih*{:.6}:w=iw*{:.6}:h=ih*{:.6}:color=red@0.6:t=2",
            rect.x.clamp(0.0, 1.0),
            rect.y.clamp(0.0, 1.0),
            rect.w.clamp(0.0, 1.0),
            rect.h.clamp(0.0, 1.0)
        ));
    }
    if filters.is_empty() {
        return Ok(());
    }
    let filter = filters.join(",");

    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-hide_banner").arg("-loglevel").arg("error");
    cmd.arg("-nostdin").arg("-y");
    if let Some(seek) = seek {
        cmd.arg("-ss").arg(format!("{seek:.3}"));
    }
    cmd.arg("-i").arg(input);
    cmd.arg("-frames:v").arg("1");
    cmd.arg("-vf").arg(filter);
    cmd.arg(&out_path);

    let output = cmd.output().await.context("running ffmpeg face dump")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg face dump failed: {}", stderr.trim());
    }
    eprintln!("clip detect: face dump -> {}", out_path.display());
    Ok(())
}

fn sanitize_filename_component(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c
            } else {
                '_'
            }
        })
        .collect()
}

async fn detect_faces_in_region(
    input: &str,
    seek: Option<f32>,
    detector: &YunetDetector,
    region: NormalizedRect,
    source_dims: Option<(u32, u32)>,
) -> Result<Vec<FaceCandidate>> {
    let frame = extract_face_frame_rgb(
        input,
        seek,
        detector.input_w,
        detector.input_h,
        region,
        source_dims,
    )
    .await?;
    Ok(detector.detect_faces(&frame))
}

async fn detect_faces_in_region_raw(
    input: &str,
    seek: Option<f32>,
    detector: &YunetDetector,
    region: NormalizedRect,
    source_dims: Option<(u32, u32)>,
) -> Result<Vec<FaceCandidate>> {
    let frame = extract_face_frame_rgb(
        input,
        seek,
        detector.input_w,
        detector.input_h,
        region,
        source_dims,
    )
    .await?;
    Ok(detector.detect_faces_raw(&frame))
}

fn face_size_status(rect: NormalizedRect, model_w: u32, model_h: u32) -> FaceSizeStatus {
    let w_px = rect.w * model_w as f32;
    let h_px = rect.h * model_h as f32;
    let max_dim = w_px.max(h_px);
    if max_dim < FACE_MIN_PX {
        FaceSizeStatus::TooSmall
    } else if max_dim > FACE_MAX_PX {
        FaceSizeStatus::TooLarge
    } else {
        FaceSizeStatus::Ok
    }
}

fn relax_face_score(base: f32, factor: f32, floor: f32) -> f32 {
    let relaxed = (base * factor).max(floor);
    if relaxed > base {
        base
    } else {
        relaxed
    }
}

fn face_score_floor(base: f32) -> f32 {
    relax_face_score(base, 0.5, 0.15)
}

fn strict_required_samples(total_samples: usize) -> usize {
    if total_samples <= 1 {
        1
    } else {
        ((total_samples as f32) * 0.5).ceil().max(2.0) as usize
    }
}

fn select_face_with_relaxation(
    samples: &[FaceSample],
    total_samples: usize,
    detector: &YunetDetector,
    base_score: f32,
) -> Option<FaceSelectionResult> {
    let relaxed_score = relax_face_score(base_score, 0.5, 0.15);
    let relaxed_min_samples = if total_samples <= 1 { 1 } else { 2 };
    let passes = [
        FaceSelectionPass {
            label: "strict",
            min_score: base_score,
            require_landmarks: FACE_LANDMARK_REQUIRED,
            region_margin: Some(FACE_REGION_MARGIN),
            model_edge_margin: FACE_EDGE_MARGIN,
            min_area: 0.003,
            max_area: 0.65,
            require_ok_size: true,
            min_samples: strict_required_samples(total_samples),
            max_drift: MAX_FACE_DRIFT,
        },
        FaceSelectionPass {
            label: "relaxed",
            min_score: relaxed_score,
            require_landmarks: false,
            region_margin: None,
            model_edge_margin: 0.0,
            min_area: 0.001,
            max_area: 0.85,
            require_ok_size: false,
            min_samples: relaxed_min_samples,
            max_drift: 0.35,
        },
    ];

    for pass in passes {
        let mut observations = Vec::new();
        for sample in samples {
            if let Some(best) = select_best_candidate_for_pass(sample, detector, &pass) {
                observations.push(FaceObservation {
                    rect: best.rect,
                    score: best.score,
                    time: sample.time,
                });
            }
        }
        if observations.is_empty() {
            continue;
        }
        let mut best = select_consensus_face(&observations);
        if let Some(consensus) = &best {
            if consensus.count < pass.min_samples || consensus.max_dist > pass.max_drift {
                best = None;
            }
        }
        if let Some(best) = best {
            return Some(FaceSelectionResult {
                best,
                observations,
                pass_label: pass.label,
            });
        }
    }

    None
}

fn select_best_candidate_for_pass(
    sample: &FaceSample,
    detector: &YunetDetector,
    pass: &FaceSelectionPass,
) -> Option<FaceCandidate> {
    let mut best: Option<(i32, FaceCandidate)> = None;
    for candidate in &sample.candidates {
        if !candidate_passes(candidate, detector, pass) {
            continue;
        }
        let priority = if region_is_full(candidate.region) {
            if face_size_status(candidate.model_rect, detector.input_w, detector.input_h)
                == FaceSizeStatus::Ok
            {
                1
            } else {
                0
            }
        } else if face_size_status(candidate.model_rect, detector.input_w, detector.input_h)
            == FaceSizeStatus::Ok
        {
            3
        } else {
            2
        };
        let replace = match best {
            None => true,
            Some((best_priority, best_candidate)) => {
                priority > best_priority
                    || (priority == best_priority && candidate.score > best_candidate.score)
            }
        };
        if replace {
            best = Some((priority, *candidate));
        }
    }
    best.map(|(_, candidate)| candidate)
}

fn candidate_passes(
    candidate: &FaceCandidate,
    detector: &YunetDetector,
    pass: &FaceSelectionPass,
) -> bool {
    if !candidate.raw_score.is_finite() || candidate.raw_score < pass.min_score {
        return false;
    }
    if !candidate.score.is_finite() || candidate.score <= 0.0 {
        return false;
    }
    if pass.require_landmarks && !candidate.landmarks_ok {
        return false;
    }
    if let Some(margin) = pass.region_margin {
        if !region_is_full(candidate.region)
            && !rect_inside_region(candidate.rect, candidate.region, margin)
        {
            return false;
        }
    }
    if pass.model_edge_margin > 0.0 {
        let rect = candidate.model_rect;
        if rect.x <= pass.model_edge_margin
            || rect.y <= pass.model_edge_margin
            || rect.x + rect.w >= 1.0 - pass.model_edge_margin
            || rect.y + rect.h >= 1.0 - pass.model_edge_margin
        {
            return false;
        }
    }
    let area = (candidate.model_rect.w * candidate.model_rect.h).max(0.0);
    if !area.is_finite() || area < pass.min_area || area > pass.max_area {
        return false;
    }
    if pass.require_ok_size
        && face_size_status(candidate.model_rect, detector.input_w, detector.input_h)
            != FaceSizeStatus::Ok
    {
        return false;
    }
    true
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_size(value: &str) -> Option<(u32, u32)> {
    let mut parts = value.split('x');
    let w = parts.next()?.trim().parse::<u32>().ok()?;
    let h = parts.next()?.trim().parse::<u32>().ok()?;
    if w > 0 && h > 0 {
        Some((w, h))
    } else {
        None
    }
}

fn build_sample_times(
    config: &ClipDetectConfig,
    is_local: bool,
    duration_secs: Option<f32>,
) -> Vec<f32> {
    if !is_local {
        let seek = config.sample_start_secs.max(0.0);
        return vec![seek];
    }

    if config.scan_full_clip {
        if let Some(duration) = duration_secs.filter(|v| v.is_finite() && *v > 0.0) {
            let start = config.sample_start_secs.max(0.0).min(duration);
            let end = if duration > 0.1 { duration - 0.05 } else { duration };
            let end = end.max(start);
            let span = end - start;
            if span <= 0.0 {
                return vec![start];
            }

            let base_step = config.sample_step_secs.max(0.05);
            let mut count = (span / base_step).floor() as usize + 1;
            let mut step = base_step;
            if count > MAX_FULL_SAMPLES {
                count = MAX_FULL_SAMPLES;
                if count > 1 {
                    step = span / (count - 1) as f32;
                } else {
                    step = 0.0;
                }
            }

            let mut times = Vec::with_capacity(count);
            for idx in 0..count {
                times.push(start + step * idx as f32);
            }
            return times;
        }
    }

    let sample_count = config.sample_count.max(1);
    let mut times = Vec::with_capacity(sample_count);
    let start = config.sample_start_secs.max(0.0);
    let step = config.sample_step_secs.max(0.01);
    for idx in 0..sample_count {
        times.push(start + step * idx as f32);
    }
    times
}

fn face_debug_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("CLIP_FACE_DEBUG")
            .ok()
            .and_then(|v| parse_bool(&v))
            .unwrap_or(false)
    })
}

fn face_dump_dir() -> Option<PathBuf> {
    static CACHED: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            std::env::var("CLIP_FACE_DUMP_DIR")
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        })
        .clone()
}

fn face_dump_raw_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("CLIP_FACE_DUMP_RAW")
            .ok()
            .and_then(|v| parse_bool(&v))
            .unwrap_or(false)
    })
}

#[derive(Clone, Copy, Debug)]
struct FaceTileConfig {
    min_score: f32,
    max_depth: usize,
}

fn face_tile_config() -> Option<FaceTileConfig> {
    let min_score_env = std::env::var("CLIP_FACE_TILE_MIN_SCORE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let min_score = match min_score_env {
        Some(v) if v <= 0.0 => return None,
        Some(v) if v.is_finite() => v,
        Some(_) => DEFAULT_FACE_TILE_MIN_SCORE,
        None => DEFAULT_FACE_TILE_MIN_SCORE,
    };
    let max_depth = std::env::var("CLIP_FACE_TILE_MAX_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.clamp(1, 6))
        .unwrap_or(DEFAULT_FACE_TILE_MAX_DEPTH);
    Some(FaceTileConfig {
        min_score: min_score.clamp(0.01, 0.99),
        max_depth,
    })
}

const YUNET_STRIDES: [usize; 3] = [8, 16, 32];

#[derive(Clone, Copy, Debug, Default)]
struct YunetOutputMap {
    cls: [Option<usize>; 3],
    obj: [Option<usize>; 3],
    bbox: [Option<usize>; 3],
    kps: [Option<usize>; 3],
}

impl YunetOutputMap {
    fn from_output_names<'a>(names: impl IntoIterator<Item = &'a str>, count: usize) -> Self {
        let mut map = YunetOutputMap::default();
        for (idx, name) in names.into_iter().enumerate() {
            match name {
                "cls_8" => map.cls[0] = Some(idx),
                "cls_16" => map.cls[1] = Some(idx),
                "cls_32" => map.cls[2] = Some(idx),
                "obj_8" => map.obj[0] = Some(idx),
                "obj_16" => map.obj[1] = Some(idx),
                "obj_32" => map.obj[2] = Some(idx),
                "bbox_8" => map.bbox[0] = Some(idx),
                "bbox_16" => map.bbox[1] = Some(idx),
                "bbox_32" => map.bbox[2] = Some(idx),
                "kps_8" => map.kps[0] = Some(idx),
                "kps_16" => map.kps[1] = Some(idx),
                "kps_32" => map.kps[2] = Some(idx),
                _ => {}
            }
        }

        if !map.is_complete() && count >= 9 {
            map.cls = [Some(0), Some(1), Some(2)];
            map.obj = [Some(3), Some(4), Some(5)];
            map.bbox = [Some(6), Some(7), Some(8)];
            if count >= 12 {
                map.kps = [Some(9), Some(10), Some(11)];
            }
        }

        map
    }

    fn from_model(model: &InferenceModel) -> Self {
        let mut names = Vec::new();
        if let Ok(outlets) = model.output_outlets() {
            for outlet in outlets.iter() {
                let name = model
                    .outlet_label(*outlet)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| model.node(outlet.node).name.clone());
                names.push(name);
            }
            return YunetOutputMap::from_output_names(
                names.iter().map(|s| s.as_str()),
                outlets.len(),
            );
        }

        YunetOutputMap::default()
    }

    #[cfg(feature = "ort")]
    fn from_session_outputs(outputs: &[ort::value::Outlet]) -> Self {
        YunetOutputMap::from_output_names(
            outputs.iter().map(|o| o.name()),
            outputs.len(),
        )
    }

    fn is_complete(&self) -> bool {
        self.cls.iter().all(|v| v.is_some())
            && self.obj.iter().all(|v| v.is_some())
            && self.bbox.iter().all(|v| v.is_some())
    }
}

#[derive(Clone, Copy, Debug)]
enum ModelLayout {
    Nchw,
    Nhwc,
}

fn resolve_model_input(
    model: &InferenceModel,
    config: &ClipDetectConfig,
) -> (u32, u32, ModelLayout) {
    let fallback = (config.frame_width, config.frame_height, ModelLayout::Nchw);
    let input_fact = match model.input_fact(0) {
        Ok(fact) => fact,
        Err(_) => return fallback,
    };
    let dims: Vec<Option<usize>> = input_fact
        .shape
        .dims()
        .map(|d| d.concretize().and_then(|d| d.to_usize().ok()))
        .collect();
    if dims.len() != 4 {
        return fallback;
    }
    let (input_h, input_w, layout) = if dims.get(1).and_then(|d| *d) == Some(3) {
        (
            dims.get(2).and_then(|d| *d),
            dims.get(3).and_then(|d| *d),
            ModelLayout::Nchw,
        )
    } else if dims.get(3).and_then(|d| *d) == Some(3) {
        (
            dims.get(1).and_then(|d| *d),
            dims.get(2).and_then(|d| *d),
            ModelLayout::Nhwc,
        )
    } else {
        return fallback;
    };
    let (Some(input_h), Some(input_w)) = (input_h, input_w) else {
        return fallback;
    };
    let input_w = match u32::try_from(input_w) {
        Ok(val) if val > 0 => val,
        _ => return fallback,
    };
    let input_h = match u32::try_from(input_h) {
        Ok(val) if val > 0 => val,
        _ => return fallback,
    };
    (input_w, input_h, layout)
}

enum YunetBackend {
    Tract(TypedRunnableModel<TypedModel>),
    #[cfg(feature = "ort")]
    Ort(std::sync::Mutex<Session>),
}

struct YunetDetector {
    backend: YunetBackend,
    input_w: u32,
    input_h: u32,
    layout: ModelLayout,
    outputs: YunetOutputMap,
    score_floor: f32,
}

impl YunetDetector {
    fn new(config: &ClipDetectConfig) -> Result<Option<Self>> {
        let Some(model_path) = config.face_model_path.as_ref() else {
            eprintln!("clip detect: face model not found; using heuristic fallback");
            return Ok(None);
        };

        let face_start = Instant::now();
        let face_tick =
            LoadingTicker::start("clip detect: loading face model", Duration::from_secs(5));
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .with_context(|| format!("loading face model at {model_path}"))?;
        let (input_w, input_h, layout) = resolve_model_input(&model, config);
        let input_shape = match layout {
            ModelLayout::Nchw => tvec!(1, 3, input_h as usize, input_w as usize),
            ModelLayout::Nhwc => tvec!(1, input_h as usize, input_w as usize, 3),
        };
        #[cfg(feature = "ort")]
        let mut outputs = YunetOutputMap::from_model(&model);
        #[cfg(not(feature = "ort"))]
        let outputs = YunetOutputMap::from_model(&model);
        let backend = match config.face_backend {
            FaceBackend::Tract => {
                let model = model
                    .with_input_fact(
                        0,
                        InferenceFact::dt_shape(f32::datum_type(), input_shape.clone()),
                    )?
                    .into_optimized()?
                    .into_runnable()?;
                YunetBackend::Tract(model)
            }
            FaceBackend::Ort => {
                #[cfg(feature = "ort")]
                {
                    drop(model);
                    let session = build_ort_session(model_path)?;
                    outputs = YunetOutputMap::from_session_outputs(session.outputs());
                    YunetBackend::Ort(std::sync::Mutex::new(session))
                }
                #[cfg(not(feature = "ort"))]
                {
                    anyhow::bail!(
                        "face backend 'ort' requested but autoclip was built without the ort feature"
                    );
                }
            }
            FaceBackend::Auto => {
                #[cfg(feature = "ort")]
                {
                    match build_ort_session(model_path) {
                        Ok(session) => {
                            outputs = YunetOutputMap::from_session_outputs(session.outputs());
                            drop(model);
                            YunetBackend::Ort(std::sync::Mutex::new(session))
                        }
                        Err(err) => {
                            eprintln!(
                                "clip detect: ORT init failed ({err:#}); falling back to tract"
                            );
                            let model = model
                                .with_input_fact(
                                    0,
                                    InferenceFact::dt_shape(
                                        f32::datum_type(),
                                        input_shape.clone(),
                                    ),
                                )?
                                .into_optimized()?
                                .into_runnable()?;
                            YunetBackend::Tract(model)
                        }
                    }
                }
                #[cfg(not(feature = "ort"))]
                {
                    let model = model
                        .with_input_fact(
                            0,
                            InferenceFact::dt_shape(f32::datum_type(), input_shape.clone()),
                        )?
                        .into_optimized()?
                        .into_runnable()?;
                    YunetBackend::Tract(model)
                }
            }
        };
        drop(face_tick);
        eprintln!(
            "clip detect: face model loaded in {:.1}s",
            face_start.elapsed().as_secs_f32()
        );
        match &backend {
            YunetBackend::Tract(_) => eprintln!("clip detect: face backend=tract"),
            #[cfg(feature = "ort")]
            YunetBackend::Ort(_) => eprintln!("clip detect: face backend=ort"),
        }

        let score_floor = face_score_floor(config.face_score_threshold);

        Ok(Some(Self {
            backend,
            input_w,
            input_h,
            layout,
            outputs,
            score_floor,
        }))
    }

    fn detect_faces(&self, frame: &FaceFrame) -> Vec<FaceCandidate> {
        self.detect_faces_internal(frame, self.score_floor, false)
    }

    fn detect_faces_raw(&self, frame: &FaceFrame) -> Vec<FaceCandidate> {
        self.detect_faces_internal(frame, 0.0, true)
    }

    fn detect_faces_internal(
        &self,
        frame: &FaceFrame,
        min_score: f32,
        raw: bool,
    ) -> Vec<FaceCandidate> {
        let rgb = &frame.rgb;
        let input = match self.layout {
            ModelLayout::Nchw => rgb_to_bgr_chw(rgb, self.input_w as usize, self.input_h as usize),
            ModelLayout::Nhwc => rgb_to_bgr_hwc(rgb, self.input_w as usize, self.input_h as usize),
        };
        let Some(input) = input else { return Vec::new(); };

        let outputs = match &self.backend {
            YunetBackend::Tract(model) => {
                let tensor = match self.layout {
                    ModelLayout::Nchw => Tensor::from_shape(
                        &[1usize, 3, self.input_h as usize, self.input_w as usize],
                        &input,
                    )
                    .ok(),
                    ModelLayout::Nhwc => Tensor::from_shape(
                        &[1usize, self.input_h as usize, self.input_w as usize, 3],
                        &input,
                    )
                    .ok(),
                };
                let Some(tensor) = tensor else { return Vec::new(); };
                model.run(tvec!(tensor.into())).ok()
            }
            #[cfg(feature = "ort")]
            YunetBackend::Ort(session) => {
                session
                    .lock()
                    .ok()
                    .and_then(|mut guard| {
                        run_ort_session(
                            &mut *guard,
                            &input,
                            self.input_w,
                            self.input_h,
                            self.layout,
                        )
                    })
            }
        };
        let Some(outputs) = outputs else { return Vec::new(); };
        if outputs.is_empty() {
            return Vec::new();
        }

        let debug = face_debug_enabled();
        if debug {
            let shapes: Vec<String> = outputs
                .iter()
                .map(|t| format!("{:?}", t.shape()))
                .collect();
            eprintln!(
                "clip detect: face outputs={} min_score={:.2}",
                shapes.join(" | "),
                min_score
            );
        }

        let mut best_raw = None;
        let mut candidates = Vec::new();
        if self.outputs.is_complete() {
            self.collect_candidates_from_yunet_outputs(
                &outputs,
                frame.mapping.as_ref(),
                frame.region,
                min_score,
                raw,
                &mut best_raw,
                &mut candidates,
            );
        } else {
            if debug {
                eprintln!("clip detect: yunet output labels missing; falling back to raw scan");
            }
            for output in outputs {
                self.collect_candidates_from_output(
                    &output,
                    frame.mapping.as_ref(),
                    frame.region,
                    min_score,
                    raw,
                    &mut best_raw,
                    &mut candidates,
                );
            }
        }

        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        if candidates.len() > MAX_FACE_CANDIDATES {
            candidates.truncate(MAX_FACE_CANDIDATES);
        }

        if debug {
            if let Some((score, rect)) = best_raw {
                eprintln!(
                    "clip detect: best raw face score={:.3} rect x={:.3} y={:.3} w={:.3} h={:.3}",
                    score, rect.x, rect.y, rect.w, rect.h
                );
            }
            if candidates.is_empty() {
                eprintln!("clip detect: no face passed threshold {:.2}", min_score);
            }
        }

        candidates
    }

    fn collect_candidates_from_output(
        &self,
        output: &Tensor,
        mapping: Option<&FrameMapping>,
        region: NormalizedRect,
        min_score: f32,
        raw: bool,
        best_raw: &mut Option<(f32, NormalizedRect)>,
        candidates: &mut Vec<FaceCandidate>,
    ) {
        let output = match output.to_array_view::<f32>() {
            Ok(val) => val,
            Err(_) => return,
        };
        let shape = output.shape();
        if shape.len() < 2 {
            return;
        }
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        let flat = match output.as_slice() {
            Some(val) => val,
            None => return,
        };
        if rows == 0 || cols < 5 || flat.len() < rows * cols {
            return;
        }

        for row in 0..rows {
            let base = row * cols;
            let x = flat[base];
            let y = flat[base + 1];
            let w = flat[base + 2];
            let h = flat[base + 3];
            let score = flat[base + 4];
            if !(score.is_finite() && x.is_finite() && y.is_finite() && w.is_finite() && h.is_finite()) {
                continue;
            }

            let rect = match normalize_rect(x, y, w, h, self.input_w, self.input_h) {
                Some(r) => r,
                None => continue,
            };
            if best_raw.map(|(s, _)| score > s).unwrap_or(true) {
                *best_raw = Some((score, rect));
            }
            if !raw && score < min_score {
                continue;
            }

            let weighted_score = if raw {
                score
            } else {
                face_candidate_weighted_score(rect, score)
            };
            if weighted_score.is_finite() && (raw || weighted_score > 0.0) {
                let mapped = match mapping {
                    Some(m) => match m.map_rect(rect) {
                        Some(val) => val,
                        None => continue,
                    },
                    None => rect,
                };
                candidates.push(FaceCandidate {
                    rect: mapped,
                    model_rect: rect,
                    raw_score: score,
                    score: weighted_score,
                    landmarks_ok: false,
                    region,
                });
            }
        }
    }

    fn collect_candidates_from_yunet_outputs(
        &self,
        outputs: &TVec<TValue>,
        mapping: Option<&FrameMapping>,
        region: NormalizedRect,
        min_score: f32,
        raw: bool,
        best_raw: &mut Option<(f32, NormalizedRect)>,
        candidates: &mut Vec<FaceCandidate>,
    ) {
        for (scale_idx, stride) in YUNET_STRIDES.iter().enumerate() {
            let Some(cls_idx) = self.outputs.cls.get(scale_idx).and_then(|v| *v) else {
                continue;
            };
            let Some(obj_idx) = self.outputs.obj.get(scale_idx).and_then(|v| *v) else {
                continue;
            };
            let Some(bbox_idx) = self.outputs.bbox.get(scale_idx).and_then(|v| *v) else {
                continue;
            };
            let kps_idx = self.outputs.kps.get(scale_idx).and_then(|v| *v);

            let cls = match outputs.get(cls_idx) {
                Some(val) => match val.to_array_view::<f32>() {
                    Ok(view) => view,
                    Err(_) => continue,
                },
                None => continue,
            };
            let obj = match outputs.get(obj_idx) {
                Some(val) => match val.to_array_view::<f32>() {
                    Ok(view) => view,
                    Err(_) => continue,
                },
                None => continue,
            };
            let bbox = match outputs.get(bbox_idx) {
                Some(val) => match val.to_array_view::<f32>() {
                    Ok(view) => view,
                    Err(_) => continue,
                },
                None => continue,
            };
            let kps = kps_idx.and_then(|idx| outputs.get(idx)).and_then(|val| {
                val.to_array_view::<f32>().ok()
            });

            let cls_vals = match cls.as_slice() {
                Some(val) => val,
                None => continue,
            };
            let obj_vals = match obj.as_slice() {
                Some(val) => val,
                None => continue,
            };
            let bbox_vals = match bbox.as_slice() {
                Some(val) => val,
                None => continue,
            };
            let kps_vals = kps.as_ref().and_then(|view| view.as_slice());
            if cls_vals.len() != obj_vals.len() {
                continue;
            }

            let stride = *stride;
            let stride_f = stride as f32;
            let grid_w = (self.input_w as usize) / stride;
            let grid_h = (self.input_h as usize) / stride;
            if grid_w * grid_h != cls_vals.len() {
                continue;
            }
            let bbox_layout = match bbox_layout(bbox.shape(), cls_vals.len()) {
                Some(layout) => layout,
                None => continue,
            };
            let kps_layout = kps
                .as_ref()
                .and_then(|view| kps_layout(view.shape(), cls_vals.len()));
            let landmarks_available = kps_vals.is_some() && kps_layout.is_some();
            if !landmarks_available && face_debug_enabled() {
                eprintln!(
                    "clip detect: missing landmarks for stride {}; using bbox-only",
                    stride
                );
            }

            for idx in 0..cls_vals.len() {
                let cls_score = sigmoid(cls_vals[idx]);
                let obj_score = sigmoid(obj_vals[idx]);
                let score = (cls_score * obj_score).sqrt();
                if !score.is_finite() || (!raw && score < min_score) {
                    continue;
                }

                let (l, t, r, b) = match bbox_at(bbox_vals, idx, bbox_layout) {
                    Some(vals) => vals,
                    None => continue,
                };
                if !(l.is_finite() && t.is_finite() && r.is_finite() && b.is_finite()) {
                    continue;
                }

                let cx = (idx % grid_w) as f32 + 0.5;
                let cy = (idx / grid_w) as f32 + 0.5;
                let cx = cx * stride_f;
                let cy = cy * stride_f;

                let x1 = (cx - l * stride_f).clamp(0.0, self.input_w as f32);
                let y1 = (cy - t * stride_f).clamp(0.0, self.input_h as f32);
                let x2 = (cx + r * stride_f).clamp(0.0, self.input_w as f32);
                let y2 = (cy + b * stride_f).clamp(0.0, self.input_h as f32);
                if x2 <= x1 || y2 <= y1 {
                    continue;
                }

                let rect = NormalizedRect {
                    x: x1 / self.input_w as f32,
                    y: y1 / self.input_h as f32,
                    w: (x2 - x1) / self.input_w as f32,
                    h: (y2 - y1) / self.input_h as f32,
                };
                if best_raw.map(|(s, _)| score > s).unwrap_or(true) {
                    *best_raw = Some((score, rect));
                }
                let mut model_rect = rect;
                let mut landmarks_ok = false;
                if let (Some(kps_vals), Some(layout)) = (kps_vals, kps_layout) {
                    if let Some(points) = decode_keypoints(
                        kps_vals,
                        idx,
                        layout,
                        grid_w,
                        stride_f,
                        self.input_w,
                        self.input_h,
                    ) {
                        if landmarks_frontal(&points, model_rect) {
                            landmarks_ok = true;
                        }
                        let center = points_center(&points);
                        model_rect = recenter_rect(model_rect, center);
                    }
                }
                let mapped_rect = match mapping {
                    Some(m) => match m.map_rect(model_rect) {
                        Some(val) => val,
                        None => continue,
                    },
                    None => model_rect,
                };
                let weighted_score = if raw {
                    score
                } else {
                    face_candidate_weighted_score(model_rect, score)
                };
                if weighted_score.is_finite() && (raw || weighted_score > 0.0) {
                    candidates.push(FaceCandidate {
                        rect: mapped_rect,
                        model_rect,
                        raw_score: score,
                        score: weighted_score,
                        landmarks_ok,
                        region,
                    });
                }
            }
        }
    }
}

#[cfg(feature = "ort")]
fn build_ort_session(model_path: &str) -> Result<Session> {
    let session = Session::builder()?
        .with_execution_providers([ep::CUDA::default().build(), ep::CPU::default().build()])?
        .commit_from_file(model_path)
        .with_context(|| format!("loading ORT session at {model_path}"))?;
    Ok(session)
}

#[cfg(feature = "ort")]
fn run_ort_session(
    session: &mut Session,
    input: &[f32],
    input_w: u32,
    input_h: u32,
    layout: ModelLayout,
) -> Option<TVec<TValue>> {
    let shape = match layout {
        ModelLayout::Nchw => [1usize, 3, input_h as usize, input_w as usize],
        ModelLayout::Nhwc => [1usize, input_h as usize, input_w as usize, 3],
    };
    let input_tensor = TensorRef::from_array_view((shape, input)).ok()?;
    let outputs = session.run(ort::inputs![input_tensor]).ok()?;
    let mut out = tvec!();
    for value in outputs.values() {
        let tensor = ort_value_to_tensor(&value)?;
        out.push(tensor.into());
    }
    Some(out)
}

#[cfg(feature = "ort")]
fn ort_value_to_tensor(
    value: &ort::value::Value,
) -> Option<Tensor> {
    let (shape, data) = value.try_extract_tensor::<f32>().ok()?;
    let shape = ort_shape_to_usize(shape)?;
    Tensor::from_shape(&shape, data).ok()
}

#[cfg(feature = "ort")]
fn ort_shape_to_usize(shape: &ort::tensor::Shape) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(shape.len());
    for dim in shape.iter() {
        let dim = usize::try_from(*dim).ok()?;
        out.push(dim);
    }
    Some(out)
}

fn rgb_to_bgr_chw(rgb: &[u8], width: usize, height: usize) -> Option<Vec<f32>> {
    let expected = width * height * 3;
    if rgb.len() < expected {
        return None;
    }
    let mut out = vec![0f32; expected];
    let area = width * height;
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let r = rgb[idx] as f32;
            let g = rgb[idx + 1] as f32;
            let b = rgb[idx + 2] as f32;
            let offset = y * width + x;
            out[offset] = b;
            out[area + offset] = g;
            out[area * 2 + offset] = r;
        }
    }
    Some(out)
}

fn rgb_to_bgr_hwc(rgb: &[u8], width: usize, height: usize) -> Option<Vec<f32>> {
    let expected = width * height * 3;
    if rgb.len() < expected {
        return None;
    }
    let mut out = vec![0f32; expected];
    for idx in (0..expected).step_by(3) {
        let r = rgb[idx] as f32;
        let g = rgb[idx + 1] as f32;
        let b = rgb[idx + 2] as f32;
        out[idx] = b;
        out[idx + 1] = g;
        out[idx + 2] = r;
    }
    Some(out)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BboxLayout {
    Nchw { plane: usize },
    Nhwc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KpsLayout {
    Nchw { plane: usize },
    Nhwc,
}

fn bbox_layout(shape: &[usize], cls_len: usize) -> Option<BboxLayout> {
    if shape.len() == 4 {
        if shape[1] == 4 {
            let plane = shape[2].saturating_mul(shape[3]);
            if plane == cls_len {
                return Some(BboxLayout::Nchw { plane });
            }
        } else if shape[3] == 4 {
            let plane = shape[1].saturating_mul(shape[2]);
            if plane == cls_len {
                return Some(BboxLayout::Nhwc);
            }
        }
    } else if shape.len() == 3 && shape[2] == 4 && shape[1] == cls_len {
        return Some(BboxLayout::Nhwc);
    } else if shape.len() == 2 && shape[1] == 4 && shape[0] == cls_len {
        return Some(BboxLayout::Nhwc);
    }
    None
}

fn kps_layout(shape: &[usize], cls_len: usize) -> Option<KpsLayout> {
    if shape.len() == 4 {
        if shape[1] == 10 {
            let plane = shape[2].saturating_mul(shape[3]);
            if plane == cls_len {
                return Some(KpsLayout::Nchw { plane });
            }
        } else if shape[3] == 10 {
            let plane = shape[1].saturating_mul(shape[2]);
            if plane == cls_len {
                return Some(KpsLayout::Nhwc);
            }
        }
    } else if shape.len() == 3 {
        if shape[2] == 10 && shape[1] == cls_len {
            return Some(KpsLayout::Nhwc);
        }
        if shape[1] == 10 && shape[2] == cls_len {
            return Some(KpsLayout::Nchw { plane: cls_len });
        }
    } else if shape.len() == 2 && shape[1] == 10 && shape[0] == cls_len {
        return Some(KpsLayout::Nhwc);
    }
    None
}

fn bbox_at(bbox_vals: &[f32], idx: usize, layout: BboxLayout) -> Option<(f32, f32, f32, f32)> {
    match layout {
        BboxLayout::Nchw { plane } => {
            if idx >= plane || bbox_vals.len() < plane * 4 {
                return None;
            }
            let l = bbox_vals[idx];
            let t = bbox_vals[plane + idx];
            let r = bbox_vals[plane * 2 + idx];
            let b = bbox_vals[plane * 3 + idx];
            Some((l, t, r, b))
        }
        BboxLayout::Nhwc => {
            let base = idx * 4;
            if base + 3 >= bbox_vals.len() {
                return None;
            }
            Some((
                bbox_vals[base],
                bbox_vals[base + 1],
                bbox_vals[base + 2],
                bbox_vals[base + 3],
            ))
        }
    }
}

fn kps_at(kps_vals: &[f32], idx: usize, layout: KpsLayout) -> Option<[f32; 10]> {
    let mut out = [0.0f32; 10];
    match layout {
        KpsLayout::Nchw { plane } => {
            if idx >= plane || kps_vals.len() < plane * 10 {
                return None;
            }
            for k in 0..10 {
                out[k] = kps_vals[plane * k + idx];
            }
        }
        KpsLayout::Nhwc => {
            let base = idx * 10;
            if base + 9 >= kps_vals.len() {
                return None;
            }
            for k in 0..10 {
                out[k] = kps_vals[base + k];
            }
        }
    }
    Some(out)
}

fn decode_keypoints(
    kps_vals: &[f32],
    idx: usize,
    layout: KpsLayout,
    grid_w: usize,
    stride_f: f32,
    input_w: u32,
    input_h: u32,
) -> Option<[NormalizedPoint; 5]> {
    let coords = kps_at(kps_vals, idx, layout)?;
    let cx = (idx % grid_w) as f32 + 0.5;
    let cy = (idx / grid_w) as f32 + 0.5;
    let cx = cx * stride_f;
    let cy = cy * stride_f;
    let mut points = [NormalizedPoint { x: 0.0, y: 0.0 }; 5];
    for i in 0..5 {
        let dx = coords[i * 2];
        let dy = coords[i * 2 + 1];
        if !(dx.is_finite() && dy.is_finite()) {
            return None;
        }
        let x = (cx + dx * stride_f).clamp(0.0, input_w as f32);
        let y = (cy + dy * stride_f).clamp(0.0, input_h as f32);
        points[i] = NormalizedPoint {
            x: clamp_unit(x / input_w as f32),
            y: clamp_unit(y / input_h as f32),
        };
    }
    Some(points)
}

fn points_center(points: &[NormalizedPoint; 5]) -> NormalizedPoint {
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    for point in points {
        sum_x += point.x;
        sum_y += point.y;
    }
    let denom = points.len().max(1) as f32;
    NormalizedPoint {
        x: clamp_unit(sum_x / denom),
        y: clamp_unit(sum_y / denom),
    }
}

#[derive(Clone, Copy, Debug)]
struct LandmarkSet {
    left_eye: NormalizedPoint,
    right_eye: NormalizedPoint,
    nose: NormalizedPoint,
    left_mouth: NormalizedPoint,
    right_mouth: NormalizedPoint,
}

fn classify_landmarks(points: &[NormalizedPoint; 5]) -> Option<LandmarkSet> {
    let mut idxs = [0usize, 1, 2, 3, 4];
    idxs.sort_by(|&a, &b| {
        points[a]
            .y
            .partial_cmp(&points[b].y)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let eye_a = points[idxs[0]];
    let eye_b = points[idxs[1]];
    let nose = points[idxs[2]];
    let mouth_a = points[idxs[3]];
    let mouth_b = points[idxs[4]];
    let (left_eye, right_eye) = if eye_a.x <= eye_b.x {
        (eye_a, eye_b)
    } else {
        (eye_b, eye_a)
    };
    let (left_mouth, right_mouth) = if mouth_a.x <= mouth_b.x {
        (mouth_a, mouth_b)
    } else {
        (mouth_b, mouth_a)
    };
    Some(LandmarkSet {
        left_eye,
        right_eye,
        nose,
        left_mouth,
        right_mouth,
    })
}

fn midpoint(a: NormalizedPoint, b: NormalizedPoint) -> NormalizedPoint {
    NormalizedPoint {
        x: clamp_unit((a.x + b.x) * 0.5),
        y: clamp_unit((a.y + b.y) * 0.5),
    }
}

fn landmarks_within_rect(points: &[NormalizedPoint; 5], rect: NormalizedRect) -> bool {
    let margin = (rect.w.min(rect.h) * 0.2).min(0.08);
    let left = (rect.x - margin).max(0.0);
    let right = (rect.x + rect.w + margin).min(1.0);
    let top = (rect.y - margin).max(0.0);
    let bottom = (rect.y + rect.h + margin).min(1.0);
    for point in points {
        if !point.x.is_finite() || !point.y.is_finite() {
            return false;
        }
        if point.x < left || point.x > right || point.y < top || point.y > bottom {
            return false;
        }
    }
    true
}

fn landmarks_frontal(points: &[NormalizedPoint; 5], rect: NormalizedRect) -> bool {
    if !landmarks_within_rect(points, rect) {
        return false;
    }
    let landmarks = match classify_landmarks(points) {
        Some(val) => val,
        None => return false,
    };
    let eye_dx = landmarks.right_eye.x - landmarks.left_eye.x;
    let eye_dy = (landmarks.right_eye.y - landmarks.left_eye.y).abs();
    if eye_dx <= 0.0 {
        return false;
    }
    let eye_dist = (eye_dx * eye_dx + eye_dy * eye_dy).sqrt();
    if eye_dist <= 1e-6 {
        return false;
    }
    let rect_w = rect.w.max(1e-4);
    let eye_ratio = eye_dx / rect_w;
    if eye_ratio < FACE_LANDMARK_MIN_EYE_RATIO || eye_ratio > 0.98 {
        return false;
    }
    if (eye_dy / eye_dist) > FACE_LANDMARK_MAX_EYE_TILT {
        return false;
    }

    let mouth_dx = landmarks.right_mouth.x - landmarks.left_mouth.x;
    if mouth_dx <= 0.0 {
        return false;
    }
    let mouth_dy = (landmarks.right_mouth.y - landmarks.left_mouth.y).abs();
    if (mouth_dy / mouth_dx) > FACE_LANDMARK_MAX_MOUTH_TILT {
        return false;
    }

    let eye_mid = midpoint(landmarks.left_eye, landmarks.right_eye);
    let mouth_mid = midpoint(landmarks.left_mouth, landmarks.right_mouth);

    let nose_offset = (landmarks.nose.x - eye_mid.x).abs() / eye_dx;
    if nose_offset > FACE_LANDMARK_NOSE_CENTER_MAX {
        return false;
    }
    let mouth_offset = (mouth_mid.x - eye_mid.x).abs() / eye_dx;
    if mouth_offset > FACE_LANDMARK_MOUTH_CENTER_MAX {
        return false;
    }

    let rect_h = rect.h.max(1e-4);
    let min_eye_nose = rect_h * FACE_LANDMARK_EYE_NOSE_MIN;
    let min_nose_mouth = rect_h * FACE_LANDMARK_NOSE_MOUTH_MIN;
    if landmarks.nose.y <= eye_mid.y + min_eye_nose {
        return false;
    }
    if mouth_mid.y <= landmarks.nose.y + min_nose_mouth {
        return false;
    }

    true
}

fn normalize_rect(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    input_w: u32,
    input_h: u32,
) -> Option<NormalizedRect> {
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    let max_val = x.max(y).max(w).max(h);
    let (mut nx, mut ny, mut nw, mut nh) = if max_val <= 1.5 {
        (x, y, w, h)
    } else {
        (
            x / input_w as f32,
            y / input_h as f32,
            w / input_w as f32,
            h / input_h as f32,
        )
    };

    nx = clamp_unit(nx);
    ny = clamp_unit(ny);
    nw = nw.clamp(0.0, 1.0 - nx);
    nh = nh.clamp(0.0, 1.0 - ny);
    if nw <= 0.0 || nh <= 0.0 {
        return None;
    }
    Some(NormalizedRect {
        x: nx,
        y: ny,
        w: nw,
        h: nh,
    })
}

fn face_candidate_weighted_score(rect: NormalizedRect, score: f32) -> f32 {
    if !score.is_finite() {
        return f32::MIN;
    }
    let area = (rect.w * rect.h).max(0.0);
    if !area.is_finite() {
        return f32::MIN;
    }
    let aspect = if rect.h > 0.0 { rect.w / rect.h } else { 0.0 };
    let aspect_weight = if (0.5..=1.8).contains(&aspect) { 1.0 } else { 0.6 };
    let size_weight = if area < 0.006 {
        0.7
    } else if area > 0.55 {
        0.6
    } else {
        1.0
    };
    let weight = size_weight * aspect_weight;
    score * weight
}

async fn extract_frame_rgb(
    input: &str,
    seek_secs: Option<f32>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-hide_banner").arg("-loglevel").arg("error");
    cmd.arg("-nostdin");
    if let Some(seek) = seek_secs {
        cmd.arg("-ss").arg(format!("{seek:.3}"));
    }
    cmd.arg("-i").arg(input);
    cmd.arg("-frames:v").arg("1");
    cmd.arg("-vf")
        .arg(format!("scale={width}:{height}:flags=bicubic"));
    cmd.arg("-pix_fmt").arg("rgb24");
    cmd.arg("-f").arg("rawvideo");
    cmd.arg("-");

    let output = cmd.output().await.context("running ffmpeg for detection frame")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg frame extract failed: {}", stderr.trim());
    }

    let expected = (width * height * 3) as usize;
    if output.stdout.len() < expected {
        anyhow::bail!(
            "ffmpeg frame extract returned {} bytes, expected {}",
            output.stdout.len(),
            expected
        );
    }
    let mut data = output.stdout;
    data.truncate(expected);
    Ok(data)
}

#[derive(Clone, Copy, Debug)]
struct FrameMapping {
    full_w: f32,
    full_h: f32,
    region_x: f32,
    region_y: f32,
    region_w: f32,
    region_h: f32,
    model_w: f32,
    model_h: f32,
    scale: f32,
    pad_x: f32,
    pad_y: f32,
}

impl FrameMapping {
    fn map_rect(&self, rect: NormalizedRect) -> Option<NormalizedRect> {
        if self.scale <= 0.0
            || self.full_w <= 0.0
            || self.full_h <= 0.0
            || self.region_w <= 0.0
            || self.region_h <= 0.0
        {
            return None;
        }
        let x0_model = rect.x * self.model_w;
        let y0_model = rect.y * self.model_h;
        let x1_model = (rect.x + rect.w) * self.model_w;
        let y1_model = (rect.y + rect.h) * self.model_h;
        let x0_region = (x0_model - self.pad_x) / self.scale;
        let y0_region = (y0_model - self.pad_y) / self.scale;
        let x1_region = (x1_model - self.pad_x) / self.scale;
        let y1_region = (y1_model - self.pad_y) / self.scale;
        let x0_src = x0_region + self.region_x;
        let y0_src = y0_region + self.region_y;
        let x1_src = x1_region + self.region_x;
        let y1_src = y1_region + self.region_y;
        if !x0_src.is_finite() || !y0_src.is_finite() || !x1_src.is_finite() || !y1_src.is_finite()
        {
            return None;
        }
        if x1_src <= 0.0
            || y1_src <= 0.0
            || x0_src >= self.full_w
            || y0_src >= self.full_h
        {
            return None;
        }
        let x0 = x0_src.clamp(0.0, self.full_w);
        let y0 = y0_src.clamp(0.0, self.full_h);
        let x1 = x1_src.clamp(0.0, self.full_w);
        let y1 = y1_src.clamp(0.0, self.full_h);
        let w = (x1 - x0).max(0.0);
        let h = (y1 - y0).max(0.0);
        if w <= 0.0 || h <= 0.0 {
            return None;
        }
        Some(NormalizedRect {
            x: clamp_unit(x0 / self.full_w),
            y: clamp_unit(y0 / self.full_h),
            w: clamp_unit(w / self.full_w),
            h: clamp_unit(h / self.full_h),
        })
    }
}

#[derive(Clone, Debug)]
struct FaceFrame {
    rgb: Vec<u8>,
    mapping: Option<FrameMapping>,
    region: NormalizedRect,
}

fn build_face_filter(model_w: u32, model_h: u32, region: NormalizedRect) -> String {
    let mut filters = Vec::new();
    if !region_is_full(region) {
        filters.push(format!(
            "crop=iw*{:.4}:ih*{:.4}:iw*{:.4}:ih*{:.4}",
            region.w, region.h, region.x, region.y
        ));
    }
    filters.push(format!(
        "scale={model_w}:{model_h}:force_original_aspect_ratio=decrease"
    ));
    filters.push(format!(
        "pad={model_w}:{model_h}:(ow-iw)/2:(oh-ih)/2"
    ));
    filters.join(",")
}

fn build_frame_mapping(
    full_w: u32,
    full_h: u32,
    model_w: u32,
    model_h: u32,
    region: NormalizedRect,
) -> FrameMapping {
    let full_w_f = full_w as f32;
    let full_h_f = full_h as f32;
    let region_x = (region.x * full_w_f).clamp(0.0, full_w_f);
    let region_y = (region.y * full_h_f).clamp(0.0, full_h_f);
    let mut region_w = (region.w * full_w_f).clamp(1.0, full_w_f - region_x);
    let mut region_h = (region.h * full_h_f).clamp(1.0, full_h_f - region_y);
    if region_x + region_w > full_w_f {
        region_w = (full_w_f - region_x).max(1.0);
    }
    if region_y + region_h > full_h_f {
        region_h = (full_h_f - region_y).max(1.0);
    }
    let model_w_f = model_w as f32;
    let model_h_f = model_h as f32;
    let scale = (model_w_f / region_w).min(model_h_f / region_h);
    let pad_x = (model_w_f - region_w * scale) / 2.0;
    let pad_y = (model_h_f - region_h * scale) / 2.0;
    FrameMapping {
        full_w: full_w_f,
        full_h: full_h_f,
        region_x,
        region_y,
        region_w,
        region_h,
        model_w: model_w_f,
        model_h: model_h_f,
        scale,
        pad_x,
        pad_y,
    }
}

async fn extract_face_frame_rgb(
    input: &str,
    seek_secs: Option<f32>,
    model_w: u32,
    model_h: u32,
    region: NormalizedRect,
    source_dims: Option<(u32, u32)>,
) -> Result<FaceFrame> {
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-hide_banner").arg("-loglevel").arg("error");
    cmd.arg("-nostdin");
    if let Some(seek) = seek_secs {
        cmd.arg("-ss").arg(format!("{seek:.3}"));
    }
    cmd.arg("-i").arg(input);
    cmd.arg("-frames:v").arg("1");
    let filter = build_face_filter(model_w, model_h, region);
    cmd.arg("-vf").arg(filter);
    cmd.arg("-pix_fmt").arg("rgb24");
    cmd.arg("-f").arg("rawvideo");
    cmd.arg("-");

    let output = cmd.output().await.context("running ffmpeg for face frame")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg face frame extract failed: {}", stderr.trim());
    }

    let expected = (model_w * model_h * 3) as usize;
    if output.stdout.len() < expected {
        anyhow::bail!(
            "ffmpeg face frame extract returned {} bytes, expected {}",
            output.stdout.len(),
            expected
        );
    }
    let mut data = output.stdout;
    data.truncate(expected);

    let mapping = source_dims.map(|(src_w, src_h)| {
        build_frame_mapping(src_w, src_h, model_w, model_h, region)
    });

    Ok(FaceFrame {
        rgb: data,
        mapping,
        region,
    })
}

async fn probe_media_duration_secs(path: &Path) -> Option<f32> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("default=nw=1:nk=1")
        .arg(path.as_os_str())
        .output()
        .await
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let secs = stdout.trim().parse::<f32>().ok()?;
    if secs.is_finite() && secs > 0.0 {
        Some(secs)
    } else {
        None
    }
}

async fn probe_media_dimensions(input: &str) -> Option<(u32, u32)> {
    if let Some(dims) = probe_media_dimensions_ffprobe(input).await {
        return Some(dims);
    }
    probe_media_dimensions_ffmpeg(input).await
}

async fn probe_media_dimensions_ffprobe(input: &str) -> Option<(u32, u32)> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("stream=width,height")
        .arg("-of")
        .arg("csv=p=0:s=x")
        .arg(input)
        .output()
        .await
        .ok()?;

    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut parts = stdout.trim().split('x');
    let w = parts.next()?.trim().parse::<u32>().ok()?;
    let h = parts.next()?.trim().parse::<u32>().ok()?;
    if w > 0 && h > 0 {
        Some((w, h))
    } else {
        None
    }
}

async fn probe_media_dimensions_ffmpeg(input: &str) -> Option<(u32, u32)> {
    let output = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-nostdin")
        .arg("-i")
        .arg(input)
        .output()
        .await
        .ok()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    parse_ffmpeg_dims(&stderr)
}

fn parse_ffmpeg_dims(text: &str) -> Option<(u32, u32)> {
    for line in text.lines() {
        if !line.contains("Video:") {
            continue;
        }
        for raw in line.split(|c: char| c.is_whitespace() || c == ',') {
            if let Some(dims) = parse_dims_token(raw) {
                return Some(dims);
            }
        }
    }
    None
}

fn parse_dims_token(token: &str) -> Option<(u32, u32)> {
    let token = token.trim_matches(|c: char| !(c.is_ascii_digit() || c == 'x'));
    if !token.contains('x') {
        return None;
    }
    let mut parts = token.split('x');
    let w = parts.next()?.parse::<u32>().ok()?;
    let h = parts.next()?.parse::<u32>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    if w >= 16 && h >= 16 {
        Some((w, h))
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug)]
struct FaceCandidate {
    rect: NormalizedRect,
    model_rect: NormalizedRect,
    raw_score: f32,
    score: f32,
    landmarks_ok: bool,
    region: NormalizedRect,
}

#[derive(Clone, Debug)]
struct FaceSample {
    time: f32,
    candidates: Vec<FaceCandidate>,
}

#[derive(Clone, Copy, Debug)]
struct FaceScanRegion {
    name: &'static str,
    rect: NormalizedRect,
}

#[derive(Clone, Copy, Debug)]
struct FaceObservation {
    rect: NormalizedRect,
    score: f32,
    time: f32,
}

#[derive(Clone, Copy, Debug)]
struct FaceConsensus {
    rect: NormalizedRect,
    score: f32,
    time: f32,
    count: usize,
    max_dist: f32,
}

#[derive(Clone, Copy, Debug)]
struct FaceSelectionPass {
    label: &'static str,
    min_score: f32,
    require_landmarks: bool,
    region_margin: Option<f32>,
    model_edge_margin: f32,
    min_area: f32,
    max_area: f32,
    require_ok_size: bool,
    min_samples: usize,
    max_drift: f32,
}

#[derive(Clone, Debug)]
struct FaceSelectionResult {
    best: FaceConsensus,
    observations: Vec<FaceObservation>,
    pass_label: &'static str,
}

fn rect_center(rect: NormalizedRect) -> NormalizedPoint {
    NormalizedPoint {
        x: rect.x + rect.w / 2.0,
        y: rect.y + rect.h / 2.0,
    }
}

fn select_consensus_face(observations: &[FaceObservation]) -> Option<FaceConsensus> {
    if observations.is_empty() {
        return None;
    }

    let mut clusters: Vec<FaceCluster> = Vec::new();
    for obs in observations {
        let center = rect_center(obs.rect);
        let mut best_idx = None;
        let mut best_dist = f32::MAX;
        for (idx, cluster) in clusters.iter().enumerate() {
            let cluster_center = cluster.center();
            let dist = (center.x - cluster_center.x).abs().max((center.y - cluster_center.y).abs());
            if dist < 0.12 && dist < best_dist {
                best_dist = dist;
                best_idx = Some(idx);
            }
        }
        if let Some(idx) = best_idx {
            clusters[idx].add(*obs);
        } else {
            clusters.push(FaceCluster::new(*obs));
        }
    }

    let mut best_cluster: Option<&FaceCluster> = None;
    for cluster in &clusters {
        let take = match best_cluster {
            None => true,
            Some(best) => {
                cluster.count > best.count
                    || (cluster.count == best.count && cluster.score_sum > best.score_sum)
            }
        };
        if take {
            best_cluster = Some(cluster);
        }
    }
    best_cluster.map(|cluster| cluster.to_consensus())
}

#[derive(Clone, Debug)]
struct FaceCluster {
    score_sum: f32,
    count: usize,
    rect_sum_x: f32,
    rect_sum_y: f32,
    rect_sum_w: f32,
    rect_sum_h: f32,
    center_sum_x: f32,
    center_sum_y: f32,
    dist_sum: f32,
    dist_count: usize,
    max_dist: f32,
    best: FaceObservation,
}

impl FaceCluster {
    fn new(obs: FaceObservation) -> Self {
        let center = rect_center(obs.rect);
        let weight = obs.score.max(0.0);
        Self {
            score_sum: weight,
            count: 1,
            rect_sum_x: obs.rect.x * weight,
            rect_sum_y: obs.rect.y * weight,
            rect_sum_w: obs.rect.w * weight,
            rect_sum_h: obs.rect.h * weight,
            center_sum_x: center.x * weight,
            center_sum_y: center.y * weight,
            dist_sum: 0.0,
            dist_count: 0,
            max_dist: 0.0,
            best: obs,
        }
    }

    fn add(&mut self, obs: FaceObservation) {
        let center_before = self.center();
        let weight = obs.score.max(0.0);
        if weight > 0.0 {
            self.score_sum += weight;
            self.count += 1;
            self.rect_sum_x += obs.rect.x * weight;
            self.rect_sum_y += obs.rect.y * weight;
            self.rect_sum_w += obs.rect.w * weight;
            self.rect_sum_h += obs.rect.h * weight;
            let center = rect_center(obs.rect);
            self.center_sum_x += center.x * weight;
            self.center_sum_y += center.y * weight;
            let dist = (center.x - center_before.x)
                .abs()
                .max((center.y - center_before.y).abs());
            self.dist_sum += dist;
            self.dist_count += 1;
            if dist > self.max_dist {
                self.max_dist = dist;
            }
        }
        if obs.score > self.best.score {
            self.best = obs;
        }
    }

    fn center(&self) -> NormalizedPoint {
        if self.score_sum > 0.0 {
            NormalizedPoint {
                x: clamp_unit(self.center_sum_x / self.score_sum),
                y: clamp_unit(self.center_sum_y / self.score_sum),
            }
        } else {
            rect_center(self.best.rect)
        }
    }

    fn to_consensus(&self) -> FaceConsensus {
        if self.score_sum > 0.0 {
            let rect = NormalizedRect {
                x: clamp_unit(self.rect_sum_x / self.score_sum),
                y: clamp_unit(self.rect_sum_y / self.score_sum),
                w: clamp_unit(self.rect_sum_w / self.score_sum),
                h: clamp_unit(self.rect_sum_h / self.score_sum),
            };
            FaceConsensus {
                rect,
                score: self.score_sum / self.count.max(1) as f32,
                time: self.best.time,
                count: self.count,
                max_dist: self.max_dist,
            }
        } else {
            FaceConsensus {
                rect: self.best.rect,
                score: self.best.score,
                time: self.best.time,
                count: self.count,
                max_dist: self.max_dist,
            }
        }
    }
}

fn recenter_rect(rect: NormalizedRect, center: NormalizedPoint) -> NormalizedRect {
    let w = rect.w.clamp(0.0, 1.0);
    let h = rect.h.clamp(0.0, 1.0);
    let mut x = center.x - w / 2.0;
    let mut y = center.y - h / 2.0;
    if x < 0.0 {
        x = 0.0;
    }
    if y < 0.0 {
        y = 0.0;
    }
    if x + w > 1.0 {
        x = 1.0 - w;
    }
    if y + h > 1.0 {
        y = 1.0 - h;
    }
    NormalizedRect {
        x: clamp_unit(x),
        y: clamp_unit(y),
        w,
        h,
    }
}

#[derive(Clone, Copy, Debug)]
struct ReticleCandidate {
    center: NormalizedPoint,
    score: f32,
}

fn detect_reticle_candidate(rgb: &[u8], width: u32, height: u32) -> Option<ReticleCandidate> {
    let w = width as usize;
    let h = height as usize;
    if w < 8 || h < 8 || rgb.len() < w * h * 3 {
        return None;
    }

    let mut gray = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let r = rgb[idx] as f32;
            let g = rgb[idx + 1] as f32;
            let b = rgb[idx + 2] as f32;
            let lum = (0.299 * r + 0.587 * g + 0.114 * b).round();
            gray[y * w + x] = lum.clamp(0.0, 255.0) as u8;
        }
    }

    let mut mag = vec![0u16; w * h];
    let mut total_mag = 0u64;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            let gx = gray[idx + 1] as i16 - gray[idx - 1] as i16;
            let gy = gray[idx + w] as i16 - gray[idx - w] as i16;
            let val = (gx.abs() + gy.abs()) as u16;
            mag[idx] = val;
            total_mag += val as u64;
        }
    }

    let mean_mag = total_mag as f32 / (w * h) as f32;
    let radius = 6i32;
    let step = 4i32;
    let mut best_score = 0.0f32;
    let mut best_center = None;
    let mut min_x = (w as f32 * 0.2) as i32;
    let mut max_x = (w as f32 * 0.8) as i32;
    let mut min_y = (h as f32 * 0.35) as i32;
    let mut max_y = (h as f32 * 0.9) as i32;
    min_x = max(min_x, radius + 1);
    min_y = max(min_y, radius + 1);
    max_x = min(max_x, w as i32 - radius - 2);
    max_y = min(max_y, h as i32 - radius - 2);

    for y in (min_y..=max_y).step_by(step as usize) {
        for x in (min_x..=max_x).step_by(step as usize) {
            let mut sum = 0u32;
            for dx in -radius..=radius {
                let ix = (x + dx) as usize;
                let iy = y as usize;
                sum += mag[iy * w + ix] as u32;
            }
            for dy in -radius..=radius {
                let ix = x as usize;
                let iy = (y + dy) as usize;
                sum += mag[iy * w + ix] as u32;
            }

            let center_bias = 1.0
                - ((x as f32 / width as f32 - 0.5).abs()
                    + (y as f32 / height as f32 - 0.55).abs())
                    .clamp(0.0, 0.8);
            let score = sum as f32 * (0.4 + center_bias * 0.6);
            if score > best_score {
                best_score = score;
                best_center = Some((x, y));
            }
        }
    }

    let min_score = (mean_mag * radius as f32 * 10.0).max(300.0);
    if best_score < min_score {
        return None;
    }

    best_center.map(|(x, y)| ReticleCandidate {
        center: NormalizedPoint {
            x: clamp_unit(x as f32 / width as f32),
            y: clamp_unit(y as f32 / height as f32),
        },
        score: best_score,
    })
}

fn clamp_unit(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(width: u32, height: u32, color: [u8; 3]) -> Vec<u8> {
        let mut data = vec![0u8; (width * height * 3) as usize];
        for idx in (0..data.len()).step_by(3) {
            data[idx] = color[0];
            data[idx + 1] = color[1];
            data[idx + 2] = color[2];
        }
        data
    }

    fn draw_cross(
        data: &mut [u8],
        width: u32,
        height: u32,
        cx: u32,
        cy: u32,
        radius: u32,
        color: [u8; 3],
    ) {
        let w = width as i32;
        let h = height as i32;
        let cx = cx as i32;
        let cy = cy as i32;
        let radius = radius as i32;
        for dx in -radius..=radius {
            let x = cx + dx;
            if x >= 0 && x < w && cy >= 0 && cy < h {
                let idx = ((cy as u32 * width + x as u32) * 3) as usize;
                data[idx] = color[0];
                data[idx + 1] = color[1];
                data[idx + 2] = color[2];
            }
        }
        for dy in -radius..=radius {
            let y = cy + dy;
            if cx >= 0 && cx < w && y >= 0 && y < h {
                let idx = ((y as u32 * width + cx as u32) * 3) as usize;
                data[idx] = color[0];
                data[idx + 1] = color[1];
                data[idx + 2] = color[2];
            }
        }
    }

    #[test]
    fn detect_reticle_candidate_finds_crosshair() {
        let width = 96;
        let height = 72;
        let mut frame = solid_frame(width, height, [20, 20, 20]);
        draw_cross(&mut frame, width, height, 48, 54, 6, [240, 240, 240]);

        let candidate = detect_reticle_candidate(&frame, width, height);
        assert!(candidate.is_some(), "expected crosshair to be detected");
        let center = candidate.unwrap().center;
        assert!((center.x - 0.5).abs() < 0.08);
        assert!((center.y - 0.75).abs() < 0.08);
    }

    #[test]
    fn build_sample_times_full_spreads_across_duration() {
        let config = ClipDetectConfig {
            enabled: true,
            sample_count: 3,
            sample_start_secs: 1.0,
            sample_step_secs: 2.0,
            frame_width: 640,
            frame_height: 360,
            face_model_path: None,
            face_backend: FaceBackend::Auto,
            face_score_threshold: 0.5,
            scan_full_clip: true,
            track_face: true,
            analysis_budget: None,
        };
        let times = build_sample_times(&config, true, Some(10.0));
        assert!(times.len() >= 2);
        let first = *times.first().unwrap();
        let last = *times.last().unwrap();
        assert!((first - 1.0).abs() < 1e-6);
        assert!(last > 7.0, "expected samples to reach near the clip end");
    }

    #[test]
    fn bbox_at_decodes_nchw() {
        let cls_len = 6usize;
        let shape = [1usize, 4, 2, 3];
        let layout = bbox_layout(&shape, cls_len).unwrap();
        assert!(matches!(layout, BboxLayout::Nchw { plane: 6 }));

        let mut bbox_vals = Vec::with_capacity(cls_len * 4);
        for v in 0..cls_len {
            bbox_vals.push(10.0 + v as f32);
        }
        for v in 0..cls_len {
            bbox_vals.push(20.0 + v as f32);
        }
        for v in 0..cls_len {
            bbox_vals.push(30.0 + v as f32);
        }
        for v in 0..cls_len {
            bbox_vals.push(40.0 + v as f32);
        }

        let (l, t, r, b) = bbox_at(&bbox_vals, 4, layout).unwrap();
        assert!((l - 14.0).abs() < 1e-6);
        assert!((t - 24.0).abs() < 1e-6);
        assert!((r - 34.0).abs() < 1e-6);
        assert!((b - 44.0).abs() < 1e-6);
    }

    #[test]
    fn bbox_at_decodes_nhwc() {
        let cls_len = 6usize;
        let shape = [1usize, 2, 3, 4];
        let layout = bbox_layout(&shape, cls_len).unwrap();
        assert_eq!(layout, BboxLayout::Nhwc);

        let mut bbox_vals = Vec::with_capacity(cls_len * 4);
        for idx in 0..cls_len {
            bbox_vals.push(100.0 + idx as f32);
            bbox_vals.push(200.0 + idx as f32);
            bbox_vals.push(300.0 + idx as f32);
            bbox_vals.push(400.0 + idx as f32);
        }

        let (l, t, r, b) = bbox_at(&bbox_vals, 3, layout).unwrap();
        assert!((l - 103.0).abs() < 1e-6);
        assert!((t - 203.0).abs() < 1e-6);
        assert!((r - 303.0).abs() < 1e-6);
        assert!((b - 403.0).abs() < 1e-6);
    }

    #[test]
    fn bbox_at_decodes_nhwc_packed() {
        let cls_len = 6usize;
        let shape = [1usize, 6, 4];
        let layout = bbox_layout(&shape, cls_len).unwrap();
        assert_eq!(layout, BboxLayout::Nhwc);

        let mut bbox_vals = Vec::with_capacity(cls_len * 4);
        for idx in 0..cls_len {
            bbox_vals.push(1.0 + idx as f32);
            bbox_vals.push(2.0 + idx as f32);
            bbox_vals.push(3.0 + idx as f32);
            bbox_vals.push(4.0 + idx as f32);
        }

        let (l, t, r, b) = bbox_at(&bbox_vals, 5, layout).unwrap();
        assert!((l - 6.0).abs() < 1e-6);
        assert!((t - 7.0).abs() < 1e-6);
        assert!((r - 8.0).abs() < 1e-6);
        assert!((b - 9.0).abs() < 1e-6);
    }

    #[test]
    fn kps_at_decodes_nchw() {
        let cls_len = 6usize;
        let shape = [1usize, 10, 2, 3];
        let layout = kps_layout(&shape, cls_len).unwrap();
        assert!(matches!(layout, KpsLayout::Nchw { plane: 6 }));

        let mut vals = Vec::with_capacity(cls_len * 10);
        for k in 0..10 {
            for idx in 0..cls_len {
                vals.push(k as f32 * 100.0 + idx as f32);
            }
        }

        let out = kps_at(&vals, 4, layout).unwrap();
        assert!((out[0] - 4.0).abs() < 1e-6);
        assert!((out[1] - 104.0).abs() < 1e-6);
        assert!((out[9] - 904.0).abs() < 1e-6);
    }

    #[test]
    fn kps_at_decodes_nhwc() {
        let cls_len = 6usize;
        let shape = [1usize, 2, 3, 10];
        let layout = kps_layout(&shape, cls_len).unwrap();
        assert_eq!(layout, KpsLayout::Nhwc);

        let mut vals = Vec::with_capacity(cls_len * 10);
        for idx in 0..cls_len {
            for k in 0..10 {
                vals.push(k as f32 * 100.0 + idx as f32);
            }
        }

        let out = kps_at(&vals, 5, layout).unwrap();
        assert!((out[0] - 5.0).abs() < 1e-6);
        assert!((out[1] - 105.0).abs() < 1e-6);
        assert!((out[9] - 905.0).abs() < 1e-6);
    }

    #[test]
    fn landmarks_frontal_accepts_centered() {
        let rect = NormalizedRect {
            x: 0.3,
            y: 0.25,
            w: 0.4,
            h: 0.5,
        };
        let points = [
            NormalizedPoint { x: 0.4, y: 0.35 },
            NormalizedPoint { x: 0.6, y: 0.36 },
            NormalizedPoint { x: 0.5, y: 0.5 },
            NormalizedPoint { x: 0.43, y: 0.65 },
            NormalizedPoint { x: 0.57, y: 0.66 },
        ];
        assert!(landmarks_frontal(&points, rect));
    }

    #[test]
    fn landmarks_frontal_rejects_profile() {
        let rect = NormalizedRect {
            x: 0.3,
            y: 0.25,
            w: 0.4,
            h: 0.5,
        };
        let points = [
            NormalizedPoint { x: 0.42, y: 0.35 },
            NormalizedPoint { x: 0.58, y: 0.36 },
            NormalizedPoint { x: 0.62, y: 0.5 },
            NormalizedPoint { x: 0.46, y: 0.66 },
            NormalizedPoint { x: 0.62, y: 0.67 },
        ];
        assert!(!landmarks_frontal(&points, rect));
    }
}
