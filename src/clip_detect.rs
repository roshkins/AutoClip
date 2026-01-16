use anyhow::{Context, Result};
use std::cmp::{max, min};
use std::path::Path;
use std::sync::OnceLock;
use tokio::process::Command;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::Factoid;
use tract_onnx::tract_hir::internal::DimLike;

use crate::clip_gameplay::{ClipGameplayDetector, GameplayObservation, read_clip_gameplay_config, log_gameplay_debug};
use crate::clip_layout::{ClipLayoutHints, NormalizedPoint, NormalizedRect};

#[derive(Clone, Debug)]
pub struct ClipDetectConfig {
    pub enabled: bool,
    pub sample_count: usize,
    pub sample_start_secs: f32,
    pub sample_step_secs: f32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub face_model_path: Option<String>,
    pub face_score_threshold: f32,
    pub scan_full_clip: bool,
    pub track_face: bool,
}

const DEFAULT_SAMPLE_COUNT: usize = 3;
const DEFAULT_SAMPLE_START: f32 = 1.0;
const DEFAULT_SAMPLE_STEP: f32 = 1.5;
const DEFAULT_FRAME_WIDTH: u32 = 960;
const DEFAULT_FRAME_HEIGHT: u32 = 540;
const DEFAULT_FACE_MODEL_PATH: &str = "models/face_detection_yunet_2023mar.onnx";
const DEFAULT_FACE_SCORE: f32 = 0.5;
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
const FACE_EDGE_MAX_DIST: f32 = 0.12;
const FACE_REGION_SCORE_MIN: f32 = 0.3;

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

    let track_face = std::env::var("CLIP_FACE_TRACK")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(true);

    let scan_full_clip = std::env::var("CLIP_DETECT_FULL")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false);

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
        face_score_threshold,
        scan_full_clip: scan_full_clip || track_face,
        track_face,
    }
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
    let mut face_samples: Vec<FaceObservation> = Vec::new();
    let mut reticle_weight = 0.0f32;
    let mut reticle_sum_x = 0.0f32;
    let mut reticle_sum_y = 0.0f32;
    let mut frames_attempted = 0usize;

    let face_detector = match YunetDetector::new(config) {
        Ok(detector) => detector,
        Err(err) => {
            eprintln!("clip detect: failed to load face model: {err:#}");
            None
        }
    };
    let gameplay_config = read_clip_gameplay_config(config.frame_width, config.frame_height);
    let gameplay_detector = match ClipGameplayDetector::new(&gameplay_config) {
        Ok(detector) => detector,
        Err(err) => {
            eprintln!("clip detect: failed to load gameplay model: {err:#}");
            None
        }
    };

    let mut gameplay_weight = 0.0f32;
    let mut gameplay_sum_x = 0.0f32;
    let mut gameplay_sum_y = 0.0f32;
    let mut gameplay_best: Option<GameplayObservation> = None;

    for seek in sample_times {
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
        frames_attempted += 1;

        if let Some(detector) = face_detector.as_ref() {
            if let Some(candidate) =
                detect_face_candidate(input, seek_arg, detector, source_dims).await
            {
                face_samples.push(FaceObservation {
                    rect: candidate.rect,
                    score: candidate.score,
                    time: seek,
                });
            }
        }

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

    if frames_attempted == 0 {
        eprintln!("clip detect: no frames sampled; skipping detection");
    }

    let mut face_best = select_consensus_face(&face_samples);
    if let Some(best) = &face_best {
        let min_samples = if total_samples <= 1 {
            1
        } else {
            ((total_samples as f32) * 0.5).ceil() as usize
        };
        let required = if total_samples <= 1 { 1 } else { min_samples.max(2) };
        if best.count < required || best.max_dist > MAX_FACE_DRIFT {
            face_best = None;
        }
    }
    hints.face_box = face_best.map(|c| c.rect);

    if config.track_face {
        if let Some(best) = &face_best {
            let best_center = rect_center(best.rect);
            let mut points: Vec<crate::clip_layout::FaceTrackPoint> = face_samples
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

fn region_is_full(region: NormalizedRect) -> bool {
    region.x <= 0.001 && region.y <= 0.001 && region.w >= 0.999 && region.h >= 0.999
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

fn edge_distance(rect: NormalizedRect) -> f32 {
    let left = rect.x;
    let top = rect.y;
    let right = 1.0 - (rect.x + rect.w);
    let bottom = 1.0 - (rect.y + rect.h);
    left.min(top).min(right).min(bottom)
}

fn is_edge_candidate(rect: NormalizedRect) -> bool {
    edge_distance(rect) <= FACE_EDGE_MAX_DIST
}

async fn detect_face_candidate(
    input: &str,
    seek: Option<f32>,
    detector: &YunetDetector,
    source_dims: Option<(u32, u32)>,
) -> Option<FaceCandidate> {
    let mut all_candidates = Vec::new();
    for region in build_face_scan_regions() {
        match detect_faces_in_region(input, seek, detector, region.rect, source_dims).await {
            Ok(mut candidates) => {
                if face_debug_enabled() {
                    eprintln!(
                        "clip detect: face scan {} -> {} candidates",
                        region.name,
                        candidates.len()
                    );
                }
                all_candidates.append(&mut candidates);
            }
            Err(err) => {
                eprintln!(
                    "clip detect: face frame extraction failed ({}): {err:#}",
                    region.name
                );
            }
        }
    }

    select_candidate(all_candidates, detector)
}

fn select_candidate(
    candidates: Vec<FaceCandidate>,
    detector: &YunetDetector,
) -> Option<FaceCandidate> {
    let mut best_sub_ok: Option<FaceCandidate> = None;
    let mut best_sub_any: Option<FaceCandidate> = None;
    let mut best_full_ok: Option<FaceCandidate> = None;
    let mut best_full_any: Option<FaceCandidate> = None;
    for candidate in candidates {
        let from_full = region_is_full(candidate.region);
        if !is_edge_candidate(candidate.rect) {
            continue;
        }
        if !from_full && !rect_inside_region(candidate.rect, candidate.region, FACE_REGION_MARGIN) {
            continue;
        }
        let ok_size =
            face_size_status(candidate.model_rect, detector.input_w, detector.input_h)
                == FaceSizeStatus::Ok;
        let slot = if from_full {
            if ok_size {
                &mut best_full_ok
            } else {
                &mut best_full_any
            }
        } else if ok_size {
            &mut best_sub_ok
        } else {
            &mut best_sub_any
        };
        let replace = slot
            .as_ref()
            .map(|best| candidate.score > best.score)
            .unwrap_or(true);
        if replace {
            *slot = Some(candidate);
        }
    }
    best_sub_ok
        .or(best_sub_any)
        .or(best_full_ok)
        .or(best_full_any)
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

const YUNET_STRIDES: [usize; 3] = [8, 16, 32];

#[derive(Clone, Copy, Debug, Default)]
struct YunetOutputMap {
    cls: [Option<usize>; 3],
    obj: [Option<usize>; 3],
    bbox: [Option<usize>; 3],
    kps: [Option<usize>; 3],
}

impl YunetOutputMap {
    fn from_model(model: &InferenceModel) -> Self {
        let mut map = YunetOutputMap::default();
        if let Ok(outlets) = model.output_outlets() {
            for (idx, outlet) in outlets.iter().enumerate() {
                let name = model
                    .outlet_label(*outlet)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| model.node(outlet.node).name.clone());
                match name.as_str() {
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

            if !map.is_complete() && outlets.len() >= 9 {
                map.cls = [Some(0), Some(1), Some(2)];
                map.obj = [Some(3), Some(4), Some(5)];
                map.bbox = [Some(6), Some(7), Some(8)];
                if outlets.len() >= 12 {
                    map.kps = [Some(9), Some(10), Some(11)];
                }
            }
        }

        map
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

struct YunetDetector {
    model: TypedRunnableModel<TypedModel>,
    input_w: u32,
    input_h: u32,
    layout: ModelLayout,
    outputs: YunetOutputMap,
    score_threshold: f32,
}

impl YunetDetector {
    fn new(config: &ClipDetectConfig) -> Result<Option<Self>> {
        let Some(model_path) = config.face_model_path.as_ref() else {
            eprintln!("clip detect: face model not found; using heuristic fallback");
            return Ok(None);
        };

        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .with_context(|| format!("loading face model at {model_path}"))?;
        let (input_w, input_h, layout) = resolve_model_input(&model, config);
        let input_shape = match layout {
            ModelLayout::Nchw => tvec!(1, 3, input_h as usize, input_w as usize),
            ModelLayout::Nhwc => tvec!(1, input_h as usize, input_w as usize, 3),
        };
        let outputs = YunetOutputMap::from_model(&model);
        let model = model
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))?
            .into_optimized()?
            .into_runnable()?;

        Ok(Some(Self {
            model,
            input_w,
            input_h,
            layout,
            outputs,
            score_threshold: config.face_score_threshold,
        }))
    }

    fn detect_faces(&self, frame: &FaceFrame) -> Vec<FaceCandidate> {
        let rgb = &frame.rgb;
        let input = match self.layout {
            ModelLayout::Nchw => rgb_to_bgr_chw(rgb, self.input_w as usize, self.input_h as usize),
            ModelLayout::Nhwc => rgb_to_bgr_hwc(rgb, self.input_w as usize, self.input_h as usize),
        };
        let Some(input) = input else { return Vec::new(); };

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

        let outputs = self.model.run(tvec!(tensor.into())).ok();
        let Some(outputs) = outputs else { return Vec::new(); };
        if outputs.is_empty() {
            return Vec::new();
        }

        let debug = face_debug_enabled();
        let min_score = if region_is_full(frame.region) {
            self.score_threshold
        } else {
            (self.score_threshold * 0.8).max(FACE_REGION_SCORE_MIN)
        };
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
                &mut best_raw,
                &mut candidates,
            );
        } else if FACE_LANDMARK_REQUIRED {
            if debug {
                eprintln!("clip detect: yunet outputs incomplete; landmarks required");
            }
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
        best_raw: &mut Option<(f32, NormalizedRect)>,
        candidates: &mut Vec<FaceCandidate>,
    ) {
        if FACE_LANDMARK_REQUIRED {
            if face_debug_enabled() {
                eprintln!("clip detect: skipping raw face scan; landmarks required");
            }
            return;
        }
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
            if rect.x <= FACE_EDGE_MARGIN
                || rect.y <= FACE_EDGE_MARGIN
                || rect.x + rect.w >= 1.0 - FACE_EDGE_MARGIN
                || rect.y + rect.h >= 1.0 - FACE_EDGE_MARGIN
            {
                continue;
            }
            if best_raw.map(|(s, _)| score > s).unwrap_or(true) {
                *best_raw = Some((score, rect));
            }
            if score < min_score {
                continue;
            }

            let score = face_candidate_score(rect, score);
            if score > 0.0 {
                let mapped = mapping.and_then(|m| m.map_rect(rect)).unwrap_or(rect);
                candidates.push(FaceCandidate {
                    rect: mapped,
                    model_rect: rect,
                    score,
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
            if FACE_LANDMARK_REQUIRED && (kps_vals.is_none() || kps_layout.is_none()) {
                if face_debug_enabled() {
                    eprintln!(
                        "clip detect: missing landmarks for stride {}; skipping",
                        stride
                    );
                }
                continue;
            }

            for idx in 0..cls_vals.len() {
                let cls_score = sigmoid(cls_vals[idx]);
                let obj_score = sigmoid(obj_vals[idx]);
                let score = (cls_score * obj_score).sqrt();
                if !score.is_finite() || score < min_score {
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
                if rect.x <= FACE_EDGE_MARGIN
                    || rect.y <= FACE_EDGE_MARGIN
                    || rect.x + rect.w >= 1.0 - FACE_EDGE_MARGIN
                    || rect.y + rect.h >= 1.0 - FACE_EDGE_MARGIN
                {
                    continue;
                }
                if best_raw.map(|(s, _)| score > s).unwrap_or(true) {
                    *best_raw = Some((score, rect));
                }
                let mut model_rect = rect;
                let mut landmarks = None;
                if let (Some(kps_vals), Some(layout)) = (kps_vals, kps_layout) {
                    landmarks = decode_keypoints(
                        kps_vals,
                        idx,
                        layout,
                        grid_w,
                        stride_f,
                        self.input_w,
                        self.input_h,
                    );
                }
                if FACE_LANDMARK_REQUIRED {
                    let Some(points) = landmarks else {
                        continue;
                    };
                    if !landmarks_frontal(&points, model_rect) {
                        continue;
                    }
                    let center = points_center(&points);
                    model_rect = recenter_rect(model_rect, center);
                } else if let Some(points) = landmarks {
                    let center = points_center(&points);
                    model_rect = recenter_rect(model_rect, center);
                }
                let mapped_rect = mapping
                    .and_then(|m| m.map_rect(model_rect))
                    .unwrap_or(model_rect);
                let score = face_candidate_score(model_rect, score);
                if score > 0.0 {
                    candidates.push(FaceCandidate {
                        rect: mapped_rect,
                        model_rect,
                        score,
                        region,
                    });
                }
            }
        }
    }
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

fn face_candidate_score(
    rect: NormalizedRect,
    score: f32,
) -> f32 {
    if !score.is_finite() {
        return f32::MIN;
    }
    let area = (rect.w * rect.h).max(0.0);
    if area < 0.003 || area > 0.65 {
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

#[derive(Clone, Copy, Debug)]
struct FaceCandidate {
    rect: NormalizedRect,
    model_rect: NormalizedRect,
    score: f32,
    region: NormalizedRect,
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
            face_score_threshold: 0.5,
            scan_full_clip: true,
            track_face: true,
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
