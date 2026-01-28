//! Face and gameplay detection pipelines.
//!
//! This module loads face detection/pose models, runs sampling across clips,
//! and produces layout hints for downstream rendering.

use anyhow::{Context, Result};
use std::cmp::{max, min};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use tokio::process::Command;
use serde::{Deserialize, Serialize};
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::Factoid;
use tract_onnx::tract_hir::internal::DimLike;

use crate::clip_gameplay::{
    ClipGameplayDetector, ClipLabelSet, ClipRegionObservation, log_gameplay_debug,
    parse_label_list, read_clip_gameplay_config,
};
use crate::clip_layout::{ClipLayoutHints, FaceFrameSpec, NormalizedPoint, NormalizedRect};
use crate::loading::LoadingTicker;
use crate::low_resource::low_resource_enabled;
use crate::profile::profile_span;

#[cfg(feature = "ort")]
use crate::gpu;
#[cfg(feature = "ort")]
use ort::ep;
#[cfg(feature = "ort")]
use ort::init_from;
#[cfg(feature = "ort")]
use ort::session::Session;
#[cfg(feature = "ort")]
use ort::value::TensorRef;

/// Backend selection for face detection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceBackend {
    Auto,
    Tract,
    Ort,
}

/// Configuration for face detection and clip sampling.
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
    pub face_track_step_secs: Option<f32>,
    pub face_budget_override: Option<Duration>,
    pub analysis_budget: Option<Duration>,
    pub gameplay_budget_override: Option<Duration>,
}

const DEFAULT_SAMPLE_COUNT: usize = 3;
const DEFAULT_SAMPLE_START: f32 = 1.0;
const DEFAULT_SAMPLE_STEP: f32 = 1.5;
const DEFAULT_FRAME_WIDTH: u32 = 960;
const DEFAULT_FRAME_HEIGHT: u32 = 540;
const DEFAULT_FACE_MODEL_PATH: &str = "models/face_detection_yunet_2023mar.onnx";
const DEFAULT_FACE_MESH_MODEL_PATH: &str = "models/face_mesh/face_mesh.onnx";
const DEFAULT_FACE_SCORE: f32 = 0.6;
const DEFAULT_FACE_TILE_MIN_SCORE: f32 = 0.60;
const DEFAULT_FACE_TILE_MAX_DEPTH: usize = 3;
const DEFAULT_FACE_PICK_RAW: bool = true;
const DEFAULT_FACE_TRACK_STEP_SECS: f32 = 2.0;
const DEFAULT_GAMEPLAY_BUDGET_MIN_SECS: f32 = 2.0;
const MAX_FULL_SAMPLES: usize = 60;
const MAX_FACE_CANDIDATES: usize = 24;
const FACE_EDGE_MARGIN: f32 = 0.02;
const MAX_FACE_DRIFT: f32 = 0.18;
const FACE_MIN_PX: f32 = 10.0;
const FACE_MAX_PX: f32 = 300.0;
const FACE_MIN_AREA_TILED_RELAXED: f32 = 0.00025;
const FACE_LANDMARK_REQUIRED: bool = true;
const FACE_LANDMARK_MAX_EYE_TILT: f32 = 0.35;
const FACE_LANDMARK_MAX_MOUTH_TILT: f32 = 0.45;
const FACE_LANDMARK_NOSE_CENTER_MAX: f32 = 0.25;
const FACE_LANDMARK_MOUTH_CENTER_MAX: f32 = 0.35;
const FACE_LANDMARK_MIN_EYE_RATIO: f32 = 0.18;
const FACE_LANDMARK_EYE_NOSE_MIN: f32 = 0.05;
const FACE_LANDMARK_NOSE_MOUTH_MIN: f32 = 0.06;
const FACE_SCAN_REGION_SIZES: [f32; 3] = [0.7, 0.5, 0.35];
const FACE_REGION_MARGIN: f32 = 0.02;
const FACE_FRAME_HEAD_TOP_DEFAULT: f32 = -0.28;
const FACE_FRAME_HEAD_TOP_MIN: f32 = -0.8;
const FACE_FRAME_HEAD_TOP_MAX: f32 = 0.2;
const FACE_FRAME_EYE_TOP_RATIO: f32 = 0.45;
const FACE_FRAME_EYE_CHIN_RATIO: f32 = 0.55;
const FACE_FRAME_SHOULDER_SCALE: f32 = 3.2;
const FACE_MESH_MODEL_MIN_MB_DEFAULT: u64 = 1;
const FACE_MESH_MODEL_MAX_MB_DEFAULT: u64 = 64;
const FACE_MESH_INPUT_SIZE_DEFAULT: u32 = 192;
const FACE_MESH_INPUT_MAX_DEFAULT: u32 = 512;
const FACE_MESH_LOAD_TIMEOUT_SECS_DEFAULT: u64 = 60;
const FACE_MESH_REGION_SCALE_DEFAULT: f32 = 1.35;
const FACE_MESH_HEADROOM_RATIO_DEFAULT: f32 = 0.12;
const FACE_MESH_MIN_POINTS: usize = 60;
const FACE_MESH_MAX_CANDIDATES: usize = 3;
const POSE_MODEL_MIN_MB_DEFAULT: u64 = 1;
const POSE_MODEL_MAX_MB_DEFAULT: u64 = 64;
const POSE_TRACT_OPT_DEFAULT: bool = false;
const POSE_INPUT_SIZE_DEFAULT: u32 = 256;
const POSE_INPUT_MAX_DEFAULT: u32 = 512;
const POSE_LOAD_TIMEOUT_SECS_DEFAULT: u64 = 60;

#[derive(Clone, Debug, PartialEq, Eq)]
struct FaceDetectorKey {
    model_path: Option<String>,
    backend: FaceBackend,
    frame_w: u32,
    frame_h: u32,
    score_threshold_bits: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PoseDetectorKey {
    enabled: bool,
    model_path: Option<String>,
    model_exists: bool,
    backend: FaceBackend,
    input_size: u32,
    input_max: u32,
    input_scale_bits: u32,
    model_min_mb: u64,
    model_max_mb: u64,
    load_timeout_secs: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FaceMeshDetectorKey {
    enabled: bool,
    model_path: Option<String>,
    model_exists: bool,
    backend: FaceBackend,
    input_size: u32,
    input_max: u32,
    input_scale_bits: u32,
    model_min_mb: u64,
    model_max_mb: u64,
    load_timeout_secs: u64,
}

struct CachedDetector<T, K> {
    key: K,
    detector: Option<Arc<T>>,
}

static FACE_DETECTOR_CACHE: OnceLock<Mutex<Option<CachedDetector<YunetDetector, FaceDetectorKey>>>> =
    OnceLock::new();
static POSE_DETECTOR_CACHE: OnceLock<Mutex<Option<CachedDetector<PoseDetector, PoseDetectorKey>>>> =
    OnceLock::new();
static FACE_MESH_DETECTOR_CACHE: OnceLock<
    Mutex<Option<CachedDetector<FaceMeshDetector, FaceMeshDetectorKey>>>,
> = OnceLock::new();

#[cfg(feature = "ort")]
static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();

fn read_env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

fn read_env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn read_env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
}

fn pose_load_timeout() -> Duration {
    let secs = read_env_u64("CLIP_POSE_LOAD_TIMEOUT_SECS", POSE_LOAD_TIMEOUT_SECS_DEFAULT);
    let secs = if secs == 0 { POSE_LOAD_TIMEOUT_SECS_DEFAULT } else { secs.min(POSE_LOAD_TIMEOUT_SECS_DEFAULT) };
    Duration::from_secs(secs)
}

fn face_mesh_load_timeout() -> Duration {
    let secs = read_env_u64(
        "CLIP_FACE_MESH_LOAD_TIMEOUT_SECS",
        FACE_MESH_LOAD_TIMEOUT_SECS_DEFAULT,
    );
    let secs = if secs == 0 {
        FACE_MESH_LOAD_TIMEOUT_SECS_DEFAULT
    } else {
        secs.min(FACE_MESH_LOAD_TIMEOUT_SECS_DEFAULT)
    };
    Duration::from_secs(secs)
}

fn face_frame_head_top_default() -> f32 {
    read_env_f32("CLIP_FACE_FRAME_HEAD_TOP", FACE_FRAME_HEAD_TOP_DEFAULT)
}

fn face_frame_head_top_min() -> f32 {
    read_env_f32("CLIP_FACE_FRAME_HEAD_TOP_MIN", FACE_FRAME_HEAD_TOP_MIN)
}

fn face_frame_head_top_max() -> f32 {
    read_env_f32("CLIP_FACE_FRAME_HEAD_TOP_MAX", FACE_FRAME_HEAD_TOP_MAX)
}

fn face_frame_eye_top_ratio() -> f32 {
    read_env_f32("CLIP_FACE_FRAME_EYE_TOP_RATIO", FACE_FRAME_EYE_TOP_RATIO)
}

fn face_frame_eye_chin_ratio() -> f32 {
    let ratio = read_env_f32("CLIP_FACE_FRAME_EYE_CHIN_RATIO", FACE_FRAME_EYE_CHIN_RATIO);
    if ratio <= 0.0 { FACE_FRAME_EYE_CHIN_RATIO } else { ratio }
}

fn face_frame_shoulder_scale() -> f32 {
    let scale = read_env_f32("CLIP_FACE_FRAME_SHOULDER_SCALE", FACE_FRAME_SHOULDER_SCALE);
    if scale <= 0.0 { FACE_FRAME_SHOULDER_SCALE } else { scale }
}

fn face_mesh_enabled() -> bool {
    std::env::var("CLIP_FACE_MESH")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(true)
}

fn face_mesh_debug_enabled() -> bool {
    std::env::var("CLIP_FACE_MESH_DEBUG")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn face_mesh_headroom_ratio() -> f32 {
    std::env::var("CLIP_FACE_MESH_HEADROOM")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(FACE_MESH_HEADROOM_RATIO_DEFAULT)
}

fn face_mesh_region_scale() -> f32 {
    std::env::var("CLIP_FACE_MESH_REGION_SCALE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.max(1.0))
        .unwrap_or(FACE_MESH_REGION_SCALE_DEFAULT)
}

fn face_mesh_input_scale() -> f32 {
    std::env::var("CLIP_FACE_MESH_INPUT_SCALE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(0.003921569)
}

fn face_mesh_input_size() -> u32 {
    let size = read_env_u32("CLIP_FACE_MESH_INPUT_SIZE", FACE_MESH_INPUT_SIZE_DEFAULT);
    if size == 0 { FACE_MESH_INPUT_SIZE_DEFAULT } else { size }
}

fn face_mesh_input_max() -> u32 {
    let max = read_env_u32("CLIP_FACE_MESH_INPUT_MAX", FACE_MESH_INPUT_MAX_DEFAULT);
    let size = face_mesh_input_size();
    if max == 0 { size } else { max.max(size) }
}

fn pose_enabled() -> bool {
    std::env::var("CLIP_POSE")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(true)
}

fn pose_debug_enabled() -> bool {
    std::env::var("CLIP_POSE_DEBUG")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn pose_keypoint_min_score() -> f32 {
    std::env::var("CLIP_POSE_SCORE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(0.30)
}

fn pose_input_scale() -> f32 {
    std::env::var("CLIP_POSE_INPUT_SCALE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(0.003921569)
}

fn pose_input_size() -> u32 {
    let size = read_env_u32("CLIP_POSE_INPUT_SIZE", POSE_INPUT_SIZE_DEFAULT);
    if size == 0 { POSE_INPUT_SIZE_DEFAULT } else { size }
}

fn pose_input_max() -> u32 {
    let max = read_env_u32("CLIP_POSE_INPUT_MAX", POSE_INPUT_MAX_DEFAULT);
    let size = pose_input_size();
    if max == 0 { size } else { max.max(size) }
}

fn face_mesh_model_min_mb() -> u64 {
    read_env_u64("CLIP_FACE_MESH_MODEL_MIN_MB", FACE_MESH_MODEL_MIN_MB_DEFAULT)
}

fn face_mesh_model_max_mb() -> u64 {
    read_env_u64("CLIP_FACE_MESH_MODEL_MAX_MB", FACE_MESH_MODEL_MAX_MB_DEFAULT)
}

fn face_mesh_tract_opt_enabled() -> bool {
    std::env::var("CLIP_FACE_MESH_TRACT_OPT")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

#[cfg(feature = "ort")]
fn ort_dylib_path() -> Option<PathBuf> {
    let raw = std::env::var("CLIP_ORT_DYLIB")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .or_else(|| {
            std::env::var("ORT_DYLIB_PATH")
                .ok()
                .filter(|v| !v.trim().is_empty())
        })?;
    let path = PathBuf::from(raw.trim());
    if path.is_dir() {
        Some(path.join("onnxruntime.dll"))
    } else {
        Some(path)
    }
}

#[cfg(feature = "ort")]
fn ensure_ort_runtime_loaded() -> Result<()> {
    let result = ORT_INIT.get_or_init(|| {
        let path = ort_dylib_path().unwrap_or_else(|| PathBuf::from("onnxruntime.dll"));
        let builder = init_from(&path).map_err(|err| {
            format!(
                "failed to load ONNX Runtime dylib from {}: {err}",
                path.display()
            )
        })?;
        let _committed = builder.commit();
        Ok(())
    });
    match result {
        Ok(()) => Ok(()),
        Err(msg) => Err(anyhow::anyhow!(
            "{msg} (set CLIP_ORT_DYLIB or update onnxruntime.dll to >=1.23.x)"
        )),
    }
}

enum LoadResult<T> {
    Ok(T),
    Timeout,
    Err(anyhow::Error),
}

fn run_with_timeout<T, F>(timeout: Duration, f: F) -> LoadResult<T>
where
    T: Send + 'static,
    F: Send + 'static + FnOnce() -> Result<T>,
{
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(timeout) {
        Ok(res) => match res {
            Ok(value) => LoadResult::Ok(value),
            Err(err) => LoadResult::Err(err),
        },
        Err(mpsc::RecvTimeoutError::Timeout) => LoadResult::Timeout,
        Err(err) => LoadResult::Err(anyhow::anyhow!("pose load thread failed: {err}")),
    }
}

fn pose_model_min_mb() -> u64 {
    read_env_u64("CLIP_POSE_MODEL_MIN_MB", POSE_MODEL_MIN_MB_DEFAULT)
}

fn pose_model_max_mb() -> u64 {
    read_env_u64("CLIP_POSE_MODEL_MAX_MB", POSE_MODEL_MAX_MB_DEFAULT)
}

fn pose_tract_opt_enabled() -> bool {
    std::env::var("CLIP_POSE_TRACT_OPT")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(POSE_TRACT_OPT_DEFAULT)
}

fn pose_head_ratio() -> f32 {
    std::env::var("CLIP_POSE_HEAD_RATIO")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(0.60)
}

fn pose_shoulder_margin() -> f32 {
    std::env::var("CLIP_POSE_SHOULDER_MARGIN")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1.10)
}

#[cfg(feature = "ort")]
fn ort_min_free_vram_mb() -> u64 {
    std::env::var("CLIP_ORT_MIN_FREE_VRAM_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(512)
}

#[cfg(feature = "ort")]
fn ort_gpu_mem_limit_mb() -> u64 {
    read_env_u64("CLIP_ORT_GPU_MEM_LIMIT_MB", 0)
}

#[cfg(feature = "ort")]
fn ort_device_id(min_free_mb: u64, label: &str) -> Option<i32> {
    let raw = std::env::var("CLIP_ORT_DEVICE").unwrap_or_default();
    let value = raw.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("auto") {
        return gpu::pick_best_nvidia_device(min_free_mb, label).map(|v| v as i32);
    }
    if value.eq_ignore_ascii_case("cpu") || value.eq_ignore_ascii_case("none") {
        return None;
    }
    match value.parse::<i32>() {
        Ok(idx) => Some(idx),
        Err(_) => {
            eprintln!("clip detect: unknown CLIP_ORT_DEVICE={value}; using auto");
            gpu::pick_best_nvidia_device(min_free_mb, label).map(|v| v as i32)
        }
    }
}

fn face_mesh_model_path() -> Option<String> {
    if !face_mesh_enabled() {
        return None;
    }
    let path = std::env::var("CLIP_FACE_MESH_MODEL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_FACE_MESH_MODEL_PATH.to_string());
    let path_ref = Path::new(&path);
    if !path_ref.exists() {
        eprintln!(
            "clip detect: face mesh model missing at {}; skipping face mesh",
            path
        );
        return None;
    }
    if let Ok(meta) = fs::metadata(path_ref) {
        let bytes = meta.len();
        let min_bytes = face_mesh_model_min_mb().saturating_mul(1024 * 1024);
        let max_bytes = face_mesh_model_max_mb().saturating_mul(1024 * 1024);
        if bytes < min_bytes || (max_bytes > 0 && bytes > max_bytes) {
            let mb = bytes as f64 / (1024.0 * 1024.0);
            eprintln!(
                "clip detect: face mesh model size {:.1} MB out of range [{}, {}]; skipping face mesh",
                mb,
                face_mesh_model_min_mb(),
                face_mesh_model_max_mb()
            );
            return None;
        }
    }
    if let Ok(mut file) = fs::File::open(path_ref) {
        let mut header = [0u8; 4];
        if let Ok(read) = file.read(&mut header) {
            if read >= 1 && header[0] == b'<' {
                eprintln!("clip detect: face mesh model looks like text/HTML; skipping face mesh");
                return None;
            }
            if read >= 2 && header[0] == b'P' && header[1] == b'K' {
                eprintln!("clip detect: face mesh model appears to be a zip; skipping face mesh");
                return None;
            }
        }
    }
    Some(path)
}

fn pose_model_path() -> Option<String> {
    if !pose_enabled() {
        return None;
    }
    let path = std::env::var("CLIP_POSE_MODEL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "models/pose/movenet_singlepose_thunder.onnx".to_string());
    let path_ref = Path::new(&path);
    if !path_ref.exists() {
        eprintln!("clip detect: pose model missing at {}; skipping pose", path);
        return None;
    }
    if let Ok(meta) = fs::metadata(path_ref) {
        let bytes = meta.len();
        let min_bytes = pose_model_min_mb().saturating_mul(1024 * 1024);
        let max_bytes = pose_model_max_mb().saturating_mul(1024 * 1024);
        if bytes < min_bytes || (max_bytes > 0 && bytes > max_bytes) {
            let mb = bytes as f64 / (1024.0 * 1024.0);
            eprintln!(
                "clip detect: pose model size {:.1} MB out of range [{}, {}]; skipping pose",
                mb,
                pose_model_min_mb(),
                pose_model_max_mb()
            );
            return None;
        }
    }
    if let Ok(mut file) = fs::File::open(path_ref) {
        let mut header = [0u8; 4];
        if let Ok(read) = file.read(&mut header) {
            if read >= 1 && header[0] == b'<' {
                eprintln!("clip detect: pose model looks like text/HTML; skipping pose");
                return None;
            }
            if read >= 2 && header[0] == b'P' && header[1] == b'K' {
                eprintln!("clip detect: pose model appears to be a zip; skipping pose");
                return None;
            }
        }
    }
    Some(path)
}

fn face_mesh_backend() -> FaceBackend {
    match std::env::var("CLIP_FACE_MESH_BACKEND") {
        Ok(value) => match parse_face_backend(&value) {
            Some(backend) => backend,
            None => {
                eprintln!(
                    "clip detect: unknown CLIP_FACE_MESH_BACKEND={value}; using auto"
                );
                FaceBackend::Auto
            }
        },
        Err(_) => FaceBackend::Auto,
    }
}

fn pose_backend() -> FaceBackend {
    match std::env::var("CLIP_POSE_BACKEND") {
        Ok(value) => match parse_face_backend(&value) {
            Some(backend) => backend,
            None => {
                eprintln!(
                    "clip detect: unknown CLIP_POSE_BACKEND={value}; using auto"
                );
                FaceBackend::Auto
            }
        },
        Err(_) => FaceBackend::Auto,
    }
}

fn parse_face_backend(value: &str) -> Option<FaceBackend> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Some(FaceBackend::Auto),
        "tract" | "cpu" => Some(FaceBackend::Tract),
        "ort" | "onnxruntime" | "cuda" | "gpu" => Some(FaceBackend::Ort),
        _ => None,
    }
}

/// Read face detection configuration from the environment with defaults.
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

    let face_track_step_secs = std::env::var("CLIP_FACE_TRACK_STEP")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.clamp(0.2, 30.0))
        .or_else(|| {
            if track_face {
                Some(DEFAULT_FACE_TRACK_STEP_SECS)
            } else {
                None
            }
        });

    let scan_full_clip = std::env::var("CLIP_DETECT_FULL")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false);

    let analysis_budget = std::env::var("CLIP_DETECT_BUDGET_SECS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(Duration::from_secs_f32);
    let face_budget_override = std::env::var("CLIP_FACE_BUDGET_SECS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(Duration::from_secs_f32);
    let gameplay_budget_override = std::env::var("CLIP_GAMEPLAY_BUDGET_SECS")
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
        face_track_step_secs,
        face_budget_override,
        analysis_budget,
        gameplay_budget_override,
    }
}

fn f32_key(value: f32) -> u32 {
    value.to_bits()
}

fn pose_model_path_raw() -> Option<String> {
    if !pose_enabled() {
        return None;
    }
    let path = std::env::var("CLIP_POSE_MODEL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "models/pose/movenet_singlepose_thunder.onnx".to_string());
    Some(path)
}

fn face_mesh_model_path_raw() -> Option<String> {
    if !face_mesh_enabled() {
        return None;
    }
    let path = std::env::var("CLIP_FACE_MESH_MODEL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_FACE_MESH_MODEL_PATH.to_string());
    Some(path)
}

fn face_detector_key(config: &ClipDetectConfig) -> FaceDetectorKey {
    FaceDetectorKey {
        model_path: config.face_model_path.clone(),
        backend: config.face_backend,
        frame_w: config.frame_width,
        frame_h: config.frame_height,
        score_threshold_bits: f32_key(config.face_score_threshold),
    }
}

fn pose_detector_key() -> PoseDetectorKey {
    let model_path = pose_model_path_raw();
    let model_exists = model_path
        .as_deref()
        .map(|path| Path::new(path).exists())
        .unwrap_or(false);
    PoseDetectorKey {
        enabled: pose_enabled(),
        model_path,
        model_exists,
        backend: pose_backend(),
        input_size: pose_input_size(),
        input_max: pose_input_max(),
        input_scale_bits: f32_key(pose_input_scale()),
        model_min_mb: pose_model_min_mb(),
        model_max_mb: pose_model_max_mb(),
        load_timeout_secs: pose_load_timeout().as_secs(),
    }
}

fn face_mesh_detector_key() -> FaceMeshDetectorKey {
    let model_path = face_mesh_model_path_raw();
    let model_exists = model_path
        .as_deref()
        .map(|path| Path::new(path).exists())
        .unwrap_or(false);
    FaceMeshDetectorKey {
        enabled: face_mesh_enabled(),
        model_path,
        model_exists,
        backend: face_mesh_backend(),
        input_size: face_mesh_input_size(),
        input_max: face_mesh_input_max(),
        input_scale_bits: f32_key(face_mesh_input_scale()),
        model_min_mb: face_mesh_model_min_mb(),
        model_max_mb: face_mesh_model_max_mb(),
        load_timeout_secs: face_mesh_load_timeout().as_secs(),
    }
}

fn cached_detector<T, K, F>(
    cache: &OnceLock<Mutex<Option<CachedDetector<T, K>>>>,
    key: K,
    build: F,
) -> Result<Option<Arc<T>>>
where
    K: PartialEq,
    F: FnOnce() -> Result<Option<T>>,
{
    let cache = cache.get_or_init(|| Mutex::new(None));
    {
        let guard = cache.lock().map_err(|_| anyhow::anyhow!("model cache lock poisoned"))?;
        if let Some(entry) = guard.as_ref() {
            if entry.key == key {
                return Ok(entry.detector.clone());
            }
        }
    }
    let detector = build()?.map(Arc::new);
    let mut guard = cache.lock().map_err(|_| anyhow::anyhow!("model cache lock poisoned"))?;
    *guard = Some(CachedDetector {
        key,
        detector: detector.clone(),
    });
    Ok(detector)
}

fn cached_face_detector(config: &ClipDetectConfig) -> Result<Option<Arc<YunetDetector>>> {
    let key = face_detector_key(config);
    cached_detector(&FACE_DETECTOR_CACHE, key, || YunetDetector::new(config))
}

fn cached_pose_detector(config: &ClipDetectConfig) -> Result<Option<Arc<PoseDetector>>> {
    let key = pose_detector_key();
    cached_detector(&POSE_DETECTOR_CACHE, key, || PoseDetector::new(config))
}

fn cached_face_mesh_detector(config: &ClipDetectConfig) -> Result<Option<Arc<FaceMeshDetector>>> {
    let key = face_mesh_detector_key();
    cached_detector(&FACE_MESH_DETECTOR_CACHE, key, || FaceMeshDetector::new(config))
}

/// Summary statistics for a face-score sweep run.
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
    face_override: Option<Duration>,
    gameplay_override: Option<Duration>,
    gameplay_enabled: bool,
) -> (Option<Duration>, Option<Duration>) {
    if let Some(face_override) = face_override {
        if let Some(total) = budget {
            let mut face_budget = face_override.min(total);
            let gameplay_budget = total.checked_sub(face_budget);
            let mut gameplay_budget = gameplay_budget
                .filter(|v| v.as_secs_f32().is_finite() && v.as_secs_f32() > 0.0);
            if gameplay_enabled && gameplay_budget.is_none() {
                let fallback = Duration::from_secs_f32(DEFAULT_GAMEPLAY_BUDGET_MIN_SECS);
                let reserve = gameplay_override.unwrap_or(fallback).min(total);
                if reserve.as_secs_f32() > 0.05 && total > reserve {
                    face_budget = total.saturating_sub(reserve);
                    gameplay_budget = Some(reserve);
                }
            }
            return (Some(face_budget), gameplay_budget);
        }
        return (Some(face_override), None);
    }
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
    let mut gameplay_budget = budget
        .checked_sub(face_budget)
        .filter(|remaining| remaining.as_secs_f32() > 0.05);
    if gameplay_enabled {
        if let Some(override_budget) = gameplay_override {
            let reserve = override_budget.min(budget);
            if reserve.as_secs_f32() > 0.05 && budget > reserve {
                gameplay_budget = Some(reserve);
                return (Some(budget.saturating_sub(reserve)), gameplay_budget);
            }
        }
    }
    (Some(face_budget), gameplay_budget)
}

fn gameplay_budget_with_fallback(
    analysis_budget: Option<Duration>,
    gameplay_budget: Option<Duration>,
) -> Option<Duration> {
    let has_budget = analysis_budget
        .filter(|budget| budget.as_secs_f32() > 0.05)
        .is_some();
    if has_budget {
        gameplay_budget.or(Some(Duration::from_secs_f32(
            DEFAULT_GAMEPLAY_BUDGET_MIN_SECS,
        )))
    } else {
        gameplay_budget
    }
}

/// Run a sweep over face-score thresholds to evaluate model behavior.
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

    let detector = match cached_face_detector(&sweep_config)? {
        Some(detector) => detector,
        None => anyhow::bail!("face model not available for sweep"),
    };

    eprintln!(
        "face sweep: loading samples (positives={}, negatives={})",
        positives.len(),
        negatives.len()
    );
    let pos_inputs = build_face_sweep_inputs(
        positives,
        &sweep_config,
        detector.as_ref(),
        "positives",
    )
    .await?;
    let neg_inputs = build_face_sweep_inputs(
        negatives,
        &sweep_config,
        detector.as_ref(),
        "negatives",
    )
    .await?;
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
                detector.as_ref(),
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
                detector.as_ref(),
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
        let mut face_frame_cache = FaceFrameCache::new(64);
        let source_dims = probe_media_dimensions(input).await;
        let sample_times = build_sample_times(config, true, None);
        let total_samples = sample_times.len();
        if total_samples == 0 {
            continue;
        }
        let mut face_samples: Vec<FaceSample> = Vec::new();
        for seek in sample_times {
            let seek_arg = if seek > 0.0 { Some(seek) } else { None };
            let candidates = detect_face_candidates(
                input,
                seek_arg,
                detector,
                source_dims,
                None,
                None,
                None,
                &mut face_frame_cache,
            )
            .await;
            if !candidates.is_empty() {
                face_samples.push(FaceSample {
                    time: seek,
                    candidates,
                    pose: None,
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
    let _span = profile_span("clip detect: layout hints");
    let mut hints = ClipLayoutHints::default();
    if !config.enabled {
        return Ok(hints);
    }

    let mut frame_cache = FrameCache::new(32);
    let mut face_frame_cache = FaceFrameCache::new(64);

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
    let face_sample_times = build_face_sample_times(config, is_local, duration_secs);
    let face_total_samples = face_sample_times.len();
    if face_total_samples > 0 {
        let mut min_t = f32::INFINITY;
        let mut max_t = f32::NEG_INFINITY;
        for t in &face_sample_times {
            min_t = min_t.min(*t);
            max_t = max_t.max(*t);
        }
        let first = if min_t.is_finite() { min_t } else { 0.0 };
        let last = if max_t.is_finite() { max_t } else { first };
        let step = config
            .face_track_step_secs
            .unwrap_or(config.sample_step_secs)
            .max(0.0);
        eprintln!(
            "clip detect: face sampling {} frame(s) from {:.2}s to {:.2}s (step {:.2}s, bisection)",
            face_total_samples,
            first,
            last,
            step
        );
    }
    let gameplay_detection = clip_region_detection_enabled();
    let gameplay_sample_times = if gameplay_detection {
        build_sample_times(config, is_local, duration_secs)
    } else {
        Vec::new()
    };
    let gameplay_total_samples = gameplay_sample_times.len();
    if gameplay_detection && gameplay_total_samples > 0 {
        let first = gameplay_sample_times.first().copied().unwrap_or(0.0);
        let last = gameplay_sample_times.last().copied().unwrap_or(first);
        eprintln!(
            "clip detect: gameplay sampling {} frame(s) from {:.2}s to {:.2}s (step {:.2}s)",
            gameplay_total_samples,
            first,
            last,
            config.sample_step_secs.max(0.0)
        );
    }
    let (face_budget, gameplay_budget) = split_analysis_budget(
        config.analysis_budget,
        config.face_budget_override,
        config.gameplay_budget_override,
        gameplay_detection,
    );
    let has_budget = config.analysis_budget.is_some();

    let mut face_samples: Vec<FaceSample> = Vec::new();
    let mut reticle_weight = 0.0f32;
    let mut reticle_sum_x = 0.0f32;
    let mut reticle_sum_y = 0.0f32;
    let mut gameplay_weight = 0.0f32;
    let mut gameplay_sum_x = 0.0f32;
    let mut gameplay_sum_y = 0.0f32;
    let mut gameplay_best: Option<ClipRegionObservation> = None;
    let mut face_best: Option<FaceConsensus> = None;
    let mut face_observations: Vec<FaceObservation> = Vec::new();
    let mut gameplay_config = None;
    let mut clip_detector: Option<Arc<ClipGameplayDetector>> = None;
    let mut gameplay_labels: Option<ClipLabelSet> = None;
    let mut cam_labels: Option<ClipLabelSet> = None;

    let face_detector = {
        let _span = profile_span("clip detect: init face detector");
        match cached_face_detector(config) {
            Ok(detector) => detector,
            Err(err) => {
                eprintln!("clip detect: failed to load face model: {err:#}");
                None
            }
        }
    };
    let face_id_matcher = FaceIdMatcher::from_env();
    let face_mesh_detector = {
        let _span = profile_span("clip detect: init face mesh detector");
        match cached_face_mesh_detector(config) {
            Ok(detector) => detector,
            Err(err) => {
                eprintln!("clip detect: failed to load face mesh model: {err:#}");
                None
            }
        }
    };
    let pose_detector = {
        let _span = profile_span("clip detect: init pose detector");
        match cached_pose_detector(config) {
            Ok(detector) => detector,
            Err(err) => {
                eprintln!("clip detect: failed to load pose model: {err:#}");
                None
            }
        }
    };

    if gameplay_detection {
        let cfg = read_clip_gameplay_config(config.frame_width, config.frame_height);
        gameplay_config = Some(cfg.clone());
        let detector = {
            let _span = profile_span("clip detect: init gameplay detector");
            match ClipGameplayDetector::get_cached(&cfg) {
                Ok(detector) => detector,
                Err(err) => {
                    eprintln!("clip detect: failed to load CLIP model: {err:#}");
                    None
                }
            }
        };
        if let Some(detector) = detector {
            let cam_score = cam_score_min(cfg.score_min);
            let cam_top_k = cam_top_k(cfg.top_k);
            let cam_labels_raw = cam_label_list();
            let cam_neg_labels_raw = cam_neg_label_list();
            if !cam_labels_raw.is_empty() && !cam_neg_labels_raw.is_empty() {
                match detector.encode_label_set(
                    &cam_labels_raw,
                    &cam_neg_labels_raw,
                    cam_score,
                    cam_top_k,
                ) {
                    Ok(labels) => cam_labels = Some(labels),
                    Err(err) => {
                        eprintln!("clip detect: failed to encode cam labels: {err:#}");
                    }
                }
            }
            gameplay_labels = Some(detector.label_set());
            clip_detector = Some(detector);
        }
    }

    if hints.face_region.is_none() {
        if let (Some(detector), Some(labels)) = (clip_detector.as_ref(), cam_labels.as_ref()) {
            let cam_seek = face_sample_times
                .first()
                .copied()
                .or_else(|| gameplay_sample_times.first().copied())
                .unwrap_or(config.sample_start_secs);
            let (cam_w, cam_h) = gameplay_config
                .as_ref()
                .map(|cfg| (cfg.frame_width, cfg.frame_height))
                .unwrap_or((config.frame_width, config.frame_height));
            let seek_arg = if cam_seek > 0.0 { Some(cam_seek) } else { None };
            match extract_frame_rgb_cached(
                Some(&mut frame_cache),
                input,
                seek_arg,
                cam_w,
                cam_h,
            )
            .await
            {
                Ok(frame) => {
                    if let Some(obs) = detector.detect_region(
                        frame.as_slice(),
                        cam_w,
                        cam_h,
                        None,
                        labels,
                    ) {
                        let region = expand_rect(obs.rect, cam_region_scale());
                        hints.face_region = Some(region);
                        eprintln!(
                            "clip detect: cam region score={:.4} x={:.3} y={:.3} w={:.3} h={:.3}",
                            obs.score, region.x, region.y, region.w, region.h
                        );
                    }
                }
                Err(err) => {
                    eprintln!("clip detect: cam frame extraction failed ({cam_seek:.2}s): {err:#}");
                }
            }
        }
    }

    if face_total_samples > 0 {
        let _span = profile_span("clip detect: face sampling");
        if let Some(detector) = face_detector.as_ref() {
            let face_start = Instant::now();
            let face_tick = Some(LoadingTicker::start(
                "clip detect: analyzing face samples",
                Duration::from_secs(5),
            ));
            let mut face_samples_attempted = 0usize;
            for (idx, seek) in face_sample_times.iter().copied().enumerate() {
                if let Some(budget) = face_budget {
                    let elapsed = face_start.elapsed();
                    if elapsed >= budget {
                        eprintln!(
                            "clip detect: face time budget {:.1}s hit after {}/{} sample(s); stopping early",
                            budget.as_secs_f32(),
                            face_samples_attempted,
                            face_total_samples
                        );
                        break;
                    }
                }
                eprintln!(
                    "clip detect: face sample {}/{} at {:.2}s",
                    idx + 1,
                    face_total_samples.max(1),
                    seek
                );
                let seek_arg = if seek > 0.0 { Some(seek) } else { None };
                let candidates = detect_face_candidates(
                    input,
                    seek_arg,
                    detector,
                    source_dims,
                    hints.face_region,
                    face_id_matcher.as_ref(),
                    face_mesh_detector.as_deref(),
                    &mut face_frame_cache,
                )
                .await;
                let pose = if !candidates.is_empty() {
                    if let Some(detector) = pose_detector.as_deref() {
                        detect_pose_observation(
                            input,
                            seek_arg,
                            detector,
                            source_dims,
                            &mut face_frame_cache,
                        )
                        .await
                    } else {
                        None
                    }
                } else {
                    None
                };
                face_samples_attempted += 1;
                if !candidates.is_empty() {
                    face_samples.push(FaceSample {
                        time: seek,
                        candidates,
                        pose,
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

    if let Some(detector) = face_detector.as_ref() {
        if let Some(result) = select_face_with_relaxation(
            &face_samples,
            face_total_samples,
            detector,
            config.face_score_threshold,
        ) {
            let accept_face =
                result.pass_label == "raw" || result.best.score >= config.face_score_threshold;
            if accept_face {
                if result.pass_label != "strict" {
                    eprintln!(
                        "clip detect: relaxing face filters -> {}",
                        result.pass_label
                    );
                }
                face_best = Some(result.best);
                face_observations = result.observations;
            } else {
                eprintln!(
                    "clip detect: best face score {:.4} below threshold {:.2}; ignoring",
                    result.best.score,
                    config.face_score_threshold
                );
            }
        }
    }
    if let (Some(matcher), Some(best)) = (face_id_matcher.as_ref(), face_best) {
        if matcher.require_motion && best.max_dist < matcher.motion_threshold {
            eprintln!(
                "face id: motion {:.4} below {:.4}; keeping face for framing but skipping face-id gating",
                best.max_dist, matcher.motion_threshold
            );
        }
    }
    if gameplay_detection && gameplay_total_samples > 0 {
        let _span = profile_span("clip detect: gameplay sampling");
        let gameplay_budget =
            gameplay_budget_with_fallback(config.analysis_budget, gameplay_budget);
        if has_budget && gameplay_budget.is_none() {
            eprintln!("clip detect: gameplay time budget 0.0s; skipping gameplay samples");
        } else {
            let gameplay_start = Instant::now();
            let gameplay_tick = Some(LoadingTicker::start(
                "clip detect: analyzing gameplay samples",
                Duration::from_secs(5),
            ));
            let mut gameplay_samples_attempted = 0usize;
            let use_reticle = clip_detector.is_none();
            let labels = gameplay_labels.as_ref();
            let cfg = gameplay_config.as_ref();
            let min_gameplay_samples = 3usize;
            for (idx, seek) in gameplay_sample_times.iter().copied().enumerate() {
                if let Some(budget) = gameplay_budget {
                    let elapsed = gameplay_start.elapsed();
                    if elapsed >= budget && gameplay_samples_attempted >= min_gameplay_samples {
                        eprintln!(
                            "clip detect: gameplay time budget {:.1}s hit after {}/{} sample(s); stopping early",
                            budget.as_secs_f32(),
                            gameplay_samples_attempted,
                            gameplay_total_samples
                        );
                        break;
                    }
                }
                eprintln!(
                    "clip detect: gameplay sample {}/{} at {:.2}s",
                    idx + 1,
                    gameplay_total_samples.max(1),
                    seek
                );
                let seek_arg = if seek > 0.0 { Some(seek) } else { None };
                let frame = match extract_frame_rgb_cached(
                    Some(&mut frame_cache),
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

                if use_reticle {
                    if let Some(reticle) =
                        detect_reticle_candidate(&frame, config.frame_width, config.frame_height)
                    {
                        reticle_sum_x += reticle.center.x * reticle.score;
                        reticle_sum_y += reticle.center.y * reticle.score;
                        reticle_weight += reticle.score;
                    }
                }

                if let (Some(detector), Some(labels), Some(cfg)) =
                    (clip_detector.as_ref(), labels, cfg)
                {
                    let exclude = hints
                        .face_region
                        .or(hints.face_box)
                        .map(|rect| expand_rect(rect, cam_region_scale()));
                    let gameplay_frame = if cfg.frame_width == config.frame_width
                        && cfg.frame_height == config.frame_height
                    {
                        None
                    } else {
                        match extract_frame_rgb_cached(
                            Some(&mut frame_cache),
                            input,
                            seek_arg,
                            cfg.frame_width,
                            cfg.frame_height,
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
                        (data.as_slice(), cfg.frame_width, cfg.frame_height)
                    } else {
                        (frame.as_slice(), config.frame_width, config.frame_height)
                    };

                    if let Some(obs) = detector.detect_region(rgb, gw, gh, exclude, labels) {
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

    let pose_available = face_samples.iter().any(|sample| sample.pose.is_some());
    if let Some(best) = face_best {
        hints.face_box = Some(best.rect);
        hints.face_frame_spec = best.frame_spec;
        if hints.face_focus.is_none() {
            hints.face_focus = best.focus;
        }
        if (face_debug_enabled() || face_mesh_debug_enabled()) && hints.face_focus.is_some() {
            if let Some(focus) = hints.face_focus {
                eprintln!(
                    "clip detect: face focus x={:.3} y={:.3}",
                    focus.x, focus.y
                );
            }
        }
        let region = pick_face_region(best.rect).or_else(|| {
            if best.region.w >= 0.25 && best.region.h >= 0.25 {
                Some(best.region)
            } else {
                None
            }
        });
        if hints.face_region.is_none() {
            if pose_available {
                hints.face_region = Some(NormalizedRect {
                    x: 0.0,
                    y: 0.0,
                    w: 1.0,
                    h: 1.0,
                });
                if pose_debug_enabled() || face_debug_enabled() {
                    eprintln!("clip detect: pose available; widening face region to full frame");
                }
            } else {
                let region = region.map(|region| {
                    let expanded = expand_rect_margins(region, 0.5, 0.5);
                    if face_debug_enabled() || pose_debug_enabled() {
                        eprintln!(
                            "clip detect: no pose; expanding face region with 0.5 margin"
                        );
                    }
                    expanded
                });
                hints.face_region = region;
            }
        }
    }

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
            } else if face_debug_enabled() {
                eprintln!(
                    "clip detect: face track has {}/2 points; using face box only",
                    points.len()
                );
            }
        }
    }

    if let Some(best) = face_best {
        eprintln!(
            "clip detect: face pick t={:.2}s score={:.4}",
            best.time, best.score
        );
    }
    if emotion_face_enabled() {
        if let Some(best) = face_best {
            let motion = best.max_dist;
            let threshold = emotion_face_motion_threshold();
            if motion >= threshold || emotion_debug_enabled() {
                eprintln!(
                    "clip detect: face emotion motion={:.4} (threshold {:.4})",
                    motion, threshold
                );
            }
        }
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
        if let Some(region) = hints.face_region {
            eprintln!(
                "clip detect: face region x={:.3} y={:.3} w={:.3} h={:.3}",
                region.x, region.y, region.w, region.h
            );
        }
    }
    if let Some(best) = gameplay_best {
        hints.game_region = Some(best.rect);
        log_gameplay_debug(&format!(
            "clip detect: gameplay pick score={:.4} center x={:.3} y={:.3} region x={:.3} y={:.3} w={:.3} h={:.3}",
            best.score,
            best.center.x,
            best.center.y,
            best.rect.x,
            best.rect.y,
            best.rect.w,
            best.rect.h
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

async fn enroll_face_id_from_input(
    input: &str,
    out_path: &Path,
    seeks: &[f32],
    strict: bool,
) -> Result<bool> {
    let cfg = read_clip_detect_config();
    let Some(detector) = cached_face_detector(&cfg)? else {
        anyhow::bail!("face detector unavailable; cannot enroll");
    };
    let face_cfg = face_id_config()
        .ok_or_else(|| anyhow::anyhow!("face id config disabled"))?;
    if !face_cfg.model_path.exists() {
        anyhow::bail!(
            "face id model not found at {}",
            face_cfg.model_path.display()
        );
    }
    let model = FaceIdModel::load(&face_cfg.model_path, face_cfg.bgr)?;
    let mut cache = FaceFrameCache::new(4);
    let source_dims = probe_media_dimensions(input).await;
    let full_region = NormalizedRect {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    };
    let mut last_err: Option<anyhow::Error> = None;
    for seek in seeks {
        let seek_arg = if *seek > 0.0 { Some(*seek) } else { None };
        let (frame, candidates) = match detect_faces_in_region_with_frame(
            input,
            seek_arg,
            detector.as_ref(),
            full_region,
            source_dims,
            &mut cache,
        )
        .await
        {
            Ok(value) => value,
            Err(err) => {
                if strict {
                    return Err(err);
                }
                last_err = Some(err);
                continue;
            }
        };
        let Some(best) = candidates
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
        else {
            continue;
        };
        let Some(embedding) = model.embed_from_face_frame(&frame, best.model_rect) else {
            continue;
        };
        write_face_id_embedding(out_path, &embedding)?;
        return Ok(true);
    }
    if let Some(err) = last_err {
        return Err(err);
    }
    Ok(false)
}

pub async fn enroll_face_id_from_image_url(
    image_url: &str,
    out_path: &Path,
) -> Result<bool> {
    enroll_face_id_from_input(image_url, out_path, &[0.0], true).await
}

pub async fn enroll_face_id_from_media_url(
    media_url: &str,
    out_path: &Path,
) -> Result<bool> {
    let seeks = [0.0, 2.0, 5.0];
    enroll_face_id_from_input(media_url, out_path, &seeks, false).await
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

fn pick_face_region(face_rect: NormalizedRect) -> Option<NormalizedRect> {
    let mut best: Option<NormalizedRect> = None;
    for region in build_face_scan_regions() {
        if region_is_full(region.rect) {
            continue;
        }
        if !rect_inside_region(face_rect, region.rect, FACE_REGION_MARGIN) {
            continue;
        }
        let area = region.rect.w * region.rect.h;
        let replace = match best {
            None => true,
            Some(current) => area > current.w * current.h,
        };
        if replace {
            best = Some(region.rect);
        }
    }
    best
}

async fn detect_face_candidates(
    input: &str,
    seek: Option<f32>,
    detector: &YunetDetector,
    source_dims: Option<(u32, u32)>,
    preferred_region: Option<NormalizedRect>,
    face_id: Option<&FaceIdMatcher>,
    face_mesh_detector: Option<&FaceMeshDetector>,
    frame_cache: &mut FaceFrameCache,
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
    if let Some(region) = preferred_region {
        if region.w > 0.0 && region.h > 0.0 {
            match detect_faces_in_region(
                input,
                seek,
                detector,
                region,
                source_dims,
                face_id,
                frame_cache,
            )
            .await
            {
                Ok(mut candidates) => {
                    if face_debug_enabled() {
                        eprintln!(
                            "clip detect: face scan cam -> {} candidates",
                            candidates.len()
                        );
                    }
                    all_candidates.append(&mut candidates);
                }
                Err(err) => {
                    eprintln!("clip detect: face frame extraction failed (cam): {err:#}");
                }
            }
            if dump_raw {
                match detect_faces_in_region_raw(
                    input,
                    seek,
                    detector,
                    region,
                    source_dims,
                    frame_cache,
                )
                .await
                {
                    Ok(raw_candidates) => {
                        maybe_dump_face_candidates(input, seek, &raw_candidates, "faces_cam_raw")
                            .await;
                    }
                    Err(err) => {
                        eprintln!("clip detect: face raw dump failed (cam): {err:#}");
                    }
                }
            }
        }
    }
    let mut full_candidates = match detect_faces_in_region(
        input,
        seek,
        detector,
        full_region,
        source_dims,
        face_id,
        frame_cache,
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
            frame_cache,
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
            face_id,
            frame_cache,
        )
        .await;
        maybe_dump_face_candidates(input, seek, &all_candidates, "faces").await;
        if let Some(detector) = face_mesh_detector {
            apply_face_mesh_to_candidates(
                input,
                seek,
                detector,
                source_dims,
                frame_cache,
                &mut all_candidates,
            )
            .await;
        }
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
        match detect_faces_in_region(
            input,
            seek,
            detector,
            region,
            source_dims,
            face_id,
            frame_cache,
        )
        .await
        {
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
        face_id,
        frame_cache,
    )
    .await;
    maybe_dump_face_candidates(input, seek, &all_candidates, "faces").await;
    if let Some(detector) = face_mesh_detector {
        apply_face_mesh_to_candidates(
            input,
            seek,
            detector,
            source_dims,
            frame_cache,
            &mut all_candidates,
        )
        .await;
    }
    all_candidates
}

async fn detect_pose_observation(
    input: &str,
    seek: Option<f32>,
    detector: &PoseDetector,
    source_dims: Option<(u32, u32)>,
    frame_cache: &mut FaceFrameCache,
) -> Option<PoseObservation> {
    let region = NormalizedRect {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    };
    let frame = extract_face_frame_rgb_cached(
        Some(frame_cache),
        input,
        seek,
        detector.input_w,
        detector.input_h,
        region,
        source_dims,
    )
    .await
    .ok()?;
    detector.detect_pose(&frame)
}

async fn detect_face_mesh_observation(
    input: &str,
    seek: Option<f32>,
    detector: &FaceMeshDetector,
    region: NormalizedRect,
    source_dims: Option<(u32, u32)>,
    frame_cache: &mut FaceFrameCache,
) -> Option<FaceMeshObservation> {
    let region = expand_rect(region, face_mesh_region_scale());
    let frame = extract_face_frame_rgb_cached(
        Some(frame_cache),
        input,
        seek,
        detector.input_w,
        detector.input_h,
        region,
        source_dims,
    )
    .await
    .ok()?;
    detector.detect_mesh(&frame)
}

async fn apply_face_mesh_to_candidates(
    input: &str,
    seek: Option<f32>,
    detector: &FaceMeshDetector,
    source_dims: Option<(u32, u32)>,
    frame_cache: &mut FaceFrameCache,
    candidates: &mut [FaceCandidate],
) {
    if candidates.is_empty() {
        return;
    }
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    indices.sort_by(|&a, &b| {
        candidates[b]
            .score
            .partial_cmp(&candidates[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let max_count = FACE_MESH_MAX_CANDIDATES.min(indices.len());
    for idx in indices.into_iter().take(max_count) {
        if candidates[idx].mesh.is_some() {
            continue;
        }
        let region = candidates[idx].rect;
        if let Some(obs) = detect_face_mesh_observation(
            input,
            seek,
            detector,
            region,
            source_dims,
            frame_cache,
        )
        .await
        {
            candidates[idx].mesh = Some(obs);
        } else if face_mesh_debug_enabled() {
            eprintln!("clip detect: face mesh failed for candidate idx={idx}");
        }
    }
}

async fn apply_face_tile_search(
    input: &str,
    seek: Option<f32>,
    detector: &YunetDetector,
    source_dims: Option<(u32, u32)>,
    mut candidates: Vec<FaceCandidate>,
    _face_id: Option<&FaceIdMatcher>,
    frame_cache: &mut FaceFrameCache,
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
                frame_cache,
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
    face_id: Option<&FaceIdMatcher>,
    frame_cache: &mut FaceFrameCache,
) -> Result<Vec<FaceCandidate>> {
    let frame = extract_face_frame_rgb_cached(
        Some(frame_cache),
        input,
        seek,
        detector.input_w,
        detector.input_h,
        region,
        source_dims,
    )
    .await?;
    let mut candidates = detector.detect_faces(frame.as_ref());
    if let Some(face_id) = face_id {
        candidates = filter_face_id_candidates(face_id, frame.as_ref(), candidates);
    }
    Ok(candidates)
}

async fn detect_faces_in_region_with_frame(
    input: &str,
    seek: Option<f32>,
    detector: &YunetDetector,
    region: NormalizedRect,
    source_dims: Option<(u32, u32)>,
    frame_cache: &mut FaceFrameCache,
) -> Result<(Arc<FaceFrame>, Vec<FaceCandidate>)> {
    let frame = extract_face_frame_rgb_cached(
        Some(frame_cache),
        input,
        seek,
        detector.input_w,
        detector.input_h,
        region,
        source_dims,
    )
    .await?;
    let candidates = detector.detect_faces(frame.as_ref());
    Ok((frame, candidates))
}

async fn detect_faces_in_region_raw(
    input: &str,
    seek: Option<f32>,
    detector: &YunetDetector,
    region: NormalizedRect,
    source_dims: Option<(u32, u32)>,
    frame_cache: &mut FaceFrameCache,
) -> Result<Vec<FaceCandidate>> {
    let frame = extract_face_frame_rgb_cached(
        Some(frame_cache),
        input,
        seek,
        detector.input_w,
        detector.input_h,
        region,
        source_dims,
    )
    .await?;
    Ok(detector.detect_faces_raw(frame.as_ref()))
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
    if face_pick_raw_enabled() {
        let min_score = raw_pick_min_score(base_score);
        if let Some(result) = select_best_raw_candidate(samples, min_score) {
            return Some(result);
        }
    }
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
                let rect = recentered_rect_from_candidate(&best);
                observations.push(FaceObservation {
                    rect,
                    score: best.score,
                    time: sample.time,
                    region: best.region,
                    frame_spec: face_frame_spec_for_candidate(&best, sample.pose.as_ref()),
                    focus: face_focus_from_candidate(&best),
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

fn select_best_raw_candidate(
    samples: &[FaceSample],
    min_score: f32,
) -> Option<FaceSelectionResult> {
    if samples.is_empty() {
        return None;
    }
    let mut observations = Vec::new();
    for sample in samples {
        let mut best: Option<(f32, f32, f32, FaceCandidate)> = None;
        let has_landmarks = sample
            .candidates
            .iter()
            .any(|c| c.landmarks_ok && c.raw_score.is_finite() && c.raw_score >= min_score);
        let enforce_landmarks = has_landmarks && !low_resource_enabled();
        for candidate in &sample.candidates {
            if !candidate.raw_score.is_finite() || candidate.raw_score < min_score {
                continue;
            }
            if enforce_landmarks && !candidate.landmarks_ok {
                continue;
            }
            let weighted =
                face_candidate_weighted_score(candidate.model_rect, candidate.raw_score);
            if !weighted.is_finite() {
                continue;
            }
            let area = rect_area(candidate.rect);
            let replace = match best {
                None => true,
                Some((best_area, best_weighted, best_raw, _)) => {
                    area > best_area
                        || (area == best_area
                            && (weighted > best_weighted
                                || (weighted == best_weighted
                                    && candidate.raw_score > best_raw)))
                }
            };
            if replace {
                best = Some((area, weighted, candidate.raw_score, *candidate));
            }
        }
        if let Some((_area, _weighted, raw, best)) = best {
            let rect = recentered_rect_from_candidate(&best);
            observations.push(FaceObservation {
                rect,
                score: raw,
                time: sample.time,
                region: best.region,
                frame_spec: face_frame_spec_for_candidate(&best, sample.pose.as_ref()),
                focus: face_focus_from_candidate(&best),
            });
        }
    }
    if observations.is_empty() {
        return None;
    }
    let best = select_consensus_face(&observations)?;
    Some(FaceSelectionResult {
        best,
        observations,
        pass_label: "raw",
    })
}

fn select_best_candidate_for_pass(
    sample: &FaceSample,
    detector: &YunetDetector,
    pass: &FaceSelectionPass,
) -> Option<FaceCandidate> {
    let mut best: Option<(i32, f32, FaceCandidate)> = None;
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
        let area = rect_area(candidate.rect);
        let replace = match best {
            None => true,
            Some((best_priority, best_area, best_candidate)) => {
                priority > best_priority
                    || (priority == best_priority
                        && (area > best_area
                            || (area == best_area
                                && candidate.score > best_candidate.score)))
            }
        };
        if replace {
            best = Some((priority, area, *candidate));
        }
    }
    best.map(|(_, _, candidate)| candidate)
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
    let min_area = if !region_is_full(candidate.region) && !pass.require_landmarks {
        pass.min_area.min(FACE_MIN_AREA_TILED_RELAXED)
    } else {
        pass.min_area
    };
    if !area.is_finite() || area < min_area || area > pass.max_area {
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
    build_sample_times_with_step(
        config.sample_count,
        config.sample_start_secs,
        config.sample_step_secs,
        config.scan_full_clip,
        is_local,
        duration_secs,
    )
}

fn build_face_sample_times(
    config: &ClipDetectConfig,
    is_local: bool,
    duration_secs: Option<f32>,
) -> Vec<f32> {
    let step = config
        .face_track_step_secs
        .unwrap_or(config.sample_step_secs);
    if is_local && config.scan_full_clip {
        if let Some(duration) = duration_secs.filter(|v| v.is_finite() && *v > 0.0) {
            let start = config.sample_start_secs.max(0.0).min(duration);
            let end = if duration > 0.1 { duration - 0.05 } else { duration };
            let end = end.max(start);
            let span = end - start;
            if span <= 0.0 {
                return vec![start];
            }
            let (count, _) = compute_full_sample_count(start, end, step);
            return build_bisection_times(start, end, count.max(1));
        }
    }
    build_sample_times_with_step(
        config.sample_count,
        config.sample_start_secs,
        step,
        config.scan_full_clip,
        is_local,
        duration_secs,
    )
}

fn build_sample_times_with_step(
    sample_count: usize,
    sample_start_secs: f32,
    sample_step_secs: f32,
    scan_full_clip: bool,
    is_local: bool,
    duration_secs: Option<f32>,
) -> Vec<f32> {
    if !is_local {
        let seek = sample_start_secs.max(0.0);
        return vec![seek];
    }

    if scan_full_clip {
        if let Some(duration) = duration_secs.filter(|v| v.is_finite() && *v > 0.0) {
            let start = sample_start_secs.max(0.0).min(duration);
            let end = if duration > 0.1 { duration - 0.05 } else { duration };
            let end = end.max(start);
            let span = end - start;
            if span <= 0.0 {
                return vec![start];
            }

            let (count, step) = compute_full_sample_count(start, end, sample_step_secs);

            let mut times = Vec::with_capacity(count);
            for idx in 0..count {
                times.push(start + step * idx as f32);
            }
            return times;
        }
    }

    let sample_count = sample_count.max(1);
    let mut times = Vec::with_capacity(sample_count);
    let start = sample_start_secs.max(0.0);
    let step = sample_step_secs.max(0.01);
    for idx in 0..sample_count {
        times.push(start + step * idx as f32);
    }
    times
}

fn compute_full_sample_count(start: f32, end: f32, sample_step_secs: f32) -> (usize, f32) {
    let span = (end - start).max(0.0);
    let base_step = sample_step_secs.max(0.05);
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
    (count.max(1), step)
}

fn build_bisection_times(start: f32, end: f32, max_samples: usize) -> Vec<f32> {
    if max_samples == 0 {
        return Vec::new();
    }
    if (end - start).abs() < 0.001 {
        return vec![start];
    }

    #[derive(Clone, Copy, Debug)]
    struct Segment {
        start: f32,
        end: f32,
        span: f32,
    }

    impl Segment {
        fn new(start: f32, end: f32) -> Option<Self> {
            let span = end - start;
            if span > 0.001 {
                Some(Self { start, end, span })
            } else {
                None
            }
        }
    }

    impl PartialEq for Segment {
        fn eq(&self, other: &Self) -> bool {
            self.span == other.span
        }
    }
    impl Eq for Segment {}
    impl PartialOrd for Segment {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.span.partial_cmp(&other.span)
        }
    }
    impl Ord for Segment {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    let mut heap = std::collections::BinaryHeap::new();
    if let Some(seg) = Segment::new(start, end) {
        heap.push(seg);
    }
    let mut times = Vec::with_capacity(max_samples);
    let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let min_span = ((end - start) / max_samples.max(1) as f32 * 0.6).max(0.05);

    while times.len() < max_samples {
        let Some(seg) = heap.pop() else { break };
        let mid = (seg.start + seg.end) * 0.5;
        let mid_q = (mid * 1000.0).round() / 1000.0;
        let key = (mid_q * 1000.0).round() as i64;
        if seen.insert(key) {
            times.push(mid_q);
        }
        if seg.span <= min_span {
            continue;
        }
        if let Some(left) = Segment::new(seg.start, mid_q) {
            heap.push(left);
        }
        if let Some(right) = Segment::new(mid_q, seg.end) {
            heap.push(right);
        }
        if heap.is_empty() {
            break;
        }
    }

    if times.is_empty() {
        times.push(start);
    }
    times
}

fn face_debug_enabled() -> bool {
    std::env::var("CLIP_FACE_DEBUG")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn emotion_face_enabled() -> bool {
    std::env::var("CLIP_EMOTION_FACE")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn emotion_debug_enabled() -> bool {
    std::env::var("CLIP_EMOTION_DEBUG")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn emotion_face_motion_threshold() -> f32 {
    std::env::var("CLIP_EMOTION_FACE_MOTION")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.035)
}

#[derive(Clone, Debug)]
struct FaceIdConfig {
    model_path: PathBuf,
    file_path: PathBuf,
    threshold: f32,
    require_motion: bool,
    motion_threshold: f32,
    bgr: bool,
    debug: bool,
}

#[derive(Serialize, Deserialize)]
struct FaceIdEmbedding {
    embedding: Vec<f32>,
}

struct FaceIdModel {
    model: TypedRunnableModel<TypedModel>,
    input_w: u32,
    input_h: u32,
    layout: ModelLayout,
    bgr: bool,
}

struct FaceIdMatcher {
    model: FaceIdModel,
    embedding: Vec<f32>,
    threshold: f32,
    require_motion: bool,
    motion_threshold: f32,
    debug: bool,
}

fn face_id_enabled() -> bool {
    std::env::var("CLIP_FACE_ID")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn face_id_config() -> Option<FaceIdConfig> {
    if !face_id_enabled() {
        return None;
    }
    let model_path = std::env::var("CLIP_FACE_ID_MODEL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "models/face_id/arcface.onnx".to_string());
    let file_path = std::env::var("CLIP_FACE_ID_FILE")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "face_id/stream.json".to_string());
    let threshold = std::env::var("CLIP_FACE_ID_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(0.35);
    let require_motion = std::env::var("CLIP_FACE_ID_REQUIRE_MOTION")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(true);
    let motion_threshold = std::env::var("CLIP_FACE_ID_MOTION")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(0.015);
    let bgr = std::env::var("CLIP_FACE_ID_BGR")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(true);
    let debug = std::env::var("CLIP_FACE_ID_DEBUG")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false);

    Some(FaceIdConfig {
        model_path: PathBuf::from(model_path),
        file_path: PathBuf::from(file_path),
        threshold,
        require_motion,
        motion_threshold,
        bgr,
        debug,
    })
}

fn load_face_id_embedding(path: &Path) -> Result<Vec<f32>> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("reading face id embedding {}", path.display()))?;
    let parsed: FaceIdEmbedding = serde_json::from_str(&data)
        .context("parsing face id embedding json")?;
    Ok(normalize_embedding(&parsed.embedding))
}

fn write_face_id_embedding(path: &Path, embedding: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let payload = FaceIdEmbedding {
        embedding: embedding.to_vec(),
    };
    let json = serde_json::to_string_pretty(&payload)
        .context("serializing face id embedding")?;
    std::fs::write(path, json)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn resolve_face_id_input(model: &InferenceModel) -> Result<(u32, u32, ModelLayout)> {
    let fact = model.input_fact(0)?.clone();
    let shape = fact
        .shape
        .as_concrete_finite()?
        .ok_or_else(|| anyhow::anyhow!("face id model input shape is not concrete"))?;
    if shape.len() != 4 {
        anyhow::bail!("face id model input must be 4D");
    }
    if shape[1] == 3 {
        let h = shape[2] as u32;
        let w = shape[3] as u32;
        Ok((w, h, ModelLayout::Nchw))
    } else if shape[3] == 3 {
        let h = shape[1] as u32;
        let w = shape[2] as u32;
        Ok((w, h, ModelLayout::Nhwc))
    } else {
        anyhow::bail!("face id model input layout is not RGB");
    }
}

fn normalize_embedding(embedding: &[f32]) -> Vec<f32> {
    let mut sum = 0.0f32;
    for v in embedding {
        sum += v * v;
    }
    let norm = sum.sqrt().max(1e-6);
    embedding.iter().map(|v| v / norm).collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

impl FaceIdModel {
    fn load(path: &Path, bgr: bool) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("loading face id model at {}", path.display()))?;
        let (input_w, input_h, layout) = resolve_face_id_input(&model)?;
        let input_shape = match layout {
            ModelLayout::Nchw => tvec!(1, 3, input_h as usize, input_w as usize),
            ModelLayout::Nhwc => tvec!(1, input_h as usize, input_w as usize, 3),
        };
        let model = model
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self {
            model,
            input_w,
            input_h,
            layout,
            bgr,
        })
    }

    fn embed_from_face_frame(
        &self,
        frame: &FaceFrame,
        rect: NormalizedRect,
    ) -> Option<Vec<f32>> {
        let rect = expand_rect(rect, 1.2);
        let src_w = frame.width as usize;
        let src_h = frame.height as usize;
        if frame.rgb.len() < src_w * src_h * 3 {
            return None;
        }
        let x0 = (rect.x * src_w as f32).floor().clamp(0.0, (src_w - 1) as f32) as usize;
        let y0 = (rect.y * src_h as f32).floor().clamp(0.0, (src_h - 1) as f32) as usize;
        let x1 = ((rect.x + rect.w) * src_w as f32)
            .ceil()
            .clamp((x0 + 1) as f32, src_w as f32) as usize;
        let y1 = ((rect.y + rect.h) * src_h as f32)
            .ceil()
            .clamp((y0 + 1) as f32, src_h as f32) as usize;
        let crop_w = x1.saturating_sub(x0).max(1);
        let crop_h = y1.saturating_sub(y0).max(1);

        let mut input = vec![0.0f32; (self.input_w * self.input_h * 3) as usize];
        for oy in 0..self.input_h as usize {
            let fy = if self.input_h > 1 {
                oy as f32 / (self.input_h - 1) as f32
            } else {
                0.0
            };
            let sy = y0 as f32 + fy * (crop_h as f32 - 1.0);
            let sy0 = sy.floor().clamp(0.0, (src_h - 1) as f32) as usize;
            let sy1 = (sy0 + 1).min(src_h - 1);
            let wy = sy - sy0 as f32;
            for ox in 0..self.input_w as usize {
                let fx = if self.input_w > 1 {
                    ox as f32 / (self.input_w - 1) as f32
                } else {
                    0.0
                };
                let sx = x0 as f32 + fx * (crop_w as f32 - 1.0);
                let sx0 = sx.floor().clamp(0.0, (src_w - 1) as f32) as usize;
                let sx1 = (sx0 + 1).min(src_w - 1);
                let wx = sx - sx0 as f32;
                let idx00 = (sy0 * src_w + sx0) * 3;
                let idx01 = (sy0 * src_w + sx1) * 3;
                let idx10 = (sy1 * src_w + sx0) * 3;
                let idx11 = (sy1 * src_w + sx1) * 3;
                for c in 0..3 {
                    let v00 = frame.rgb[idx00 + c] as f32;
                    let v01 = frame.rgb[idx01 + c] as f32;
                    let v10 = frame.rgb[idx10 + c] as f32;
                    let v11 = frame.rgb[idx11 + c] as f32;
                    let v0 = v00 + (v01 - v00) * wx;
                    let v1 = v10 + (v11 - v10) * wx;
                    let v = v0 + (v1 - v0) * wy;
                    let channel = if self.bgr { 2 - c } else { c };
                    let out_idx = (oy * self.input_w as usize + ox) * 3 + channel;
                    input[out_idx] = (v - 127.5) / 128.0;
                }
            }
        }

        let tensor = match self.layout {
            ModelLayout::Nchw => {
                let mut chw = vec![0.0f32; (self.input_w * self.input_h * 3) as usize];
                for y in 0..self.input_h as usize {
                    for x in 0..self.input_w as usize {
                        let base = (y * self.input_w as usize + x) * 3;
                        for c in 0..3 {
                            let idx = c * (self.input_w * self.input_h) as usize
                                + y * self.input_w as usize
                                + x;
                            chw[idx] = input[base + c];
                        }
                    }
                }
                Tensor::from_shape(
                    &[1, 3, self.input_h as usize, self.input_w as usize],
                    &chw,
                )
                .ok()?
            }
            ModelLayout::Nhwc => Tensor::from_shape(
                &[1, self.input_h as usize, self.input_w as usize, 3],
                &input,
            )
            .ok()?,
        };

        let outputs = self.model.run(tvec!(tensor.into())).ok()?;
        let output = outputs.get(0)?;
        let view = output.to_array_view::<f32>().ok()?;
        let embedding: Vec<f32> = view.iter().copied().collect();
        Some(normalize_embedding(&embedding))
    }
}

impl FaceIdMatcher {
    fn from_env() -> Option<Self> {
        let cfg = face_id_config()?;
        if !cfg.model_path.exists() {
            eprintln!(
                "face id: model not found at {}; disabling",
                cfg.model_path.display()
            );
            return None;
        }
        if !cfg.file_path.exists() {
            eprintln!(
                "face id: embedding not found at {}; disabling",
                cfg.file_path.display()
            );
            return None;
        }
        let model = FaceIdModel::load(&cfg.model_path, cfg.bgr).ok()?;
        let embedding = load_face_id_embedding(&cfg.file_path).ok()?;
        Some(Self {
            model,
            embedding,
            threshold: cfg.threshold,
            require_motion: cfg.require_motion,
            motion_threshold: cfg.motion_threshold,
            debug: cfg.debug,
        })
    }

    fn matches_candidate(&self, frame: &FaceFrame, rect: NormalizedRect) -> Option<f32> {
        let embed = self.model.embed_from_face_frame(frame, rect)?;
        Some(cosine_similarity(&embed, &self.embedding))
    }
}

fn filter_face_id_candidates(
    matcher: &FaceIdMatcher,
    frame: &FaceFrame,
    candidates: Vec<FaceCandidate>,
) -> Vec<FaceCandidate> {
    let all_candidates = candidates;
    let mut kept = Vec::new();
    let mut best_sim: Option<f32> = None;
    for candidate in all_candidates.iter().copied() {
        let sim = match matcher.matches_candidate(frame, candidate.model_rect) {
            Some(v) => v,
            None => continue,
        };
        if matcher.debug {
            best_sim = Some(best_sim.map(|b| b.max(sim)).unwrap_or(sim));
        }
        if sim >= matcher.threshold {
            kept.push(candidate);
        }
    }
    if matcher.debug {
        if let Some(sim) = best_sim {
            eprintln!(
                "face id: best similarity {:.4} (threshold {:.4})",
                sim, matcher.threshold
            );
        } else {
            eprintln!("face id: no candidates to score");
        }
    }
    if kept.is_empty() {
        eprintln!("face id: no candidates matched; falling back to raw faces");
        all_candidates
    } else {
        kept
    }
}

fn face_dump_dir() -> Option<PathBuf> {
    std::env::var("CLIP_FACE_DUMP_DIR")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn face_dump_raw_enabled() -> bool {
    std::env::var("CLIP_FACE_DUMP_RAW")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

fn face_pick_raw_enabled() -> bool {
    std::env::var("CLIP_FACE_PICK_RAW")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(DEFAULT_FACE_PICK_RAW)
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

fn raw_pick_min_score(base_score: f32) -> f32 {
    if let Some(tile) = face_tile_config() {
        base_score.max(tile.min_score)
    } else {
        base_score
    }
}

fn clip_region_detection_enabled() -> bool {
    std::env::var("CLIP_REGION_DETECT")
        .ok()
        .and_then(|v| parse_bool(&v))
        .or_else(|| {
            std::env::var("CLIP_GAMEPLAY_DETECT")
                .ok()
                .and_then(|v| parse_bool(&v))
        })
        .unwrap_or(true)
}

fn cam_label_list() -> Vec<String> {
    parse_label_list(
        std::env::var("CLIP_CAM_LABELS")
            .ok()
            .unwrap_or_else(|| {
                "webcam overlay|streamer cam|facecam|person portrait|streamer".to_string()
            }),
    )
}

fn cam_neg_label_list() -> Vec<String> {
    parse_label_list(
        std::env::var("CLIP_CAM_NEG_LABELS")
            .ok()
            .unwrap_or_else(|| {
                "video game gameplay|game UI|game HUD|chat box|text overlay".to_string()
            }),
    )
}

fn cam_score_min(default_score: f32) -> f32 {
    std::env::var("CLIP_CAM_SCORE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(-1.0, 1.0))
        .unwrap_or(default_score)
}

fn cam_top_k(default_top_k: usize) -> usize {
    std::env::var("CLIP_CAM_TOPK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.clamp(1, 24))
        .unwrap_or(default_top_k)
}

fn cam_region_scale() -> f32 {
    std::env::var("CLIP_CAM_REGION_SCALE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(1.0, 3.0))
        .unwrap_or(1.6)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PoseInputSource {
    Model,
    Fallback,
    Clamped,
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

fn resolve_pose_input_from_dims(
    dims: &[Option<usize>],
    fallback_size: u32,
    max_side: u32,
) -> (u32, u32, ModelLayout, PoseInputSource) {
    let mut layout = ModelLayout::Nchw;
    let mut source = PoseInputSource::Fallback;
    let mut input_w = fallback_size;
    let mut input_h = fallback_size;
    let mut layout_known = false;

    if dims.len() == 4 {
        if dims.get(1).and_then(|d| *d) == Some(3) {
            layout = ModelLayout::Nchw;
            layout_known = true;
        } else if dims.get(3).and_then(|d| *d) == Some(3) {
            layout = ModelLayout::Nhwc;
            layout_known = true;
        }
    }

    if layout_known && dims.len() == 4 {
        let (input_h_opt, input_w_opt) = match layout {
            ModelLayout::Nchw => (dims.get(2).and_then(|d| *d), dims.get(3).and_then(|d| *d)),
            ModelLayout::Nhwc => (dims.get(1).and_then(|d| *d), dims.get(2).and_then(|d| *d)),
        };
        if let (Some(input_h_raw), Some(input_w_raw)) = (input_h_opt, input_w_opt) {
            if let (Ok(input_h_val), Ok(input_w_val)) = (
                u32::try_from(input_h_raw),
                u32::try_from(input_w_raw),
            ) {
                if input_h_val > 0 && input_w_val > 0 {
                    input_h = input_h_val;
                    input_w = input_w_val;
                    source = PoseInputSource::Model;
                }
            }
        }
    }

    if input_w > max_side || input_h > max_side {
        input_w = fallback_size;
        input_h = fallback_size;
        source = PoseInputSource::Clamped;
    }

    (input_w, input_h, layout, source)
}

#[cfg(feature = "ort")]
fn resolve_model_input_from_ort(
    session: &Session,
    fallback_size: u32,
    max_side: u32,
) -> (u32, u32, ModelLayout, PoseInputSource) {
    let input = match session.inputs().first() {
        Some(val) => val,
        None => {
            return (
                fallback_size,
                fallback_size,
                ModelLayout::Nchw,
                PoseInputSource::Fallback,
            );
        }
    };
    let shape = match input.dtype().tensor_shape() {
        Some(val) => val,
        None => {
            return (
                fallback_size,
                fallback_size,
                ModelLayout::Nchw,
                PoseInputSource::Fallback,
            );
        }
    };
    let dims: Vec<Option<usize>> = shape
        .iter()
        .map(|d| if *d > 0 { usize::try_from(*d).ok() } else { None })
        .collect();
    resolve_pose_input_from_dims(&dims, fallback_size, max_side)
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
                    let session = build_ort_session(model_path, "face")?;
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
                    match build_ort_session(model_path, "face") {
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

        if face_pick_raw_enabled() {
            candidates.sort_by(|a, b| {
                b.raw_score
                    .partial_cmp(&a.raw_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            candidates.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
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
                    landmarks: None,
                    mesh: None,
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
                let mut landmarks: Option<[NormalizedPoint; 5]> = None;
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
                        landmarks = map_landmarks(mapping, points);
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
                        landmarks,
                        mesh: None,
                        region,
                    });
                }
            }
        }
    }
}

#[cfg(feature = "ort")]
fn build_ort_session(model_path: &str, label: &str) -> Result<Session> {
    ensure_ort_runtime_loaded()?;
    let min_free = ort_min_free_vram_mb();
    let device = ort_device_id(min_free, label);
    let allow_gpu = match device {
        Some(dev) => gpu::gpu_vram_allows(min_free, Some(dev as u32), label),
        None => false,
    };
    let mut providers = Vec::new();
    if allow_gpu {
        let mut cuda = ep::CUDA::default()
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic)
            .with_conv_max_workspace(false);
        if let Some(dev) = device {
            cuda = cuda.with_device_id(dev);
        }
        let mem_limit = ort_gpu_mem_limit_mb();
        if mem_limit > 0 {
            cuda = cuda.with_memory_limit(mem_limit.saturating_mul(1024 * 1024) as usize);
        }
        providers.push(cuda.build());
        if let Some(dev) = device {
            eprintln!("clip detect: ORT using CUDA device {dev} for {label}");
        }
    }
    providers.push(ep::CPU::default().build());
    let session = Session::builder()?
        .with_execution_providers(providers)?
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
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        let shape = ort_shape_to_usize(shape)?;
        return Tensor::from_shape(&shape, data).ok();
    }
    let (shape, data) = value.try_extract_tensor::<half::f16>().ok()?;
    let shape = ort_shape_to_usize(shape)?;
    let data: Vec<f32> = data.iter().map(|v| f32::from(*v)).collect();
    Tensor::from_shape(&shape, &data).ok()
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

fn rgb_to_rgb_chw_scaled(
    rgb: &[u8],
    width: usize,
    height: usize,
    scale: f32,
) -> Option<Vec<f32>> {
    let expected = width * height * 3;
    if rgb.len() < expected {
        return None;
    }
    let scale = if scale.is_finite() { scale } else { 1.0 };
    let mut out = vec![0f32; expected];
    let area = width * height;
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let r = rgb[idx] as f32 * scale;
            let g = rgb[idx + 1] as f32 * scale;
            let b = rgb[idx + 2] as f32 * scale;
            let offset = y * width + x;
            out[offset] = r;
            out[area + offset] = g;
            out[area * 2 + offset] = b;
        }
    }
    Some(out)
}

fn rgb_to_rgb_hwc_scaled(
    rgb: &[u8],
    width: usize,
    height: usize,
    scale: f32,
) -> Option<Vec<f32>> {
    let expected = width * height * 3;
    if rgb.len() < expected {
        return None;
    }
    let scale = if scale.is_finite() { scale } else { 1.0 };
    let mut out = vec![0f32; expected];
    for idx in (0..expected).step_by(3) {
        out[idx] = rgb[idx] as f32 * scale;
        out[idx + 1] = rgb[idx + 1] as f32 * scale;
        out[idx + 2] = rgb[idx + 2] as f32 * scale;
    }
    Some(out)
}

enum PoseBackend {
    Tract(TypedRunnableModel<TypedModel>),
    #[cfg(feature = "ort")]
    Ort(std::sync::Mutex<Session>),
}

struct PoseDetector {
    backend: PoseBackend,
    input_w: u32,
    input_h: u32,
    layout: ModelLayout,
    input_scale: f32,
}

impl PoseDetector {
    fn new(_config: &ClipDetectConfig) -> Result<Option<Self>> {
        let Some(model_path) = pose_model_path() else {
            return Ok(None);
        };
        let pose_start = Instant::now();
        let pose_timeout = pose_load_timeout();
        let pose_deadline = pose_start + pose_timeout;
        let pose_tick =
            LoadingTicker::start("clip detect: loading pose model", Duration::from_secs(5));
        let backend_choice = pose_backend();
        let timeout_secs = pose_timeout.as_secs_f32();

        #[cfg(feature = "ort")]
        if matches!(backend_choice, FaceBackend::Ort | FaceBackend::Auto) {
            let Some(remaining) = pose_deadline.checked_duration_since(Instant::now()) else {
                drop(pose_tick);
                eprintln!(
                    "clip detect: pose model load timed out after {:.1}s; skipping pose",
                    timeout_secs
                );
                return Ok(None);
            };
            let model_path_clone = model_path.clone();
            let fallback_size = pose_input_size();
            let max_side = pose_input_max();
            let ort_result = run_with_timeout(remaining, move || {
                let session = build_ort_session(&model_path_clone, "pose")?;
                let (input_w, input_h, layout, source) =
                    resolve_model_input_from_ort(&session, fallback_size, max_side);
                Ok((session, input_w, input_h, layout, source))
            });
            match ort_result {
                LoadResult::Ok((session, input_w, input_h, layout, source)) => {
                    drop(pose_tick);
                    eprintln!(
                        "clip detect: pose model loaded in {:.1}s",
                        pose_start.elapsed().as_secs_f32()
                    );
                    if matches!(source, PoseInputSource::Fallback) {
                        eprintln!(
                            "clip detect: pose input unresolved; using {}x{}",
                            input_w, input_h
                        );
                    } else if matches!(source, PoseInputSource::Clamped) {
                        eprintln!(
                            "clip detect: pose input exceeds max {}; using {}x{}",
                            max_side, input_w, input_h
                        );
                    }
                    eprintln!("clip detect: pose backend=ort");
                    return Ok(Some(Self {
                        backend: PoseBackend::Ort(std::sync::Mutex::new(session)),
                        input_w,
                        input_h,
                        layout,
                        input_scale: pose_input_scale(),
                    }));
                }
                LoadResult::Timeout => {
                    drop(pose_tick);
                    eprintln!(
                        "clip detect: pose model load timed out after {:.1}s; skipping pose",
                        timeout_secs
                    );
                    return Ok(None);
                }
                LoadResult::Err(err) => {
                    if matches!(backend_choice, FaceBackend::Ort) {
                        return Err(err);
                    }
                    eprintln!(
                        "clip detect: pose ORT init failed ({err:#}); falling back to tract"
                    );
                }
            }
        }

        #[cfg(not(feature = "ort"))]
        if matches!(backend_choice, FaceBackend::Ort) {
            anyhow::bail!(
                "pose backend 'ort' requested but autoclip was built without the ort feature"
            );
        }

        let Some(remaining) = pose_deadline.checked_duration_since(Instant::now()) else {
            drop(pose_tick);
            eprintln!(
                "clip detect: pose model load timed out after {:.1}s; skipping pose",
                timeout_secs
            );
            return Ok(None);
        };
        let fallback_size = pose_input_size();
        let max_side = pose_input_max();
        let model_path_clone = model_path.clone();
        let tract_result = run_with_timeout(remaining, move || {
            let model = tract_onnx::onnx()
                .model_for_path(&model_path_clone)
                .with_context(|| format!("loading pose model at {model_path_clone}"))?;
            let (input_w, input_h, layout, source) = match model.input_fact(0) {
                Ok(fact) => {
                    let dims: Vec<Option<usize>> = fact
                        .shape
                        .dims()
                        .map(|d| d.concretize().and_then(|d| d.to_usize().ok()))
                        .collect();
                    resolve_pose_input_from_dims(&dims, fallback_size, max_side)
                }
                Err(_) => (
                    fallback_size,
                    fallback_size,
                    ModelLayout::Nchw,
                    PoseInputSource::Fallback,
                ),
            };
            let input_shape = match layout {
                ModelLayout::Nchw => tvec!(1, 3, input_h as usize, input_w as usize),
                ModelLayout::Nhwc => tvec!(1, input_h as usize, input_w as usize, 3),
            };
            let model = model.with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), input_shape),
            )?;
            let model = if pose_tract_opt_enabled() {
                model.into_optimized()?
            } else {
                model.into_typed()?
            };
            let backend = PoseBackend::Tract(model.into_runnable()?);
            Ok((backend, input_w, input_h, layout, source))
        });
        match tract_result {
            LoadResult::Ok((backend, input_w, input_h, layout, source)) => {
                drop(pose_tick);
                eprintln!(
                    "clip detect: pose model loaded in {:.1}s",
                    pose_start.elapsed().as_secs_f32()
                );
                if matches!(source, PoseInputSource::Fallback) {
                    eprintln!(
                        "clip detect: pose input unresolved; using {}x{}",
                        input_w, input_h
                    );
                } else if matches!(source, PoseInputSource::Clamped) {
                    eprintln!(
                        "clip detect: pose input exceeds max {}; using {}x{}",
                        max_side, input_w, input_h
                    );
                }
                eprintln!("clip detect: pose backend=tract");
                Ok(Some(Self {
                    backend,
                    input_w,
                    input_h,
                    layout,
                    input_scale: pose_input_scale(),
                }))
            }
            LoadResult::Timeout => {
                drop(pose_tick);
                eprintln!(
                    "clip detect: pose model load timed out after {:.1}s; skipping pose",
                    timeout_secs
                );
                Ok(None)
            }
            LoadResult::Err(err) => {
                drop(pose_tick);
                Err(err)
            }
        }
    }

    fn detect_pose(&self, frame: &FaceFrame) -> Option<PoseObservation> {
        let rgb = &frame.rgb;
        let input = match self.layout {
            ModelLayout::Nchw => rgb_to_rgb_chw_scaled(
                rgb,
                self.input_w as usize,
                self.input_h as usize,
                self.input_scale,
            ),
            ModelLayout::Nhwc => rgb_to_rgb_hwc_scaled(
                rgb,
                self.input_w as usize,
                self.input_h as usize,
                self.input_scale,
            ),
        };
        let Some(input) = input else { return None };
        let outputs = match &self.backend {
            PoseBackend::Tract(model) => {
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
                let Some(tensor) = tensor else { return None };
                model.run(tvec!(tensor.into())).ok()
            }
            #[cfg(feature = "ort")]
            PoseBackend::Ort(session) => {
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
        }?;
        if outputs.is_empty() {
            return None;
        }
        let output = outputs.first()?;
        let mut keypoints = decode_movenet_keypoints(output)?;
        if let Some(mapping) = frame.mapping.as_ref() {
            if let Some(mapped) = map_pose_keypoints(mapping, &keypoints) {
                keypoints = mapped;
            }
        }
        let score = keypoints
            .iter()
            .map(|kp| if kp.score.is_finite() { kp.score.max(0.0) } else { 0.0 })
            .sum::<f32>()
            / POSE_KEYPOINT_COUNT.max(1) as f32;
        let observation = PoseObservation { keypoints, score };
        if pose_debug_enabled() {
            let eye = observation.keypoints[POSE_KP_LEFT_EYE];
            let shoulder = observation.keypoints[POSE_KP_LEFT_SHOULDER];
            eprintln!(
                "clip detect: pose score={:.3} eye=({:.3},{:.3}) shoulder=({:.3},{:.3})",
                observation.score, eye.x, eye.y, shoulder.x, shoulder.y
            );
        }
        Some(observation)
    }
}

enum FaceMeshBackend {
    Tract(TypedRunnableModel<TypedModel>),
    #[cfg(feature = "ort")]
    Ort(std::sync::Mutex<Session>),
}

struct FaceMeshDetector {
    backend: FaceMeshBackend,
    input_w: u32,
    input_h: u32,
    layout: ModelLayout,
    input_scale: f32,
}

impl FaceMeshDetector {
    fn new(_config: &ClipDetectConfig) -> Result<Option<Self>> {
        let Some(model_path) = face_mesh_model_path() else {
            return Ok(None);
        };
        let mesh_start = Instant::now();
        let mesh_timeout = face_mesh_load_timeout();
        let mesh_deadline = mesh_start + mesh_timeout;
        let mesh_tick = LoadingTicker::start(
            "clip detect: loading face mesh model",
            Duration::from_secs(5),
        );
        let backend_choice = face_mesh_backend();
        let timeout_secs = mesh_timeout.as_secs_f32();

        #[cfg(feature = "ort")]
        if matches!(backend_choice, FaceBackend::Ort | FaceBackend::Auto) {
            let Some(remaining) = mesh_deadline.checked_duration_since(Instant::now()) else {
                drop(mesh_tick);
                eprintln!(
                    "clip detect: face mesh model load timed out after {:.1}s; skipping face mesh",
                    timeout_secs
                );
                return Ok(None);
            };
            let model_path_clone = model_path.clone();
            let fallback_size = face_mesh_input_size();
            let max_side = face_mesh_input_max();
            let ort_result = run_with_timeout(remaining, move || {
                let session = build_ort_session(&model_path_clone, "face mesh")?;
                let (input_w, input_h, layout, source) =
                    resolve_model_input_from_ort(&session, fallback_size, max_side);
                Ok((session, input_w, input_h, layout, source))
            });
            match ort_result {
                LoadResult::Ok((session, input_w, input_h, layout, source)) => {
                    drop(mesh_tick);
                    eprintln!(
                        "clip detect: face mesh model loaded in {:.1}s",
                        mesh_start.elapsed().as_secs_f32()
                    );
                    if matches!(source, PoseInputSource::Fallback) {
                        eprintln!(
                            "clip detect: face mesh input unresolved; using {}x{}",
                            input_w, input_h
                        );
                    } else if matches!(source, PoseInputSource::Clamped) {
                        eprintln!(
                            "clip detect: face mesh input exceeds max {}; using {}x{}",
                            max_side, input_w, input_h
                        );
                    }
                    eprintln!("clip detect: face mesh backend=ort");
                    return Ok(Some(Self {
                        backend: FaceMeshBackend::Ort(std::sync::Mutex::new(session)),
                        input_w,
                        input_h,
                        layout,
                        input_scale: face_mesh_input_scale(),
                    }));
                }
                LoadResult::Timeout => {
                    drop(mesh_tick);
                    eprintln!(
                        "clip detect: face mesh model load timed out after {:.1}s; skipping face mesh",
                        timeout_secs
                    );
                    return Ok(None);
                }
                LoadResult::Err(err) => {
                    if matches!(backend_choice, FaceBackend::Ort) {
                        return Err(err);
                    }
                    eprintln!(
                        "clip detect: face mesh ORT init failed ({err:#}); falling back to tract"
                    );
                }
            }
        }

        #[cfg(not(feature = "ort"))]
        if matches!(backend_choice, FaceBackend::Ort) {
            anyhow::bail!(
                "face mesh backend 'ort' requested but autoclip was built without the ort feature"
            );
        }

        let Some(remaining) = mesh_deadline.checked_duration_since(Instant::now()) else {
            drop(mesh_tick);
            eprintln!(
                "clip detect: face mesh model load timed out after {:.1}s; skipping face mesh",
                timeout_secs
            );
            return Ok(None);
        };
        let fallback_size = face_mesh_input_size();
        let max_side = face_mesh_input_max();
        let model_path_clone = model_path.clone();
        let tract_result = run_with_timeout(remaining, move || {
            let model = tract_onnx::onnx()
                .model_for_path(&model_path_clone)
                .with_context(|| format!("loading face mesh model at {model_path_clone}"))?;
            let (input_w, input_h, layout, source) = match model.input_fact(0) {
                Ok(fact) => {
                    let dims: Vec<Option<usize>> = fact
                        .shape
                        .dims()
                        .map(|d| d.concretize().and_then(|d| d.to_usize().ok()))
                        .collect();
                    resolve_pose_input_from_dims(&dims, fallback_size, max_side)
                }
                Err(_) => (
                    fallback_size,
                    fallback_size,
                    ModelLayout::Nchw,
                    PoseInputSource::Fallback,
                ),
            };
            let input_shape = match layout {
                ModelLayout::Nchw => tvec!(1, 3, input_h as usize, input_w as usize),
                ModelLayout::Nhwc => tvec!(1, input_h as usize, input_w as usize, 3),
            };
            let model = model.with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), input_shape),
            )?;
            let model = if face_mesh_tract_opt_enabled() {
                model.into_optimized()?
            } else {
                model.into_typed()?
            };
            let backend = FaceMeshBackend::Tract(model.into_runnable()?);
            Ok((backend, input_w, input_h, layout, source))
        });
        match tract_result {
            LoadResult::Ok((backend, input_w, input_h, layout, source)) => {
                drop(mesh_tick);
                eprintln!(
                    "clip detect: face mesh model loaded in {:.1}s",
                    mesh_start.elapsed().as_secs_f32()
                );
                if matches!(source, PoseInputSource::Fallback) {
                    eprintln!(
                        "clip detect: face mesh input unresolved; using {}x{}",
                        input_w, input_h
                    );
                } else if matches!(source, PoseInputSource::Clamped) {
                    eprintln!(
                        "clip detect: face mesh input exceeds max {}; using {}x{}",
                        max_side, input_w, input_h
                    );
                }
                eprintln!("clip detect: face mesh backend=tract");
                Ok(Some(Self {
                    backend,
                    input_w,
                    input_h,
                    layout,
                    input_scale: face_mesh_input_scale(),
                }))
            }
            LoadResult::Timeout => {
                drop(mesh_tick);
                eprintln!(
                    "clip detect: face mesh model load timed out after {:.1}s; skipping face mesh",
                    timeout_secs
                );
                Ok(None)
            }
            LoadResult::Err(err) => {
                drop(mesh_tick);
                Err(err)
            }
        }
    }

    fn detect_mesh(&self, frame: &FaceFrame) -> Option<FaceMeshObservation> {
        let rgb = &frame.rgb;
        let input = match self.layout {
            ModelLayout::Nchw => rgb_to_rgb_chw_scaled(
                rgb,
                self.input_w as usize,
                self.input_h as usize,
                self.input_scale,
            ),
            ModelLayout::Nhwc => rgb_to_rgb_hwc_scaled(
                rgb,
                self.input_w as usize,
                self.input_h as usize,
                self.input_scale,
            ),
        };
        let Some(input) = input else { return None };
        let outputs = match &self.backend {
            FaceMeshBackend::Tract(model) => {
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
                let Some(tensor) = tensor else { return None };
                model.run(tvec!(tensor.into())).ok()
            }
            #[cfg(feature = "ort")]
            FaceMeshBackend::Ort(session) => {
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
        }?;
        if outputs.is_empty() {
            return None;
        }
        let output = select_face_mesh_output(&outputs)?;
        let (mut left_x, mut right_x, mut top_y, mut bottom_y) =
            decode_face_mesh_bounds(output, self.input_w, self.input_h)?;
        if let Some(mapping) = frame.mapping.as_ref() {
            let left = mapping.map_point(NormalizedPoint { x: left_x, y: 0.5 })?;
            let right = mapping.map_point(NormalizedPoint { x: right_x, y: 0.5 })?;
            let top = mapping.map_point(NormalizedPoint { x: 0.5, y: top_y })?;
            let bottom = mapping.map_point(NormalizedPoint { x: 0.5, y: bottom_y })?;
            left_x = left.x;
            right_x = right.x;
            top_y = top.y;
            bottom_y = bottom.y;
        }
        if !left_x.is_finite()
            || !right_x.is_finite()
            || !top_y.is_finite()
            || !bottom_y.is_finite()
            || right_x <= left_x
            || bottom_y <= top_y
        {
            return None;
        }
        let observation = FaceMeshObservation {
            left_x,
            right_x,
            top_y,
            bottom_y,
        };
        if face_mesh_debug_enabled() {
            eprintln!(
                "clip detect: face mesh bounds x={:.3}->{:.3} y={:.3}->{:.3}",
                observation.left_x,
                observation.right_x,
                observation.top_y,
                observation.bottom_y
            );
        }
        Some(observation)
    }
}

fn select_face_mesh_output(outputs: &[TValue]) -> Option<&TValue> {
    let mut best: Option<&TValue> = None;
    let mut best_len = 0usize;
    for output in outputs {
        let Ok(data) = output.as_slice::<f32>() else { continue; };
        let len = data.len();
        if len < FACE_MESH_MIN_POINTS.saturating_mul(3) || len % 3 != 0 {
            continue;
        }
        if len > best_len {
            best = Some(output);
            best_len = len;
        }
    }
    best
}

fn decode_face_mesh_bounds(
    output: &TValue,
    input_w: u32,
    input_h: u32,
) -> Option<(f32, f32, f32, f32)> {
    let data = output.as_slice::<f32>().ok()?;
    if data.len() < FACE_MESH_MIN_POINTS.saturating_mul(3) {
        return None;
    }
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for chunk in data.chunks_exact(3) {
        let x = chunk[0];
        let y = chunk[1];
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return None;
    }
    let max_abs = min_x
        .abs()
        .max(max_x.abs())
        .max(min_y.abs())
        .max(max_y.abs());
    if max_abs > 2.0 {
        let w = input_w as f32;
        let h = input_h as f32;
        if w > 0.0 {
            min_x /= w;
            max_x /= w;
        }
        if h > 0.0 {
            min_y /= h;
            max_y /= h;
        }
    } else if min_y < -0.5 || max_y > 1.5 || min_x < -0.5 || max_x > 1.5 {
        min_x = (min_x + 1.0) * 0.5;
        max_x = (max_x + 1.0) * 0.5;
        min_y = (min_y + 1.0) * 0.5;
        max_y = (max_y + 1.0) * 0.5;
    }
    min_x = clamp_unit(min_x);
    max_x = clamp_unit(max_x);
    min_y = clamp_unit(min_y);
    max_y = clamp_unit(max_y);
    if max_x <= min_x || max_y <= min_y {
        return None;
    }
    Some((min_x, max_x, min_y, max_y))
}

fn decode_movenet_keypoints(output: &Tensor) -> Option<[PoseKeypoint; POSE_KEYPOINT_COUNT]> {
    let data = output.as_slice::<f32>().ok()?;
    if data.len() < POSE_KEYPOINT_COUNT * 3 {
        return None;
    }
    let mut keypoints = [PoseKeypoint { x: 0.0, y: 0.0, score: 0.0 }; POSE_KEYPOINT_COUNT];
    let base = 0usize;
    for idx in 0..POSE_KEYPOINT_COUNT {
        let offset = base + idx * 3;
        let y = data[offset];
        let x = data[offset + 1];
        let score = data[offset + 2];
        let x = if x.is_finite() { clamp_unit(x) } else { 0.0 };
        let y = if y.is_finite() { clamp_unit(y) } else { 0.0 };
        let score = if score.is_finite() { score } else { 0.0 };
        keypoints[idx] = PoseKeypoint { x, y, score };
    }
    Some(keypoints)
}

fn map_pose_keypoints(
    mapping: &FrameMapping,
    keypoints: &[PoseKeypoint; POSE_KEYPOINT_COUNT],
) -> Option<[PoseKeypoint; POSE_KEYPOINT_COUNT]> {
    let mut out = [PoseKeypoint { x: 0.0, y: 0.0, score: 0.0 }; POSE_KEYPOINT_COUNT];
    for (idx, kp) in keypoints.iter().enumerate() {
        let mapped = mapping.map_point(NormalizedPoint { x: kp.x, y: kp.y })?;
        out[idx] = PoseKeypoint {
            x: mapped.x,
            y: mapped.y,
            score: kp.score,
        };
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

fn recenter_rect_x(rect: NormalizedRect, center_x: f32) -> NormalizedRect {
    let w = rect.w.clamp(0.0, 1.0);
    let h = rect.h.clamp(0.0, 1.0);
    let mut x = center_x - w / 2.0;
    if x < 0.0 {
        x = 0.0;
    }
    if x + w > 1.0 {
        x = 1.0 - w;
    }
    NormalizedRect {
        x: clamp_unit(x),
        y: rect.y.clamp(0.0, 1.0),
        w,
        h,
    }
}

fn recentered_rect_from_candidate(candidate: &FaceCandidate) -> NormalizedRect {
    if !low_resource_enabled() {
        return candidate.rect;
    }
    if let Some(mesh) = candidate.mesh {
        let center_x = (mesh.left_x + mesh.right_x) * 0.5;
        if center_x.is_finite() {
            let rect_center = rect_center(candidate.rect);
            let dx = (center_x - rect_center.x).abs();
            if dx > candidate.rect.w * 0.05 {
                return recenter_rect_x(candidate.rect, center_x);
            }
        }
    }
    let Some(points) = candidate.landmarks else {
        return candidate.rect;
    };
    if !landmarks_within_rect(&points, candidate.rect) {
        return candidate.rect;
    }
    let landmarks = classify_landmarks(&points);
    let mut center_x = landmarks
        .map(|landmarks| midpoint(landmarks.left_eye, landmarks.right_eye).x)
        .unwrap_or_else(|| points_center(&points).x);
    if !center_x.is_finite() {
        return candidate.rect;
    }
    if let Some(landmarks) = landmarks {
        let eye_mid = midpoint(landmarks.left_eye, landmarks.right_eye);
        let eye_dx = (landmarks.right_eye.x - landmarks.left_eye.x).abs().max(1e-4);
        let nose_bias = (landmarks.nose.x - eye_mid.x) / eye_dx;
        if nose_bias.is_finite() && nose_bias.abs() > 0.08 {
            let shift = (-nose_bias).clamp(-1.0, 1.0) * candidate.rect.w * 0.6;
            center_x = (center_x + shift).clamp(0.0, 1.0);
        }
    }
    let rect_center = rect_center(candidate.rect);
    let dx = (center_x - rect_center.x).abs();
    if dx <= candidate.rect.w * 0.05 {
        return candidate.rect;
    }
    recenter_rect_x(candidate.rect, center_x)
}

fn face_focus_from_candidate(candidate: &FaceCandidate) -> Option<NormalizedPoint> {
    if let Some(mesh) = candidate.mesh {
        let center_x = (mesh.left_x + mesh.right_x) * 0.5;
        let center_y = (mesh.top_y + mesh.bottom_y) * 0.5;
        if center_x.is_finite() && center_y.is_finite() {
            return Some(NormalizedPoint {
                x: clamp_unit(center_x),
                y: clamp_unit(center_y),
            });
        }
    }
    let points = candidate.landmarks?;
    if let Some(landmarks) = classify_landmarks(&points) {
        let eye_mid = midpoint(landmarks.left_eye, landmarks.right_eye);
        let mouth_mid = midpoint(landmarks.left_mouth, landmarks.right_mouth);
        let center_y = (eye_mid.y + mouth_mid.y) * 0.5;
        return Some(NormalizedPoint {
            x: clamp_unit(eye_mid.x),
            y: clamp_unit(center_y),
        });
    }
    let center = points_center(&points);
    if center.x.is_finite() && center.y.is_finite() {
        Some(NormalizedPoint {
            x: clamp_unit(center.x),
            y: clamp_unit(center.y),
        })
    } else {
        None
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

fn face_frame_spec_for_candidate(
    candidate: &FaceCandidate,
    pose: Option<&PoseObservation>,
) -> Option<FaceFrameSpec> {
    if candidate.mesh.is_some() {
        return Some(face_frame_spec_from_landmarks(candidate));
    }
    if let Some(pose) = pose {
        return pose_frame_spec_for_candidate(candidate, pose);
    }
    Some(face_frame_spec_from_landmarks(candidate))
}

fn face_frame_spec_from_landmarks(candidate: &FaceCandidate) -> FaceFrameSpec {
    let mut head_top_offset = face_frame_head_top_default();
    let min_offset = face_frame_head_top_min();
    let max_offset = face_frame_head_top_max();
    let (min_offset, max_offset) = if min_offset <= max_offset {
        (min_offset, max_offset)
    } else {
        (max_offset, min_offset)
    };
    if let Some(mesh) = candidate.mesh {
        let span = (mesh.bottom_y - mesh.top_y).max(1e-4);
        let head_top = mesh.top_y - span * face_mesh_headroom_ratio();
        let denom = candidate.rect.h.max(1e-4);
        let offset = (head_top - candidate.rect.y) / denom;
        if offset.is_finite() {
            head_top_offset = offset;
        }
    } else if candidate.landmarks_ok {
        if let Some(points) = candidate.landmarks {
            if let Some(landmarks) = classify_landmarks(&points) {
                let eye_mid = midpoint(landmarks.left_eye, landmarks.right_eye);
                let chin_y = candidate.rect.y + candidate.rect.h;
                let eye_to_chin = chin_y - eye_mid.y;
                if eye_to_chin.is_finite() && eye_to_chin > 1e-4 {
                    let eye_top_ratio = face_frame_eye_top_ratio();
                    let eye_chin_ratio = face_frame_eye_chin_ratio();
                    let head_height = eye_to_chin / eye_chin_ratio;
                    let head_top = eye_mid.y - eye_top_ratio * head_height;
                    let denom = candidate.rect.h.max(1e-4);
                    let offset = (head_top - candidate.rect.y) / denom;
                    if offset.is_finite() {
                        head_top_offset = offset;
                    }
                }
            }
        }
    } else if head_top_offset > min_offset {
        head_top_offset = min_offset;
    }
    head_top_offset = head_top_offset.clamp(min_offset, max_offset);
    FaceFrameSpec {
        head_top_offset,
        shoulder_width_scale: face_frame_shoulder_scale(),
    }
}

fn pose_frame_spec_for_candidate(
    candidate: &FaceCandidate,
    pose: &PoseObservation,
) -> Option<FaceFrameSpec> {
    let min_score = pose_keypoint_min_score();
    if !pose_matches_face(candidate.rect, pose, min_score) && pose_debug_enabled() {
        eprintln!("clip detect: pose does not match face; applying pose framing");
    }

    let mut head_top_offset = face_frame_head_top_default();
    let mut shoulder_width_scale = face_frame_shoulder_scale();

    if let Some((eye_mid, head_anchor, shoulder_mid)) =
        pose_head_and_shoulders(pose, min_score)
    {
        let head_span = shoulder_mid.y - eye_mid.y;
        if head_span.is_finite() && head_span > 1e-4 {
            let head_top = head_anchor.y - head_span * pose_head_ratio();
            let denom = candidate.rect.h.max(1e-4);
            let offset = (head_top - candidate.rect.y) / denom;
            if offset.is_finite() {
                head_top_offset = offset;
            }
        }
    }
    let min_offset = face_frame_head_top_min();
    let max_offset = face_frame_head_top_max();
    let (min_offset, max_offset) = if min_offset <= max_offset {
        (min_offset, max_offset)
    } else {
        (max_offset, min_offset)
    };
    head_top_offset = head_top_offset.clamp(min_offset, max_offset);

    if let Some(width) = pose_shoulder_width(pose, min_score) {
        let denom = candidate.rect.w.max(1e-4);
        let scale = (width * pose_shoulder_margin()) / denom;
        if scale.is_finite() && scale > 0.0 {
            shoulder_width_scale = scale.clamp(1.0, 12.0);
        }
    }

    let spec = FaceFrameSpec {
        head_top_offset,
        shoulder_width_scale,
    };
    if pose_debug_enabled() {
        eprintln!(
            "clip detect: pose frame spec head_top={:.3} shoulder_scale={:.2}",
            spec.head_top_offset, spec.shoulder_width_scale
        );
    }
    Some(spec)
}

fn pose_matches_face(
    rect: NormalizedRect,
    pose: &PoseObservation,
    min_score: f32,
) -> bool {
    let nose = pose_keypoint(pose, POSE_KP_NOSE, min_score);
    let eye = pose_midpoint(
        pose_keypoint(pose, POSE_KP_LEFT_EYE, min_score),
        pose_keypoint(pose, POSE_KP_RIGHT_EYE, min_score),
    );
    let point = nose.or(eye);
    let Some(point) = point else { return false };
    rect_contains_point(rect, point, 0.06)
}

fn pose_head_and_shoulders(
    pose: &PoseObservation,
    min_score: f32,
) -> Option<(NormalizedPoint, NormalizedPoint, NormalizedPoint)> {
    let left_eye = pose_keypoint(pose, POSE_KP_LEFT_EYE, min_score);
    let right_eye = pose_keypoint(pose, POSE_KP_RIGHT_EYE, min_score);
    let nose = pose_keypoint(pose, POSE_KP_NOSE, min_score);
    let eye_mid = pose_midpoint(left_eye, right_eye).or(nose);

    let mut head_candidates = Vec::new();
    if let Some(pt) = left_eye {
        head_candidates.push(pt);
    }
    if let Some(pt) = right_eye {
        head_candidates.push(pt);
    }
    if let Some(pt) = pose_keypoint(pose, POSE_KP_LEFT_EAR, min_score) {
        head_candidates.push(pt);
    }
    if let Some(pt) = pose_keypoint(pose, POSE_KP_RIGHT_EAR, min_score) {
        head_candidates.push(pt);
    }
    if let Some(pt) = nose {
        head_candidates.push(pt);
    }
    let head_anchor = head_candidates
        .into_iter()
        .min_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal));

    let shoulder_mid = pose_midpoint(
        pose_keypoint(pose, POSE_KP_LEFT_SHOULDER, min_score),
        pose_keypoint(pose, POSE_KP_RIGHT_SHOULDER, min_score),
    );

    match (eye_mid, head_anchor, shoulder_mid) {
        (Some(eye_mid), Some(head_anchor), Some(shoulder_mid)) => {
            Some((eye_mid, head_anchor, shoulder_mid))
        }
        _ => None,
    }
}

fn pose_shoulder_width(pose: &PoseObservation, min_score: f32) -> Option<f32> {
    let left = pose_keypoint(pose, POSE_KP_LEFT_SHOULDER, min_score)?;
    let right = pose_keypoint(pose, POSE_KP_RIGHT_SHOULDER, min_score)?;
    let width = (right.x - left.x).abs();
    if width.is_finite() && width > 0.0 {
        Some(width)
    } else {
        None
    }
}

fn pose_keypoint(
    pose: &PoseObservation,
    idx: usize,
    min_score: f32,
) -> Option<NormalizedPoint> {
    let kp = pose.keypoints.get(idx)?;
    if kp.score < min_score {
        return None;
    }
    if !(kp.x.is_finite() && kp.y.is_finite()) {
        return None;
    }
    Some(NormalizedPoint {
        x: clamp_unit(kp.x),
        y: clamp_unit(kp.y),
    })
}

fn pose_midpoint(
    left: Option<NormalizedPoint>,
    right: Option<NormalizedPoint>,
) -> Option<NormalizedPoint> {
    match (left, right) {
        (Some(left), Some(right)) => Some(midpoint(left, right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        _ => None,
    }
}

fn rect_contains_point(rect: NormalizedRect, point: NormalizedPoint, margin: f32) -> bool {
    let min_x = rect.x - margin;
    let max_x = rect.x + rect.w + margin;
    let min_y = rect.y - margin;
    let max_y = rect.y + rect.h + margin;
    point.x >= min_x && point.x <= max_x && point.y >= min_y && point.y <= max_y
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct FrameKey {
    seek_ms: i64,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct FaceFrameKey {
    seek_ms: i64,
    model_w: u32,
    model_h: u32,
    region_x: u16,
    region_y: u16,
    region_w: u16,
    region_h: u16,
    src_w: u32,
    src_h: u32,
}

struct FrameCache {
    max_entries: usize,
    order: VecDeque<FrameKey>,
    entries: HashMap<FrameKey, Arc<Vec<u8>>>,
}

impl FrameCache {
    fn new(max_entries: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    fn get(&self, key: &FrameKey) -> Option<Arc<Vec<u8>>> {
        self.entries.get(key).cloned()
    }

    fn insert(&mut self, key: FrameKey, value: Arc<Vec<u8>>) {
        if self.entries.contains_key(&key) {
            return;
        }
        self.entries.insert(key, value);
        self.order.push_back(key);
        self.evict_if_needed();
    }

    fn evict_if_needed(&mut self) {
        while self.entries.len() > self.max_entries {
            if let Some(key) = self.order.pop_front() {
                self.entries.remove(&key);
            } else {
                break;
            }
        }
    }
}

struct FaceFrameCache {
    max_entries: usize,
    order: VecDeque<FaceFrameKey>,
    entries: HashMap<FaceFrameKey, Arc<FaceFrame>>,
}

impl FaceFrameCache {
    fn new(max_entries: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    fn get(&self, key: &FaceFrameKey) -> Option<Arc<FaceFrame>> {
        self.entries.get(key).cloned()
    }

    fn insert(&mut self, key: FaceFrameKey, value: Arc<FaceFrame>) {
        if self.entries.contains_key(&key) {
            return;
        }
        self.entries.insert(key, value);
        self.order.push_back(key);
        self.evict_if_needed();
    }

    fn evict_if_needed(&mut self) {
        while self.entries.len() > self.max_entries {
            if let Some(key) = self.order.pop_front() {
                self.entries.remove(&key);
            } else {
                break;
            }
        }
    }
}

fn seek_to_ms(seek: Option<f32>) -> i64 {
    seek.map(|s| (s.max(0.0) * 1000.0).round() as i64)
        .unwrap_or(-1)
}

fn quantize_unit(value: f32) -> u16 {
    let clamped = value.clamp(0.0, 1.0);
    (clamped * 10_000.0).round().clamp(0.0, 10_000.0) as u16
}

fn face_frame_key(
    seek: Option<f32>,
    model_w: u32,
    model_h: u32,
    region: NormalizedRect,
    source_dims: Option<(u32, u32)>,
) -> FaceFrameKey {
    let (src_w, src_h) = source_dims.unwrap_or((0, 0));
    FaceFrameKey {
        seek_ms: seek_to_ms(seek),
        model_w,
        model_h,
        region_x: quantize_unit(region.x),
        region_y: quantize_unit(region.y),
        region_w: quantize_unit(region.w),
        region_h: quantize_unit(region.h),
        src_w,
        src_h,
    }
}

async fn extract_frame_rgb_cached(
    mut cache: Option<&mut FrameCache>,
    input: &str,
    seek_secs: Option<f32>,
    width: u32,
    height: u32,
) -> Result<Arc<Vec<u8>>> {
    let key = FrameKey {
        seek_ms: seek_to_ms(seek_secs),
        width,
        height,
    };
    if let Some(cache) = cache.as_deref_mut() {
        if let Some(frame) = cache.get(&key) {
            return Ok(frame);
        }
    }
    let data = Arc::new(extract_frame_rgb(input, seek_secs, width, height).await?);
    if let Some(cache) = cache.as_deref_mut() {
        cache.insert(key, data.clone());
    }
    Ok(data)
}

async fn run_ffmpeg_rgb_frame(
    input: &str,
    seek_secs: Option<f32>,
    filter: &str,
    seek_after_input: bool,
    context: &'static str,
) -> Result<Vec<u8>> {
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-hide_banner").arg("-loglevel").arg("error");
    cmd.arg("-nostdin");
    if !seek_after_input {
        if let Some(seek) = seek_secs {
            cmd.arg("-ss").arg(format!("{seek:.3}"));
        }
    }
    cmd.arg("-i").arg(input);
    if seek_after_input {
        if let Some(seek) = seek_secs {
            cmd.arg("-ss").arg(format!("{seek:.3}"));
        }
    }
    cmd.arg("-frames:v").arg("1");
    cmd.arg("-vf").arg(filter);
    cmd.arg("-pix_fmt").arg("rgb24");
    cmd.arg("-f").arg("rawvideo");
    cmd.arg("-");

    let output = cmd.output().await.context(context)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg frame extract failed: {}", stderr.trim());
    }
    Ok(output.stdout)
}

async fn extract_frame_rgb(
    input: &str,
    seek_secs: Option<f32>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let _span = profile_span("clip detect: extract frame");
    let filter = format!("scale={width}:{height}:flags=bicubic");
    let mut data = run_ffmpeg_rgb_frame(
        input,
        seek_secs,
        &filter,
        false,
        "running ffmpeg for detection frame",
    )
    .await?;
    let expected = (width * height * 3) as usize;
    if data.len() < expected {
        if seek_secs.is_some() {
            data = run_ffmpeg_rgb_frame(
                input,
                seek_secs,
                &filter,
                true,
                "running ffmpeg for detection frame (accurate seek)",
            )
            .await?;
        }
        if data.len() < expected {
            anyhow::bail!(
                "ffmpeg frame extract returned {} bytes, expected {}",
                data.len(),
                expected
            );
        }
    }
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

    fn map_point(&self, point: NormalizedPoint) -> Option<NormalizedPoint> {
        if self.scale <= 0.0
            || self.full_w <= 0.0
            || self.full_h <= 0.0
            || self.region_w <= 0.0
            || self.region_h <= 0.0
        {
            return None;
        }
        let x_model = point.x * self.model_w;
        let y_model = point.y * self.model_h;
        let x_region = (x_model - self.pad_x) / self.scale;
        let y_region = (y_model - self.pad_y) / self.scale;
        let x_src = x_region + self.region_x;
        let y_src = y_region + self.region_y;
        if !x_src.is_finite() || !y_src.is_finite() {
            return None;
        }
        let x = x_src.clamp(0.0, self.full_w);
        let y = y_src.clamp(0.0, self.full_h);
        Some(NormalizedPoint {
            x: clamp_unit(x / self.full_w),
            y: clamp_unit(y / self.full_h),
        })
    }
}

fn map_landmarks(
    mapping: Option<&FrameMapping>,
    points: [NormalizedPoint; 5],
) -> Option<[NormalizedPoint; 5]> {
    if let Some(mapper) = mapping {
        let mut out = [NormalizedPoint { x: 0.0, y: 0.0 }; 5];
        for (idx, point) in points.iter().copied().enumerate() {
            let mapped = mapper.map_point(point)?;
            out[idx] = mapped;
        }
        Some(out)
    } else {
        Some(points)
    }
}

#[derive(Clone, Debug)]
struct FaceFrame {
    rgb: Vec<u8>,
    mapping: Option<FrameMapping>,
    region: NormalizedRect,
    width: u32,
    height: u32,
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
    let _span = profile_span("clip detect: extract face frame");
    let filter = build_face_filter(model_w, model_h, region);
    let mut data = run_ffmpeg_rgb_frame(
        input,
        seek_secs,
        &filter,
        false,
        "running ffmpeg for face frame",
    )
    .await?;
    let expected = (model_w * model_h * 3) as usize;
    if data.len() < expected {
        if let Some(seek) = seek_secs {
            data = run_ffmpeg_rgb_frame(
                input,
                seek_secs,
                &filter,
                true,
                "running ffmpeg for face frame (accurate seek)",
            )
            .await?;
            if data.len() < expected && seek > 0.25 {
                let retry_seek = (seek - 0.25).max(0.0);
                data = run_ffmpeg_rgb_frame(
                    input,
                    Some(retry_seek),
                    &filter,
                    true,
                    "running ffmpeg for face frame (fallback seek)",
                )
                .await?;
            }
        }
        if data.len() < expected {
            anyhow::bail!(
                "ffmpeg face frame extract returned {} bytes, expected {}",
                data.len(),
                expected
            );
        }
    }
    data.truncate(expected);

    let mapping = source_dims.map(|(src_w, src_h)| {
        build_frame_mapping(src_w, src_h, model_w, model_h, region)
    });

    Ok(FaceFrame {
        rgb: data,
        mapping,
        region,
        width: model_w,
        height: model_h,
    })
}

async fn extract_face_frame_rgb_cached(
    mut cache: Option<&mut FaceFrameCache>,
    input: &str,
    seek_secs: Option<f32>,
    model_w: u32,
    model_h: u32,
    region: NormalizedRect,
    source_dims: Option<(u32, u32)>,
) -> Result<Arc<FaceFrame>> {
    let key = face_frame_key(seek_secs, model_w, model_h, region, source_dims);
    if let Some(cache) = cache.as_deref_mut() {
        if let Some(frame) = cache.get(&key) {
            return Ok(frame);
        }
    }
    let frame = Arc::new(
        extract_face_frame_rgb(input, seek_secs, model_w, model_h, region, source_dims).await?,
    );
    if let Some(cache) = cache.as_deref_mut() {
        cache.insert(key, frame.clone());
    }
    Ok(frame)
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
struct FaceMeshObservation {
    left_x: f32,
    right_x: f32,
    top_y: f32,
    bottom_y: f32,
}

#[derive(Clone, Copy, Debug)]
struct FaceCandidate {
    rect: NormalizedRect,
    model_rect: NormalizedRect,
    raw_score: f32,
    score: f32,
    landmarks_ok: bool,
    landmarks: Option<[NormalizedPoint; 5]>,
    mesh: Option<FaceMeshObservation>,
    region: NormalizedRect,
}

const POSE_KEYPOINT_COUNT: usize = 17;
const POSE_KP_NOSE: usize = 0;
const POSE_KP_LEFT_EYE: usize = 1;
const POSE_KP_RIGHT_EYE: usize = 2;
const POSE_KP_LEFT_EAR: usize = 3;
const POSE_KP_RIGHT_EAR: usize = 4;
const POSE_KP_LEFT_SHOULDER: usize = 5;
const POSE_KP_RIGHT_SHOULDER: usize = 6;

#[derive(Clone, Copy, Debug)]
struct PoseKeypoint {
    x: f32,
    y: f32,
    score: f32,
}

#[derive(Clone, Copy, Debug)]
struct PoseObservation {
    keypoints: [PoseKeypoint; POSE_KEYPOINT_COUNT],
    score: f32,
}

#[derive(Clone, Debug)]
struct FaceSample {
    time: f32,
    candidates: Vec<FaceCandidate>,
    pose: Option<PoseObservation>,
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
    region: NormalizedRect,
    frame_spec: Option<FaceFrameSpec>,
    focus: Option<NormalizedPoint>,
}

#[derive(Clone, Copy, Debug)]
struct FaceConsensus {
    rect: NormalizedRect,
    score: f32,
    time: f32,
    count: usize,
    max_dist: f32,
    region: NormalizedRect,
    frame_spec: Option<FaceFrameSpec>,
    focus: Option<NormalizedPoint>,
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

fn rect_area(rect: NormalizedRect) -> f32 {
    let area = rect.w * rect.h;
    if !area.is_finite() {
        return 0.0;
    }
    area.max(0.0)
}

fn active_face_motion_threshold() -> f32 {
    std::env::var("CLIP_FACE_ACTIVE_MOTION")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(0.015)
}

fn active_face_area_ratio() -> f32 {
    std::env::var("CLIP_FACE_ACTIVE_AREA_RATIO")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.clamp(0.05, 1.0))
        .unwrap_or(0.35)
}

fn expand_rect(rect: NormalizedRect, scale: f32) -> NormalizedRect {
    let scale = if scale.is_finite() { scale.max(1.0) } else { 1.0 };
    let center = rect_center(rect);
    let w = (rect.w * scale).clamp(0.0, 1.0);
    let h = (rect.h * scale).clamp(0.0, 1.0);
    recenter_rect(
        NormalizedRect {
            x: 0.0,
            y: 0.0,
            w,
            h,
        },
        center,
    )
}

fn expand_rect_margins(rect: NormalizedRect, x_margin: f32, y_margin: f32) -> NormalizedRect {
    let x_margin = if x_margin.is_finite() { x_margin.max(0.0) } else { 0.0 };
    let y_margin = if y_margin.is_finite() { y_margin.max(0.0) } else { 0.0 };
    let extra_w = rect.w * x_margin;
    let extra_h = rect.h * y_margin;
    let left = (rect.x - extra_w).max(0.0);
    let right = (rect.x + rect.w + extra_w).min(1.0);
    let top = (rect.y - extra_h).max(0.0);
    let bottom = (rect.y + rect.h + extra_h).min(1.0);
    NormalizedRect {
        x: left,
        y: top,
        w: (right - left).max(0.0),
        h: (bottom - top).max(0.0),
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

    let max_area = clusters
        .iter()
        .fold(0.0f32, |best, cluster| best.max(cluster.max_area));
    let mut best_active: Option<&FaceCluster> = None;
    let motion_threshold = active_face_motion_threshold();
    let area_ratio = active_face_area_ratio();
    let strong_motion = motion_threshold * 2.0;
    for cluster in &clusters {
        if cluster.max_dist < motion_threshold {
            continue;
        }
        if max_area > 0.0
            && cluster.max_area < max_area * area_ratio
            && cluster.max_dist < strong_motion
        {
            continue;
        }
        let take = match best_active {
            None => true,
            Some(best) => {
                cluster.max_dist > best.max_dist
                    || (cluster.max_dist == best.max_dist
                        && (cluster.max_area > best.max_area
                            || (cluster.max_area == best.max_area
                                && (cluster.count > best.count
                                    || (cluster.count == best.count
                                        && cluster.score_sum > best.score_sum)))))
            }
        };
        if take {
            best_active = Some(cluster);
        }
    }
    if let Some(cluster) = best_active {
        return Some(cluster.to_consensus());
    }

    let mut best_cluster: Option<&FaceCluster> = None;
    for cluster in &clusters {
        let take = match best_cluster {
            None => true,
            Some(best) => {
                cluster.max_area > best.max_area
                    || (cluster.max_area == best.max_area
                        && (cluster.count > best.count
                            || (cluster.count == best.count
                                && cluster.score_sum > best.score_sum)))
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
    max_area: f32,
    rect_sum_x: f32,
    rect_sum_y: f32,
    rect_sum_w: f32,
    rect_sum_h: f32,
    center_sum_x: f32,
    center_sum_y: f32,
    focus_sum_x: f32,
    focus_sum_y: f32,
    focus_weight: f32,
    dist_sum: f32,
    dist_count: usize,
    max_dist: f32,
    best: FaceObservation,
}

impl FaceCluster {
    fn new(obs: FaceObservation) -> Self {
        let center = rect_center(obs.rect);
        let weight = obs.score.max(0.0);
        let area = rect_area(obs.rect);
        let (focus_sum_x, focus_sum_y, focus_weight) = if weight > 0.0 {
            if let Some(focus) = obs.focus {
                (focus.x * weight, focus.y * weight, weight)
            } else {
                (0.0, 0.0, 0.0)
            }
        } else {
            (0.0, 0.0, 0.0)
        };
        Self {
            score_sum: weight,
            count: 1,
            max_area: area,
            rect_sum_x: obs.rect.x * weight,
            rect_sum_y: obs.rect.y * weight,
            rect_sum_w: obs.rect.w * weight,
            rect_sum_h: obs.rect.h * weight,
            center_sum_x: center.x * weight,
            center_sum_y: center.y * weight,
            focus_sum_x,
            focus_sum_y,
            focus_weight,
            dist_sum: 0.0,
            dist_count: 0,
            max_dist: 0.0,
            best: obs,
        }
    }

    fn add(&mut self, obs: FaceObservation) {
        let center_before = self.center();
        let area = rect_area(obs.rect);
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
            if let Some(focus) = obs.focus {
                self.focus_sum_x += focus.x * weight;
                self.focus_sum_y += focus.y * weight;
                self.focus_weight += weight;
            }
            let dist = (center.x - center_before.x)
                .abs()
                .max((center.y - center_before.y).abs());
            self.dist_sum += dist;
            self.dist_count += 1;
            if dist > self.max_dist {
                self.max_dist = dist;
            }
        }
        if area > self.max_area {
            self.max_area = area;
            self.best = obs;
        } else if area == self.max_area && obs.score > self.best.score {
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
        let focus = if self.focus_weight > 0.0 {
            Some(NormalizedPoint {
                x: clamp_unit(self.focus_sum_x / self.focus_weight),
                y: clamp_unit(self.focus_sum_y / self.focus_weight),
            })
        } else {
            self.best.focus
        };
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
                region: self.best.region,
                frame_spec: self.best.frame_spec,
                focus,
            }
        } else {
            FaceConsensus {
                rect: self.best.rect,
                score: self.best.score,
                time: self.best.time,
                count: self.count,
                max_dist: self.max_dist,
                region: self.best.region,
                frame_spec: self.best.frame_spec,
                focus,
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

    fn rect_at(cx: f32, cy: f32, w: f32, h: f32) -> NormalizedRect {
        NormalizedRect {
            x: cx - w / 2.0,
            y: cy - h / 2.0,
            w,
            h,
        }
    }

    fn obs_at(cx: f32, cy: f32, w: f32, h: f32, time: f32) -> FaceObservation {
        let rect = rect_at(cx, cy, w, h);
        FaceObservation {
            rect,
            score: 0.9,
            time,
            region: rect,
            frame_spec: Some(FaceFrameSpec {
                head_top_offset: 0.0,
                shoulder_width_scale: 1.0,
            }),
            focus: None,
        }
    }

    fn pose_obs(points: &[(usize, f32, f32, f32)]) -> PoseObservation {
        let mut keypoints = [PoseKeypoint { x: 0.0, y: 0.0, score: 0.0 }; POSE_KEYPOINT_COUNT];
        for (idx, x, y, score) in points {
            if *idx < keypoints.len() {
                keypoints[*idx] = PoseKeypoint {
                    x: *x,
                    y: *y,
                    score: *score,
                };
            }
        }
        let score = keypoints
            .iter()
            .map(|kp| kp.score.max(0.0))
            .sum::<f32>()
            / POSE_KEYPOINT_COUNT.max(1) as f32;
        PoseObservation { keypoints, score }
    }

    #[test]
    fn resolve_pose_input_prefers_model_dims() {
        let dims = vec![Some(1), Some(3), Some(256), Some(256)];
        let (w, h, layout, source) = resolve_pose_input_from_dims(&dims, 256, 512);
        assert_eq!((w, h), (256, 256));
        assert!(matches!(layout, ModelLayout::Nchw));
        assert_eq!(source, PoseInputSource::Model);
    }

    #[test]
    fn resolve_pose_input_clamps_large_dims() {
        let dims = vec![Some(1), Some(3), Some(1080), Some(1920)];
        let (w, h, _layout, source) = resolve_pose_input_from_dims(&dims, 256, 512);
        assert_eq!((w, h), (256, 256));
        assert_eq!(source, PoseInputSource::Clamped);
    }

    #[test]
    fn resolve_pose_input_falls_back_for_dynamic_dims() {
        let dims = vec![Some(1), None, None, Some(3)];
        let (w, h, layout, source) = resolve_pose_input_from_dims(&dims, 256, 512);
        assert_eq!((w, h), (256, 256));
        assert!(matches!(layout, ModelLayout::Nhwc));
        assert_eq!(source, PoseInputSource::Fallback);
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
            face_track_step_secs: None,
            face_budget_override: None,
            analysis_budget: None,
            gameplay_budget_override: None,
        };
        let times = build_sample_times(&config, true, Some(10.0));
        assert!(times.len() >= 2);
        let first = *times.first().unwrap();
        let last = *times.last().unwrap();
        assert!((first - 1.0).abs() < 1e-6);
        assert!(last > 7.0, "expected samples to reach near the clip end");
    }

    #[test]
    fn split_analysis_budget_reserves_gameplay_when_face_override_consumes_total() {
        let total = Duration::from_secs_f32(15.0);
        let (face_budget, gameplay_budget) =
            split_analysis_budget(Some(total), Some(total), None, true);
        let face_budget = face_budget.expect("expected face budget");
        let gameplay_budget = gameplay_budget.expect("expected gameplay budget");
        assert!(
            gameplay_budget.as_secs_f32() > 0.05,
            "gameplay budget should be reserved"
        );
        let summed = face_budget.as_secs_f32() + gameplay_budget.as_secs_f32();
        assert!(
            summed <= total.as_secs_f32() + 0.01,
            "face + gameplay should fit in total budget"
        );
    }

    #[test]
    fn gameplay_budget_fallback_applies_when_total_budget_present() {
        let total = Duration::from_secs_f32(15.0);
        let fallback = gameplay_budget_with_fallback(Some(total), None);
        assert_eq!(
            fallback,
            Some(Duration::from_secs_f32(DEFAULT_GAMEPLAY_BUDGET_MIN_SECS))
        );
    }

    #[test]
    fn gameplay_budget_fallback_respects_missing_budget() {
        let fallback = gameplay_budget_with_fallback(None, None);
        assert_eq!(fallback, None);
    }

    #[test]
    fn moving_face_beats_static_when_area_close() {
        let mut observations = Vec::new();
        observations.push(obs_at(0.8, 0.5, 0.5, 0.4, 1.0));
        observations.push(obs_at(0.8, 0.5, 0.5, 0.4, 2.0));
        observations.push(obs_at(0.8, 0.5, 0.5, 0.4, 3.0));

        observations.push(obs_at(0.2, 0.5, 0.4, 0.3, 1.0));
        observations.push(obs_at(0.25, 0.5, 0.4, 0.3, 2.0));
        observations.push(obs_at(0.3, 0.5, 0.4, 0.3, 3.0));

        let consensus = select_consensus_face(&observations).expect("expected consensus");
        let center = rect_center(consensus.rect);
        assert!(center.x < 0.5, "moving face should be preferred");
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

    #[test]
    fn pose_frame_spec_uses_shoulders() {
        let candidate = FaceCandidate {
            rect: NormalizedRect {
                x: 0.4,
                y: 0.2,
                w: 0.2,
                h: 0.4,
            },
            model_rect: NormalizedRect {
                x: 0.4,
                y: 0.2,
                w: 0.2,
                h: 0.4,
            },
            raw_score: 0.9,
            score: 0.9,
            landmarks_ok: false,
            landmarks: None,
            mesh: None,
            region: NormalizedRect {
                x: 0.0,
                y: 0.0,
                w: 1.0,
                h: 1.0,
            },
        };
        let pose = pose_obs(&[
            (POSE_KP_LEFT_EYE, 0.45, 0.25, 0.95),
            (POSE_KP_RIGHT_EYE, 0.55, 0.25, 0.95),
            (POSE_KP_NOSE, 0.50, 0.30, 0.90),
            (POSE_KP_LEFT_SHOULDER, 0.35, 0.60, 0.90),
            (POSE_KP_RIGHT_SHOULDER, 0.65, 0.60, 0.90),
        ]);
        let spec = pose_frame_spec_for_candidate(&candidate, &pose)
            .expect("expected pose-based frame spec");
        let expected_scale =
            (0.30 * pose_shoulder_margin()) / candidate.rect.w.max(1e-4);
        assert!((spec.shoulder_width_scale - expected_scale).abs() < 0.05);
        assert!(
            spec.head_top_offset < 0.0,
            "expected pose to push head top above face box"
        );
    }
}
