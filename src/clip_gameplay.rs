use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tract_onnx::prelude::*;

use crate::clip_layout::{NormalizedPoint, NormalizedRect};
use crate::loading::LoadingTicker;

const DEFAULT_MODEL_DIR: &str = "models/clip-vit-base-patch32-xenova";
const FALLBACK_MODEL_DIR: &str = "models/clip-vit-base-patch32";
const DEFAULT_STRIDE: u32 = 112;
const DEFAULT_TOP_K: usize = 6;
const DEFAULT_SCORE_MIN: f32 = 0.12;
const PATCH_SIZE: usize = 224;
const SEQ_LEN: usize = 77;

const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

#[derive(Clone, Debug)]
pub struct ClipGameplayConfig {
    pub enabled: bool,
    pub text_model_path: PathBuf,
    pub vision_model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub stride: u32,
    pub top_k: usize,
    pub score_min: f32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub positive_labels: Vec<String>,
    pub negative_labels: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct ClipRegionObservation {
    pub rect: NormalizedRect,
    pub center: NormalizedPoint,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct ClipLabelSet {
    pub positive: Vec<Vec<f32>>,
    pub negative: Vec<Vec<f32>>,
    pub score_min: f32,
    pub top_k: usize,
}

pub fn read_clip_gameplay_config(
    default_w: u32,
    default_h: u32,
) -> ClipGameplayConfig {
    let enabled = std::env::var("CLIP_GAMEPLAY")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(true);

    let model_dir = std::env::var("CLIP_GAMEPLAY_MODEL_DIR")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| {
            if PathBuf::from(DEFAULT_MODEL_DIR).exists() {
                DEFAULT_MODEL_DIR.to_string()
            } else {
                FALLBACK_MODEL_DIR.to_string()
            }
        });
    let model_dir = PathBuf::from(model_dir);
    let onnx_dir = model_dir.join("onnx");

    let text_model_path = std::env::var("CLIP_GAMEPLAY_TEXT_MODEL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            default_onnx_path(
                &onnx_dir,
                &["clip_text_fixed.onnx", "clip_text.onnx", "text_model.onnx"],
            )
        });
    let vision_model_path = std::env::var("CLIP_GAMEPLAY_VISION_MODEL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| default_onnx_path(&onnx_dir, &["clip_vision.onnx", "vision_model.onnx"]));
    let tokenizer_path = std::env::var("CLIP_GAMEPLAY_TOKENIZER")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| model_dir.join("tokenizer.json"));

    let stride = std::env::var("CLIP_GAMEPLAY_STRIDE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .map(|v| v.clamp(32, 224))
        .unwrap_or(DEFAULT_STRIDE);

    let top_k = std::env::var("CLIP_GAMEPLAY_TOPK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.clamp(1, 24))
        .unwrap_or(DEFAULT_TOP_K);

    let score_min = std::env::var("CLIP_GAMEPLAY_SCORE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(-1.0, 1.0))
        .unwrap_or(DEFAULT_SCORE_MIN);

    let (frame_width, frame_height) = std::env::var("CLIP_GAMEPLAY_SIZE")
        .ok()
        .and_then(|v| parse_size(&v))
        .unwrap_or((default_w.max(64), default_h.max(64)));

    let positive_labels = parse_label_list(
        std::env::var("CLIP_GAMEPLAY_LABELS")
            .ok()
            .unwrap_or_else(|| "video game gameplay|in-game scene|game HUD|game UI".to_string()),
    );
    let negative_labels = parse_label_list(
        std::env::var("CLIP_GAMEPLAY_NEG_LABELS")
            .ok()
            .unwrap_or_else(
                || "webcam|face|stream overlay|chat box|text overlay".to_string(),
            ),
    );

    ClipGameplayConfig {
        enabled,
        text_model_path,
        vision_model_path,
        tokenizer_path,
        stride,
        top_k,
        score_min,
        frame_width,
        frame_height,
        positive_labels,
        negative_labels,
    }
}

fn default_onnx_path(onnx_dir: &PathBuf, candidates: &[&str]) -> PathBuf {
    for name in candidates {
        let path = onnx_dir.join(name);
        if path.exists() {
            return path;
        }
    }
    onnx_dir.join(candidates[0])
}

pub struct ClipGameplayDetector {
    text_model: TypedRunnableModel<TypedModel>,
    vision_model: TypedRunnableModel<TypedModel>,
    tokenizer: Tokenizer,
    pad_id: u32,
    positive: Vec<Vec<f32>>,
    negative: Vec<Vec<f32>>,
    stride: u32,
    top_k: usize,
    score_min: f32,
}

impl ClipGameplayDetector {
    pub fn new(config: &ClipGameplayConfig) -> Result<Option<Self>> {
        if !config.enabled {
            return Ok(None);
        }
        if !config.text_model_path.exists() || !config.vision_model_path.exists() {
            return Ok(None);
        }
        if !config.tokenizer_path.exists() {
            return Ok(None);
        }

        let tokenizer_start = Instant::now();
        let tokenizer_tick =
            LoadingTicker::start("clip gameplay: loading tokenizer", Duration::from_secs(5));
        let tokenizer = Tokenizer::from_file(&config.tokenizer_path).map_err(|err| {
            anyhow::anyhow!(
                "loading tokenizer at {} failed: {err}",
                config.tokenizer_path.display()
            )
        })?;
        drop(tokenizer_tick);
        eprintln!(
            "clip gameplay: tokenizer loaded in {:.1}s",
            tokenizer_start.elapsed().as_secs_f32()
        );
        let pad_id = tokenizer
            .token_to_id("<|endoftext|>")
            .unwrap_or(0);

        let text_start = Instant::now();
        let text_tick =
            LoadingTicker::start("clip gameplay: loading text model", Duration::from_secs(5));
        let text_model = tract_onnx::onnx()
            .model_for_path(&config.text_model_path)
            .with_context(|| format!("loading text model at {}", config.text_model_path.display()))?;
        let text_model = apply_text_input_facts(text_model)?;
        let text_model = text_model.into_optimized()?.into_runnable()?;
        drop(text_tick);
        eprintln!(
            "clip gameplay: text model loaded in {:.1}s",
            text_start.elapsed().as_secs_f32()
        );

        let vision_start = Instant::now();
        let vision_tick =
            LoadingTicker::start("clip gameplay: loading vision model", Duration::from_secs(5));
        let vision_model = tract_onnx::onnx()
            .model_for_path(&config.vision_model_path)
            .with_context(|| {
                format!("loading vision model at {}", config.vision_model_path.display())
            })?;
        let vision_model = vision_model
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec!(1usize, 3usize, PATCH_SIZE, PATCH_SIZE),
                ),
            )?
            .into_optimized()?
            .into_runnable()?;
        drop(vision_tick);
        eprintln!(
            "clip gameplay: vision model loaded in {:.1}s",
            vision_start.elapsed().as_secs_f32()
        );

        let mut detector = Self {
            text_model,
            vision_model,
            tokenizer,
            pad_id,
            positive: Vec::new(),
            negative: Vec::new(),
            stride: config.stride,
            top_k: config.top_k,
            score_min: config.score_min,
        };

        detector.positive = detector.encode_prompts(&config.positive_labels)?;
        detector.negative = detector.encode_prompts(&config.negative_labels)?;

        Ok(Some(detector))
    }

    pub fn label_set(&self) -> ClipLabelSet {
        ClipLabelSet {
            positive: self.positive.clone(),
            negative: self.negative.clone(),
            score_min: self.score_min,
            top_k: self.top_k,
        }
    }

    pub fn encode_label_set(
        &self,
        positive_labels: &[String],
        negative_labels: &[String],
        score_min: f32,
        top_k: usize,
    ) -> Result<ClipLabelSet> {
        let positive = self.encode_prompts(positive_labels)?;
        let negative = self.encode_prompts(negative_labels)?;
        Ok(ClipLabelSet {
            positive,
            negative,
            score_min,
            top_k,
        })
    }

    pub fn detect_region(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
        exclude: Option<NormalizedRect>,
        labels: &ClipLabelSet,
    ) -> Option<ClipRegionObservation> {
        self.detect_region_with_labels(rgb, width, height, exclude, labels)
    }

    fn detect_region_with_labels(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
        exclude: Option<NormalizedRect>,
        labels: &ClipLabelSet,
    ) -> Option<ClipRegionObservation> {
        if width < PATCH_SIZE as u32 || height < PATCH_SIZE as u32 {
            return None;
        }
        let expected = (width * height * 3) as usize;
        if rgb.len() < expected {
            return None;
        }
        let exclude = normalize_rect(exclude);

        let mut scores = self.score_patches(
            rgb,
            width,
            height,
            exclude,
            &labels.positive,
            &labels.negative,
        )?;

        if scores.is_empty() {
            return None;
        }

        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = labels.top_k.min(scores.len()).max(1);
        let mut weight_sum = 0.0f32;
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let best_score = scores[0].score;
        let patch_w = PATCH_SIZE as f32 / width as f32;
        let patch_h = PATCH_SIZE as f32 / height as f32;
        let mut min_x = 1.0f32;
        let mut min_y = 1.0f32;
        let mut max_x = 0.0f32;
        let mut max_y = 0.0f32;

        for patch in scores.iter().take(top_k) {
            if patch.score < labels.score_min {
                continue;
            }
            let weight = patch.score.max(0.0);
            if weight <= 0.0 {
                continue;
            }
            weight_sum += weight;
            sum_x += patch.cx * weight;
            sum_y += patch.cy * weight;
            let x0 = (patch.cx - patch_w / 2.0).clamp(0.0, 1.0);
            let y0 = (patch.cy - patch_h / 2.0).clamp(0.0, 1.0);
            let x1 = (patch.cx + patch_w / 2.0).clamp(0.0, 1.0);
            let y1 = (patch.cy + patch_h / 2.0).clamp(0.0, 1.0);
            min_x = min_x.min(x0);
            min_y = min_y.min(y0);
            max_x = max_x.max(x1);
            max_y = max_y.max(y1);
        }

        if weight_sum <= 0.0 {
            if best_score < labels.score_min {
                return None;
            }
            let best = scores[0];
            weight_sum = 1.0;
            sum_x = best.cx;
            sum_y = best.cy;
            let x0 = (best.cx - patch_w / 2.0).clamp(0.0, 1.0);
            let y0 = (best.cy - patch_h / 2.0).clamp(0.0, 1.0);
            let x1 = (best.cx + patch_w / 2.0).clamp(0.0, 1.0);
            let y1 = (best.cy + patch_h / 2.0).clamp(0.0, 1.0);
            min_x = x0;
            min_y = y0;
            max_x = x1;
            max_y = y1;
        }

        let rect = NormalizedRect {
            x: clamp_unit(min_x),
            y: clamp_unit(min_y),
            w: clamp_unit((max_x - min_x).max(patch_w)),
            h: clamp_unit((max_y - min_y).max(patch_h)),
        };

        Some(ClipRegionObservation {
            rect,
            center: NormalizedPoint {
                x: clamp_unit(sum_x / weight_sum),
                y: clamp_unit(sum_y / weight_sum),
            },
            score: best_score,
        })
    }

    fn score_patches(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
        exclude: Option<NormalizedRect>,
        positive: &[Vec<f32>],
        negative: &[Vec<f32>],
    ) -> Option<Vec<PatchScore>> {
        let stride = self.stride.max(1) as usize;
        let patch = PATCH_SIZE;
        let mut scores: Vec<PatchScore> = Vec::new();

        let max_y = height as usize - patch;
        let max_x = width as usize - patch;
        let step_y = stride.min(max_y.max(1));
        let step_x = stride.min(max_x.max(1));

        for y in (0..=max_y).step_by(step_y) {
            for x in (0..=max_x).step_by(step_x) {
                let cx = (x + patch / 2) as f32 / width as f32;
                let cy = (y + patch / 2) as f32 / height as f32;
                if exclude.map(|rect| rect_contains(rect, cx, cy)).unwrap_or(false) {
                    continue;
                }
                let tensor = match patch_to_tensor(rgb, width as usize, height as usize, x, y) {
                    Some(val) => val,
                    None => continue,
                };
                let output = match self.vision_model.run(tvec!(tensor.into())) {
                    Ok(val) => val,
                    Err(_) => continue,
                };
                let embed = output
                    .get(0)
                    .and_then(|v| v.to_array_view::<f32>().ok())
                    .and_then(|v| v.as_slice().map(|s| s.to_vec()))?;
                let mut embed = embed;
                if !normalize_vec(&mut embed) {
                    continue;
                }
                let pos = max_similarity(&embed, positive);
                let neg = max_similarity(&embed, negative);
                let score = pos - neg;
                if score.is_finite() {
                    scores.push(PatchScore { score, cx, cy });
                }
            }
        }

        Some(scores)
    }

    fn encode_prompts(&self, prompts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::new();
        let input_count = self.text_model.model().input_outlets()?.len();
        if input_count == 0 {
            anyhow::bail!("text model has no inputs");
        }
        for prompt in prompts {
            let (ids, mask) = encode_prompt(&self.tokenizer, self.pad_id, prompt)?;
            let input_ids = Tensor::from_shape(&[1usize, SEQ_LEN], &ids)?;
            let output = match input_count {
                1 => self.text_model.run(tvec!(input_ids.into())),
                2 => {
                    let attention = Tensor::from_shape(&[1usize, SEQ_LEN], &mask)?;
                    self.text_model.run(tvec!(input_ids.into(), attention.into()))
                }
                other => anyhow::bail!(
                    "text model has unsupported input count {other}; expected 1 or 2"
                ),
            }
            .with_context(|| format!("text model failed on prompt '{prompt}'"))?;
            let embed = output
                .get(0)
                .and_then(|v| v.to_array_view::<f32>().ok())
                .and_then(|v| v.as_slice().map(|s| s.to_vec()))
                .context("missing text embedding output")?;
            let mut embed = embed;
            if normalize_vec(&mut embed) {
                out.push(embed);
            }
        }
        Ok(out)
    }
}

#[derive(Clone, Copy)]
struct PatchScore {
    score: f32,
    cx: f32,
    cy: f32,
}

fn max_similarity(embed: &[f32], bank: &[Vec<f32>]) -> f32 {
    if bank.is_empty() {
        return 0.0;
    }
    let mut best = f32::MIN;
    for candidate in bank {
        let mut sum = 0.0f32;
        let len = embed.len().min(candidate.len());
        for idx in 0..len {
            sum += embed[idx] * candidate[idx];
        }
        if sum > best {
            best = sum;
        }
    }
    best
}

fn normalize_vec(values: &mut [f32]) -> bool {
    let mut sum = 0.0f32;
    for v in values.iter() {
        sum += v * v;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return false;
    }
    let norm = sum.sqrt();
    if norm <= 0.0 {
        return false;
    }
    for v in values.iter_mut() {
        *v /= norm;
    }
    true
}

fn patch_to_tensor(
    rgb: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
) -> Option<Tensor> {
    let patch = PATCH_SIZE;
    if x0 + patch > width || y0 + patch > height {
        return None;
    }
    let mut data = vec![0f32; patch * patch * 3];
    let area = patch * patch;
    for y in 0..patch {
        let src_y = y0 + y;
        let src_row = src_y * width;
        for x in 0..patch {
            let src_x = x0 + x;
            let src_idx = (src_row + src_x) * 3;
            let r = rgb[src_idx] as f32 / 255.0;
            let g = rgb[src_idx + 1] as f32 / 255.0;
            let b = rgb[src_idx + 2] as f32 / 255.0;
            let dst = y * patch + x;
            data[dst] = (r - CLIP_MEAN[0]) / CLIP_STD[0];
            data[area + dst] = (g - CLIP_MEAN[1]) / CLIP_STD[1];
            data[area * 2 + dst] = (b - CLIP_MEAN[2]) / CLIP_STD[2];
        }
    }
    Tensor::from_shape(&[1usize, 3, patch, patch], &data).ok()
}

fn encode_prompt(
    tokenizer: &Tokenizer,
    pad_id: u32,
    prompt: &str,
) -> Result<(Vec<i64>, Vec<i64>)> {
    let encoding = tokenizer.encode(prompt, true).map_err(|err| {
        anyhow::anyhow!("tokenizing prompt '{prompt}' failed: {err}")
    })?;
    let mut ids: Vec<i64> = encoding.get_ids().iter().map(|v| *v as i64).collect();
    if ids.len() > SEQ_LEN {
        ids.truncate(SEQ_LEN);
    }
    let mut mask = vec![1i64; ids.len()];
    while ids.len() < SEQ_LEN {
        ids.push(pad_id as i64);
        mask.push(0);
    }
    Ok((ids, mask))
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

pub fn parse_label_list(value: String) -> Vec<String> {
    value
        .split(|c| c == '|' || c == ',')
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .collect()
}

fn apply_text_input_facts(mut model: InferenceModel) -> Result<InferenceModel> {
    let input_count = model.input_outlets()?.len();
    for idx in 0..input_count {
        model = model.with_input_fact(
            idx,
            InferenceFact::dt_shape(i64::datum_type(), tvec!(1usize, SEQ_LEN)),
        )?;
    }
    Ok(model)
}

fn clamp_unit(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn rect_contains(rect: NormalizedRect, cx: f32, cy: f32) -> bool {
    let x0 = rect.x.min(1.0).max(0.0);
    let y0 = rect.y.min(1.0).max(0.0);
    let x1 = (rect.x + rect.w).min(1.0).max(0.0);
    let y1 = (rect.y + rect.h).min(1.0).max(0.0);
    if x1 <= x0 || y1 <= y0 {
        return false;
    }
    cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1
}

fn normalize_rect(rect: Option<NormalizedRect>) -> Option<NormalizedRect> {
    let rect = rect?;
    if !rect.x.is_finite()
        || !rect.y.is_finite()
        || !rect.w.is_finite()
        || !rect.h.is_finite()
    {
        return None;
    }
    if rect.w <= 0.0 || rect.h <= 0.0 {
        return None;
    }
    Some(NormalizedRect {
        x: clamp_unit(rect.x),
        y: clamp_unit(rect.y),
        w: rect.w.clamp(0.0, 1.0),
        h: rect.h.clamp(0.0, 1.0),
    })
}

fn gameplay_debug_enabled() -> bool {
    std::env::var("CLIP_GAMEPLAY_DEBUG")
        .ok()
        .and_then(|v| parse_bool(&v))
        .unwrap_or(false)
}

pub fn log_gameplay_debug(message: &str) {
    if gameplay_debug_enabled() {
        eprintln!("{message}");
    }
}
