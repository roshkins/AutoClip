use std::env;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipLayoutMode {
    Full,
    Stacked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceAnchor {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Center,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizedPoint {
    pub x: f32,
    pub y: f32,
}

impl NormalizedPoint {
    fn clamp_unit(self) -> Self {
        Self {
            x: clamp_unit(self.x),
            y: clamp_unit(self.y),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizedRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl NormalizedRect {
    fn center(&self) -> NormalizedPoint {
        NormalizedPoint {
            x: self.x + self.w / 2.0,
            y: self.y + self.h / 2.0,
        }
    }

    #[cfg(test)]
    fn expanded(&self, scale: f32) -> Self {
        let scale = scale.max(1.0);
        let center = self.center().clamp_unit();
        let w = (self.w * scale).clamp(MIN_CROP_RATIO, 1.0);
        let h = (self.h * scale).clamp(MIN_CROP_RATIO, 1.0);
        Self::from_center(center, w, h)
    }

    fn from_center(center: NormalizedPoint, w: f32, h: f32) -> Self {
        let w = w.clamp(MIN_CROP_RATIO, 1.0);
        let h = h.clamp(MIN_CROP_RATIO, 1.0);
        let max_w = (center.x.min(1.0 - center.x) * 2.0).max(0.0);
        let max_h = (center.y.min(1.0 - center.y) * 2.0).max(0.0);
        let w = if max_w > 0.0 { w.min(max_w) } else { w };
        let h = if max_h > 0.0 { h.min(max_h) } else { h };
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
        Self {
            x: clamp_unit(x),
            y: clamp_unit(y),
            w,
            h,
        }
    }
}

fn expand_rect_width_to_aspect(rect: NormalizedRect, target_aspect: f32) -> NormalizedRect {
    let target_aspect = if target_aspect.is_finite() && target_aspect > 0.0 {
        target_aspect
    } else {
        return rect;
    };
    let aspect = rect.w / rect.h;
    if !aspect.is_finite() || aspect >= target_aspect {
        return rect;
    }
    let new_w = (rect.h * target_aspect).clamp(MIN_CROP_RATIO, 1.0);
    NormalizedRect::from_center(rect.center().clamp_unit(), new_w, rect.h)
}

fn frame_rect_for_face(
    face_rect: NormalizedRect,
    spec: FaceFrameSpec,
    target_aspect: f32,
    bounds: NormalizedRect,
) -> NormalizedRect {
    let bounds = normalize_bounds(bounds);
    let mut head_top = face_rect.y + face_rect.h * spec.head_top_offset;
    if !head_top.is_finite() {
        head_top = face_rect.y;
    }
    let mut width = if spec.shoulder_width_scale.is_finite() && spec.shoulder_width_scale > 0.0 {
        face_rect.w * spec.shoulder_width_scale
    } else {
        face_rect.w
    };
    if !width.is_finite() || width <= 0.0 {
        width = face_rect.w.max(MIN_CROP_RATIO);
    }
    width = width.clamp(MIN_CROP_RATIO, bounds.w.max(MIN_CROP_RATIO));
    let mut height = if target_aspect.is_finite() && target_aspect > 0.0 {
        width / target_aspect
    } else {
        width
    };
    if !height.is_finite() || height <= 0.0 {
        height = face_rect.h.max(MIN_CROP_RATIO);
    }
    if height > bounds.h {
        height = bounds.h.max(MIN_CROP_RATIO);
        if target_aspect.is_finite() && target_aspect > 0.0 {
            width = (height * target_aspect).min(bounds.w);
        }
    }
    let center_x = face_rect.x + face_rect.w / 2.0;
    let mut x = center_x - width / 2.0;
    let mut y = head_top;
    let min_x = bounds.x;
    let max_x = (bounds.x + bounds.w - width).max(min_x);
    let min_y = bounds.y;
    let max_y = (bounds.y + bounds.h - height).max(min_y);
    if x < min_x {
        x = min_x;
    }
    if x > max_x {
        x = max_x;
    }
    if y < min_y {
        y = min_y;
    }
    if y > max_y {
        y = max_y;
    }
    NormalizedRect {
        x: clamp_unit(x),
        y: clamp_unit(y),
        w: width,
        h: height,
    }
}

#[derive(Clone, Debug)]
pub struct FaceTrackPoint {
    pub time: f32,
    pub rect: NormalizedRect,
}

#[derive(Clone, Debug)]
pub struct FaceTrack {
    pub points: Vec<FaceTrackPoint>,
}

#[derive(Clone, Debug)]
pub struct ClipLayoutConfig {
    pub mode: ClipLayoutMode,
    pub face_ratio: f32,
    pub face_crop: Option<String>,
    pub face_anchor: FaceAnchor,
    pub face_context_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceFrameSpec {
    pub head_top_offset: f32,
    pub shoulder_width_scale: f32,
}

#[derive(Clone, Debug, Default)]
pub struct ClipLayoutHints {
    pub face_box: Option<NormalizedRect>,
    pub face_track: Option<FaceTrack>,
    pub face_region: Option<NormalizedRect>,
    pub face_frame_spec: Option<FaceFrameSpec>,
    pub game_center: Option<NormalizedPoint>,
    pub game_region: Option<NormalizedRect>,
}

#[derive(Clone, Debug)]
pub enum FilterGraph {
    Vf(String),
    Complex { graph: String, output: String },
}

#[derive(Clone, Copy, Debug)]
pub struct StackedLayoutDims {
    pub face_h: u32,
    pub game_h: u32,
}

const DEFAULT_FACE_RATIO: f32 = 0.40;
const DEFAULT_FACE_CONTEXT_SCALE: f32 = 6.0;
const FORCE_HALF_FACE_CONTEXT: f32 = 3.0;
const DEFAULT_GAME_CENTER_X: f32 = 0.50;
const DEFAULT_GAME_CENTER_Y: f32 = 0.50;
const MIDSHOT_FACE_RATIO: f32 = 0.50;
const MIN_CROP_RATIO: f32 = 0.20;
const MIN_FACE_REFRAME_SECS: f32 = 6.0;
const KALMAN_PROCESS_VAR: f32 = 0.0005;
const KALMAN_MEASURE_VAR: f32 = 0.0025;

pub fn read_clip_layout_config() -> ClipLayoutConfig {
    let mode = env::var("CLIP_LAYOUT")
        .ok()
        .map(|v| v.to_ascii_lowercase())
        .and_then(|v| match v.trim() {
            "stacked" | "tiktok" | "stack" | "split" => Some(ClipLayoutMode::Stacked),
            "full" | "default" | "" => Some(ClipLayoutMode::Full),
            _ => None,
        })
        .unwrap_or(ClipLayoutMode::Stacked);

    let face_ratio = env::var("CLIP_FACE_RATIO")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(0.2, 0.8))
        .unwrap_or(DEFAULT_FACE_RATIO);

    let face_crop = env::var("CLIP_FACE_CROP")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty() && !v.eq_ignore_ascii_case("none"))
        .map(|v| v.strip_prefix("crop=").unwrap_or(&v).to_string());

    let face_anchor = env::var("CLIP_FACE_ANCHOR")
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

    let face_context_scale = env::var("CLIP_FACE_CONTEXT")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.max(1.0))
        .unwrap_or(DEFAULT_FACE_CONTEXT_SCALE);

    ClipLayoutConfig {
        mode,
        face_ratio,
        face_crop,
        face_anchor,
        face_context_scale,
    }
}

pub fn read_clip_layout_hints() -> ClipLayoutHints {
    let face_box = env::var("CLIP_FACE_BOX").ok().and_then(|v| parse_rect(&v));
    let face_region = env::var("CLIP_FACE_REGION").ok().and_then(|v| parse_rect(&v));
    let game_center = env::var("CLIP_GAME_CENTER").ok().and_then(|v| parse_point(&v));
    let game_region = env::var("CLIP_GAME_REGION").ok().and_then(|v| parse_rect(&v));

    ClipLayoutHints {
        face_box,
        face_track: None,
        face_region,
        face_frame_spec: None,
        game_center,
        game_region,
    }
}

pub fn resolve_layout_heights(out_h: u32, face_ratio: f32) -> (u32, u32) {
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

pub fn resolve_stacked_layout_dims(
    out_w: u32,
    out_h: u32,
    layout: &ClipLayoutConfig,
    hints: &ClipLayoutHints,
) -> StackedLayoutDims {
    let base_face_h = resolve_layout_heights(out_h, layout.face_ratio).0;
    let base_aspect = out_w as f32 / base_face_h as f32;
    let face_rect = face_crop_rect(layout, hints, base_aspect);
    let tracked_max = tracked_face_max_rect(layout, hints, base_aspect);
    let force_half = layout.face_context_scale > FORCE_HALF_FACE_CONTEXT
        && (tracked_max.is_some() || face_rect.is_some());
    let mut face_ratio = if hints.face_frame_spec.is_some() {
        layout.face_ratio
    } else {
        tracked_max
            .and_then(|rect| face_ratio_from_rect(rect, out_w, out_h))
            .or_else(|| face_rect.and_then(|rect| face_ratio_from_rect(rect, out_w, out_h)))
            .unwrap_or(layout.face_ratio)
    };
    if force_half {
        face_ratio = MIDSHOT_FACE_RATIO;
    }
    let (face_h, game_h) = resolve_layout_heights(out_h, face_ratio);
    StackedLayoutDims {
        face_h,
        game_h,
    }
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

fn face_crop_rect(
    layout: &ClipLayoutConfig,
    hints: &ClipLayoutHints,
    target_aspect: f32,
) -> Option<NormalizedRect> {
    if layout.face_crop.is_some() {
        return None;
    }
    let face_box = hints.face_box?;
    let bounds = hints.face_region.unwrap_or(NormalizedRect {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    });
    if let Some(spec) = hints.face_frame_spec {
        return Some(frame_rect_for_face(face_box, spec, target_aspect, bounds));
    }
    Some(expand_rect_in_bounds(
        face_box,
        layout.face_context_scale,
        bounds,
    ))
}

fn build_face_crop_expr(layout: &ClipLayoutConfig, rect: Option<NormalizedRect>) -> String {
    if let Some(face_crop) = layout.face_crop.clone() {
        return face_crop;
    }
    if let Some(rect) = rect {
        return crop_expr_from_rect(rect);
    }
    default_face_crop_expr(layout.face_anchor)
}

fn crop_expr_from_rect(rect: NormalizedRect) -> String {
    let w = format_ratio(rect.w);
    let h = format_ratio(rect.h);
    let x = format_ratio(rect.x);
    let y = format_ratio(rect.y);
    format!("iw*{w}:ih*{h}:iw*{x}:ih*{y}")
}

fn axis_center_expr(axis: &str, crop_dim: u32, center: f32) -> String {
    let center = clamp_unit(center);
    let half = crop_dim as f32 / 2.0;
    let half_str = format!("{half:.1}");
    format!(
        "max(min({axis}*{center:.4}-{half_str}\\, {axis}-{crop_dim})\\, 0)"
    )
}

pub fn build_stacked_filter_graph(
    out_w: u32,
    out_h: u32,
    layout: &ClipLayoutConfig,
    hints: &ClipLayoutHints,
) -> FilterGraph {
    let dims = resolve_stacked_layout_dims(out_w, out_h, layout, hints);
    let face_h = dims.face_h;
    let game_h = dims.game_h;
    let target_aspect = out_w as f32 / face_h as f32;
    let face_rect = face_crop_rect(layout, hints, target_aspect);
    let face_rect = face_rect.map(|rect| expand_rect_width_to_aspect(rect, target_aspect));
    let tracked = build_tracked_face_crop(layout, hints, target_aspect);
    let face_crop = if let Some(tracked) = tracked {
        tracked.crop_expr
    } else {
        build_face_crop_expr(layout, face_rect)
    };
    let game_center = hints.game_center.unwrap_or_else(default_game_center);
    let game_x = axis_center_expr("iw", out_w, game_center.x);
    let game_y = axis_center_expr("ih", game_h, game_center.y);

    let face_chain = format!(
        "crop={face_crop},scale={out_w}:{face_h}:force_original_aspect_ratio=increase,crop={out_w}:{face_h}"
    );
    let game_chain = format!(
        "scale={out_w}:{game_h}:force_original_aspect_ratio=increase,crop={out_w}:{game_h}:{game_x}:{game_y}"
    );
    let graph = format!(
        "[0:v]split=2[face_src][game_src];[face_src]{face_chain}[face];[game_src]{game_chain}[game];[face][game]vstack=inputs=2,format=yuv420p[v]"
    );
    FilterGraph::Complex {
        graph,
        output: "v".to_string(),
    }
}

pub fn build_face_only_filter_graph(
    out_w: u32,
    out_h: u32,
    layout: &ClipLayoutConfig,
    hints: &ClipLayoutHints,
) -> FilterGraph {
    let target_aspect = out_w as f32 / out_h as f32;
    let face_rect = face_crop_rect(layout, hints, target_aspect);
    let face_rect = face_rect.map(|rect| expand_rect_width_to_aspect(rect, target_aspect));
    let tracked = build_tracked_face_crop(layout, hints, target_aspect);
    let face_crop = if let Some(tracked) = tracked {
        tracked.crop_expr
    } else {
        build_face_crop_expr(layout, face_rect)
    };
    let chain = format!(
        "crop={face_crop},scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},format=yuv420p"
    );
    FilterGraph::Vf(chain)
}

pub fn build_full_frame_fill_filter_graph(
    out_w: u32,
    out_h: u32,
    center: NormalizedPoint,
) -> FilterGraph {
    let x = axis_center_expr("iw", out_w, center.x);
    let y = axis_center_expr("ih", out_h, center.y);
    let chain = format!(
        "scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h}:{x}:{y},format=yuv420p"
    );
    FilterGraph::Vf(chain)
}

pub fn build_tracked_full_frame_fill_filter_graph(
    out_w: u32,
    out_h: u32,
    track: &FaceTrack,
) -> Option<FilterGraph> {
    if track.points.len() < 2 {
        return None;
    }
    let mut samples: Vec<TrackSample> = track
        .points
        .iter()
        .map(|point| TrackSample {
            time: point.time.max(0.0),
            center: point.rect.center().clamp_unit(),
            w: point.rect.w,
            h: point.rect.h,
        })
        .collect();
    samples.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    let mut deduped: Vec<TrackSample> = Vec::with_capacity(samples.len());
    for sample in samples {
        if let Some(last) = deduped.last_mut() {
            if (sample.time - last.time).abs() < 0.001 {
                *last = sample;
                continue;
            }
        }
        deduped.push(sample);
    }
    if deduped.len() < 2 {
        return None;
    }

    let smoothed = kalman_smooth_samples(&deduped);
    let downsampled = downsample_track_samples(&smoothed, MIN_FACE_REFRAME_SECS);
    if downsampled.len() < 2 {
        return None;
    }

    let center_x_expr = piecewise_lerp_expr(&downsampled, |s| s.center.x);
    let center_y_expr = piecewise_lerp_expr(&downsampled, |s| s.center.y);
    let x_expr = axis_center_expr_expr("iw", out_w, &center_x_expr);
    let y_expr = axis_center_expr_expr("ih", out_h, &center_y_expr);
    let chain = format!(
        "scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h}:{x_expr}:{y_expr},format=yuv420p"
    );
    Some(FilterGraph::Vf(chain))
}

struct TrackedFaceCrop {
    crop_expr: String,
}

#[derive(Clone, Copy, Debug)]
struct TrackSample {
    time: f32,
    center: NormalizedPoint,
    w: f32,
    h: f32,
}

#[derive(Clone, Copy, Debug)]
struct Kalman1D {
    x: f32,
    v: f32,
    p00: f32,
    p01: f32,
    p10: f32,
    p11: f32,
}

impl Kalman1D {
    fn new(value: f32) -> Self {
        Self {
            x: value,
            v: 0.0,
            p00: 1.0,
            p01: 0.0,
            p10: 0.0,
            p11: 1.0,
        }
    }

    fn update(&mut self, measurement: f32, dt: f32) -> f32 {
        if !measurement.is_finite() {
            return self.x;
        }
        let dt = dt.max(0.001);
        self.x += self.v * dt;

        let p00 = self.p00 + dt * (self.p10 + self.p01) + dt * dt * self.p11;
        let p01 = self.p01 + dt * self.p11;
        let p10 = self.p10 + dt * self.p11;
        let p11 = self.p11;

        let q00 = 0.25 * dt * dt * dt * dt * KALMAN_PROCESS_VAR;
        let q01 = 0.5 * dt * dt * dt * KALMAN_PROCESS_VAR;
        let q11 = dt * dt * KALMAN_PROCESS_VAR;

        let p00 = p00 + q00;
        let p01 = p01 + q01;
        let p10 = p10 + q01;
        let p11 = p11 + q11;

        let s = p00 + KALMAN_MEASURE_VAR;
        let k0 = p00 / s;
        let k1 = p10 / s;
        let y = measurement - self.x;

        self.x += k0 * y;
        self.v += k1 * y;
        self.p00 = (1.0 - k0) * p00;
        self.p01 = (1.0 - k0) * p01;
        self.p10 = p10 - k1 * p00;
        self.p11 = p11 - k1 * p01;

        self.x
    }
}

fn track_sample_area(sample: &TrackSample) -> f32 {
    let area = sample.w * sample.h;
    if !area.is_finite() {
        return 0.0;
    }
    area.max(0.0)
}

fn kalman_smooth_samples(samples: &[TrackSample]) -> Vec<TrackSample> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(samples.len());
    let mut kx = Kalman1D::new(samples[0].center.x);
    let mut ky = Kalman1D::new(samples[0].center.y);
    let mut kw = Kalman1D::new(samples[0].w);
    let mut kh = Kalman1D::new(samples[0].h);
    let mut last_time = samples[0].time;
    out.push(samples[0]);
    for sample in samples.iter().skip(1) {
        let dt = (sample.time - last_time).max(0.001);
        let x = kx.update(sample.center.x, dt);
        let y = ky.update(sample.center.y, dt);
        let w = kw.update(sample.w, dt);
        let h = kh.update(sample.h, dt);
        out.push(TrackSample {
            time: sample.time,
            center: NormalizedPoint {
                x: clamp_unit(x),
                y: clamp_unit(y),
            },
            w: w.clamp(MIN_CROP_RATIO, 1.0),
            h: h.clamp(MIN_CROP_RATIO, 1.0),
        });
        last_time = sample.time;
    }
    out
}

fn downsample_track_samples(samples: &[TrackSample], min_interval: f32) -> Vec<TrackSample> {
    if samples.is_empty() {
        return Vec::new();
    }
    let min_interval = min_interval.max(0.0);
    let mut out: Vec<TrackSample> = Vec::new();
    for sample in samples {
        if let Some(last) = out.last_mut() {
            if sample.time - last.time < min_interval {
                if track_sample_area(sample) > track_sample_area(last) {
                    *last = *sample;
                }
                continue;
            }
        }
        out.push(*sample);
    }
    out
}

fn build_tracked_face_crop(
    layout: &ClipLayoutConfig,
    hints: &ClipLayoutHints,
    target_aspect: f32,
) -> Option<TrackedFaceCrop> {
    let track = hints.face_track.as_ref()?;
    if track.points.len() < 2 || layout.face_crop.is_some() {
        return None;
    }

    let mut points = track.points.clone();
    points.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    let bounds = hints.face_region.unwrap_or(NormalizedRect {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    });

    let mut samples: Vec<TrackSample> = Vec::with_capacity(points.len());
    let frame_spec = hints.face_frame_spec;
    for point in points {
        let rect = if let Some(spec) = frame_spec {
            frame_rect_for_face(point.rect, spec, target_aspect, bounds)
        } else {
            expand_rect_in_bounds(point.rect, layout.face_context_scale, bounds)
        };
        let rect = expand_rect_width_to_aspect(rect, target_aspect);
        samples.push(TrackSample {
            time: point.time.max(0.0),
            center: rect.center().clamp_unit(),
            w: rect.w,
            h: rect.h,
        });
    }

    samples.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    let mut deduped: Vec<TrackSample> = Vec::with_capacity(samples.len());
    for sample in samples {
        if let Some(last) = deduped.last_mut() {
            if (sample.time - last.time).abs() < 0.001 {
                *last = sample;
                continue;
            }
        }
        deduped.push(sample);
    }

    if deduped.len() < 2 {
        return None;
    }

    let smoothed = kalman_smooth_samples(&deduped);
    let downsampled = downsample_track_samples(&smoothed, MIN_FACE_REFRAME_SECS);
    if downsampled.len() < 2 {
        return None;
    }

    let center_x_expr = piecewise_lerp_expr(&downsampled, |s| s.center.x);
    let center_y_expr = piecewise_lerp_expr(&downsampled, |s| s.center.y);
    let width_expr = clamp_ratio_expr(&piecewise_lerp_expr(&downsampled, |s| s.w));
    let height_expr = clamp_ratio_expr(&piecewise_lerp_expr(&downsampled, |s| s.h));
    let x_expr = axis_center_expr_ratio_expr("iw", &width_expr, &center_x_expr);
    let y_expr = axis_center_expr_ratio_expr("ih", &height_expr, &center_y_expr);
    let crop_expr = format!(
        "iw*({width_expr}):ih*({height_expr}):{x_expr}:{y_expr}"
    );

    Some(TrackedFaceCrop { crop_expr })
}

fn tracked_face_max_rect(
    layout: &ClipLayoutConfig,
    hints: &ClipLayoutHints,
    target_aspect: f32,
) -> Option<NormalizedRect> {
    let track = hints.face_track.as_ref()?;
    if track.points.is_empty() || layout.face_crop.is_some() {
        return None;
    }
    let bounds = hints.face_region.unwrap_or(NormalizedRect {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    });
    let frame_spec = hints.face_frame_spec;
    let mut max_w = MIN_CROP_RATIO;
    let mut max_h = MIN_CROP_RATIO;
    for point in &track.points {
        let rect = if let Some(spec) = frame_spec {
            frame_rect_for_face(point.rect, spec, target_aspect, bounds)
        } else {
            expand_rect_in_bounds(point.rect, layout.face_context_scale, bounds)
        };
        max_w = max_w.max(rect.w);
        max_h = max_h.max(rect.h);
    }
    Some(NormalizedRect {
        x: 0.0,
        y: 0.0,
        w: max_w,
        h: max_h,
    })
}

fn clamp_ratio_expr(expr: &str) -> String {
    format!("min(max({expr}\\,{MIN_CROP_RATIO:.4})\\,1.0)")
}

fn axis_center_expr_ratio_expr(axis: &str, ratio_expr: &str, center_expr: &str) -> String {
    let size_expr = format!("{axis}*({ratio_expr})");
    let half_expr = format!("{axis}*({ratio_expr})/2");
    let center_expr = format!("{axis}*({center_expr})");
    format!("max(min({center_expr}-{half_expr}\\, {axis}-{size_expr})\\, 0)")
}

fn axis_center_expr_expr(axis: &str, crop_dim: u32, center_expr: &str) -> String {
    let half = crop_dim as f32 / 2.0;
    let half_str = format!("{half:.1}");
    let center_expr = format!("{axis}*({center_expr})");
    format!(
        "max(min({center_expr}-{half_str}\\, {axis}-{crop_dim})\\, 0)"
    )
}

fn piecewise_lerp_expr(samples: &[TrackSample], getter: fn(&TrackSample) -> f32) -> String {
    let last = samples
        .last()
        .expect("piecewise_lerp_expr expects at least one sample");
    let mut expr = format!("{:.4}", getter(last));
    if samples.len() < 2 {
        return expr;
    }
    for idx in (0..samples.len() - 1).rev() {
        let start = &samples[idx];
        let end = &samples[idx + 1];
        let start_t = start.time;
        let end_t = end.time;
        let start_val = getter(start);
        let end_val = getter(end);
        let span = end_t - start_t;
        let segment_expr = if span.abs() < 0.001 {
            format!("{start_val:.4}")
        } else {
            format!(
                "{start_val:.4}+({end_val:.4}-{start_val:.4})*(t-{start_t:.3})/{span:.3}"
            )
        };
        expr = format!(
            "if(between(t\\,{start_t:.3}\\,{end_t:.3})\\,{segment_expr}\\,{expr})"
        );
    }
    expr
}

fn face_ratio_from_rect(rect: NormalizedRect, out_w: u32, out_h: u32) -> Option<f32> {
    if rect.w <= 0.0 || rect.h <= 0.0 || out_h == 0 {
        return None;
    }
    let aspect = rect.w / rect.h;
    if !aspect.is_finite() || aspect <= 0.0 {
        return None;
    }
    let face_h = out_w as f32 / aspect;
    let ratio = face_h / out_h as f32;
    if !ratio.is_finite() {
        return None;
    }
    Some(ratio.clamp(0.2, 0.8))
}

fn default_game_center() -> NormalizedPoint {
    NormalizedPoint {
        x: DEFAULT_GAME_CENTER_X,
        y: DEFAULT_GAME_CENTER_Y,
    }
}

fn parse_rect(value: &str) -> Option<NormalizedRect> {
    let vals = parse_unit_list(value, 4)?;
    let rect = NormalizedRect {
        x: vals[0],
        y: vals[1],
        w: vals[2],
        h: vals[3],
    };
    if rect.w <= 0.0 || rect.h <= 0.0 {
        return None;
    }
    Some(rect)
}

fn parse_point(value: &str) -> Option<NormalizedPoint> {
    let vals = parse_unit_list(value, 2)?;
    Some(NormalizedPoint { x: vals[0], y: vals[1] })
}

fn parse_unit_list(value: &str, expected: usize) -> Option<Vec<f32>> {
    let mut out = Vec::new();
    for part in value.split(|c| c == ',' || c == ':') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val = trimmed.parse::<f32>().ok()?;
        let val = parse_unit_value(val)?;
        out.push(val);
    }
    if out.len() == expected {
        Some(out)
    } else {
        None
    }
}

fn expand_rect_in_bounds(
    rect: NormalizedRect,
    scale: f32,
    bounds: NormalizedRect,
) -> NormalizedRect {
    let scale = if scale.is_finite() { scale.max(1.0) } else { 1.0 };
    let bounds = normalize_bounds(bounds);
    let desired_w = (rect.w * scale)
        .max(rect.w)
        .clamp(MIN_CROP_RATIO, bounds.w.max(MIN_CROP_RATIO));
    let desired_h = (rect.h * scale)
        .max(rect.h)
        .clamp(MIN_CROP_RATIO, bounds.h.max(MIN_CROP_RATIO));
    let mut x = rect.x + rect.w / 2.0 - desired_w / 2.0;
    let mut y = rect.y + rect.h / 2.0 - desired_h / 2.0;
    let min_x = bounds.x;
    let max_x = (bounds.x + bounds.w - desired_w).max(min_x);
    let min_y = bounds.y;
    let max_y = (bounds.y + bounds.h - desired_h).max(min_y);
    if x < min_x {
        x = min_x;
    }
    if x > max_x {
        x = max_x;
    }
    if y < min_y {
        y = min_y;
    }
    if y > max_y {
        y = max_y;
    }
    NormalizedRect {
        x: clamp_unit(x),
        y: clamp_unit(y),
        w: desired_w,
        h: desired_h,
    }
}

fn normalize_bounds(bounds: NormalizedRect) -> NormalizedRect {
    if bounds.w <= 0.0 || bounds.h <= 0.0 {
        return NormalizedRect {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        };
    }
    let x = bounds.x.clamp(0.0, 1.0);
    let y = bounds.y.clamp(0.0, 1.0);
    let mut w = bounds.w.clamp(0.0, 1.0);
    let mut h = bounds.h.clamp(0.0, 1.0);
    if x + w > 1.0 {
        w = (1.0 - x).max(0.0);
    }
    if y + h > 1.0 {
        h = (1.0 - y).max(0.0);
    }
    if w <= 0.0 || h <= 0.0 {
        return NormalizedRect {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        };
    }
    NormalizedRect { x, y, w, h }
}

fn parse_unit_value(value: f32) -> Option<f32> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Some(value)
    } else {
        None
    }
}

fn clamp_unit(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn format_ratio(value: f32) -> String {
    format!("{value:.4}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_layout_heights_returns_even_sizes() {
        let (face_h, game_h) = resolve_layout_heights(1920, 0.41);
        assert_eq!(face_h + game_h, 1920);
        assert_eq!(face_h % 2, 0);
        assert_eq!(game_h % 2, 0);
        assert!(face_h >= 2);
        assert!(game_h >= 2);
    }

    #[test]
    fn expanded_face_rect_clamps_to_bounds() {
        let rect = NormalizedRect {
            x: 0.0,
            y: 0.0,
            w: 0.2,
            h: 0.2,
        };
        let expanded = rect.expanded(2.0);
        assert!((expanded.w - 0.2).abs() < 1e-6);
        assert!((expanded.h - 0.2).abs() < 1e-6);
        assert!((expanded.x - 0.0).abs() < 1e-6);
        assert!((expanded.y - 0.0).abs() < 1e-6);
    }

    #[test]
    fn stacked_graph_uses_face_box_and_game_center() {
        let layout = ClipLayoutConfig {
            mode: ClipLayoutMode::Stacked,
            face_ratio: 0.4,
            face_crop: None,
            face_anchor: FaceAnchor::TopLeft,
            face_context_scale: 1.5,
        };
        let hints = ClipLayoutHints {
            face_box: Some(NormalizedRect {
                x: 0.1,
                y: 0.1,
                w: 0.2,
                h: 0.2,
            }),
            face_track: None,
            face_region: None,
            face_frame_spec: None,
            game_center: Some(NormalizedPoint { x: 0.5, y: 0.6 }),
            game_region: None,
        };
        let FilterGraph::Complex { graph, output } =
            build_stacked_filter_graph(1080, 1920, &layout, &hints)
        else {
            panic!("expected complex filter graph");
        };
        assert_eq!(output, "v");
        assert!(
            graph.contains("crop=iw*0.3000:ih*0.3000:iw*0.0500:ih*0.0500"),
            "face crop should expand around the detected face"
        );
        assert!(
            graph.contains("crop=1080:840:max(min(iw*0.5000-540.0\\, iw-1080)\\, 0):max(min(ih*0.6000-420.0\\, ih-840)\\, 0)"),
            "game crop should center on the provided reticle hint"
        );
    }

    #[test]
    fn tracked_full_frame_fill_uses_dynamic_center() {
        let track = FaceTrack {
            points: vec![
                FaceTrackPoint {
                    time: 0.0,
                    rect: NormalizedRect {
                        x: 0.1,
                        y: 0.2,
                        w: 0.2,
                        h: 0.2,
                    },
                },
                FaceTrackPoint {
                    time: 6.0,
                    rect: NormalizedRect {
                        x: 0.6,
                        y: 0.4,
                        w: 0.2,
                        h: 0.2,
                    },
                },
            ],
        };
        let FilterGraph::Vf(chain) =
            build_tracked_full_frame_fill_filter_graph(1080, 1920, &track)
                .expect("expected tracked full-frame graph")
        else {
            panic!("expected Vf filter graph");
        };
        assert!(
            chain.contains("scale=1080:1920:force_original_aspect_ratio=increase"),
            "expected full-frame scale"
        );
        assert!(
            chain.contains("crop=1080:1920:"),
            "expected full-frame crop"
        );
        assert!(
            chain.contains("if(between(t"),
            "expected dynamic center expression"
        );
    }
}
