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

#[derive(Clone, Debug, Default)]
pub struct ClipLayoutHints {
    pub face_box: Option<NormalizedRect>,
    pub face_track: Option<FaceTrack>,
    pub game_center: Option<NormalizedPoint>,
}

#[derive(Clone, Debug)]
pub enum FilterGraph {
    Vf(String),
    Complex { graph: String, output: String },
}

const DEFAULT_FACE_RATIO: f32 = 0.40;
const DEFAULT_FACE_CONTEXT_SCALE: f32 = 3.0;
const DEFAULT_GAME_CENTER_X: f32 = 0.50;
const DEFAULT_GAME_CENTER_Y: f32 = 0.56;
const MIDSHOT_FACE_RATIO: f32 = 0.50;
const MIN_CROP_RATIO: f32 = 0.20;

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
        .map(|v| v.clamp(1.0, 6.0))
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
    let game_center = env::var("CLIP_GAME_CENTER").ok().and_then(|v| parse_point(&v));

    ClipLayoutHints {
        face_box,
        face_track: None,
        game_center,
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

fn face_crop_rect(layout: &ClipLayoutConfig, hints: &ClipLayoutHints) -> Option<NormalizedRect> {
    if layout.face_crop.is_some() {
        return None;
    }
    hints
        .face_box
        .map(|face_box| face_box.expanded(layout.face_context_scale))
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
    let face_rect = face_crop_rect(layout, hints);
    let tracked_max = tracked_face_max_rect(layout, hints);
    let force_half = layout.face_context_scale > DEFAULT_FACE_CONTEXT_SCALE
        && (tracked_max.is_some() || face_rect.is_some());
    let mut face_ratio = tracked_max
        .and_then(|rect| face_ratio_from_rect(rect, out_w, out_h))
        .or_else(|| face_rect.and_then(|rect| face_ratio_from_rect(rect, out_w, out_h)))
        .unwrap_or(layout.face_ratio);
    if force_half {
        face_ratio = MIDSHOT_FACE_RATIO;
    }
    let (face_h, game_h) = resolve_layout_heights(out_h, face_ratio);
    let target_aspect = out_w as f32 / face_h as f32;
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

    let mut samples: Vec<TrackSample> = Vec::with_capacity(points.len());
    for point in points {
        let rect = point
            .rect
            .expanded(layout.face_context_scale);
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

    let center_x_expr = piecewise_lerp_expr(&deduped, |s| s.center.x);
    let center_y_expr = piecewise_lerp_expr(&deduped, |s| s.center.y);
    let width_expr = clamp_ratio_expr(&piecewise_lerp_expr(&deduped, |s| s.w));
    let height_expr = clamp_ratio_expr(&piecewise_lerp_expr(&deduped, |s| s.h));
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
) -> Option<NormalizedRect> {
    let track = hints.face_track.as_ref()?;
    if track.points.is_empty() || layout.face_crop.is_some() {
        return None;
    }
    let mut max_w = MIN_CROP_RATIO;
    let mut max_h = MIN_CROP_RATIO;
    for point in &track.points {
        let rect = point.rect.expanded(layout.face_context_scale);
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
            game_center: Some(NormalizedPoint { x: 0.5, y: 0.6 }),
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
}
