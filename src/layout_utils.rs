//! Layout math helpers shared by layout selection and low-resource heuristics.
//!
//! These functions operate on normalized (0..=1) rectangles and points and are
//! intentionally lightweight so they can be used in both the live and offline
//! layout paths.

use crate::clip_layout;
use crate::clip_layout::NormalizedRect;

/// Parse a resolution string like `"1080x1920"` into `(width, height)`.
///
/// Returns `None` for malformed input or non-numeric dimensions.
pub fn parse_resolution(res: &str) -> Option<(u32, u32)> {
    let parts: Vec<_> = res.split('x').collect();
    if parts.len() != 2 {
        return None;
    }
    let w = parts[0].parse().ok()?;
    let h = parts[1].parse().ok()?;
    Some((w, h))
}

/// Compute the area of a normalized rectangle.
///
/// Any non-finite values are treated as `0.0`.
pub fn rect_area(rect: NormalizedRect) -> f32 {
    let area = rect.w * rect.h;
    if !area.is_finite() {
        return 0.0;
    }
    area.max(0.0)
}

/// Extract the face area from layout hints, if a face box/track exists.
///
/// This is a convenience for quick "face present" checks in heuristics.
#[allow(dead_code)]
pub fn face_area_from_hints(hints: &clip_layout::ClipLayoutHints) -> Option<f32> {
    let rect = hints.face_box.or_else(|| {
        hints
            .face_track
            .as_ref()
            .and_then(|track| track.points.first().map(|p| p.rect))
    })?;
    Some(rect_area(rect))
}

/// Return the most suitable face rectangle from layout hints.
///
/// Prefers `face_region`, falls back to `face_box`, then the first track point.
pub fn face_rect_from_hints(hints: &clip_layout::ClipLayoutHints) -> Option<NormalizedRect> {
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

/// Return a face rectangle suitable for gameplay-guessing heuristics.
///
/// This intentionally prefers the raw face box/track so the heuristic can
/// reason about face position and size relative to the frame.
pub fn face_rect_for_gameplay_guess(
    hints: &clip_layout::ClipLayoutHints,
) -> Option<NormalizedRect> {
    hints.face_box.or_else(|| {
        hints
            .face_track
            .as_ref()
            .and_then(|track| track.points.first().map(|p| p.rect))
    })
}

/// Low-resource heuristic that guesses whether gameplay is visible.
///
/// Returns `(gameplay_present, face_area, edge_bias)` when a face is found.
/// The heuristic biases toward "gameplay present" when faces are small or
/// near the frame edges, and toward "face-only" when large or centered.
pub fn guess_gameplay_low_resource(
    hints: &clip_layout::ClipLayoutHints,
    face_only_area_threshold: f32,
) -> Option<(bool, f32, f32)> {
    let rect = face_rect_for_gameplay_guess(hints)?;
    let area = rect_area(rect);
    if !area.is_finite() {
        return None;
    }
    let center_x = (rect.x + rect.w / 2.0).clamp(0.0, 1.0);
    let center_y = (rect.y + rect.h / 2.0).clamp(0.0, 1.0);
    let edge_bias = center_x
        .min(center_y)
        .min(1.0 - center_x)
        .min(1.0 - center_y);
    const TINY_FACE_AREA: f32 = 0.03;
    const EDGE_BIAS_THRESHOLD: f32 = 0.22;
    const CENTER_BIAS_THRESHOLD: f32 = 0.30;
    if area >= face_only_area_threshold {
        return Some((false, area, edge_bias));
    }
    if area <= TINY_FACE_AREA {
        let gameplay_present = edge_bias <= EDGE_BIAS_THRESHOLD;
        return Some((gameplay_present, area, edge_bias));
    }
    if edge_bias <= EDGE_BIAS_THRESHOLD {
        return Some((true, area, edge_bias));
    }
    if edge_bias >= CENTER_BIAS_THRESHOLD {
        return Some((false, area, edge_bias));
    }
    None
}

/// Test whether `inner` is fully contained in `outer` with a margin (in
/// normalized units).
pub fn rect_contains_rect(outer: NormalizedRect, inner: NormalizedRect, margin: f32) -> bool {
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
