//! MPEG-TS parsing helpers.
//!
//! Each TS segment is a sequence of 188-byte packets. The orchestrator uses
//! these helpers to recover real wall-clock durations from a TS segment
//! independently of playlist-declared durations, which can drift on live HLS
//! streams. Audio PTS is preferred over video PTS; PCR is the fallback when
//! no PES packet headers are present.
//!
//! All entry points are pure — give them a TS payload, get back a duration
//! or `None`. They are unit-tested in this module.

use std::time::Duration;

pub const TS_PACKET_SIZE: usize = 188;
pub const PTS_HZ: f64 = 90_000.0;
pub const PTS_WRAP: u64 = 1 << 33;
pub const PCR_HZ: f64 = 27_000_000.0;
pub const PCR_WRAP: u64 = (1 << 33) * 300;

/// Prefer the PTS-derived duration when valid, otherwise fall back to the
/// playlist's declared duration. Keeps live clip lengths anchored to real
/// wall-clock spans.
pub fn choose_segment_duration(pts: Option<Duration>, playlist: Duration) -> Duration {
    if let Some(pts_dur) = pts {
        let secs = pts_dur.as_secs_f32();
        if secs.is_finite() && secs > 0.0 {
            return pts_dur;
        }
    }
    playlist
}

/// Best-effort PTS span recovery: tries audio PTS first, then video, then
/// PCR. Returns `None` if neither a usable span is present.
pub fn pts_duration_from_ts(data: &[u8]) -> Option<Duration> {
    if let Some((start, end)) = pts_span_from_ts(data) {
        if let Some(dur) = duration_from_span(start, end, PTS_WRAP, PTS_HZ) {
            return Some(dur);
        }
    }
    pcr_duration_from_ts(data)
}

pub fn pts_span_from_ts(data: &[u8]) -> Option<(u64, u64)> {
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

pub fn pcr_duration_from_ts(data: &[u8]) -> Option<Duration> {
    let (start, end) = pcr_span_from_ts(data)?;
    duration_from_span(start, end, PCR_WRAP, PCR_HZ)
}

pub fn pcr_span_from_ts(data: &[u8]) -> Option<(u64, u64)> {
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

pub fn duration_from_span(start: u64, end: u64, wrap: u64, hz: f64) -> Option<Duration> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_segment_duration_prefers_pts() {
        let pts = Some(Duration::from_secs(6));
        let playlist = Duration::from_secs(4);
        assert_eq!(choose_segment_duration(pts, playlist), Duration::from_secs(6));
    }

    #[test]
    fn choose_segment_duration_falls_back_to_playlist() {
        assert_eq!(
            choose_segment_duration(None, Duration::from_secs(4)),
            Duration::from_secs(4)
        );
        // Zero-second PTS is not "valid" — fall back.
        assert_eq!(
            choose_segment_duration(Some(Duration::ZERO), Duration::from_secs(4)),
            Duration::from_secs(4)
        );
    }

    #[test]
    fn duration_from_span_handles_wrap() {
        // 1 second @ 90kHz.
        assert_eq!(
            duration_from_span(0, 90_000, PTS_WRAP, PTS_HZ),
            Some(Duration::from_secs(1))
        );
        // Wrap-around: end is small, start is near wrap.
        let near_wrap = PTS_WRAP - 45_000;
        let after_wrap = 45_000u64;
        assert_eq!(
            duration_from_span(near_wrap, after_wrap, PTS_WRAP, PTS_HZ),
            Some(Duration::from_secs(1))
        );
    }

    #[test]
    fn duration_from_span_rejects_zero_delta() {
        assert!(duration_from_span(0, 0, PTS_WRAP, PTS_HZ).is_none());
    }

    #[test]
    fn stream_id_classification() {
        assert!(is_audio_stream_id(0xC0));
        assert!(is_audio_stream_id(0xDF));
        assert!(!is_audio_stream_id(0xBF));
        assert!(is_video_stream_id(0xE0));
        assert!(!is_video_stream_id(0xDF));
    }

    #[test]
    fn parse_pts_recovers_known_value() {
        // Build a synthetic 5-byte PTS header for value 0x123456789.
        // PTS layout (33 bits across 5 bytes):
        //   byte0: 0010 PPP1  (top 3 bits + marker)
        //   byte1: PPPPPPPP
        //   byte2: PPPPPPPM
        //   byte3: PPPPPPPP
        //   byte4: PPPPPPPM
        let target: u64 = 0x123456789;
        let b0 = 0x20 | (((target >> 30) & 0x07) as u8) << 1 | 0x01;
        let b1 = ((target >> 22) & 0xFF) as u8;
        let b2 = (((target >> 15) & 0x7F) as u8) << 1 | 0x01;
        let b3 = ((target >> 7) & 0xFF) as u8;
        let b4 = (((target) & 0x7F) as u8) << 1 | 0x01;
        let bytes = [b0, b1, b2, b3, b4];
        assert_eq!(parse_pts(&bytes), Some(target));
    }

    #[test]
    fn find_ts_sync_locates_first_aligned_pair() {
        // Two sync bytes 188 apart starting at offset 2.
        let mut data = vec![0u8; TS_PACKET_SIZE * 3];
        data[2] = 0x47;
        data[2 + TS_PACKET_SIZE] = 0x47;
        assert_eq!(find_ts_sync(&data), Some(2));
        // No sync at all.
        let zeros = vec![0u8; TS_PACKET_SIZE * 2];
        assert_eq!(find_ts_sync(&zeros), None);
    }
}
