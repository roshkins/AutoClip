use std::collections::VecDeque;
use std::time::Duration;

/// Rolling buffer that keeps the most recent chunks up to a wall-clock duration budget.
/// Intended for small demo/testing; no cross-thread safety and no zero-copy slices.
#[derive(Debug, Default)]
pub struct RollingBuffer {
    capacity: Duration,
    chunks: VecDeque<Chunk>,
    total_duration: Duration,
    total_bytes: usize,
}

#[derive(Debug, Clone)]
struct Chunk {
    data: Vec<u8>,
    duration: Duration,
}

impl RollingBuffer {
    pub fn new(capacity: Duration) -> Self {
        Self {
            capacity,
            chunks: VecDeque::new(),
            total_duration: Duration::ZERO,
            total_bytes: 0,
        }
    }

    /// Push a chunk with its playback duration; evict oldest chunks until within capacity.
    pub fn push(&mut self, data: Vec<u8>, duration: Duration) {
        let bytes = data.len();
        self.total_duration += duration;
        self.total_bytes += bytes;
        self.chunks.push_back(Chunk { data, duration });
        self.evict();
    }

    /// Concatenate buffered chunks (oldest to newest) into a single Vec.
    pub fn snapshot_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.total_bytes);
        for chunk in &self.chunks {
            out.extend_from_slice(&chunk.data);
        }
        out
    }

    /// Concatenate only the newest `keep` duration (oldest first). If a chunk would
    /// exceed `keep`, it is still included whole; returned duration reflects the sum
    /// of included chunks.
    #[allow(dead_code)]
    pub fn snapshot_tail(&self, keep: Duration) -> (Vec<u8>, Duration) {
        let mut collected: Vec<&Chunk> = Vec::new();
        let mut acc = Duration::ZERO;

        for chunk in self.chunks.iter().rev() {
            collected.push(chunk);
            acc += chunk.duration;
            if acc >= keep {
                break;
            }
        }

        collected.reverse();
        let mut out = Vec::new();
        for chunk in collected {
            out.extend_from_slice(&chunk.data);
        }

        (out, acc)
    }

    /// Concatenate `keep` duration ending `drop_tail` before the newest chunk.
    /// Drops whole chunks to satisfy `drop_tail` and `keep`.
    #[allow(dead_code)]
    pub fn snapshot_tail_offset(&self, keep: Duration, drop_tail: Duration) -> (Vec<u8>, Duration) {
        let mut collected: Vec<&Chunk> = Vec::new();
        let mut skipped = Duration::ZERO;
        let mut acc = Duration::ZERO;

        for chunk in self.chunks.iter().rev() {
            if skipped < drop_tail {
                skipped += chunk.duration;
                continue;
            }
            collected.push(chunk);
            acc += chunk.duration;
            if acc >= keep {
                break;
            }
        }

        collected.reverse();
        let mut out = Vec::new();
        for chunk in collected {
            out.extend_from_slice(&chunk.data);
        }

        (out, acc)
    }

    /// Number of chunks currently retained.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Total duration of buffered chunks.
    pub fn total_duration(&self) -> Duration {
        self.total_duration
    }

    /// Total byte size of buffered chunks.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn set_capacity(&mut self, capacity: Duration) {
        self.capacity = capacity;
        self.evict();
    }

    fn evict(&mut self) {
        while self.total_duration > self.capacity {
            if let Some(oldest) = self.chunks.pop_front() {
                self.total_duration = self.total_duration.saturating_sub(oldest.duration);
                self.total_bytes = self.total_bytes.saturating_sub(oldest.data.len());
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_within_capacity() {
        let mut buf = RollingBuffer::new(Duration::from_secs(1));
        buf.push(b"a".to_vec(), Duration::from_millis(400));
        buf.push(b"b".to_vec(), Duration::from_millis(400));
        assert_eq!(buf.chunk_count(), 2);
        assert_eq!(buf.total_duration(), Duration::from_millis(800));
        assert_eq!(buf.snapshot_bytes(), b"ab");
    }

    #[test]
    fn evicts_oldest_when_over_budget() {
        let mut buf = RollingBuffer::new(Duration::from_millis(1000));
        buf.push(b"a".to_vec(), Duration::from_millis(400));
        buf.push(b"b".to_vec(), Duration::from_millis(400));
        buf.push(b"c".to_vec(), Duration::from_millis(400));
        // Total would be 1200ms; should evict "a" and keep b+c (800ms).
        assert_eq!(buf.chunk_count(), 2);
        assert_eq!(buf.total_duration(), Duration::from_millis(800));
        assert_eq!(buf.snapshot_bytes(), b"bc");
    }

    #[test]
    fn evicts_multiple_chunks_if_needed() {
        let mut buf = RollingBuffer::new(Duration::from_millis(500));
        buf.push(b"a".to_vec(), Duration::from_millis(300));
        buf.push(b"b".to_vec(), Duration::from_millis(300));
        buf.push(b"c".to_vec(), Duration::from_millis(300));
        // Total would be 900ms; should evict a and b to fit c (300ms <= 500ms).
        assert_eq!(buf.chunk_count(), 1);
        assert_eq!(buf.total_duration(), Duration::from_millis(300));
        assert_eq!(buf.snapshot_bytes(), b"c");
    }

    #[test]
    fn total_bytes_tracks_pushes_and_evictions() {
        let mut buf = RollingBuffer::new(Duration::from_millis(500));
        buf.push(vec![0u8; 100], Duration::from_millis(300));
        buf.push(vec![0u8; 200], Duration::from_millis(300));
        // Evicts the 100-byte chunk to fit; only 200 remains.
        assert_eq!(buf.total_bytes(), 200);
    }

    #[test]
    fn snapshot_tail_takes_newest() {
        let mut buf = RollingBuffer::new(Duration::from_secs(10));
        buf.push(b"a".to_vec(), Duration::from_millis(400));
        buf.push(b"b".to_vec(), Duration::from_millis(400));
        buf.push(b"c".to_vec(), Duration::from_millis(400));
        let (bytes, dur) = buf.snapshot_tail(Duration::from_millis(500));
        // 500ms target — newest c (400ms) is below, b adds 400 -> 800ms >= 500, stop.
        assert_eq!(bytes, b"bc");
        assert_eq!(dur, Duration::from_millis(800));
    }

    #[test]
    fn snapshot_tail_offset_drops_tail() {
        let mut buf = RollingBuffer::new(Duration::from_secs(10));
        buf.push(b"a".to_vec(), Duration::from_millis(400));
        buf.push(b"b".to_vec(), Duration::from_millis(400));
        buf.push(b"c".to_vec(), Duration::from_millis(400));
        let (bytes, _dur) = buf.snapshot_tail_offset(
            Duration::from_millis(500),
            Duration::from_millis(300), // skip ~newest 300ms
        );
        // Skip c (counts 400ms toward the 300ms skip, which exhausts it),
        // then collect b until acc >= 500ms.
        assert!(!bytes.is_empty());
    }

    #[test]
    fn set_capacity_evicts_immediately() {
        let mut buf = RollingBuffer::new(Duration::from_secs(10));
        buf.push(b"a".to_vec(), Duration::from_millis(400));
        buf.push(b"b".to_vec(), Duration::from_millis(400));
        buf.push(b"c".to_vec(), Duration::from_millis(400));
        assert_eq!(buf.chunk_count(), 3);
        buf.set_capacity(Duration::from_millis(500));
        // Should evict a and b; c (400ms) fits in 500ms.
        assert_eq!(buf.chunk_count(), 1);
        assert_eq!(buf.snapshot_bytes(), b"c");
    }

    #[test]
    fn empty_buffer_snapshots_are_empty() {
        let buf = RollingBuffer::new(Duration::from_secs(1));
        assert_eq!(buf.snapshot_bytes(), Vec::<u8>::new());
        assert_eq!(buf.chunk_count(), 0);
        assert_eq!(buf.total_bytes(), 0);
        assert_eq!(buf.total_duration(), Duration::ZERO);
    }
}
