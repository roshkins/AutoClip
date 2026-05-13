use std::process::Command;
#[cfg(feature = "whisper")]
use std::sync::{Mutex, OnceLock};

fn parse_nvidia_smi_entries(stdout: &str) -> Vec<(u32, u64)> {
    let mut entries = Vec::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 2 {
            if let (Ok(idx), Ok(mem)) = (parts[0].parse::<u32>(), parts[1].parse::<u64>()) {
                entries.push((idx, mem));
            }
        }
    }
    entries
}

fn parse_nvidia_smi_csv(stdout: &str, device: Option<u32>) -> Option<u64> {
    let mut entries = parse_nvidia_smi_entries(stdout);
    if entries.is_empty() {
        return None;
    }
    if let Some(target) = device {
        return entries
            .iter()
            .find(|(idx, _)| *idx == target)
            .map(|(_, mem)| *mem);
    }
    if let Some((_, mem)) = entries.iter().find(|(idx, _)| *idx == 0) {
        return Some(*mem);
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries.first().map(|(_, mem)| *mem)
}

pub fn query_nvidia_free_vram_mb(device: Option<u32>) -> Option<u64> {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=index,memory.free")
        .arg("--format=csv,noheader,nounits")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_nvidia_smi_csv(&stdout, device)
}

#[cfg(feature = "ort")]
pub fn query_nvidia_free_vram_all() -> Option<Vec<(u32, u64)>> {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=index,memory.free")
        .arg("--format=csv,noheader,nounits")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let entries = parse_nvidia_smi_entries(&stdout);
    if entries.is_empty() {
        None
    } else {
        Some(entries)
    }
}

#[cfg(feature = "ort")]
pub fn pick_best_nvidia_device(min_free_mb: u64, label: &str) -> Option<u32> {
    let mut entries = query_nvidia_free_vram_all()?;
    entries.retain(|(_, mem)| *mem >= min_free_mb);
    if entries.is_empty() {
        eprintln!("gpu vram guard: no GPU meets {min_free_mb} MB for {label}");
        return None;
    }
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries.first().map(|(idx, _)| *idx)
}

pub fn gpu_vram_allows(min_free_mb: u64, device: Option<u32>, label: &str) -> bool {
    if min_free_mb == 0 {
        return true;
    }
    let Some(free_mb) = query_nvidia_free_vram_mb(device) else {
        eprintln!("gpu vram guard: unable to query VRAM; allowing GPU for {label}");
        return true;
    };
    if free_mb < min_free_mb {
        eprintln!(
            "gpu vram guard: {label} free {free_mb} MB < {min_free_mb} MB; forcing CPU",
        );
        return false;
    }
    true
}

#[cfg(feature = "whisper")]
pub struct GpuLease {
    _guard: std::sync::MutexGuard<'static, ()>,
}

#[cfg(feature = "whisper")]
pub fn try_acquire_gpu_lease(label: &str) -> Option<GpuLease> {
    static GPU_LEASE: OnceLock<Mutex<()>> = OnceLock::new();
    let lock = GPU_LEASE.get_or_init(|| Mutex::new(()));
    match lock.try_lock() {
        Ok(guard) => Some(GpuLease { _guard: guard }),
        Err(err) => {
            eprintln!("gpu lease: unavailable for {label} ({err}); forcing CPU");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nvidia_smi_prefers_device_zero() {
        let out = "0, 1234\n1, 5678\n";
        assert_eq!(parse_nvidia_smi_csv(out, None), Some(1234));
    }

    #[test]
    fn parse_nvidia_smi_picks_requested_device() {
        let out = "0, 1234\n1, 5678\n";
        assert_eq!(parse_nvidia_smi_csv(out, Some(1)), Some(5678));
        assert_eq!(parse_nvidia_smi_csv(out, Some(3)), None);
    }

    #[test]
    fn parse_nvidia_smi_falls_back_to_lowest_index() {
        let out = "2, 900\n4, 800\n";
        assert_eq!(parse_nvidia_smi_csv(out, None), Some(900));
    }

    #[test]
    fn parse_nvidia_smi_rejects_invalid_input() {
        assert_eq!(parse_nvidia_smi_csv("bad\n", None), None);
    }

    #[test]
    fn parse_nvidia_smi_entries_collects_all() {
        let out = "0, 1234\n1, 5678\n";
        assert_eq!(parse_nvidia_smi_entries(out), vec![(0, 1234), (1, 5678)]);
    }

    #[test]
    fn gpu_vram_allows_skips_check_when_disabled() {
        assert!(gpu_vram_allows(0, None, "test"));
    }
}
