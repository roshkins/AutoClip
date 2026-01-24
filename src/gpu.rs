//! GPU helper utilities (VRAM queries + lease guards).
//!
//! These helpers call `nvidia-smi` to decide whether GPU usage is safe and
//! provide a simple lease mechanism to avoid concurrent GPU-heavy workloads.

use std::env;
use std::process::Command;
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

fn parse_override_entries(raw: &str) -> Vec<(u32, u64)> {
    let mut entries = Vec::new();
    for part in raw.split(|c| c == ',' || c == ';') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((idx_raw, mem_raw)) = trimmed.split_once(':') {
            if let (Ok(idx), Ok(mem)) =
                (idx_raw.trim().parse::<u32>(), mem_raw.trim().parse::<u64>())
            {
                entries.push((idx, mem));
            }
        } else if let Ok(mem) = trimmed.parse::<u64>() {
            entries.push((0, mem));
        }
    }
    entries
}

fn override_vram_entries() -> Option<Vec<(u32, u64)>> {
    if let Ok(raw) = env::var("GPU_VRAM_OVERRIDE_LIST") {
        let entries = parse_override_entries(&raw);
        if !entries.is_empty() {
            return Some(entries);
        }
    }
    if let Ok(raw) = env::var("GPU_VRAM_OVERRIDE_MB") {
        let entries = parse_override_entries(&raw);
        if !entries.is_empty() {
            return Some(entries);
        }
    }
    None
}

fn override_vram_mb(device: Option<u32>) -> Option<u64> {
    let entries = override_vram_entries()?;
    if let Some(target) = device {
        return entries
            .iter()
            .find(|(idx, _)| *idx == target)
            .map(|(_, mem)| *mem);
    }
    if let Some((_, mem)) = entries.iter().find(|(idx, _)| *idx == 0) {
        return Some(*mem);
    }
    entries.first().map(|(_, mem)| *mem)
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

/// Query free VRAM (MB) for a given NVIDIA device.
///
/// Returns `None` if `nvidia-smi` is unavailable or parsing fails.
pub fn query_nvidia_free_vram_mb(device: Option<u32>) -> Option<u64> {
    if let Some(mem) = override_vram_mb(device) {
        return Some(mem);
    }
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

/// Query free VRAM (MB) for all NVIDIA devices.
pub fn query_nvidia_free_vram_all() -> Option<Vec<(u32, u64)>> {
    if let Some(entries) = override_vram_entries() {
        return Some(entries);
    }
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

/// Pick the NVIDIA device with the most free VRAM that meets `min_free_mb`.
pub fn pick_best_nvidia_device(min_free_mb: u64, label: &str) -> Option<u32> {
    let reserve = gpu_vram_reserve_mb();
    let required = min_free_mb.saturating_add(reserve);
    let mut entries = query_nvidia_free_vram_all()?;
    entries.retain(|(_, mem)| *mem >= required);
    if entries.is_empty() {
        eprintln!(
            "gpu vram guard: no GPU meets {required} MB (min {min_free_mb} + reserve {reserve}) for {label}"
        );
        return None;
    }
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries.first().map(|(idx, _)| *idx)
}

fn gpu_vram_reserve_mb() -> u64 {
    env::var("GPU_VRAM_RESERVE_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(512)
}

/// Determine if a GPU should be used given a minimum free VRAM threshold.
pub fn gpu_vram_allows(min_free_mb: u64, device: Option<u32>, label: &str) -> bool {
    let reserve = gpu_vram_reserve_mb();
    let required = min_free_mb.saturating_add(reserve);
    if required == 0 {
        return true;
    }
    let Some(free_mb) = query_nvidia_free_vram_mb(device) else {
        eprintln!("gpu vram guard: unable to query VRAM; allowing GPU for {label}");
        return true;
    };
    if free_mb < required {
        eprintln!(
            "gpu vram guard: {label} free {free_mb} MB < {required} MB (min {min_free_mb} + reserve {reserve}); forcing CPU",
        );
        return false;
    }
    true
}

/// Guard object representing an exclusive GPU lease.
pub struct GpuLease {
    _guard: std::sync::MutexGuard<'static, ()>,
}

/// Attempt to acquire a process-wide GPU lease.
///
/// This is a best-effort guard to prevent overlapping GPU-heavy tasks.
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
    use crate::test_util::EnvGuard;

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

    #[test]
    fn override_vram_single_value() {
        let mut env = EnvGuard::new();
        env.set("GPU_VRAM_OVERRIDE_MB", "2048");
        assert_eq!(query_nvidia_free_vram_mb(None), Some(2048));
        assert_eq!(query_nvidia_free_vram_mb(Some(0)), Some(2048));
    }

    #[test]
    fn override_vram_list() {
        let mut env = EnvGuard::new();
        env.set("GPU_VRAM_OVERRIDE_LIST", "0:3000,1:6000");
        assert_eq!(query_nvidia_free_vram_mb(Some(1)), Some(6000));
        let all = query_nvidia_free_vram_all().expect("expected override list");
        assert!(all.iter().any(|(idx, mem)| *idx == 0 && *mem == 3000));
        assert!(all.iter().any(|(idx, mem)| *idx == 1 && *mem == 6000));
    }

    #[test]
    fn vram_reserve_is_enforced() {
        let mut env = EnvGuard::new();
        env.set("GPU_VRAM_OVERRIDE_MB", "1024");
        env.set("GPU_VRAM_RESERVE_MB", "512");
        assert!(gpu_vram_allows(400, None, "test"));
        assert!(!gpu_vram_allows(600, None, "test"));
    }

    #[test]
    fn pick_best_device_respects_reserve() {
        let mut env = EnvGuard::new();
        env.set("GPU_VRAM_OVERRIDE_LIST", "0:3000,1:7000");
        env.set("GPU_VRAM_RESERVE_MB", "512");
        let best = pick_best_nvidia_device(3000, "test").expect("expected device");
        assert_eq!(best, 1);
    }
}
