//! Integration test for the `reprocess-ts` subcommand.
//!
//! Generate a fixture with `scripts/generate_test_fixture.ps1`, then run:
//! `cargo test --no-default-features --test reprocess_ts -- --ignored`.
//! The test is explicitly ignored by default. When requested, missing media
//! or tools are failures instead of a successful test that did no rendering.
//!
//! Asserts after a successful run:
//!   1. `autoclip reprocess-ts <fixture>` exits 0.
//!   2. The produced .mp4 exists and is non-empty.
//!   3. Clip duration via `ffprobe` matches the source TS within +/- 0.5s.
//!
//! Requires `ffmpeg`/`ffprobe` on PATH. Cargo supplies the matching binary.

use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture_path() -> PathBuf {
    workspace_root()
        .join("tests")
        .join("data")
        .join("sample.ts")
}

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_autoclip"))
}

fn ffprobe_duration_secs(path: &Path) -> Option<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim().parse::<f64>().ok()
}

#[test]
#[ignore = "requires a local TS fixture and FFmpeg; see tests/data/README.md"]
fn reprocess_ts_produces_clip_matching_source_duration() {
    let fixture = fixture_path();
    assert!(
        fixture.exists(),
        "Generate the fixture first: scripts/generate_test_fixture.ps1"
    );

    let bin = binary_path();
    assert!(
        bin.exists(),
        "Cargo-built autoclip binary is missing: {}",
        bin.display()
    );
    let source_secs = ffprobe_duration_secs(&fixture)
        .expect("ffprobe must be on PATH and the fixture must contain valid media");

    // reprocess-ts writes to ./clips, relative to the process working directory.
    // Use a temp directory so output and live config cannot touch the checkout.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let sandbox = std::env::temp_dir().join(format!("autoclip_reprocess_test_{stamp}"));
    std::fs::create_dir_all(&sandbox).expect("create sandbox");

    let sandboxed_ts = sandbox.join("clip_001.ts");
    std::fs::copy(&fixture, &sandboxed_ts).expect("copy fixture into sandbox");
    let config = sandbox.join("test-config.env");
    std::fs::write(
        &config,
        concat!(
            "CLIP_LAYOUT=full\nCLIP_DETECT=0\nCLIP_GAMEPLAY=0\n",
            "CLIP_FACE_MESH=0\nCLIP_POSE=0\nCLIP_CAPTIONS=0\n",
            "CLIP_CLOSED_CAPTIONS=0\nCLIP_LLM_ENABLE=0\n",
            "FFMPEG_HWACCEL=none\nFFMPEG_ENCODER=libx264\n"
        ),
    )
    .expect("write CPU-only test config");

    let status = Command::new(&bin)
        .current_dir(&sandbox)
        .arg("reprocess-ts")
        .arg(&sandboxed_ts)
        .env("CLIP_LIVE_CONFIG", &config)
        .status()
        .expect("run autoclip reprocess-ts");
    assert!(status.success(), "reprocess-ts exited with {:?}", status);

    // The application stores new output under the sandbox's clips directory.
    let mut produced = None;
    if let Ok(entries) = std::fs::read_dir(sandbox.join("clips")) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("mp4") {
                produced = Some(p);
                break;
            }
        }
    }
    let produced = produced.expect("expected an mp4 in the sandbox clips directory");
    let size = std::fs::metadata(&produced).map(|m| m.len()).unwrap_or(0);
    assert!(size > 0, "produced clip is empty");

    let out_secs = ffprobe_duration_secs(&produced).expect("ffprobe should read the produced mp4");
    let drift = (out_secs - source_secs).abs();
    assert!(
        drift < 0.5,
        "duration drift {drift:.3}s exceeds 0.5s tolerance (src={source_secs:.3}, out={out_secs:.3})"
    );

    // Best-effort cleanup.
    let _ = std::fs::remove_dir_all(&sandbox);
}
