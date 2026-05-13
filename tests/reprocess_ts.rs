//! Integration test for the `reprocess-ts` subcommand.
//!
//! Drop a small TS file at `tests/data/sample.ts` to enable this test. When
//! the fixture is missing the test prints a skip notice and passes — that
//! way the CI green path doesn't require a binary fixture committed to the
//! repo, but a developer who copies a real TS into `tests/data/` gets a
//! meaningful regression check for free.
//!
//! Asserts after a successful run:
//!   1. `autoclip reprocess-ts <fixture>` exits 0.
//!   2. The produced .mp4 exists and is non-empty.
//!   3. Clip duration via `ffprobe` matches the source TS within +/- 0.5s.
//!
//! Requires `ffmpeg`/`ffprobe` on PATH and a working build (`cargo build`).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture_path() -> PathBuf {
    workspace_root().join("tests").join("data").join("sample.ts")
}

fn binary_path() -> PathBuf {
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace_root().join("target"));
    // Tests are typically run via `cargo test`, which builds the `debug` profile.
    target_dir
        .join("debug")
        .join(if cfg!(windows) { "autoclip.exe" } else { "autoclip" })
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
fn reprocess_ts_produces_clip_matching_source_duration() {
    let fixture = fixture_path();
    if !fixture.exists() {
        eprintln!(
            "[skip] reprocess_ts integration test — no fixture at {}.\n       \
             Drop a small TS (<= 30s) at that path to enable.",
            fixture.display()
        );
        return;
    }

    let bin = binary_path();
    if !bin.exists() {
        eprintln!(
            "[skip] reprocess_ts integration test — binary not built at {}.\n       \
             Run `cargo build` first.",
            bin.display()
        );
        return;
    }

    // ffprobe must be available to validate.
    if ffprobe_duration_secs(&fixture).is_none() {
        eprintln!(
            "[skip] reprocess_ts integration test — ffprobe missing or fixture invalid"
        );
        return;
    }

    let source_secs = ffprobe_duration_secs(&fixture).expect("source duration");

    // Run reprocess-ts. autoclip writes the output next to the input, with the
    // counter file in the same directory as the TS, so we sandbox via a temp dir.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let sandbox = std::env::temp_dir().join(format!("autoclip_reprocess_test_{stamp}"));
    std::fs::create_dir_all(&sandbox).expect("create sandbox");

    let sandboxed_ts = sandbox.join("clip_001.ts");
    std::fs::copy(&fixture, &sandboxed_ts).expect("copy fixture into sandbox");

    let status = Command::new(&bin)
        .arg("reprocess-ts")
        .arg(&sandboxed_ts)
        .env("SKIP_CLIP_SAVE", "")
        // Avoid live-config polling racing with the test.
        .env("CLIP_LIVE_CONFIG", "")
        .status()
        .expect("run autoclip reprocess-ts");
    assert!(status.success(), "reprocess-ts exited with {:?}", status);

    // The reprocessed clip lives in the same directory with .mp4 extension and
    // the same stem; autoclip's renamer may add a suffix but the base is stable.
    let mut produced = None;
    if let Ok(entries) = std::fs::read_dir(&sandbox) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("mp4") {
                produced = Some(p);
                break;
            }
        }
    }
    let produced = produced.expect("expected an mp4 next to the TS fixture");
    let size = std::fs::metadata(&produced).map(|m| m.len()).unwrap_or(0);
    assert!(size > 0, "produced clip is empty");

    let out_secs = ffprobe_duration_secs(&produced)
        .expect("ffprobe should read the produced mp4");
    let drift = (out_secs - source_secs).abs();
    assert!(
        drift < 0.5,
        "duration drift {drift:.3}s exceeds 0.5s tolerance (src={source_secs:.3}, out={out_secs:.3})"
    );

    // Best-effort cleanup.
    let _ = std::fs::remove_dir_all(&sandbox);
    let _ = Duration::from_secs(0); // silence unused-import lint if we drop time below
}
