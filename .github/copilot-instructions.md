# Copilot / AI Agent Instructions for AutoClip

These notes orient AI agents working in this repo. Skim before edits.

## What AutoClip is

A Rust binary that watches a live stream (Kick / Twitch / TikTok), runs
Whisper on the audio to detect a wake phrase, and on trigger saves a
vertical-format MP4 (face cam stacked over gameplay) via FFmpeg.

The whole pipeline is end-to-end working today; refactors should preserve
behavior, not re-invent it. The original "MVP stub" language in the source
is historical — the binary is the real product.

## Layout (top-level)

- `src/main.rs` — orchestration loop, HLS client, CLI parsing, config
  writer, ffmpeg invocations. **Currently ~8k LOC and being decomposed.**
  Pull new logic into a sibling module unless it is genuinely
  orchestration-glue.
- `src/clip_detect.rs` — face / mesh / pose / ID detection (Tract + optional
  ORT). **Currently ~6.8k LOC; deferred deep refactor.** Internal `// =====
  SECTION ... =====` dividers mark the conceptual chunks (config, env helpers,
  YuNet, ORT glue, FaceMesh, Pose, FaceID, debug-dump). Future split target:
  `clip_detect/{config,backend,face,face_ort,mesh,pose,face_id,sweep}.rs`.
  Until then, jump by section dividers; the module docstring at the top of
  the file lists them all.
- `src/clip_gameplay.rs` — CLIP-based gameplay region detection.
- `src/clip_layout.rs` — FFmpeg filter-graph builders.
- `src/stream_audio_wake.rs` (+ `_stub.rs`) — Whisper wake detection. The
  stub mirrors the real public API for `#[cfg(not(feature = "whisper"))]`
  builds; keep them in lockstep.
- `src/rolling_buffer.rs` — bounded TS segment buffer.
- `src/gpu.rs` — `nvidia-smi` queries + GPU lease mutex.
- `src/loading.rs` / `src/profile.rs` — small support modules.
- `scripts/` — Playwright m3u8 capture, model download helpers, build/run
  helpers (PowerShell + .bat).
- `config.env` — hot-reloaded config; the binary polls it while running.

## Conventions

- Anything settable via env var should also be settable via CLI: lowercase
  the env name and replace `_` with `-`, e.g. `CLIP_FACE_RATIO` →
  `--clip-face-ratio=0.45`. Plumb new flags through the existing parser in
  `src/main.rs`.
- Filenames are `{file_name_stub}_{NNN}.mp4` with a hidden counter file in
  `save_path`; use `next_output_path()`.
- **Never shorten clip duration from internal buffer estimates** — see
  `docs/clip-length-guardrails.md`. Probe with ffprobe if you must validate.
- Keep the README, DESIGN_DOC, and `--help` output in sync with code in the
  same change.
- Tract and ORT must both compile; gate ORT-only code with
  `#[cfg(feature = "ort")]` and provide a Tract fallback.

## Build / run

- `cargo check --no-default-features` is the fastest sanity check (skips
  whisper.cpp + ORT). Use this for refactors.
- Full CUDA build: `.\scripts\setup_gpu_build.ps1 -CargoProfile debug`.
- Run prebuilt: `.\run_autoclip_with_cuda.ps1 -NoBuild "<stream-url>"`.
- See `BUILD_NOTES.md` for the Windows CUDA gauntlet (VS dev env, explicit
  cl/rc/mt/nvcc paths, Ninja, `target/release/build/whisper-rs-sys-*` cache
  invalidation).

## Testing

- `cargo test --no-default-features` runs the unit tests under the Tract-only
  path (fast, no whisper.cpp).
- Pure-function targets (RollingBuffer, parse helpers, filter-graph builders)
  should grow tests as logic moves into them.
- For end-to-end checks use the `reprocess-ts` subcommand against a saved TS
  file — no live stream required.

## When making changes

1. Make the smallest edit that solves the problem.
2. Run `cargo check --no-default-features` and (when relevant)
   `cargo test --no-default-features`. Fix warnings before merging.
3. If CLI flags change, update `--help` text and README in the same PR.
4. If detection / framing thresholds change, update `config.env` defaults
   and note the change near the relevant code.

## Jujutsu (jj) workflow

Optional but supported. Working copy is always a commit; prefer change IDs.
`jj status`, `jj log`, `jj describe -m`, `jj new`, `jj squash`,
`jj rebase -s … -o …`. Git interop via `jj git clone` / `git push`.

## When something is missing

Ask. `DESIGN_DOC.md` and `README.md` are the source of truth for intended
behavior; everything else should match.
