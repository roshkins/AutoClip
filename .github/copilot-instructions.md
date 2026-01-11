# Copilot Instructions for AutoClip

These notes help AI agents work effectively in this repository. They summarize the current design and expected behaviors; adjust as real code lands.

## Purpose & Flow
- Goal: voice-activated streamer clipper that saves a vertical, resized video file.
- Inputs: `kick_url`, `activation_phrase`, `before_buffer_length`, `after_buffer_length`, `resolution`, `vram_allocation`, `save_path`, `file_name_stub`.
- Outputs: processed clip file (design doc currently says MP3; likely MP4/vertical video—confirm).
- Pipeline: fetch m3u8 playlist → pick highest-res stream → continuously buffer; CPU-bound Whisper listens for wake phrase; on trigger, wait `after_buffer_length`, deep-copy buffer, enqueue post-process; post-process via FFmpeg to vertical format; save to `save_path` with incrementing filename; free copy; if playlist ends, periodically refresh or detect changed m3u8 URI.

## Architecture Expectations (Rust)
- Primary language: Rust. Expect modules for streaming ingest (m3u8/HLS), audio transcription (Whisper CPU), buffering with VRAM-to-RAM spillover, FFmpeg-driven transcode, and storage naming.
- Concurrency: likely async runtime (Tokio) for streaming/buffering and a worker queue for post-processing tasks.
- Resource management: use `vram_allocation` to cap GPU buffer; spill to CPU RAM when exceeded.
- Trigger handling: wake-word detection gates clip creation; ensure `before_buffer_length`/`after_buffer_length` windowing is honored.

## Conventions to Follow
- Configuration surface should mirror the inputs listed above; keep naming consistent across CLI/env/config structs.
- Filenames: prefix with `file_name_stub`, append incrementing counter; ensure directory exists under `save_path`.
- Video processing: enforce vertical resize/crop in FFmpeg; keep source highest available resolution; avoid re-encoding audio unless required.
- Resilience: on playlist end, implement retry/refresh logic before aborting; handle m3u8 URL changes.

## External Dependencies (likely)
- FFmpeg for transcode/resize.
- Whisper (CPU) for wake-word detection; plan for model asset management and thread pinning.
- HLS/m3u8 client for stream ingest.

## Implementation Hints
- Buffer design: rolling buffer sized by `before_buffer_length`; deep-copy on trigger to decouple post-processing latency from ingest.
- Queueing: background worker processes copies sequentially or in bounded concurrency to avoid VRAM spikes.
- Metrics/logging: log trigger events, playlist refresh attempts, FFmpeg failures, and save destinations.

## Workflow & Testing
- Keep changes small and incremental; prefer a series of tiny edits over large batches.
- After each change, run the fastest applicable checks (unit/linters once added); fix issues immediately.
- Run the integration test suite after every code change to verify no regressions; if commands are not defined yet, ask for or add a canonical `cargo test` (or project-specific) integration target and document it.
- Keep comments in sync with behavior. When code paths change (e.g., swapping HTML scraping for headless m3u8 capture), update inline docs and module headers in the same PR.
- When adding or changing flags/flows, update the CLI help output (`-h/--help`) in `src/main.rs` in the same change.

## Jujutsu (jj) Workflow
- Working copy is always a commit; each change has a stable change ID and a rewritable commit hash—prefer change IDs in commands.
- Common commands: `jj status` (or `jj st`), `jj log` (revsets like `@`, `root()`, `bookmarks()`, `@::trunk()`), `jj diff`, `jj describe -m`, `jj new` to start a fresh change, `jj squash`/`jj squash -i` to move edits into parents, `jj rebase -s <rev> -o <dest>` to move a stack.
- No `git add`: edits auto-amend the working-copy commit; use `.gitignore` plus `jj file untrack <path>` to stop tracking.
- Conflicts: rebases complete even with conflicts; resolve on a child commit, then `jj squash` into the conflicted one. `jj resolve` or manual markers both work.
- Operation log: `jj op log` shows history of operations; `jj undo` reverts the last operation (including rebases/squashes) and updates the working copy.
- Git interop: clone with `jj git clone`; bookmarks map to Git branches. Use `git status`/`git push` as usual; keep `jj` and Git views aligned.

## If Something Is Missing
- This repo currently only contains the design doc ([DESIGN_DOC.md](../DESIGN_DOC.md)). Ask for details on build/run/test commands, directory layout, and target platforms before proceeding.
