AutoClip Design Overview

Summary
- AutoClip discovers an HLS stream from a page URL, buffers audio/video, detects a wake phrase, and renders a vertical clip via FFmpeg.
- Layout decisions can stack face/gameplay or render full-frame based on detected hints (face, reticle, gameplay).
- Configuration is driven by config.env and can be hot-reloaded for live tuning.

Core Flow
1) Discover HLS
   - Headless script (Playwright) or platform-specific HTTP logic resolves a master playlist.
   - Highest-variant stream is selected and periodically refreshed.
2) Buffer
   - RollingBuffer stores recent TS segments for before/after clipping.
   - Buffer length is adjusted for wake detection latency to avoid drift.
3) Wake Detection
   - Whisper listens to stream audio (or mic) for configured wake phrases.
   - On detection, a window of before/after audio+video is cut.
4) Layout Hints
   - Face detection + optional face mesh/pose for headroom framing.
   - Gameplay detection / reticle hints to decide stacked vs full-frame.
   - Low-resource mode skips heavy detection and uses heuristics.
5) Render
   - FFmpeg encodes to a vertical output, optionally stacked.
   - Captions/LLM metadata can be added if enabled.
6) Save
   - Output files are named with an incrementing counter and saved to clips/.

Key Modules
- src/main.rs: Orchestration, CLI/env parsing, main pipeline, and FFmpeg render entrypoints.
- src/clip_detect.rs: Face, mesh, pose, and gameplay detection; layout hint extraction.
- src/clip_layout.rs: Layout decisions, crop math, tracking/Kalman smoothing.
- src/rolling_buffer.rs: Rolling buffer for segments.
- src/stream_audio_wake.rs: Whisper wake-word detection and audio handling.

Configuration
- config.env is the primary live config file.
- env vars can be overridden via CLI flags (lowercase + dash).

Performance/Resource Modes
- CLIP_LOW_RESOURCES lowers detection budgets and disables heavy features.
- GPU use is preferred for Whisper and ONNX runtime when available and allowed.
