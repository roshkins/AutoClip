# AutoClip — Voice-Activated Stream Clipper

AutoClip watches a live stream (Kick, Twitch, or TikTok), listens for a wake
phrase, and saves a vertical-format MP4 clip whenever the phrase is heard. The
saved clip stacks a tracked face crop over the gameplay region so it reads
well on phone screens.

## Inputs

CLI arg or `CLIP_PAGE_URL` env: the streamer page to watch. Everything else is
configured through CLI flags and/or `config.env` (see README + the `Config`
struct in `src/main.rs`). Notable knobs:

- `activation_phrase` / `CLIP_WAKE_WORDS` — wake-word(s) for the Whisper
  trigger.
- The live trigger path currently fixes the intended clip window at 50
  seconds before the trigger and 10 seconds after it. The `Config` fields
  `before_buffer_length` / `after_buffer_length` are used by other demo
  paths; they do not currently override that live window.
- `resolution` — output canvas, default `1080x1920` (vertical 9:16).
- `save_path` / `file_name_stub` — output directory and filename prefix; the
  binary appends an incrementing index.
- `vram_allocation`, `FFMPEG_MIN_FREE_VRAM_MB`, `WHISPER_MIN_FREE_VRAM_MB` —
  GPU VRAM guards. AutoClip falls back to CPU when VRAM is below threshold.
- `use_mic_for_wake` / `MIC_DEVICE` — listen to a local microphone instead of
  stream audio (useful for testing without a live stream).

## Output

A vertical MP4 saved to `save_path/{file_name_stub}_{NNN}.mp4`. The optional
LLM-titler renames the file based on the transcript when
`CLIP_LLM_ENABLE=1`.

## Pipeline

1. **Page → m3u8.** A headless Node + Playwright script (`scripts/capture_m3u8.js`
   or `_tiktok.js`) navigates the streamer page and prints the captured m3u8
   URL. `HlsClient::fetch_master_from_page_with_headers` returns the master
   playlist plus the request headers (Referer / Origin / Cookie) needed to
   fetch segments.
2. **Variant pick.** `HlsClient::highest_variant_url` chooses the highest
   resolution variant from the master playlist.
3. **Rolling buffer.** `RollingBuffer` keeps the most recent TS segments
   bounded by wall-clock duration; size = clip window + latency headroom.
4. **Wake detect.** Whisper runs in a worker (`stream_audio_wake.rs`), either
   tailing the HLS stream or a microphone (via FFmpeg). When `WHISPER_ISOLATE=1`
   the worker runs as a separate process talking to the main loop through a
   status file — this isolates crashes from the orchestrator.
5. **Trigger.** When the wake phrase fires, the main loop collects the
   10-second post-trigger tail, then snapshots the relevant buffer contents.
6. **Detect layout.** `clip_detect.rs` runs face detection (Tract or ORT),
   optional FaceMesh, optional MoveNet pose, and optional ArcFace identity
   gating to pick a face crop and gameplay region.
7. **Encode.** `clip_layout.rs` builds an FFmpeg filter graph (stacked /
   face-only / full-frame variants), with optional NVENC acceleration and
   loudnorm audio. Output is the requested vertical resolution.
8. **Save + free.** The clip is written to disk, the buffer copy is dropped,
   and the orchestrator resumes ingest. Optional LLM titling renames the file.
9. **Playlist resilience.** If the m3u8 URL expires (signed-URL drift) or the
   stream goes offline, the orchestrator periodically re-fetches the master
   playlist; offline streams are retried up to `CLIP_STREAM_OFFLINE_SECS`.

## Concurrency model

- Tokio runtime for HLS ingest, ffmpeg invocations, headless calls.
- A separate OS thread monitors observed audio→now latency and grows the
  rolling buffer target if real latency exceeds the configured headroom.
- A separate OS thread (or separate process under `WHISPER_ISOLATE`) runs
  Whisper inference and stamps `detect_ns` / `audio_ns` atomics.
- Post-process tasks are spawned as `tokio::task` join handles so the
  orchestrator can pipeline a new clip while the previous one encodes.

## VRAM management

- `gpu::query_nvidia_free_vram_mb` polls `nvidia-smi` before claiming GPU work.
- Whisper, ORT face/mesh/pose, and FFmpeg NVENC each have a configurable
  minimum free-VRAM threshold; the path falls back to CPU when not met.
- A singleton `GpuLease` mutex (in `gpu.rs`) serializes participating Whisper
  GPU work within one process. It does not coordinate separate worker
  processes or provide a global reservation for FFmpeg/ONNX workloads.

## Out of scope

- Audio-only clipping (the design has always been video).
- Real-time uploads — AutoClip writes to disk only; pushing to YouTube /
  TikTok / Twitter is a future concern.
- Cloud / remote stream sources beyond what the headless capture supports.

## See also

- `BUILD_NOTES.md` — CUDA build pitfalls on Windows.
- `docs/clip-length-guardrails.md` — never shorten output clips from internal
  buffer estimates; always probe TS PTS with ffprobe.
- `README.md` — project overview, development disclosure and build/run steps.
- `docs/code-walkthrough.md` — source reading path and test coverage boundaries.
