# Code walkthrough

AutoClip is an experimental media pipeline implemented in Rust, Tokio and
Node.js. It was entirely vibe coded with AI tools. This guide points to
concrete code and tests so a reviewer can assess the implementation directly.

## Follow one clip

```text
Stream page
  -> HLS discovery / playlist refresh       src/hls.rs + scripts/capture_m3u8*.js
  -> Fetch segments / retain recent video  src/main.rs + src/rolling_buffer.rs
  -> Whisper wake event                    src/stream_audio_wake.rs
  -> Collect tail / snapshot buffer        AutoClip::run_until_wake_and_clip
  -> Detect layout / encode MP4            src/clip_detect.rs + src/clip_layout.rs
```

Start with the [rolling buffer](../src/rolling_buffer.rs): it owns a
`VecDeque` of byte chunks, tracks their durations and byte count, evicts the
oldest chunks, and can concatenate a snapshot. Tests cover eviction,
snapshots and resizing. Its capacity is a duration budget; it is not a
hard memory bound or a shared concurrent queue.

Next read [HlsClient](../src/hls.rs). It uses async `reqwest` requests with
a 15-second client timeout, chooses stream variants and tracks signed URL
expiry. The orchestration in [main.rs](../src/main.rs) handles stale/offline
playlists and reacquires stream information. Tests exercise URL normalization,
variant selection and expiry parsing without needing a live channel.

Finally, follow `AutoClip::run_until_wake_and_clip` in `main.rs`: ingestion,
buffer warm-up, wake detection, collection of the post-trigger tail and
the spawned save task. The live path currently fixes the intended window
at 50 seconds before the trigger plus 10 seconds after it, with additional
buffer headroom for inference latency.

## Async and resource boundaries

| Boundary | Mechanism | Tradeoff to inspect |
| --- | --- | --- |
| Network and external processes | Tokio runtime, async HTTP and process APIs | Async I/O shares the runtime with coordination work. |
| Wake inference | OS worker thread; optional separate worker process | Audio timing and wake state cross the boundary through atomics or status files. |
| Inference lag | A monitor thread increases the buffer target using observed latency | Extra history helps preserve the requested clip, but also increases retained data. |
| Multi-stream admission | `spawn_stream_task` acquires an owned semaphore permit | Bounds active stream tasks; does not bound every render spawned by a stream. |
| Clip saves | Tokio tasks hold owned snapshots while ingestion continues | Concurrent output avoids waiting for each encode, but costs extra memory. |
| GPU use | Free-VRAM checks and a process-local `GpuLease` mutex | Availability checks are advisory; the mutex does not coordinate separate processes. |

These are implementation details to review, not measured performance
results or guarantees of production reliability. The code includes
synchronous filesystem/process calls and several synchronization mechanisms;
there is room to simplify the orchestrator and tighten resource bounds.

## Timing and correctness

[ts.rs](../src/ts.rs) parses MPEG-TS presentation and clock timestamps,
handles wraparound and selects a segment duration. Its tests cover known
timestamp values, wraparound and fallback behavior.

[clip_layout.rs](../src/clip_layout.rs) constructs FFmpeg filter graphs.
Tests check crop bounds, even output dimensions and stacked layouts.
The [duration guardrails](clip-length-guardrails.md) explain why actual
media timestamps matter when reprocessing saved segments.

## What the checks establish

```powershell
cargo test --locked --no-default-features
cargo run --locked --no-default-features -- --help
```

These commands check the build and deterministic utility tests without
CUDA or downloaded model assets. Two network tests only perform requests
when `M3U8_TEST_URL` or `M3U8_PAGE_URL` is supplied. A passing default suite
therefore does not establish live platform compatibility.

The [media integration test](../tests/reprocess_ts.rs) is explicitly ignored
by default. To exercise a real local TS-to-MP4 render, follow the
[README instructions](../README.md#run-a-media-regression-check). It checks
duration preservation with CPU encoding, without claiming to validate
Whisper, face inference or the entire live pipeline.
