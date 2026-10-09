# AutoClip

Voice-triggered livestream clips, formatted for vertical video.

AutoClip watches a live stream, keeps recent video in a rolling buffer, and
uses Whisper to listen for a phrase such as "clip that". A trigger saves the
surrounding footage as an MP4, with optional face/gameplay framing, captions
and transcript-based titles.

**Status:** experimental personal project, primarily developed for Windows
with NVIDIA hardware. The code has paths for Kick, Twitch and TikTok; live
compatibility depends on each platform's current stream access and page layout.

**Development:** this project was entirely vibe coded using AI coding tools.
It is shared as an experiment in AI-assisted product development and
automated media workflows.

## What is interesting in the code

| Area | Implementation | Start here |
| --- | --- | --- |
| Async external integrations | HLS requests, playlist refresh, signed-URL expiry and headless stream discovery | [src/hls.rs](src/hls.rs) |
| Event-driven orchestration | Tokio tasks, wake events, clip saves and a semaphore for multi-stream admission | [src/main.rs](src/main.rs) |
| Stream state | Duration-based segment eviction and snapshots of recent video | [src/rolling_buffer.rs](src/rolling_buffer.rs) |
| Media timing | MPEG-TS PTS/PCR parsing, timestamp wraparound and duration selection | [src/ts.rs](src/ts.rs) |
| Resource tradeoffs | GPU availability checks, inference workers and CPU fallback paths | [src/gpu.rs](src/gpu.rs), [src/stream_audio_wake.rs](src/stream_audio_wake.rs) |
| Output | Face/gameplay detection and FFmpeg filter graphs for vertical layouts | [src/clip_detect.rs](src/clip_detect.rs), [src/clip_layout.rs](src/clip_layout.rs) |

See the [code walkthrough](docs/code-walkthrough.md) for a short reading path,
concurrency boundaries and limitations. The [design document](DESIGN_DOC.md)
describes the full pipeline.

## Check the Rust code without a GPU

Install Rust and the native compiler tools for your platform. On Windows,
use Visual Studio 2022 Build Tools with the C++ workload and Windows SDK.
From the repository root:

```powershell
cargo test --locked --no-default-features
cargo run --locked --no-default-features -- --help
```

This builds the Rust code with Whisper and ONNX Runtime features disabled.
It exercises unit tests for CLI parsing, HLS helpers, the rolling buffer,
timestamps, layouts and other utilities. It does **not** enable live voice
triggers or validate CUDA inference. Network tests require explicit URLs;
the media integration test is opt-in, as described below.

## Run the live pipeline on Windows

The default Cargo features enable CUDA-backed Whisper and dynamic ONNX
Runtime support. A full live run requires more setup than the checks above:

1. Install Rust, Visual Studio 2022 C++ tools, the Windows SDK, CMake,
   Ninja, the CUDA Toolkit and an NVIDIA driver. Put FFmpeg and ffprobe on
   `PATH`. NVENC is optional for video encoding.
2. Install Node.js and the headless browser helper:

   ```powershell
   npm ci
   npx playwright install chromium
   ```

3. Obtain a Whisper **GGML** model using the
   [whisper.cpp model instructions](https://github.com/ggml-org/whisper.cpp/tree/master/models).
   For example, place `ggml-base.en.bin` in `models/`. Model files are not
   included in this repository. Optional face/pose models and the ONNX
   Runtime DLL are separate dependencies; the example below disables
   model-based layout detection for an initial run.
4. Build using the existing helper. Set the CUDA architecture to match
   your GPU; `86` below is an example, not a universal setting:

   ```powershell
   .\scripts\setup_gpu_build.ps1 -WhisperCudaFlags "-DGGML_CUDA_FORCE_MMQ" -WhisperCudaArch "86" -CargoProfile debug
   ```

5. Run from the repository root, replacing the example channel URL:

   ```powershell
   .\run_autoclip_with_cuda.ps1 -NoBuild "https://kick.com/your_channel" --phrase "clip that" --whisper-model "models/ggml-base.en.bin" --clip-layout=full --clip-detect=false --clip-gameplay=false --clip-face-mesh=false --clip-pose=false --clip-captions=false --clip-llm-enable=false
   ```

   The live path currently targets about 50 seconds before the trigger and
   10 seconds after it. Let the buffer warm up before triggering a clip.
   Output is saved under `clips/`, with a default 1080x1920 canvas.

For subsequent runs, edit the local, gitignored `config.env`. Supported
settings are polled during live runs; some startup settings need a restart.
The larger [example configuration](config.env.example) contains experimental
tuning values, so adjust the wake phrase and model paths for your setup.
See [configuration notes](docs/configuration.md) and
[Windows CUDA troubleshooting](BUILD_NOTES.md) for advanced setup.

## Run a media regression check

With FFmpeg and ffprobe on `PATH`, generate a short synthetic TS fixture
and explicitly run the ignored integration test:

```powershell
.\scripts\generate_test_fixture.ps1
cargo test --locked --no-default-features --test reprocess_ts -- --ignored --nocapture
```

The test uses CPU encoding in a temporary directory and checks that
`reprocess-ts` produces a nonempty MP4 whose duration matches the source
within 0.5 seconds. It does not test speech recognition or live stream access.
See [fixture notes](tests/data/README.md) for using your own media.

## Current limits

- Setup is manual; GPU inference and live platform access need validation
  on the machine and stream being used.
- The rolling buffer is bounded by duration, not a hard byte limit.
  Concurrent renders can increase memory and GPU pressure.
- Platform changes and expired stream credentials can interrupt capture.
- Captions, face/pose analysis and LLM titles require additional assets or
  services; no throughput or latency benchmark is claimed here.
- Clips are saved locally. Automatic uploads are outside the current scope.
