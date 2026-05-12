# AutoClip

**Build (CUDA, PowerShell)**
- Ensure: Rust (cargo), Visual Studio 2022 C++ tools, CUDA Toolkit, CMake, Ninja, and FFmpeg.
- Run the CUDA build helper:
  - `.\scripts\setup_gpu_build.ps1 -WhisperCudaFlags "-DGGML_CUDA_FORCE_MMQ" -WhisperCudaArch "86" -CargoProfile debug`

**Run (prebuilt EXE, no rebuild)**
- Use the CUDA run helper (quotes recommended for URLs with `?`):
  - `.\run_autoclip_with_cuda.ps1 -NoBuild "https://kick.com/leckatv"`
  - `.\run_autoclip_with_cuda.ps1 -NoBuild "https://www.twitch.tv/divatopia?twitch5=0"`
- Or run the EXE directly:
  - `.\target\debug\autoclip.exe "https://kick.com/leckatv"`

**Run (cargo)**
- Make CUDA DLLs available in the current shell:
  - `. .\scripts\enable_cuda_env.ps1`
- Then run:
  - `cargo run -- "https://kick.com/leckatv"`

**Live config**
- Copy `config.env.example` to `config.env` once; the binary polls it while
  running so edits apply within ~2s without restart.
- `config.env` is gitignored so per-developer paths (model locations, dll
  paths, etc.) stay local.

**Notes**
- `cargo run` needs CUDA DLLs on PATH (the helper above sets them).
- The EXE does not need `--` before the URL; `cargo run` does.
- Default layout is stacked; override with `--clip-layout=full` or `CLIP_LAYOUT=full`.
- Any env var can be set via CLI by lowercasing and replacing `_` with `-` (e.g., `CLIP_FACE_RATIO` -> `--clip-face-ratio=0.45`).
- Optional CUDA-backed face detection (ONNX Runtime): build with `--features ort` and set `CLIP_FACE_BACKEND=ort` (requires `onnxruntime.dll` + CUDA provider DLLs on PATH or `ORT_DYLIB_PATH`).
- Face debug dumps: set `CLIP_FACE_DUMP_DIR=.\face_debug` (or `--clip-face-dump-dir=.\face_debug`) to write annotated frame images with all detected face rectangles.
- Raw face dump + tile search: add `--clip-face-dump-raw` to dump raw candidates; tile search defaults to min score 0.60 and max depth 3 (override with `--clip-face-tile-min-score` / `--clip-face-tile-max-depth`, set min score <= 0 to disable). Raw score picking is enabled by default (disable with `--clip-face-pick-raw=false`). Face crops expand by default with `CLIP_FACE_CONTEXT=3.0` for a mid-shot (max 6.0).
- Tracking samples are spaced every 2 seconds by default (`CLIP_FACE_TRACK_STEP=2.0`) and chosen via bisection across the clip to keep faces in frame while respecting the budget.
- Audio is normalized by default (`CLIP_AUDIO_NORM=true`) using FFmpeg loudnorm; disable with `--clip-audio-norm=false`.
- When `CLIP_FACE_CONTEXT` is raised above the default (3.0), the face panel is pinned to 50% height to keep gameplay visible.
- Face crops widen to match the face panel aspect ratio so the top pane never letterboxes.
- Use `CLIP_FACE_BUDGET_SECS` to override the face analysis time budget (otherwise capped at 20s).
- Reprocess a saved TS snapshot with `.\target\debug\autoclip.exe reprocess-ts .\clips\clip_001.ts`.
- Validate the gameplay CLIP model with `.\target\debug\autoclip.exe check-gameplay-model .\models\clip-vit-base-patch32-xenova`.
- For help: `.\target\debug\autoclip.exe --help`
