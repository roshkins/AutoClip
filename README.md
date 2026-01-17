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

**Notes**
- `cargo run` needs CUDA DLLs on PATH (the helper above sets them).
- The EXE does not need `--` before the URL; `cargo run` does.
- Default layout is stacked; override with `--clip-layout=full` or `CLIP_LAYOUT=full`.
- Any env var can be set via CLI by lowercasing and replacing `_` with `-` (e.g., `CLIP_FACE_RATIO` -> `--clip-face-ratio=0.45`).
- Optional CUDA-backed face detection (ONNX Runtime): build with `--features ort` and set `CLIP_FACE_BACKEND=ort` (requires `onnxruntime.dll` + CUDA provider DLLs on PATH or `ORT_DYLIB_PATH`).
- Face debug dumps: set `CLIP_FACE_DUMP_DIR=.\face_debug` (or `--clip-face-dump-dir=.\face_debug`) to write annotated frame images with all detected face rectangles.
- Raw face dump + tile search: add `--clip-face-dump-raw` to dump raw candidates; tile search defaults to min score 0.60 and max depth 3 (override with `--clip-face-tile-min-score` / `--clip-face-tile-max-depth`, set min score <= 0 to disable).
- Reprocess a saved TS snapshot with `.\target\debug\autoclip.exe reprocess-ts .\clips\clip_001.ts`.
- Validate the gameplay CLIP model with `.\target\debug\autoclip.exe check-gameplay-model .\models\clip-vit-base-patch32-xenova`.
- For help: `.\target\debug\autoclip.exe --help`
