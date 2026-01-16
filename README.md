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
- For help: `.\target\debug\autoclip.exe --help`
