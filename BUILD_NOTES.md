# Build Notes: Whisper CUDA on Windows

## Symptom
- CMake "hangs" or build appears stuck, or MSBuild exits with no errors.
- CMake reports the C compiler is "broken" or cannot run `rc.exe`.
- Generator mismatch errors when switching to Ninja.

## Root Causes Observed
- Visual Studio developer environment not loaded, so `cl`, `rc`, `mt`, SDK
  includes/libs, and MSVC toolchain vars were missing.
- `pwsh -Command` one-liners had malformed quoting and dropped into an
  interactive prompt (looked like a hang).
- Old CMake cache directories were left behind when switching generators.
- `nvcc` failed when `cl.exe` on PATH did not match the `-ccbin` compiler
  (often caused by VS Insiders paths leaking into PATH).

## Reliable Fix
- Always load the VS dev environment before running CMake:
  `VsDevCmd.bat -arch=x64 -host_arch=x64`
- Force explicit toolchain paths for CMake:
  - `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` -> MSVC `cl.exe`
  - `CMAKE_RC_COMPILER` -> Windows SDK `rc.exe`
  - `CMAKE_MT` -> Windows SDK `mt.exe`
  - `CMAKE_CUDA_COMPILER` -> `nvcc.exe`
- Use Ninja consistently with a clean build directory.
- If changing generator or flags, delete `target\release\build\whisper-rs-sys-*`
  to avoid cache conflicts.

## Manual Known-Good Configure/Build (reference)
From an x64 VS dev shell:
```
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set CUDAToolkit_ROOT=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%

cmake -S vendor\whisper-rs-sys\whisper.cpp -B target\manual_whisper\build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON ^
  -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=OFF ^
  -DGGML_CUDA_ARCHITECTURES=86 -DWHISPER_EXTRA_FLAGS=-DGGML_CUDA_FORCE_MMQ ^
  -DCUDAToolkit_ROOT=%CUDA_PATH% -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  -DCMAKE_RC_COMPILER=rc -DCMAKE_MT=mt ^
  -DCMAKE_INSTALL_PREFIX=target\manual_whisper\install

set CMAKE_BUILD_PARALLEL_LEVEL=1
ninja -C target\manual_whisper\build -v -j1 install
```

## Integration Notes
- `scripts/setup_gpu_build.ps1` now loads the VS dev environment, sets explicit
  compiler/tool paths, trims PATH for build, and defaults build parallelism to 1.
- `whisper-rs-sys/build.rs` passes those paths (including ASM) to CMake.
