@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CMAKE_GENERATOR=Ninja
set CMAKE_MAKE_PROGRAM=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
set WHISPER_CUBLAS=1
set GGML_LOG_LEVEL=1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64;%PATH%
set BASE_DIR=%CD%\target\debug\build
set WHISPER_OUT=
for /d %%d in ("%BASE_DIR%\whisper-rs-sys-*") do set WHISPER_OUT=%%d\out
if defined WHISPER_OUT if exist "%WHISPER_OUT%\lib\static\whisper.lib" (
	if not exist "%WHISPER_OUT%\build\Release" mkdir "%WHISPER_OUT%\build\Release"
	copy /Y "%WHISPER_OUT%\lib\static\whisper.lib" "%WHISPER_OUT%\build\Release\whisper.lib" >nul
)
cargo test -q
