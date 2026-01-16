@echo off
setlocal

if "%CUDA_PATH_OVERRIDE%"=="" (
  set "CUDA_PATH_OVERRIDE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
)
set "CUDA_PATH=%CUDA_PATH_OVERRIDE%"
set "CUDAToolkit_ROOT=%CUDA_PATH_OVERRIDE%"
if exist "%CUDA_PATH%\bin" (
  set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%"
)

set "PROFILE=%AUTOCLIP_PROFILE%"
if "%PROFILE%"=="" (
  if exist "target\debug\autoclip.exe" (
    set "PROFILE=debug"
  ) else if exist "target\release\autoclip.exe" (
    set "PROFILE=release"
  ) else (
    set "PROFILE=debug"
  )
)

set "EXE=target\%PROFILE%\autoclip.exe"
if not exist "%EXE%" (
  if /I "%AUTOCLIP_NO_BUILD%"=="1" (
    echo autoclip.exe not found in target\%PROFILE% (AUTOCLIP_NO_BUILD=1)
    exit /b 1
  )
  if /I "%PROFILE%"=="release" (
    echo autoclip.exe not found; building release...
    cargo build --release
  ) else if /I "%PROFILE%"=="debug" (
    echo autoclip.exe not found; building debug...
    cargo build
  ) else (
    echo autoclip.exe not found; building profile "%PROFILE%"...
    cargo build --profile %PROFILE%
  )
  if errorlevel 1 exit /b 1
)

"%EXE%" %*
