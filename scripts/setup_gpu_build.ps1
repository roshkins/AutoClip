#requires -version 7.0
$ErrorActionPreference = 'Stop'

param(
    [switch]$SkipWhisper,
    [string]$FfmpegZipUrl = 'https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1-essentials_build.zip',
    [string]$FfmpegInstallDir = 'tools/ffmpeg',
    [string]$CargoProfile = 'release'
)

function Test-FfmpegNvenc {
    try {
        $ffmpeg = Get-Command ffmpeg -ErrorAction Stop
    } catch {
        return @{ Present = $false; Nvenc = $false; Path = $null }
    }

    $enc = & $ffmpeg.Source -hide_banner -encoders 2>$null
    $hasNvenc = $enc -match 'h264_nvenc' -or $enc -match 'hevc_nvenc'
    return @{ Present = $true; Nvenc = [bool]$hasNvenc; Path = $ffmpeg.Source }
}

function Ensure-Ffmpeg {
    $probe = Test-FfmpegNvenc
    if ($probe.Present -and $probe.Nvenc) {
        Write-Host "FFmpeg found with NVENC: $($probe.Path)" -ForegroundColor Green
        return $probe.Path
    }

    Write-Host "FFmpeg with NVENC not found; downloading essentials build..." -ForegroundColor Yellow
    $destRoot = Join-Path $PSScriptRoot '..'
    $destDir = Join-Path $destRoot $FfmpegInstallDir
    $zipPath = Join-Path $destRoot 'tools/ffmpeg.zip'
    New-Item -ItemType Directory -Force -Path (Split-Path $zipPath) | Out-Null

    Write-Host "Downloading $FfmpegZipUrl -> $zipPath"
    Invoke-WebRequest -Uri $FfmpegZipUrl -OutFile $zipPath

    if (Test-Path $destDir) { Remove-Item -Recurse -Force $destDir }
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null

    Write-Host "Extracting..."
    Expand-Archive -Path $zipPath -DestinationPath $destDir -Force

    $ffmpegExe = Get-ChildItem -Path $destDir -Recurse -Filter ffmpeg.exe | Select-Object -First 1
    if (-not $ffmpegExe) { throw "ffmpeg.exe not found after extraction" }

    Write-Host "FFmpeg installed at $($ffmpegExe.FullName)" -ForegroundColor Green
    Write-Host "Add to PATH for this shell: `$Env:PATH = '$($ffmpegExe.DirectoryName);' + `$Env:PATH" -ForegroundColor Yellow
    return $ffmpegExe.FullName
}

function Build-WhisperCuda {
    Write-Host "Building autoclip with WHISPER_CUBLAS=1 (CUDA) ..." -ForegroundColor Cyan
    $env:WHISPER_CUBLAS = '1'
    $env:WHISPER_STRIP = '1'
    $cmd = "cargo build --$CargoProfile"
    Write-Host "Running: $cmd"
    Invoke-Expression $cmd
    Write-Host "Whisper build finished." -ForegroundColor Green
}

# Main
$ff = Ensure-Ffmpeg
$probe = Test-FfmpegNvenc
if (-not $probe.Nvenc) {
    Write-Host "Warning: FFmpeg still without NVENC. Check PATH and driver/toolkit." -ForegroundColor Yellow
}

if (-not $SkipWhisper) {
    Build-WhisperCuda
} else {
    Write-Host "Skipping Whisper build as requested." -ForegroundColor Yellow
}

Write-Host "Done. If FFmpeg was installed locally, prepend its folder to PATH before running autoclip." -ForegroundColor Cyan
