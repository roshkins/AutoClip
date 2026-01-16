#requires -version 7.0
param(
    [switch]$SkipWhisper,
    [string]$WhisperCudaFlags = '',
    [string]$WhisperCudaArch = '',
    [string]$CudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1',
    [string]$FfmpegZipUrl = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip',
    [string]$FfmpegInstallDir = 'tools/ffmpeg',
    [string]$CargoProfile = 'release',
    [switch]$RunAfterBuild,
    [string]$RunArgs = ''
)

$ErrorActionPreference = 'Stop'

function Import-VsDevEnv {
    $vsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $vsDevCmd)) {
        return $false
    }
    $output = & cmd /c "`"$vsDevCmd`" -arch=x64 -host_arch=x64 && set"
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    foreach ($line in $output) {
        if ($line -match '^([^=]+)=(.*)$') {
            $name = $matches[1]
            $value = $matches[2]
            Set-Item -Path "Env:$name" -Value $value
        }
    }
    return $true
}

function Convert-ToCMakePath {
    param([string]$PathValue)
    if (-not $PathValue) {
        return $PathValue
    }
    return ($PathValue -replace '\\', '/')
}

function Convert-ToNativePath {
    param([string]$PathValue)
    if (-not $PathValue) {
        return $PathValue
    }
    return ($PathValue -replace '/', '\')
}

function Stop-WhisperBuildProcesses {
    $names = @('cargo', 'rustc', 'cmake', 'ninja', 'msbuild', 'cl', 'link', 'rc', 'mt', 'nvcc')
    Get-Process -Name $names -ErrorAction SilentlyContinue | Stop-Process -Force
}

function Remove-LockFile {
    param([string]$PathValue)
    if (-not $PathValue) { return }
    if (-not (Test-Path $PathValue)) { return }
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Remove-Item -Force $PathValue -ErrorAction Stop
            return
        } catch {
            Stop-WhisperBuildProcesses
            Start-Sleep -Seconds 1
        }
    }
}

function Get-ShortPath {
    param([string]$PathValue)
    if (-not $PathValue) {
        return $PathValue
    }
    try {
        $escaped = $PathValue -replace '"', '""'
        $short = & cmd /c "for %I in (`"$escaped`") do @echo %~sI"
        if ($LASTEXITCODE -eq 0 -and $short) {
            return $short.Trim()
        }
    } catch {
        # Fall back to the original path if short-name lookup fails.
    }
    return $PathValue
}

function Reduce-PathForBuild {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CudaRoot,
        [string]$CargoBin = ''
    )

    $entries = @()

    function Add-PathEntry([string]$path) {
        if (-not $path) { return }
        $trimmed = (Convert-ToNativePath $path).Trim('"')
        if (Test-Path $trimmed -PathType Leaf) {
            $trimmed = Split-Path $trimmed -Parent
        }
        if ($trimmed -and (Test-Path $trimmed -PathType Container)) {
            $entries += $trimmed.TrimEnd('\')
        }
    }

    Add-PathEntry $env:WHISPER_CMAKE_C_COMPILER
    Add-PathEntry $env:WHISPER_CMAKE_CXX_COMPILER
    Add-PathEntry $env:WHISPER_CMAKE_ASM_COMPILER
    Add-PathEntry $env:CMAKE_ASM_COMPILER
    Add-PathEntry $env:WHISPER_CMAKE_CUDA_COMPILER
    Add-PathEntry $env:WHISPER_CMAKE_CUDA_HOST_COMPILER
    Add-PathEntry $env:WHISPER_CMAKE_RC_COMPILER
    Add-PathEntry $env:WHISPER_CMAKE_MT
    Add-PathEntry $env:CMAKE
    Add-PathEntry $env:WHISPER_CMAKE_MAKE_PROGRAM
    Add-PathEntry (Join-Path $CudaRoot "bin")
    Add-PathEntry (Join-Path $CudaRoot "bin\x64")
    if ($CargoBin) {
        Add-PathEntry $CargoBin
    }
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if ($gitCmd -and $gitCmd.Source) {
        Add-PathEntry $gitCmd.Source
    }
    $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE ".cargo" }
    Add-PathEntry (Join-Path $cargoHome "bin")

    $sysRoot = $env:SystemRoot
    Add-PathEntry (Join-Path $sysRoot "System32")
    Add-PathEntry $sysRoot
    Add-PathEntry (Join-Path $sysRoot "System32\Wbem")
    Add-PathEntry (Join-Path $sysRoot "System32\WindowsPowerShell\v1.0")

    $unique = $entries | Select-Object -Unique
    if ($unique -and $unique.Count -gt 0) {
        $before = $env:PATH.Length
        $env:PATH = ($unique -join ';')
        $after = $env:PATH.Length
        Write-Host "PATH trimmed from $before to $after chars for build." -ForegroundColor Yellow
    }
}

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

    $downloadUrls = @(
        $FfmpegZipUrl,
        'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip',
        'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip?download=1'
    ) | Select-Object -Unique

    $downloaded = $false
    foreach ($url in $downloadUrls) {
        try {
            Write-Host "Downloading $url -> $zipPath"
            Invoke-WebRequest -Uri $url -OutFile $zipPath -ErrorAction Stop
            $downloaded = $true
            break
        } catch {
            Write-Host "Download failed: $url ($($_.Exception.Message))" -ForegroundColor Yellow
        }
    }
    if (-not $downloaded) {
        throw "FFmpeg download failed. Provide -FfmpegZipUrl or install manually."
    }

    if (Test-Path $destDir) { Remove-Item -Recurse -Force $destDir }
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null

    Write-Host "Extracting..."
    Expand-Archive -Path $zipPath -DestinationPath $destDir -Force

    $ffmpegExe = Get-ChildItem -Path $destDir -Recurse -Filter ffmpeg.exe | Select-Object -First 1
    if (-not $ffmpegExe) { throw "ffmpeg.exe not found after extraction" }

    Write-Host "FFmpeg installed at $($ffmpegExe.FullName)" -ForegroundColor Green
    if ($env:PATH -notmatch [regex]::Escape($ffmpegExe.DirectoryName)) {
        $env:PATH = "$($ffmpegExe.DirectoryName);$env:PATH"
    }
    Write-Host "Added to PATH for this shell: $($ffmpegExe.DirectoryName)" -ForegroundColor Yellow
    return $ffmpegExe.FullName
}

function Build-WhisperCuda {
    $originalPath = $env:PATH
    $loadedVs = Import-VsDevEnv
    if ($loadedVs) {
        Write-Host "Loaded Visual Studio developer environment." -ForegroundColor Yellow
    }
    if (-not (Test-Path $CudaRoot)) {
        throw "CUDA root not found at: $CudaRoot"
    }
    $cargoCmd = Get-Command cargo -ErrorAction SilentlyContinue
    $cargoBin = if ($cargoCmd -and $cargoCmd.Source) { Split-Path $cargoCmd.Source -Parent } else { $null }
    $env:CUDA_PATH = $CudaRoot
    $env:CUDAToolkit_ROOT = $CudaRoot
    $env:PATH = "$CudaRoot\bin;$CudaRoot\bin\x64;$env:PATH"
    if (-not $env:CMAKE) {
        $cmake = "C:\Program Files\CMake\bin\cmake.exe"
        if (Test-Path $cmake) {
            $env:CMAKE = $cmake
            Write-Host "Using CMake: $cmake" -ForegroundColor Yellow
        } else {
            $vsCmake = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
            if (Test-Path $vsCmake) {
                $env:CMAKE = $vsCmake
                Write-Host "Using Visual Studio CMake: $vsCmake" -ForegroundColor Yellow
            }
        }
    }
    $nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
    if (Test-Path $nvcc) {
        $nvccFull = (Get-Item -LiteralPath $nvcc).FullName
        $nvccCmake = Convert-ToCMakePath $nvccFull
        $env:WHISPER_CMAKE_CUDA_COMPILER = $nvccCmake
        $env:WHISPER_CMAKE_CUDA_COMPILER_FORCED = "1"
        $env:WHISPER_CMAKE_CUDA_COMPILER_WORKS = "1"
    }
    $clCandidates = Get-ChildItem -Path "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    if ($clCandidates -and $clCandidates.Count -gt 0) {
        $clPath = Join-Path $clCandidates[0].FullName "bin\Hostx64\x64\cl.exe"
        if (Test-Path $clPath) {
            $clPathFull = (Get-Item -LiteralPath $clPath).FullName
            $clPathForNvcc = Get-ShortPath $clPathFull
            $clPathCmake = Convert-ToCMakePath $clPathForNvcc
            $hostCompilerDir = Split-Path $clPathForNvcc -Parent
            $hostCompilerDirCmake = Convert-ToCMakePath $hostCompilerDir
            Write-Host "Using MSVC host compiler: $clPathForNvcc" -ForegroundColor Yellow
            $env:WHISPER_CMAKE_CUDA_HOST_COMPILER = $clPathCmake
            $env:CMAKE_CUDA_HOST_COMPILER = $clPathCmake
            if (-not $env:CUDAHOSTCXX) {
                $env:CUDAHOSTCXX = $clPathForNvcc
            }
            $env:WHISPER_CMAKE_C_COMPILER = $clPathCmake
            $env:WHISPER_CMAKE_CXX_COMPILER = $clPathCmake
            $env:WHISPER_CMAKE_ASM_COMPILER = $clPathCmake
            $env:CC = $clPathForNvcc
            $env:CXX = $clPathForNvcc
            if (Test-Path Env:NVCC_PREPEND_FLAGS) { Remove-Item Env:NVCC_PREPEND_FLAGS -ErrorAction SilentlyContinue }
            if (Test-Path Env:CMAKE_C_COMPILER) { Remove-Item Env:CMAKE_C_COMPILER -ErrorAction SilentlyContinue }
            if (Test-Path Env:CMAKE_CXX_COMPILER) { Remove-Item Env:CMAKE_CXX_COMPILER -ErrorAction SilentlyContinue }
            if (Test-Path Env:CMAKE_ASM_COMPILER) { Remove-Item Env:CMAKE_ASM_COMPILER -ErrorAction SilentlyContinue }
        }
    }
    $sdkBins = Get-ChildItem -Path "C:\Program Files (x86)\Windows Kits\10\bin" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    foreach ($sdk in $sdkBins) {
        $rc = Join-Path $sdk.FullName "x64\rc.exe"
        $mt = Join-Path $sdk.FullName "x64\mt.exe"
        if ((Test-Path $rc) -and (Test-Path $mt)) {
            $rcFull = (Get-Item -LiteralPath $rc).FullName
            $mtFull = (Get-Item -LiteralPath $mt).FullName
            $rcCmake = Convert-ToCMakePath $rcFull
            $mtCmake = Convert-ToCMakePath $mtFull
            $env:WHISPER_CMAKE_RC_COMPILER = $rcCmake
            $env:WHISPER_CMAKE_MT = $mtCmake
            $env:CMAKE_RC_COMPILER = $rcCmake
            $env:CMAKE_MT = $mtCmake
            break
        }
    }
    $ninja = Get-Command ninja -ErrorAction SilentlyContinue
    if ($ninja -and $ninja.Source) {
        $env:WHISPER_CMAKE_GENERATOR = "Ninja"
        $env:WHISPER_CMAKE_MAKE_PROGRAM = $ninja.Source
        $env:CMAKE_GENERATOR = "Ninja"
        $env:CMAKE_MAKE_PROGRAM = $ninja.Source
        Write-Host "Using Ninja generator: $($ninja.Source)" -ForegroundColor Yellow
    }
    if (-not $env:CMAKE_BUILD_PARALLEL_LEVEL) {
        $env:CMAKE_BUILD_PARALLEL_LEVEL = "1"
    }
    if (-not $env:CARGO_BUILD_JOBS) {
        $env:CARGO_BUILD_JOBS = "1"
    }
    Reduce-PathForBuild -CudaRoot $CudaRoot -CargoBin $cargoBin
    if ($env:WHISPER_CMAKE_CUDA_HOST_COMPILER) {
        $pathEntries = @()
        if ($hostCompilerDir) {
            $pathEntries += $hostCompilerDir
        } else {
            $pathEntries += (Split-Path (Convert-ToNativePath $env:WHISPER_CMAKE_CUDA_HOST_COMPILER) -Parent)
        }
        if ($env:WHISPER_CMAKE_CUDA_COMPILER) {
            $pathEntries += (Split-Path (Convert-ToNativePath $env:WHISPER_CMAKE_CUDA_COMPILER) -Parent)
        }
        if ($env:WHISPER_CMAKE_MAKE_PROGRAM) {
            $pathEntries += (Split-Path (Convert-ToNativePath $env:WHISPER_CMAKE_MAKE_PROGRAM) -Parent)
        }
        if ($env:CMAKE) {
            $pathEntries += (Split-Path (Convert-ToNativePath $env:CMAKE) -Parent)
        }
        $pathEntries += (Join-Path $CudaRoot "bin")
        $pathEntries += (Join-Path $CudaRoot "bin\x64")
        if ($env:WHISPER_CMAKE_RC_COMPILER) {
            $pathEntries += (Split-Path (Convert-ToNativePath $env:WHISPER_CMAKE_RC_COMPILER) -Parent)
        }
        if ($env:WHISPER_CMAKE_MT) {
            $pathEntries += (Split-Path (Convert-ToNativePath $env:WHISPER_CMAKE_MT) -Parent)
        }
        if ($env:CMAKE_RC_COMPILER) {
            $pathEntries += (Split-Path (Convert-ToNativePath $env:CMAKE_RC_COMPILER) -Parent)
        }
        if ($env:CMAKE_MT) {
            $pathEntries += (Split-Path (Convert-ToNativePath $env:CMAKE_MT) -Parent)
        }
        if ($cargoBin) {
            $pathEntries += $cargoBin
        }
        $gitCmd = Get-Command git -ErrorAction SilentlyContinue
        if ($gitCmd -and $gitCmd.Source) {
            $pathEntries += (Split-Path (Convert-ToNativePath $gitCmd.Source) -Parent)
        }
        $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE ".cargo" }
        $pathEntries += (Join-Path $cargoHome "bin")
        $pathEntries += (Join-Path $env:SystemRoot "System32")
        $pathEntries += $env:SystemRoot
        $pathEntries += (Join-Path $env:SystemRoot "System32\Wbem")
        $pathEntries += (Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0")
        $env:PATH = ($pathEntries | Select-Object -Unique) -join ';'
        Write-Host "PATH reset to match CUDA host compiler." -ForegroundColor Yellow
        if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
            $fallbackCargo = Join-Path $env:USERPROFILE ".cargo\bin"
            if (Test-Path $fallbackCargo) {
                $env:PATH = "$fallbackCargo;$env:PATH"
            }
        }
    }

    $profileLower = $CargoProfile.ToLowerInvariant()
    $targetProfileDir = if ($profileLower -eq 'release') { 'release' } elseif ($profileLower -in @('debug', 'dev')) { 'debug' } else { $CargoProfile }
    $markerDir = Join-Path $PSScriptRoot "..\\target\\$targetProfileDir"
    New-Item -ItemType Directory -Force -Path $markerDir | Out-Null
    $flagsPath = Join-Path $markerDir ".whisper_cuda_flags"
    $successPath = Join-Path $markerDir ".whisper_cuda_last_success"

    Write-Host "Building autoclip with GGML_CUDA=1 (CUDA) ..." -ForegroundColor Cyan
    $env:GGML_CUDA = '1'
    if (Test-Path Env:WHISPER_CUBLAS) {
        Remove-Item Env:WHISPER_CUBLAS -ErrorAction SilentlyContinue
    }
    $env:WHISPER_STRIP = if ($profileLower -eq 'release') { '1' } else { '0' }
    if ($WhisperCudaFlags) {
        $env:WHISPER_EXTRA_FLAGS = $WhisperCudaFlags
        Write-Host "Using WHISPER_EXTRA_FLAGS='$WhisperCudaFlags'" -ForegroundColor Yellow
    }
    if ($WhisperCudaArch) {
        $env:GGML_CUDA_ARCHITECTURES = $WhisperCudaArch
        $env:CMAKE_CUDA_ARCHITECTURES = $WhisperCudaArch
        Write-Host "Using CMAKE_CUDA_ARCHITECTURES='$WhisperCudaArch' (GGML_CUDA_ARCHITECTURES)" -ForegroundColor Yellow
    }
    $currentFlags = "flags=$WhisperCudaFlags`narch=$WhisperCudaArch"
    $lastFlags = if (Test-Path $flagsPath) { (Get-Content -Raw $flagsPath).Trim() } else { '' }
    $currentFlags = $currentFlags.Trim()
    $lastSucceeded = Test-Path $successPath
    $shouldClean = $false
    if ((-not $WhisperCudaFlags) -and (-not $WhisperCudaArch)) {
        $shouldClean = $true
    } elseif (-not $lastSucceeded) {
        $shouldClean = $true
    } elseif ($currentFlags -ne $lastFlags) {
        $shouldClean = $true
    }

    if ($shouldClean) {
        Write-Host "Cleaning whisper-rs-sys to apply CUDA build flags..." -ForegroundColor Yellow
        Stop-WhisperBuildProcesses
        $lockPath = Join-Path $PSScriptRoot "..\\target\\$targetProfileDir\\.cargo-lock"
        Remove-LockFile -PathValue $lockPath
        Invoke-Expression "cargo clean -p whisper-rs-sys"
        $buildRoot = Join-Path $PSScriptRoot "..\\target\\$targetProfileDir\\build"
        if (Test-Path $buildRoot) {
            $dirs = Get-ChildItem -Path $buildRoot -Directory -Filter "whisper-rs-sys-*" -ErrorAction SilentlyContinue
            if ($dirs) {
                $maxAttempts = 3
                for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
                    $remaining = @()
                    foreach ($dir in $dirs) {
                        if (-not (Test-Path $dir.FullName)) {
                            continue
                        }
                        try {
                            Write-Host "Removing cached build dir (attempt $attempt/$maxAttempts): $($dir.FullName)" -ForegroundColor Yellow
                            Remove-Item -Recurse -Force $dir.FullName -ErrorAction Stop
                        } catch {
                            $remaining += $dir
                        }
                    }
                    if (-not $remaining -or $remaining.Count -eq 0) {
                        break
                    }
                    if ($attempt -lt $maxAttempts) {
                        Write-Host "Build dirs still locked; terminating build tools and retrying..." -ForegroundColor Yellow
                        Stop-WhisperBuildProcesses
                        Remove-LockFile -PathValue $lockPath
                        Start-Sleep -Seconds 2
                    } else {
                        $locked = $remaining | ForEach-Object { $_.FullName }
                        throw "Unable to remove cached build dirs due to file locks: $($locked -join '; ')"
                    }
                }
            }
        }
    } else {
        Write-Host "Skipping whisper-rs-sys clean (flags unchanged and last build succeeded)." -ForegroundColor Yellow
    }
    if ($profileLower -eq 'release') {
        $cmd = "cargo build --release"
        $runPrefix = "cargo run --release --"
    } elseif ($profileLower -in @('debug', 'dev', '')) {
        $cmd = "cargo build"
        $runPrefix = "cargo run --"
    } else {
        $cmd = "cargo build --profile $CargoProfile"
        $runPrefix = "cargo run --profile $CargoProfile --"
    }
    Write-Host "Running: $cmd"
    $buildSucceeded = $false
    try {
        Invoke-Expression $cmd
        $buildSucceeded = $true
        Write-Host "Whisper build finished." -ForegroundColor Green
        if ($RunAfterBuild) {
            $runCmd = if ($RunArgs) { "$runPrefix $RunArgs" } else { $runPrefix.TrimEnd() }
            Write-Host "Running: $runCmd" -ForegroundColor Yellow
            Invoke-Expression $runCmd
        }
    } catch {
        $buildSucceeded = $false
        throw
    } finally {
        if ($buildSucceeded) {
            Set-Content -Path $flagsPath -Value $currentFlags
            New-Item -ItemType File -Force -Path $successPath | Out-Null
        } else {
            if (Test-Path $successPath) {
                Remove-Item -Force $successPath
            }
        }
        if ($originalPath) {
            $env:PATH = $originalPath
        }
    }
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

