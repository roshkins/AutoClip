[CmdletBinding(PositionalBinding = $false)]
param(
    [switch]$KillLocks,
    [switch]$SkipBindgen,
    [switch]$NoNinja,
    [string]$CudaArch,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args = @()
)

$ErrorActionPreference = "Stop"

$originalPath = $env:Path
$cargoPath = (Get-Command cargo -ErrorAction SilentlyContinue).Source
$cargoDir = if ($cargoPath) { Split-Path $cargoPath } else { $null }

if (-not $PSBoundParameters.ContainsKey('KillLocks')) {
    $KillLocks = $true
}

function Get-BasePath {
    param([string]$ExtraPath)
    $parts = @()
    foreach ($p in @(
        [Environment]::GetEnvironmentVariable("Path", "Machine"),
        [Environment]::GetEnvironmentVariable("Path", "User"),
        $ExtraPath
    )) {
        if (-not $p) { continue }
        $parts += ($p -split ';') | Where-Object { $_ }
    }
    $parts = $parts | Select-Object -Unique
    if ($parts.Count -eq 0) { return $null }
    return [string]::Join(';', $parts)
}

function Import-VsDevEnv {
    param([string]$FallbackPath)
    $vsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $vsDevCmd)) {
        return $false
    }
    $cmdLine = "`"$vsDevCmd`" -arch=x64 -host_arch=x64 && set"
    $output = & cmd /c $cmdLine 2>&1
    $needsFallback = ($LASTEXITCODE -ne 0) -or ($output | Select-String -SimpleMatch -Pattern "input line is too long" -Quiet)
    if ($needsFallback -and $FallbackPath) {
        $cmdLine = "set `"PATH=$FallbackPath`" && `"$vsDevCmd`" -arch=x64 -host_arch=x64 && set"
        $output = & cmd /c $cmdLine 2>&1
    }
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

function Stop-WhisperBuildProcesses {
    $names = @('cargo', 'rustc', 'cmake', 'ninja', 'msbuild', 'cl', 'link', 'rc', 'mt', 'nvcc')
    Get-Process -Name $names -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

function Remove-LockFile {
    param([string]$PathValue)
    if (-not $PathValue) { return $false }
    if (-not (Test-Path $PathValue)) { return $false }
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Remove-Item -Force $PathValue -ErrorAction Stop
            return $true
        } catch {
            Stop-WhisperBuildProcesses
            Start-Sleep -Seconds 1
        }
    }
    return $false
}

function Set-EnvIfEmpty {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Value
    )
    $current = (Get-Item -Path "Env:$Name" -ErrorAction SilentlyContinue).Value
    if (-not $current) {
        Set-Item -Path "Env:$Name" -Value $Value
        return $true
    }
    return $false
}

function Normalize-PathEntry {
    param([string]$PathEntry)
    if (-not $PathEntry) { return $null }
    return $PathEntry.Trim().TrimEnd('\')
}

function Add-ToPathIfMissing {
    param(
        [string]$Dir,
        [switch]$Prepend
    )
    if (-not $Dir) { return $false }
    if (-not (Test-Path $Dir)) { return $false }
    $normDir = Normalize-PathEntry $Dir
    $parts = $env:Path -split ';' | Where-Object { $_ }
    $normParts = $parts | ForEach-Object { Normalize-PathEntry $_ }
    if ($normParts -contains $normDir) { return $false }
    if ($Prepend) {
        $env:Path = ($Dir + ';' + $env:Path.TrimStart(';'))
    } else {
        $env:Path = ($env:Path.TrimEnd(';') + ';' + $Dir)
    }
    return $true
}

function Resolve-CudaPath {
    if ($env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) {
        return $env:CUDA_PATH
    }
    $root = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (-not (Test-Path $root)) {
        return $null
    }
    $candidates = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "bin\\nvcc.exe") }
    if (-not $candidates) {
        return $null
    }
    $best = $candidates |
        ForEach-Object {
            $name = $_.Name.TrimStart('v')
            $version = $null
            try { $version = [version]$name } catch { }
            [PSCustomObject]@{ Dir = $_.FullName; Version = $version }
        } |
        Sort-Object -Property Version -Descending |
        Select-Object -First 1
    return $best.Dir
}

$cpu = [Environment]::ProcessorCount
if ($cpu -lt 1) { $cpu = 1 }

$fallbackPath = Get-BasePath -ExtraPath $cargoDir
$loadedVs = Import-VsDevEnv -FallbackPath $fallbackPath
if ($loadedVs) {
    Write-Host "Loaded Visual Studio developer environment." -ForegroundColor Yellow
}

function Set-LibClangEnv {
    if ($env:LIBCLANG_PATH -and (Test-Path (Join-Path $env:LIBCLANG_PATH "libclang.dll"))) {
        return $false
    }
    $candidates = @(
        "C:\Program Files\LLVM\bin",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin"
    )
    foreach ($dir in $candidates) {
        if (Test-Path (Join-Path $dir "libclang.dll")) {
            $env:LIBCLANG_PATH = $dir
            if (-not $env:CLANG_PATH -and (Test-Path (Join-Path $dir "clang.exe"))) {
                $env:CLANG_PATH = (Join-Path $dir "clang.exe")
            }
            return $true
        }
    }
    return $false
}

$setLibclang = Set-LibClangEnv
if ($setLibclang) {
    Write-Host "Using libclang from $env:LIBCLANG_PATH" -ForegroundColor Yellow
}

$cudaPath = Resolve-CudaPath
if (-not $env:CUDA_PATH -and $cudaPath) {
    $env:CUDA_PATH = $cudaPath
}
if ($cudaPath) {
    $addedCuda = $false
    $addedCuda = (Add-ToPathIfMissing -Dir (Join-Path $cudaPath "bin\\x64") -Prepend) -or $addedCuda
    $addedCuda = (Add-ToPathIfMissing -Dir (Join-Path $cudaPath "bin")) -or $addedCuda
    if ($addedCuda) {
        Write-Host "Added CUDA runtime paths to PATH." -ForegroundColor Yellow
    }
}

if (-not $env:BINDGEN_EXTRA_CLANG_ARGS) {
    $env:BINDGEN_EXTRA_CLANG_ARGS = "--target=x86_64-pc-windows-msvc"
}
if (-not $env:BINDGEN_EXTRA_CLANG_ARGS_x86_64_pc_windows_msvc) {
    $env:BINDGEN_EXTRA_CLANG_ARGS_x86_64_pc_windows_msvc = "--target=x86_64-pc-windows-msvc"
}

if ($KillLocks) {
    $repoRoot = Resolve-Path $PSScriptRoot
    $targetRoot = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $repoRoot "target" }
    $removed = @()
    foreach ($profile in @("debug", "release")) {
        $lockPath = Join-Path (Join-Path $targetRoot $profile) ".cargo-lock"
        if (Remove-LockFile -PathValue $lockPath) {
            $removed += $lockPath
        }
    }
    if ($removed.Count -gt 0) {
        Write-Host "Cleared build locks: $($removed -join ', ')" -ForegroundColor Yellow
    }
}

function Remove-StaleWhisperBindings {
    param([string]$TargetRoot)
    if (-not $TargetRoot -or -not (Test-Path $TargetRoot)) {
        return
    }
    $profiles = @("debug", "release")
    foreach ($profile in $profiles) {
        $buildRoot = Join-Path $TargetRoot $profile
        $buildRoot = Join-Path $buildRoot "build"
        if (-not (Test-Path $buildRoot)) {
            continue
        }
        $dirs = Get-ChildItem -Path $buildRoot -Directory -Filter "whisper-rs-sys-*" -ErrorAction SilentlyContinue
        foreach ($dir in $dirs) {
            $bindings = Join-Path $dir.FullName "out\\bindings.rs"
            if (-not (Test-Path $bindings)) {
                continue
            }
            $head = Get-Content -Path $bindings -TotalCount 40 -ErrorAction SilentlyContinue
            $hasGlibc = $head | Select-String -Pattern "__GLIBC__" -SimpleMatch -Quiet
            if ($hasGlibc) {
                Remove-Item -Force $bindings -ErrorAction SilentlyContinue
                Write-Host "Removed stale glibc bindings: $bindings" -ForegroundColor Yellow
            }
        }
    }
}

Remove-StaleWhisperBindings -TargetRoot $targetRoot

$setJobs = Set-EnvIfEmpty -Name "CARGO_BUILD_JOBS" -Value "$cpu"
$setCmake = Set-EnvIfEmpty -Name "CMAKE_BUILD_PARALLEL_LEVEL" -Value "$cpu"
$setBindings = $false
if ($SkipBindgen) {
    $setBindings = Set-EnvIfEmpty -Name "WHISPER_DONT_GENERATE_BINDINGS" -Value "1"
} elseif ($env:WHISPER_DONT_GENERATE_BINDINGS) {
    Write-Host "Clearing WHISPER_DONT_GENERATE_BINDINGS (unsafe for Windows bindings)." -ForegroundColor Yellow
    Remove-Item Env:WHISPER_DONT_GENERATE_BINDINGS -ErrorAction SilentlyContinue
}

$ninja = Get-Command ninja -ErrorAction SilentlyContinue
$setNinja = $false
if ($NoNinja -or $env:WHISPER_NO_NINJA) {
    Remove-Item Env:WHISPER_CMAKE_GENERATOR -ErrorAction SilentlyContinue
    Remove-Item Env:WHISPER_CMAKE_MAKE_PROGRAM -ErrorAction SilentlyContinue
} elseif ($ninja) {
    $setNinja = Set-EnvIfEmpty -Name "WHISPER_CMAKE_GENERATOR" -Value "Ninja"
    $setNinja = (Set-EnvIfEmpty -Name "WHISPER_CMAKE_MAKE_PROGRAM" -Value $ninja.Source) -or $setNinja
}

$setArch = $false
if ($CudaArch) {
    $archList = $CudaArch.Trim()
    if ($archList) {
        $setArch = Set-EnvIfEmpty -Name "GGML_CUDA_ARCHITECTURES" -Value $archList
        $setArch = (Set-EnvIfEmpty -Name "CMAKE_CUDA_ARCHITECTURES" -Value $archList) -or $setArch
        $setArch = (Set-EnvIfEmpty -Name "CUDAARCHS" -Value $archList) -or $setArch
    }
} elseif (-not $env:GGML_CUDA_ARCHITECTURES) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        try {
            $raw = & $smi.Source --query-gpu=compute_cap --format=csv,noheader 2>$null
            $caps = $raw | ForEach-Object { $_.Trim() } | Where-Object { $_ }
            if ($caps.Count -gt 0) {
                $archs = $caps | ForEach-Object { $_ -replace '\.', '' } |
                    Where-Object { $_ -match '^[0-9]+$' } |
                    ForEach-Object { [int]$_ } |
                    Select-Object -Unique
                $archs = $archs | Where-Object { $_ -ge 50 -and $_ -le 120 } |
                    ForEach-Object { $_.ToString() } |
                    Select-Object -Unique
                if ($archs.Count -gt 0) {
                    $archList = ($archs -join ';')
                    $setArch = Set-EnvIfEmpty -Name "GGML_CUDA_ARCHITECTURES" -Value $archList
                    $setArch = (Set-EnvIfEmpty -Name "CMAKE_CUDA_ARCHITECTURES" -Value $archList) -or $setArch
                    $setArch = (Set-EnvIfEmpty -Name "CUDAARCHS" -Value $archList) -or $setArch
                } else {
                    Write-Host "Warning: skipping GGML_CUDA_ARCHITECTURES auto-set (unsupported or unknown compute capability)." -ForegroundColor Yellow
                }
            }
        } catch {
            # Ignore GPU arch detection failures.
        }
    }
}

Write-Host "fast test env:" -ForegroundColor Cyan
Write-Host "  jobs=$env:CARGO_BUILD_JOBS (set=$setJobs)"
Write-Host "  cmake_parallel=$env:CMAKE_BUILD_PARALLEL_LEVEL (set=$setCmake)"
Write-Host "  skip_bindgen=$env:WHISPER_DONT_GENERATE_BINDINGS (set=$setBindings)"
if ($env:GGML_CUDA_ARCHITECTURES) {
    Write-Host "  cuda_arch=$env:GGML_CUDA_ARCHITECTURES (set=$setArch)"
}
if ($ninja) {
    Write-Host "  ninja=$env:WHISPER_CMAKE_GENERATOR (set=$setNinja)"
}

$cargo = Get-Command cargo -ErrorAction Stop
& $cargo.Source test @Args
exit $LASTEXITCODE
