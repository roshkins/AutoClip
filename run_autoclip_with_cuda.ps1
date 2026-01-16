[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Profile = '',
    [switch]$NoBuild,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArgs = @()
)

$ErrorActionPreference = "Stop"

$cudaOverride = $env:CUDA_PATH_OVERRIDE
if ($cudaOverride) {
    $env:CUDA_PATH = $cudaOverride
} elseif (-not $env:CUDA_PATH -or $env:CUDA_PATH.Trim().Length -eq 0) {
    $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
}

$cudaBin = Join-Path $env:CUDA_PATH "bin"
$cudaBin64 = Join-Path $env:CUDA_PATH "bin\x64"
if (Test-Path $cudaBin) {
    $env:PATH = "$cudaBin;$cudaBin64;$env:PATH"
}

$chosenProfile = $Profile
if (-not $chosenProfile) {
    if (Test-Path "target\debug\autoclip.exe") {
        $chosenProfile = "debug"
    } elseif (Test-Path "target\release\autoclip.exe") {
        $chosenProfile = "release"
    } else {
        $chosenProfile = "debug"
    }
}

$exe = Join-Path ("target\" + $chosenProfile) "autoclip.exe"
if (-not (Test-Path $exe)) {
    if ($NoBuild) {
        throw "autoclip.exe not found in target\$chosenProfile (NoBuild set)."
    }
    $cargo = Get-Command cargo -ErrorAction SilentlyContinue
    if (-not $cargo) {
        throw "cargo not found in PATH. Install Rust or add cargo to PATH."
    }
    if ($chosenProfile -eq "release") {
        Write-Host "autoclip.exe not found; building release..."
        & $cargo.Source build --release
    } elseif ($chosenProfile -eq "debug") {
        Write-Host "autoclip.exe not found; building debug..."
        & $cargo.Source build
    } else {
        Write-Host "autoclip.exe not found; building profile '$chosenProfile'..."
        & $cargo.Source build --profile $chosenProfile
    }
}

& $exe @RunArgs
