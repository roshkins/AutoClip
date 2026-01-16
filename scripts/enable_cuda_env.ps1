[CmdletBinding()]
param(
    [string]$CudaPath = $env:CUDA_PATH_OVERRIDE
)

$ErrorActionPreference = "Stop"

if (-not $CudaPath -or $CudaPath.Trim().Length -eq 0) {
    $CudaPath = $env:CUDA_PATH
}
if (-not $CudaPath -or $CudaPath.Trim().Length -eq 0) {
    $CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
}

$cudaBin = Join-Path $CudaPath "bin"
$cudaBin64 = Join-Path $CudaPath "bin\x64"

if (-not (Test-Path $cudaBin)) {
    throw "CUDA bin directory not found: $cudaBin"
}

$env:CUDA_PATH = $CudaPath
$pathEntries = @($cudaBin)
if (Test-Path $cudaBin64) {
    $pathEntries += $cudaBin64
}
$pathEntries += $env:PATH
$env:PATH = ($pathEntries | Select-Object -Unique) -join ';'

Write-Host "CUDA_PATH set to $CudaPath"
Write-Host "PATH updated with CUDA bin directories for this shell."
