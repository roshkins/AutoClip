$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
if (-not (Test-Path $cudaRoot)) {
    Write-Error "CUDA root not found at: $cudaRoot"
    exit 1
}

$env:CUDA_PATH = $cudaRoot
$env:CUDAToolkit_ROOT = $cudaRoot
$env:PATH = "$cudaRoot\bin;$cudaRoot\bin\x64;$env:PATH"

cargo test @Args
