param(
    [string]$Destination = (Join-Path $PSScriptRoot "..\\models\\pose\\movenet_singlepose_thunder.onnx")
)

$source = "https://huggingface.co/Xenova/movenet-singlepose-thunder/resolve/main/onnx/model.onnx"
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$destinationDir = Split-Path -Parent $destinationPath

if (Test-Path $destinationPath) {
    Write-Host "MoveNet Thunder already present at $destinationPath"
    exit 0
}

if (-not (Test-Path $destinationDir)) {
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null
}

Write-Host "Downloading MoveNet Thunder (Apache-2.0) from $source"
Invoke-WebRequest -Uri $source -OutFile $destinationPath
Write-Host "Saved $destinationPath"
