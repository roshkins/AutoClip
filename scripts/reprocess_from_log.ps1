param(
    [Parameter(Mandatory = $true)]
    [string]$LogPath,
    [Parameter(Mandatory = $true)]
    [string]$ClipName,
    [Parameter(Mandatory = $true)]
    [Alias("Input")]
    [string]$InputPath,
    [string]$Output,
    [string]$Resolution = "1080x1920",
    [switch]$UseNvenc
)

if (-not (Test-Path -LiteralPath $LogPath)) {
    throw "Log file not found: $LogPath"
}

if (-not (Test-Path -LiteralPath $InputPath)) {
    throw "Input not found: $InputPath"
}

$needle = [Regex]::Escape($ClipName)
$line = Get-Content -LiteralPath $LogPath | Where-Object {
    $_ -match "wake clip timing:" -and $_ -match $needle
} | Select-Object -Last 1

if (-not $line) {
    throw "No wake clip timing line found for '$ClipName' in $LogPath"
}

$match = [regex]::Match($line, "detect_offset_secs=([0-9.]+)")
if (-not $match.Success) {
    throw "detect_offset_secs not found in line: $line"
}

$wakeSeconds = [double]$match.Groups[1].Value
Write-Host ("Using wake seconds {0:N3} from log" -f $wakeSeconds)

$script = Join-Path $PSScriptRoot "reprocess_ts.ps1"
$args = @(
    "-InputPath", $InputPath,
    "-Resolution", $Resolution,
    "-WakeSeconds", $wakeSeconds
)
if ($Output) {
    $args += @("-Output", $Output)
}
if ($UseNvenc) {
    $args += "-UseNvenc"
}

& $script @args
