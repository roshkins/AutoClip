param(
    [Parameter(Mandatory = $true)]
    [Alias("Input")]
    [string]$InputPath,
    [string]$Output,
    [string]$Resolution = "1080x1920",
    [double]$Start = 0,
    [double]$Duration = 0,
    [double]$WakeSeconds = 0,
    [double]$WakeAt = 50,
    [double]$ClipLength = 60,
    [switch]$UseNvenc
)

$ffmpeg = if ($env:FFMPEG_BIN) { $env:FFMPEG_BIN } else { "ffmpeg" }
$ffprobe = if ($env:FFPROBE_BIN) { $env:FFPROBE_BIN } else { "ffprobe" }

if (-not (Test-Path -LiteralPath $InputPath)) {
    throw "Input not found: $InputPath"
}

if (-not $Output) {
    $Output = [System.IO.Path]::ChangeExtension($InputPath, ".mp4")
}

$w = 0
$h = 0
$parts = $Resolution -split "x"
if ($parts.Length -ne 2 -or -not [int]::TryParse($parts[0], [ref]$w) -or -not [int]::TryParse($parts[1], [ref]$h)) {
    throw "Resolution must be WxH (e.g., 1080x1920)"
}

$actual = 0.0
try {
    $probe = & $ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $InputPath
    if ($probe) {
        [void][double]::TryParse($probe.Trim(), [ref]$actual)
    }
} catch {
    $actual = 0.0
}

if ($WakeSeconds -gt 0) {
    $Start = $WakeSeconds - $WakeAt
    if ($Start -lt 0) {
        $Start = 0
    }
    $Duration = $ClipLength
}

if ($Duration -le 0 -and $actual -gt 0) {
    $Duration = $actual
}

if ($Duration -gt 0 -and $actual -gt 0) {
    $remaining = $actual - $Start
    if ($remaining -gt 0 -and $remaining + 0.5 -lt $Duration) {
        Write-Warning ("Clamping duration to {0:N1}s based on input length {1:N1}s" -f $remaining, $actual)
        $Duration = $remaining
    }
}

$vf = "scale=$w`:$h`:force_original_aspect_ratio=decrease,pad=$w`:$h`:(ow-iw)/2:(oh-ih)/2,format=yuv420p"

$args = @("-y", "-loglevel", "warning", "-fflags", "+genpts", "-i", $InputPath)
if ($Start -gt 0) {
    $args += @("-ss", ("{0:F3}" -f $Start))
}
if ($Duration -gt 0) {
    $args += @("-t", ("{0:F3}" -f $Duration))
}

$args += @("-map", "0:v:0", "-map", "0:a:0?", "-vf", $vf, "-fps_mode", "vfr")

if ($UseNvenc) {
    $args += @("-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-b:v", "0", "-cq", "23")
} else {
    $args += @("-c:v", "libx264", "-preset", "veryfast", "-crf", "23")
}

$args += @("-c:a", "aac", "-b:a", "160k", $Output)

& $ffmpeg @args
if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg failed with exit code $LASTEXITCODE"
}

Write-Host ("Wrote {0}" -f $Output)
