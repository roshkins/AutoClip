param(
    [string[]]$Clips,
    [int]$ClipCount = 8,
    [int]$FrameChecks = 5,
    [double]$StepSeconds = 1.5,
    [double]$FaceScore = 0.5,
    [double]$ContextScale = 1.8,
    [double]$CenterThresh = 0.4,
    [switch]$FullScan = $true,
    [switch]$SaveFrames = $true
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$autoclip = Join-Path $root "target\\release\\autoclip.exe"

if (!(Test-Path $autoclip)) {
    throw "autoclip.exe not found. Build first: cargo build --release"
}

$modelPath = Join-Path $root "models\\face_detection_yunet_2023mar.onnx"
if (!(Test-Path $modelPath)) {
    Write-Warning "Face model missing at $modelPath"
}

$cudaPath = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1"
if (Test-Path $cudaPath) {
    $env:CUDA_PATH = $cudaPath
    $env:PATH = "$cudaPath\\bin;$cudaPath\\bin\\x64;$env:PATH"
}

function Parse-Float([string]$value) {
    return [double]::Parse($value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Run-Cmd([string]$cmd) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        return cmd /c $cmd 2>&1
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Get-Duration([string]$path) {
    $cmd = "ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 `"$path`" 2>nul"
    $output = Run-Cmd $cmd
    if (!$output) {
        return $null
    }
    try {
        return Parse-Float(($output | Select-Object -First 1).Trim())
    } catch {
        return $null
    }
}

function Run-Autoclip([string]$clipPath) {
    $cmd = "`"$autoclip`" demo-detect `"$clipPath`""
    return Run-Cmd $cmd
}

function Extract-Crop([string]$clipPath, [double]$time, [pscustomobject]$crop, [string]$outPath) {
    $timeStr = $time.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    $cropExpr = "crop=iw*$($crop.W):ih*$($crop.H):iw*$($crop.X):ih*$($crop.Y)"
    $cmd = "ffmpeg -hide_banner -loglevel error -ss $timeStr -i `"$clipPath`" -frames:v 1 -vf `"$cropExpr`" -y `"$outPath`""
    $null = Run-Cmd $cmd
    return Test-Path $outPath
}

function Parse-Detection([string[]]$lines) {
    $text = ($lines -join "`n")
    $faceMatch = [regex]::Match($text, "demo-detect: face x=([0-9.]+) y=([0-9.]+) w=([0-9.]+) h=([0-9.]+)")
    if (!$faceMatch.Success) {
        return [pscustomobject]@{
            Found = $false
        }
    }

    $timeMatch = [regex]::Match($text, "clip detect: face pick t=([0-9.]+)s score=([0-9.]+)")
    $time = 0.0
    $score = 0.0
    if ($timeMatch.Success) {
        $time = Parse-Float $timeMatch.Groups[1].Value
        $score = Parse-Float $timeMatch.Groups[2].Value
    }

    return [pscustomobject]@{
        Found = $true
        X = Parse-Float $faceMatch.Groups[1].Value
        Y = Parse-Float $faceMatch.Groups[2].Value
        W = Parse-Float $faceMatch.Groups[3].Value
        H = Parse-Float $faceMatch.Groups[4].Value
        Time = $time
        Score = $score
    }
}

function Clamp([double]$v, [double]$min, [double]$max) {
    if ($v -lt $min) { return $min }
    if ($v -gt $max) { return $max }
    return $v
}

function Rect-Center([pscustomobject]$rect) {
    return [pscustomobject]@{
        X = $rect.X + $rect.W / 2.0
        Y = $rect.Y + $rect.H / 2.0
    }
}

function Expand-Rect([double]$x, [double]$y, [double]$w, [double]$h, [double]$scale) {
    $scale = [math]::Max($scale, 1.0)
    $cx = $x + $w / 2.0
    $cy = $y + $h / 2.0
    $newW = Clamp ($w * $scale) 0.12 1.0
    $newH = Clamp ($h * $scale) 0.12 1.0
    $maxW = [math]::Max([math]::Min($cx, 1.0 - $cx) * 2.0, 0.0)
    $maxH = [math]::Max([math]::Min($cy, 1.0 - $cy) * 2.0, 0.0)
    if ($maxW -gt 0.0) { $newW = [math]::Min($newW, $maxW) }
    if ($maxH -gt 0.0) { $newH = [math]::Min($newH, $maxH) }
    $newX = $cx - $newW / 2.0
    $newY = $cy - $newH / 2.0
    $newX = Clamp $newX 0.0 (1.0 - $newW)
    $newY = Clamp $newY 0.0 (1.0 - $newH)
    return [pscustomobject]@{
        X = $newX
        Y = $newY
        W = $newW
        H = $newH
    }
}

function Build-TrackTimes([double]$duration, [double]$start, [double]$step) {
    $start = [math]::Max(0.0, $start)
    if (!$duration -or $duration -le 0.0) {
        return @($start)
    }
    $end = [math]::Max($start, $duration - 0.05)
    $span = $end - $start
    if ($span -le 0.0) {
        return @($start)
    }
    $step = [math]::Max(0.05, $step)
    $count = [int][math]::Floor($span / $step) + 1
    if ($count -gt 60) {
        $count = 60
        $step = if ($count -gt 1) { $span / ($count - 1) } else { 0.0 }
    }
    $times = @()
    for ($i = 0; $i -lt $count; $i++) {
        $times += ($start + $step * $i)
    }
    return $times
}

function Select-TrackPoints([object[]]$points) {
    if ($points.Count -eq 0) {
        return @()
    }
    $clusters = @()
    foreach ($p in $points) {
        $center = Rect-Center $p.Rect
        $bestIdx = $null
        $bestDist = 1.0
        for ($i = 0; $i -lt $clusters.Count; $i++) {
            $c = $clusters[$i]
            $dist = [math]::Max([math]::Abs($center.X - $c.CenterX), [math]::Abs($center.Y - $c.CenterY))
            if ($dist -lt 0.12 -and $dist -lt $bestDist) {
                $bestDist = $dist
                $bestIdx = $i
            }
        }
        if ($null -eq $bestIdx) {
            $clusters += [pscustomobject]@{
                CenterX = $center.X
                CenterY = $center.Y
                SumX = $center.X
                SumY = $center.Y
                Count = 1
            }
        } else {
            $c = $clusters[$bestIdx]
            $c.Count += 1
            $c.SumX += $center.X
            $c.SumY += $center.Y
            $c.CenterX = $c.SumX / $c.Count
            $c.CenterY = $c.SumY / $c.Count
            $clusters[$bestIdx] = $c
        }
    }
    $best = $clusters | Sort-Object -Property Count -Descending | Select-Object -First 1
    $filtered = @()
    foreach ($p in $points) {
        $center = Rect-Center $p.Rect
        $dist = [math]::Max([math]::Abs($center.X - $best.CenterX), [math]::Abs($center.Y - $best.CenterY))
        if ($dist -le 0.12) {
            $filtered += $p
        }
    }
    return $filtered
}

function Track-For-Clip([string]$clipPath, [double]$duration, [double]$step, [double]$faceScore, [double]$contextScale) {
    $times = Build-TrackTimes $duration 1.0 $step
    $points = @()
    foreach ($t in $times) {
        $env:CLIP_DETECT_FULL = "0"
        $env:CLIP_DETECT_SAMPLES = "1"
        $env:CLIP_DETECT_START = $t.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        $env:CLIP_DETECT_STEP = "0.1"
        $env:CLIP_FACE_SCORE = $faceScore.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        $env:CLIP_FACE_TRACK = "0"
        $output = Run-Autoclip $clipPath
        $det = Parse-Detection $output
        if ($det.Found) {
            $points += [pscustomobject]@{
                Time = $t
                Rect = [pscustomobject]@{
                    X = $det.X
                    Y = $det.Y
                    W = $det.W
                    H = $det.H
                }
            }
        }
    }
    $points = Select-TrackPoints $points
    $points = $points | Sort-Object -Property Time
    if ($points.Count -eq 0) {
        return $null
    }
    $maxW = 0.12
    $maxH = 0.12
    $expanded = @()
    foreach ($p in $points) {
        $rect = Expand-Rect $p.Rect.X $p.Rect.Y $p.Rect.W $p.Rect.H $contextScale
        $maxW = [math]::Max($maxW, $rect.W)
        $maxH = [math]::Max($maxH, $rect.H)
        $center = Rect-Center $rect
        $expanded += [pscustomobject]@{
            Time = $p.Time
            CenterX = $center.X
            CenterY = $center.Y
            W = $rect.W
            H = $rect.H
        }
    }
    return [pscustomobject]@{
        Points = $expanded
        MaxW = $maxW
        MaxH = $maxH
    }
}

function Interpolate-TrackPoint([object[]]$points, [double]$time) {
    if (!$points -or $points.Count -eq 0) {
        return $null
    }
    if ($points.Count -eq 1) {
        return $points[0]
    }
    if ($time -le $points[0].Time) {
        return $points[0]
    }
    $last = $points[$points.Count - 1]
    if ($time -ge $last.Time) {
        return $last
    }
    for ($i = 0; $i -lt $points.Count - 1; $i++) {
        $a = $points[$i]
        $b = $points[$i + 1]
        if ($time -le $b.Time) {
            $span = $b.Time - $a.Time
            if ($span -le 0.0001) {
                return $b
            }
            $t = ($time - $a.Time) / $span
            return [pscustomobject]@{
                CenterX = $a.CenterX + ($b.CenterX - $a.CenterX) * $t
                CenterY = $a.CenterY + ($b.CenterY - $a.CenterY) * $t
                W = $a.W + ($b.W - $a.W) * $t
                H = $a.H + ($b.H - $a.H) * $t
            }
        }
    }
    return $last
}

function Crop-At-Time([pscustomobject]$track, [double]$time) {
    if ($null -eq $track) {
        return $null
    }
    $points = $track.Points
    $selected = Interpolate-TrackPoint $points $time
    if ($null -eq $selected) {
        return $null
    }
    $w = Clamp $selected.W 0.12 1.0
    $h = Clamp $selected.H 0.12 1.0
    $centerX = Clamp $selected.CenterX 0.0 1.0
    $centerY = Clamp $selected.CenterY 0.0 1.0
    $x = Clamp ($centerX - $w / 2.0) 0.0 (1.0 - $w)
    $y = Clamp ($centerY - $h / 2.0) 0.0 (1.0 - $h)
    return [pscustomobject]@{
        X = $x
        Y = $y
        W = $w
        H = $h
    }
}

if (!$Clips -or $Clips.Count -eq 0) {
    $all = Get-ChildItem -Path (Join-Path $root "clips") -Filter "clip_*.mp4" -File |
        Where-Object { $_.Length -gt 0 } |
        Sort-Object Name
    if ($all.Count -eq 0) {
        throw "No clip_*.mp4 files found under clips"
    }
    if ($ClipCount -ge $all.Count) {
        $Clips = $all.FullName
    } else {
        $Clips = @()
        for ($i = 0; $i -lt $ClipCount; $i++) {
            $idx = [int][math]::Round($i * ($all.Count - 1) / ($ClipCount - 1))
            $Clips += $all[$idx].FullName
        }
    }
}

$outDir = Join-Path $root "face_eval"
$frameDir = Join-Path $outDir "frames"
$cropDir = Join-Path $outDir "crops"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
if ($SaveFrames) {
    New-Item -ItemType Directory -Force -Path $frameDir | Out-Null
    New-Item -ItemType Directory -Force -Path $cropDir | Out-Null
}

$summary = @()
$details = @()

foreach ($clip in $Clips) {
    $clipPath = Resolve-Path $clip
    Write-Host "==> $clipPath"
    $duration = Get-Duration $clipPath

    $env:CLIP_DETECT_FULL = if ($FullScan) { "1" } else { "0" }
    $env:CLIP_DETECT_STEP = $StepSeconds.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    $env:CLIP_FACE_SCORE = $FaceScore.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    $env:CLIP_DETECT_SAMPLES = "3"
    $env:CLIP_DETECT_START = "1.0"
    $env:CLIP_FACE_TRACK = if ($FullScan) { "1" } else { "0" }

    $pickOutput = Run-Autoclip $clipPath
    $pick = Parse-Detection $pickOutput
    if (!$pick.Found) {
        $summary += [pscustomobject]@{
            Clip = $clipPath.Path
            Duration = $duration
            FaceFound = $false
            FaceTime = $null
            FaceScore = $null
            CenterPassRate = 0
            FramesChecked = 0
        }
        Write-Host "  no face detected"
        continue
    }

    $track = if ($FullScan) {
        Track-For-Clip $clipPath $duration $StepSeconds $FaceScore $ContextScale
    } else {
        $null
    }
    $staticCrop = Expand-Rect $pick.X $pick.Y $pick.W $pick.H $ContextScale

    $times = @()
    if ($duration -and $FrameChecks -gt 1) {
        $start = [math]::Max(0.0, [math]::Min(1.0, $duration * 0.05))
        $end = [math]::Max($start, $duration - 0.1)
        for ($i = 0; $i -lt $FrameChecks; $i++) {
            $t = $start + ($end - $start) * $i / ($FrameChecks - 1)
            $times += $t
        }
    } else {
        $times += $pick.Time
    }

    $framesChecked = 0
    $framesPassing = 0
    $fullPassing = 0
    $cropChecked = 0
    $cropPassing = 0
    $cropFoundCount = 0
    foreach ($t in $times) {
        $env:CLIP_DETECT_FULL = "0"
        $env:CLIP_DETECT_SAMPLES = "1"
        $env:CLIP_DETECT_START = $t.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        $env:CLIP_DETECT_STEP = "0.1"
        $env:CLIP_FACE_SCORE = $FaceScore.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        $env:CLIP_FACE_TRACK = "0"
        $frameOutput = Run-Autoclip $clipPath
        $frame = Parse-Detection $frameOutput
        $crop = if ($track) { Crop-At-Time $track $t } else { $staticCrop }
        if ($null -eq $crop) { $crop = $staticCrop }
        $centerX = $crop.X + $crop.W / 2.0
        $centerY = $crop.Y + $crop.H / 2.0
        $halfW = $crop.W / 2.0
        $halfH = $crop.H / 2.0
        $cropPass = $false
        $cropCenterErr = $null
        $cropFound = $false
        if ($SaveFrames) {
            $base = [System.IO.Path]::GetFileNameWithoutExtension($clipPath.Path)
            $timeTag = ("{0:N2}" -f $t).Replace(".", "_")
            $cropPath = Join-Path $cropDir "$base`_t$timeTag`_crop.png"
            if (Extract-Crop $clipPath $t $crop $cropPath) {
                $env:CLIP_DETECT_START = "0.0"
                $env:CLIP_FACE_TRACK = "0"
                $cropOutput = Run-Autoclip $cropPath
                $cropDetect = Parse-Detection $cropOutput
                if ($cropDetect.Found) {
                    $cropFound = $true
                    $cx = $cropDetect.X + $cropDetect.W / 2.0
                    $cy = $cropDetect.Y + $cropDetect.H / 2.0
                    $cropCenterErr = [math]::Max([math]::Abs($cx - 0.5) / 0.5, [math]::Abs($cy - 0.5) / 0.5)
                    $cropPass = $cropCenterErr -le $CenterThresh
                }
            }
            $cropChecked += 1
            if ($cropFound) { $cropFoundCount += 1 }
            if ($cropPass) { $cropPassing += 1 }
        }

        if ($frame.Found) {
            $fx = $frame.X + $frame.W / 2.0
            $fy = $frame.Y + $frame.H / 2.0
            $dx = if ($halfW -gt 0.0) { [math]::Abs($fx - $centerX) / $halfW } else { 0.0 }
            $dy = if ($halfH -gt 0.0) { [math]::Abs($fy - $centerY) / $halfH } else { 0.0 }
            $centerErr = [math]::Max($dx, $dy)
            $inside = ($fx -ge $crop.X) -and ($fx -le ($crop.X + $crop.W)) -and
                ($fy -ge $crop.Y) -and ($fy -le ($crop.Y + $crop.H))
            $pass = $inside -and ($centerErr -le $CenterThresh)
            $combinedPass = $pass -and (!$SaveFrames -or $cropPass)
            if ($pass) { $fullPassing += 1 }
            if ($combinedPass) { $framesPassing += 1 }
            $framesChecked += 1
            $details += [pscustomobject]@{
                Clip = $clipPath.Path
                SampleTime = $t
                FaceFound = $true
                FaceX = $frame.X
                FaceY = $frame.Y
                FaceW = $frame.W
                FaceH = $frame.H
                CropX = $crop.X
                CropY = $crop.Y
                CropW = $crop.W
                CropH = $crop.H
                CenterErr = $centerErr
                Pass = $pass
                CropFaceFound = $cropFound
                CropCenterErr = $cropCenterErr
                CropPass = $cropPass
                CombinedPass = $combinedPass
            }
        } else {
            $framesChecked += 1
            $details += [pscustomobject]@{
                Clip = $clipPath.Path
                SampleTime = $t
                FaceFound = $false
                FaceX = $null
                FaceY = $null
                FaceW = $null
                FaceH = $null
                CropX = $crop.X
                CropY = $crop.Y
                CropW = $crop.W
                CropH = $crop.H
                CenterErr = $null
                Pass = $false
                CropFaceFound = $cropFound
                CropCenterErr = $cropCenterErr
                CropPass = $cropPass
                CombinedPass = $false
            }
        }
    }

    $fullRate = if ($framesChecked -gt 0) { $fullPassing / $framesChecked } else { 0 }
    $combinedRate = if ($framesChecked -gt 0) { $framesPassing / $framesChecked } else { 0 }
    $cropPassRate = if ($cropChecked -gt 0) { $cropPassing / $cropChecked } else { 0 }
    $cropFoundRate = if ($cropChecked -gt 0) { $cropFoundCount / $cropChecked } else { 0 }
    $summary += [pscustomobject]@{
        Clip = $clipPath.Path
        Duration = $duration
        FaceFound = $true
        FaceTime = $pick.Time
        FaceScore = $pick.Score
        FullPassRate = $fullRate
        CropPassRate = $cropPassRate
        CropFoundRate = $cropFoundRate
        CombinedPassRate = $combinedRate
        FramesChecked = $framesChecked
    }

    Write-Host ("  face at t={0:N2}s score={1:N3} combined pass={2:P0}" -f $pick.Time, $pick.Score, $combinedRate)
}

$summaryPath = Join-Path $outDir "summary.csv"
$detailsPath = Join-Path $outDir "details.csv"
$summary | Export-Csv -NoTypeInformation -Path $summaryPath
$details | Export-Csv -NoTypeInformation -Path $detailsPath

Write-Host ""
Write-Host "Summary -> $summaryPath"
Write-Host "Details -> $detailsPath"
