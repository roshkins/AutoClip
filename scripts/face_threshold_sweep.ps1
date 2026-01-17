param(
    [string[]]$Positives,
    [string[]]$Negatives,
    [double]$Start = 0.2,
    [double]$End = 0.7,
    [double]$Step = 0.05,
    [int]$SampleCount = 1,
    [double]$SampleStart = 0.0,
    [double]$SampleStep = 1.0,
    [switch]$Batch,
    [switch]$Release
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$autoclip = if ($Release) {
    Join-Path $root "target\\release\\autoclip.exe"
} else {
    Join-Path $root "target\\debug\\autoclip.exe"
}

if (!(Test-Path $autoclip)) {
    throw "autoclip.exe not found at $autoclip. Build first."
}

$defaultPos = Join-Path $root "face_eval\\positives"
$defaultNeg = Join-Path $root "face_eval\\negatives"
if (!$Positives -and (Test-Path $defaultPos)) {
    $Positives = @($defaultPos)
}
if (!$Negatives -and (Test-Path $defaultNeg)) {
    $Negatives = @($defaultNeg)
}
if (!$Positives -or !$Negatives) {
    throw "Provide -Positives and -Negatives or create face_eval\\positives and face_eval\\negatives"
}

$outDir = Join-Path $root "face_eval"
if ($Batch) {
    if ($Positives.Count -ne 1 -or !(Test-Path $Positives[0] -PathType Container)) {
        throw "Batch mode requires -Positives to point at a single directory"
    }
    if ($Negatives.Count -ne 1 -or !(Test-Path $Negatives[0] -PathType Container)) {
        throw "Batch mode requires -Negatives to point at a single directory"
    }
    $posDir = (Resolve-Path $Positives[0]).Path
    $negDir = (Resolve-Path $Negatives[0]).Path
    $outPath = Join-Path $outDir "threshold_sweep.csv"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $cmd = "`"$autoclip`" face-sweep `"$posDir`" `"$negDir`" $Start $End $Step `"$outPath`" --clip-detect-samples $SampleCount --clip-detect-start $SampleStart --clip-detect-step $SampleStep"
    Write-Host "Running batch sweep: $cmd"
    cmd /c $cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Batch sweep failed with exit code $LASTEXITCODE"
    }
    if (!(Test-Path $outPath)) {
        throw "Batch sweep completed without creating $outPath"
    }
    return
}

$exts = @("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.mp4", "*.mkv", "*.mov", "*.ts")

function Expand-Inputs([string[]]$paths) {
    $items = @()
    foreach ($p in $paths) {
        if (!$p) { continue }
        $resolved = Resolve-Path $p -ErrorAction SilentlyContinue
        foreach ($r in $resolved) {
            if (Test-Path $r -PathType Container) {
                foreach ($ext in $exts) {
                    $items += Get-ChildItem -Path $r -Filter $ext -File -ErrorAction SilentlyContinue
                }
            } elseif (Test-Path $r -PathType Leaf) {
                $items += Get-Item -LiteralPath $r
            }
        }
    }
    return $items | Select-Object -Unique
}

function Format-Float([double]$value) {
    return $value.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

function Run-Detect([string]$path, [double]$score) {
    $env:CLIP_DETECT_FULL = "0"
    $env:CLIP_DETECT_SAMPLES = $SampleCount.ToString()
    $env:CLIP_DETECT_START = Format-Float $SampleStart
    $env:CLIP_DETECT_STEP = Format-Float $SampleStep
    $env:CLIP_FACE_SCORE = Format-Float $score
    $env:CLIP_FACE_TRACK = "0"
    $cmd = "`"$autoclip`" demo-detect `"$path`" --clip-gameplay=0"
    return cmd /c $cmd 2>&1
}

function Has-Face([string[]]$lines) {
    $text = ($lines -join "`n")
    return $text -match "demo-detect: face x="
}

$posItems = Expand-Inputs $Positives
$negItems = Expand-Inputs $Negatives
if ($posItems.Count -eq 0 -or $negItems.Count -eq 0) {
    throw "No samples found after expanding inputs."
}

$results = @()
for ($score = $Start; $score -le ($End + 0.0001); $score += $Step) {
    $posFound = 0
    foreach ($item in $posItems) {
        $lines = Run-Detect $item.FullName $score
        if (Has-Face $lines) {
            $posFound += 1
        }
    }

    $negFound = 0
    foreach ($item in $negItems) {
        $lines = Run-Detect $item.FullName $score
        if (Has-Face $lines) {
            $negFound += 1
        }
    }

    $posRate = if ($posItems.Count -gt 0) { $posFound / $posItems.Count } else { 0 }
    $negRate = if ($negItems.Count -gt 0) { $negFound / $negItems.Count } else { 0 }
    $precision = if (($posFound + $negFound) -gt 0) { $posFound / ($posFound + $negFound) } else { 0 }
    $scoreMetric = $posRate - $negRate
    $results += [pscustomobject]@{
        Score = $score
        Positives = $posItems.Count
        PosFound = $posFound
        PosRate = $posRate
        Negatives = $negItems.Count
        NegFound = $negFound
        NegRate = $negRate
        Precision = $precision
        ScoreMetric = $scoreMetric
    }
    Write-Host ("score={0:N2} pos={1}/{2} ({3:P0}) neg={4}/{5} ({6:P0})" -f $score, $posFound, $posItems.Count, $posRate, $negFound, $negItems.Count, $negRate)
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outPath = Join-Path $outDir "threshold_sweep.csv"
$results | Export-Csv -NoTypeInformation -Path $outPath
Write-Host "Wrote $outPath"

$best = $results | Sort-Object -Property ScoreMetric -Descending | Select-Object -First 1
if ($best) {
    Write-Host ("Recommended CLIP_FACE_SCORE={0:N2} (pos={1:P0} neg={2:P0})" -f $best.Score, $best.PosRate, $best.NegRate)
}
