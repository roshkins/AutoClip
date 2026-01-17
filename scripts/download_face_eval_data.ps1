param(
    [int]$MaxPositives = 250,
    [int]$MaxNegatives = 250,
    [string]$OutDir = "face_eval",
    [string]$FacesArchive = "",
    [switch]$ForceDownload
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$outDir = Join-Path $root $OutDir
$datasetDir = Join-Path $outDir "datasets"
$posDir = Join-Path $outDir "positives"
$negDir = Join-Path $outDir "negatives"

New-Item -ItemType Directory -Force -Path $datasetDir | Out-Null
New-Item -ItemType Directory -Force -Path $posDir | Out-Null
New-Item -ItemType Directory -Force -Path $negDir | Out-Null

function Download-File([string[]]$urls, [string]$dest) {
    if ((Test-Path $dest) -and -not $ForceDownload) {
        Write-Host "Using cached file: $dest"
        return
    }
    foreach ($url in $urls) {
        try {
            Write-Host "Downloading $url -> $dest"
            Invoke-WebRequest -Uri $url -OutFile $dest -ErrorAction Stop
            return
        } catch {
            Write-Host "Download failed: $url ($($_.Exception.Message))"
        }
    }
    throw "All download attempts failed for $dest"
}

function Extract-Tgz([string]$archive, [string]$dest) {
    if (!(Test-Path $archive)) {
        throw "Archive not found: $archive"
    }
    if (Test-Path $dest) {
        return
    }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    tar -xf $archive -C $dest
}

function Extract-Zip([string]$archive, [string]$dest) {
    if (!(Test-Path $archive)) {
        throw "Archive not found: $archive"
    }
    if (Test-Path $dest) {
        return
    }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Expand-Archive -Path $archive -DestinationPath $dest -Force
}

function Sample-Files([object[]]$files, [int]$maxCount) {
    if ($maxCount -gt 0 -and $files.Count -gt $maxCount) {
        return $files | Get-Random -Count $maxCount
    }
    return $files
}

function Copy-Sampled([object[]]$files, [string]$destDir, [string]$prefix) {
    $idx = 0
    foreach ($file in $files) {
        $name = [System.IO.Path]::GetFileName($file.FullName)
        $target = Join-Path $destDir ("{0}_{1:D4}_{2}" -f $prefix, $idx, $name)
        Copy-Item -LiteralPath $file.FullName -Destination $target -Force
        $idx += 1
    }
}

function Get-ImageFiles([string]$rootDir) {
    if (!(Test-Path $rootDir)) {
        return @()
    }
    $exts = @(".jpg", ".jpeg", ".png")
    return Get-ChildItem -Path $rootDir -Recurse -File | Where-Object { $exts -contains $_.Extension.ToLowerInvariant() }
}

$facesArchivePath = $FacesArchive
if (!$facesArchivePath -or $facesArchivePath.Trim().Length -eq 0) {
    $facesArchivePath = Join-Path $datasetDir "archive.zip"
}
$facesDir = Join-Path $datasetDir "faces"
$useLocalFaces = Test-Path $facesArchivePath

$posFiles = @()
if ($useLocalFaces) {
    Extract-Zip $facesArchivePath $facesDir
    $posFiles = Get-ImageFiles $facesDir
} else {
    $fddbArchive = Join-Path $datasetDir "fddb_originalPics.tar.gz"
    $fddbDir = Join-Path $datasetDir "fddb"
    Download-File @(
        "https://vis-www.cs.umass.edu/fddb/originalPics.tar.gz",
        "http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz"
    ) $fddbArchive
    Extract-Tgz $fddbArchive $fddbDir
    $posFiles = Get-ImageFiles $fddbDir
}

$bsdsArchive = Join-Path $datasetDir "bsds500.tgz"
$bsdsDir = Join-Path $datasetDir "bsds500"
Download-File @(
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz",
    "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
) $bsdsArchive
Extract-Tgz $bsdsArchive $bsdsDir

$negFiles = Get-ImageFiles $bsdsDir

if ($posFiles.Count -eq 0) {
    if ($useLocalFaces) {
        throw "No face images found under $facesDir"
    }
    throw "No FDDB images found under $fddbDir"
}
if ($negFiles.Count -eq 0) { throw "No BSDS500 images found under $bsdsDir" }

$posSample = Sample-Files $posFiles $MaxPositives
$negSample = Sample-Files $negFiles $MaxNegatives

Get-ChildItem -Path $posDir -File | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $negDir -File | Remove-Item -Force -ErrorAction SilentlyContinue

Copy-Sampled $posSample $posDir "pos"
Copy-Sampled $negSample $negDir "neg"

Write-Host "Positives: $($posSample.Count) -> $posDir"
Write-Host "Negatives: $($negSample.Count) -> $negDir"
