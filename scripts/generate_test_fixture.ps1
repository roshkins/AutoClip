# Generate a small, local-only fixture for the opt-in TS-to-MP4 test.
# No stream access or downloaded model is required.
$ErrorActionPreference = 'Stop'
$fixtureDir = Join-Path $PSScriptRoot '..\tests\data'
$fixturePath = Join-Path $fixtureDir 'sample.ts'
New-Item -ItemType Directory -Force -Path $fixtureDir | Out-Null
& ffmpeg -hide_banner -loglevel error -y `
    -f lavfi -i 'testsrc2=size=640x360:rate=24' `
    -f lavfi -i 'sine=frequency=440:sample_rate=48000' `
    -t 6 -c:v mpeg2video -c:a mp2 -f mpegts $fixturePath
if ($LASTEXITCODE -ne 0) {
    throw "FFmpeg could not generate the test fixture (exit $LASTEXITCODE)."
}
Write-Host "Generated synthetic fixture: $fixturePath"
