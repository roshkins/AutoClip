# Integration test fixtures

Generate a short synthetic TS file from the repository root:

```powershell
.\scripts\generate_test_fixture.ps1
cargo test --locked --no-default-features --test reprocess_ts -- --ignored --nocapture
```

This requires FFmpeg/ffprobe, including the `libx264` encoder. The fixture
contains a test pattern and a tone; no stream, model download or GPU is
needed. Generated TS files are gitignored.

The test is ignored by default. When explicitly requested, it fails if the
fixture, required tools or render are missing. Cargo supplies the exact
application binary; output goes into a temporary working directory's
`clips/` folder. The test checks file existence, nonzero size and source/
output duration agreement within 0.5 seconds. It does not test voice triggers
or live platform access.

You can also replace `tests/data/sample.ts` with your own fixture:

A good fixture:
- ≤30 seconds, single video + audio track.
- Captured with `autoclip` itself (`run_autoclip_with_cuda.ps1` produces TS
  alongside the final mp4 by default), or recorded with
  `ffmpeg -t 20 -i <hls-url> -c copy sample.ts`.
- ≤10 MB so it stays cheap to copy around.
