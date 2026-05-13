# Integration test fixtures

Drop a small (≤30s) TS file at `sample.ts` in this directory to enable the
`reprocess_ts_produces_clip_matching_source_duration` integration test (see
`tests/reprocess_ts.rs`).

When the fixture is missing the test passes with a skip notice — this keeps
CI green without requiring a binary fixture committed to the repo.

A good fixture:
- ≤30 seconds, single video + audio track.
- Captured with `autoclip` itself (`run_autoclip_with_cuda.ps1` produces TS
  alongside the final mp4 by default), or recorded with
  `ffmpeg -t 20 -i <hls-url> -c copy sample.ts`.
- ≤10 MB so it stays cheap to copy around.
