# Configuration notes

Run from the repository root so relative helper and model paths resolve.
`config.env` is gitignored. The binary creates a local configuration when
needed and polls it during live runs. Values read only at startup need a
restart even if the file is updated.

The CLI supports the settings listed by `autoclip --help` and
`ENV_SPECS` in [src/cli.rs](../src/cli.rs). For those settings, lowercase
the name and replace underscores with hyphens: `CLIP_FACE_RATIO` becomes
`--clip-face-ratio=0.45`. This does not apply to every possible environment
variable. `cargo run` needs `--` before application arguments; the built
executable does not.

## Layout and audio

- Default layout is stacked. Use `--clip-layout=full` for full-frame video.
- Audio normalization uses FFmpeg loudnorm by default; disable with
  `--clip-audio-norm=false`.
- CUDA-backed ONNX face detection uses the `ort` Cargo feature (included in
  the default features). Set `CLIP_FACE_BACKEND=ort` and supply the ONNX
  Runtime DLL and CUDA provider DLLs through `PATH`, `ORT_DYLIB_PATH` or
  the supported runtime-path setting. Model files are separate assets.
- `CLIP_FACE_DUMP_DIR` writes annotated detection frames. Raw dumps use
  `--clip-face-dump-raw`; disable raw-score picking with
  `--clip-face-pick-raw=false`.
- Tile search defaults to minimum score 0.60 and maximum depth 3.
  `CLIP_FACE_TILE_MIN_SCORE` at or below zero disables tile search.
- `CLIP_FACE_CONTEXT` expands the face crop for a mid-shot (default 3.0,
  maximum 6.0). Increasing it above the default pins the face panel to 50%
  height. Crops widen to match the panel aspect ratio.
- Tracking samples default to 2-second spacing via
  `CLIP_FACE_TRACK_STEP`, sampled across the clip subject to a budget.
- `CLIP_FACE_BUDGET_SECS` overrides the face-analysis budget.

These tuning options have separate compute/model requirements. Start with
the full-frame example in the README before enabling additional detectors.

## Other commands

After a full build, reprocess a saved snapshot or validate an installed
gameplay model:

```powershell
.\target\debug\autoclip.exe reprocess-ts .\clips\clip_001.ts
.\target\debug\autoclip.exe check-gameplay-model .\models\clip-vit-base-patch32-xenova
.\target\debug\autoclip.exe --help
```

`reprocess-ts` writes a new MP4 under the example output path
(`./clips` relative to the working directory); it does not overwrite the
input TS. A successful local reprocess is separate from a live voice-trigger
test.

For CUDA build failures and toolchain setup, see
[BUILD_NOTES.md](../BUILD_NOTES.md). Its manual `vendor/whisper-rs-sys`
example describes a local troubleshooting checkout; `vendor/` is not
included in the repository, and a normal build uses Cargo dependencies.
