# Agents Notes

- When using `cargo_*_fast.ps1`, pass app arguments via `-Args` (PowerShell treats `--` as ambiguous), e.g.:
  `powershell -File .\cargo_run_fast.ps1 -Args '--' 'reprocess-ts' 'clips\clip_403.ts'`.
