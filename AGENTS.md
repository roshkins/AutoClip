# Agents Notes

- When using `cargo_*_fast.ps1`, pass app arguments positionally (PowerShell treats `--` as ambiguous), e.g.:
  `powershell -File .\cargo_run_fast.ps1 reprocess-ts clips\clip_403.ts`.
