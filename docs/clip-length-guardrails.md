# Clip Length Guardrails

This project relies on ffmpeg to cut fixed-length clips from buffered TS data.
Do not clamp output duration using internal buffer estimates. Those estimates
are based on playlist durations and can drift from actual TS timestamps.

Rules:
- Keep the requested clip length fixed (e.g., 60s) when the TS exists.
- If you must validate length, probe the saved TS with ffprobe and base
  decisions on real PTS spans, not RollingBuffer totals.
- Use buffer estimates only for warnings/telemetry, never to shorten output.
- When a duration is specified, avoid `-shortest` unless you explicitly want
  to cut to the shortest stream.

If a future change proposes shortening clips, re-check this file first.
