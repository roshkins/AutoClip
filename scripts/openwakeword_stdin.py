import os
import sys
import time
from typing import Optional

import numpy as np

try:
    from openwakeword.model import Model
except ImportError as exc:  # pragma: no cover - import guard
    sys.stderr.write(
        "openwakeword not installed. Run: pip install openwakeword numpy\n"
    )
    raise exc


# Defaults tuned for demo. Adjust via environment variables.
WAKE_THRESHOLD = float(os.environ.get("WAKE_THRESHOLD", "0.5"))
WAKE_MODELS = os.environ.get("WAKE_MODELS")
WAKE_MODELS_LIST = [m for m in (WAKE_MODELS.split(",") if WAKE_MODELS else []) if m]
CHUNK_MS = int(os.environ.get("WAKE_CHUNK_MS", "500"))  # 500ms frames
SAMPLE_RATE = 16000
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * (CHUNK_MS / 1000.0))
BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * 2  # int16

# Initialize model once.
model = Model(wakeword_models=WAKE_MODELS_LIST or None)


def process_stream(stream) -> None:
    """Read 16k mono int16 PCM from stdin, emit detections to stdout."""
    buf = b""
    last_emit: dict[str, float] = {}
    while True:
        data = stream.read(BYTES_PER_CHUNK)
        if not data:
            break
        buf += data
        while len(buf) >= BYTES_PER_CHUNK:
            chunk = buf[:BYTES_PER_CHUNK]
            buf = buf[BYTES_PER_CHUNK:]
            audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            scores = model.predict(audio)
            now = time.time()
            for name, score in scores.items():
                if score >= WAKE_THRESHOLD:
                    last_time = last_emit.get(name, 0)
                    if now - last_time > 0.75:  # simple debounce
                        sys.stdout.write(f"WAKE {name} {score:.3f}\n")
                        sys.stdout.flush()
                        last_emit[name] = now


def main() -> int:
    try:
        process_stream(sys.stdin.buffer)
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
