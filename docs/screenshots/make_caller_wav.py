"""Make the simulated caller for the live-call screenshot: Piper lines separated by silence, as one WAV.

Chrome plays it as the microphone (`--use-file-for-fake-audio-capture`), so the screenshot shows a real browser call
through the whole pipeline. Gaps are long enough for Callie to answer each line.
Run: uv run python docs/screenshots/make_caller_wav.py data/hero_caller.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from callie.audio.pcm import write_wav
from callie.audio.resample import resample
from callie.config import Settings
from callie.tts import PiperTTS

LINES = [
    (12.0, "Hi, I'd like to book a cleaning for next Tuesday afternoon, please."),
    (19.0, "The three o'clock one, please."),
    (14.0, "It's Jane Doe, and my number is five five five, one two three, four five six seven."),
    (21.0, "Yes, that's right."),
    (14.0, "Do you take Delta Dental?"),
    (16.0, "No, that's all. Thank you, bye!"),
]


def main(out: Path) -> None:
    tts = PiperTTS(Settings().models_dir / "en_US-libritts_r-medium.onnx", speaker_id=12)
    parts = []
    for gap, text in LINES:
        parts.append(np.zeros(int(gap * 16000), dtype=np.float32))
        speech = tts.synthesize_sync(text)
        audio = resample(speech.audio, speech.rate, 16000)
        parts.append((audio / (np.max(np.abs(audio)) + 1e-6) * 0.6).astype(np.float32))
    parts.append(np.zeros(16000 * 8, dtype=np.float32))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out, np.concatenate(parts), 16000)
    print(out, sum(len(p) for p in parts) / 16000, "s")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/hero_caller.wav"))
