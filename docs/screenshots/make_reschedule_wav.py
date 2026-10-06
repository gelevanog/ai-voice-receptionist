"""Simulated caller for the interruption + reschedule demo call (see make_caller_wav.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from callie.audio.pcm import write_wav
from callie.audio.resample import resample
from callie.config import Settings
from callie.tts import PiperTTS

LINES = [
    (12.0, "Hi, it's Sofia Rossi. I have a cleaning tomorrow afternoon and I need to move it."),
    (15.0, "Do you have anything on Thursday?"),
    (7.5, "Sorry, sorry, in the morning, please."),
    (16.0, "The first one works."),
    (20.0, "Yes, that's right."),
    (15.0, "No, that's all. Thanks, bye!"),
]

if __name__ == "__main__":
    tts = PiperTTS(Settings().models_dir / "en_US-libritts_r-medium.onnx", speaker_id=199)
    parts = []
    for gap, text in LINES:
        parts.append(np.zeros(int(gap * 16000), dtype=np.float32))
        speech = tts.synthesize_sync(text)
        audio = resample(speech.audio, speech.rate, 16000)
        parts.append((audio / (np.max(np.abs(audio)) + 1e-6) * 0.6).astype(np.float32))
    parts.append(np.zeros(16000 * 8, dtype=np.float32))
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/reschedule_caller.wav")
    write_wav(out, np.concatenate(parts), 16000)
    print(out)
