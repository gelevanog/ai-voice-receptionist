"""Text to speech, one sentence at a time (the chunker feeds sentences as soon as the model completes them).

- `KokoroTTS`: Kokoro-82M (Apache-2.0 weights) via kokoro-onnx, fp32 ONNX on CPU, 24 kHz. Chosen for the agent
  voice: far more natural than Piper at ~0.25 real-time factor here. The int8 export was ~4x *slower* on this
  CPU (dynamic-quantized convolutions), so fp32 is used.
- `PiperTTS`: Piper VITS voices via piper-tts (GPL-3.0, optional extra), 22.05 kHz, ~0.04 real-time factor.
  Used for the simulated callers in the evaluation: a different engine and different voices than the agent.
- `FakeTTS`: deterministic tones whose length follows the text, for tests, CI and the zero-download demo.

Both real engines phonemize with eSpeak NG (GPL-3.0) through their Python packages.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from callie.audio.pcm import Audio


@dataclass(frozen=True)
class Speech:
    audio: Audio
    rate: int
    seconds: float  # synthesis time


class TTS(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def rate(self) -> int: ...

    async def synthesize(self, text: str) -> Speech: ...


class FakeTTS:
    """A soft two-tone hum, 60 ms per character: audible, deterministic and sized like real speech."""

    def __init__(self, rate: int = 16000, ms_per_char: float = 60.0, amplitude: float = 0.2) -> None:
        self._rate = rate
        self.ms_per_char = ms_per_char
        self.amplitude = amplitude

    @property
    def name(self) -> str:
        return "fake-tones"

    @property
    def rate(self) -> int:
        return self._rate

    def render(self, text: str) -> Audio:
        seconds = max(0.25, len(text) * self.ms_per_char / 1000)
        t = np.arange(round(seconds * self._rate)) / self._rate
        base = 180 + (sum(map(ord, text)) % 60)
        envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t) ** 2  # syllable-like modulation
        fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.02)
        audio = (np.sin(2 * np.pi * base * t) + 0.4 * np.sin(2 * np.pi * 2.5 * base * t)) * envelope * fade
        return np.asarray(self.amplitude * audio / 1.4, dtype=np.float32)

    async def synthesize(self, text: str) -> Speech:
        return Speech(self.render(text), self._rate, 0.0)


def _speakable(text: str) -> str:
    # TTS engines read "4 5 6 7" digit by digit; "$120" and "PM" are handled by the phonemizer.
    return text.replace(" & ", " and ")


class KokoroTTS:
    def __init__(
        self, model_path: Path, voices_path: Path, *, voice: str = "af_heart", speed: float = 1.05, threads: int = 4
    ) -> None:
        self.model_path = model_path
        self.voices_path = voices_path
        self.voice = voice
        self.speed = speed
        self.threads = threads
        self._engine: Any = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    @property
    def name(self) -> str:
        return f"kokoro-82m/{self.voice}"

    @property
    def rate(self) -> int:
        return 24000

    def load(self) -> None:
        if self._engine is None:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro

            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(self.model_path), options, providers=["CPUExecutionProvider"])
            self._engine = Kokoro.from_session(session, str(self.voices_path))

    def synthesize_sync(self, text: str) -> Speech:
        self.load()
        started = time.perf_counter()
        audio, rate = self._engine.create(_speakable(text), voice=self.voice, speed=self.speed, lang="en-us")
        return Speech(np.asarray(audio, dtype=np.float32), int(rate), time.perf_counter() - started)

    async def synthesize(self, text: str) -> Speech:
        return await asyncio.get_running_loop().run_in_executor(self._executor, self.synthesize_sync, text)


class PiperTTS:
    def __init__(self, model_path: Path, *, speaker_id: int | None = None, length_scale: float | None = None) -> None:
        self.model_path = model_path
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self._voice: Any = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piper")

    @property
    def name(self) -> str:
        speaker = f"#{self.speaker_id}" if self.speaker_id is not None else ""
        return f"piper/{self.model_path.stem}{speaker}"

    @property
    def rate(self) -> int:
        self.load()
        return int(self._voice.config.sample_rate)

    def load(self) -> None:
        if self._voice is None:
            from piper import PiperVoice

            self._voice = PiperVoice.load(str(self.model_path))

    def synthesize_sync(self, text: str) -> Speech:
        from piper.config import SynthesisConfig

        self.load()
        started = time.perf_counter()
        config = SynthesisConfig(speaker_id=self.speaker_id, length_scale=self.length_scale)
        chunks = [chunk.audio_float_array for chunk in self._voice.synthesize(_speakable(text), syn_config=config)]
        audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
        return Speech(audio, int(self._voice.config.sample_rate), time.perf_counter() - started)

    async def synthesize(self, text: str) -> Speech:
        return await asyncio.get_running_loop().run_in_executor(self._executor, self.synthesize_sync, text)
