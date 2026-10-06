"""Call recording: caller and agent on one timeline, saved as a stereo WAV (left: caller, right: Callie).

The caller track is written as audio arrives; the agent track is written where each chunk starts playing, and
erased past the point of a barge-in (the client dropped that audio, so the caller never heard it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from callie.audio.pcm import PIPELINE_RATE, Audio, write_wav
from callie.audio.resample import resample


class Recorder:
    def __init__(self, rate: int = PIPELINE_RATE, max_seconds: float = 1800.0) -> None:
        self.rate = rate
        self.max_samples = round(max_seconds * rate)
        self._caller = np.zeros(rate * 60, dtype=np.float32)
        self._agent = np.zeros(rate * 60, dtype=np.float32)
        self.caller_samples = 0
        self.agent_end = 0

    def _ensure(self, length: int) -> None:
        length = min(length, self.max_samples)
        if length > len(self._caller):
            size = max(length, len(self._caller) * 2)
            self._caller = np.pad(self._caller, (0, size - len(self._caller)))
            self._agent = np.pad(self._agent, (0, size - len(self._agent)))

    def add_caller(self, audio: Audio) -> None:
        end = min(self.caller_samples + len(audio), self.max_samples)
        self._ensure(end)
        self._caller[self.caller_samples : end] = audio[: end - self.caller_samples]
        self.caller_samples = end

    def add_agent(self, audio: Audio, rate: int, at_seconds: float) -> None:
        pcm = resample(audio, rate, self.rate) if rate != self.rate else audio
        start = max(0, round(at_seconds * self.rate))
        end = min(start + len(pcm), self.max_samples)
        if end <= start:
            return
        self._ensure(end)
        self._agent[start:end] += pcm[: end - start]
        self.agent_end = max(self.agent_end, end)

    def cut_agent_after(self, at_seconds: float) -> None:
        start = max(0, round(at_seconds * self.rate))
        if start < len(self._agent):
            self._agent[start:] = 0.0
            self.agent_end = min(self.agent_end, start)

    def stereo(self) -> Audio:
        length = max(self.caller_samples, self.agent_end)
        return np.stack([self._caller[:length], np.clip(self._agent[:length], -1, 1)], axis=1).astype(np.float32)

    def mono_mix(self) -> Audio:
        stereo = self.stereo()
        return np.asarray(np.clip(stereo.sum(axis=1) * 0.8, -1.0, 1.0), dtype=np.float32)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(path, self.stereo().reshape(-1), self.rate, channels=2)
        return path

    @property
    def seconds(self) -> float:
        return max(self.caller_samples, self.agent_end) / self.rate
