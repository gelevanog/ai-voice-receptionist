"""Speech to text for one caller turn (the endpointer hands over the whole utterance).

- `WhisperSTT`: faster-whisper (CTranslate2) on CPU, int8. `base.en` is the default: in the benchmark on this
  machine it transcribed a ~2 s utterance in ~0.27 s with a WER close to `small.en` on clean speech
  (`small.en` is ~3x slower and more robust in noise; see README > Results).
- `FakeSTT`: returns scripted lines, deterministic, for tests, CI and the zero-download demo.

Transcription is chunked per turn rather than streamed word by word: with end-of-turn detection already
waiting ~0.5 s of silence, transcribing the finished utterance on CPU adds ~0.3 s and gives Whisper the whole
sentence, which is more accurate than partial hypotheses.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from callie.audio.pcm import Audio

DOMAIN_PROMPT = (
    "Brightside Dental. Appointment, cleaning, checkup, filling, whitening, Dr. Patel, Delta Dental, Cigna, "
    "MetLife, Aetna, reschedule, cancel."
)


@dataclass(frozen=True)
class Transcript:
    text: str
    seconds: float  # processing time
    confidence: float = 1.0  # mean token probability where available
    model: str = ""


class STT(Protocol):
    @property
    def name(self) -> str: ...

    async def transcribe(self, audio: Audio) -> Transcript: ...


class FakeSTT:
    """Returns the next scripted line for every utterance (empty once the script is exhausted)."""

    def __init__(self, lines: Iterable[str] = ()) -> None:
        self.lines = list(lines)
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake-scripted"

    def queue(self, *lines: str) -> None:
        self.lines.extend(lines)

    async def transcribe(self, audio: Audio) -> Transcript:
        self.calls += 1
        text = self.lines.pop(0) if self.lines else ""
        return Transcript(text=text, seconds=0.0, model=self.name)


class WhisperSTT:
    def __init__(
        self,
        model: str = "base.en",
        *,
        threads: int = 4,
        compute_type: str = "int8",
        beam_size: int = 1,
        prompt: str | None = DOMAIN_PROMPT,
        download_root: str | None = None,
    ) -> None:
        self.model_name = model
        self.threads = threads
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.prompt = prompt
        self.download_root = download_root
        self._model: Any = None
        # One worker: CTranslate2 already parallelizes inside a call; concurrent calls would only fight for cores.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

    @property
    def name(self) -> str:
        return f"faster-whisper/{self.model_name}/{self.compute_type}"

    def load(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=self.threads,
                download_root=self.download_root,
            )

    def transcribe_sync(self, audio: Audio) -> Transcript:
        self.load()
        started = time.perf_counter()
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=self.beam_size,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
            initial_prompt=self.prompt,
        )
        parts = list(segments)
        text = " ".join(s.text.strip() for s in parts).strip()
        probs = [float(2.718281828**s.avg_logprob) for s in parts if s.avg_logprob is not None]
        no_speech = max((float(s.no_speech_prob) for s in parts), default=0.0)
        if no_speech > 0.8 and (not probs or max(probs) < 0.4):
            text = ""  # Whisper's hallucinations on noise ("Thank you.") are dropped
        return Transcript(
            text=text,
            seconds=time.perf_counter() - started,
            confidence=sum(probs) / len(probs) if probs else 0.0,
            model=self.name,
        )

    async def transcribe(self, audio: Audio) -> Transcript:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.transcribe_sync, audio)
