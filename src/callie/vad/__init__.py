"""Voice activity detection and endpointing (when has the caller finished their turn?).

Frames are 512 samples at 16 kHz (32 ms), the size Silero VAD expects. Two detectors share one interface:
- `SileroVAD`: the Silero VAD v6 network via ONNX Runtime (MIT, ~2 MB, ~0.15 ms per frame on CPU);
- `EnergyVAD`: a deterministic energy detector with an adaptive noise floor, used by the tests, CI and the
  zero-download demo.

`Endpointer` turns per-frame speech probabilities into events: speech start (after `min_speech_ms` of speech,
so a click is not a turn), ongoing speech (drives barge-in), and end of turn after `end_silence_ms` of silence.
The end-of-turn silence is the main latency/turn-taking trade-off: shorter answers faster but cuts callers off
mid-sentence when they pause.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from callie.audio.pcm import PIPELINE_RATE, Audio

FRAME = 512
FRAME_S = FRAME / PIPELINE_RATE


class VADModel(Protocol):
    def prob(self, frame: Audio) -> float: ...

    def reset(self) -> None: ...


class EnergyVAD:
    """Speech probability from frame energy relative to an adaptive noise floor (deterministic)."""

    def __init__(self, margin_db: float = 12.0, floor_db: float = -60.0, min_db: float = -45.0) -> None:
        self.margin_db = margin_db
        self.initial_floor = floor_db
        self.min_db = min_db
        self.floor_db = floor_db

    def prob(self, frame: Audio) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if len(frame) else 0.0
        db = 20.0 * float(np.log10(max(rms, 1e-6)))
        # The floor follows quiet frames quickly and loud frames very slowly.
        rate = 0.05 if db < self.floor_db + self.margin_db else 0.001
        self.floor_db += rate * (db - self.floor_db)
        threshold = max(self.floor_db + self.margin_db, self.min_db)
        return float(1.0 / (1.0 + np.exp(-(db - threshold) / 2.0)))

    def reset(self) -> None:
        self.floor_db = self.initial_floor


class SileroVAD:
    """Silero VAD v6 (ONNX). Keeps the recurrent state and the 64-sample context between frames."""

    CONTEXT = 64

    def __init__(self, model_path: Path, threads: int = 1) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self._sr = np.array(PIPELINE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def prob(self, frame: Audio) -> float:
        if len(frame) != FRAME:
            frame = np.pad(frame, (0, max(0, FRAME - len(frame))))[:FRAME]
        x = np.concatenate([self._context, frame.reshape(1, -1).astype(np.float32)], axis=1)
        out, self._state = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -self.CONTEXT :]
        return float(out[0][0])


EventKind = Literal["speech_start", "speech_ongoing", "speech_end"]


@dataclass
class VADEvent:
    kind: EventKind
    t: float  # stream time (s) of the frame that produced the event
    onset: float  # stream time the speech started
    speech_s: float = 0.0  # speech so far (ongoing) or in total (end)
    last_speech_t: float = 0.0  # stream time the last speech frame ended
    audio: Audio = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


class Endpointer:
    def __init__(
        self,
        vad: VADModel,
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 120,
        end_silence_ms: int = 550,
        preroll_ms: int = 320,
        max_utterance_s: float = 25.0,
    ) -> None:
        self.vad = vad
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.05)
        self.min_speech_frames = max(1, round(min_speech_ms / 1000 / FRAME_S))
        self.end_silence_frames = max(1, round(end_silence_ms / 1000 / FRAME_S))
        self.max_frames = round(max_utterance_s / FRAME_S)
        self._preroll: deque[Audio] = deque(maxlen=max(1, round(preroll_ms / 1000 / FRAME_S)))
        self._pending = np.zeros(0, dtype=np.float32)
        self.frames_seen = 0
        self.in_speech = False
        self._run: list[Audio] = []
        self._buffer: list[Audio] = []
        self._silence_frames = 0
        self._speech_frames = 0
        self._onset = 0.0
        self._last_speech_t = 0.0

    @property
    def stream_time(self) -> float:
        return self.frames_seen * FRAME_S

    def feed(self, audio: Audio) -> list[VADEvent]:
        """Feed any amount of 16 kHz audio; returns the events of the complete frames in it."""
        self._pending = np.concatenate([self._pending, audio.astype(np.float32, copy=False)])
        events: list[VADEvent] = []
        while len(self._pending) >= FRAME:
            frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
            events.extend(self._frame(frame))
        return events

    def _frame(self, frame: Audio) -> list[VADEvent]:
        prob = self.vad.prob(frame)
        self.frames_seen += 1
        now = self.stream_time
        if not self.in_speech:
            if prob >= self.threshold:
                if not self._run:
                    self._onset = now - FRAME_S
                self._run.append(frame)
                if len(self._run) >= self.min_speech_frames:
                    self.in_speech = True
                    self._buffer = [*self._preroll, *self._run]
                    self._speech_frames = len(self._run)
                    self._silence_frames = 0
                    self._last_speech_t = now
                    self._run = []
                    self._preroll.clear()
                    return [VADEvent("speech_start", now, self._onset, self._speech_frames * FRAME_S, now)]
            elif prob < self.neg_threshold:
                for held in self._run:
                    self._preroll.append(held)
                self._run = []
                self._preroll.append(frame)
            else:
                self._preroll.append(frame)
            return []
        self._buffer.append(frame)
        if prob >= self.neg_threshold:
            self._silence_frames = 0
            self._speech_frames += 1
            self._last_speech_t = now
        else:
            self._silence_frames += 1
        if self._silence_frames >= self.end_silence_frames or len(self._buffer) >= self.max_frames:
            return [self._end(now)]
        return [VADEvent("speech_ongoing", now, self._onset, self._speech_frames * FRAME_S, self._last_speech_t)]

    def _end(self, now: float) -> VADEvent:
        keep_tail = min(self._silence_frames, round(0.2 / FRAME_S))  # keep ~200 ms of trailing silence
        frames = self._buffer[: len(self._buffer) - self._silence_frames + keep_tail]
        audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        event = VADEvent("speech_end", now, self._onset, self._speech_frames * FRAME_S, self._last_speech_t, audio)
        self.in_speech = False
        self._buffer = []
        self._silence_frames = 0
        self._speech_frames = 0
        return event

    def reset(self) -> None:
        self.vad.reset()
        self.in_speech = False
        self._run, self._buffer = [], []
        self._preroll.clear()
        self._silence_frames = self._speech_frames = 0
