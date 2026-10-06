"""Test doubles: a recording transport, synthetic "speech", and a session factory with fake components."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import numpy as np

from callie.audio.pcm import Audio
from callie.config import Settings
from callie.llm.base import ChatModel
from callie.pipeline.session import CallSession, SessionConfig
from callie.runtime import Runtime, build_runtime
from callie.speech import SpeechStack
from callie.stt import FakeSTT
from callie.tts import FakeTTS
from callie.vad import EnergyVAD

RATE = 16000


def speech_like(seconds: float, amplitude: float = 0.3, seed: int = 0) -> Audio:
    """Voiced, syllable-modulated noise: loud enough for the energy VAD, sized like an utterance."""
    rng = np.random.default_rng(seed)
    n = round(seconds * RATE)
    t = np.arange(n) / RATE
    carrier = np.sin(2 * np.pi * 160 * t) + 0.5 * np.sin(2 * np.pi * 320 * t) + 0.2 * rng.standard_normal(n)
    envelope = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 3.0 * t))
    return (amplitude * carrier * envelope / 1.7).astype(np.float32)


def quiet(seconds: float) -> Audio:
    return (np.random.default_rng(1).standard_normal(round(seconds * RATE)) * 1e-4).astype(np.float32)


class RecordingTransport:
    name = "test"

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.audio_chunks = 0
        self.audio_seconds = 0.0
        self.clears = 0
        self.transferred: str | None = None
        self.hung_up = False

    async def send_audio(self, audio: Audio, rate: int) -> None:
        self.audio_chunks += 1
        self.audio_seconds += len(audio) / rate

    async def clear_audio(self) -> None:
        self.clears += 1

    async def send_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    async def transfer(self, reason: str) -> None:
        self.transferred = reason

    async def hangup(self) -> None:
        self.hung_up = True

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == kind]


def fake_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "now": "2026-10-06T09:30",
        "database_url": "sqlite://",
        "llm_provider": "fake",
        "llm_ledger": None,
        "seed_demo_data": False,
        "endpoint_silence_ms": 400,
    }
    values.update(overrides)
    return Settings(**values)


def make_session(
    lines: list[str],
    *,
    ms_per_char: float = 4.0,
    settings: Settings | None = None,
    config: SessionConfig | None = None,
    llm: ChatModel | None = None,
    runtime: Runtime | None = None,
    caller_phone: str | None = None,
) -> tuple[CallSession, RecordingTransport, FakeSTT]:
    settings = settings or fake_settings()
    runtime = runtime or build_runtime(settings)
    stt = FakeSTT(lines)
    speech = SpeechStack(stt=stt, tts=FakeTTS(ms_per_char=ms_per_char), vad_factory=EnergyVAD, settings=settings)
    transport = RecordingTransport()
    session = CallSession(
        runtime,
        speech,
        transport,
        config=config or SessionConfig(silence_prompt_s=30.0, save_record=True, record=True, recordings_dir=_tmp_dir()),
        llm=llm,
        caller_phone=caller_phone,
    )
    return session, transport, stt


def _tmp_dir() -> Any:
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp(prefix="callie-test-"))


async def stream(session: CallSession, audio: Audio, chunk_s: float = 0.02, speed: float = 1.0) -> None:
    """Feed audio in (approximately) real time, as a phone or browser would."""
    size = round(chunk_s * RATE)
    for start in range(0, len(audio), size):
        session.feed_audio(audio[start : start + size])
        await asyncio.sleep(chunk_s / speed)


async def say(session: CallSession, seconds: float = 0.6, *, trailing_silence: float = 0.6) -> None:
    await stream(session, np.concatenate([speech_like(seconds), quiet(trailing_silence)]))


async def wait_for(condition: Callable[[], bool], limit: float = 10.0, step: float = 0.02) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit
    while not condition():
        if loop.time() > deadline:
            raise TimeoutError("condition not met")
        await asyncio.sleep(step)


async def keep_line_open(session: CallSession, seconds: float) -> None:
    """Background line noise while waiting, like a real call (the endpointer needs a continuous stream)."""
    await stream(session, quiet(seconds))


async def until_agent_done(session: CallSession, limit: float = 15.0) -> None:
    """Feed silence until the current turn has finished speaking."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit
    await stream(session, quiet(0.1))
    while (
        session.player.active or (session._turn_task is not None and not session._turn_task.done())
    ) and not session.closed.is_set():
        if loop.time() > deadline:
            raise TimeoutError("agent still speaking")
        await stream(session, quiet(0.05))
