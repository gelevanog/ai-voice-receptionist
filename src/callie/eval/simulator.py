"""Simulated phone calls through the real pipeline, in real time.

A `LineFeeder` plays the caller's side as a continuous 20 ms audio stream (line noise between utterances, like a
real call), so VAD, endpointing, barge-in and the latency waterfall run exactly as they do for a browser or
Twilio call. The caller's words come from an LLM (or a script), are spoken by Piper (a different engine and voice
than Callie's Kokoro), and pass through the scenario's channel: clean 16 kHz, a phone line (band-limited, 8 kHz
μ-law, the same codec path as Twilio), or a phone line with babble noise at 10 dB SNR.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from callie.audio.effects import add_noise, babble_noise, phone_channel
from callie.audio.pcm import PIPELINE_RATE, Audio, read_wav
from callie.audio.resample import resample
from callie.config import Settings
from callie.eval.caller import Caller
from callie.eval.checks import check_call
from callie.eval.metrics import corpus_wer
from callie.eval.scenarios import Scenario, get_scenario
from callie.llm.base import ChatModel, JsonDict
from callie.llm.factory import build_chat_model
from callie.pipeline.session import CallSession, SessionConfig
from callie.runtime import Runtime, build_runtime
from callie.scheduling.calendar import seed_demo_appointments
from callie.scheduling.db import Appointment
from callie.speech import SpeechStack, build_speech
from callie.tts import TTS, FakeTTS, PiperTTS

FRAME_S = 0.02
FRAME = round(FRAME_S * PIPELINE_RATE)
SPEECH_DBFS = -20.0


class SimTransport:
    name = "simulated"

    def __init__(self) -> None:
        self.clears: list[float] = []
        self.events: list[JsonDict] = []
        self.transferred: str | None = None
        self.hung_up = False
        self.audio_seconds = 0.0

    async def send_audio(self, audio: Audio, rate: int) -> None:
        self.audio_seconds += len(audio) / rate

    async def clear_audio(self) -> None:
        self.clears.append(time.monotonic())

    async def send_event(self, event: JsonDict) -> None:
        self.events.append(event)

    async def transfer(self, reason: str) -> None:
        self.transferred = reason

    async def hangup(self) -> None:
        self.hung_up = True


class LineFeeder:
    """Feeds the caller's side of the line in real time: queued utterances, otherwise line noise."""

    def __init__(self, session: CallSession, channel: str, rng: np.random.Generator) -> None:
        self.session = session
        self.channel = channel
        self.rng = rng
        self._queue: deque[Audio] = deque()
        self._current: Audio | None = None
        self._pos = 0
        self._noise = self._noise_bed()
        self._noise_pos = 0
        self._task: asyncio.Task[None] | None = None
        self.t0 = 0.0
        self.samples_fed = 0
        self.speech_started_wall: list[float] = []  # wall time of the first voiced sample of each utterance
        self._voiced_offset: deque[int] = deque()
        self.idle = asyncio.Event()
        self.idle.set()

    def _noise_bed(self) -> Audio:
        seconds = 6.0
        n = round(seconds * PIPELINE_RATE)
        if self.channel == "phone_noisy":
            level = 10 ** ((SPEECH_DBFS - 10.0) / 20)  # babble at 10 dB below speech level
            return phone_channel((babble_noise(n, PIPELINE_RATE, self.rng) * level).astype(np.float32), PIPELINE_RATE)
        floor = (self.rng.standard_normal(n) * 10 ** (-65 / 20)).astype(np.float32)
        return phone_channel(floor, PIPELINE_RATE) if self.channel == "phone" else floor

    def enqueue(self, audio: Audio, voiced_offset: int) -> None:
        self._queue.append(audio)
        self._voiced_offset.append(voiced_offset)
        self.idle.clear()

    def start(self) -> None:
        self.t0 = time.monotonic()
        self._task = asyncio.create_task(self._run(), name="line-feeder")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def stream_time(self) -> float:
        return self.samples_fed / PIPELINE_RATE

    async def _run(self) -> None:
        frames = 0
        while not self.session.closed.is_set():
            if self._current is None and self._queue:
                self._current = self._queue.popleft()
                self._pos = 0
                offset = self._voiced_offset.popleft()
                self.speech_started_wall.append(time.monotonic() + offset / PIPELINE_RATE)
            if self._current is not None:
                frame = self._current[self._pos : self._pos + FRAME]
                self._pos += FRAME
                if self._pos >= len(self._current):
                    self._current = None
                    if not self._queue:
                        self.idle.set()
                if len(frame) < FRAME:
                    frame = np.concatenate([frame, self._noise_frame(FRAME - len(frame))])
            else:
                frame = self._noise_frame(FRAME)
            self.session.feed_audio(frame)
            self.samples_fed += len(frame)
            frames += 1
            delay = self.t0 + frames * FRAME_S - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

    def _noise_frame(self, n: int) -> Audio:
        if self._noise_pos + n > len(self._noise):
            self._noise_pos = 0
        out = self._noise[self._noise_pos : self._noise_pos + n]
        self._noise_pos += n
        return out


def _normalize_level(audio: Audio, dbfs: float = SPEECH_DBFS) -> Audio:
    voiced = audio[np.abs(audio) > 1e-3]
    rms = float(np.sqrt(np.mean(np.square(voiced)))) if len(voiced) else 0.0
    if rms <= 0:
        return audio
    return np.asarray(np.clip(audio * (10 ** (dbfs / 20) / rms), -1, 1), dtype=np.float32)


def apply_channel(audio: Audio, channel: str, rng: np.random.Generator) -> Audio:
    if channel == "phone":
        return phone_channel(audio, PIPELINE_RATE)
    if channel == "phone_noisy":
        return phone_channel(add_noise(audio, 10.0, PIPELINE_RATE, rng, kind="babble"), PIPELINE_RATE)
    return audio


@dataclass
class CallerUtterance:
    text: str
    clean: Audio
    channel_audio: Audio
    stream_t: float  # when it started on the call timeline (s)
    interrupt: bool = False


@dataclass
class SimResult:
    scenario: Scenario
    summary: dict[str, Any]
    check: dict[str, Any]
    utterances: list[CallerUtterance] = field(default_factory=list)
    agent_lines: list[dict[str, Any]] = field(default_factory=list)
    barge_in_truth: list[dict[str, Any]] = field(default_factory=list)
    recording: str | None = None
    caller_errors: list[str] = field(default_factory=list)
    pipeline_wer: float | None = None
    raw_stt: list[str] = field(default_factory=list)


def prepare_runtime(settings: Settings, scenario: Scenario, llm: ChatModel, db_path: Path) -> tuple[Runtime, list[str]]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)
    scenario_settings = settings.model_copy(
        update={"now": scenario.now or settings.now or "2026-10-06T09:30", "seed_demo_data": False}
    )
    runtime = build_runtime(scenario_settings, llm=llm, database_url=f"sqlite:///{db_path}")
    seed_demo_appointments(runtime.calendar)
    ids: list[str] = []
    for item in scenario.setup:
        service = runtime.clinic.service(item.service)
        assert service is not None, item.service
        start = datetime.fromisoformat(item.start).replace(tzinfo=runtime.clinic.tz)
        with runtime.sessions() as db, db.begin():  # clear any seeded booking in the way
            for row in (
                db.query(Appointment)
                .filter(Appointment.resource_id == service.resource, Appointment.status == "booked")
                .all()
            ):
                if row.start < start + (row.end - row.start) and row.end > start:
                    db.delete(row)
        # Existing bookings are made "now" without the notice rule, as if booked weeks ago.
        appointment = Appointment(
            id=f"A-SETUP{len(ids)}",
            service_id=service.id,
            resource_id=service.resource,
            start=start,
            end=start + timedelta(minutes=service.minutes),
            patient_name=item.name,
            phone=item.phone,
            source="seed",
        )
        with runtime.sessions() as db, db.begin():
            db.add(appointment)
        ids.append(appointment.id)
    return runtime, ids


async def _wait_agent_done(session: CallSession, settle: float = 0.5, limit: float = 90.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit
    quiet_since: float | None = None
    while not session.closed.is_set() and loop.time() < deadline:
        busy = (
            session.player.active
            or (session._turn_task is not None and not session._turn_task.done())
            or session.endpointer.in_speech
        )
        if busy:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = loop.time()
        elif loop.time() - quiet_since >= settle:
            return
        await asyncio.sleep(0.05)


async def _wait_turn_started(session: CallSession, turn: int, limit: float = 6.0) -> None:
    """Until the utterance just spoken became a turn (end-of-turn silence + STT), or it was dropped."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit
    while not session.closed.is_set() and session._turn_no < turn and loop.time() < deadline:
        await asyncio.sleep(0.05)


async def _wait_agent_speaking(session: CallSession, turn: int, limit: float = 60.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit
    while not session.closed.is_set() and loop.time() < deadline:
        metrics = session._turn_metrics
        if metrics is not None and metrics.turn == turn and metrics.playback_start is not None:
            return True
        if (
            session._turn_task is not None
            and session._turn_task.done()
            and not session.player.active
            and metrics
            and metrics.turn == turn
        ):
            return False
        await asyncio.sleep(0.02)
    return False


def _heard_since(session: CallSession, after_turn: int) -> str:
    return " ".join(
        u.heard_text() for u in session.player.history if u.turn > after_turn and u.source != "filler"
    ).strip()


async def simulate_call(
    settings: Settings,
    scenario: Scenario,
    *,
    speech: SpeechStack,
    agent_llm: ChatModel,
    caller_llm: ChatModel | None,
    caller_tts: TTS,
    work_dir: Path,
    max_seconds: float = 200.0,
) -> SimResult:
    rng = np.random.default_rng(abs(hash(scenario.id)) % (2**32))
    runtime, setup_ids = prepare_runtime(settings, scenario, agent_llm, work_dir / "db" / f"{scenario.id}.db")
    transport = SimTransport()
    config = SessionConfig(
        barge_in_min_speech_s=settings.barge_in_min_speech_ms / 1000,
        hard_interrupt_s=settings.hard_interrupt_ms / 1000,
        # The simulated caller's own LLM may take many seconds to "think"; that is not a silent caller.
        silence_prompt_s=6.0 if scenario.silent else 45.0,
        recordings_dir=work_dir / "recordings",
    )
    session = CallSession(
        runtime,
        speech,
        transport,
        config=config,
        caller_phone=scenario.caller_id,
        scenario=scenario.id,
        call_id=f"sim_{scenario.id}_{int(time.time())}",
    )
    await preload(speech, caller_tts)
    feeder = LineFeeder(session, scenario.channel, rng)
    caller = Caller(scenario, caller_llm, max_turns=6)
    result = SimResult(scenario=scenario, summary={}, check={})
    started = time.monotonic()
    feeder.start()
    await session.start()
    last_turn_heard = -1
    try:
        while not session.closed.is_set() and time.monotonic() - started < max_seconds:
            await _wait_agent_done(session)
            if session.closed.is_set():
                break
            heard = _heard_since(session, last_turn_heard)
            last_turn_heard = session._turn_no
            result.agent_lines.append({"t": round(time.monotonic() - feeder.t0, 2), "text": heard})
            if scenario.silent:
                await asyncio.wait_for(session.closed.wait(), timeout=40)
                break
            caller.heard(heard)
            text, ends = await caller.next_line()
            expected_turn = session._turn_no + 1
            await _say(session, feeder, caller_tts, text, scenario, rng, result)
            await _wait_turn_started(session, expected_turn)
            interrupt = scenario.interrupt
            if (
                interrupt is not None
                and interrupt.turn == caller.turns
                and await _wait_agent_speaking(session, expected_turn)
            ):
                await asyncio.sleep(interrupt.after_s)
                caller.heard(_heard_since(session, last_turn_heard))
                clears_before = len(transport.clears)
                onset = await _say(session, feeder, caller_tts, interrupt.text, scenario, rng, result, interrupt=True)
                caller.said(interrupt.text)
                result.barge_in_truth.append(
                    {"onset": onset, "clears_before": clears_before, "backchannel": interrupt.backchannel}
                )
                await feeder.idle.wait()
                await asyncio.sleep(1.2)
                for truth in result.barge_in_truth:
                    later = [c for c in transport.clears[truth["clears_before"] :] if c >= truth["onset"]]
                    truth["reaction_ms"] = round((later[0] - truth["onset"]) * 1000) if later else None
            if ends:
                await feeder.idle.wait()
                await _wait_agent_done(session, limit=30)
                break
    finally:
        await asyncio.sleep(0.3)
        if not session.closed.is_set():
            await session.close("caller_hung_up")
        await feeder.stop()
    summary = session.summary()
    result.summary = summary
    result.caller_errors = caller.errors
    result.raw_stt = list(session.raw_caller)
    references = [u.text for u in result.utterances]
    if references and result.raw_stt:
        result.pipeline_wer = round(corpus_wer([(" ".join(references), " ".join(result.raw_stt))]), 4)
    result.recording = str(work_dir / "recordings" / f"{session.call_id}.wav")
    result.check = check_call(scenario, summary, runtime.sessions, runtime.clinic, setup_ids, call_id=session.call_id)
    return result


async def _say(
    session: CallSession,
    feeder: LineFeeder,
    tts: TTS,
    text: str,
    scenario: Scenario,
    rng: np.random.Generator,
    result: SimResult,
    *,
    interrupt: bool = False,
) -> float:
    if isinstance(tts, PiperTTS):
        tts.speaker_id = scenario.voice
    speech = await tts.synthesize(text)
    clean = _normalize_level(resample(speech.audio, speech.rate, PIPELINE_RATE))
    audio = apply_channel(clean, scenario.channel, rng)
    voiced = np.flatnonzero(np.abs(clean) > 0.01)
    offset = int(voiced[0]) if len(voiced) else 0
    stream_t = feeder.stream_time()
    feeder.enqueue(np.concatenate([audio, np.zeros(0, dtype=np.float32)]), offset)
    while len(feeder.speech_started_wall) < len(result.utterances) + 1:
        await asyncio.sleep(0.005)
    onset = feeder.speech_started_wall[len(result.utterances)]
    result.utterances.append(CallerUtterance(text, clean, audio, stream_t, interrupt))
    if not interrupt:
        await feeder.idle.wait()
    return onset


async def preload(speech: SpeechStack, *engines: object) -> None:
    """Load models before the call starts, so the first turn does not pay for it."""
    for component in (speech.stt, speech.tts, *engines):
        loader = getattr(component, "load", None)
        if loader is not None:
            await asyncio.to_thread(loader)


def caller_tts_for(settings: Settings, fake: bool = False) -> TTS:
    if fake:
        return FakeTTS()
    return PiperTTS(settings.models_dir / "en_US-libritts_r-medium.onnx")


async def simulate_one(
    settings: Settings,
    scenario_id: str,
    *,
    caller_mode: str = "scripted",
    channel: str | None = None,
    caller_model: str = "liquid/lfm-2.5-2.6b:free",
) -> dict[str, Any]:
    scenario = get_scenario(scenario_id)
    if channel:
        scenario = scenario.model_copy(update={"channel": channel})
    speech = build_speech(settings)
    from callie.clinic import load_clinic

    clinic = load_clinic(settings.clinic_file)
    agent_llm = build_chat_model(settings, clinic)
    caller_llm = (
        build_chat_model(
            settings.model_copy(update={"llm_max_retries": 3}),
            clinic,
            provider="openrouter",
            model=caller_model,
            fallback_models=[],
            tag="caller",
        )
        if caller_mode == "llm"
        else None
    )
    fake_voice = settings.tts_provider == "fake" and settings.stt_provider == "fake"
    result = await simulate_call(
        settings,
        scenario,
        speech=speech,
        agent_llm=agent_llm,
        caller_llm=caller_llm,
        caller_tts=caller_tts_for(settings, fake_voice),
        work_dir=Path("data/eval/single"),
    )
    return {"check": result.check, "summary": result.summary, "recording": result.recording}


async def run_wav_call(settings: Settings, wav: Path, out: Path) -> dict[str, Any]:
    """Play a recorded caller WAV into the pipeline (then 8 s of line noise) and save the call recording."""
    from callie.clinic import load_clinic

    audio, rate = read_wav(wav)
    audio = resample(audio, rate, PIPELINE_RATE)
    clinic = load_clinic(settings.clinic_file)
    runtime = build_runtime(settings, llm=build_chat_model(settings, clinic))
    speech = build_speech(settings)
    session = CallSession(runtime, speech, SimTransport(), config=SessionConfig(recordings_dir=out.parent))
    feeder = LineFeeder(session, "clean", np.random.default_rng(0))
    feeder.start()
    await session.start()
    await _wait_agent_done(session)
    feeder.enqueue(audio, 0)
    await feeder.idle.wait()
    await _wait_agent_done(session, settle=1.5)
    await session.close("caller_hung_up")
    await feeder.stop()
    produced = out.parent / f"{session.call_id}.wav"
    if produced.exists():
        produced.replace(out)
    return session.summary()
