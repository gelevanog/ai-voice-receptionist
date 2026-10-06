"""A live call: caller audio in, agent audio out, everything streamed and interruptible.

    caller audio -> VAD/endpointer -> (end of turn) STT -> agent (rules / LLM stream + tools)
                 -> sentence chunks -> TTS (per sentence) -> paced player -> caller

Barge-in, in two stages so a backchannel does not cost the caller the rest of the answer:
1. VAD: once the caller has spoken `barge_in_min_speech_ms` while the agent is talking, playback pauses at once.
2. STT: when that utterance ends, a backchannel ("mm-hm", "okay") resumes playback where it paused; anything else
   drops the rest of the queued answer, cancels generation, trims the history to what the caller heard, and
   becomes the next turn. Speech longer than `hard_interrupt_ms` interrupts without waiting for STT.

If the caller speaks again while the agent is still thinking (before any audio), the turn is cancelled and the
two utterances are answered together. Silence prompts ("Are you still there?") and a polite hang-up handle
callers who go quiet.

Per turn the session records a latency waterfall in wall-clock time: caller stops speaking -> end of turn
detected -> transcript ready -> first model token -> first sentence ready -> first audio synthesized -> first
audio playing. Voice-to-voice latency is the last minus the first.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from callie.agent.agent import Agent, CallAction, LLMTiming, Sentence, ToolEvent
from callie.agent.policy import is_backchannel
from callie.audio.pcm import PIPELINE_RATE, Audio
from callie.llm.base import ChatModel, JsonDict
from callie.logging_config import get_logger
from callie.pipeline.player import AudioSink, Player, Utterance
from callie.pipeline.recorder import Recorder
from callie.privacy import mask_phone
from callie.runtime import Runtime
from callie.scheduling.db import CallRecord
from callie.speech import SpeechStack
from callie.vad import VADEvent

log = get_logger(__name__)


class Transport(Protocol):
    name: str

    async def send_audio(self, audio: Audio, rate: int) -> None: ...

    async def clear_audio(self) -> None: ...

    async def send_event(self, event: JsonDict) -> None: ...

    async def transfer(self, reason: str) -> None: ...

    async def hangup(self) -> None: ...


class _Sink(AudioSink):
    def __init__(self, transport: Transport) -> None:
        self.transport = transport

    async def send_audio(self, audio: Audio, rate: int) -> None:
        await self.transport.send_audio(audio, rate)

    async def clear_audio(self) -> None:
        await self.transport.clear_audio()


@dataclass
class TurnMetrics:
    turn: int
    caller_text: str = ""
    path: str = ""
    model: str = ""
    served_model: str | None = None
    cached: bool = False
    llm_calls: int = 0
    user_stop: float | None = None  # wall clock, s
    vad_end: float | None = None
    stt_done: float | None = None
    llm_first_token: float | None = None
    first_sentence: float | None = None
    tts_first_audio: float | None = None
    playback_start: float | None = None
    filler_start: float | None = None
    stt_s: float | None = None
    tts_s: float | None = None
    interrupted: bool = False
    typed: bool = False

    def waterfall(self) -> dict[str, Any]:
        """Stage durations in ms (None when a stage did not happen, e.g. a rules-only turn has no LLM)."""

        def ms(a: float | None, b: float | None) -> float | None:
            return round((b - a) * 1000) if a is not None and b is not None else None

        llm_end = self.llm_first_token or self.first_sentence
        return {
            "turn": self.turn,
            "path": self.path,
            "model": self.served_model or self.model,
            "cached": self.cached,
            "llm_calls": self.llm_calls,
            "typed": self.typed,
            "endpointing_ms": ms(self.user_stop, self.vad_end),
            "stt_ms": ms(self.vad_end, self.stt_done),
            "llm_first_token_ms": ms(self.stt_done, self.llm_first_token),
            "to_first_sentence_ms": ms(llm_end, self.first_sentence) if self.llm_first_token else None,
            "tts_first_audio_ms": ms(self.first_sentence, self.tts_first_audio),
            "send_ms": ms(self.tts_first_audio, self.playback_start),
            "voice_to_voice_ms": ms(self.user_stop, self.playback_start),
            "first_audio_ms": ms(
                self.user_stop, min(x for x in (self.filler_start, self.playback_start) if x is not None)
            )
            if (self.filler_start or self.playback_start)
            else None,
            "filler": self.filler_start is not None,
            "interrupted": self.interrupted,
        }


@dataclass
class BargeIn:
    onset_wall: float
    paused_wall: float | None = None
    turn: int | None = None
    hard: bool = False


@dataclass
class SessionConfig:
    barge_in_min_speech_s: float = 0.25
    hard_interrupt_s: float = 1.2
    backchannel_max_s: float = 1.0
    silence_prompt_s: float = 9.0
    record: bool = True
    recordings_dir: Path = Path("data/recordings")
    save_record: bool = True


@dataclass
class SessionStats:
    barge_ins: list[dict[str, Any]] = field(default_factory=list)
    backchannels: int = 0
    silence_prompts: int = 0
    unsupported_times: list[str] = field(default_factory=list)
    replaced_advice: int = 0


class CallSession:
    def __init__(
        self,
        runtime: Runtime,
        speech: SpeechStack,
        transport: Transport,
        *,
        config: SessionConfig | None = None,
        caller_phone: str | None = None,
        call_id: str | None = None,
        scenario: str | None = None,
        llm: ChatModel | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime = runtime
        self.speech = speech
        self.transport = transport
        self.config = config or SessionConfig(
            barge_in_min_speech_s=runtime.settings.barge_in_min_speech_ms / 1000,
            hard_interrupt_s=runtime.settings.hard_interrupt_ms / 1000,
            silence_prompt_s=runtime.settings.silence_prompt_seconds,
            recordings_dir=runtime.settings.recordings_dir,
        )
        self.clock = clock
        self.call_id = call_id or f"call_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.ctx = runtime.new_context(caller_phone=caller_phone, call_id=self.call_id)
        self.agent: Agent = runtime.new_agent(self.ctx, llm)
        self.scenario = scenario
        self.endpointer = speech.endpointer()
        self.player = Player(_Sink(transport), clock, on_chunk=self._on_chunk)
        self.recorder = Recorder() if self.config.record else None
        self.started_wall = clock()
        self.started_at = datetime.now(UTC)
        self.transcript: list[JsonDict] = []
        self.turns: list[TurnMetrics] = []
        self.stats = SessionStats()
        self.closed = asyncio.Event()
        self.end_reason = ""
        self._turn_no = 0
        self._turn_task: asyncio.Task[None] | None = None
        self._turn_metrics: TurnMetrics | None = None
        self._turn_audio_started = False
        self._barge: BargeIn | None = None
        self._last_feed_wall = clock()
        self._last_activity = clock()
        self._prompts = 0
        self._tasks: set[asyncio.Task[Any]] = set()
        self._ending = False

    # -- lifecycle ---------------------------------------------------------------------------------------------
    async def start(self) -> None:
        self.player.start()
        self._spawn(self._watchdog(), "watchdog")
        await self._emit({"type": "call_started", "call_id": self.call_id, "stack": self.stack()})
        await self._speak_rules(self.agent.greeting(), turn=0)

    def stack(self) -> dict[str, str]:
        return {**self.speech.describe(), "llm": self.agent.llm.label}

    async def close(self, reason: str = "caller_hung_up") -> None:
        if self.closed.is_set():
            return
        self.end_reason = self.end_reason or reason
        self.closed.set()
        current = asyncio.current_task()
        if self._turn_task is not None and not self._turn_task.done() and self._turn_task is not current:
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._turn_task
        await self.player.stop()
        for task in list(self._tasks):
            if task is not current:
                task.cancel()
        outcome = self.ctx.outcome.label()
        if self.end_reason == "silence_timeout" and outcome == "info_only":
            outcome = "no_response"
        await self._emit({"type": "call_ended", "call_id": self.call_id, "outcome": outcome, "reason": self.end_reason})
        if self.config.save_record:
            self._save(outcome)

    def _spawn(self, coro: Any, name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # -- clocks ------------------------------------------------------------------------------------------------
    def stream_now(self) -> float:
        """Current position on the caller-audio timeline (s)."""
        return self.endpointer.stream_time + (self.clock() - self._last_feed_wall)

    def _wall_of(self, stream_t: float) -> float:
        return self._last_feed_wall - (self.endpointer.stream_time - stream_t)

    def _since_start(self) -> float:
        return round(self.clock() - self.started_wall, 2)

    # -- input -------------------------------------------------------------------------------------------------
    def feed_audio(self, audio: Audio) -> None:
        """Caller audio at 16 kHz, any chunk size, in real time."""
        if self.closed.is_set():
            return
        if self.recorder is not None:
            self.recorder.add_caller(audio)
        self._last_feed_wall = self.clock()
        for event in self.endpointer.feed(audio):
            self._on_vad(event)

    async def feed_text(self, text: str) -> None:
        """Typed input from the dashboard (skips VAD and STT)."""
        text = text.strip()
        if not text or self.closed.is_set():
            return
        self._last_activity = self.clock()
        self._prompts = 0
        if self.player.active:
            await self._interrupt()
        metrics = TurnMetrics(turn=self._turn_no + 1, typed=True)
        now = self.clock()
        metrics.user_stop = metrics.vad_end = metrics.stt_done = now
        await self._start_turn(text, metrics, merge=False)

    def _on_vad(self, event: VADEvent) -> None:
        now = self.clock()
        if event.kind == "speech_start":
            self._last_activity = now
            self._prompts = 0
            if self.player.active and not self.player.paused and self._barge is None:
                self._barge = BargeIn(onset_wall=self._wall_of(event.onset), turn=self.player.current_turn)
            self._spawn(self._emit({"type": "state", "state": "caller_speaking"}), "emit")
        elif event.kind == "speech_ongoing":
            self._last_activity = now
            barge = self._barge
            if barge is not None and barge.paused_wall is None and event.speech_s >= self.config.barge_in_min_speech_s:
                barge.paused_wall = now
                self._spawn(self._pause_for(barge), "barge-pause")
            elif (
                barge is not None
                and barge.paused_wall is not None
                and not barge.hard
                and event.speech_s >= self.config.hard_interrupt_s
            ):
                barge.hard = True
                self._spawn(self._interrupt(), "barge-hard")
        elif event.kind == "speech_end":
            self._last_activity = now
            self._spawn(self._on_utterance(event, now), "utterance")

    async def _pause_for(self, barge: BargeIn) -> None:
        await self.player.pause()
        if self.recorder is not None:
            self.recorder.cut_agent_after(self.stream_now())
        reaction = round((self.clock() - barge.onset_wall) * 1000)
        self.stats.barge_ins.append(
            {"turn": barge.turn, "reaction_ms": reaction, "result": "paused", "t": self._since_start()}
        )
        await self._emit({"type": "barge_in", "stage": "paused", "reaction_ms": reaction})

    async def _on_utterance(self, event: VADEvent, vad_end_wall: float) -> None:
        metrics = TurnMetrics(turn=self._turn_no + 1)
        metrics.user_stop = self._wall_of(event.last_speech_t)
        metrics.vad_end = vad_end_wall
        await self._emit({"type": "state", "state": "transcribing"})
        transcript = await self.speech.stt.transcribe(event.audio)
        metrics.stt_done = self.clock()
        metrics.stt_s = transcript.seconds
        text = transcript.text.strip()
        barge = self._barge
        self._barge = None
        if barge is not None and barge.paused_wall is not None:
            backchannel = is_backchannel(text) or (event.speech_s < 0.6 and len(text.split()) <= 2)
            if not barge.hard and backchannel and event.speech_s <= self.config.backchannel_max_s:
                self.stats.backchannels += 1
                if self.stats.barge_ins:
                    self.stats.barge_ins[-1]["result"] = "backchannel_resumed"
                self.player.resume()
                await self._emit({"type": "barge_in", "stage": "resumed", "text": self.ctx.masker.mask(text)})
                return
            if self.stats.barge_ins:
                self.stats.barge_ins[-1]["result"] = "interrupted"
            if not barge.hard:
                await self._interrupt()
        if not text:
            await self._emit({"type": "state", "state": "listening"})
            return
        merge = False
        running = self._turn_task is not None and not self._turn_task.done()
        if running and self.agent.turn_committed and self._turn_task is not None:
            # The previous turn already changed something (e.g. booked): let it finish and say so first.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._turn_task
        elif running and not self._turn_audio_started and self._turn_task is not None:
            # Still thinking about the previous utterance: answer both together.
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._turn_task
            merge = True
            if self._turn_metrics is not None and self._turn_metrics in self.turns:
                self.turns.remove(self._turn_metrics)
        await self._start_turn(text, metrics, merge=merge)

    async def _interrupt(self) -> None:
        """Hard barge-in: drop the rest of the answer, stop generating, keep only what the caller heard."""
        turn = self.player.current_turn
        if turn is None and self._turn_task is not None and not self._turn_task.done():
            turn = self._turn_no
        await self.player.drop()
        if self.recorder is not None:
            self.recorder.cut_agent_after(self.stream_now())
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._turn_task
        if turn is not None:
            heard = self.player.heard_text(turn)
            self.agent.note_interrupted(heard)
            for entry in reversed(self.transcript):
                if entry["role"] == "agent" and entry.get("turn") == turn:
                    entry["text"] = self.ctx.masker.mask(heard) or "…"
                    entry["interrupted"] = True
                    break
            for metrics in self.turns:
                if metrics.turn == turn:
                    metrics.interrupted = True
        await self._emit({"type": "barge_in", "stage": "interrupted"})

    # -- turns -------------------------------------------------------------------------------------------------
    async def _start_turn(self, text: str, metrics: TurnMetrics, *, merge: bool) -> None:
        if merge:
            metrics.turn = self._turn_no
            for entry in reversed(self.transcript):
                if entry["role"] == "caller":
                    entry["text"] = f"{entry['text']} {self.ctx.masker.mask(text)}"
                    break
        else:
            self._turn_no += 1
            metrics.turn = self._turn_no
            self.transcript.append(
                {
                    "t": self._since_start(),
                    "role": "caller",
                    "turn": metrics.turn,
                    "text": self.ctx.masker.mask(text),
                    "typed": metrics.typed,
                }
            )
        metrics.caller_text = self.ctx.masker.mask(text)
        await self._emit(
            {
                "type": "transcript",
                "role": "caller",
                "turn": metrics.turn,
                "text": self.ctx.masker.mask(text),
                "merged": merge,
            }
        )
        self.turns.append(metrics)
        self._turn_metrics = metrics
        self._turn_audio_started = False
        self._turn_task = asyncio.create_task(self._run_turn(text, metrics, merge), name=f"turn-{metrics.turn}")

    async def _run_turn(self, text: str, metrics: TurnMetrics, merge: bool) -> None:
        turn = metrics.turn
        queue: asyncio.Queue[Sentence | None] = asyncio.Queue()
        worker = asyncio.create_task(self._tts_worker(turn, queue, metrics), name=f"tts-{turn}")
        action: CallAction | None = None
        spoken: list[str] = []
        await self._emit({"type": "state", "state": "thinking"})
        try:
            async for event in self.agent.respond(text, merge=merge):
                if isinstance(event, Sentence):
                    if event.source != "filler" and metrics.first_sentence is None:
                        metrics.first_sentence = self.clock()
                    spoken.append(event.text)
                    await queue.put(event)
                    await self._emit(
                        {
                            "type": "agent_text",
                            "turn": turn,
                            "text": self.ctx.masker.mask(event.text),
                            "source": event.source,
                        }
                    )
                elif isinstance(event, ToolEvent):
                    entry = self.ctx.tool_log[-1] if self.ctx.tool_log else {}
                    await self._emit({"type": "tool", "turn": turn, **entry})
                elif isinstance(event, LLMTiming):
                    metrics.model = event.model
                    metrics.served_model = event.served_model or metrics.served_model
                    metrics.cached = metrics.cached or event.cached
                    if metrics.llm_first_token is None and event.first_token_s is not None:
                        metrics.llm_first_token = (metrics.stt_done or self.clock()) + event.first_token_s
                elif isinstance(event, CallAction):
                    action = event
            stats = self.agent.last_stats
            metrics.path, metrics.llm_calls = stats.path, stats.llm_calls
            self.stats.unsupported_times.extend(stats.unsupported_times)
            self.stats.replaced_advice += stats.replaced_advice
            self._record_agent_text(turn, " ".join(s for s in spoken))
            await queue.put(None)
            await worker
            if action is not None:
                await self.player.wait_idle()
                await self._perform(action)
            else:
                await self._emit({"type": "metrics", **metrics.waterfall()})
        except asyncio.CancelledError:
            worker.cancel()
            stats = self.agent.last_stats
            metrics.path, metrics.llm_calls = stats.path, stats.llm_calls
            if spoken:
                self._record_agent_text(turn, " ".join(spoken))
            raise
        finally:
            if not worker.done():
                worker.cancel()

    def _record_agent_text(self, turn: int, text: str) -> None:
        for entry in reversed(self.transcript):
            if entry["role"] == "agent" and entry.get("turn") == turn:
                entry["text"] = self.ctx.masker.mask(text)
                return
        self.transcript.append(
            {"t": self._since_start(), "role": "agent", "turn": turn, "text": self.ctx.masker.mask(text)}
        )

    async def _tts_worker(self, turn: int, queue: asyncio.Queue[Sentence | None], metrics: TurnMetrics | None) -> None:
        while (sentence := await queue.get()) is not None:
            speech = await self.speech.tts.synthesize(sentence.text)
            if metrics is not None:
                metrics.tts_s = (metrics.tts_s or 0.0) + speech.seconds
                if sentence.source != "filler" and metrics.tts_first_audio is None:
                    metrics.tts_first_audio = self.clock()
            self.player.enqueue(Utterance(sentence.text, speech.audio, speech.rate, sentence.source, turn))

    def _on_chunk(self, utterance: Utterance, chunk: Audio, start_at: float) -> None:
        if self.recorder is not None:
            at = self.endpointer.stream_time + (start_at - self._last_feed_wall)
            self.recorder.add_agent(chunk, utterance.rate, at)
        metrics = self._turn_metrics
        if metrics is not None and utterance.turn == metrics.turn:
            if utterance.source == "filler":
                metrics.filler_start = metrics.filler_start or start_at
            elif metrics.playback_start is None:
                metrics.playback_start = start_at
                self._turn_audio_started = True
                self._spawn(self._emit({"type": "state", "state": "speaking"}), "emit")
            if utterance.source == "filler":
                self._turn_audio_started = True

    async def _speak_rules(self, text: str, *, turn: int) -> None:
        speech = await self.speech.tts.synthesize(text)
        self.transcript.append(
            {"t": self._since_start(), "role": "agent", "turn": turn, "text": self.ctx.masker.mask(text)}
        )
        await self._emit({"type": "agent_text", "turn": turn, "text": text, "source": "rules"})
        self.player.enqueue(Utterance(text, speech.audio, speech.rate, "rules", turn))

    async def _perform(self, action: CallAction) -> None:
        self.end_reason = "transferred" if action.kind == "transfer" else "agent_hung_up"
        await self._emit({"type": "action", "action": action.kind, "reason": action.reason})
        if action.kind == "transfer":
            await self.transport.transfer(action.reason)
        else:
            await self.transport.hangup()
        await self.close(self.end_reason)

    async def _watchdog(self) -> None:
        while not self.closed.is_set():
            await asyncio.sleep(0.25)
            busy = (
                self.player.active
                or self.endpointer.in_speech
                or (self._turn_task is not None and not self._turn_task.done())
            )
            if busy:
                self._last_activity = self.clock()
                continue
            if self.clock() - self._last_activity < self.config.silence_prompt_s:
                continue
            self._prompts += 1
            self.stats.silence_prompts += 1
            self._last_activity = self.clock()
            if self._prompts == 1:
                text = "Are you still there?"
                self.agent.say_directly(text)
                await self._speak_rules(text, turn=self._turn_no)
            else:
                text = f"It seems we got disconnected. Please call {self.ctx.clinic.name} back anytime. Goodbye!"
                self.agent.say_directly(text)
                await self._speak_rules(text, turn=self._turn_no)
                await self.player.wait_idle()
                self.end_reason = "silence_timeout"
                await self.transport.hangup()
                await self.close("silence_timeout")
                return

    # -- events and persistence --------------------------------------------------------------------------------
    async def _emit(self, event: JsonDict) -> None:
        with contextlib.suppress(Exception):
            await self.transport.send_event({"t": self._since_start(), **event})

    def summary(self) -> JsonDict:
        return {
            "call_id": self.call_id,
            "outcome": self.ctx.outcome.label(),
            "outcome_detail": asdict(self.ctx.outcome),
            "end_reason": self.end_reason,
            "transcript": self.transcript,
            "tool_calls": self.ctx.tool_log,
            "turns": [t.waterfall() for t in self.turns],
            "stats": asdict(self.stats),
            "stack": self.stack(),
        }

    def _save(self, outcome: str) -> None:
        recording = None
        if self.recorder is not None and self.recorder.seconds > 0.5:
            path = self.config.recordings_dir / f"{self.call_id}.wav"
            recording = str(self.recorder.save(path))
        detail = asdict(self.ctx.outcome)
        with self.runtime.sessions() as session, session.begin():
            session.add(
                CallRecord(
                    id=self.call_id,
                    started_at=self.started_at,
                    ended_at=datetime.now(UTC),
                    transport=self.transport.name,
                    caller=mask_phone(self.ctx.caller_phone) if self.ctx.caller_phone else None,
                    outcome=outcome,
                    outcome_detail=str({k: v for k, v in detail.items() if v}),
                    transcript=self.transcript,
                    tool_calls=self.ctx.tool_log,
                    turns=[t.waterfall() for t in self.turns],
                    recording_path=recording,
                    duration_s=round(self.clock() - self.started_wall, 1),
                    stack=self.stack(),
                    scenario=self.scenario,
                )
            )


def silence_frames(seconds: float) -> Audio:
    return np.zeros(round(seconds * PIPELINE_RATE), dtype=np.float32)
