"""Outbound audio: sends the agent's speech in 20 ms chunks paced to real time.

Pacing matters for barge-in. If a whole answer were pushed to the client at once, "stop talking" would have to
chase seconds of buffered audio; here the client never holds more than `lead_s` (120 ms) of audio, so a pause
takes effect within one network round trip, and the player knows exactly how much of each sentence the caller
has heard (for an honest conversation history after an interruption).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np

from callie.audio.pcm import Audio


class AudioSink:
    """What the player needs from a transport."""

    async def send_audio(self, audio: Audio, rate: int) -> None:
        raise NotImplementedError

    async def clear_audio(self) -> None:
        raise NotImplementedError


@dataclass
class Utterance:
    text: str
    audio: Audio
    rate: int
    source: str
    turn: int
    sent: int = 0  # samples sent to the client
    heard: int = 0  # samples the client has played (known after the fact)
    started_at: float | None = None  # wall time its first sample plays
    done: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def duration(self) -> float:
        return len(self.audio) / self.rate

    def heard_text(self) -> str:
        if not len(self.audio):
            return self.text
        fraction = min(1.0, self.heard / len(self.audio))
        if fraction >= 0.97:
            return self.text
        words = self.text.split()
        return " ".join(words[: int(len(words) * fraction)])


ChunkCallback = Callable[[Utterance, Audio, float], None]


class Player:
    CHUNK_S = 0.02

    def __init__(
        self,
        sink: AudioSink,
        clock: Callable[[], float],
        *,
        lead_s: float = 0.12,
        on_chunk: ChunkCallback | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.sink = sink
        self.clock = clock
        self.lead_s = lead_s
        self.on_chunk = on_chunk
        self._sleep = sleep
        self._queue: deque[Utterance] = deque()
        self._current: Utterance | None = None
        self._paused = False
        self._play_end = 0.0  # wall time the client finishes what it has received
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self.history: list[Utterance] = []
        self._task: asyncio.Task[None] | None = None

    # -- state ---------------------------------------------------------------------------------------------------
    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def active(self) -> bool:
        """The caller is hearing (or about to hear) the agent."""
        return self._current is not None or bool(self._queue) or self.clock() < self._play_end

    @property
    def current_turn(self) -> int | None:
        if self._current is not None:
            return self._current.turn
        return self._queue[0].turn if self._queue else None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="player")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def enqueue(self, utterance: Utterance) -> None:
        self._queue.append(utterance)
        self.history.append(utterance)
        self._idle.clear()
        self._wake.set()

    async def wait_idle(self) -> None:
        """Until everything queued has been sent and played."""
        while True:
            await self._idle.wait()
            remaining = self._play_end - self.clock()
            if remaining <= 0 and self._current is None and not self._queue:
                return
            await self._sleep(max(remaining, 0.01))

    # -- control -------------------------------------------------------------------------------------------------
    def _rewind_to_heard(self) -> None:
        """Mark what was actually played and move the read position back to it."""
        now = self.clock()
        unplayed_s = max(0.0, self._play_end - now)
        current = self._current
        if current is not None:
            unplayed = min(current.sent, round(unplayed_s * current.rate))
            current.heard = current.sent - unplayed
            current.sent = current.heard
        self._play_end = min(self._play_end, now)

    async def pause(self) -> None:
        """Stop the caller hearing the agent now; `resume` continues from the point they last heard."""
        if self._paused:
            return
        self._paused = True
        self._rewind_to_heard()
        await self.sink.clear_audio()

    def resume(self) -> None:
        if self._paused:
            self._paused = False
            self._wake.set()

    async def drop(self) -> None:
        """Barge-in: stop and discard everything queued (the rest of the answer is never spoken)."""
        if not self._paused:
            self._rewind_to_heard()
            await self.sink.clear_audio()
        self._paused = False
        if self._current is not None:
            self._current.done.set()
        for utterance in self._queue:
            utterance.done.set()
        self._current = None
        self._queue.clear()
        self._idle.set()

    def heard_text(self, turn: int) -> str:
        return " ".join(t for u in self.history if u.turn == turn and (t := u.heard_text())).strip()

    # -- loop ----------------------------------------------------------------------------------------------------
    async def _run(self) -> None:
        while True:
            if self._paused or (self._current is None and not self._queue):
                if self._current is None and not self._queue:
                    self._idle.set()
                self._wake.clear()
                await self._wake.wait()
                continue
            if self._current is None:
                self._current = self._queue.popleft()
            utterance = self._current
            now = self.clock()
            if self._play_end > now + self.lead_s:
                await self._sleep(self._play_end - now - self.lead_s)
                continue
            size = max(1, round(self.CHUNK_S * utterance.rate))
            chunk = utterance.audio[utterance.sent : utterance.sent + size]
            if len(chunk) == 0:
                utterance.heard = len(utterance.audio)
                utterance.done.set()
                self._current = None
                continue
            start_at = max(self._play_end, now)
            if utterance.started_at is None:
                utterance.started_at = start_at
            await self.sink.send_audio(np.ascontiguousarray(chunk), utterance.rate)
            if self._paused or self._current is not utterance:
                continue  # paused or dropped while sending
            utterance.sent += len(chunk)
            utterance.heard = max(utterance.heard, utterance.sent - len(chunk))
            self._play_end = start_at + len(chunk) / utterance.rate
            if self.on_chunk is not None:
                self.on_chunk(utterance, chunk, start_at)
