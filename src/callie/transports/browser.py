"""Browser calls over one WebSocket.

Client -> server: binary frames of 16-bit little-endian PCM, mono, 16 kHz (the page downsamples the microphone in
an AudioWorklet); JSON text frames `{"type": "text", "text": ...}` (typed input) and `{"type": "hangup"}`.
Server -> client: binary frames of 16-bit PCM at the TTS sample rate (announced in an `audio_format` event), and
JSON events (state, transcript, tool calls, latency waterfall, barge-in). `{"type": "clear"}` tells the page to
stop and discard queued audio immediately (barge-in).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from callie.audio.pcm import Audio, float_to_int16_bytes, int16_bytes_to_float
from callie.llm.base import JsonDict
from callie.logging_config import get_logger
from callie.pipeline.session import CallSession
from callie.runtime import Runtime
from callie.speech import SpeechStack

log = get_logger(__name__)


class BrowserTransport:
    name = "browser"

    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        self._rate: int | None = None
        self._lock = asyncio.Lock()
        self.closed = False

    async def _send_json(self, payload: JsonDict) -> None:
        if self.closed or self.ws.application_state != WebSocketState.CONNECTED:
            return
        async with self._lock:
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await self.ws.send_text(json.dumps(payload, default=str))

    async def send_audio(self, audio: Audio, rate: int) -> None:
        if rate != self._rate:
            self._rate = rate
            await self._send_json({"type": "audio_format", "rate": rate, "encoding": "pcm_s16le"})
        if self.closed or self.ws.application_state != WebSocketState.CONNECTED:
            return
        async with self._lock:
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await self.ws.send_bytes(float_to_int16_bytes(audio))

    async def clear_audio(self) -> None:
        await self._send_json({"type": "clear"})

    async def send_event(self, event: JsonDict) -> None:
        await self._send_json(event)

    async def transfer(self, reason: str) -> None:
        # A browser call has nobody to transfer to: show it, then end (the Twilio transport really dials).
        await self._send_json(
            {"type": "transfer", "reason": reason, "note": "On a phone call this dials the front desk."}
        )

    async def hangup(self) -> None:
        await self._send_json({"type": "hangup"})


async def serve_browser_call(
    websocket: WebSocket, runtime: Runtime, speech: SpeechStack, **session_kwargs: Any
) -> None:
    await websocket.accept()
    transport = BrowserTransport(websocket)
    session = CallSession(runtime, speech, transport, **session_kwargs)
    await session.start()
    try:
        while not session.closed.is_set():
            receive = asyncio.ensure_future(websocket.receive())
            closed = asyncio.ensure_future(session.closed.wait())
            done, _ = await asyncio.wait({receive, closed}, return_when=asyncio.FIRST_COMPLETED)
            if closed in done:
                receive.cancel()
                break
            closed.cancel()
            message = receive.result()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                session.feed_audio(int16_bytes_to_float(message["bytes"]))
            elif message.get("text"):
                data = json.loads(message["text"])
                if data.get("type") == "text":
                    await session.feed_text(str(data.get("text", "")))
                elif data.get("type") == "hangup":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        transport.closed = True
        await session.close(session.end_reason or "caller_hung_up")
        if websocket.application_state == WebSocketState.CONNECTED:
            with contextlib.suppress(RuntimeError):
                await websocket.close()
