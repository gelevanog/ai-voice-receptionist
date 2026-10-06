"""Twilio phone calls: the voice webhook answers with TwiML that connects a bidirectional Media Stream.

    caller -> Twilio -> POST /twilio/voice  -> <Connect><Stream url="wss://.../twilio/media"/></Connect>
    Twilio <-> WS /twilio/media: JSON messages; audio is base64 G.711 μ-law, 8 kHz, mono, 20 ms per frame

Inbound `media` frames are decoded, resampled 8 -> 16 kHz (stateful filter, no clicks between frames) and fed to
the session. Outbound speech is resampled to 8 kHz, μ-law encoded and sent as 20 ms `media` messages; barge-in
sends `clear` so Twilio drops what it has buffered. A transfer updates the live call with `<Dial>` TwiML through
the REST API (the media stream then ends and Twilio dials the front desk).

Message shapes follow Twilio's Media Streams documentation (`connected`, `start`, `media`, `mark`, `dtmf`, `stop`)
and are tested with recorded fixtures. No live Twilio call was made while building this (no account was used).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
from collections.abc import Mapping
from typing import Any
from xml.sax.saxutils import quoteattr

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from callie.audio import mulaw
from callie.audio.pcm import Audio
from callie.audio.resample import StreamResampler
from callie.llm.base import JsonDict
from callie.logging_config import get_logger
from callie.pipeline.session import CallSession
from callie.runtime import Runtime
from callie.speech import SpeechStack
from callie.transports.twilio_rest import transfer_twiml

log = get_logger(__name__)
FRAME_BYTES = 160  # 20 ms at 8 kHz, one byte per μ-law sample


def compute_signature(auth_token: str, url: str, params: Mapping[str, str]) -> str:
    """X-Twilio-Signature: base64(HMAC-SHA1(auth token, URL + every POST param name+value, sorted by name))."""
    payload = url + "".join(f"{key}{params[key]}" for key in sorted(params))
    digest = hmac.new(auth_token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


def validate_signature(auth_token: str, url: str, params: Mapping[str, str], signature: str | None) -> bool:
    if not signature:
        return False
    return hmac.compare_digest(compute_signature(auth_token, url, params), signature)


def stream_twiml(stream_url: str, *, caller: str | None, call_sid: str | None) -> str:
    parameters = "".join(
        f"<Parameter name={quoteattr(name)} value={quoteattr(value)}/>"
        for name, value in (("from", caller or ""), ("callSid", call_sid or ""))
        if value
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><Response>'
        f"<Connect><Stream url={quoteattr(stream_url)}>{parameters}</Stream></Connect>"
        "</Response>"
    )


class TwilioTransport:
    name = "twilio"

    def __init__(self, websocket: WebSocket, runtime: Runtime) -> None:
        self.ws = websocket
        self.runtime = runtime
        self.stream_sid = ""
        self.call_sid = ""
        self._out: dict[int, StreamResampler] = {}
        self._pending = b""
        self._lock = asyncio.Lock()
        self.closed = False
        self.marks_sent = 0
        self.transferred_to: str | None = None

    async def _send(self, payload: JsonDict) -> None:
        if self.closed or self.ws.application_state != WebSocketState.CONNECTED:
            return
        async with self._lock:
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await self.ws.send_text(json.dumps(payload))

    async def send_audio(self, audio: Audio, rate: int) -> None:
        resampler = self._out.get(rate)
        if resampler is None:
            resampler = self._out[rate] = StreamResampler(rate, 8000)
        data = self._pending + mulaw.encode(resampler.process(audio))
        whole = len(data) - len(data) % FRAME_BYTES
        self._pending = data[whole:]
        for start in range(0, whole, FRAME_BYTES):
            payload = base64.b64encode(data[start : start + FRAME_BYTES]).decode("ascii")
            await self._send({"event": "media", "streamSid": self.stream_sid, "media": {"payload": payload}})

    async def clear_audio(self) -> None:
        self._pending = b""
        for resampler in self._out.values():
            resampler.reset()
        await self._send({"event": "clear", "streamSid": self.stream_sid})

    async def mark(self, name: str) -> None:
        self.marks_sent += 1
        await self._send({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": name}})

    async def send_event(self, event: JsonDict) -> None:
        if event.get("type") == "agent_text":
            await self.mark(f"turn-{event.get('turn')}")

    async def transfer(self, reason: str) -> None:
        number = self.runtime.settings.transfer_number
        self.transferred_to = number
        twiml = transfer_twiml(number, "Connecting you to our front desk now.")
        await asyncio.to_thread(self.runtime.twilio.redirect_call, self.call_sid, twiml)

    async def hangup(self) -> None:
        # Closing the stream ends <Connect>; with no TwiML after it, Twilio ends the call.
        self.closed = True
        if self.ws.application_state == WebSocketState.CONNECTED:
            with contextlib.suppress(RuntimeError):
                await self.ws.close()


def decode_media(payload: str, resampler: StreamResampler) -> Audio:
    """One inbound `media` payload (base64 μ-law, 8 kHz) -> float audio at 16 kHz."""
    return resampler.process(mulaw.decode(base64.b64decode(payload)))


async def serve_twilio_stream(websocket: WebSocket, runtime: Runtime, speech: SpeechStack, **kwargs: Any) -> None:
    await websocket.accept()
    transport = TwilioTransport(websocket, runtime)
    inbound = StreamResampler(8000, 16000)
    session: CallSession | None = None
    try:
        while True:
            message = json.loads(await websocket.receive_text())
            event = message.get("event")
            if event == "connected":
                continue
            if event == "start":
                start = message.get("start", {})
                transport.stream_sid = message.get("streamSid") or start.get("streamSid", "")
                transport.call_sid = start.get("callSid", "")
                media_format = start.get("mediaFormat", {})
                if media_format and (
                    media_format.get("encoding") != "audio/x-mulaw" or int(media_format.get("sampleRate", 8000)) != 8000
                ):
                    log.warning("twilio.unexpected_format", media_format=media_format)
                caller = (start.get("customParameters") or {}).get("from") or None
                session = CallSession(runtime, speech, transport, caller_phone=caller, **kwargs)
                await session.start()
            elif event == "media" and session is not None:
                media = message.get("media", {})
                if media.get("track", "inbound") != "inbound":
                    continue
                session.feed_audio(decode_media(media["payload"], inbound))
            elif event == "dtmf" and session is not None:
                digit = str(message.get("dtmf", {}).get("digit", ""))
                if digit == "0":  # press 0 for a person
                    await session.feed_text("I'd like to speak to a person")
            elif event == "stop":
                break
            if session is not None and session.closed.is_set():
                break
    except WebSocketDisconnect:
        pass
    finally:
        transport.closed = True
        if session is not None:
            await session.close(session.end_reason or "caller_hung_up")
        if websocket.application_state == WebSocketState.CONNECTED:
            with contextlib.suppress(RuntimeError):
                await websocket.close()


def silence_payload(ms: int = 20) -> str:
    return base64.b64encode(mulaw.encode(np.zeros(8 * ms, dtype=np.float32))).decode("ascii")
