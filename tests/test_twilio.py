"""Twilio: webhook signature and TwiML, Media Streams message handling with a recorded fixture, μ-law framing,
clear on barge-in, transfer via the REST API (mocked), SMS (mocked)."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import select

from callie.audio import mulaw
from callie.audio.resample import StreamResampler
from callie.runtime import build_runtime
from callie.scheduling.db import CallRecord
from callie.speech import SpeechStack
from callie.stt import FakeSTT
from callie.transports.twilio import compute_signature, decode_media, stream_twiml, validate_signature
from callie.transports.twilio_rest import TwilioRest, transfer_twiml
from callie.tts import FakeTTS
from callie.vad import EnergyVAD
from callie.web.app import create_app
from tests.helpers import fake_settings

FIXTURES = Path(__file__).parent / "fixtures" / "twilio"


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


class TestWebhook:
    def test_signature_matches_twilio_reference_values(self) -> None:
        webhook = load("voice_webhook.json")
        docs = webhook["docs_example"]
        assert compute_signature(docs["auth_token"], docs["url"], docs["params"]) == docs["signature"]
        assert validate_signature(webhook["auth_token"], webhook["url"], webhook["params"], webhook["signature"])
        tampered = {**webhook["params"], "From": "+15555550999"}
        assert not validate_signature(webhook["auth_token"], webhook["url"], tampered, webhook["signature"])
        assert not validate_signature(webhook["auth_token"], webhook["url"], webhook["params"], None)

    def test_twiml_connects_a_stream_with_caller_parameters(self) -> None:
        twiml = stream_twiml("wss://example.test/twilio/media", caller="+15555550123", call_sid="CA1")
        assert '<Connect><Stream url="wss://example.test/twilio/media">' in twiml
        assert '<Parameter name="from" value="+15555550123"/>' in twiml
        assert twiml.startswith('<?xml version="1.0" encoding="UTF-8"?><Response>')

    def test_voice_endpoint_validates_signature(self) -> None:
        webhook = load("voice_webhook.json")
        settings = fake_settings(
            twilio_auth_token=webhook["auth_token"], public_base_url="https://callie.example.ngrok.app"
        )
        client = TestClient(create_app(settings, speech=_speech(settings, [])))
        ok = client.post("/twilio/voice", data=webhook["params"], headers={"X-Twilio-Signature": webhook["signature"]})
        assert ok.status_code == 200 and ok.headers["content-type"].startswith("application/xml")
        assert 'url="wss://callie.example.ngrok.app/twilio/media"' in ok.text
        forged = client.post("/twilio/voice", data=webhook["params"], headers={"X-Twilio-Signature": "bogus"})
        assert forged.status_code == 403


def _speech(settings: Any, lines: list[str]) -> SpeechStack:
    return SpeechStack(stt=FakeSTT(lines), tts=FakeTTS(ms_per_char=4), vad_factory=EnergyVAD, settings=settings)


class TestMediaStream:
    def test_inbound_decoding_resamples_to_16k(self) -> None:
        resampler = StreamResampler(8000, 16000)
        tone = (0.4 * np.sin(2 * np.pi * 400 * np.arange(160) / 8000)).astype(np.float32)
        out = decode_media(base64.b64encode(mulaw.encode(tone)).decode(), resampler)
        assert len(out) == 320 and out.dtype == np.float32

    def test_fixture_call_end_to_end(self) -> None:
        settings = fake_settings()
        runtime = build_runtime(settings)
        app = create_app(settings, runtime=runtime, speech=_speech(settings, ["Do you take Delta Dental?"]))
        messages = load("media_stream_call.json")
        outbound: list[dict[str, Any]] = []
        with TestClient(app) as client, client.websocket_connect("/twilio/media") as ws:
            for message in messages[:-1]:
                ws.send_text(json.dumps(message))
                if message["event"] == "media":
                    time.sleep(0.02)  # real time, like Twilio
            time.sleep(2.5)  # let Callie answer
            ws.send_text(json.dumps(messages[-1]))  # stop
            try:
                while True:
                    outbound.append(json.loads(ws.receive_text()))
            except Exception:
                pass
        media = [m for m in outbound if m["event"] == "media"]
        stream_sid = messages[1]["streamSid"]
        assert media and all(m["streamSid"] == stream_sid for m in media)
        assert all(len(base64.b64decode(m["media"]["payload"])) == 160 for m in media)  # 20 ms frames
        assert any(m["event"] == "mark" for m in outbound)
        with runtime.sessions() as db:
            record = db.scalars(select(CallRecord)).one()
        assert record.transport == "twilio" and record.caller == "***-***-0123"
        assert any(t["name"] == "answer_faq" for t in record.tool_calls)
        assert any(e["role"] == "caller" and "Delta" in e["text"] for e in record.transcript)

    async def test_outbound_framing_and_clear(self) -> None:
        from callie.transports.twilio import TwilioTransport

        sent: list[str] = []

        class FakeWS:
            application_state = __import__("starlette.websockets", fromlist=["WebSocketState"]).WebSocketState.CONNECTED

            async def send_text(self, text: str) -> None:
                sent.append(text)

        transport = TwilioTransport(FakeWS(), build_runtime(fake_settings()))  # type: ignore[arg-type]
        transport.stream_sid = "MZ1"
        await transport.send_audio(np.zeros(24000 // 50 * 3 + 100, dtype=np.float32), 24000)  # 60 ms + a bit
        frames = [json.loads(s) for s in sent]
        assert [f["event"] for f in frames] == ["media", "media", "media"]
        assert transport._pending  # the partial frame waits for more audio
        await transport.clear_audio()
        assert json.loads(sent[-1]) == {"event": "clear", "streamSid": "MZ1"} and transport._pending == b""


class TestRest:
    def test_transfer_and_sms_requests(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(201, json={"sid": "SM123"})

        rest = TwilioRest("AC1", "token", "+15555550100", transport=httpx.MockTransport(handler))
        assert rest.send_sms("+15555550123", "hello").status == "sent"
        assert rest.redirect_call("CA9", transfer_twiml("+15555550199"))
        sms, redirect = requests
        assert sms.url.path == "/2010-04-01/Accounts/AC1/Messages.json"
        assert parse_qs(sms.content.decode()) == {"To": ["+15555550123"], "From": ["+15555550100"], "Body": ["hello"]}
        assert redirect.url.path == "/2010-04-01/Accounts/AC1/Calls/CA9.json"
        twiml = parse_qs(redirect.content.decode())["Twiml"][0]
        assert "<Dial" in twiml and "+15555550199" in twiml
        assert sms.headers["authorization"].startswith("Basic ")

    def test_dry_run_without_credentials(self) -> None:
        rest = TwilioRest("", "", "")
        assert rest.send_sms("+15555550123", "hi").status == "dry_run"
        assert rest.redirect_call("CA1", "<Response/>") is False
