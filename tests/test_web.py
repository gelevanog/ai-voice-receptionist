"""The browser-call WebSocket endpoint with fake components, and the dashboard pages."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from callie.audio.pcm import float_to_int16_bytes
from callie.runtime import build_runtime
from callie.speech import SpeechStack
from callie.stt import FakeSTT
from callie.tts import FakeTTS
from callie.vad import EnergyVAD
from callie.web.app import create_app
from tests.helpers import fake_settings, quiet, speech_like


def make_client(lines: list[str]) -> tuple[TestClient, Any]:
    settings = fake_settings(seed_demo_data=True)
    runtime = build_runtime(settings)
    speech = SpeechStack(stt=FakeSTT(lines), tts=FakeTTS(ms_per_char=3), vad_factory=EnergyVAD, settings=settings)
    return TestClient(create_app(settings, runtime=runtime, speech=speech)), runtime


def collect(ws: Any, until: str, limit: int = 2000) -> list[Any]:
    out: list[Any] = []
    for _ in range(limit):
        message = ws.receive()
        if message.get("text"):
            event = json.loads(message["text"])
            out.append(event)
            if event.get("type") == until:
                return out
        elif message.get("bytes") is not None:
            out.append(message["bytes"])
    raise AssertionError(f"no {until} event")


def test_browser_call_over_websocket() -> None:
    client, _runtime = make_client(["How much is a cleaning without insurance?"])
    with client, client.websocket_connect("/ws/call") as ws:
        first = collect(ws, "audio_format")
        assert first[0]["type"] == "call_started" and first[-1]["rate"] == 16000
        greeting = (
            collect(ws, "agent_text")
            if not any(isinstance(e, dict) and e.get("type") == "agent_text" for e in first)
            else first
        )
        assert greeting
        # Speak: 20 ms PCM16 frames at 16 kHz, in real time.
        audio = np.concatenate([quiet(0.6), speech_like(0.7), quiet(0.7)])
        for start in range(0, len(audio), 320):
            ws.send_bytes(float_to_int16_bytes(audio[start : start + 320]))
            time.sleep(0.02)
        events = collect(ws, "metrics")
        kinds = [e["type"] for e in events if isinstance(e, dict)]
        assert "transcript" in kinds and "tool" in kinds and "agent_text" in kinds
        assert any(isinstance(e, bytes) and len(e) % 2 == 0 for e in events)  # PCM16 audio frames
        metrics = next(e for e in events if isinstance(e, dict) and e["type"] == "metrics")
        assert metrics["voice_to_voice_ms"] is not None and metrics["stt_ms"] is not None
        ws.send_text(json.dumps({"type": "text", "text": "Do you take Delta Dental?"}))
        typed = collect(ws, "metrics")
        assert any(isinstance(e, dict) and e.get("type") == "tool" and e["name"] == "answer_faq" for e in typed)
        ws.send_text(json.dumps({"type": "hangup"}))
    calls = client.get("/api/calls").json()
    assert calls and calls[0]["transport"] == "browser"


def test_pages_render() -> None:
    client, _ = make_client([])
    with client:
        for path in ["/", "/calls", "/calendar", "/calendar?week=2026-10-12", "/eval", "/health", "/api/appointments"]:
            response = client.get(path)
            assert response.status_code == 200, path
        assert "Brightside Dental" in client.get("/").text
        assert "appointments this week" in client.get("/calendar").text
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["stack"]["llm"] == "fake/callie-rules"
        appointments = client.get("/api/appointments?days=14").json()
        assert appointments and all("*" in a["patient"] for a in appointments)  # masked
