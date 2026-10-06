"""Regenerate `media_stream_call.json`: a Twilio Media Streams session as the WebSocket receives it.

Message shapes follow Twilio's Media Streams docs (connected, start, media, dtmf, stop). The caller audio is a
synthetic voiced burst (no real person), encoded exactly like Twilio's: 20 ms frames of base64 μ-law at 8 kHz.
Run: uv run python tests/fixtures/twilio/generate.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from callie.audio import mulaw
from callie.audio.resample import resample

ACCOUNT = "AC" + "0" * 32
CALL = "CA" + "1" * 32
STREAM = "MZ" + "2" * 32


def caller_audio() -> np.ndarray:
    rate = 16000
    t = np.arange(round(0.9 * rate)) / rate
    rng = np.random.default_rng(7)
    voiced = np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 300 * t) + 0.2 * rng.standard_normal(len(t))
    voiced = 0.3 * voiced * (0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 3 * t))) / 1.7
    quiet = lambda s: rng.standard_normal(round(s * rate)) * 1e-4  # noqa: E731
    audio = np.concatenate([quiet(1.2), voiced, quiet(1.0)]).astype(np.float32)
    return resample(audio, rate, 8000)


def main() -> None:
    audio = caller_audio()
    encoded = mulaw.encode(audio)
    messages: list[dict[str, object]] = [{"event": "connected", "protocol": "Call", "version": "1.0.0"}]
    messages.append(
        {
            "event": "start",
            "sequenceNumber": "1",
            "streamSid": STREAM,
            "start": {
                "accountSid": ACCOUNT,
                "streamSid": STREAM,
                "callSid": CALL,
                "tracks": ["inbound"],
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
                "customParameters": {"from": "+15555550123", "callSid": CALL},
            },
        }
    )
    for index in range(0, len(encoded) - len(encoded) % 160, 160):
        chunk = index // 160 + 1
        messages.append(
            {
                "event": "media",
                "sequenceNumber": str(chunk + 1),
                "streamSid": STREAM,
                "media": {
                    "track": "inbound",
                    "chunk": str(chunk),
                    "timestamp": str(chunk * 20),
                    "payload": base64.b64encode(encoded[index : index + 160]).decode("ascii"),
                },
            }
        )
    messages.append(
        {
            "event": "dtmf",
            "streamSid": STREAM,
            "sequenceNumber": str(len(messages)),
            "dtmf": {"track": "inbound_track", "digit": "1"},
        }
    )
    messages.append(
        {
            "event": "stop",
            "sequenceNumber": str(len(messages)),
            "streamSid": STREAM,
            "stop": {"accountSid": ACCOUNT, "callSid": CALL},
        }
    )
    out = Path(__file__).with_name("media_stream_call.json")
    out.write_text(json.dumps(messages, indent=0) + "\n", encoding="utf-8")
    print(f"{out}: {len(messages)} messages, {len(encoded) / 8000:.2f} s of audio")


if __name__ == "__main__":
    main()
