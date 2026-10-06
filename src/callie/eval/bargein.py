"""Barge-in benchmark: real VAD, STT and TTS, no API calls.

Each trial asks a question that has a long answer (typed, so the trial does not depend on the model), waits
until Callie has been speaking for a random 0.8-2.5 s, then plays a caller utterance into the line: either an
interruption ("Sorry, can I ask something else?") or a backchannel ("Mm-hmm."). Measured:
- reaction time: from the first voiced sample of the caller's speech to the moment Callie's audio is cleared;
- interruptions handled: the rest of the answer was dropped and the caller's words became the next turn;
- backchannels handled: playback resumed and the whole answer was heard (no false interruption).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from callie.audio.pcm import PIPELINE_RATE
from callie.audio.resample import resample
from callie.clinic import load_clinic
from callie.config import Settings
from callie.eval.metrics import summarize
from callie.eval.simulator import LineFeeder, SimTransport, _normalize_level, preload
from callie.llm.fake import FakeReceptionist
from callie.pipeline.session import CallSession, SessionConfig
from callie.runtime import build_runtime
from callie.speech import build_speech
from callie.tts import PiperTTS

console = Console()
INTERRUPTIONS = [
    "Sorry, can I ask something else?",
    "Wait, wait, hold on a second.",
    "Actually, I just need the address.",
    "Sorry to interrupt, do you take my insurance?",
    "Hang on, I think I need to cancel instead.",
]
BACKCHANNELS = ["Mm-hmm.", "Okay.", "Uh-huh.", "Right.", "Yeah."]
QUESTION = "How much is a cleaning without insurance?"  # a long answer (the full price list)


async def _trial(
    settings: Settings,
    speech: Any,
    caller_tts: PiperTTS,
    text: str,
    backchannel: bool,
    offset_s: float,
    speaker: int,
    seed: int,
) -> dict[str, Any]:
    clinic = load_clinic(settings.clinic_file)
    runtime = build_runtime(
        settings.model_copy(update={"database_url": "sqlite://", "seed_demo_data": False}), llm=FakeReceptionist(clinic)
    )
    transport = SimTransport()
    session = CallSession(
        runtime,
        speech,
        transport,
        config=SessionConfig(
            save_record=False,
            record=False,
            barge_in_min_speech_s=settings.barge_in_min_speech_ms / 1000,
            hard_interrupt_s=settings.hard_interrupt_ms / 1000,
        ),
    )
    feeder = LineFeeder(session, "clean", np.random.default_rng(seed))
    feeder.start()
    await session.start()
    while session.player.active:
        await asyncio.sleep(0.05)
    await session.feed_text(QUESTION)
    while not (session._turn_metrics and session._turn_metrics.playback_start):
        await asyncio.sleep(0.01)
    caller_tts.speaker_id = speaker
    spoken = await caller_tts.synthesize(text)
    clean = _normalize_level(resample(spoken.audio, spoken.rate, PIPELINE_RATE))
    voiced = np.flatnonzero(np.abs(clean) > 0.01)
    await asyncio.sleep(max(0.0, offset_s - (time.monotonic() - session._turn_metrics.playback_start)))
    clears_before = len(transport.clears)
    feeder.enqueue(clean, int(voiced[0]) if len(voiced) else 0)
    while not feeder.speech_started_wall:
        await asyncio.sleep(0.005)
    onset = feeder.speech_started_wall[0]
    await feeder.idle.wait()
    heard_before = len(session.raw_caller)
    deadline = time.monotonic() + 4  # end-of-turn silence + STT for the caller's words
    while time.monotonic() < deadline and len(session.raw_caller) == heard_before and not session.stats.backchannels:
        await asyncio.sleep(0.05)
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline and (
        session.player.active or (session._turn_task and not session._turn_task.done())
    ):
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.3)
    clears = [c for c in transport.clears[clears_before:] if c >= onset]
    answer = [u for u in session.player.history if u.turn == 1]
    result = session.stats.barge_ins[0]["result"] if session.stats.barge_ins else "not_detected"
    record = {
        "kind": "backchannel" if backchannel else "interruption",
        "text": text,
        "offset_s": round(offset_s, 2),
        "reaction_ms": round((clears[0] - onset) * 1000) if clears else None,
        "session_reaction_ms": session.stats.barge_ins[0]["reaction_ms"] if session.stats.barge_ins else None,
        "result": result,
        "answer_fully_heard": bool(answer) and all(u.heard_text() == u.text for u in answer),
        "stt": session.raw_caller[-1] if session.raw_caller else "",
        "correct": (result == "backchannel_resumed") if backchannel else (result == "interrupted"),
    }
    await session.close("benchmark")
    await feeder.stop()
    return record


async def run_bargein(settings: Settings, trials: int, out_dir: Path) -> dict[str, Any]:
    speech = build_speech(settings)
    caller_tts = PiperTTS(settings.models_dir / "en_US-libritts_r-medium.onnx")
    await preload(speech, caller_tts)
    rng = random.Random(11)
    plan = [(INTERRUPTIONS[i % len(INTERRUPTIONS)], False) for i in range(trials)]
    plan += [(BACKCHANNELS[i % len(BACKCHANNELS)], True) for i in range(trials)]
    rng.shuffle(plan)
    records = []
    for index, (text, backchannel) in enumerate(plan):
        record = await _trial(
            settings,
            speech,
            caller_tts,
            text,
            backchannel,
            rng.uniform(0.8, 2.5),
            rng.choice([12, 45, 101, 230, 377, 512, 640, 803]),
            index,
        )
        records.append(record)
        mark = "[green]ok[/]" if record["correct"] else "[red]wrong[/]"
        console.print(
            f"{index + 1:>2}/{len(plan)} {record['kind']:<12} {mark} {record['result']:<20} "
            f"reaction {record['reaction_ms']} ms  stt={record['stt']!r}"
        )
    interruptions = [r for r in records if r["kind"] == "interruption"]
    backchannels = [r for r in records if r["kind"] == "backchannel"]
    summary = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "trials": len(records),
        "settings": {
            "barge_in_min_speech_ms": settings.barge_in_min_speech_ms,
            "hard_interrupt_ms": settings.hard_interrupt_ms,
            "vad": settings.vad_provider,
            "stt": settings.stt_model,
        },
        "reaction_ms": summarize([r["reaction_ms"] for r in records if r["reaction_ms"] is not None]),
        "interruptions_handled": [sum(r["correct"] for r in interruptions), len(interruptions)],
        "backchannels_handled": [sum(r["correct"] for r in backchannels), len(backchannels)],
        "false_interruptions": sum(1 for r in backchannels if r["result"] == "interrupted"),
        "records": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bargein.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    console.print_json(json.dumps({k: v for k, v in summary.items() if k != "records"}))
    return summary
