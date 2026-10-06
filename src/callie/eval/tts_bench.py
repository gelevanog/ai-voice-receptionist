"""TTS choice: time to synthesize a sentence on this CPU (Kokoro fp32 / int8, Piper), and whether Callie's voice is
intelligible (Whisper word error rate on Kokoro's own output)."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from callie.audio.resample import resample
from callie.config import Settings
from callie.eval.metrics import corpus_wer, summarize
from callie.stt import WhisperSTT
from callie.tts import TTS, KokoroTTS, PiperTTS

SENTENCES = [
    "Thanks for calling Brightside Dental.",
    "I have Tuesday, October 13th at 2:30 PM, 3 PM or 3:30 PM.",
    "Just to confirm: a cleaning and checkup for Maria Gonzalez on Tuesday, October 13th at 2:30 PM.",
    "I'll text the confirmation to the number ending in 8 8 3 9. Is that right?",
    "We are in network with Delta Dental, Cigna, MetLife, Aetna and Guardian PPO plans.",
    "If this is life-threatening, please hang up and call 911.",
    "Our office is at 418 Linden Street, on the second floor above the pharmacy.",
    "Without insurance, a cleaning and checkup is $120.",
]


def _bench(tts: TTS, repeats: int = 2) -> dict[str, Any]:
    synth = tts.synthesize_sync  # type: ignore[attr-defined]
    synth("Warm up.")
    seconds, audio_seconds = [], 0.0
    for _ in range(repeats):
        for sentence in SENTENCES:
            started = time.perf_counter()
            speech = synth(sentence)
            seconds.append(time.perf_counter() - started)
            audio_seconds += len(speech.audio) / speech.rate
    return {"per_sentence_s": summarize(seconds), "real_time_factor": round(sum(seconds) / audio_seconds, 3)}


def run_tts_bench(settings: Settings, out_dir: Path) -> dict[str, Any]:
    models = settings.models_dir
    engines: dict[str, TTS] = {
        "kokoro-82m fp32 (agent voice)": KokoroTTS(models / "kokoro-v1.0.onnx", models / "voices-v1.0.bin", threads=4),
    }
    if (models / "kokoro-v1.0.int8.onnx").exists():
        engines["kokoro-82m int8"] = KokoroTTS(models / "kokoro-v1.0.int8.onnx", models / "voices-v1.0.bin", threads=4)
    engines["piper libritts_r medium (caller voices)"] = PiperTTS(
        models / "en_US-libritts_r-medium.onnx", speaker_id=12
    )
    results: dict[str, Any] = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "cpu_count": os.cpu_count(),
        "load_avg_1m": os.getloadavg()[0],
        "sentences": len(SENTENCES),
        "engines": {name: _bench(engine) for name, engine in engines.items()},
    }
    stt = WhisperSTT(settings.stt_model, threads=settings.stt_threads)
    kokoro = next(iter(engines.values()))
    pairs = []
    for sentence in SENTENCES:
        speech = kokoro.synthesize_sync(sentence)  # type: ignore[attr-defined]
        pairs.append((sentence, stt.transcribe_sync(resample(speech.audio, speech.rate, 16000)).text))
    results["agent_voice_intelligibility"] = {
        "stt": stt.name,
        "wer": round(corpus_wer(pairs), 4),
        "examples": [{"ref": r, "hyp": h} for r, h in pairs],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tts_bench.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results
