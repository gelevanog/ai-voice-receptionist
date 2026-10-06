"""End-to-end evaluation: every scenario as a simulated phone call through audio, graded deterministically."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from callie.audio.pcm import write_wav
from callie.clinic import load_clinic
from callie.config import Settings
from callie.eval.scenarios import load_scenarios
from callie.eval.simulator import SimResult, caller_tts_for, preload, simulate_call
from callie.llm.factory import build_chat_model
from callie.speech import build_speech

console = Console()
WORK_DIR = Path("data/eval")


async def _sample_load(samples: list[float], stop: asyncio.Event) -> None:
    while not stop.is_set():
        samples.append(os.getloadavg()[0])
        try:
            await asyncio.wait_for(stop.wait(), timeout=10)
        except TimeoutError:
            continue


def _record(result: SimResult) -> dict[str, Any]:
    summary = result.summary
    return {
        **result.check,
        "call_id": summary["call_id"],
        "end_reason": summary["end_reason"],
        "transcript": summary["transcript"],
        "tool_calls": summary["tool_calls"],
        "turns": summary["turns"],
        "stats": summary["stats"],
        "barge_in_truth": [{k: v for k, v in t.items() if k != "onset"} for t in result.barge_in_truth],
        "pipeline_wer": result.pipeline_wer,
        "caller_errors": result.caller_errors,
        "recording": result.recording,
        "stack": summary["stack"],
    }


async def run_e2e(
    settings: Settings,
    *,
    caller_mode: str,
    caller_model: str,
    only: list[str] | None,
    out_dir: Path,
    tag: str,
) -> dict[str, Any]:
    scenarios = [s for s in load_scenarios() if not only or s.id in only]
    clinic = load_clinic(settings.clinic_file)
    speech = build_speech(settings)
    agent_llm = build_chat_model(settings, clinic, tag="agent")
    caller_llm = (
        build_chat_model(
            settings,
            clinic,
            provider="openrouter",
            model=caller_model,
            fallback_models=[],
            tag="caller",
        )
        if caller_mode == "llm"
        else None
    )
    caller_tts = caller_tts_for(settings, fake=settings.tts_provider == "fake")
    await preload(speech, caller_tts)
    out_dir.mkdir(parents=True, exist_ok=True)
    utterance_dir = WORK_DIR / "utterances"
    utterance_dir.mkdir(parents=True, exist_ok=True)
    manifest = (WORK_DIR / "utterances.jsonl").open("a", encoding="utf-8")
    load: list[float] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(_sample_load(load, stop))
    started = datetime.now(UTC)
    records: list[dict[str, Any]] = []
    path = out_dir / f"{tag}.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() and only else None
    try:
        for index, scenario in enumerate(scenarios, 1):
            t0 = time.monotonic()
            console.print(f"[bold]{index}/{len(scenarios)}[/] {scenario.id} ({scenario.category}, {scenario.channel})")
            result = await simulate_call(settings, scenario, speech=speech, agent_llm=agent_llm, caller_llm=caller_llm,
                                         caller_tts=caller_tts, work_dir=WORK_DIR)  # fmt: skip
            record = _record(result)
            record["wall_s"] = round(time.monotonic() - t0, 1)
            records.append(record)
            for n, utterance in enumerate(result.utterances):
                wav = utterance_dir / f"{scenario.id}_{n:02d}.wav"
                write_wav(wav, utterance.clean, 16000)
                manifest.write(json.dumps({"scenario": scenario.id, "channel": scenario.channel, "voice": scenario.voice,
                                           "text": utterance.text, "wav": str(wav), "interrupt": utterance.interrupt}) + "\n")  # fmt: skip
            manifest.flush()
            verdict = "[green]PASS[/]" if record["passed"] else f"[red]FAIL[/] {record['problems']}"
            console.print(
                f"   {verdict} outcome={record['outcome']} turns={record['caller_turns']} {record['wall_s']}s"
            )
            _write(path, settings, caller_mode, caller_model, started, load, _merge(previous, records))
    finally:
        stop.set()
        await sampler
        manifest.close()
    return _write(path, settings, caller_mode, caller_model, started, load, _merge(previous, records))


def _merge(previous: dict[str, Any] | None, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not previous:
        return records
    fresh = {r["scenario"] for r in records}
    return [r for r in previous["scenarios"] if r["scenario"] not in fresh] + records


def _write(
    path: Path,
    settings: Settings,
    caller_mode: str,
    caller_model: str,
    started: datetime,
    load: list[float],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    stack = records[-1]["stack"] if records else {}
    data = {
        "run": {
            "started": started.isoformat(timespec="seconds"),
            "finished": datetime.now(UTC).isoformat(timespec="seconds"),
            "agent_llm": stack.get("llm"),
            "agent_fallbacks": settings.llm_fallback_models,
            "caller": f"openrouter/{caller_model}" if caller_mode == "llm" else "scripted",
            "stt": stack.get("stt"),
            "tts_agent": stack.get("tts"),
            "tts_caller": "piper/en_US-libritts_r-medium (one speaker per scenario)",
            "vad": stack.get("vad"),
            "endpoint_silence_ms": settings.endpoint_silence_ms,
            "barge_in_min_speech_ms": settings.barge_in_min_speech_ms,
            "cpu_count": os.cpu_count(),
            "load_avg_1m": {
                "mean": round(sum(load) / len(load), 2) if load else None,
                "max": max(load) if load else None,
            },
            "clinic_clock": settings.now,
        },
        "scenarios": records,
    }
    path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    return data
