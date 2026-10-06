"""STT word error rate on the simulated callers' utterances: clean 16 kHz vs phone line vs phone line + noise.

The utterances are the ones the end-to-end run synthesized (LLM-written caller lines spoken by Piper voices), so
this measures recognition of synthetic speech, not of real callers; treat it as a relative comparison between
models and channels. Each condition is rendered from the same clean audio with a fixed seed.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from callie.audio.pcm import read_wav
from callie.config import Settings
from callie.eval.metrics import corpus_wer, summarize
from callie.eval.simulator import apply_channel
from callie.stt import WhisperSTT

console = Console()
MANIFEST = Path("data/eval/utterances.jsonl")
CONDITIONS = ["clean", "phone", "phone_noisy"]


def load_manifest(path: Path = MANIFEST) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"{path} not found: run `callie eval run` first (it saves the caller utterances)")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:  # the latest run wins for each file
        seen[row["wav"]] = row
    return [r for r in seen.values() if Path(r["wav"]).exists() and r["text"].strip()]


def run_wer(settings: Settings, models: list[str], out_dir: Path) -> dict[str, Any]:
    rows = load_manifest()
    audio = {r["wav"]: read_wav(Path(r["wav"]))[0] for r in rows}
    rendered: dict[str, dict[str, np.ndarray]] = {}
    for condition in CONDITIONS:
        rng = np.random.default_rng(42)
        rendered[condition] = {key: apply_channel(clip, condition, rng) for key, clip in audio.items()}
    results: dict[str, Any] = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "utterances": len(rows),
        "words": sum(len(r["text"].split()) for r in rows),
        "note": "Synthetic caller speech (Piper libritts_r voices), LLM-written lines; not real callers.",
        "models": {},
    }
    table = Table("model", *CONDITIONS, "latency p50 / p95 (s)")
    for name in models:
        stt = WhisperSTT(name, threads=settings.stt_threads)
        stt.load()
        entry: dict[str, Any] = {}
        latencies: list[float] = []
        for condition in CONDITIONS:
            pairs = []
            for row in rows:
                started = time.perf_counter()
                hypothesis = stt.transcribe_sync(rendered[condition][row["wav"]]).text
                latencies.append(time.perf_counter() - started)
                pairs.append((row["text"], hypothesis))
            entry[condition] = round(corpus_wer(pairs), 4)
            if name == settings.stt_model:
                entry[f"examples_{condition}"] = [{"ref": r, "hyp": h} for r, h in pairs[:12]]
        entry["latency_s"] = summarize(latencies)
        results["models"][name] = entry
        table.add_row(name, *(f"{entry[c]:.1%}" for c in CONDITIONS),
                      f"{entry['latency_s']['p50']} / {entry['latency_s']['p95']}")  # fmt: skip
        console.print(table)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wer.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results
