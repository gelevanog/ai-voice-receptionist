"""List free OpenRouter models and smoke-test candidates for voice: tool calling and time to first token.

One streaming request per candidate (retries and the call are recorded in the ledger), with the real system
prompt and tools: "I'd like to book a cleaning next Tuesday afternoon" must produce a `check_availability` call
with the caller's words in `when`.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from callie.agent.prompts import greeting, system_prompt
from callie.agent.tools import tool_specs
from callie.clinic import load_clinic
from callie.config import Settings
from callie.llm.base import ProviderError, TextDelta, ToolCall, ensure_free_models
from callie.llm.factory import build_chat_model
from callie.llm.openai_compat import list_free_models
from callie.scheduling.timeparse import clinic_now

DEFAULT_CANDIDATES = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "liquid/lfm-2.5-2.6b:free",
    "dots-studio/dots-3-note-preview:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]
console = Console()


async def list_and_smoke(settings: Settings, smoke: int, candidates: list[str] | None) -> None:
    models = await list_free_models(os.environ.get("OPENROUTER_API_KEY"))
    table = Table("free model", "context", "tools")
    for model in sorted(models, key=lambda m: m["id"]):
        table.add_row(model["id"], str(model["context_length"]), "yes" if model["tools"] else "")
    console.print(table)
    out = Path(settings.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    listing = {"date": datetime.now(UTC).isoformat(timespec="seconds"), "free_models": models}
    if not smoke:
        (out / "free_models.json").write_text(json.dumps(listing, indent=2), encoding="utf-8")
        return
    available = {m["id"] for m in models if m["tools"]}
    chosen = [c for c in (candidates or DEFAULT_CANDIDATES) if c in available][:smoke]
    ensure_free_models(chosen)
    results = [await smoke_one(settings, model) for model in chosen]
    listing["smoke"] = results
    (out / "smoke_models.json").write_text(json.dumps(listing, indent=2), encoding="utf-8")
    table = Table("model", "ok", "tool call", "when", "first token s", "total s", "error")
    for r in results:
        table.add_row(r["model"], str(r["ok"]), r.get("tool") or "", r.get("when") or "", str(r.get("ttft_s")),
                      str(r.get("total_s")), (r.get("error") or "")[:60])  # fmt: skip
    console.print(table)


async def smoke_one(settings: Settings, model: str) -> dict[str, object]:
    clinic = load_clinic(settings.clinic_file)
    now = clinic_now(clinic.timezone, settings.now or "2026-10-06T09:30")
    chat = build_chat_model(settings, clinic, provider="openrouter", model=model, fallback_models=[], tag="smoke")
    messages = [
        {"role": "system", "content": system_prompt(clinic, now, None)},
        {"role": "assistant", "content": greeting(clinic)},
        {"role": "user", "content": "Hi, I'd like to book a cleaning next Tuesday afternoon."},
    ]
    started = time.monotonic()
    first: float | None = None
    text, tool, when = "", None, None
    try:
        async for event in chat.stream(messages, tool_specs(clinic), max_tokens=300, temperature=0.3):
            if isinstance(event, TextDelta | ToolCall) and first is None:
                first = time.monotonic() - started
            if isinstance(event, TextDelta):
                text += event.text
            elif isinstance(event, ToolCall) and tool is None:
                tool, when = event.name, str(event.arguments.get("when", ""))
    except ProviderError as exc:
        return {"model": model, "ok": False, "error": str(exc)[:300], "total_s": round(time.monotonic() - started, 2)}
    return {
        "model": model,
        "ok": tool == "check_availability",
        "tool": tool,
        "when": when,
        "text": text[:200],
        "ttft_s": round(first, 2) if first is not None else None,
        "total_s": round(time.monotonic() - started, 2),
    }
