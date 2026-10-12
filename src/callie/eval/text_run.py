"""Text-level conversation evaluation: the same scenarios, caller and checks as `eval run`, without audio.

The caller's words go straight to the agent and the agent's full answer goes straight back to the caller, so only
the LLM brain is measured: booking correctness, tool use, confirmation and escalation policy, and per-turn model
latency. Speech recognition is perfect here by construction; the end-to-end run through audio is `eval run`.

Audio-only parts of a scenario are adapted: an interruption is delivered as the caller's next turn (a backchannel
is dropped), the barge-in and backchannel checks are skipped, and the silent caller (handled by rules, no model
call) is left out. Records use the same format as `eval run`, so `report.e2e_summary` summarizes them.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from callie.agent.agent import CallAction, LLMTiming, Sentence, ToolEvent
from callie.config import Settings
from callie.eval.caller import Caller
from callie.eval.checks import check_call
from callie.eval.scenarios import Scenario, load_scenarios
from callie.eval.simulator import prepare_runtime
from callie.llm.base import ChatModel, JsonDict
from callie.llm.resilient import CallLedger

console = Console()
WORK_DIR = Path("data/eval/text")


def text_mode_scenarios(*, include_extended: bool = False, only: list[str] | None = None) -> list[Scenario]:
    """The scenarios that need a model in text mode (the silent caller is pure rules) with audio checks off."""
    chosen = [
        s
        for s in load_scenarios(include_extended=include_extended or bool(only))
        if (not only or s.id in only) and not s.silent
    ]
    return [
        s.model_copy(update={"expect": s.expect.model_copy(update={"barge_in": False, "backchannel": False})})
        for s in chosen
    ]


async def run_text_scenario(
    settings: Settings,
    scenario: Scenario,
    *,
    agent_llm: ChatModel,
    caller_llm: ChatModel | None,
    work_dir: Path = WORK_DIR,
    max_caller_turns: int = 6,
) -> dict[str, Any]:
    runtime, setup_ids = prepare_runtime(settings, scenario, agent_llm, work_dir / "db" / f"{scenario.id}.db")
    call_id = f"text_{scenario.id}_{int(time.time())}"
    ctx = runtime.new_context(caller_phone=scenario.caller_id, call_id=call_id)
    agent = runtime.new_agent(ctx)
    caller = Caller(scenario, caller_llm, max_turns=max_caller_turns)
    started = time.monotonic()
    transcript: list[JsonDict] = []
    turns: list[JsonDict] = []
    unsupported: list[str] = []
    replaced_advice = 0
    end_reason = "caller_hung_up"

    def since() -> float:
        return round(time.monotonic() - started, 2)

    greeting = agent.greeting()
    transcript.append({"t": since(), "role": "agent", "turn": 0, "text": ctx.masker.mask(greeting)})
    caller.heard(greeting)

    async def agent_turn(text: str, turn: int) -> tuple[str, CallAction | None]:
        nonlocal replaced_advice
        transcript.append({"t": since(), "role": "caller", "turn": turn, "text": ctx.masker.mask(text), "typed": False})
        t0 = time.monotonic()
        first_sentence: float | None = None
        first_token: float | None = None
        filler = False
        model = ""
        served: str | None = None
        errors: list[str] = []
        spoken: list[str] = []
        action: CallAction | None = None
        async for event in agent.respond(text, turn=turn):
            if isinstance(event, Sentence):
                if event.source == "filler":
                    filler = True
                    continue
                first_sentence = first_sentence or time.monotonic()
                spoken.append(event.text)
            elif isinstance(event, LLMTiming):
                model = event.model
                served = event.served_model or served
                if first_token is None and event.first_token_s is not None:
                    first_token = event.first_token_s
                if event.error:
                    errors.append(event.error)
            elif isinstance(event, CallAction):
                action = event
            elif isinstance(event, ToolEvent):
                pass
        stats = agent.last_stats
        unsupported.extend(stats.unsupported_times)
        replaced_advice += stats.replaced_advice
        said = " ".join(spoken).strip()
        transcript.append({"t": since(), "role": "agent", "turn": turn, "text": ctx.masker.mask(said)})
        turns.append(
            {
                "turn": turn,
                "path": stats.path,
                "model": served or model,
                "cached": False,
                "llm_calls": stats.llm_calls,
                "typed": False,
                "llm_first_token_ms": round(first_token * 1000) if first_token is not None else None,
                "agent_ms": round((first_sentence - t0) * 1000) if first_sentence is not None else None,
                "turn_ms": round((time.monotonic() - t0) * 1000),
                "filler": filler,
                "llm_errors": errors,
            }
        )
        return said, action

    turn = 0
    while True:
        text, ends = await caller.next_line()
        turn += 1
        said, action = await agent_turn(text, turn)
        if action is not None:
            end_reason = "transferred" if action.kind == "transfer" else "agent_hung_up"
            break
        if ends:
            break
        caller.heard(said)
        interrupt = scenario.interrupt
        if interrupt is not None and interrupt.turn == caller.turns and not interrupt.backchannel:
            # No audio to talk over: the interruption becomes the caller's next turn.
            caller.said(interrupt.text)
            turn += 1
            said, action = await agent_turn(interrupt.text, turn)
            if action is not None:
                end_reason = "transferred" if action.kind == "transfer" else "agent_hung_up"
                break
            caller.heard(said)

    for entry in transcript:  # names learned later in the call are masked in earlier lines too
        entry["text"] = ctx.masker.mask(entry["text"])
    summary: JsonDict = {
        "call_id": call_id,
        "outcome": ctx.outcome.label(),
        "outcome_detail": asdict(ctx.outcome),
        "end_reason": end_reason,
        "transcript": transcript,
        "tool_calls": ctx.tool_log,
        "turns": turns,
        "stats": {
            "barge_ins": [],
            "backchannels": 0,
            "silence_prompts": 0,
            "unsupported_times": unsupported,
            "replaced_advice": replaced_advice,
        },
        "stack": {"mode": "text", "llm": agent_llm.label},
    }
    check = check_call(scenario, summary, runtime.sessions, runtime.clinic, setup_ids, call_id=call_id)
    tool_errors = [
        {"turn": c["turn"], "name": c["name"], "error": str(c["result"].get("error", ""))[:160]}
        for c in ctx.tool_log
        if c["result"].get("status") == "error"
    ]
    return {
        **check,
        "call_id": call_id,
        "end_reason": end_reason,
        "transcript": transcript,
        "tool_calls": ctx.tool_log,
        "turns": turns,
        "stats": summary["stats"],
        "tool_errors": tool_errors,
        "caller_errors": caller.errors,
        "stack": summary["stack"],
    }


def _ledger_rows(ledger: CallLedger, start: int) -> list[JsonDict]:
    if ledger.path is None or not ledger.path.exists():
        return []
    lines = ledger.path.read_text(encoding="utf-8").splitlines()[start:]
    return [json.loads(line) for line in lines if line.strip()]


def _ledger_len(ledger: CallLedger) -> int:
    if ledger.path is None or not ledger.path.exists():
        return 0
    return sum(1 for line in ledger.path.read_text(encoding="utf-8").splitlines() if line.strip())


def _usage(rows: list[JsonDict]) -> JsonDict:
    usage: JsonDict = {}
    for tag in sorted({str(r.get("tag")) for r in rows}):
        tagged = [r for r in rows if r.get("tag") == tag]
        usage[tag] = {
            "requests": len(tagged),
            "failed": sum(1 for r in tagged if r["status"] != "ok"),
            "input_tokens": sum(r.get("input_tokens") or 0 for r in tagged),
            "output_tokens": sum(r.get("output_tokens") or 0 for r in tagged),
        }
    return usage


async def run_text_eval(
    settings: Settings,
    scenarios: list[Scenario],
    *,
    agent_llm: ChatModel,
    caller_llm: ChatModel | None,
    caller_label: str,
    ledger: CallLedger,
    out_path: Path,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    records: list[dict[str, Any]] = []
    previous = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    for index, scenario in enumerate(scenarios, 1):
        console.print(f"[bold]{index}/{len(scenarios)}[/] {scenario.id} ({scenario.category})")
        before = _ledger_len(ledger)
        t0 = time.monotonic()
        record = await run_text_scenario(
            settings,
            scenario,
            agent_llm=agent_llm,
            caller_llm=caller_llm,
            work_dir=WORK_DIR / out_path.stem,  # one database directory per run, so runs can go in parallel
        )
        record["wall_s"] = round(time.monotonic() - t0, 1)
        record["usage"] = _usage(_ledger_rows(ledger, before))
        records.append(record)
        verdict = "[green]PASS[/]" if record["passed"] else f"[red]FAIL[/] {record['problems']}"
        agent_tokens = record["usage"].get("agent", {})
        console.print(
            f"   {verdict} outcome={record['outcome']} turns={record['caller_turns']} {record['wall_s']}s "
            f"agent tokens in/out={agent_tokens.get('input_tokens', 0)}/{agent_tokens.get('output_tokens', 0)}"
        )
        data = _write(out_path, settings, agent_llm, caller_label, started, _merge(previous, records))
    return data


def _merge(previous: dict[str, Any] | None, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not previous:
        return records
    fresh = {r["scenario"] for r in records}
    return [r for r in previous["scenarios"] if r["scenario"] not in fresh] + records


def _write(
    path: Path,
    settings: Settings,
    agent_llm: ChatModel,
    caller_label: str,
    started: datetime,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    data = {
        "run": {
            "mode": "text (no audio: caller text in, agent text out; perfect speech recognition by construction)",
            "started": started.isoformat(timespec="seconds"),
            "finished": datetime.now(UTC).isoformat(timespec="seconds"),
            "agent_llm": agent_llm.label,
            "agent_fallbacks": settings.llm_fallback_models,
            "agent_temperature": settings.llm_temperature if settings.llm_send_temperature else None,
            "agent_max_tokens": settings.llm_max_tokens,
            "caller": caller_label,
            "clinic_clock": settings.now,
        },
        "scenarios": records,
    }
    path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    return data
