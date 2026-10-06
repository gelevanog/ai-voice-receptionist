"""Combine result files into results/summary.json and results/report.md; summarize the call ledger."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from callie.eval.metrics import summarize

STAGES = [
    ("endpointing_ms", "End of turn (silence wait)"),
    ("stt_ms", "STT (faster-whisper)"),
    ("llm_first_token_ms", "LLM time to first token"),
    ("agent_ms", "LLM + tools to first sentence"),
    ("tts_ms", "TTS to first audio"),
    ("voice_to_voice_ms", "Voice to voice"),
    ("first_audio_ms", "First audio incl. filler"),
]


def ledger_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"calls": 0}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    models = Counter(r.get("requested_model") for r in rows)
    served = Counter(r.get("served_model") for r in rows if r.get("served_model"))
    ok = [r for r in rows if r["status"] == "ok"]
    ttft_by_tag: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        if r.get("ttft_s") is not None:
            ttft_by_tag[r["tag"]].append(r["ttft_s"])
    return {
        "calls": len(rows),
        "by_tag": dict(Counter(r["tag"] for r in rows)),
        "by_status": dict(Counter(r["status"] for r in rows)),
        "requested_models": dict(models),
        "served_models": dict(served),
        "all_requested_free": all(str(m).endswith(":free") for m in models),
        "all_served_free": all(str(m).endswith(":free") for m in served),
        "ttft_s_by_tag": {tag: summarize(values) for tag, values in ttft_by_tag.items()},
        "input_tokens": sum(r.get("input_tokens") or 0 for r in rows),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in rows),
        "first": rows[0]["ts"] if rows else None,
        "last": rows[-1]["ts"] if rows else None,
    }


def _offered_and_booked(record: dict[str, Any]) -> tuple[int, int]:
    """Bookings/reschedules whose slot was never returned by check_availability in the same call."""
    offered: set[str] = set()
    changes = unoffered = 0
    for call in record["tool_calls"]:
        result = call["result"]
        if call["name"] == "check_availability":
            offered.update(s["slot_id"] for s in result.get("slots", []) + result.get("alternatives", []))
        if result.get("status") in {"booked", "rescheduled"}:
            changes += 1
            if call["arguments"].get("slot_id") not in offered:
                unoffered += 1
    return changes, unoffered


def e2e_summary(data: dict[str, Any]) -> dict[str, Any]:
    scenarios = data["scenarios"]
    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_channel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in scenarios:
        for bucket, key in ((by_category, record["category"]), (by_channel, record["channel"])):
            bucket[key][0] += int(record["passed"])
            bucket[key][1] += 1
    turns = [t for r in scenarios for t in r["turns"] if not t.get("typed")]
    llm_turns = [t for t in turns if t["path"] == "llm" and not t.get("cached")]
    rule_turns = [t for t in turns if t["path"] in {"fast_confirm", "rules"}]
    latency = {
        "llm_turns": {key: summarize([t[key] for t in llm_turns if t.get(key) is not None]) for key, _ in STAGES},
        "rule_turns": {key: summarize([t[key] for t in rule_turns if t.get(key) is not None]) for key, _ in STAGES},
        "all_turns": {
            key: summarize([t[key] for t in turns if t.get(key) is not None and not t.get("cached")])
            for key, _ in STAGES
        },
        "filler_share": round(sum(1 for t in llm_turns if t.get("filler")) / len(llm_turns), 3) if llm_turns else None,
        "cached_turns": sum(1 for t in turns if t.get("cached")),
    }
    confirmations = [r["confirmation"] for r in scenarios]
    grounding = [r["grounding"] for r in scenarios]
    escalation_expected = [r for r in scenarios if r["category"] == "escalation"]
    unoffered = [_offered_and_booked(r) for r in scenarios]
    barge_truth = [t for r in scenarios for t in r.get("barge_in_truth", [])]
    return {
        "run": data["run"],
        "task_success": {
            "passed": sum(r["passed"] for r in scenarios),
            "total": len(scenarios),
            "by_category": dict(by_category),
            "by_channel": dict(by_channel),
            "failures": [{"scenario": r["scenario"], "problems": r["problems"]} for r in scenarios if not r["passed"]],
        },
        "caller_turns": summarize([r["caller_turns"] for r in scenarios]),
        "confirmation": {
            "changes": sum(c["changes"] for c in confirmations),
            "with_readback_and_yes": sum(c["with_readback_and_yes"] for c in confirmations),
            "blocked_unconfirmed_attempts": sum(c["blocked_unconfirmed"] for c in confirmations),
            "false_completion_claims": sum(r["false_claims"] for r in scenarios),
        },
        "hallucinated_availability": {
            "bookings_on_unoffered_slots": sum(u for _, u in unoffered),
            "calendar_changes": sum(c for c, _ in unoffered),
            "unoffered_slot_attempts_blocked": sum(g["unoffered_slot_attempts"] for g in grounding),
            "ungrounded_times_spoken_by_llm": sum(len(g["unsupported_time_mentions"]) for g in grounding),
            "examples": [t for g in grounding for t in g["unsupported_time_mentions"]][:10],
        },
        "escalation": {
            "expected": len(escalation_expected),
            "transferred_when_expected": sum(1 for r in escalation_expected if r["transferred"]),
            "transferred_when_not_expected": sum(
                1 for r in scenarios if r["transferred"] and r["category"] != "escalation"
            ),
        },
        "advice_sentences_replaced": sum(r["replaced_advice"] for r in scenarios),
        "latency_ms": latency,
        "barge_in_e2e": {
            "events": len(barge_truth),
            "reaction_ms": summarize([t["reaction_ms"] for t in barge_truth if t.get("reaction_ms") is not None]),
        },
        "pipeline_wer": summarize([r["pipeline_wer"] for r in scenarios if r.get("pipeline_wer") is not None]),
        "scenarios": [
            {
                "scenario": r["scenario"],
                "category": r["category"],
                "channel": r["channel"],
                "passed": r["passed"],
                "outcome": r["outcome"],
                "caller_turns": r["caller_turns"],
                "problems": r["problems"],
                "v2v_p50_ms": summarize(
                    [t["voice_to_voice_ms"] for t in r["turns"] if t.get("voice_to_voice_ms") is not None]
                )["p50"],
                "call_id": r["call_id"],
            }
            for r in scenarios
        ],
    }


def build_report(out_dir: Path) -> str:
    summary: dict[str, Any] = {}
    e2e_path = out_dir / "e2e.json"
    if e2e_path.exists():
        summary["e2e"] = e2e_summary(json.loads(e2e_path.read_text(encoding="utf-8")))
    for name in ("wer", "bargein", "smoke_models", "tts_bench"):
        path = out_dir / f"{name}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if name == "bargein":
                data = {k: v for k, v in data.items() if k != "records"}
            summary[name] = data
    summary["calls"] = ledger_summary(out_dir / "calls.jsonl")
    (out_dir / "calls_summary.json").write_text(json.dumps(summary["calls"], indent=1), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    report = render_markdown(summary)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return report


def _pct(passed: int, total: int) -> str:
    return f"{passed}/{total} ({passed / total:.0%})" if total else "–"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Callie evaluation report", ""]
    e2e = summary.get("e2e")
    if e2e:
        run = e2e["run"]
        lines += [
            f"Run {run['started']} to {run['finished']} (UTC). Agent LLM `{run['agent_llm']}`, "
            f"caller `{run['caller']}`, STT `{run['stt']}`, agent TTS `{run['tts_agent']}`, "
            f"caller TTS {run['tts_caller']}, VAD `{run['vad']}`. {run['cpu_count']} CPU cores, "
            f"1-minute load average mean {run['load_avg_1m']['mean']} / max {run['load_avg_1m']['max']}.",
            "",
            "## Task success",
            "",
            f"**{_pct(e2e['task_success']['passed'], e2e['task_success']['total'])}** scenarios passed.",
            "",
            "| Category | Passed |",
            "|---|---|",
        ]
        lines += [f"| {k} | {_pct(*v)} |" for k, v in sorted(e2e["task_success"]["by_category"].items())]
        lines += ["", "| Channel | Passed |", "|---|---|"]
        lines += [f"| {k} | {_pct(*v)} |" for k, v in sorted(e2e["task_success"]["by_channel"].items())]
        if e2e["task_success"]["failures"]:
            lines += ["", "Failures:", ""]
            lines += [f"- `{f['scenario']}`: {'; '.join(f['problems'])}" for f in e2e["task_success"]["failures"]]
        conf, hall, esc = e2e["confirmation"], e2e["hallucinated_availability"], e2e["escalation"]
        lines += [
            "",
            "## Safety and grounding",
            "",
            f"- Calendar changes with a read-back and a yes: {conf['with_readback_and_yes']}/{conf['changes']}; "
            "confirmed=true attempts refused because the caller had not said yes: "
            f"{conf['blocked_unconfirmed_attempts']}; "
            f"agent claims of a completed change without one: {conf['false_completion_claims']}.",
            "- Bookings on slots not offered by a tool: "
            f"{hall['bookings_on_unoffered_slots']}/{hall['calendar_changes']}; "
            f"attempts blocked: {hall['unoffered_slot_attempts_blocked']}; ungrounded clock times spoken by the LLM: "
            f"{hall['ungrounded_times_spoken_by_llm']} {hall['examples']}.",
            f"- Escalations: {esc['transferred_when_expected']}/{esc['expected']} expected transfers happened; "
            f"{esc['transferred_when_not_expected']} unexpected transfers.",
            f"- Caller turns per call: {e2e['caller_turns']}.",
            "",
            "## Latency (ms)",
            "",
            "| Stage | LLM turns p50 | p95 | rule turns p50 | p95 |",
            "|---|---|---|---|---|",
        ]
        for key, label in STAGES:
            a, b = e2e["latency_ms"]["llm_turns"][key], e2e["latency_ms"]["rule_turns"][key]
            lines.append(f"| {label} | {a['p50']} | {a['p95']} | {b['p50']} | {b['p95']} |")
        lines += [
            "",
            f"Barge-in reaction in the scenarios: {e2e['barge_in_e2e']}.",
            f"In-pipeline WER per call: {e2e['pipeline_wer']}.",
        ]
    bargein = summary.get("bargein")
    if bargein:
        lines += [
            "",
            "## Barge-in benchmark",
            "",
            f"Reaction time {bargein['reaction_ms']}; interruptions handled {_pct(*bargein['interruptions_handled'])}; "
            f"backchannels handled {_pct(*bargein['backchannels_handled'])}; "
            f"false interruptions {bargein['false_interruptions']}.",
        ]
    wer = summary.get("wer")
    if wer:
        lines += [
            "",
            "## STT word error rate",
            "",
            f"{wer['utterances']} utterances, {wer['words']} words. {wer['note']}",
            "",
            "| Model | clean | phone | phone + noise | latency p50 / p95 s |",
            "|---|---|---|---|---|",
        ]
        for name, entry in wer["models"].items():
            lines.append(
                f"| {name} | {entry['clean']:.1%} | {entry['phone']:.1%} | {entry['phone_noisy']:.1%} | "
                f"{entry['latency_s']['p50']} / {entry['latency_s']['p95']} |"
            )
    calls = summary["calls"]
    lines += [
        "",
        "## API calls",
        "",
        f"{calls.get('calls', 0)} real requests; all requested ids `:free`: {calls.get('all_requested_free')}; "
        f"all served ids `:free`: {calls.get('all_served_free')}. "
        f"By tag {calls.get('by_tag')}, by status {calls.get('by_status')}.",
    ]
    return "\n".join(lines) + "\n"
