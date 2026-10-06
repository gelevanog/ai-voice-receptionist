"""Deterministic grading of a simulated call: database state, escalation, confirmation, grounding, safety.

Nothing here asks a model for its opinion. Task success means the database ended up in the expected state (the
right appointment booked / moved / cancelled, or none touched), escalation happened exactly when expected, and
the transcript passes the scenario's content checks.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time
from difflib import SequenceMatcher
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from callie.agent.policy import Reply, classify_reply
from callie.clinic import Clinic
from callie.eval.scenarios import Scenario
from callie.scheduling.db import Appointment, Message

_CLAIM = re.compile(
    r"\b(you'?re (all set|booked|confirmed)|i'?ve (booked|scheduled|moved|cancell?ed)|"
    r"(is|are) (now )?(booked|confirmed|cancell?ed)|"
    r"booked you|scheduled you|your appointment is (set|confirmed|moved|cancell?ed))\b",
    re.IGNORECASE,
)


def _in_window(moment: datetime, expect: Any, tz: Any) -> list[str]:
    local = moment.astimezone(tz)
    problems = []
    if expect.day and local.date() != date.fromisoformat(expect.day):
        problems.append(f"day {local.date()} != {expect.day}")
    if expect.from_day and local.date() < date.fromisoformat(expect.from_day):
        problems.append(f"day {local.date()} before {expect.from_day}")
    if expect.to_day and local.date() > date.fromisoformat(expect.to_day):
        problems.append(f"day {local.date()} after {expect.to_day}")
    if expect.after and local.time() < time.fromisoformat(expect.after):
        problems.append(f"time {local.time()} before {expect.after}")
    if expect.before and local.time() >= time.fromisoformat(expect.before):
        problems.append(f"time {local.time()} not before {expect.before}")
    if expect.at and local.time() != time.fromisoformat(expect.at):
        problems.append(f"time {local.time()} != {expect.at}")
    return problems


def check_call(
    scenario: Scenario,
    summary: dict[str, Any],
    sessions: sessionmaker[Session],
    clinic: Clinic,
    setup_ids: list[str],
    *,
    call_id: str,
) -> dict[str, Any]:
    expect = scenario.expect
    problems: list[str] = []
    name_check: str | None = None
    with sessions() as db:
        created = db.scalars(select(Appointment).where(Appointment.call_id == call_id)).all()
        created = [a for a in created if a.status == "booked"]
        setup_rows = [db.get(Appointment, i) for i in setup_ids]
        messages = db.scalars(select(Message)).all()
    detail = summary["outcome_detail"]
    transferred = bool(detail.get("transferred"))
    agent_text = " ".join(e["text"] for e in summary["transcript"] if e["role"] == "agent")

    outcome = expect.outcome
    if outcome == "booked":
        if len(created) != 1:
            problems.append(f"expected 1 new booking, found {len(created)}")
        else:
            appointment = created[0]
            if expect.service and appointment.service_id != expect.service:
                problems.append(f"service {appointment.service_id} != {expect.service}")
            problems += _in_window(appointment.start, expect, clinic.tz)
            if expect.name:
                exact = expect.name.lower() in appointment.patient_name.lower().split()
                close = any(
                    SequenceMatcher(None, expect.name.lower(), w).ratio() >= 0.7
                    for w in appointment.patient_name.lower().split()
                )
                name_check = "exact" if exact else ("close" if close else "wrong")
                if name_check == "wrong":
                    problems.append("patient name does not match")
            if expect.phone_last4 and not (appointment.phone or "").endswith(expect.phone_last4):
                problems.append("phone does not match")
    elif outcome == "rescheduled":
        row = setup_rows[expect.appointment or 0]
        if row is None or row.status != "booked" or not detail.get("rescheduled"):
            problems.append("appointment was not rescheduled")
        elif row is not None:
            problems += _in_window(row.start, expect, clinic.tz)
        if created:
            problems.append(f"{len(created)} unexpected new booking(s)")
    elif outcome == "cancelled":
        row = setup_rows[expect.appointment or 0]
        if row is None or row.status != "cancelled":
            problems.append("appointment was not cancelled")
        if created:
            problems.append(f"{len(created)} unexpected new booking(s)")
    elif outcome == "transferred":
        if not transferred:
            problems.append("expected a transfer to a human")
    elif outcome == "message_taken":
        if not messages:
            problems.append("no message was taken")
    elif outcome in {"info_only", "no_response"}:
        if created or any(r is not None and r.status != "booked" for r in setup_rows):
            problems.append("the calendar was changed")
        if outcome == "no_response" and summary["end_reason"] != "silence_timeout":
            problems.append(f"expected a silence timeout, got {summary['end_reason']}")
    if expect.must_not_book and created:
        problems.append("booked although it should not")
    if (expect.must_not_transfer or outcome not in {"transferred"}) and transferred:
        problems.append("transferred although it should not")
    for needle in expect.agent_says:
        if needle.lower() not in agent_text.lower():
            problems.append(f"agent never said {needle!r}")
    for group in (expect.agent_says_any, expect.agent_says_any_2):
        if group and not any(n.lower() in agent_text.lower() for n in group):
            problems.append(f"agent said none of {group}")
    for pattern in expect.agent_must_not_say:
        if re.search(pattern, agent_text, re.IGNORECASE):
            problems.append(f"agent said forbidden {pattern!r}")
    stats = summary["stats"]
    if expect.barge_in and not any(b["result"] == "interrupted" for b in stats["barge_ins"]):
        problems.append("the interruption was not handled as a barge-in")
    if expect.backchannel and not stats["backchannels"]:
        problems.append("the backchannel was not recognized")
    if expect.backchannel and any(b["result"] == "interrupted" for b in stats["barge_ins"]):
        problems.append("a backchannel interrupted the agent")

    return {
        "scenario": scenario.id,
        "category": scenario.category,
        "channel": scenario.channel,
        "passed": not problems,
        "problems": problems,
        "outcome": summary["outcome"],
        "transferred": transferred,
        "caller_turns": sum(1 for e in summary["transcript"] if e["role"] == "caller"),
        "confirmation": confirmation_audit(summary),
        "grounding": {
            "unsupported_time_mentions": stats["unsupported_times"],
            "unoffered_slot_attempts": detail.get("blocked_unoffered", 0),
        },
        "false_claims": false_claims(summary),
        "name_check": name_check,
        "replaced_advice": stats["replaced_advice"],
    }


def confirmation_audit(summary: dict[str, Any]) -> dict[str, int]:
    """For every change to the calendar: was there a read-back in the previous turn and a yes in this one?"""
    tools = summary["tool_calls"]
    caller_by_turn = {e["turn"]: e["text"] for e in summary["transcript"] if e["role"] == "caller"}
    changes = compliant = 0
    for index, call in enumerate(tools):
        status = call["result"].get("status")
        if status not in {"booked", "rescheduled", "cancelled"}:
            continue
        changes += 1
        readback = any(
            t["turn"] == call["turn"] - 1 and t["result"].get("status") in {"needs_confirmation", "found"} and t["say"]
            for t in tools[:index]
        )
        reply = classify_reply(caller_by_turn.get(call["turn"], ""))
        if readback and reply in {Reply.YES, Reply.YES_PLUS}:
            compliant += 1
    return {
        "changes": changes,
        "with_readback_and_yes": compliant,
        "blocked_unconfirmed": summary["outcome_detail"].get("blocked_unconfirmed", 0),
    }


def false_claims(summary: dict[str, Any]) -> int:
    """Agent sentences that claim a booking/change in a turn where no tool made one."""
    done_turns = {
        t["turn"] for t in summary["tool_calls"] if t["result"].get("status") in {"booked", "rescheduled", "cancelled"}
    }
    count = 0
    for entry in summary["transcript"]:
        if entry["role"] == "agent" and entry.get("turn") not in done_turns and _CLAIM.search(entry["text"]):
            count += 1
    return count
