"""Metrics (WER, percentiles), scenario file, deterministic checks, report rendering, and CLI smoke runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from callie.cli import app
from callie.eval.metrics import corpus_wer, normalize_for_wer, percentile, summarize, wer, wer_counts
from callie.eval.report import ledger_summary, render_markdown
from callie.eval.scenarios import load_scenarios


class TestWer:
    def test_basic_counts(self) -> None:
        counts = wer_counts("book a cleaning on tuesday", "book cleaning on a thursday")
        assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 1, 1)
        assert counts.reference_words == 5 and counts.wer == pytest.approx(0.6)

    def test_normalization_makes_spoken_forms_equal(self) -> None:
        assert wer("Three thirty P.M. on October twentieth", "3:30 pm on October 20th") == 0.0
        assert wer("Mm-hmm, okay.", "mhm OK") == 0.0
        assert wer("It's $120.", "it's 120 dollars") == 0.0
        assert normalize_for_wer("Twenty five") == ["25"]

    def test_edge_cases(self) -> None:
        assert wer("", "") == 0.0 and wer("", "noise") == 1.0 and wer("hello", "") == 1.0
        assert corpus_wer([("a b c d", "a b c d"), ("e f", "x f")]) == pytest.approx(1 / 6)

    def test_percentiles(self) -> None:
        assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([], 50) is None
        assert summarize([1.0, 2.0, 3.0])["p95"] == pytest.approx(2.9)


def test_scenarios_are_valid_and_cover_the_categories() -> None:
    scenarios = load_scenarios()
    assert 25 <= len(scenarios) <= 40
    categories = {s.category for s in scenarios}
    assert {"booking", "reschedule", "cancel", "faq", "escalation", "interruption", "change_of_mind"} <= categories
    assert {s.channel for s in scenarios} == {"clean", "phone", "phone_noisy"}
    for s in scenarios:
        assert s.silent or s.opening, s.id
        if s.expect.outcome in {"rescheduled", "cancelled"}:
            assert s.setup, s.id


def test_report_renders(tmp_path: Path) -> None:
    ledger = tmp_path / "calls.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "ts": "t",
                "tag": "agent",
                "requested_model": "a/b:free",
                "served_model": "a/b:free",
                "status": "ok",
                "ttft_s": 1.2,
            }
        )
        + "\n"
    )
    summary = {"calls": ledger_summary(ledger)}
    assert summary["calls"]["all_requested_free"] and summary["calls"]["calls"] == 1
    assert "1 real requests" in render_markdown(summary)


runner = CliRunner()


def test_cli_text_call_books_with_the_fake_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CALLIE_DATABASE_URL", f"sqlite:///{tmp_path / 'c.db'}")
    monkeypatch.setenv("CALLIE_NOW", "2026-10-06T09:30")
    monkeypatch.setenv("CALLIE_LLM_PROVIDER", "fake")
    from callie.config import get_settings

    get_settings.cache_clear()
    result = runner.invoke(
        app,
        [
            "call",
            "I'd like a cleaning next Tuesday after lunch",
            "The 3 PM one",
            "My name is Jane Doe, 555 123 4567",
            "Yes",
            "No, that's all, thanks",
        ],
    )
    get_settings.cache_clear()
    assert result.exit_code == 0, result.output
    assert "Just to confirm" in result.output and "outcome: booked" in result.output


def test_cli_lists_scenarios_and_help() -> None:
    assert runner.invoke(app, ["--help"]).exit_code == 0
    listing = runner.invoke(app, ["eval", "scenarios"])
    assert listing.exit_code == 0 and "book_basic" in listing.output


def test_import_calls_copies_records_and_bookings(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from callie.eval.importer import import_calls
    from callie.scheduling.db import Appointment, CallRecord, make_session_factory

    source = tmp_path / "eval"
    source.mkdir()
    origin = make_session_factory(f"sqlite:///{source / 'a.db'}")
    start = datetime(2026, 10, 13, 18, 30, tzinfo=UTC)
    with origin() as db, db.begin():
        db.add(
            CallRecord(
                id="sim_a", transport="simulated", outcome="booked", transcript=[], tool_calls=[], turns=[], stack={}
            )
        )
        db.add(
            Appointment(
                id="A-1",
                service_id="cleaning",
                resource_id="hygienist",
                start=start,
                end=start.replace(hour=19),
                patient_name="Jane Doe",
                phone=None,
                source="call",
            )
        )
    target = tmp_path / "dash.db"
    assert import_calls(source, target) == {"calls": 1, "bookings": 1, "skipped_overlapping_bookings": 0}
    assert import_calls(source, target)["calls"] == 0  # idempotent
