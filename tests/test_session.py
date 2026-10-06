"""The live call session with fake components: audio in, turns, tools, barge-in, backchannels, silence prompts."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from callie.pipeline.session import CallSession, SessionConfig
from callie.runtime import build_runtime
from callie.scheduling.db import Appointment, CallRecord
from callie.speech import SpeechStack
from callie.stt import FakeSTT
from callie.tts import FakeTTS
from callie.vad import EnergyVAD
from tests.helpers import (
    RecordingTransport,
    fake_settings,
    make_session,
    quiet,
    say,
    speech_like,
    stream,
    until_agent_done,
    wait_for,
)


async def test_booking_call_end_to_end_over_audio() -> None:
    session, transport, stt = make_session(
        [
            "Hi, I'd like to book a cleaning next Tuesday after lunch",
            "The 3 PM one please",
            "My name is Jane Doe and my number is 555 123 4567",
            "Yes, that's right",
            "No, that's all, thanks",
        ]
    )
    await session.start()
    await until_agent_done(session)
    for _ in range(5):
        await say(session)
        await until_agent_done(session)
        if session.closed.is_set():
            break
    await wait_for(session.closed.is_set, limit=5)
    assert stt.calls == 5
    with session.runtime.sessions() as db:
        booked = db.scalars(select(Appointment).where(Appointment.source == "call")).all()
        record = db.get(CallRecord, session.call_id)
    assert len(booked) == 1 and booked[0].patient_name == "Jane Doe"
    assert booked[0].start.astimezone(session.ctx.clinic.tz).hour == 15
    assert transport.hung_up and session.end_reason == "agent_hung_up"
    # The read-back was spoken before the booking, and the yes executed it without another model call.
    names = [t["name"] for t in session.ctx.tool_log]
    assert names == ["check_availability", "book_appointment", "book_appointment", "end_call"]
    assert session.ctx.tool_log[1]["result"]["status"] == "needs_confirmation"
    assert session.turns[3].path == "fast_confirm"
    # Persisted call record: masked transcript, latency waterfall, recording.
    assert record is not None and record.outcome == "booked" and record.recording_path
    text = " ".join(e["text"] for e in record.transcript)
    assert "Jane" not in text and "4567" in text and "555 123" not in text
    waterfall = record.turns[0]
    for key in ("endpointing_ms", "stt_ms", "tts_first_audio_ms", "voice_to_voice_ms"):
        assert waterfall[key] is not None and waterfall[key] >= 0, key
    assert 300 <= waterfall["endpointing_ms"] <= 700  # 400 ms end-of-turn silence + frame granularity
    assert transport.of("metrics") and transport.of("tool")


async def test_greeting_discloses_ai_and_recording() -> None:
    session, transport, _ = make_session([])
    await session.start()
    first = transport.of("agent_text")[0]["text"]
    assert "AI assistant" in first and "recorded" in first
    await session.close()


async def test_barge_in_drops_the_rest_of_the_answer() -> None:
    session, transport, _stt = make_session(
        ["What's your cancellation policy?", "Sorry, actually I want to book a filling"], ms_per_char=40.0
    )
    await session.start()
    await until_agent_done(session, limit=30)
    await say(session)
    await wait_for(lambda: session.player.active and session.player.current_turn == 1, limit=5)
    await stream(session, quiet(0.6))  # let Callie talk for a moment
    played_before = transport.audio_seconds
    await stream(session, speech_like(1.6, seed=3))  # long enough for a hard interrupt
    await stream(session, quiet(0.6))
    await wait_for(lambda: session.turns[-1].turn == 2, limit=5)
    assert transport.clears >= 1
    barge = session.stats.barge_ins[0]
    assert barge["result"] == "interrupted"
    assert 200 <= barge["reaction_ms"] <= 600  # 250 ms of speech + VAD framing
    heard = next(e for e in session.transcript if e["role"] == "agent" and e["turn"] == 1)
    assert heard.get("interrupted")
    full_answer_chars = len(session.player.history[-1].text) if session.player.history else 0
    assert full_answer_chars >= 0 and transport.audio_seconds - played_before < 3.0
    assert any("[interrupted by the caller]" in str(m.get("content")) for m in session.agent.messages)
    await until_agent_done(session, limit=30)
    await session.close()


async def test_backchannel_does_not_interrupt() -> None:
    session, _transport, _stt = make_session(["Do you take Delta Dental?", "mm-hmm"], ms_per_char=30.0)
    await session.start()
    await until_agent_done(session, limit=30)
    await say(session)
    await wait_for(lambda: session.player.active and session.player.current_turn == 1, limit=5)
    await stream(session, quiet(0.4))
    await stream(session, speech_like(0.4, seed=5))  # a short "mm-hmm"
    await stream(session, quiet(0.6))
    await until_agent_done(session, limit=30)
    assert session.stats.backchannels == 1
    assert session.stats.barge_ins[0]["result"] == "backchannel_resumed"
    answer = [u for u in session.player.history if u.turn == 1]
    assert answer and all(u.heard_text() == u.text for u in answer)  # the whole answer was heard
    assert len(session.turns) == 1  # "mm-hmm" did not become a turn
    await session.close()


async def test_silence_prompt_then_polite_hangup() -> None:
    session, transport, _ = make_session(
        [], config=SessionConfig(silence_prompt_s=0.6, save_record=False, record=False)
    )
    await session.start()
    for _ in range(80):
        if session.closed.is_set():
            break
        await stream(session, quiet(0.1))
    assert session.closed.is_set()
    lines = [e["text"] for e in transport.of("agent_text")]
    assert "Are you still there?" in lines
    assert any("Goodbye" in line for line in lines)
    assert transport.hung_up and session.end_reason == "silence_timeout"


async def test_typed_input_and_emergency_transfer() -> None:
    session, transport, _ = make_session([])
    await session.start()
    await session.feed_text("My jaw is swollen and I have trouble breathing")
    await wait_for(session.closed.is_set, limit=10)
    assert transport.transferred is not None
    assert session.ctx.outcome.label() == "transferred"
    spoken = " ".join(e["text"] for e in transport.of("agent_text"))
    assert "911" in spoken


async def test_speaking_again_while_thinking_merges_the_turns() -> None:
    from callie.clinic import load_clinic
    from callie.llm.fake import FakeReceptionist

    slow = FakeReceptionist(load_clinic(), delay_s=1.5)
    session, _transport, _stt = make_session(["I'd like to book a cleaning", "next Thursday morning"], llm=slow)
    await session.start()
    await until_agent_done(session, limit=30)
    await say(session, trailing_silence=0.5)
    await say(session, trailing_silence=0.5)  # while the first answer is still being generated
    await until_agent_done(session, limit=30)
    callers = [e for e in session.transcript if e["role"] == "caller"]
    assert len(callers) == 1 and "Thursday" in callers[0]["text"]
    assert session.ctx.tool_log and session.ctx.tool_log[0]["name"] == "check_availability"
    await session.close()


def test_settings_drive_session_config() -> None:
    settings = fake_settings(barge_in_min_speech_ms=300, hard_interrupt_ms=900, silence_prompt_seconds=5)
    runtime = build_runtime(settings)
    speech = SpeechStack(stt=FakeSTT(), tts=FakeTTS(), vad_factory=EnergyVAD, settings=settings)
    session = CallSession(runtime, speech, RecordingTransport())
    assert session.config.barge_in_min_speech_s == pytest.approx(0.3)
    assert session.config.hard_interrupt_s == pytest.approx(0.9)
    assert session.config.silence_prompt_s == 5
