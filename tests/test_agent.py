"""Tools and the rules around them: offered-slots-only booking, the read-back/yes gate, escalation, FAQ grounding,
sentence-chunked streaming, the medical-advice filter, privacy masking."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from callie.agent.agent import Agent, CallAction, Sentence, ToolEvent, split_sentences
from callie.agent.grounding import extract_times, unsupported_times
from callie.agent.policy import Escalation, EscalationState, Reply, analyze_turn, classify_reply, is_backchannel
from callie.agent.tools import ToolBox
from callie.kb.retriever import KnowledgeBase
from callie.llm.base import Completed, JsonDict, LLMEvent, TextDelta, ToolCall
from callie.privacy import Masker, digits_from_speech, mask_name, mask_phone, normalize_phone
from callie.runtime import Runtime, build_runtime
from callie.tts.chunker import SentenceChunker, clean_for_speech
from tests.helpers import fake_settings


@pytest.fixture
def runtime() -> Runtime:
    return build_runtime(fake_settings())


def toolbox(runtime: Runtime, caller_phone: str | None = None) -> ToolBox:
    return ToolBox(runtime.new_context(caller_phone=caller_phone))


def offer(box: ToolBox, when: str = "next Tuesday after lunch") -> dict[str, Any]:
    box.ctx.turn += 1
    result = box.execute("check_availability", {"service": "cleaning", "when": when})
    assert result.data["status"] == "ok", result.data
    return result.data


class TestBookingGate:
    def test_read_back_then_yes_then_booked(self, runtime: Runtime) -> None:
        box = toolbox(runtime)
        slot = offer(box)["slots"][0]["slot_id"]
        args = {"service": "cleaning", "slot_id": slot, "patient_name": "jane doe", "phone": "555 123 4567"}
        box.ctx.turn += 1
        first = box.execute("book_appointment", {**args, "confirmed": False})
        assert first.data["status"] == "needs_confirmation"
        assert (
            "Jane Doe" in (first.say or "") and "4 5 6 7" in (first.say or "") and "Is that right?" in (first.say or "")
        )
        box.ctx.turn += 1
        box.register_reply(affirmed=True, declined=False)
        done = box.execute("book_appointment", {**args, "confirmed": True})
        assert done.data["status"] == "booked" and box.ctx.outcome.booked == [done.data["appointment_id"]]
        assert runtime.calendar.get(done.data["appointment_id"]) is not None

    def test_confirmed_true_without_a_yes_is_refused(self, runtime: Runtime) -> None:
        box = toolbox(runtime)
        slot = offer(box)["slots"][0]["slot_id"]
        args = {
            "service": "cleaning",
            "slot_id": slot,
            "patient_name": "Jane Doe",
            "phone": "5551234567",
            "confirmed": True,
        }
        result = box.execute("book_appointment", args)  # skipped the read-back entirely
        assert result.data["status"] == "needs_confirmation" and box.ctx.outcome.booked == []
        box.ctx.turn += 1
        box.register_reply(affirmed=False, declined=False)  # caller said something else
        assert box.execute("book_appointment", args).data["status"] == "needs_confirmation"
        assert box.ctx.outcome.blocked_unconfirmed == 2

    def test_yes_must_come_right_after_the_read_back(self, runtime: Runtime) -> None:
        box = toolbox(runtime)
        slot = offer(box)["slots"][0]["slot_id"]
        args = {"service": "cleaning", "slot_id": slot, "patient_name": "Jane Doe", "phone": "5551234567"}
        box.ctx.turn += 1
        box.execute("book_appointment", {**args, "confirmed": False})
        box.ctx.turn += 2  # a turn in between
        box.register_reply(affirmed=True, declined=False)
        assert box.execute("book_appointment", {**args, "confirmed": True}).data["status"] == "needs_confirmation"

    def test_changed_details_need_a_new_read_back(self, runtime: Runtime) -> None:
        box = toolbox(runtime)
        slots = offer(box)["slots"]
        args = {
            "service": "cleaning",
            "slot_id": slots[0]["slot_id"],
            "patient_name": "Jane Doe",
            "phone": "5551234567",
        }
        box.ctx.turn += 1
        box.execute("book_appointment", {**args, "confirmed": False})
        box.ctx.turn += 1
        box.register_reply(affirmed=True, declined=False)
        changed = box.execute("book_appointment", {**args, "slot_id": slots[1]["slot_id"], "confirmed": True})
        assert changed.data["status"] == "needs_confirmation"

    def test_no_clears_the_pending_action(self, runtime: Runtime) -> None:
        box = toolbox(runtime)
        slot = offer(box)["slots"][0]["slot_id"]
        box.ctx.turn += 1
        box.execute(
            "book_appointment",
            {
                "service": "cleaning",
                "slot_id": slot,
                "patient_name": "Jane Doe",
                "phone": "5551234567",
                "confirmed": False,
            },
        )
        box.ctx.turn += 1
        box.register_reply(affirmed=False, declined=True)
        assert box.ctx.pending is None

    def test_only_offered_slots_can_be_booked(self, runtime: Runtime) -> None:
        box = toolbox(runtime)
        result = box.execute(
            "book_appointment",
            {
                "service": "cleaning",
                "slot_id": "2026-10-13T15:00",
                "patient_name": "Jane Doe",
                "phone": "5551234567",
                "confirmed": False,
            },
        )
        assert result.data["status"] == "error" and "not offered" in result.data["error"]
        assert box.ctx.outcome.blocked_unoffered == 1
        offer(box)
        wrong_service = box.execute(
            "book_appointment",
            {
                "service": "filling",
                "slot_id": next(iter(box.ctx.offered)),
                "patient_name": "Jane Doe",
                "confirmed": False,
            },
        )
        assert wrong_service.data["status"] == "error"

    def test_caller_id_supplies_the_phone(self, runtime: Runtime) -> None:
        box = toolbox(runtime, caller_phone="+15558821190")
        slot = offer(box)["slots"][0]["slot_id"]
        result = box.execute(
            "book_appointment", {"service": "cleaning", "slot_id": slot, "patient_name": "Ann Lee", "confirmed": False}
        )
        assert "1 1 9 0" in (result.say or "")

    def test_missing_phone_is_asked_for(self, runtime: Runtime) -> None:
        box = toolbox(runtime)
        slot = offer(box)["slots"][0]["slot_id"]
        result = box.execute(
            "book_appointment", {"service": "cleaning", "slot_id": slot, "patient_name": "Ann Lee", "confirmed": False}
        )
        assert result.data["status"] == "need_phone"


class TestOtherTools:
    def test_availability_reports_interpretation_and_alternatives(self, runtime: Runtime) -> None:
        box = toolbox(runtime)
        data = offer(box, "next Tuesday after lunch")
        assert data["understood_as"] == "Tuesday, October 13th, 1 PM to 6 PM"
        sunday = box.execute("check_availability", {"service": "cleaning", "when": "Sunday"})
        assert sunday.data["status"] == "no_slots_in_window" and sunday.data["alternatives"]
        assert "next openings" in (sunday.say or "")
        unclear = box.execute("check_availability", {"service": "cleaning", "when": "whenever"})
        assert unclear.data["status"] == "need_when"

    def test_exact_time_available(self, runtime: Runtime) -> None:
        result = toolbox(runtime).execute("check_availability", {"service": "filling", "when": "Thursday at 2:30 pm"})
        assert result.data.get("exact_match") == "2026-10-08T14:30" and "is open" in (result.say or "")

    def test_reschedule_and_cancel_with_read_back(self, runtime: Runtime) -> None:
        cleaning = runtime.clinic.service("cleaning")
        assert cleaning is not None
        from datetime import datetime

        from tests.conftest import NY

        existing = runtime.calendar.book(cleaning, datetime(2026, 10, 8, 10, tzinfo=NY), "Sofia Rossi", "+15553492216")
        box = toolbox(runtime)
        box.ctx.turn = 1
        found = box.execute(
            "find_appointment", {"patient_name": "Sophia Rossi", "purpose": "reschedule"}
        )  # STT spelling
        assert found.data["status"] == "found"
        box.ctx.turn = 2
        offered = box.execute(
            "check_availability", {"service": "cleaning", "when": "Friday morning", "appointment_id": existing.id}
        )
        slot = offered.data["slots"][0]["slot_id"]
        box.ctx.turn = 3
        read_back = box.execute(
            "reschedule_appointment", {"appointment_id": existing.id, "slot_id": slot, "confirmed": False}
        )
        assert "from Thursday, October 8th at 10 AM" in (read_back.say or "")
        box.ctx.turn = 4
        box.register_reply(affirmed=True, declined=False)
        moved = box.execute(
            "reschedule_appointment", {"appointment_id": existing.id, "slot_id": slot, "confirmed": True}
        )
        assert moved.data["status"] == "rescheduled"
        box.ctx.turn = 5
        cancel = box.execute("find_appointment", {"patient_name": "Rossi", "purpose": "cancel"})
        assert "cancel it?" in (cancel.say or "") and box.ctx.pending is not None
        box.ctx.turn = 6
        box.register_reply(affirmed=True, declined=False)
        done = box.execute("cancel_appointment", {"appointment_id": existing.id, "confirmed": True})
        assert done.data["status"] == "cancelled"
        row = runtime.calendar.get(existing.id)
        assert row is not None and row.status == "cancelled"

    def test_faq_grounded_or_no_answer(self, runtime: Runtime) -> None:
        box = toolbox(runtime)
        answer = box.execute("answer_faq", {"question": "Do you take Delta Dental?"})
        assert answer.data["passages"][0]["title"] == "Insurance" and answer.say is None
        unknown = box.execute("answer_faq", {"question": "What's the meaning of life?"})
        assert unknown.data["status"] == "no_answer" and "take a message" in (unknown.say or "")
        hours = box.execute("answer_faq", {"question": "What are your opening hours?"})
        assert hours.data["passages"][0]["title"] == "Opening hours" and "5 PM" in box.ctx.known_times

    def test_transfer_open_vs_closed(self, runtime: Runtime) -> None:
        box = toolbox(runtime)
        assert box.execute("transfer_to_human", {"reason": "caller asked"}).action == "transfer"
        evening = build_runtime(fake_settings(now="2026-10-06T20:15"))
        closed = toolbox(evening).execute("transfer_to_human", {"reason": "caller asked"})
        assert closed.action is None and "closed" in (closed.say or "")
        emergency = toolbox(evening).execute("transfer_to_human", {"reason": "emergency"})
        assert emergency.action == "transfer" and "911" in (emergency.say or "")

    def test_message_and_sms_outbox_are_masked(self, runtime: Runtime) -> None:
        box = toolbox(runtime)
        result = box.execute(
            "take_message",
            {"caller_name": "peter novak", "message": "Call me about my bill, 555 812 0965", "phone": "555-812-0965"},
        )
        assert result.data["status"] == "message_taken" and "0 9 6 5" in (result.say or "")
        logged = box.ctx.tool_log[-1]
        assert logged["arguments"]["phone"] == "***-***-0965" and "812" not in str(logged)

    def test_unknown_tool_and_bad_arguments(self, runtime: Runtime) -> None:
        box = toolbox(runtime)
        assert box.execute("find_availability", {}).data["status"] == "error"
        assert box.execute("check_availability", {"_invalid_json": "{"}).data["status"] == "error"


class TestPolicies:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Yes", Reply.YES),
            ("yeah that's right", Reply.YES),
            ("Correct.", Reply.YES),
            ("um, yes please", Reply.YES),
            ("Sounds good, thanks!", Reply.YES),
            ("That sounds perfect.", Reply.YES),
            ("Yes, but can we make it 3 instead?", Reply.YES_PLUS),
            ("No, I said Thursday", Reply.NO),
            ("Actually, wait", Reply.NO),
            ("What about Friday?", Reply.OTHER),
            ("", Reply.OTHER),
        ],
    )
    def test_reply_classification(self, text: str, expected: Reply) -> None:
        assert classify_reply(text) == expected

    @pytest.mark.parametrize("text", ["mm-hmm", "Uh-huh.", "okay", "Right.", "yeah", "M-hm", "and mhm", ""])
    def test_backchannels(self, text: str) -> None:
        assert is_backchannel(text)

    @pytest.mark.parametrize("text", ["Wait, stop", "Sorry, can I ask something?", "No, Thursday", "I want to cancel"])
    def test_not_backchannels(self, text: str) -> None:
        assert not is_backchannel(text)

    @pytest.mark.parametrize(
        "text",
        [
            "My face is swollen and it's hard to swallow",
            "I can't breathe properly",
            "the bleeding won't stop",
            "I think I broke my jaw",
            "This is an emergency",
        ],
    )
    def test_emergencies(self, text: str) -> None:
        assert analyze_turn(text).escalation is Escalation.EMERGENCY

    def test_emergency_word_alone_is_not_an_emergency(self) -> None:
        assert analyze_turn("Do you take emergency appointments?").escalation is None

    def test_human_request_and_anger(self) -> None:
        assert analyze_turn("Can I talk to a real person?").escalation is Escalation.HUMAN_REQUESTED
        state = EscalationState()
        assert state.update(analyze_turn("This is ridiculous, I'm so frustrated")) is Escalation.ANGRY

    def test_repeated_misunderstanding(self) -> None:
        state = EscalationState()
        assert state.update(analyze_turn("What? I don't understand")) is None
        assert state.update(analyze_turn("That's not what I said")) is Escalation.MISUNDERSTANDING

    def test_goodbyes(self) -> None:
        for text in ["No, that's all, thanks", "Thanks, bye!", "Thank you so much. Have a great day!", "I'm all set"]:
            assert analyze_turn(text).goodbye, text
        assert not analyze_turn("Is that all you have?").goodbye


class ScriptedLLM:
    """A model that replays fixed events, to test the agent's streaming and safety handling."""

    def __init__(self, turns: list[list[LLMEvent]]) -> None:
        self.turns = turns

    @property
    def label(self) -> str:
        return "test/scripted"

    @property
    def is_remote(self) -> bool:
        return False

    async def stream(
        self, messages: list[JsonDict], tools: list[JsonDict], *, max_tokens: int, temperature: float
    ) -> AsyncIterator[LLMEvent]:
        for event in self.turns.pop(0):
            yield event


async def collect(agent: Agent, text: str) -> list[Any]:
    return [event async for event in agent.respond(text)]


class TestAgentStreaming:
    async def test_sentences_are_released_as_they_complete(self, runtime: Runtime) -> None:
        tokens = ["Sure", ", we're ", "open on ", "Saturday mornings. ", "Anything ", "else?"]
        agent = runtime.new_agent(
            runtime.new_context(), ScriptedLLM([[*(TextDelta(t) for t in tokens), Completed("stop")]])
        )
        events = await collect(agent, "Are you open on Saturday?")
        assert [e.text for e in events if isinstance(e, Sentence)] == [
            "Sure, we're open on Saturday mornings.",
            "Anything else?",
        ]

    async def test_medical_advice_is_replaced(self, runtime: Runtime) -> None:
        llm = ScriptedLLM(
            [
                [
                    TextDelta("You should take 400 mg of ibuprofen. "),
                    TextDelta("Want to book a visit?"),
                    Completed("stop"),
                ]
            ]
        )
        agent = runtime.new_agent(runtime.new_context(), llm)
        sentences = [e.text for e in await collect(agent, "Should I take ibuprofen?") if isinstance(e, Sentence)]
        assert "ibuprofen" not in " ".join(sentences) and "not able to give medical advice" in sentences[0]
        assert agent.last_stats.replaced_advice == 1

    async def test_ungrounded_times_are_counted(self, runtime: Runtime) -> None:
        llm = ScriptedLLM([[TextDelta("I have an opening at 4:30 PM tomorrow."), Completed("stop")]])
        agent = runtime.new_agent(runtime.new_context(), llm)
        await collect(agent, "Anything tomorrow?")
        assert agent.last_stats.unsupported_times == ["4:30 PM"]

    async def test_tool_say_ends_the_turn_and_history_is_consistent(self, runtime: Runtime) -> None:
        call = ToolCall("c1", "check_availability", {"service": "cleaning", "when": "next Tuesday after lunch"})
        agent = runtime.new_agent(
            runtime.new_context(), ScriptedLLM([[TextDelta("Let me check. "), call, Completed("tool_calls")]])
        )
        events = await collect(agent, "Cleaning next Tuesday after lunch?")
        sentences = [e for e in events if isinstance(e, Sentence)]
        assert sentences[0].text == "Let me check." and sentences[1].source == "tool"
        roles = [m["role"] for m in agent.messages]
        assert roles[-3:] == ["assistant", "tool", "assistant"] and agent.messages[-3]["tool_calls"][0]["id"] == "c1"

    async def test_end_call_and_transfer_actions(self, runtime: Runtime) -> None:
        agent = runtime.new_agent(runtime.new_context(), ScriptedLLM([]))
        events = await collect(agent, "No, that's all, thanks. Bye!")
        assert isinstance(events[-1], CallAction) and events[-1].kind == "hangup"
        assert any(isinstance(e, ToolEvent) and e.result.name == "end_call" for e in events)

    async def test_interrupted_answer_keeps_only_what_was_heard(self, runtime: Runtime) -> None:
        llm = ScriptedLLM(
            [[TextDelta("Our office is at 418 Linden Street. Parking is behind the building."), Completed("stop")]]
        )
        agent = runtime.new_agent(runtime.new_context(), llm)
        await collect(agent, "Where are you?")
        agent.note_interrupted("Our office is at 418")
        assert agent.messages[-1]["content"] == "Our office is at 418 [interrupted by the caller]"


class TestChunking:
    def test_split_and_clean(self) -> None:
        assert split_sentences("Just to confirm: a cleaning on Tuesday. Is that right?") == [
            "Just to confirm:",
            "a cleaning on Tuesday.",
            "Is that right?",
        ]
        assert clean_for_speech("**Hours:** 8 AM – 5 PM 😀") == "Hours: 8 AM to 5 PM "

    @pytest.mark.parametrize("step", [1, 3, 7, 50])
    def test_token_size_does_not_change_the_chunks(self, step: int) -> None:
        text = "Dr. Patel is in at 1:30 p.m. today. It costs $1.50, or so! Bye."
        chunker = SentenceChunker()
        chunks = [c for i in range(0, len(text), step) for c in chunker.push(text[i : i + step])] + chunker.flush()
        assert chunks == ["Dr. Patel is in at 1:30 p.m. today.", "It costs $1.50, or so!", "Bye."]


class TestPrivacyAndGrounding:
    def test_masking(self) -> None:
        masker = Masker()
        masker.register_name("Maria Gonzalez")
        text = "Maria Gonzalez, 555-214-8839, maria@example.com, five five five two one four eight eight three nine"
        masked = masker.mask(text)
        assert "Maria" not in masked and "214" not in masked and "@" not in masked and "8839" in masked
        assert masker.mask("My name is Tom Becker") == "My name is T** B*****"
        assert mask_phone("+15552148839") == "***-***-8839" and mask_name("Jo Li") == "J** L**"

    def test_phone_normalization(self) -> None:
        assert digits_from_speech("five five five, double two, one oh nine") == "55522109"
        assert normalize_phone("(555) 214-8839") == "+15552148839"
        assert normalize_phone("1 555 214 8839") == "+15552148839"
        assert normalize_phone("12") is None

    def test_time_grounding(self) -> None:
        assert extract_times("Open 8 AM to noon, and at 1:30 PM") == {"8 AM", "noon", "1:30 PM"}
        assert unsupported_times("I can book you at 3 PM.", {"3 PM"}) == []
        assert unsupported_times("We close at 5 PM.", set()) == []  # not an availability offer


def test_knowledge_base_retrieval() -> None:
    from callie.clinic import load_clinic

    kb = KnowledgeBase.from_markdown(load_clinic().knowledge_text())
    cases = {
        "do you take delta dental": "insurance",
        "where can I park": "parking",
        "are you open on sunday": "opening-hours",
        "how much does whitening cost": "teeth-whitening",
        "do you speak spanish": "accessibility-and-languages",
        "is there a fee if I cancel": "cancellation-policy",
        "do you do braces": "services",
    }
    for question, section in cases.items():
        assert kb.search(question)[0].passage.id == section, question
    assert kb.search("what is the meaning of life") == []


def test_unclear_phone_is_re_asked_then_booking_proceeds_without_text(runtime: Runtime) -> None:
    box = toolbox(runtime)
    slot = offer(box)["slots"][0]["slot_id"]
    args = {
        "service": "cleaning",
        "slot_id": slot,
        "patient_name": "Maria Gonzalez",
        "phone": "5. 155. 2. 114,8,8. 139.",
    }
    first = box.execute("book_appointment", {**args, "confirmed": False})
    assert first.data["status"] == "phone_unclear" and "13 digits" in (first.say or "")
    second = box.execute("book_appointment", {**args, "confirmed": False})
    assert second.data["status"] == "needs_confirmation" and "won't be able to text" in (second.say or "")


def test_non_names_are_refused_and_never_masked(runtime: Runtime) -> None:
    box = toolbox(runtime)
    slot = offer(box)["slots"][0]["slot_id"]
    for bogus in ["Tuesday, October 6th at 1 p.m.", "me", "5551234567"]:
        result = box.execute("book_appointment", {"service": "cleaning", "slot_id": slot, "patient_name": bogus,
                                                  "phone": "5551234567", "confirmed": False})  # fmt: skip
        assert result.data["status"] == "need_name", bogus
    assert box.ctx.masker.mask("Let me check Tuesday at 1 PM") == "Let me check Tuesday at 1 PM"
