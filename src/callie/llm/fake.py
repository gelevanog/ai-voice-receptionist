"""A deterministic, rule-based stand-in for the LLM: same interface, same tools, no keys, no network.

It reads the conversation (user turns and tool results) and decides the next step with keyword rules, which is
enough for the tests, CI and a zero-key demo to exercise the real pipeline end to end: booking with read-back
and confirmation, rescheduling, cancelling, FAQ answers and messages. It is not a language model and makes no
claims about real-model quality; the evaluation uses real models.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

from callie.agent.grounding import extract_times
from callie.clinic import Clinic
from callie.llm.base import Completed, JsonDict, LLMEvent, TextDelta, ToolCall, message_text
from callie.privacy import digits_from_speech

_NAME = re.compile(
    r"\b(?:my name is|my name's|name is|this is|it's|i'm|i am|under|for)\s+([A-Za-z][a-z'’-]+(?:\s+[A-Za-z][a-z'’-]+)?)",
    re.IGNORECASE,
)
_NOT_NAME = {
    "calling", "looking", "wondering", "trying", "free", "fine", "good", "not", "a", "an", "the", "just", "sorry",
    "available", "interested", "new", "busy", "here", "okay", "sure", "me", "my", "next", "this", "that",
}  # fmt: skip
_DAY_WORDS = re.compile(
    r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|this week|weekend|"
    r"morning|afternoon|evening|asap|soon|earliest|\d{1,2}(?:st|nd|rd|th)|january|february|march|april|may|june|july|"
    r"august|september|october|november|december)\b"
)
_QUESTION = re.compile(
    r"\?|^(?:what|when|where|how|do you|does|are you|is there|is it|can i|can you|which|who)\b|\b(how much|"
    r"do you (?:take|accept|have|offer|do)|are you open|what are your|where are you|is there parking)\b"
)


def _tool_results(messages: list[JsonDict]) -> list[tuple[str, JsonDict, JsonDict, int]]:
    """(tool name, arguments, result, message index) for every tool call in the history, in order."""
    calls: dict[str, tuple[str, JsonDict]] = {}
    out = []
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                calls[call["id"]] = (function.get("name", ""), json.loads(function.get("arguments") or "{}"))
        elif message.get("role") == "tool":
            name, arguments = calls.get(message.get("tool_call_id", ""), ("", {}))
            try:
                result = json.loads(message_text(message))
            except json.JSONDecodeError:
                result = {}
            out.append((name, arguments, result, index))
    return out


def extract_name(text: str) -> str | None:
    for match in _NAME.finditer(text):
        words = [w for w in match.group(1).split() if w.lower() not in _NOT_NAME]
        if words and words[0].lower() == match.group(1).split()[0].lower():
            return " ".join(w.capitalize() for w in words)
    bare = text.strip(" .,!?")
    if re.fullmatch(r"[A-Z][a-z'’-]+ [A-Z][a-z'’-]+", bare):
        return bare
    return None


def extract_phone(text: str) -> str | None:
    digits = digits_from_speech(text)
    return digits if len(digits) >= 10 else None


def pick_slot(text: str, offered: list[JsonDict]) -> JsonDict | None:
    if not offered:
        return None
    lowered = text.lower()
    mentioned = extract_times(text)
    for slot in offered:
        if any(t in slot["time"] for t in mentioned):
            return slot
    if re.search(r"\b(first|earliest|earlier|sooner|1st)\b", lowered):
        return offered[0]
    if re.search(r"\b(second|middle|2nd)\b", lowered) and len(offered) > 1:
        return offered[1]
    if re.search(r"\b(last|latest|later|third|3rd)\b", lowered):
        return offered[-1]
    if len(offered) == 1 and re.search(r"\b(yes|yeah|sure|that works|ok|okay|perfect|great)\b", lowered):
        return offered[0]
    return None


class FakeReceptionist:
    def __init__(self, clinic: Clinic, *, delay_s: float = 0.0) -> None:
        self.clinic = clinic
        self.delay_s = delay_s
        self._counter = 0

    @property
    def label(self) -> str:
        return "fake/callie-rules"

    @property
    def is_remote(self) -> bool:
        return False

    async def stream(
        self, messages: list[JsonDict], tools: list[JsonDict], *, max_tokens: int, temperature: float
    ) -> AsyncIterator[LLMEvent]:
        text, calls = self.decide(messages)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        for word in re.findall(r"\S+\s*", text):
            yield TextDelta(word)
        for name, arguments in calls:
            self._counter += 1
            yield ToolCall(id=f"fake_{self._counter}", name=name, arguments=arguments, raw_arguments=json.dumps(arguments))
        yield Completed(finish_reason="tool_calls" if calls else "stop", served_model=self.label)

    # -- the rules ---------------------------------------------------------------------------------------------
    def decide(self, messages: list[JsonDict]) -> tuple[str, list[tuple[str, JsonDict]]]:
        last = messages[-1]
        results = _tool_results(messages)
        if last.get("role") == "tool":
            return self._after_tool(results[-1] if results else ("", {}, {}, 0)), []
        user_turns = [message_text(m) for m in messages if m.get("role") == "user"]
        text = user_turns[-1] if user_turns else ""
        lowered = text.lower()
        all_user = " ".join(user_turns)

        name = next((n for n in (extract_name(t) for t in reversed(user_turns)) if n), None)
        phone = next((p for p in (extract_phone(t) for t in reversed(user_turns)) if p), None)
        service = next((s for s in (self.clinic.match_service(t) for t in reversed(user_turns)) if s), None)
        last_tool = results[-1] if results else None
        found = next((r for r in reversed(results) if r[0] == "find_appointment" and r[2].get("appointments")), None)
        offered_result = next((r for r in reversed(results) if r[0] == "check_availability"), None)
        offered = (offered_result[2].get("slots") or offered_result[2].get("alternatives") or []) if offered_result else []
        purpose = "reschedule" if re.search(r"\b(reschedul\w*|move|change|push|different (?:day|time))\b", all_user.lower()) else (
            "cancel" if re.search(r"\bcancel\w*\b", all_user.lower()) else None)  # fmt: skip

        # A read-back is waiting for a yes or a no.
        if last_tool and last_tool[2].get("status") == "needs_confirmation":
            if re.match(r"^\W*(yes|yeah|yep|correct|that's right|right|sure|perfect|sounds good)\b", lowered):
                return "", [(last_tool[0], {**last_tool[1], "confirmed": True})]
            if re.match(r"^\W*(no|nope|not quite|actually|wait)\b", lowered):
                return "No problem. What would you like to change?", []
        if last_tool and last_tool[0] == "find_appointment" and last_tool[2].get("pending") == "cancel_appointment":
            if re.match(r"^\W*(yes|yeah|yep|please|sure|correct)\b", lowered):
                appointment = last_tool[2]["appointments"][0]["appointment_id"]
                return "", [("cancel_appointment", {"appointment_id": appointment, "confirmed": True})]

        if re.search(r"\b(leave a message|take a message|pass (?:on )?a message|message for)\b", lowered):
            if not name:
                return "Of course. Can I get your name first?", []
            return "", [("take_message", {"caller_name": name, "message": text, **({"phone": phone} if phone else {})})]

        if purpose and not found:
            if not name and not phone:
                return "Sure. What's the name the appointment is under?", []
            return "", [("find_appointment", {"patient_name": name or "", "purpose": purpose, **({"phone": phone} if phone else {})})]
        if found and purpose == "reschedule":
            appointment = found[2]["appointments"][0]
            pick = pick_slot(text, offered) if offered_result and results.index(offered_result) > results.index(found) else None
            if pick:
                return "", [("reschedule_appointment", {"appointment_id": appointment["appointment_id"], "slot_id": pick["slot_id"], "confirmed": False})]
            if extract_times(text) or re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|tomorrow|next|week|morning|afternoon)\b", lowered):
                return "", [("check_availability", {"service": appointment["service"], "when": text, "appointment_id": appointment["appointment_id"]})]

        if offered and offered_result and not found:
            # Which slot did the caller pick, this turn or since the slots were offered?
            trigger = max((i for i, m in enumerate(messages[: offered_result[3]]) if m.get("role") == "user"), default=0)
            since = [message_text(m) for m in messages[trigger:] if m.get("role") == "user"]
            exact = offered_result[2].get("exact_match")
            pick = next((p for p in (pick_slot(t, offered) for t in reversed(since)) if p), None)
            if pick is None and exact and len(since) > 1:
                pick = next((slot for slot in offered if slot["slot_id"] == exact), None)
            asked_new_time = extract_times(text) and pick_slot(text, offered) is None
            if asked_new_time or (pick is None and _DAY_WORDS.search(lowered)):
                when = text if _DAY_WORDS.search(lowered) else f"{offered_result[1].get('when', '')} {text}"
                return "", [("check_availability", {"service": offered_result[1].get("service"), "when": when})]
            if pick:
                if not name:
                    return "Great. Can I get your first and last name?", []
                arguments: JsonDict = {"service": offered_result[1].get("service"), "slot_id": pick["slot_id"], "patient_name": name, "confirmed": False}
                if phone:
                    arguments["phone"] = phone
                return "", [("book_appointment", arguments)]

        if _QUESTION.search(lowered) and not re.search(r"\b(book|schedule|appointment|available|availability|opening)\b", lowered):
            return "", [("answer_faq", {"question": text})]

        wants_booking = re.search(r"\b(book|schedule|appointment|come in|get in|see the dentist|availab\w*|opening|slot)\b", lowered)
        if wants_booking or service:
            if service is None:
                return "Sure, I can help with that. What kind of appointment do you need?", []
            when = text if (extract_times(text) or re.search(
                r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|morning|afternoon|evening|soon|asap|earliest|\d{1,2}(st|nd|rd|th))\b",
                lowered)) else ""  # fmt: skip
            if not when:
                return f"Sure, a {service.name}. What day and time work best for you?", []
            return "Let me check that for you.", [("check_availability", {"service": service.id, "when": when})]
        if extract_times(text) or re.search(r"\b(tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|next week)\b", lowered):
            if service:
                return "", [("check_availability", {"service": service.id, "when": text})]
            return "What kind of appointment is it for?", []
        return "I can help you book, change or cancel an appointment, answer questions about the clinic, or take a message. What can I do for you?", []

    def _after_tool(self, result: tuple[str, JsonDict, JsonDict, int]) -> str:
        name, arguments, data, _ = result
        if name == "answer_faq" and data.get("passages"):
            passage = data["passages"][0]["text"]
            question = set(re.findall(r"[a-z]{4,}", str(arguments.get("question", "")).lower()))
            sentences = re.split(r"(?<=[.!?])\s+", passage)
            best = max(sentences, key=lambda s: len(question & set(re.findall(r"[a-z]{4,}", s.lower()))))
            return best + " Is there anything else I can help with?"
        if data.get("status") == "error":
            return "Sorry, let me look at that again. What day and time would you like?"
        return "Okay. Anything else I can help with?"
