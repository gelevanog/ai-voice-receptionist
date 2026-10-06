"""The receptionist's tools and the rules around them.

Three rules are enforced here, in code, whatever the model does:

1. **Availability only comes from tool results.** `check_availability` records the slots it offered in this call;
   booking and rescheduling accept only those slot ids, and the calendar re-checks the slot inside a lock.
2. **Nothing changes without a read-back and a yes.** The first booking / reschedule / cancel call (confirmed=false)
   stores a pending action and returns a deterministic read-back for the agent to say. A confirmed call succeeds
   only if the details match that read-back and the caller's very next turn was a clear yes.
3. **Exact facts are spoken by code.** Slot offers, read-backs and confirmations come back as a `say` string the
   pipeline speaks verbatim, so dates and times are never paraphrased by the model.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from callie.agent.grounding import extract_times
from callie.clinic import Clinic, Service
from callie.kb.retriever import KnowledgeBase
from callie.llm.base import JsonDict
from callie.privacy import Masker, looks_like_name, mask_phone, normalize_phone
from callie.scheduling.calendar import Calendar, Slot, SlotUnavailableError
from callie.scheduling.db import Appointment, Message, SmsOutbox
from callie.scheduling.google_calendar import GoogleCalendarClient, GoogleCalendarError
from callie.scheduling.timeparse import parse_when, speak_date, speak_slot, speak_time
from callie.transports.twilio_rest import TwilioRest

CONFIRMING_TOOLS = {"book_appointment", "reschedule_appointment", "cancel_appointment"}


def _fn(name: str, description: str, properties: JsonDict, required: list[str]) -> JsonDict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def tool_specs(clinic: Clinic) -> list[JsonDict]:
    service_ids = [s.id for s in clinic.services]
    service = {"type": "string", "enum": service_ids, "description": "Service id."}
    return [
        _fn(
            "check_availability",
            "Find free appointment times. Put the caller's own words for the day and time in `when` "
            "(e.g. 'next Tuesday after lunch', 'as soon as possible'); never convert dates yourself. "
            "For a reschedule, pass the appointment_id from find_appointment.",
            {"service": service, "when": {"type": "string"}, "appointment_id": {"type": "string"}},
            ["service", "when"],
        ),
        _fn(
            "book_appointment",
            "Book a slot returned by check_availability. First call with confirmed=false: the system reads the "
            "details back to the caller. Call again with confirmed=true only after the caller says yes.",
            {
                "service": service,
                "slot_id": {"type": "string", "description": "slot_id from check_availability"},
                "patient_name": {"type": "string", "description": "Full name as the caller said it."},
                "phone": {"type": "string", "description": "Caller's phone number, digits."},
                "confirmed": {"type": "boolean"},
            },
            ["service", "slot_id", "patient_name", "confirmed"],
        ),
        _fn(
            "find_appointment",
            "Look up the caller's upcoming appointments by name and/or phone. purpose: reschedule, cancel or check.",
            {
                "patient_name": {"type": "string"},
                "phone": {"type": "string"},
                "purpose": {"type": "string", "enum": ["reschedule", "cancel", "check"]},
            },
            ["purpose"],
        ),
        _fn(
            "reschedule_appointment",
            "Move an appointment to a slot from check_availability. confirmed=false first, true after a yes.",
            {"appointment_id": {"type": "string"}, "slot_id": {"type": "string"}, "confirmed": {"type": "boolean"}},
            ["appointment_id", "slot_id", "confirmed"],
        ),
        _fn(
            "cancel_appointment",
            "Cancel an appointment found with find_appointment. confirmed=false first, true after a yes.",
            {"appointment_id": {"type": "string"}, "confirmed": {"type": "boolean"}},
            ["appointment_id", "confirmed"],
        ),
        _fn(
            "answer_faq",
            "Search the clinic's knowledge base (hours, prices, insurance, location, parking, services, policies). "
            "Answer only from what it returns.",
            {"question": {"type": "string"}},
            ["question"],
        ),
        _fn(
            "take_message",
            "Leave a message for the clinic team, who will call back.",
            {
                "caller_name": {"type": "string"},
                "message": {"type": "string"},
                "phone": {"type": "string"},
                "urgent": {"type": "boolean"},
            },
            ["caller_name", "message"],
        ),
        _fn(
            "transfer_to_human",
            "Transfer the call to the front desk: the caller asks for a person, is upset, it is an emergency, "
            "or you cannot help.",
            {"reason": {"type": "string"}},
            ["reason"],
        ),
        _fn(
            "send_confirmation_sms",
            "Text the appointment details to the caller again.",
            {"appointment_id": {"type": "string"}},
            ["appointment_id"],
        ),
        _fn("end_call", "Say goodbye and hang up when the caller is done.", {"reason": {"type": "string"}}, []),
    ]


@dataclass
class PendingAction:
    tool: str
    details: JsonDict
    read_back: str
    turn: int
    affirmed_turn: int | None = None


@dataclass
class ToolResult:
    name: str
    arguments: JsonDict
    data: JsonDict
    say: str | None = None
    action: str | None = None  # transfer | hangup
    elapsed_ms: float = 0.0

    def for_model(self) -> str:
        import json

        payload = dict(self.data)
        if self.say:
            payload["spoken_to_caller"] = self.say
        return json.dumps(payload, ensure_ascii=False, default=str)


@dataclass
class Outcome:
    booked: list[str] = field(default_factory=list)
    rescheduled: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    messages: int = 0
    transferred: str | None = None
    ended_by_agent: bool = False
    blocked_unconfirmed: int = 0  # confirmed=true calls refused because the caller had not said yes
    blocked_unoffered: int = 0  # slot ids that were never offered in this call

    def label(self) -> str:
        if self.transferred:
            return "transferred"
        if self.booked:
            return "booked"
        if self.rescheduled:
            return "rescheduled"
        if self.cancelled:
            return "cancelled"
        if self.messages:
            return "message_taken"
        return "info_only"


@dataclass
class CallContext:
    call_id: str
    clinic: Clinic
    calendar: Calendar
    kb: KnowledgeBase
    sessions: sessionmaker[Session]
    now: Callable[[], datetime]
    twilio: TwilioRest
    transfer_number: str
    caller_phone: str | None = None
    google: GoogleCalendarClient | None = None
    masker: Masker = field(default_factory=Masker)
    offered: dict[str, Slot] = field(default_factory=dict)
    pending: PendingAction | None = None
    turn: int = 0
    outcome: Outcome = field(default_factory=Outcome)
    tool_log: list[JsonDict] = field(default_factory=list)
    known_times: set[str] = field(default_factory=set)  # times that tools or the caller mentioned
    phone_failures: int = 0
    phone_skipped: bool = False
    reschedule_slots: dict[str, str] = field(default_factory=dict)  # slot key -> appointment being moved


def _digits_spoken(phone: str | None) -> str:
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    return " ".join(digits[-4:])


class ToolBox:
    def __init__(self, ctx: CallContext) -> None:
        self.ctx = ctx

    # -- dispatch ----------------------------------------------------------------------------------------------
    def execute(self, name: str, arguments: JsonDict) -> ToolResult:
        started = time.perf_counter()
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None or "_invalid_json" in arguments:
            data: JsonDict = {"status": "error", "error": f"unknown tool or invalid arguments: {name}"}
            result = ToolResult(name, arguments, data)
        else:
            try:
                result = handler(arguments)
            except (KeyError, TypeError, ValueError) as exc:
                result = ToolResult(name, arguments, {"status": "error", "error": f"bad arguments: {exc}"})
        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        self.ctx.tool_log.append(
            {
                "turn": self.ctx.turn,
                "name": name,
                "arguments": self.ctx.masker.mask_value(arguments),
                "result": self.ctx.masker.mask_value(result.data),
                "say": self.ctx.masker.mask(result.say or ""),
                "ms": result.elapsed_ms,
            }
        )
        return result

    def register_reply(self, affirmed: bool, declined: bool) -> None:
        """Called once per caller turn, before any tool runs in that turn."""
        pending = self.ctx.pending
        if pending is None:
            return
        if affirmed and pending.turn == self.ctx.turn - 1:
            pending.affirmed_turn = self.ctx.turn
        elif declined:
            self.ctx.pending = None

    # -- helpers -----------------------------------------------------------------------------------------------
    def _service(self, value: Any) -> Service | None:
        if not value:
            return None
        return self.ctx.clinic.service(str(value)) or self.ctx.clinic.match_service(str(value))

    def _remember_times(self, *moments: datetime) -> None:
        for moment in moments:
            local = moment.astimezone(self.ctx.clinic.tz)
            self.ctx.known_times.add(speak_time(local.time()))

    def _confirm_gate(self, tool: str, details: JsonDict, read_back: str, confirmed: bool) -> ToolResult | None:
        """None when the action may run now; otherwise the result to return (read-back or refusal)."""
        pending = self.ctx.pending
        same = pending is not None and pending.tool == tool and pending.details == details
        if confirmed and same and pending is not None and pending.affirmed_turn == self.ctx.turn:
            return None
        if confirmed and same:
            self.ctx.outcome.blocked_unconfirmed += 1
            return ToolResult(
                tool,
                details,
                {"status": "needs_confirmation", "error": "the caller has not said yes to the read-back yet"},
                say=read_back,
            )
        if confirmed:
            self.ctx.outcome.blocked_unconfirmed += 1
        self.ctx.pending = PendingAction(tool, details, read_back, self.ctx.turn)
        return ToolResult(tool, details, {"status": "needs_confirmation", "read_back": read_back}, say=read_back)

    def _slot(self, slot_id: str, service: Service) -> Slot | None:
        slot = self.ctx.offered.get(str(slot_id).strip())
        if slot is None or slot.service_id != service.id:
            return None
        return slot

    def _unoffered(self, name: str, arguments: JsonDict) -> ToolResult:
        self.ctx.outcome.blocked_unoffered += 1
        return ToolResult(
            name,
            arguments,
            {
                "status": "error",
                "error": "slot_id was not offered in this call; call check_availability and offer only its slots",
                "offered_slot_ids": list(self.ctx.offered),
            },
        )

    def _mirror(self, appointment: Appointment, change: str) -> None:
        google = self.ctx.google
        if google is None:
            return
        try:
            if change == "book":
                service = self.ctx.clinic.service(appointment.service_id)
                appointment.external_id = google.create_event(
                    f"{service.name if service else appointment.service_id}: "
                    f"{self.ctx.masker.mask(appointment.patient_name)}",
                    appointment.start,
                    appointment.end,
                    "Booked by Callie",
                )
                with self.ctx.sessions() as session, session.begin():
                    row = session.get(Appointment, appointment.id)
                    if row is not None:
                        row.external_id = appointment.external_id
            elif change == "move" and appointment.external_id:
                google.move_event(appointment.external_id, appointment.start, appointment.end)
            elif change == "cancel" and appointment.external_id:
                google.delete_event(appointment.external_id)
        except GoogleCalendarError:
            pass  # the clinic's own calendar is the source of truth; the mirror is best effort

    def _sms(self, appointment: Appointment, prefix: str) -> str:
        if not appointment.phone:
            return "no_phone"
        service = self.ctx.clinic.service(appointment.service_id)
        body = (
            f"{self.ctx.clinic.name}: {prefix} {service.name if service else 'appointment'} on "
            f"{speak_slot(appointment.start.astimezone(self.ctx.clinic.tz))}. {self.ctx.clinic.address}. "
            f"Reply C to cancel or call {self.ctx.clinic.phone}."
        )
        result = self.ctx.twilio.send_sms(appointment.phone, body)
        with self.ctx.sessions() as session, session.begin():
            session.add(
                SmsOutbox(
                    to_masked=mask_phone(appointment.phone),
                    body=self.ctx.masker.mask(body),
                    status=result.status,
                    provider_id=result.provider_id,
                    appointment_id=appointment.id,
                )
            )
        return result.status

    # -- tools -------------------------------------------------------------------------------------------------
    def _tool_check_availability(self, args: JsonDict) -> ToolResult:
        service = self._service(args.get("service"))
        when_text = str(args.get("when") or "").strip()
        if service is None:
            return ToolResult(
                "check_availability",
                args,
                {"status": "need_service", "services": [s.name for s in self.ctx.clinic.services]},
            )
        parsed = parse_when(when_text, self.ctx.now())
        exclude = str(args.get("appointment_id") or "") or None
        if not parsed.understood:
            return ToolResult(
                "check_availability",
                args,
                {"status": "need_when", "understood_as": "not understood"},
                say="What day and time would work best for you?",
            )
        slots = self.ctx.calendar.find_slots(service, parsed, exclude_id=exclude)
        alternatives: list[Slot] = []
        if not slots:
            after = max(w.end for w in parsed.windows)
            alternatives = self.ctx.calendar.next_free(service, max(after, self.ctx.now()), limit=2)
        offered = slots or alternatives
        for slot in offered:
            self.ctx.offered[slot.key] = slot
            if exclude:
                self.ctx.reschedule_slots[slot.key] = exclude
        self._remember_times(*(s.start for s in offered))
        data: JsonDict = {
            "status": "ok" if slots else "no_slots_in_window",
            "service": service.id,
            "understood_as": parsed.describe(),
            "assumptions": parsed.assumptions,
            "slots": [{"slot_id": s.key, "time": s.spoken()} for s in slots],
            "alternatives": [{"slot_id": s.key, "time": s.spoken()} for s in alternatives],
        }
        preferred = next((w.preferred for w in parsed.windows if w.preferred is not None), None)
        exact = next((s for s in slots if preferred is not None and s.start == preferred), None)
        if exact is not None:
            data["exact_match"] = exact.key
            say = f"Yes, {exact.spoken()} is open. Would you like me to book it?"
        else:
            say = self._offer_text(service, parsed.describe(), slots, alternatives, parsed.asap)
        return ToolResult("check_availability", args, data, say=say)

    def _offer_text(
        self, service: Service, understood: str, slots: list[Slot], alternatives: list[Slot], asap: bool
    ) -> str:
        def join(items: list[str]) -> str:
            return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]

        def times(found: list[Slot]) -> str:
            by_day: dict[str, list[str]] = {}
            for slot in found:
                by_day.setdefault(speak_date(slot.start.date()), []).append(speak_time(slot.start.time()))
            return "; or ".join(f"{day} at {join(ts)}" for day, ts in by_day.items())

        if slots:
            lead = "The earliest I have is" if asap else "I have"
            question = "Does that work?" if len(slots) == 1 else "Which would you prefer?"
            return f"{lead} {times(slots)}. {question}"
        if alternatives:
            return (
                f"I'm sorry, I don't have anything for {understood}. "
                f"The next openings are {times(alternatives)}. Would one of those work?"
            )
        return f"I'm sorry, I don't see any openings for a {service.name} around then. Is there another day that works?"

    def _tool_book_appointment(self, args: JsonDict) -> ToolResult:
        service = self._service(args.get("service"))
        if service is None:
            return ToolResult("book_appointment", args, {"status": "error", "error": "unknown service"})
        slot = self._slot(str(args.get("slot_id", "")), service)
        if slot is None:
            return self._unoffered("book_appointment", args)
        moving = self.ctx.reschedule_slots.get(slot.key)
        if moving:
            # The slot came from a reschedule search: booking it would leave the old appointment in place.
            return ToolResult("book_appointment", args, {
                "status": "error",
                "error": f"the caller is moving appointment {moving}; call reschedule_appointment with "
                f"appointment_id={moving} and slot_id={slot.key} instead of booking a second appointment",
            })  # fmt: skip
        name = " ".join(str(args.get("patient_name") or "").split()).title()
        if not looks_like_name(name):
            return ToolResult(
                "book_appointment", args, {"status": "need_name"}, say="Can I get your first and last name, please?"
            )
        phone = normalize_phone(str(args.get("phone") or "")) or self.ctx.caller_phone
        if phone is None and not self.ctx.phone_skipped:
            heard = "".join(ch for ch in str(args.get("phone") or "") if ch.isdigit())
            if not heard:
                return ToolResult(
                    "book_appointment",
                    args,
                    {"status": "need_phone"},
                    say="And what's the best phone number for your confirmation text?",
                )
            self.ctx.phone_failures += 1
            if self.ctx.phone_failures >= 2:
                self.ctx.phone_skipped = True  # book anyway, without the text; the read-back says so
            else:
                return ToolResult(
                    "book_appointment",
                    args,
                    {"status": "phone_unclear", "heard_digits": len(heard)},
                    say=f"Sorry, I got {len(heard)} digits instead of 10. Could you say your number again, "
                    "one digit at a time?",
                )
        self.ctx.masker.register_name(name)
        details = {"service": service.id, "slot_id": slot.key, "patient_name": name, "phone": phone}
        texting = (
            f"and I'll text the confirmation to the number ending in {_digits_spoken(phone)}"
            if phone
            else "and since I couldn't catch your number, I won't be able to text you a confirmation"
        )
        read_back = f"Just to confirm: a {service.name} for {name} on {slot.spoken()}, {texting}. Is that right?"
        gate = self._confirm_gate("book_appointment", details, read_back, bool(args.get("confirmed")))
        if gate is not None:
            return gate
        try:
            appointment = self.ctx.calendar.book(service, slot.start, name, phone, call_id=self.ctx.call_id)
        except SlotUnavailableError as exc:
            self.ctx.pending = None
            self.ctx.offered.pop(slot.key, None)
            return ToolResult(
                "book_appointment",
                args,
                {"status": "slot_unavailable", "error": str(exc)},
                say="I'm sorry, that time was just taken. Would you like me to look for another time?",
            )
        self.ctx.pending = None
        self.ctx.outcome.booked.append(appointment.id)
        self._mirror(appointment, "book")
        sms = self._sms(appointment, "your")
        texted = " I've sent you a confirmation text." if sms in {"sent", "dry_run"} else ""
        return ToolResult(
            "book_appointment",
            args,
            {"status": "booked", "appointment_id": appointment.id, "time": slot.spoken(), "sms": sms},
            say=f"You're all set for {slot.spoken()}.{texted} Is there anything else I can help you with?",
        )

    def _tool_find_appointment(self, args: JsonDict) -> ToolResult:
        name = str(args.get("patient_name") or "").strip() or None
        phone = normalize_phone(str(args.get("phone") or "")) or (self.ctx.caller_phone if not name else None)
        purpose = str(args.get("purpose") or "check")
        if not name and not phone:
            return ToolResult(
                "find_appointment",
                args,
                {"status": "need_identity"},
                say="Sure. What's the name the appointment is under?",
            )
        self.ctx.masker.register_name(name)
        found = self.ctx.calendar.find_appointments(name=name, phone=phone)
        if not found and name and phone:
            found = self.ctx.calendar.find_appointments(name=name)
        items = []
        for appointment in found[:3]:
            service = self.ctx.clinic.service(appointment.service_id)
            local = appointment.start.astimezone(self.ctx.clinic.tz)
            self._remember_times(local)
            items.append(
                {
                    "appointment_id": appointment.id,
                    "service": service.id if service else appointment.service_id,
                    "service_name": service.name if service else appointment.service_id,
                    "time": speak_slot(local),
                }
            )
        if not items:
            return ToolResult(
                "find_appointment",
                args,
                {"status": "not_found"},
                say="I couldn't find an upcoming appointment under that name. "
                "Could you spell the last name for me, or give me the phone number on file?",
            )
        if len(items) > 1:
            listed = " and ".join(f"a {i['service_name']} on {i['time']}" for i in items)
            return ToolResult(
                "find_appointment",
                args,
                {"status": "found", "appointments": items},
                say=f"I see {listed}. Which one is it about?",
            )
        item = items[0]
        if purpose == "cancel":
            read_back = f"I found your {item['service_name']} on {item['time']}. Would you like me to cancel it?"
            details = {"appointment_id": item["appointment_id"]}
            self.ctx.pending = PendingAction("cancel_appointment", details, read_back, self.ctx.turn)
            return ToolResult(
                "find_appointment",
                args,
                {"status": "found", "appointments": items, "pending": "cancel_appointment"},
                say=read_back,
            )
        if purpose == "reschedule":
            return ToolResult(
                "find_appointment",
                args,
                {"status": "found", "appointments": items},
                say=f"I found your {item['service_name']} on {item['time']}. What day and time would you like instead?",
            )
        return ToolResult(
            "find_appointment",
            args,
            {"status": "found", "appointments": items},
            say=f"You have a {item['service_name']} on {item['time']}. Is there anything else I can help with?",
        )

    def _tool_reschedule_appointment(self, args: JsonDict) -> ToolResult:
        appointment = self.ctx.calendar.get(str(args.get("appointment_id", "")))
        if appointment is None or appointment.status != "booked":
            return ToolResult("reschedule_appointment", args, {"status": "error", "error": "appointment not found"})
        service = self.ctx.clinic.service(appointment.service_id)
        assert service is not None
        slot = self._slot(str(args.get("slot_id", "")), service)
        if slot is None:
            return self._unoffered("reschedule_appointment", args)
        old = speak_slot(appointment.start.astimezone(self.ctx.clinic.tz))
        details = {"appointment_id": appointment.id, "slot_id": slot.key}
        read_back = f"Just to confirm: I'll move your {service.name} from {old} to {slot.spoken()}. Is that right?"
        gate = self._confirm_gate("reschedule_appointment", details, read_back, bool(args.get("confirmed")))
        if gate is not None:
            return gate
        try:
            moved = self.ctx.calendar.reschedule(appointment.id, slot.start)
        except SlotUnavailableError as exc:
            self.ctx.pending = None
            return ToolResult(
                "reschedule_appointment",
                args,
                {"status": "slot_unavailable", "error": str(exc)},
                say="I'm sorry, that time was just taken. Shall I look for another one?",
            )
        self.ctx.pending = None
        self.ctx.outcome.rescheduled.append(moved.id)
        self._mirror(moved, "move")
        sms = self._sms(moved, "we moved your")
        texted = " I've texted you the new time." if sms in {"sent", "dry_run"} else ""
        return ToolResult(
            "reschedule_appointment",
            args,
            {"status": "rescheduled", "appointment_id": moved.id, "time": slot.spoken()},
            say=f"Done. Your {service.name} is now on {slot.spoken()}.{texted} Anything else?",
        )

    def _tool_cancel_appointment(self, args: JsonDict) -> ToolResult:
        appointment = self.ctx.calendar.get(str(args.get("appointment_id", "")))
        if appointment is None or appointment.status != "booked":
            return ToolResult("cancel_appointment", args, {"status": "error", "error": "appointment not found"})
        service = self.ctx.clinic.service(appointment.service_id)
        when = speak_slot(appointment.start.astimezone(self.ctx.clinic.tz))
        details = {"appointment_id": appointment.id}
        read_back = (
            f"Just to confirm: you'd like to cancel your {service.name if service else 'appointment'} on {when}?"
        )
        gate = self._confirm_gate("cancel_appointment", details, read_back, bool(args.get("confirmed")))
        if gate is not None:
            return gate
        self.ctx.calendar.cancel(appointment.id)
        self.ctx.pending = None
        self.ctx.outcome.cancelled.append(appointment.id)
        self._mirror(appointment, "cancel")
        late = appointment.start - self.ctx.now() < timedelta(hours=24)
        fee = " Since it's less than 24 hours away, a $40 late cancellation fee may apply." if late else ""
        return ToolResult(
            "cancel_appointment",
            args,
            {"status": "cancelled", "appointment_id": appointment.id, "late": late},
            say=f"Your appointment on {when} is cancelled.{fee} Would you like to book a new time?",
        )

    def _tool_answer_faq(self, args: JsonDict) -> ToolResult:
        question = str(args.get("question") or "")
        hits = self.ctx.kb.search(question, k=2)
        if not hits:
            return ToolResult(
                "answer_faq",
                args,
                {"status": "no_answer"},
                say="I'm sorry, I don't have that information. "
                "I can take a message and someone from the team will call you back.",
            )
        for hit in hits:
            self.ctx.known_times.update(extract_times(hit.passage.text))
        return ToolResult(
            "answer_faq",
            args,
            {"status": "ok", "passages": [{"title": h.passage.title, "text": h.passage.text} for h in hits]},
        )

    def _tool_take_message(self, args: JsonDict) -> ToolResult:
        name = " ".join(str(args.get("caller_name") or "").split()).title() or "Unknown caller"
        text = str(args.get("message") or "").strip()
        phone = normalize_phone(str(args.get("phone") or "")) or self.ctx.caller_phone
        if not text:
            return ToolResult(
                "take_message",
                args,
                {"status": "need_message"},
                say="Of course. What message would you like me to pass on?",
            )
        heard = "".join(ch for ch in str(args.get("phone") or "") if ch.isdigit())
        if phone is None and heard and self.ctx.phone_failures < 1:
            self.ctx.phone_failures += 1
            return ToolResult(
                "take_message",
                args,
                {"status": "phone_unclear", "heard_digits": len(heard)},
                say=f"Sorry, I got {len(heard)} digits instead of 10. "
                "Could you say the number again, one digit at a time?",
            )
        if phone is None and not heard:
            return ToolResult(
                "take_message", args, {"status": "need_phone"}, say="And what number should they call you back on?"
            )
        self.ctx.masker.register_name(name)
        with self.ctx.sessions() as session, session.begin():
            session.add(
                Message(
                    call_id=None,
                    name=name,
                    phone=phone,
                    text=self.ctx.masker.mask(text),
                    urgency="urgent" if args.get("urgent") else "normal",
                )
            )
        self.ctx.outcome.messages += 1
        reopen = self.ctx.clinic.next_open(self.ctx.now())
        when = (
            "shortly"
            if self.ctx.clinic.is_open(self.ctx.now())
            else (f"when we open, {speak_slot(reopen)}" if reopen else "as soon as possible")
        )
        return ToolResult(
            "take_message",
            args,
            {"status": "message_taken"},
            say=f"Thanks, {name.split()[0]}. I've passed that on, and someone will call you back "
            + (
                f"at the number ending in {_digits_spoken(phone)} {when}. Anything else?"
                if phone
                else f"{when}. I couldn't catch your number, so please call us back if you don't hear from us. "
                "Anything else?"
            ),
        )

    def _tool_transfer_to_human(self, args: JsonDict) -> ToolResult:
        reason = str(args.get("reason") or "caller request")
        emergency = "emerg" in reason.lower()
        if self.ctx.clinic.is_open(self.ctx.now()) or emergency:
            self.ctx.outcome.transferred = reason
            advised = "911" in reason
            prefix = "If this is life-threatening, please hang up and call 911. " if emergency and not advised else ""
            return ToolResult(
                "transfer_to_human",
                args,
                {"status": "transferring", "to": "front desk"},
                say=f"{prefix}I'm connecting you with our team now. Please hold for a moment.",
                action="transfer",
            )
        reopen = self.ctx.clinic.next_open(self.ctx.now())
        when = speak_slot(reopen) if reopen else "the next business day"
        return ToolResult(
            "transfer_to_human",
            args,
            {"status": "front_desk_closed", "reopens": when},
            say=f"Our front desk is closed right now; we open again {when}. "
            "I can take a message so someone calls you back. Would you like that?",
        )

    def _tool_send_confirmation_sms(self, args: JsonDict) -> ToolResult:
        appointment = self.ctx.calendar.get(str(args.get("appointment_id", "")))
        if appointment is None or appointment.status != "booked":
            return ToolResult("send_confirmation_sms", args, {"status": "error", "error": "appointment not found"})
        status = self._sms(appointment, "your")
        return ToolResult(
            "send_confirmation_sms",
            args,
            {"status": status},
            say="I've sent the details by text."
            if status in {"sent", "dry_run"}
            else "I'm sorry, I couldn't send the text.",
        )

    def _tool_end_call(self, args: JsonDict) -> ToolResult:
        self.ctx.outcome.ended_by_agent = True
        return ToolResult(
            "end_call",
            args,
            {"status": "ending"},
            say=f"Thanks for calling {self.ctx.clinic.name}. Have a great day. Goodbye!",
            action="hangup",
        )
