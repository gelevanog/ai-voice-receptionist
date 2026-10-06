"""Process-wide wiring: settings -> clinic, database, calendar, knowledge base, telephony client, models."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from callie.agent.agent import Agent
from callie.agent.tools import CallContext
from callie.clinic import Clinic, load_clinic
from callie.config import Settings
from callie.kb.retriever import KnowledgeBase
from callie.llm.base import ChatModel
from callie.llm.factory import build_chat_model
from callie.scheduling.calendar import Calendar, seed_demo_appointments
from callie.scheduling.db import Appointment, make_session_factory
from callie.scheduling.google_calendar import GoogleCalendarClient
from callie.scheduling.timeparse import clinic_now
from callie.transports.twilio_rest import TwilioRest


@dataclass
class Runtime:
    settings: Settings
    clinic: Clinic
    sessions: sessionmaker[Session]
    calendar: Calendar
    kb: KnowledgeBase
    twilio: TwilioRest
    llm: ChatModel
    google: GoogleCalendarClient | None = None
    now: Callable[[], datetime] = field(default=lambda: datetime.now())
    speech: Any = None  # callie.pipeline.components.SpeechStack, built lazily (models load on first use)

    def new_context(self, *, caller_phone: str | None = None, call_id: str | None = None) -> CallContext:
        return CallContext(
            call_id=call_id or f"call_{uuid.uuid4().hex[:12]}",
            clinic=self.clinic,
            calendar=self.calendar,
            kb=self.kb,
            sessions=self.sessions,
            now=self.now,
            twilio=self.twilio,
            transfer_number=self.settings.transfer_number,
            caller_phone=caller_phone,
            google=self.google,
        )

    def new_agent(self, ctx: CallContext, llm: ChatModel | None = None) -> Agent:
        return Agent(
            llm or self.llm,
            ctx,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            filler_after_s=self.settings.filler_after_ms / 1000 if self.settings.filler_after_ms > 0 else None,
            fast_confirm=self.settings.fast_confirm,
        )


def build_runtime(settings: Settings, *, llm: ChatModel | None = None, database_url: str | None = None) -> Runtime:
    clinic = load_clinic(settings.clinic_file)
    sessions = make_session_factory(database_url or settings.database_url)
    frozen = settings.now

    def now() -> datetime:
        return clinic_now(clinic.timezone, frozen)

    google = None
    if settings.google_calendar_id and settings.google_access_token:
        google = GoogleCalendarClient(
            settings.google_calendar_id, settings.google_access_token, timezone=clinic.timezone
        )
    calendar = Calendar(clinic, sessions, now, external_busy=google.busy if google else None)
    if settings.seed_demo_data:
        with sessions() as session:
            empty = session.query(Appointment).count() == 0
        if empty:
            seed_demo_appointments(calendar)
    return Runtime(
        settings=settings,
        clinic=clinic,
        sessions=sessions,
        calendar=calendar,
        kb=KnowledgeBase.from_markdown(clinic.knowledge_text()),
        twilio=TwilioRest(settings.twilio_account_sid, settings.twilio_auth_token, settings.twilio_from_number),
        llm=llm or build_chat_model(settings, clinic),
        google=google,
        now=now,
    )
