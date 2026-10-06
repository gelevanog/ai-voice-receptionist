"""Storage: appointments, calls, messages and the SMS outbox (SQLAlchemy 2.0; SQLite by default, Postgres works).

Personal data is kept to what running the clinic needs: an appointment row holds the patient's name and phone
number (the confirmation text goes to that number). Call records, transcripts, tool logs and evaluation
artifacts only ever hold masked forms ("J*** D**", "***-***-4567"); see `callie.privacy`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator[datetime]):
    """Stores aware datetimes as UTC and always returns aware UTC datetimes (SQLite drops the offset)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime given to a UTC column")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Appointment(Base):
    __tablename__ = "appointments"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    service_id: Mapped[str] = mapped_column(String(64))
    resource_id: Mapped[str] = mapped_column(String(64), index=True)
    start: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    end: Mapped[datetime] = mapped_column(UTCDateTime())
    patient_name: Mapped[str] = mapped_column(String(120))
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="booked", index=True)  # booked | cancelled
    source: Mapped[str] = mapped_column(String(16), default="call")  # call | seed | google | dashboard
    call_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    external_id: Mapped[str | None] = mapped_column(String(128), nullable=True)  # Google Calendar event id
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class CallRecord(Base):
    __tablename__ = "calls"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    transport: Mapped[str] = mapped_column(String(16))  # browser | twilio | simulated | text
    caller: Mapped[str | None] = mapped_column(String(40), nullable=True)  # masked
    outcome: Mapped[str] = mapped_column(String(32), default="in_progress")
    outcome_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # masked
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # masked
    turns: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # latency per turn
    recording_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    stack: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # models used
    scenario: Mapped[str | None] = mapped_column(String(64), nullable=True)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str | None] = mapped_column(ForeignKey("calls.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(120))
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    urgency: Mapped[str] = mapped_column(String(16), default="normal")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class SmsOutbox(Base):
    __tablename__ = "sms_outbox"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    to_masked: Mapped[str] = mapped_column(String(40))
    body: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))  # sent | dry_run | failed
    provider_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    appointment_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    if url in {"sqlite://", "sqlite:///:memory:"}:
        engine = create_engine(url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    elif url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(url, pool_pre_ping=True)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    Base.metadata.create_all(engine)
    return engine


def make_session_factory(url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=make_engine(url), expire_on_commit=False)
