"""The appointment book: business hours, service durations, one schedule per resource, no double booking.

Availability only ever comes from here. The agent cannot offer or book a time this module did not return:
`book` re-checks the slot inside a lock, and the tools refuse slots that were not offered in the same call.
"""

from __future__ import annotations

import random
import re
import secrets
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from difflib import SequenceMatcher

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from callie.clinic import Clinic, Service
from callie.scheduling.db import Appointment
from callie.scheduling.timeparse import ParsedWhen, TimeWindow, speak_slot

BusyProvider = Callable[[datetime, datetime], list[tuple[datetime, datetime]]]


class SlotUnavailableError(Exception):
    """The slot is outside business hours, too soon, too far out, or overlaps another booking."""


@dataclass(frozen=True)
class Slot:
    service_id: str
    resource_id: str
    start: datetime  # clinic local time
    end: datetime

    @property
    def key(self) -> str:
        """Stable id the model passes back when booking: the local start time, minute precision."""
        return self.start.strftime("%Y-%m-%dT%H:%M")

    def spoken(self) -> str:
        return speak_slot(self.start)


class Calendar:
    def __init__(
        self,
        clinic: Clinic,
        sessions: sessionmaker[Session],
        now: Callable[[], datetime],
        external_busy: BusyProvider | None = None,
    ) -> None:
        self.clinic = clinic
        self.sessions = sessions
        self.now = now
        self.external_busy = external_busy
        self._lock = threading.Lock()

    # -- availability ------------------------------------------------------------------------------------------
    def _candidate_starts(self, service: Service, day: date) -> Iterable[datetime]:
        step = timedelta(minutes=self.clinic.slot_minutes)
        length = timedelta(minutes=service.minutes)
        for open_at, close_at in self.clinic.intervals(day):
            cursor = open_at
            while cursor + length <= close_at:
                yield cursor
                cursor += step

    def _busy(
        self, session: Session, resource_id: str, start: datetime, end: datetime, exclude_id: str | None = None
    ) -> list[tuple[datetime, datetime]]:
        query = select(Appointment).where(
            Appointment.resource_id == resource_id,
            Appointment.status == "booked",
            Appointment.start < end,
            Appointment.end > start,
        )
        if exclude_id:
            query = query.where(Appointment.id != exclude_id)
        rows = session.scalars(query).all()
        busy = [(row.start, row.end) for row in rows]
        if self.external_busy is not None:
            busy.extend(self.external_busy(start, end))
        return busy

    def _bookable(self, moment: datetime) -> bool:
        now = self.now()
        earliest = now + timedelta(minutes=self.clinic.min_notice_minutes)
        latest = now + timedelta(days=self.clinic.booking_horizon_days)
        return earliest <= moment <= latest

    def check(self, service: Service, start: datetime, *, exclude_id: str | None = None) -> Slot:
        """Validate one exact start time; raises SlotUnavailableError with a reason."""
        local = start.astimezone(self.clinic.tz)
        end = local + timedelta(minutes=service.minutes)
        if not any(local == c for c in self._candidate_starts(service, local.date())):
            raise SlotUnavailableError("outside opening hours or not on the slot grid")
        if not self._bookable(local):
            raise SlotUnavailableError("too soon or too far in the future")
        with self.sessions() as session:
            busy = self._busy(session, service.resource, local, end, exclude_id)
        if busy:
            raise SlotUnavailableError("already booked")
        return Slot(service.id, service.resource, local, end)

    def free_slots(self, service: Service, window: TimeWindow, *, exclude_id: str | None = None) -> list[Slot]:
        day = window.start.astimezone(self.clinic.tz).date()
        candidates = [
            c for c in self._candidate_starts(service, day) if window.start <= c < window.end and self._bookable(c)
        ]
        if not candidates:
            return []
        length = timedelta(minutes=service.minutes)
        with self.sessions() as session:
            busy = self._busy(session, service.resource, candidates[0], candidates[-1] + length, exclude_id)
        return [
            Slot(service.id, service.resource, c, c + length)
            for c in candidates
            if all(not (c < e and c + length > s) for s, e in busy)
        ]

    def find_slots(
        self, service: Service, when: ParsedWhen, *, limit: int | None = None, exclude_id: str | None = None
    ) -> list[Slot]:
        """Up to `limit` free slots matching the request, spread out so the options are actually different.

        With a preferred time ("at 3:30") the exact slot comes first, then the nearest free ones that day.
        Otherwise the earliest slot of each day, then later ones at least two hours apart.
        """
        limit = limit or self.clinic.max_offered_slots
        per_window = [self.free_slots(service, w, exclude_id=exclude_id) for w in when.windows]
        preferred = [w.preferred for w in when.windows if w.preferred is not None]
        if preferred:
            target = preferred[0]
            pool = [s for slots in per_window for s in slots]
            same_day = [s for s in pool if s.start.date() == target.date()]
            ranked = sorted(same_day or pool, key=lambda s: (abs((s.start - target).total_seconds()), s.start))
            return sorted(ranked[:limit], key=lambda s: s.start)
        chosen: list[Slot] = []
        by_day: dict[date, list[Slot]] = {}
        for slots in per_window:
            for slot in slots:
                by_day.setdefault(slot.start.date(), []).append(slot)
        days = sorted(by_day)
        # Round-robin over days (first slot of each day, then later ones), preferring options at least two hours
        # apart; the spacing is relaxed when the window is too small to give `limit` different options.
        for gap_hours in (2.0, 1.0, 0.5):
            for round_index in range(3):
                for day in days:
                    if len(chosen) >= limit:
                        break
                    picked_today = [s for s in chosen if s.start.date() == day]
                    if len(picked_today) != round_index:
                        continue
                    for slot in by_day[day]:
                        if slot in chosen:
                            continue
                        if all(abs((slot.start - p.start).total_seconds()) >= gap_hours * 3600 for p in picked_today):
                            chosen.append(slot)
                            break
            if len(chosen) >= limit:
                break
        return sorted(chosen, key=lambda s: s.start)

    def next_free(self, service: Service, after: datetime, days: int = 14, limit: int | None = None) -> list[Slot]:
        windows = []
        start_day = after.astimezone(self.clinic.tz).date()
        for offset in range(days):
            day = start_day + timedelta(days=offset)
            day_start = datetime.combine(day, time(0, 0), tzinfo=self.clinic.tz)
            windows.append(TimeWindow(max(day_start, after), day_start + timedelta(days=1)))
        return self.find_slots(service, ParsedWhen(text="next available", windows=windows), limit=limit)

    # -- changes -----------------------------------------------------------------------------------------------
    def book(
        self,
        service: Service,
        start: datetime,
        patient_name: str,
        phone: str | None,
        *,
        call_id: str | None = None,
        source: str = "call",
    ) -> Appointment:
        with self._lock:
            slot = self.check(service, start)
            appointment = Appointment(
                id=_new_id(),
                service_id=service.id,
                resource_id=service.resource,
                start=slot.start,
                end=slot.end,
                patient_name=patient_name.strip(),
                phone=phone,
                status="booked",
                source=source,
                call_id=call_id,
            )
            with self.sessions() as session, session.begin():
                session.add(appointment)
            return appointment

    def reschedule(self, appointment_id: str, new_start: datetime) -> Appointment:
        with self._lock:
            with self.sessions() as session:
                current = session.get(Appointment, appointment_id)
                if current is None or current.status != "booked":
                    raise SlotUnavailableError("appointment not found")
                service = self.clinic.service(current.service_id)
            if service is None:
                raise SlotUnavailableError("unknown service")
            slot = self.check(service, new_start, exclude_id=appointment_id)
            with self.sessions() as session, session.begin():
                row = session.get(Appointment, appointment_id)
                assert row is not None
                row.start, row.end = slot.start, slot.end
            return row

    def cancel(self, appointment_id: str) -> Appointment:
        with self._lock, self.sessions() as session, session.begin():
            row = session.get(Appointment, appointment_id)
            if row is None or row.status != "booked":
                raise SlotUnavailableError("appointment not found")
            row.status = "cancelled"
            return row

    # -- lookups -----------------------------------------------------------------------------------------------
    def get(self, appointment_id: str) -> Appointment | None:
        with self.sessions() as session:
            return session.get(Appointment, appointment_id)

    def find_appointments(self, *, name: str | None = None, phone: str | None = None) -> list[Appointment]:
        """Upcoming booked appointments matching the caller's phone, or name (case-insensitive, last name ok)."""
        with self.sessions() as session:
            rows = session.scalars(
                select(Appointment)
                .where(Appointment.status == "booked", Appointment.start >= self.now())
                .order_by(Appointment.start)
            ).all()
        matches = []
        for row in rows:
            if (phone and row.phone and _digits(row.phone)[-10:] == _digits(phone)[-10:]) or (
                name and _name_matches(name, row.patient_name)
            ):
                matches.append(row)
        return matches

    def between(self, start: datetime, end: datetime, *, include_cancelled: bool = False) -> list[Appointment]:
        with self.sessions() as session:
            query = select(Appointment).where(Appointment.start >= start, Appointment.start < end)
            if not include_cancelled:
                query = query.where(Appointment.status == "booked")
            return list(session.scalars(query.order_by(Appointment.start)).all())


def _new_id() -> str:
    return "A-" + secrets.token_hex(3).upper()


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _name_matches(query: str, full_name: str) -> bool:
    """Tolerant of speech-recognition spellings: "Sophia Rossi" finds "Sofia Rossi", "Gonzales" finds "Gonzalez".

    The last name must be close (similarity >= 0.8); a first name, when given, must be close too or be a short
    form of it ("Liz" / "Elizabeth" does not count: that one is left to a phone-number lookup).
    """
    q = [part for part in re.sub(r"[^a-z\s]", " ", query.lower()).split() if len(part) > 1]
    names = full_name.lower().split()
    if not q or not names:
        return False
    last_ok = _similar(q[-1], names[-1]) >= 0.8
    if len(q) == 1:
        return last_ok
    first_ok = _similar(q[0], names[0]) >= 0.7 or names[0].startswith(q[0])
    return last_ok and first_ok


SEED_FIRST = ["Olivia", "Liam", "Emma", "Noah", "Ava", "Elijah", "Sophia", "Lucas", "Mia", "Mateo", "Harper", "Ethan"]
SEED_LAST = ["Nguyen", "Okafor", "Kowalski", "Haddad", "Larsen", "Moreau", "Tanaka", "Silva", "Brennan", "Iyer"]


def seed_demo_appointments(calendar: Calendar, *, days: int = 21, occupancy: float = 0.45, seed: int = 7) -> int:
    """Fill the next `days` days with fictional bookings so availability looks like a real practice."""
    rng = random.Random(seed)
    now = calendar.now()
    created = 0
    for offset in range(days):
        day = now.astimezone(calendar.clinic.tz).date() + timedelta(days=offset)
        for service in (calendar.clinic.service("cleaning"), calendar.clinic.service("filling")):
            assert service is not None
            for start in list(calendar._candidate_starts(service, day)):
                if rng.random() > occupancy / 2:
                    continue
                name = f"{rng.choice(SEED_FIRST)} {rng.choice(SEED_LAST)}"
                try:
                    calendar.book(service, start, name, f"+1555{rng.randint(1000000, 9999999)}", source="seed")
                    created += 1
                except SlotUnavailableError:
                    continue
    return created
