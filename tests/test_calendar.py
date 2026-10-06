"""Business hours, durations, conflicts per resource, notice and horizon, slot selection, Google Calendar sync."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from callie.clinic import Clinic
from callie.scheduling.calendar import Calendar, SlotUnavailableError, seed_demo_appointments
from callie.scheduling.google_calendar import GoogleCalendarClient, GoogleCalendarError
from callie.scheduling.timeparse import parse_when
from tests.conftest import FROZEN_NOW, NY


def at(day: int, hour: int, minute: int = 0, month: int = 10) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=NY)


def service(clinic: Clinic, service_id: str):  # type: ignore[no-untyped-def]
    found = clinic.service(service_id)
    assert found is not None
    return found


class TestClinic:
    def test_service_matching(self, clinic: Clinic) -> None:
        assert clinic.match_service("I'd like a teeth cleaning").id == "cleaning"  # type: ignore[union-attr]
        assert clinic.match_service("my tooth hurts, it's a toothache").id == "emergency_visit"  # type: ignore[union-attr]
        assert clinic.match_service("whitening").id == "whitening"  # type: ignore[union-attr]
        assert clinic.match_service("kids_cleaning").id == "kids_cleaning"  # type: ignore[union-attr]
        assert clinic.match_service("a haircut") is None

    def test_hours(self, clinic: Clinic) -> None:
        assert clinic.is_open(at(6, 10))
        assert not clinic.is_open(at(6, 12, 30))  # lunch
        assert not clinic.is_open(at(11, 10))  # Sunday
        assert not clinic.is_open(datetime(2026, 11, 26, 10, tzinfo=NY))  # Thanksgiving
        assert clinic.next_open(at(6, 12, 30)) == at(6, 13)
        assert clinic.next_open(at(10, 14)) == at(12, 8)  # Saturday afternoon -> Monday


class TestAvailability:
    def test_slots_respect_hours_lunch_duration_and_notice(self, calendar: Calendar, clinic: Clinic) -> None:
        cleaning = service(clinic, "cleaning")
        slots = calendar.free_slots(cleaning, parse_when("today", FROZEN_NOW).windows[0])
        starts = [s.start for s in slots]
        # 9:30 now + 2 h notice = 11:30, but a 60-minute cleaning at 11:30 would run into lunch.
        assert starts[0] == at(6, 13)
        assert at(6, 11) not in starts and at(6, 11, 30) not in starts
        assert at(6, 16) in starts and at(6, 16, 30) not in starts  # must end by 5 PM
        whitening = service(clinic, "whitening")  # 90 minutes
        friday = calendar.free_slots(whitening, parse_when("friday", FROZEN_NOW).windows[0])
        assert friday[-1].start == at(9, 14, 30)  # Friday closes at 4 PM

    def test_closed_days_and_horizon(self, calendar: Calendar, clinic: Clinic) -> None:
        cleaning = service(clinic, "cleaning")
        assert calendar.find_slots(cleaning, parse_when("sunday", FROZEN_NOW)) == []
        assert calendar.find_slots(cleaning, parse_when("November 26", FROZEN_NOW)) == []
        assert calendar.find_slots(cleaning, parse_when("January 20", FROZEN_NOW)) == []  # beyond 60 days

    def test_conflicts_are_per_resource(self, calendar: Calendar, clinic: Clinic) -> None:
        cleaning, filling = service(clinic, "cleaning"), service(clinic, "filling")
        calendar.book(cleaning, at(8, 9), "Ana Lima", "+15550001111")
        with pytest.raises(SlotUnavailableError, match="already booked"):
            calendar.book(cleaning, at(8, 9, 30), "Bo Chen", None)  # overlaps 9:00-10:00 (hygienist)
        calendar.book(filling, at(8, 9), "Bo Chen", None)  # the dentist is free at 9
        calendar.book(cleaning, at(8, 10), "Cy Dror", None)  # back to back is fine
        thursday = [s.start for s in calendar.free_slots(cleaning, parse_when("thursday", FROZEN_NOW).windows[0])]
        assert at(8, 9) not in thursday and at(8, 9, 30) not in thursday and at(8, 8, 30) not in thursday
        assert at(8, 8) in thursday

    def test_check_rejects_invalid_times(self, calendar: Calendar, clinic: Clinic) -> None:
        cleaning = service(clinic, "cleaning")
        for bad, reason in [
            (at(8, 9, 15), "slot grid"),
            (at(8, 12), "opening hours"),
            (at(11, 10), "opening hours"),
            (at(6, 10), "too soon"),
        ]:
            with pytest.raises(SlotUnavailableError, match=reason):
                calendar.check(cleaning, bad)

    def test_preferred_time_returns_exact_then_nearest(self, calendar: Calendar, clinic: Clinic) -> None:
        cleaning = service(clinic, "cleaning")
        exact = calendar.find_slots(cleaning, parse_when("thursday at 2pm", FROZEN_NOW))
        assert at(8, 14) in [s.start for s in exact]
        calendar.book(cleaning, at(8, 14), "Ana Lima", None)
        nearest = calendar.find_slots(cleaning, parse_when("thursday at 2pm", FROZEN_NOW))
        assert at(8, 14) not in [s.start for s in nearest]
        assert {s.start for s in nearest} <= {at(8, 13), at(8, 13, 30), at(8, 15), at(8, 15, 30)}

    def test_options_are_spread_out(self, calendar: Calendar, clinic: Clinic) -> None:
        cleaning = service(clinic, "cleaning")
        options = calendar.find_slots(cleaning, parse_when("next tuesday after lunch", FROZEN_NOW))
        # 1 PM and 3 PM are two hours apart; the third option relaxes the spacing to one hour.
        assert [s.start for s in options] == [at(13, 13), at(13, 14), at(13, 15)]
        week = calendar.find_slots(cleaning, parse_when("next week", FROZEN_NOW))
        assert len(week) == 3 and len({s.start.date() for s in week}) == 3

    def test_dst_bookings_store_utc_correctly(self, calendar: Calendar, clinic: Clinic) -> None:
        cleaning = service(clinic, "cleaning")
        before = calendar.book(cleaning, datetime(2026, 10, 30, 9, tzinfo=NY), "Ana Lima", None)
        after = calendar.book(cleaning, datetime(2026, 11, 2, 9, tzinfo=NY), "Bo Chen", None)
        stored_before, stored_after = calendar.get(before.id), calendar.get(after.id)
        assert stored_before is not None and stored_after is not None
        assert stored_before.start == datetime(2026, 10, 30, 13, tzinfo=UTC)
        assert stored_after.start == datetime(2026, 11, 2, 14, tzinfo=UTC)
        assert stored_after.start.astimezone(NY).hour == 9


class TestChanges:
    def test_reschedule_frees_the_old_slot(self, calendar: Calendar, clinic: Clinic) -> None:
        cleaning = service(clinic, "cleaning")
        booked = calendar.book(cleaning, at(8, 9), "Ana Lima", "+15550001111")
        calendar.reschedule(booked.id, at(8, 9, 30))  # overlaps its own old slot only: allowed
        moved = calendar.get(booked.id)
        assert moved is not None and moved.start == at(8, 9, 30).astimezone(UTC)
        with pytest.raises(SlotUnavailableError):
            calendar.book(cleaning, at(8, 10), "Bo Chen", None)

    def test_cancel(self, calendar: Calendar, clinic: Clinic) -> None:
        cleaning = service(clinic, "cleaning")
        booked = calendar.book(cleaning, at(8, 9), "Ana Lima", None)
        calendar.cancel(booked.id)
        with pytest.raises(SlotUnavailableError):
            calendar.cancel(booked.id)
        calendar.book(cleaning, at(8, 9), "Bo Chen", None)  # slot is free again

    def test_find_appointments_by_phone_or_name(self, calendar: Calendar, clinic: Clinic) -> None:
        cleaning = service(clinic, "cleaning")
        booked = calendar.book(cleaning, at(8, 9), "Maria Gonzalez", "+15550001111")
        assert [a.id for a in calendar.find_appointments(phone="(555) 000-1111")] == [booked.id]
        assert [a.id for a in calendar.find_appointments(name="maria gonzalez")] == [booked.id]
        assert [a.id for a in calendar.find_appointments(name="Gonzalez")] == [booked.id]
        assert calendar.find_appointments(name="Maria Smith") == []

    def test_seed_is_deterministic(self, clinic: Clinic, sessions, now) -> None:  # type: ignore[no-untyped-def]
        count = seed_demo_appointments(Calendar(clinic, sessions, now))
        assert count > 20
        from callie.scheduling.db import make_session_factory

        again = seed_demo_appointments(Calendar(clinic, make_session_factory("sqlite://"), now))
        assert again == count


class TestGoogleCalendar:
    def make(self, handler) -> GoogleCalendarClient:  # type: ignore[no-untyped-def]
        return GoogleCalendarClient(
            "clinic@group.calendar.google.com",
            "token",
            timezone="America/New_York",
            transport=httpx.MockTransport(handler),
        )

    def test_busy_times_block_slots(self, clinic: Clinic, sessions, now) -> None:  # type: ignore[no-untyped-def]
        seen: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["authorization"] == "Bearer token"
            assert request.url.path == "/calendar/v3/freeBusy"
            seen.append(json.loads(request.content))
            busy = [{"start": "2026-10-08T13:00:00Z", "end": "2026-10-08T15:00:00Z"}]  # 9-11 AM EDT
            return httpx.Response(200, json={"calendars": {"clinic@group.calendar.google.com": {"busy": busy}}})

        google = self.make(handler)
        calendar = Calendar(clinic, sessions, now, external_busy=google.busy)
        starts = [
            s.start
            for s in calendar.free_slots(clinic.service("cleaning"), parse_when("thursday", FROZEN_NOW).windows[0])
        ]  # type: ignore[arg-type]
        assert at(8, 8) in starts and at(8, 9) not in starts and at(8, 10, 30) not in starts and at(8, 11) in starts
        assert seen and seen[0]["items"] == [{"id": "clinic@group.calendar.google.com"}]

    def test_errors_fail_closed(self) -> None:
        google = self.make(lambda request: httpx.Response(403, json={"error": "forbidden"}))
        with pytest.raises(GoogleCalendarError):
            google.busy(FROZEN_NOW, FROZEN_NOW + timedelta(days=1))

    def test_event_lifecycle(self) -> None:
        calls: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append((request.method, request.url.path))
            if request.method == "POST":
                body = json.loads(request.content)
                assert body["start"]["timeZone"] == "America/New_York"
                return httpx.Response(200, json={"id": "evt123"})
            return httpx.Response(204)

        google = self.make(handler)
        event_id = google.create_event("Cleaning: A. L.", at(8, 9), at(8, 10))
        google.move_event(event_id, at(8, 10), at(8, 11))
        google.delete_event(event_id)
        base = "/calendar/v3/calendars/clinic@group.calendar.google.com/events"
        assert calls == [("POST", base), ("PATCH", f"{base}/evt123"), ("DELETE", f"{base}/evt123")]
