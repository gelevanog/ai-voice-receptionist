"""Spoken date/time normalization: relative days, weekdays, explicit dates, parts of day, timezones and DST."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from callie.scheduling.timeparse import clinic_now, normalize, parse_when, speak_date, speak_slot, speak_time

NY = ZoneInfo("America/New_York")
# Tuesday, October 6, 2026, 9:30 AM in the clinic's timezone.
NOW = datetime(2026, 10, 6, 9, 30, tzinfo=NY)


def days_of(text: str, now: datetime = NOW) -> list[date]:
    return parse_when(text, now).days


def single(text: str, now: datetime = NOW) -> tuple[date, time, time]:
    parsed = parse_when(text, now)
    assert len(parsed.windows) == 1, parsed.windows
    window = parsed.windows[0]
    return window.start.date(), window.start.timetz().replace(tzinfo=None), window.end.timetz().replace(tzinfo=None)


def preferred(text: str, now: datetime = NOW) -> datetime:
    parsed = parse_when(text, now)
    assert parsed.windows and parsed.windows[0].preferred is not None, parsed
    return parsed.windows[0].preferred


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("today", [date(2026, 10, 6)]),
        ("tomorrow", [date(2026, 10, 7)]),
        ("the day after tomorrow", [date(2026, 10, 8)]),
        ("Thursday", [date(2026, 10, 8)]),
        ("this Thursday", [date(2026, 10, 8)]),
        ("this Tuesday", [date(2026, 10, 6)]),  # "this" may be today
        ("Tuesday", [date(2026, 10, 13)]),  # a bare weekday is never today
        ("next Tuesday", [date(2026, 10, 13)]),
        ("next Thursday", [date(2026, 10, 15)]),  # next calendar week
        ("Thursday after next", [date(2026, 10, 15)]),
        ("a week from Friday", [date(2026, 10, 16)]),
        ("a week from today", [date(2026, 10, 13)]),
        ("in two weeks", [date(2026, 10, 20)]),
        ("in 3 days", [date(2026, 10, 9)]),
        ("in a couple of days", [date(2026, 10, 8)]),
        ("October 20th", [date(2026, 10, 20)]),
        ("the 20th of October", [date(2026, 10, 20)]),
        ("Oct 21", [date(2026, 10, 21)]),
        ("twenty first of october", [date(2026, 10, 21)]),
        ("november third", [date(2026, 11, 3)]),
        ("the 13th", [date(2026, 10, 13)]),
        ("the 2nd", [date(2026, 11, 2)]),  # already passed this month -> next month
        ("10/15", [date(2026, 10, 15)]),
        ("January 5", [date(2027, 1, 5)]),
        ("October 1", [date(2027, 10, 1)]),  # passed this year -> next year
        ("Wednesday the fourteenth", [date(2026, 10, 14)]),
        ("early next week", [date(2026, 10, 12), date(2026, 10, 13)]),
        ("this weekend", [date(2026, 10, 10), date(2026, 10, 11)]),
        ("Monday or Tuesday", [date(2026, 10, 12), date(2026, 10, 13)]),
        ("between Monday and Wednesday", [date(2026, 10, 12), date(2026, 10, 13), date(2026, 10, 14)]),
    ],
)
def test_days(text: str, expected: list[date]) -> None:
    assert days_of(text) == expected


def test_next_week_spans_monday_to_saturday() -> None:
    assert days_of("sometime next week") == [date(2026, 10, 12) + timedelta(days=i) for i in range(6)]


def test_this_week_and_end_of_week() -> None:
    assert days_of("later this week")[0] == date(2026, 10, 6)
    assert days_of("later this week")[-1] == date(2026, 10, 10)
    assert days_of("end of the week") == [date(2026, 10, 8), date(2026, 10, 9), date(2026, 10, 10)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Thursday at 3", datetime(2026, 10, 8, 15, 0, tzinfo=NY)),
        ("Thursday at 3pm", datetime(2026, 10, 8, 15, 0, tzinfo=NY)),
        ("Thursday at 3 p.m.", datetime(2026, 10, 8, 15, 0, tzinfo=NY)),
        ("Thursday at 9", datetime(2026, 10, 8, 9, 0, tzinfo=NY)),
        ("Thursday at 10:30", datetime(2026, 10, 8, 10, 30, tzinfo=NY)),
        ("Thursday at two thirty", datetime(2026, 10, 8, 14, 30, tzinfo=NY)),
        ("Thursday at ten fifteen", datetime(2026, 10, 8, 10, 15, tzinfo=NY)),
        ("Thursday at one forty five", datetime(2026, 10, 8, 13, 45, tzinfo=NY)),
        ("Thursday at nine oh five", datetime(2026, 10, 8, 9, 5, tzinfo=NY)),
        ("half past nine on Thursday", datetime(2026, 10, 8, 9, 30, tzinfo=NY)),
        ("quarter to four tomorrow", datetime(2026, 10, 7, 15, 45, tzinfo=NY)),
        ("a quarter past ten tomorrow", datetime(2026, 10, 7, 10, 15, tzinfo=NY)),
        ("tomorrow at noon", datetime(2026, 10, 7, 12, 0, tzinfo=NY)),
        ("tomorrow at eleven o'clock", datetime(2026, 10, 7, 11, 0, tzinfo=NY)),
        ("tomorrow evening at 6", datetime(2026, 10, 7, 18, 0, tzinfo=NY)),
        ("tomorrow morning at 8", datetime(2026, 10, 7, 8, 0, tzinfo=NY)),
        ("the 13th at 3pm", datetime(2026, 10, 13, 15, 0, tzinfo=NY)),
        ("10/15 at 9am", datetime(2026, 10, 15, 9, 0, tzinfo=NY)),
        ("Friday around 10", datetime(2026, 10, 9, 10, 0, tzinfo=NY)),
    ],
)
def test_exact_times(text: str, expected: datetime) -> None:
    assert preferred(text) == expected


@pytest.mark.parametrize(
    ("text", "day", "start", "end"),
    [
        ("next Tuesday after lunch", date(2026, 10, 13), time(13, 0), time(18, 0)),
        ("tomorrow morning", date(2026, 10, 7), time(7, 0), time(12, 0)),
        ("tomorrow afternoon", date(2026, 10, 7), time(12, 0), time(18, 0)),
        ("Thursday late afternoon", date(2026, 10, 8), time(15, 0), time(18, 0)),
        ("Thursday early morning", date(2026, 10, 8), time(7, 0), time(10, 0)),
        ("first thing Thursday", date(2026, 10, 8), time(7, 0), time(9, 30)),
        ("Thursday before lunch", date(2026, 10, 8), time(7, 0), time(12, 0)),
        ("Friday after 3", date(2026, 10, 9), time(15, 0), time(23, 59)),
        ("I am free after 3 on friday", date(2026, 10, 9), time(15, 0), time(23, 59)),
        ("tomorrow before 11", date(2026, 10, 7), time(0, 0), time(11, 0)),
        ("between 2 and 4 on Wednesday", date(2026, 10, 7), time(14, 0), time(16, 0)),
        ("Wednesday from 10am to noon", date(2026, 10, 7), time(10, 0), time(12, 0)),
        ("Thursday afternoon after 3", date(2026, 10, 8), time(15, 0), time(18, 0)),
    ],
)
def test_parts_of_day_and_ranges(text: str, day: date, start: time, end: time) -> None:
    assert single(text) == (day, start, end)


def test_or_lists_share_the_time_of_day() -> None:
    parsed = parse_when("Monday or Tuesday afternoon", NOW)
    assert [(w.start.date(), w.start.hour, w.end.hour) for w in parsed.windows] == [
        (date(2026, 10, 12), 12, 18),
        (date(2026, 10, 13), 12, 18),
    ]


def test_or_lists_with_their_own_times() -> None:
    parsed = parse_when("Thursday morning or Friday after 2", NOW)
    assert [(w.start.date(), w.start.hour) for w in parsed.windows] == [
        (date(2026, 10, 8), 7),
        (date(2026, 10, 9), 14),
    ]


def test_time_without_day_is_the_next_occurrence() -> None:
    assert single("in the afternoon")[0] == date(2026, 10, 6)
    late = datetime(2026, 10, 6, 18, 30, tzinfo=NY)
    assert single("in the morning", late)[0] == date(2026, 10, 7)
    parsed = parse_when("at 3", NOW)
    assert parsed.windows[0].preferred == datetime(2026, 10, 6, 15, 0, tzinfo=NY)
    assert any("no day given" in a for a in parsed.assumptions)


def test_windows_never_start_in_the_past() -> None:
    parsed = parse_when("today", NOW)
    assert parsed.windows[0].start == NOW
    assert parse_when("this morning", datetime(2026, 10, 6, 13, 0, tzinfo=NY)).windows == []


def test_asap() -> None:
    parsed = parse_when("as soon as possible", NOW)
    assert parsed.asap and parsed.windows[0].start == NOW
    assert parsed.describe() == "the earliest available time"
    assert parse_when("whatever the first available is", NOW).asap


def test_ambiguity_is_reported() -> None:
    monday = datetime(2026, 10, 5, 9, 0, tzinfo=NY)
    parsed = parse_when("next Tuesday", monday)
    assert parsed.days == [date(2026, 10, 13)]
    assert parsed.ambiguous
    assert "Tuesday, October 6th" in parsed.assumptions[0]
    mismatch = parse_when("Thursday the fourteenth", NOW)
    assert mismatch.days == [date(2026, 10, 14)] and mismatch.ambiguous


def test_am_pm_assumptions_are_recorded() -> None:
    assert any("3 PM" in a for a in parse_when("Thursday at 3", NOW).assumptions)
    assert parse_when("Thursday at 3pm", NOW).assumptions == []


def test_not_understood() -> None:
    for text in ["whenever you guys are free I guess", "hmm", ""]:
        parsed = parse_when(text, NOW)
        assert not parsed.understood
        assert parsed.describe() == "not understood"


def test_describe_reads_naturally() -> None:
    assert parse_when("next Tuesday after lunch", NOW).describe() == "Tuesday, October 13th, 1 PM to 6 PM"
    assert parse_when("Friday after 3", NOW).describe() == "Friday, October 9th, after 3 PM"
    assert parse_when("tomorrow before 11", NOW).describe() == "Wednesday, October 7th, before 11 AM"
    assert parse_when("next week", NOW).describe() == "Monday, October 12th through Saturday, October 17th"
    assert parse_when("Thursday at 2:30", NOW).describe() == "Thursday, October 8th, at 2:30 PM"


def test_needs_an_aware_now() -> None:
    with pytest.raises(ValueError):
        parse_when("tomorrow", datetime(2026, 10, 6, 9, 30))


class TestTimezonesAndDst:
    """US daylight saving time ends on Sunday, November 1, 2026 (2 AM EDT -> 1 AM EST)."""

    def test_same_wall_clock_time_across_the_dst_change(self) -> None:
        before = preferred("Friday at 9am", datetime(2026, 10, 27, 10, 0, tzinfo=NY))
        after = preferred("Monday at 9am", datetime(2026, 10, 30, 10, 0, tzinfo=NY))
        assert before.astimezone(UTC).hour == 13  # EDT, UTC-4
        assert after.astimezone(UTC).hour == 14  # EST, UTC-5
        assert (before.hour, after.hour) == (9, 9)

    def test_in_a_week_keeps_wall_clock_time_not_168_hours(self) -> None:
        now = datetime(2026, 10, 29, 10, 0, tzinfo=NY)
        parsed = parse_when("in a week at 10am", now)
        moment = parsed.windows[0].preferred
        assert moment is not None
        assert moment.date() == date(2026, 11, 5) and moment.hour == 10
        assert (moment - now) == timedelta(days=7)  # aware arithmetic on the same tz: wall clock
        assert (moment.astimezone(UTC) - now.astimezone(UTC)) == timedelta(days=7, hours=1)

    def test_other_timezones(self) -> None:
        la = ZoneInfo("America/Los_Angeles")
        now = datetime(2026, 10, 6, 22, 30, tzinfo=la)  # already Wednesday in UTC
        assert days_of("tomorrow", now) == [date(2026, 10, 7)]
        assert preferred("tomorrow at 9am", now).utcoffset() == timedelta(hours=-7)
        berlin = ZoneInfo("Europe/Berlin")  # EU DST ends a week earlier: October 25, 2026
        moment = preferred("next Monday at 9am", datetime(2026, 10, 22, 12, 0, tzinfo=berlin))
        assert moment.utcoffset() == timedelta(hours=1)

    def test_clinic_now(self) -> None:
        frozen = clinic_now("America/New_York", "2026-10-06T09:30")
        assert frozen == NOW and frozen.tzinfo is not None
        assert clinic_now("America/New_York").tzinfo is not None


def test_normalize_number_words() -> None:
    assert normalize("Two thirty P.M.") == "2:30pm"
    assert normalize("noon") == "12:00pm"
    assert normalize("the twenty first") == "the 21"
    assert normalize("I need a second opinion") == "i need a second opinion"


def test_spoken_formats() -> None:
    assert speak_date(date(2026, 10, 13)) == "Tuesday, October 13th"
    assert speak_date(date(2026, 10, 22)) == "Thursday, October 22nd"
    assert speak_date(date(2026, 10, 11)) == "Sunday, October 11th"
    assert speak_time(time(13, 30)) == "1:30 PM"
    assert speak_time(time(9, 0)) == "9 AM"
    assert speak_time(time(12, 0)) == "noon"
    assert speak_slot(datetime(2026, 10, 13, 15, 0, tzinfo=NY)) == "Tuesday, October 13th at 3 PM"
