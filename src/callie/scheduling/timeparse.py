"""Deterministic normalization of spoken dates and times ("next Tuesday after lunch") in the clinic's timezone.

The LLM never computes dates. It passes the caller's words to `check_availability(when=...)`, this module turns
them into concrete windows in the clinic's timezone, and the tool result says how they were understood
("Tuesday, October 13, after 1 PM"), so the read-back catches a misunderstanding before anything is booked.

Conventions (each one is reported as an assumption when it was applied):
- a bare weekday ("Tuesday") is the next one after today; "this Tuesday" may be today;
- "next Tuesday" is the Tuesday of next calendar week (Monday-based); when that differs from the nearest
  Tuesday the result is flagged ambiguous;
- an hour without am/pm is read in business hours: 7-11 -> morning, 12-6 -> afternoon;
- a date without a year is the next such date; "the 13th" is this month's if it has not passed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
_MONTH_ABBR = {m[:3]: i + 1 for i, m in enumerate(MONTHS)} | {"sept": 9}

_UNITS = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
}
_SMALL_COUNTS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "couple": 2, "few": 3}

# Parts of the day as [start, end) local times.
PARTS_OF_DAY: dict[str, tuple[time, time]] = {
    "first thing": (time(7, 0), time(9, 30)),
    "early morning": (time(7, 0), time(10, 0)),
    "late morning": (time(10, 0), time(12, 0)),
    "before lunch": (time(7, 0), time(12, 0)),
    "morning": (time(7, 0), time(12, 0)),
    "around lunch": (time(11, 30), time(14, 0)),
    "lunchtime": (time(12, 0), time(13, 0)),
    "lunch time": (time(12, 0), time(13, 0)),
    "after lunch": (time(13, 0), time(18, 0)),
    "early afternoon": (time(12, 0), time(15, 0)),
    "late afternoon": (time(15, 0), time(18, 0)),
    "afternoon": (time(12, 0), time(18, 0)),
    "end of the day": (time(15, 0), time(18, 0)),
    "after work": (time(16, 30), time(21, 0)),
    "after school": (time(15, 0), time(18, 0)),
    "evening": (time(17, 0), time(21, 0)),
    "tonight": (time(17, 0), time(21, 0)),
    "midday": (time(11, 30), time(13, 30)),
}
_PART_PATTERN = "|".join(sorted((re.escape(p) for p in PARTS_OF_DAY), key=len, reverse=True))


@dataclass(frozen=True)
class TimeWindow:
    start: datetime
    end: datetime
    preferred: datetime | None = None  # a specific time the caller asked for ("at 3:30")
    label: str = ""  # the time part in words: "", "after 3 PM", "1 PM to 6 PM", "at 2:30 PM"

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end


@dataclass
class ParsedWhen:
    text: str
    windows: list[TimeWindow] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    ambiguous: bool = False
    asap: bool = False

    @property
    def understood(self) -> bool:
        return bool(self.windows)

    @property
    def days(self) -> list[date]:
        seen: list[date] = []
        for window in self.windows:
            if window.start.date() not in seen:
                seen.append(window.start.date())
        return seen

    def describe(self) -> str:
        """How the request was understood, in words the agent can read back."""
        if not self.windows:
            return "not understood"
        if self.asap:
            return "the earliest available time"
        groups: list[tuple[list[date], str]] = []
        for window in self.windows:
            day = window.start.date()
            if groups and groups[-1][1] == window.label and (day - groups[-1][0][-1]).days == 1:
                groups[-1][0].append(day)
            else:
                groups.append(([day], window.label))
        parts = []
        for days, label in groups:
            text = speak_date(days[0]) if len(days) < 3 else f"{speak_date(days[0])} through {speak_date(days[-1])}"
            if len(days) == 2:
                text = f"{speak_date(days[0])} or {speak_date(days[1])}"
            parts.append(f"{text}, {label}" if label else text)
        return "; or ".join(parts)


# ---------------------------------------------------------------------------------------------------------------
# Spoken output
# ---------------------------------------------------------------------------------------------------------------


def ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def speak_date(day: date) -> str:
    return f"{WEEKDAYS[day.weekday()].capitalize()}, {MONTHS[day.month - 1].capitalize()} {ordinal(day.day)}"


def speak_time(moment: time) -> str:
    if moment == time(12, 0):
        return "noon"
    hour = moment.hour % 12 or 12
    suffix = "AM" if moment.hour < 12 else "PM"
    return f"{hour} {suffix}" if moment.minute == 0 else f"{hour}:{moment.minute:02d} {suffix}"


def speak_slot(moment: datetime) -> str:
    return f"{speak_date(moment.date())} at {speak_time(moment.time())}"


# ---------------------------------------------------------------------------------------------------------------
# Text normalization: number words -> digits, "half past three" -> "3:30", "1 p.m." -> "1pm"
# ---------------------------------------------------------------------------------------------------------------


def _number_word(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return _UNITS.get(token, _TENS.get(token))


def _compound(tokens: list[str], i: int) -> tuple[int | None, int]:
    """Parse 'twenty five' / 'forty' / 'seven' at tokens[i]; returns (value, tokens consumed)."""
    if i >= len(tokens):
        return None, 0
    if tokens[i] in _TENS:
        value = _TENS[tokens[i]]
        if i + 1 < len(tokens) and tokens[i + 1] in _UNITS and 0 < _UNITS[tokens[i + 1]] < 10:
            return value + _UNITS[tokens[i + 1]], 2
        return value, 1
    if tokens[i] in _UNITS:
        return _UNITS[tokens[i]], 1
    return None, 0


def _words_to_times(text: str) -> str:
    text = re.sub(r"\b(a\s+)?quarter\s+(past|after)\s+(\w+)", lambda m: _rel_time(m.group(3), 15), text)
    text = re.sub(r"\b(a\s+)?quarter\s+(to|till|til|of)\s+(\w+)", lambda m: _rel_time(m.group(3), -15), text)
    text = re.sub(r"\bhalf\s+(past|after)\s+(\w+)", lambda m: _rel_time(m.group(2), 30), text)
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        hour = _UNITS.get(token)
        if hour is not None and 1 <= hour <= 12 and token != "oh":
            # "two thirty", "nine oh five", "one forty five", "eleven o'clock"
            if i + 1 < len(tokens) and tokens[i + 1] in {"o'clock", "oclock"}:
                out.append(str(hour))
                i += 2
                continue
            if i + 2 < len(tokens) and tokens[i + 1] == "oh" and tokens[i + 2] in _UNITS:
                minute = _UNITS[tokens[i + 2]]
                if minute < 10:
                    out.append(f"{hour}:{minute:02d}")
                    i += 3
                    continue
            compound, used = _compound(tokens, i + 1)
            if compound is not None and used and 10 <= compound <= 59 and tokens[i + 1] in {*_TENS, "fifteen"}:
                out.append(f"{hour}:{compound:02d}")
                i += 1 + used
                continue
            out.append(str(hour))
            i += 1
            continue
        value, used = _compound(tokens, i)
        if value is not None and used and token not in {"oh", "zero"}:
            out.append(str(value))
            i += used
            continue
        out.append(token)
        i += 1
    return " ".join(out)


def _rel_time(hour_word: str, minutes: int) -> str:
    hour = _number_word(hour_word)
    if hour is None or not 1 <= hour <= 12:
        return hour_word
    if minutes < 0:
        hour = hour - 1 or 12
        minutes += 60
    return f"{hour}:{minutes:02d}"


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"\b([ap])\.\s?m\.?", r"\1m", text)
    text = re.sub(r"[,;!?]", " ", text)
    text = re.sub(r"\.(?!\d)", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", text)
    text = re.sub(r"\b(o'clock|oclock)\b", " o'clock", text)
    text = re.sub(r"\b12 ?noon\b|\bnoon\b|\bmidday time\b", "12:00pm", text)
    text = re.sub(
        r"\btwenty (first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)\b",
        lambda m: str(20 + _ORDINALS[m.group(1)]),
        text,
    )
    text = re.sub(r"\bthirty first\b", "31", text)
    # Ordinals only next to a month or after "the" ("the thirteenth"), so "a second" stays a word.
    month_names = "|".join(MONTHS + list(_MONTH_ABBR))
    ordinal_words = "|".join(_ORDINALS)
    text = re.sub(
        rf"\b(the|{month_names})\s+({ordinal_words})\b(?!\s+(available|opening|thing|one|time|slot|appointment))",
        lambda m: f"{m.group(1)} {_ORDINALS[m.group(2)]}",
        text,
    )
    text = re.sub(
        rf"\b({ordinal_words})\s+of\s+({month_names})\b", lambda m: f"{_ORDINALS[m.group(1)]} of {m.group(2)}", text
    )
    text = _words_to_times(" ".join(text.split()))
    text = re.sub(
        r"\b(\d{1,2})(?::(\d{2}))?\s+(am|pm)\b",
        lambda m: f"{m.group(1)}{':' + m.group(2) if m.group(2) else ''}{m.group(3)}",
        text,
    )
    return " ".join(text.split())


# ---------------------------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------------------------

_TIME_TOKEN = r"(\d{1,2})(?::(\d{2}))?(am|pm)?"


@dataclass
class _Segment:
    days: list[date] = field(default_factory=list)
    ranges: list[tuple[time, time]] = field(default_factory=list)
    preferred: time | None = None


class _Parser:
    def __init__(self, now: datetime, result: ParsedWhen) -> None:
        self.now = now
        self.today = now.date()
        self.result = result

    def assume(self, note: str) -> None:
        if note not in self.result.assumptions:
            self.result.assumptions.append(note)

    # -- days ------------------------------------------------------------------------------------------------
    def weekday_date(self, name: str, modifier: str | None) -> date:
        target = WEEKDAYS.index(name)
        ahead = (target - self.today.weekday()) % 7
        nearest_inclusive = self.today + timedelta(days=ahead)
        nearest_future = nearest_inclusive if ahead else self.today + timedelta(days=7)
        if modifier in {"this", "coming", "this coming"}:
            return nearest_inclusive
        if modifier in {"next", "the following"}:
            next_monday = self.today + timedelta(days=7 - self.today.weekday())
            chosen = next_monday + timedelta(days=target)
            if chosen != nearest_future:
                self.result.ambiguous = True
                self.assume(f'"next {name}" read as {speak_date(chosen)} (next week), not {speak_date(nearest_future)}')
            return chosen
        if modifier == "after next":
            return nearest_future + timedelta(days=7)
        if ahead == 0:
            self.assume(f'"{name}" read as next {name}, {speak_date(nearest_future)}, not today')
        return nearest_future

    def month_day(self, month: int, day: int) -> date | None:
        for year in (self.today.year, self.today.year + 1):
            try:
                candidate = date(year, month, day)
            except ValueError:
                return None
            if candidate >= self.today:
                return candidate
        return None

    def parse_days(self, text: str) -> list[date]:
        days: list[date] = []
        span: list[date] = []

        def add(d: date | None) -> None:
            if d is not None and d not in days:
                days.append(d)

        if re.search(
            r"\b(as soon as possible|asap|earliest|soonest|first available|first opening|next available)\b", text
        ):
            self.result.asap = True
            span = [self.today + timedelta(days=i) for i in range(14)]
        if re.search(r"\bday after tomorrow\b", text):
            add(self.today + timedelta(days=2))
            text = text.replace("day after tomorrow", " ")
        for match in re.finditer(r"\ba week from (today|tomorrow|" + "|".join(WEEKDAYS) + r")\b", text):
            anchor = match.group(1)
            base = (
                self.today
                if anchor == "today"
                else self.today + timedelta(days=1)
                if anchor == "tomorrow"
                else self.weekday_date(anchor, None)
            )
            add(base + timedelta(days=7))
            text = text.replace(match.group(0), " ")
        explicit: list[date] = []

        def add_explicit(d: date | None) -> None:
            if d is not None and d not in explicit:
                explicit.append(d)

        month_names = "|".join(MONTHS + list(_MONTH_ABBR))
        for match in re.finditer(rf"\b({month_names})\s+(\d{{1,2}})\b(?!\s*(?::|am|pm))", text):
            add_explicit(self.month_day(_month_number(match.group(1)), int(match.group(2))))
        for match in re.finditer(rf"\b(\d{{1,2}})\s+(?:of\s+)?({month_names})\b", text):
            add_explicit(self.month_day(_month_number(match.group(2)), int(match.group(1))))
        for match in re.finditer(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", text):
            add_explicit(self.month_day(int(match.group(1)), int(match.group(2))))
            self.assume(f"{match.group(0)} read as month/day")
        if not explicit:
            for match in re.finditer(r"\bthe (\d{1,2})\b(?!\s*(?::|am|pm|o'clock))", text):
                day_number = int(match.group(1))
                if 1 <= day_number <= 31:
                    candidate = None
                    month_start = self.today.replace(day=1)
                    for _ in range(3):  # this month, else the next month that has that day
                        try:
                            candidate = month_start.replace(day=day_number)
                        except ValueError:
                            candidate = None
                        if candidate is not None and candidate >= self.today:
                            break
                        candidate = None
                        month_start = (month_start + timedelta(days=32)).replace(day=1)
                    add_explicit(candidate)
        weekday_names = re.findall(r"\b(" + "|".join(WEEKDAYS) + r")\b", text)
        if explicit:
            for d in explicit:
                add(d)
            for name in weekday_names:
                if all(WEEKDAYS[d.weekday()] != name for d in explicit):
                    self.result.ambiguous = True
                    self.assume(
                        f"{name} does not match {', '.join(speak_date(d) for d in explicit)}; the date was used"
                    )
            weekday_matches: list[re.Match[str]] = []
        else:
            weekday_re = (
                r"\b(this coming|this|next|coming|the following)?\s*(" + "|".join(WEEKDAYS) + r")s?(\s+after next)?\b"
            )
            weekday_matches = list(re.finditer(weekday_re, text))
            range_match = re.search(
                r"\b(?:between|from)?\s*("
                + "|".join(WEEKDAYS)
                + r")\s+(?:through|thru|to|and|until|till)\s+("
                + "|".join(WEEKDAYS)
                + r")\b",
                text,
            )
            if range_match and (
                "between" in text or "through" in text or "thru" in text or " to " in range_match.group(0)
            ):
                modifier = "next" if re.search(r"\bnext\s+" + range_match.group(1), text) else None
                first = self.weekday_date(range_match.group(1), modifier)
                last_index = WEEKDAYS.index(range_match.group(2))
                cursor = first
                while cursor.weekday() != last_index and len(span) < 7:
                    span.append(cursor)
                    cursor += timedelta(days=1)
                span.append(cursor)
            else:
                for match in weekday_matches:
                    modifier = match.group(1)
                    if match.group(3):
                        modifier = "after next"
                    add(self.weekday_date(match.group(2), modifier))
        if re.search(r"\b(today|tonight|this (morning|afternoon|evening))\b", text):
            add(self.today)
        if re.search(r"\btomorrow\b", text):
            add(self.today + timedelta(days=1))
        for match in re.finditer(
            r"\bin (a|an|one|two|three|four|a couple of|couple of|a few|few|\d+) (day|week)s?\b", text
        ):
            count_word = (
                match.group(1).replace("a couple of", "couple").replace("couple of", "couple").replace("a few", "few")
            )
            count = int(count_word) if count_word.isdigit() else _SMALL_COUNTS.get(count_word, 1)
            add(self.today + timedelta(days=count * (7 if match.group(2) == "week" else 1)))
        if re.search(r"\b(this|the) weekend\b|\bweekend\b", text):
            saturday = self.today + timedelta(days=(5 - self.today.weekday()) % 7)
            if "next weekend" in text:
                saturday = self.today + timedelta(days=7 - self.today.weekday() + 5)
            span.extend([saturday, saturday + timedelta(days=1)])
        elif re.search(r"\bearly next week\b", text):
            monday = self.today + timedelta(days=7 - self.today.weekday())
            span.extend([monday, monday + timedelta(days=1)])
        elif re.search(r"\b(end|later part) of next week\b", text):
            monday = self.today + timedelta(days=7 - self.today.weekday())
            span.extend([monday + timedelta(days=3), monday + timedelta(days=4)])
        elif re.search(r"\bnext week\b", text) and not weekday_matches:
            monday = self.today + timedelta(days=7 - self.today.weekday())
            span.extend(monday + timedelta(days=i) for i in range(6))
        elif re.search(r"\b(end of (the|this) week|later this week|rest of (the|this) week|this week)\b", text):
            start = (
                self.today
                if "end of" not in text
                else max(self.today, self.today + timedelta(days=3 - self.today.weekday()))
            )
            saturday = self.today + timedelta(days=(5 - self.today.weekday()) % 7)
            cursor = start
            while cursor <= saturday:
                span.append(cursor)
                cursor += timedelta(days=1)
        for d in span:
            add(d)
        return sorted(days)

    # -- times -----------------------------------------------------------------------------------------------
    def to_time(self, hour: int, minute: int, meridiem: str | None, context: str) -> time | None:
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        elif meridiem is None and hour <= 12:
            if re.search(r"\b(morning|before lunch)\b", context) and hour < 12:
                pass
            elif re.search(r"\b(afternoon|evening|tonight|after lunch|after work)\b", context) and hour < 12:
                hour += 12
            elif 1 <= hour <= 6:
                hour += 12
                self.assume(f"{hour - 12} o'clock read as {hour - 12} PM")
            elif hour == 12:
                pass
            else:
                self.assume(f"{hour} o'clock read as {hour} AM")
        return time(hour, minute)

    def parse_times(self, text: str, segment: _Segment) -> None:
        for part in sorted(PARTS_OF_DAY, key=len, reverse=True):
            if re.search(rf"\b{re.escape(part)}\b", text):
                segment.ranges.append(PARTS_OF_DAY[part])
                text = re.sub(rf"\b{re.escape(part)}\b", " ", text)
        between = re.search(rf"\b(?:between|from)\s+{_TIME_TOKEN}\s+(?:and|to|till|until)\s+{_TIME_TOKEN}\b", text)
        if between:
            end_meridiem = between.group(6)
            start_t = self.to_time(
                int(between.group(1)), int(between.group(2) or 0), between.group(3) or end_meridiem, text
            )
            end_t = self.to_time(int(between.group(4)), int(between.group(5) or 0), end_meridiem, text)
            if start_t and end_t and end_t > start_t:
                segment.ranges = [(start_t, end_t)]
            return
        after = re.search(rf"\b(?:after|later than|from)\s+{_TIME_TOKEN}\b", text)
        before = re.search(rf"\b(?:before|by|no later than|earlier than)\s+{_TIME_TOKEN}\b", text)
        if after or before:
            low = time(0, 0)
            high = time(23, 59)
            if after:
                parsed = self.to_time(int(after.group(1)), int(after.group(2) or 0), after.group(3), text)
                low = parsed or low
            if before:
                parsed = self.to_time(int(before.group(1)), int(before.group(2) or 0), before.group(3), text)
                high = parsed or high
            if segment.ranges:
                part_start, part_end = segment.ranges[0]
                low, high = max(low, part_start), min(high, part_end)
            if high > low:
                segment.ranges = [(low, high)]
            return
        cue = r"(?:at|around|about|say|how about|maybe|like|ideally)"
        for match in re.finditer(rf"(?:\b({cue})\s+)?\b{_TIME_TOKEN}(\s+o'clock)?(?=\s|$)", text):
            has_cue, minutes, meridiem, oclock = match.group(1), match.group(3), match.group(4), match.group(5)
            if not (has_cue or minutes or meridiem or oclock):
                continue  # a bare number ("the 13", "in 2 days") is not a time
            parsed = self.to_time(int(match.group(2)), int(minutes or 0), meridiem, text)
            if parsed is not None:
                segment.preferred = parsed
                return

    def build(self, segments: list[_Segment]) -> None:
        tz = self.now.tzinfo
        # Segments with days but no times borrow the next (or previous) segment's times, and vice versa.
        for index, segment in enumerate(segments):
            if segment.days and not segment.ranges and segment.preferred is None:
                donor = next((s for s in segments[index + 1 :] if s.ranges or s.preferred), None) or next(
                    (s for s in reversed(segments[:index]) if s.ranges or s.preferred), None
                )
                if donor:
                    segment.ranges = list(donor.ranges)
                    segment.preferred = donor.preferred
            if not segment.days and (segment.ranges or segment.preferred):
                donor = next((s for s in reversed(segments[:index]) if s.days), None)
                if donor:
                    segment.days = list(donor.days)
        if all(not s.days for s in segments) and any(s.ranges or s.preferred for s in segments):
            # Only a time ("in the morning", "at 3"): the next time it occurs.
            for segment in segments:
                latest = max([end for _, end in segment.ranges] + ([segment.preferred] if segment.preferred else []))
                day = self.today if latest > self.now.time() else self.today + timedelta(days=1)
                segment.days = [day]
                self.assume(f"no day given, read as {'today' if day == self.today else 'tomorrow'}")
        for segment in segments:
            for day in segment.days:
                if segment.preferred is not None:
                    preferred = datetime.combine(day, segment.preferred, tzinfo=tz)
                    start = datetime.combine(day, time(0, 0), tzinfo=tz)
                    label = f"at {speak_time(segment.preferred)}"
                    self.result.windows.append(TimeWindow(start, start + timedelta(days=1), preferred, label))
                elif segment.ranges:
                    for low, high in segment.ranges:
                        self.result.windows.append(
                            TimeWindow(
                                datetime.combine(day, low, tzinfo=tz),
                                datetime.combine(day, high, tzinfo=tz),
                                label=_range_label(low, high),
                            )
                        )
                else:
                    start = datetime.combine(day, time(0, 0), tzinfo=tz)
                    self.result.windows.append(
                        TimeWindow(start, datetime.combine(day + timedelta(days=1), time(0, 0), tzinfo=tz))
                    )
        # Never in the past.
        self.result.windows = [
            TimeWindow(max(w.start, self.now), w.end, w.preferred, w.label)
            for w in self.result.windows
            if w.end > self.now
        ]


def _range_label(low: time, high: time) -> str:
    if high >= time(23, 59):
        return f"after {speak_time(low)}"
    if low == time(0, 0):
        return f"before {speak_time(high)}"
    return f"{speak_time(low)} to {speak_time(high)}"


def _month_number(name: str) -> int:
    if name in MONTHS:
        return MONTHS.index(name) + 1
    return _MONTH_ABBR[name[:4] if name.startswith("sept") else name[:3]]


def parse_when(text: str, now: datetime) -> ParsedWhen:
    """Turn a spoken date/time phrase into windows in `now`'s timezone (`now` must be timezone-aware)."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware (the clinic's timezone)")
    result = ParsedWhen(text=text)
    parser = _Parser(now, result)
    normalized = normalize(text)
    pieces = [p for p in re.split(r"\s+or\s+|\s+otherwise\s+|\s+and also\s+", normalized) if p.strip()]
    # "between monday and wednesday" / "between 2 and 4" must not be split; neither is "or" splitting needed then.
    if re.search(r"\b(between|through|thru)\b", normalized):
        pieces = [normalized]
    segments: list[_Segment] = []
    for piece in pieces:
        segment = _Segment(days=parser.parse_days(piece))
        parser.parse_times(piece, segment)
        segments.append(segment)
    parser.build(segments)
    if result.asap and not result.windows:
        start = now
        result.windows = [TimeWindow(start, start + timedelta(days=14))]
    return result


def clinic_now(tz_name: str, frozen: str | None = None) -> datetime:
    tz = ZoneInfo(tz_name)
    if frozen:
        parsed = datetime.fromisoformat(frozen)
        return parsed.replace(tzinfo=tz) if parsed.tzinfo is None else parsed.astimezone(tz)
    return datetime.now(tz)
