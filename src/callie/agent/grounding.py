"""Grounding check for the hallucinated-availability metric.

Every clock time the agent says is compared with the times that tools returned in this call (offered slots,
existing appointments, opening hours from the knowledge base) and the times the caller said. A time that comes
from none of them, in a sentence that offers or confirms availability, counts as hallucinated availability.
The `say` strings from tools are grounded by construction; this check matters for the model's own sentences.
"""

from __future__ import annotations

import re
from datetime import time

from callie.scheduling.timeparse import normalize, speak_time

_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|\b(\d{1,2}):(\d{2})\b|\b(noon)\b")
_OFFER = re.compile(
    r"\b(available|availability|opening|openings|open slot|free|i have|we have|there's|there is|i can (?:do|fit|offer|book)|"
    r"works|booked|book you|schedule you|how about|would .* work|slot|appointment)\b",
    re.IGNORECASE,
)


def extract_times(text: str) -> set[str]:
    """Clock times mentioned in `text`, normalized to the spoken form ("1:30 PM", "9 AM", "noon")."""
    found: set[str] = set()
    for match in _TIME.finditer(normalize(text)):
        if match.group(6):
            found.add("noon")
            continue
        hour = int(match.group(1) or match.group(4))
        minute = int(match.group(2) or match.group(5) or 0)
        meridiem = match.group(3)
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        elif meridiem is None and 1 <= hour <= 6:
            hour += 12  # "at 2:30" in business hours
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            found.add(speak_time(time(hour, minute)))
    return found


def unsupported_times(sentence: str, known: set[str]) -> list[str]:
    """Times in an availability-offering sentence that no tool and no caller mentioned."""
    if not _OFFER.search(sentence):
        return []
    return sorted(t for t in extract_times(sentence) if t not in known)
