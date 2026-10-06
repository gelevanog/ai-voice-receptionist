from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.orm import Session, sessionmaker

from callie.clinic import Clinic, load_clinic
from callie.scheduling.calendar import Calendar
from callie.scheduling.db import make_session_factory

NY = ZoneInfo("America/New_York")
# Tuesday, October 6, 2026, 9:30 AM: the frozen clinic clock used across the tests.
FROZEN_NOW = datetime(2026, 10, 6, 9, 30, tzinfo=NY)


@pytest.fixture
def clinic() -> Clinic:
    return load_clinic()


@pytest.fixture
def sessions() -> sessionmaker[Session]:
    return make_session_factory("sqlite://")


@pytest.fixture
def now() -> Callable[[], datetime]:
    return lambda: FROZEN_NOW


@pytest.fixture
def calendar(clinic: Clinic, sessions: sessionmaker[Session], now: Callable[[], datetime]) -> Calendar:
    return Calendar(clinic, sessions, now)
