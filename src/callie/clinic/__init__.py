"""The business Callie answers for: hours, services, resources, knowledge base (a YAML file + Markdown)."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from functools import cached_property
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, Field, field_validator

PACKAGE_DIR = Path(__file__).parent
DEFAULT_CLINIC_FILE = PACKAGE_DIR / "clinic.yaml"
DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class Service(BaseModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    minutes: int
    resource: str


class Resource(BaseModel):
    id: str
    name: str


class Clinic(BaseModel):
    name: str
    assistant_name: str = "Callie"
    timezone: str
    phone: str
    address: str
    emergency_line: str = "911"
    hours: dict[str, list[tuple[str, str]]]
    holidays: list[date] = Field(default_factory=list)
    slot_minutes: int = 30
    min_notice_minutes: int = 120
    booking_horizon_days: int = 60
    max_offered_slots: int = 3
    resources: list[Resource]
    services: list[Service]
    knowledge_file: Path | None = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        ZoneInfo(value)  # raises for unknown zones
        return value

    @cached_property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def service(self, service_id: str) -> Service | None:
        return next((s for s in self.services if s.id == service_id), None)

    def resource(self, resource_id: str) -> Resource:
        return next(r for r in self.resources if r.id == resource_id)

    def match_service(self, text: str) -> Service | None:
        """Map a caller's words ("a cleaning", "my tooth hurts") to a service id; None when unclear."""
        lowered = text.lower().strip()
        exact = self.service(lowered.replace(" ", "_"))
        if exact:
            return exact
        best: tuple[int, Service] | None = None
        for service in self.services:
            for alias in [service.name, *service.aliases]:
                if re.search(rf"\b{re.escape(alias.lower())}\b", lowered) and (best is None or len(alias) > best[0]):
                    best = (len(alias), service)
        return best[1] if best else None

    def intervals(self, day: date) -> list[tuple[datetime, datetime]]:
        """Opening intervals of a day as aware datetimes (empty on closed days and holidays)."""
        if day in self.holidays:
            return []
        out = []
        for start, end in self.hours.get(DAY_KEYS[day.weekday()], []):
            out.append(
                (
                    datetime.combine(day, time.fromisoformat(start), tzinfo=self.tz),
                    datetime.combine(day, time.fromisoformat(end), tzinfo=self.tz),
                )
            )
        return out

    def is_open(self, moment: datetime) -> bool:
        local = moment.astimezone(self.tz)
        return any(start <= local < end for start, end in self.intervals(local.date()))

    def next_open(self, moment: datetime) -> datetime | None:
        local = moment.astimezone(self.tz)
        for offset in range(14):
            for start, end in self.intervals(local.date() + timedelta(days=offset)):
                if end > local:
                    return max(start, local)
        return None

    def knowledge_text(self) -> str:
        path = self.knowledge_file or next((PACKAGE_DIR / "knowledge").glob("*.md"))
        return path.read_text(encoding="utf-8")


def load_clinic(path: Path | None = None) -> Clinic:
    source = path or DEFAULT_CLINIC_FILE
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if data.get("knowledge_file"):
        data["knowledge_file"] = (source.parent / data["knowledge_file"]).resolve()
    return Clinic.model_validate(data)
