"""Scenario definitions for the simulated callers (see scenarios.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

SCENARIO_FILE = Path(__file__).with_name("scenarios.yaml")
Channel = Literal["clean", "phone", "phone_noisy"]


class Interrupt(BaseModel):
    turn: int
    after_s: float
    text: str
    backchannel: bool = False


class Setup(BaseModel):
    service: str
    start: str  # local ISO, clinic timezone
    name: str
    phone: str | None = None


class Expect(BaseModel):
    model_config = {"extra": "allow"}

    outcome: str
    service: str | None = None
    day: str | None = None
    from_day: str | None = None
    to_day: str | None = None
    after: str | None = None
    before: str | None = None
    at: str | None = None
    name: str | None = None
    phone_last4: str | None = None
    appointment: int | None = None  # index into `setup`
    must_not_book: bool = False
    must_not_transfer: bool = False
    agent_says: list[str] = Field(default_factory=list)
    agent_says_any: list[str] = Field(default_factory=list)
    agent_says_any_2: list[str] = Field(default_factory=list)
    agent_must_not_say: list[str] = Field(default_factory=list)
    barge_in: bool = False
    backchannel: bool = False


class Scenario(BaseModel):
    id: str
    tier: Literal["core", "extended"] = "core"
    category: str
    channel: Channel = "clean"
    voice: int = 0
    persona: str
    goal: str
    facts: dict[str, Any] = Field(default_factory=dict)
    opening: str = ""
    caller_id: str | None = None
    now: str | None = None
    setup: list[Setup] = Field(default_factory=list)
    interrupt: Interrupt | None = None
    silent: bool = False
    script: list[str] = Field(default_factory=list)
    expect: Expect


def load_scenarios(path: Path = SCENARIO_FILE, *, include_extended: bool = True) -> list[Scenario]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    scenarios = [Scenario.model_validate(item) for item in data]
    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario ids")
    return [s for s in scenarios if include_extended or s.tier == "core"]


def get_scenario(scenario_id: str) -> Scenario:
    for scenario in load_scenarios():
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(f"unknown scenario {scenario_id!r}")
