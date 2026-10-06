"""Budget-safe wrapper for real APIs: retries with backoff, model fallback, disk cache, call budget, call ledger.

Free models are rate limited ("429 rate-limited upstream" is common), so a request is retried with exponential
backoff (or `Retry-After`), then moved to the next fallback model. Retries are only possible before the first
token reached the caller; after that an error ends the turn and the agent apologizes.

Every real request, retries included, becomes one JSON line in the ledger (model ids, status, latency,
time to first token, token counts; never prompts), so "how many calls did this cost" is read from a file.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from callie.llm.base import (
    BudgetExceededError,
    ChatModel,
    Completed,
    JsonDict,
    LLMEvent,
    ProviderError,
    RetryableError,
    TextDelta,
    ToolCall,
)
from callie.logging_config import get_logger

log = get_logger(__name__)


class DiskCache:
    """One JSON file per request hash: re-running the evaluation replays answers instead of paying again."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @staticmethod
    def key(label: str, payload: JsonDict) -> str:
        material = json.dumps({"provider": label, **payload}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> list[JsonDict] | None:
        path = self._path(key)
        if not path.exists():
            return None
        data: list[JsonDict] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def put(self, key: str, events: list[JsonDict]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")


class Throttle:
    """Spaces request starts at least `min_interval` seconds apart."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._next_start = 0.0
        self._lock = threading.Lock()

    async def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + self.min_interval
        if start > now:
            await asyncio.sleep(start - now)


class CallLedger:
    """Counts real requests against a hard budget; one JSON line per request."""

    def __init__(self, path: Path | None, max_calls: int) -> None:
        self.path = path
        self.max_calls = max_calls
        self.calls = 0
        self._lock = threading.Lock()
        if path is not None and path.exists():
            with path.open(encoding="utf-8") as handle:
                self.calls = sum(1 for line in handle if line.strip())

    def reserve(self) -> None:
        with self._lock:
            if self.calls >= self.max_calls:
                raise BudgetExceededError(f"call budget of {self.max_calls} real requests reached ({self.path})")
            self.calls += 1

    def record(self, **entry: object) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **entry}
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _event_to_json(event: LLMEvent) -> JsonDict:
    return {"type": type(event).__name__, **asdict(event)}


def _event_from_json(data: JsonDict) -> LLMEvent:
    kind = data.pop("type")
    if kind == "TextDelta":
        return TextDelta(**data)
    if kind == "ToolCall":
        return ToolCall(**data)
    return Completed(**{**data, "cached": True})


class ResilientChat:
    def __init__(
        self,
        models: list[ChatModel],
        *,
        ledger: CallLedger | None = None,
        cache: DiskCache | None = None,
        throttle: Throttle | None = None,
        max_retries: int = 3,
        retry_base_seconds: float = 2.0,
        tag: str = "agent",
        sleep: Callable[[float], object] | None = None,
    ) -> None:
        if not models:
            raise ValueError("at least one model")
        self.models = models
        self.ledger = ledger or CallLedger(None, max_calls=10**9)
        self.cache = cache
        self.throttle = throttle
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.tag = tag
        self._sleep = sleep

    @property
    def label(self) -> str:
        return self.models[0].label

    @property
    def is_remote(self) -> bool:
        return any(model.is_remote for model in self.models)

    def with_tag(self, tag: str) -> ResilientChat:
        return ResilientChat(
            self.models,
            ledger=self.ledger,
            cache=self.cache,
            throttle=self.throttle,
            max_retries=self.max_retries,
            retry_base_seconds=self.retry_base_seconds,
            tag=tag,
            sleep=self._sleep,
        )

    async def stream(
        self, messages: list[JsonDict], tools: list[JsonDict], *, max_tokens: int, temperature: float
    ) -> AsyncIterator[LLMEvent]:
        payload = {"messages": messages, "tools": tools, "max_tokens": max_tokens, "temperature": temperature}
        key = DiskCache.key("|".join(m.label for m in self.models), payload) if self.cache else ""
        if self.cache and (hit := self.cache.get(key)) is not None:
            for item in hit:
                yield _event_from_json(dict(item))
            return
        last_error: ProviderError | None = None
        for model in self.models:
            for attempt in range(self.max_retries + 1):
                await self._before_request(model)
                started = time.monotonic()
                first_token: float | None = None
                recorded: list[JsonDict] = []
                emitted = False
                try:
                    async for event in model.stream(messages, tools, max_tokens=max_tokens, temperature=temperature):
                        if first_token is None and isinstance(event, TextDelta | ToolCall):
                            first_token = time.monotonic() - started
                        recorded.append(_event_to_json(event))
                        emitted = True
                        yield event
                except RetryableError as exc:
                    last_error = exc
                    self._record(model, "retryable_error", started, first_token, error=str(exc))
                    if emitted:
                        raise
                    if attempt < self.max_retries:
                        delay = self._backoff(model, attempt, exc)
                        log.warning(
                            "llm.retry",
                            model=model.label,
                            attempt=attempt + 1,
                            delay=round(delay, 1),
                            error=str(exc)[:160],
                        )
                        await self._pause(delay)
                    continue
                except ProviderError as exc:
                    self._record(model, "error", started, first_token, error=str(exc))
                    raise
                completed = next((e for e in recorded if e["type"] == "Completed"), {})
                self._record(model, "ok", started, first_token, completed=completed)
                if self.cache:
                    self.cache.put(key, recorded)
                return
            log.warning("llm.fallback", failed=model.label, error=str(last_error)[:160])
        raise last_error or ProviderError("request failed")

    async def _pause(self, delay: float) -> None:
        if self._sleep is not None:
            self._sleep(delay)
            return
        await asyncio.sleep(delay)

    async def _before_request(self, model: ChatModel) -> None:
        if model.is_remote:
            self.ledger.reserve()
            if self.throttle:
                await self.throttle.wait()

    def _backoff(self, model: ChatModel, attempt: int, error: RetryableError) -> float:
        if not model.is_remote:
            return 0.0
        if error.retry_after:
            return min(error.retry_after, 20.0)
        return float(min(self.retry_base_seconds * 2.0**attempt, 20.0) * (0.75 + random.random() / 2))

    def _record(
        self,
        model: ChatModel,
        status: str,
        started: float,
        first_token: float | None,
        *,
        error: str | None = None,
        completed: JsonDict | None = None,
    ) -> None:
        if not model.is_remote:
            return
        completed = completed or {}
        self.ledger.record(
            tag=self.tag,
            provider=model.label,
            requested_model=model.label.split("/", 1)[1] if "/" in model.label else model.label,
            served_model=completed.get("served_model"),
            status=status,
            latency_s=round(time.monotonic() - started, 2),
            ttft_s=round(first_token, 2) if first_token is not None else None,
            input_tokens=completed.get("input_tokens", 0),
            output_tokens=completed.get("output_tokens", 0),
            error=error[:300] if error else None,
        )
