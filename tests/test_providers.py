"""LLM providers: the free-only guard (ids and served model), SSE parsing with tool calls, retries, fallback,
cache and ledger, Anthropic conversion, the fake receptionist."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from callie.config import Settings
from callie.llm.anthropic_provider import to_anthropic
from callie.llm.base import (
    BudgetExceededError,
    Completed,
    FreeModelGuardError,
    JsonDict,
    LLMEvent,
    ProviderError,
    RetryableError,
    TextDelta,
    ToolCall,
    ensure_free_models,
)
from callie.llm.factory import build_chat_model
from callie.llm.openai_compat import OpenAICompatibleChat
from callie.llm.resilient import CallLedger, DiskCache, ResilientChat


def sse(*chunks: JsonDict) -> bytes:
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def openrouter(handler: Any, model: str = "x/y:free", **kwargs: Any) -> OpenAICompatibleChat:
    return OpenAICompatibleChat(kind="openrouter", model=model, api_key="test", base_url="https://openrouter.test/api/v1",
                                transport=httpx.MockTransport(handler), **kwargs)  # fmt: skip


async def drain(chat: Any) -> list[LLMEvent]:
    return [e async for e in chat.stream([{"role": "user", "content": "hi"}], [], max_tokens=10, temperature=0)]


class TestFreeOnlyGuard:
    def test_ids(self) -> None:
        ensure_free_models(["a/b:free", "c/d:free"])
        with pytest.raises(FreeModelGuardError):
            ensure_free_models(["a/b:free", "openai/gpt-5"])

    def test_constructor_refuses_paid_model_and_fallbacks(self) -> None:
        with pytest.raises(FreeModelGuardError):
            openrouter(lambda r: httpx.Response(200), model="anthropic/claude-sonnet-5")
        with pytest.raises(FreeModelGuardError):
            openrouter(lambda r: httpx.Response(200), fallback_models=["meta/llama-4:paid"])

    def test_factory_refuses_before_any_request(self) -> None:
        settings = Settings(llm_provider="openrouter", llm_model="openai/gpt-5", llm_ledger=None)
        from callie.clinic import load_clinic

        with pytest.raises(FreeModelGuardError):
            build_chat_model(settings, load_clinic())

    def test_guard_can_be_disabled_for_clients(self) -> None:
        chat = openrouter(lambda r: httpx.Response(200), model="openai/gpt-5", require_free=False)
        assert chat.build_body([], [], max_tokens=1, temperature=0)["model"] == "openai/gpt-5"

    async def test_answer_served_by_a_paid_model_is_rejected(self) -> None:
        body = sse({"model": "openai/gpt-5", "choices": [{"delta": {"content": "hi"}}]})
        chat = openrouter(lambda r: httpx.Response(200, content=body))
        with pytest.raises(FreeModelGuardError):
            await drain(chat)

    def test_request_body_carries_fallbacks_and_tools(self) -> None:
        chat = openrouter(lambda r: httpx.Response(200), fallback_models=["b/c:free"])
        body = chat.build_body([{"role": "user", "content": "x"}], [{"type": "function", "function": {"name": "t"}}],
                               max_tokens=5, temperature=0.1)  # fmt: skip
        assert body["models"] == ["x/y:free", "b/c:free"] and body["tool_choice"] == "auto" and body["stream"] is True


class TestStreaming:
    async def test_text_and_tool_call_deltas(self) -> None:
        body = sse(
            {"model": "x/y:free", "choices": [{"delta": {"content": "Let me "}}]},
            {"choices": [{"delta": {"content": "check."}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "check_availability", "arguments": '{"serv'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ice": "cleaning"}'}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 9}},
        )  # fmt: skip
        events = await drain(openrouter(lambda r: httpx.Response(200, content=body)))
        assert [e.text for e in events if isinstance(e, TextDelta)] == ["Let me ", "check."]
        call = next(e for e in events if isinstance(e, ToolCall))
        assert call.arguments == {"service": "cleaning"} and call.id == "c1"
        done = events[-1]
        assert isinstance(done, Completed) and done.finish_reason == "tool_calls" and done.input_tokens == 50

    async def test_error_mapping(self) -> None:
        for status, error in [(429, RetryableError), (503, RetryableError), (400, ProviderError)]:
            chat = openrouter(lambda r, s=status: httpx.Response(s, json={"error": {"message": "nope"}}))
            with pytest.raises(error):
                await drain(chat)
        empty = openrouter(lambda r: httpx.Response(200, content=sse({"choices": [{"delta": {}, "finish_reason": "length"}]})))
        with pytest.raises(ProviderError, match="max_tokens"):
            await drain(empty)

    async def test_error_inside_the_stream(self) -> None:
        body = sse({"error": {"code": 429, "message": "rate-limited upstream"}})
        with pytest.raises(RetryableError):
            await drain(openrouter(lambda r: httpx.Response(200, content=body)))


class FlakyModel:
    def __init__(self, name: str, failures: int, remote: bool = True) -> None:
        self.name = name
        self.failures = failures
        self.calls = 0
        self.remote = remote

    @property
    def label(self) -> str:
        return f"openrouter/{self.name}"

    @property
    def is_remote(self) -> bool:
        return self.remote

    async def stream(self, messages: list[JsonDict], tools: list[JsonDict], *, max_tokens: int, temperature: float) -> AsyncIterator[LLMEvent]:
        self.calls += 1
        if self.calls <= self.failures:
            raise RetryableError("rate limited", retry_after=0.0)
        yield TextDelta(f"hello from {self.name}")
        yield Completed("stop", served_model=self.name)


class TestResilient:
    async def test_retry_then_fallback_then_ledger_and_cache(self, tmp_path: Path) -> None:
        primary, backup = FlakyModel("a/b:free", failures=10), FlakyModel("c/d:free", failures=1)
        ledger = CallLedger(tmp_path / "calls.jsonl", max_calls=100)
        chat = ResilientChat([primary, backup], ledger=ledger, cache=DiskCache(tmp_path / "cache"), max_retries=1,
                             retry_base_seconds=0.0, sleep=lambda s: None)  # fmt: skip
        events = await drain(chat)
        assert isinstance(events[0], TextDelta) and "c/d:free" in events[0].text
        assert primary.calls == 2 and backup.calls == 2
        rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
        assert [r["status"] for r in rows] == ["retryable_error", "retryable_error", "retryable_error", "ok"]
        assert rows[-1]["ttft_s"] is not None and "messages" not in rows[-1]  # never prompts
        cached = await drain(chat)
        assert backup.calls == 2 and isinstance(cached[-1], Completed) and cached[-1].cached

    async def test_budget(self, tmp_path: Path) -> None:
        chat = ResilientChat([FlakyModel("a/b:free", failures=0)], ledger=CallLedger(tmp_path / "l.jsonl", max_calls=1))
        await drain(chat)
        with pytest.raises(BudgetExceededError):
            await chat.stream([{"role": "user", "content": "other"}], [], max_tokens=10, temperature=0).__anext__()


def test_anthropic_conversion() -> None:
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "assistant", "content": "Hi, how can I help?"},
        {"role": "user", "content": "Book a cleaning"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "check_availability", "arguments": '{"when": "Tuesday"}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": '{"status": "ok"}'},
        {"role": "assistant", "content": "I have Tuesday at 3 PM."},
    ]  # fmt: skip
    tools = [{"type": "function", "function": {"name": "check_availability", "description": "d", "parameters": {"type": "object"}}}]
    system, converted, converted_tools = to_anthropic(messages, tools)
    assert system == "Be brief."
    assert converted[0]["role"] == "user"  # a conversation must start with the caller
    assert converted[-2]["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "content": '{"status": "ok"}'}
    assert any(b.get("type") == "tool_use" and b["input"] == {"when": "Tuesday"} for m in converted for b in m["content"])
    assert converted_tools[0]["input_schema"] == {"type": "object"} and converted_tools[0]["eager_input_streaming"]


async def test_fake_receptionist_is_deterministic() -> None:
    from callie.clinic import load_clinic
    from callie.llm.fake import FakeReceptionist

    fake = FakeReceptionist(load_clinic())
    messages = [{"role": "user", "content": "I'd like to book a cleaning next Tuesday afternoon"}]
    first = [e async for e in fake.stream(messages, [], max_tokens=10, temperature=0)]
    second = [e async for e in fake.stream(messages, [], max_tokens=10, temperature=0)]
    calls = [e for e in first if isinstance(e, ToolCall)]
    assert calls and calls[0].name == "check_availability"
    assert [type(e) for e in first] == [type(e) for e in second]
