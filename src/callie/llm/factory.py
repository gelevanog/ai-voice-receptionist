"""Build the configured chat model, wrapped for retries, fallbacks, caching and the call budget."""

from __future__ import annotations

import os

from callie.clinic import Clinic
from callie.config import Settings
from callie.llm.anthropic_provider import AnthropicChat
from callie.llm.base import ChatModel, ensure_free_models
from callie.llm.fake import FakeReceptionist
from callie.llm.openai_compat import OPENROUTER_BASE_URL, OpenAICompatibleChat
from callie.llm.resilient import CallLedger, DiskCache, ResilientChat, Throttle

_LEDGERS: dict[str, CallLedger] = {}


def shared_ledger(settings: Settings) -> CallLedger:
    """One ledger per file per process, so every component counts against the same budget."""
    key = str(settings.llm_ledger)
    if key not in _LEDGERS:
        _LEDGERS[key] = CallLedger(settings.llm_ledger, settings.llm_max_calls)
    return _LEDGERS[key]


def build_chat_model(
    settings: Settings,
    clinic: Clinic,
    *,
    provider: str | None = None,
    model: str | None = None,
    fallback_models: list[str] | None = None,
    tag: str = "agent",
    fake_delay_s: float = 0.0,
    reasoning: dict[str, object] | None = None,
) -> ChatModel:
    provider = provider or settings.llm_provider
    model = model or (settings.resolved_llm_model() if provider == settings.llm_provider else "")
    fallbacks = list(settings.llm_fallback_models if fallback_models is None else fallback_models)
    if provider == "fake":
        return FakeReceptionist(clinic, delay_s=fake_delay_s)
    models: list[ChatModel]
    if provider == "openrouter":
        if settings.require_free_models:
            ensure_free_models([model, *fallbacks])
        # Each model is tried in turn on our side; OpenRouter's own `models` fallback is not used, so the ledger
        # always names the model that was asked and the free-only guard sees every id.
        models = [
            OpenAICompatibleChat(
                kind="openrouter",
                model=name,
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
                require_free=settings.require_free_models,
                timeout_seconds=settings.llm_timeout_seconds,
                # Voice needs the first words fast: keep hidden reasoning minimal and out of the stream.
                extra_body={"reasoning": reasoning or {"effort": "low", "exclude": True}},
            )
            for name in [model, *fallbacks]
        ]
    elif provider == "openai":
        models = [
            OpenAICompatibleChat(
                kind="openai",
                model=name,
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                require_free=False,
                timeout_seconds=settings.llm_timeout_seconds,
            )
            for name in [model or "gpt-5-mini", *fallbacks]
        ]
    elif provider == "anthropic":
        models = [AnthropicChat(model=model or "claude-sonnet-5", api_key=os.environ.get("ANTHROPIC_API_KEY"))]
    else:
        raise ValueError(f"unknown LLM provider {provider!r}")
    return ResilientChat(
        models,
        ledger=shared_ledger(settings),
        cache=DiskCache(settings.llm_cache_dir),
        throttle=Throttle(settings.llm_min_seconds_between_requests),
        max_retries=settings.llm_max_retries,
        tag=tag,
    )
