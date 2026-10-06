"""Streaming chat-model interface: OpenAI-format messages and tools in, text deltas and tool calls out.

Voice needs the first words as early as possible, so every provider streams; the pipeline speaks the first
sentence while the rest is still being generated. Tool calls are emitted once their arguments are complete.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: JsonDict
    raw_arguments: str = ""


@dataclass(frozen=True)
class Completed:
    finish_reason: str
    served_model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    extra: JsonDict = field(default_factory=dict)


LLMEvent = TextDelta | ToolCall | Completed


class ProviderError(RuntimeError):
    """The request failed and should not be retried."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class RetryableError(ProviderError):
    """Rate limit, overload, timeout or an empty answer: worth retrying with backoff or another model."""

    def __init__(self, message: str, retry_after: float | None = None, status_code: int = 503) -> None:
        super().__init__(message, status_code)
        self.retry_after = retry_after


class FreeModelGuardError(ProviderError):
    """A non-`:free` model id was about to be used with OpenRouter while the free-only guard is on."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


class BudgetExceededError(ProviderError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=429)


def ensure_free_models(model_ids: Iterable[str]) -> None:
    """Refuse any model id that is not an OpenRouter free variant (suffix `:free`)."""
    paid = [str(model) for model in model_ids if not str(model).endswith(":free")]
    if paid:
        raise FreeModelGuardError(
            f"free-only guard: refusing non-free OpenRouter model id(s): {', '.join(paid)} "
            "(set CALLIE_REQUIRE_FREE_MODELS=false to allow paid models)"
        )


class ChatModel(Protocol):
    @property
    def label(self) -> str:
        """Provider and model, e.g. "openrouter/google/gemma-4-31b-it:free"."""
        ...

    @property
    def is_remote(self) -> bool:
        """True for real APIs (retried, cached and counted in the call ledger)."""
        ...

    def stream(
        self, messages: list[JsonDict], tools: list[JsonDict], *, max_tokens: int, temperature: float
    ) -> AsyncIterator[LLMEvent]: ...


def message_text(message: JsonDict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""
