"""OpenRouter, OpenAI and any OpenAI-compatible server (vLLM, LiteLLM, a local Ollama) over streaming HTTP.

Plain httpx + SSE instead of an SDK: OpenRouter-specific fields (`models` fallback list, `reasoning`,
`provider` routing) pass through untouched, and the free-only guard sees exactly what is sent.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx

from callie.llm.base import (
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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/gelevanog/ai-voice-receptionist",
    "X-Title": "Callie voice receptionist",
}


class OpenAICompatibleChat:
    def __init__(
        self,
        *,
        kind: Literal["openai", "openrouter"],
        model: str,
        api_key: str | None,
        base_url: str,
        fallback_models: list[str] | None = None,
        require_free: bool = True,
        timeout_seconds: float = 60.0,
        extra_body: JsonDict | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        local = any(host in base_url for host in ("localhost", "127.0.0.1", "0.0.0.0", "ollama"))
        if not api_key and not local:
            raise ProviderError(f"{'OPENROUTER' if kind == 'openrouter' else 'OPENAI'}_API_KEY is not set", 500)
        self.kind = kind
        self.model = model
        self.fallback_models = list(fallback_models or [])
        self.require_free = require_free and kind == "openrouter"
        if self.require_free:
            ensure_free_models([model, *self.fallback_models])
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key or 'unused'}"}
        if kind == "openrouter":
            self._headers |= OPENROUTER_HEADERS
        self._timeout = timeout_seconds
        self._extra = dict(extra_body or {})
        self._transport = transport

    @property
    def label(self) -> str:
        return f"{self.kind}/{self.model}"

    @property
    def is_remote(self) -> bool:
        return True

    def build_body(
        self, messages: list[JsonDict], tools: list[JsonDict], *, max_tokens: int, temperature: float
    ) -> JsonDict:
        body: JsonDict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **self._extra,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self.kind == "openrouter":
            body["usage"] = {"include": True}
            if self.fallback_models:
                body["models"] = [self.model, *self.fallback_models]
            if self.require_free:
                ensure_free_models([body["model"], *body.get("models", [])])
        else:
            body["stream_options"] = {"include_usage": True}
        return body

    def _check_served(self, served: object) -> None:
        if self.require_free and isinstance(served, str) and served and not served.endswith(":free"):
            raise FreeModelGuardError(f"OpenRouter served non-free model {served!r}; refusing the answer")

    async def stream(
        self, messages: list[JsonDict], tools: list[JsonDict], *, max_tokens: int, temperature: float
    ) -> AsyncIterator[LLMEvent]:
        body = self.build_body(messages, tools, max_tokens=max_tokens, temperature=temperature)
        calls: dict[int, dict[str, Any]] = {}
        finish = "stop"
        served: str | None = None
        usage: JsonDict = {}
        produced = False
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client,
                client.stream("POST", self._url, json=body, headers=self._headers) as response,
            ):
                if response.status_code >= 400:
                    await response.aread()
                    raise_for_status(response.status_code, response.text, response.headers.get("retry-after"))
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue  # keep-alives and ": OPENROUTER PROCESSING" comments
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk: JsonDict = json.loads(payload)
                    if "error" in chunk:
                        error = chunk["error"] if isinstance(chunk["error"], dict) else {"message": chunk["error"]}
                        code = error.get("code")
                        raise_for_status(code if isinstance(code, int) else 502, json.dumps(chunk), None)
                    if chunk.get("model") and served is None:
                        served = str(chunk["model"])
                        self._check_served(served)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            produced = True
                            yield TextDelta(delta["content"])
                        for call in delta.get("tool_calls") or []:
                            produced = True
                            slot = calls.setdefault(int(call.get("index", 0)), {"id": "", "name": "", "arguments": ""})
                            slot["id"] = call.get("id") or slot["id"]
                            function = call.get("function") or {}
                            slot["name"] = function.get("name") or slot["name"]
                            slot["arguments"] += function.get("arguments") or ""
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
        except httpx.TimeoutException as exc:
            raise RetryableError(f"upstream timeout: {exc}", status_code=504) from exc
        except httpx.TransportError as exc:
            raise RetryableError(f"upstream connection error: {exc}") from exc
        for index in sorted(calls):
            slot = calls[index]
            yield ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments=parse_arguments(slot["arguments"]),
                raw_arguments=slot["arguments"],
            )
        if not produced:
            if finish == "length":
                raise ProviderError("empty answer: max_tokens reached before any output", 502)
            raise RetryableError(f"empty answer (finish_reason={finish})")
        yield Completed(
            finish_reason=finish,
            served_model=served,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )


def parse_arguments(raw: str) -> JsonDict:
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_invalid_json": raw[:500]}
    return value if isinstance(value, dict) else {"_invalid_json": raw[:500]}


def raise_for_status(status: int, text: str, retry_after: str | None) -> None:
    message = text[:300]
    try:
        data = json.loads(text)
        error = data.get("error", data) if isinstance(data, dict) else data
        if isinstance(error, dict):
            metadata = error.get("metadata")
            raw = metadata.get("raw") if isinstance(metadata, dict) else None
            message = str(raw or error.get("message") or error)[:300]
    except (json.JSONDecodeError, AttributeError):
        pass
    if status == 429:
        try:
            delay = float(retry_after) if retry_after else None
        except ValueError:
            delay = None
        raise RetryableError(f"rate limited: {message}", retry_after=delay, status_code=429)
    if status >= 500 or status in {408, 409}:
        raise RetryableError(f"upstream {status}: {message}", status_code=502)
    raise ProviderError(f"upstream {status}: {message}", status)


async def list_free_models(api_key: str | None, base_url: str = OPENROUTER_BASE_URL) -> list[JsonDict]:
    """Models whose id ends with `:free`, with the fields that matter for voice (context, tools support)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(base_url.rstrip("/") + "/models", headers=headers)
        response.raise_for_status()
    models = response.json().get("data", [])
    return [
        {
            "id": m["id"],
            "context_length": m.get("context_length"),
            "tools": "tools" in (m.get("supported_parameters") or []),
            "created": m.get("created"),
        }
        for m in models
        if str(m.get("id", "")).endswith(":free")
    ]
