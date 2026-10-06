"""Claude through the official `anthropic` SDK (Messages API, streaming, client-side tools).

OpenAI-format messages and tools are converted at this edge, so the agent only knows one format.
Not exercised against the live API in this repository (no Anthropic key was used); the conversion and the
event mapping are unit-tested with a stubbed stream.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from callie.llm.base import Completed, JsonDict, LLMEvent, ProviderError, RetryableError, TextDelta, ToolCall

DEFAULT_MODEL = "claude-sonnet-5"


def to_anthropic(messages: list[JsonDict], tools: list[JsonDict]) -> tuple[str, list[JsonDict], list[JsonDict]]:
    """OpenAI chat messages + function tools -> (system, messages, tools) for the Messages API."""
    system_parts: list[str] = []
    out: list[JsonDict] = []

    def append(role: str, blocks: list[JsonDict]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if role == "system":
            system_parts.append(text)
        elif role == "user":
            append("user", [{"type": "text", "text": text}] if text else [])
        elif role == "assistant":
            blocks: list[JsonDict] = [{"type": "text", "text": text}] if text else []
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                blocks.append({"type": "tool_use", "id": call["id"], "name": function.get("name", ""), "input": arguments})
            append("assistant", blocks)
        elif role == "tool":
            append("user", [{"type": "tool_result", "tool_use_id": message.get("tool_call_id", ""), "content": text}])
    converted_tools = [
        {
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "input_schema": tool["function"].get("parameters") or {"type": "object", "properties": {}},
            # Stream tool inputs as they are generated; the agent validates every argument before running a tool.
            "eager_input_streaming": True,
        }
        for tool in tools
        if tool.get("type") == "function"
    ]
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(call connected)"}]})
    return "\n\n".join(p for p in system_parts if p), out, converted_tools


class AnthropicChat:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        effort: str = "low",
        timeout_seconds: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        # Retries live in the resilient wrapper (one ledger, one backoff policy for every provider).
        self._client = client or AsyncAnthropic(api_key=api_key, max_retries=0, timeout=timeout_seconds)

    @property
    def label(self) -> str:
        return f"anthropic/{self.model}"

    @property
    def is_remote(self) -> bool:
        return True

    async def stream(
        self, messages: list[JsonDict], tools: list[JsonDict], *, max_tokens: int, temperature: float
    ) -> AsyncIterator[LLMEvent]:
        system, converted, converted_tools = to_anthropic(messages, tools)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": converted,
            # A phone answer is short: low effort keeps time-to-first-token down. Sampling parameters are not
            # sent; current Claude models reject `temperature`.
            "output_config": {"effort": self.effort},
        }
        if system:
            kwargs["system"] = system
        if converted_tools:
            kwargs["tools"] = converted_tools
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_delta" and getattr(event.delta, "type", "") == "text_delta":
                        yield TextDelta(event.delta.text)
                final = await stream.get_final_message()
        except anthropic.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after") if exc.response is not None else None
            raise RetryableError(f"rate limited: {exc.message}", float(retry_after) if retry_after else None, 429) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise RetryableError(f"anthropic {exc.status_code}: {exc.message}") from exc
            raise ProviderError(f"anthropic {exc.status_code}: {exc.message}", exc.status_code) from exc
        except anthropic.APIConnectionError as exc:
            raise RetryableError(f"anthropic connection error: {exc}") from exc
        if final.stop_reason == "refusal":
            raise ProviderError("the model declined to answer (refusal)", 422)
        tool_uses = [block for block in final.content if block.type == "tool_use"]
        if final.stop_reason == "max_tokens" and tool_uses:
            raise ProviderError("tool input truncated at max_tokens", 502)
        for block in tool_uses:
            arguments = block.input if isinstance(block.input, dict) else {"_invalid_json": str(block.input)[:500]}
            yield ToolCall(id=block.id, name=block.name, arguments=arguments, raw_arguments=json.dumps(arguments))
        yield Completed(
            finish_reason="tool_calls" if tool_uses else "stop",
            served_model=final.model,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )
