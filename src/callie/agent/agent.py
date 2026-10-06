"""One caller turn in, a stream of speakable sentences, tool events and call actions out.

Order of decisions in a turn:
1. deterministic rules: emergencies, requests for a person, anger, repeated misunderstanding, goodbyes;
2. a plain "yes" to a pending read-back executes the confirmed action directly (no model round trip);
3. otherwise the model streams; sentences are released as soon as they are complete, tool calls run as they
   arrive, and a tool's `say` text ends the turn (exact facts spoken by code).

The generator can be cancelled at any `yield` (barge-in). History is updated before each yield, so a cancelled
turn leaves a consistent conversation; `note_interrupted` then trims the last answer to what was heard.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from callie.agent.grounding import extract_times, unsupported_times
from callie.agent.policy import (
    SAFE_REPLACEMENT,
    Escalation,
    EscalationState,
    Reply,
    analyze_turn,
    classify_reply,
    violates_advice_policy,
)
from callie.agent.prompts import greeting, system_prompt
from callie.agent.tools import CallContext, ToolBox, ToolResult, tool_specs
from callie.llm.base import ChatModel, Completed, JsonDict, ProviderError, TextDelta, ToolCall
from callie.logging_config import get_logger
from callie.tts.chunker import SentenceChunker

log = get_logger(__name__)
Source = Literal["llm", "tool", "rules", "filler"]
FILLERS = ["One moment.", "Let me check.", "Sure, one second."]
SIDE_EFFECT_TOOLS = {
    "book_appointment",
    "reschedule_appointment",
    "cancel_appointment",
    "take_message",
    "transfer_to_human",
    "end_call",
}


@dataclass(frozen=True)
class Sentence:
    text: str
    source: Source


@dataclass(frozen=True)
class ToolEvent:
    result: ToolResult


@dataclass(frozen=True)
class CallAction:
    kind: Literal["transfer", "hangup"]
    reason: str = ""


@dataclass(frozen=True)
class LLMTiming:
    model: str
    first_token_s: float | None
    total_s: float
    cached: bool
    served_model: str | None
    error: str | None = None


AgentEvent = Sentence | ToolEvent | CallAction | LLMTiming


@dataclass
class TurnStats:
    path: str = "llm"  # llm | rules | fast_confirm
    llm_calls: int = 0
    escalation: str | None = None
    unsupported_times: list[str] = field(default_factory=list)
    replaced_advice: int = 0


class Agent:
    def __init__(
        self,
        llm: ChatModel,
        ctx: CallContext,
        *,
        max_tokens: int = 400,
        temperature: float = 0.3,
        filler_after_s: float | None = 1.8,
        fast_confirm: bool = True,
        max_steps: int = 4,
    ) -> None:
        self.llm = llm
        self.ctx = ctx
        self.tools = ToolBox(ctx)
        self.specs = tool_specs(ctx.clinic)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.filler_after_s = filler_after_s if llm.is_remote else None
        self.fast_confirm = fast_confirm
        self.max_steps = max_steps
        self.escalation = EscalationState()
        self.messages: list[JsonDict] = [
            {"role": "system", "content": system_prompt(ctx.clinic, ctx.now(), ctx.caller_phone)}
        ]
        self._fillers = itertools.cycle(FILLERS)
        self.last_stats = TurnStats()
        self.turn_committed = False  # a tool with side effects ran in the current turn

    def greeting(self) -> str:
        text = greeting(self.ctx.clinic)
        self.messages.append({"role": "assistant", "content": text})
        return text

    def note_interrupted(self, heard: str) -> None:
        """Barge-in: keep only what the caller actually heard of the last answer."""
        for message in reversed(self.messages):
            if message["role"] == "assistant" and message.get("content"):
                message["content"] = (heard.strip() + " [interrupted by the caller]").strip()
                return

    def say_directly(self, text: str) -> None:
        """Record a line the pipeline spoke on its own (silence prompts)."""
        self.messages.append({"role": "assistant", "content": text})

    # -- the turn ----------------------------------------------------------------------------------------------
    async def respond(
        self, user_text: str, *, merge: bool = False, turn: int | None = None
    ) -> AsyncIterator[AgentEvent]:
        """`turn` is the caller-turn number (the session's count); merged utterances keep their turn."""
        stats = TurnStats()
        self.last_stats = stats
        self.turn_committed = False
        if merge and self.messages[-1]["role"] == "user":
            self.messages[-1]["content"] = f"{self.messages[-1]['content']} {user_text}".strip()
            user_text = self.messages[-1]["content"]
        else:
            self.messages.append({"role": "user", "content": user_text})
        if turn is not None:
            self.ctx.turn = turn
        elif not merge:
            self.ctx.turn += 1
        self.ctx.known_times.update(extract_times(user_text))
        reply = classify_reply(user_text)
        self.tools.register_reply(affirmed=reply is Reply.YES or reply is Reply.YES_PLUS, declined=reply is Reply.NO)
        signals = analyze_turn(user_text)
        escalation = self.escalation.update(signals)

        if escalation is not None:
            stats.path, stats.escalation = "rules", escalation.value
            async for event in self._escalate(escalation):
                yield event
            return
        pending = self.ctx.pending
        if self.fast_confirm and pending is not None and reply is Reply.YES and pending.affirmed_turn == self.ctx.turn:
            stats.path = "fast_confirm"
            async for event in self._run_rule_tool(pending.tool, {**pending.details, "confirmed": True}):
                yield event
            return
        if signals.goodbye and pending is None:
            stats.path = "rules"
            async for event in self._run_rule_tool("end_call", {"reason": "caller said goodbye"}):
                yield event
            return
        async for event in self._llm_turn(stats):
            yield event

    async def _escalate(self, escalation: Escalation) -> AsyncIterator[AgentEvent]:
        lead = {
            Escalation.EMERGENCY: (
                "That sounds serious. If you have swelling that is spreading, trouble breathing or swallowing, "
                f"or bleeding that won't stop, please hang up and call {self.ctx.clinic.emergency_line} now."
            ),
            Escalation.ANGRY: "I'm really sorry for the frustration.",
            Escalation.MISUNDERSTANDING: "I'm sorry, I'm having trouble understanding.",
            Escalation.HUMAN_REQUESTED: "",
        }[escalation]
        if lead:
            self.messages.append({"role": "assistant", "content": lead})
            for part in split_sentences(lead):
                yield Sentence(part, "rules")
        reason = {Escalation.EMERGENCY: "emergency, 911 advised"}.get(escalation, escalation.value.replace("_", " "))
        async for event in self._run_rule_tool("transfer_to_human", {"reason": reason}):
            yield event

    async def _run_rule_tool(self, name: str, arguments: JsonDict) -> AsyncIterator[AgentEvent]:
        call_id = f"rule_{uuid.uuid4().hex[:8]}"
        result = self.tools.execute(name, arguments)
        self.turn_committed = self.turn_committed or name in SIDE_EFFECT_TOOLS
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": call_id, "type": "function", "function": {"name": name, "arguments": _json(arguments)}}
                ],
            }
        )
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": result.for_model()})
        if result.say:
            self.messages.append({"role": "assistant", "content": result.say})
        yield ToolEvent(result)
        if result.say:
            for part in split_sentences(result.say):
                yield Sentence(part, "tool")
        if result.action:
            yield CallAction(result.action, str(arguments.get("reason", "")))  # type: ignore[arg-type]

    async def _llm_turn(self, stats: TurnStats) -> AsyncIterator[AgentEvent]:
        filler_used = False
        for _step in range(self.max_steps):
            chunker = SentenceChunker()
            spoken: list[str] = []
            calls: list[ToolCall] = []
            started = time.monotonic()
            first_token: float | None = None
            completed: Completed | None = None
            stats.llm_calls += 1
            stream = self.llm.stream(
                self.messages, self.specs, max_tokens=self.max_tokens, temperature=self.temperature
            )
            iterator = stream.__aiter__()
            pending_first: asyncio.Future[Any] | None = None
            try:
                if self.filler_after_s and not filler_used:
                    pending_first = asyncio.ensure_future(iterator.__anext__())
                    done, _ = await asyncio.wait({pending_first}, timeout=self.filler_after_s)
                    if not done:
                        filler_used = True
                        yield Sentence(next(self._fillers), "filler")
                    events: AsyncIterator[object] = _prepend(pending_first, iterator)
                else:
                    events = iterator
                async for event in events:
                    if isinstance(event, TextDelta):
                        first_token = first_token if first_token is not None else time.monotonic() - started
                        for sentence in chunker.push(event.text):
                            async for out in self._release(sentence, spoken, stats):
                                yield out
                    elif isinstance(event, ToolCall):
                        first_token = first_token if first_token is not None else time.monotonic() - started
                        calls.append(event)
                    elif isinstance(event, Completed):
                        completed = event
                for sentence in chunker.flush():
                    async for out in self._release(sentence, spoken, stats):
                        yield out
            except ProviderError as exc:
                log.warning("agent.llm_error", error=str(exc)[:200])
                self.escalation.failures += 1
                yield LLMTiming(self.llm.label, first_token, time.monotonic() - started, False, None, str(exc)[:200])
                apology = "Sorry, I'm having trouble on my end. Could you say that again?"
                if self.escalation.failures >= 2:
                    async for event in self._escalate(Escalation.MISUNDERSTANDING):
                        yield event
                    return
                self.messages.append({"role": "assistant", "content": apology})
                yield Sentence(apology, "rules")
                return
            finally:
                if pending_first is not None and not pending_first.done():
                    pending_first.cancel()
                    with contextlib.suppress(BaseException):
                        await pending_first
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    with contextlib.suppress(RuntimeError, StopAsyncIteration):
                        await aclose()
            self.escalation.failures = 0
            yield LLMTiming(
                self.llm.label,
                first_token,
                time.monotonic() - started,
                bool(completed and completed.cached),
                completed.served_model if completed else None,
            )
            text = " ".join(spoken).strip()
            if not calls:
                if text:
                    self.messages.append({"role": "assistant", "content": text})
                else:
                    fallback = "Sorry, could you say that again?"
                    self.messages.append({"role": "assistant", "content": fallback})
                    yield Sentence(fallback, "rules")
                return
            self.messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": c.raw_arguments or _json(c.arguments)},
                        }
                        for c in calls
                    ],
                }
            )
            results = [self.tools.execute(call.name, call.arguments) for call in calls]
            self.turn_committed = self.turn_committed or any(c.name in SIDE_EFFECT_TOOLS for c in calls)
            for call, result in zip(calls, results, strict=True):
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result.for_model()})
            says = " ".join(r.say for r in results if r.say)
            if says:
                self.messages.append({"role": "assistant", "content": says})
            for result in results:
                yield ToolEvent(result)
            if says:
                for part in split_sentences(says):
                    yield Sentence(part, "tool")
                for result in results:
                    if result.action:
                        yield CallAction(result.action, str(result.arguments.get("reason", "")))  # type: ignore[arg-type]
                return
        # The model kept calling tools without saying anything.
        fallback = "Sorry, let me get someone to help with that."
        yield Sentence(fallback, "rules")

    async def _release(self, sentence: str, spoken: list[str], stats: TurnStats) -> AsyncIterator[AgentEvent]:
        if violates_advice_policy(sentence):
            stats.replaced_advice += 1
            if SAFE_REPLACEMENT in spoken:
                return
            sentence = SAFE_REPLACEMENT
        stats.unsupported_times.extend(unsupported_times(sentence, self.ctx.known_times))
        spoken.append(sentence)
        yield Sentence(sentence, "llm")


def split_sentences(text: str) -> list[str]:
    """Speak code-written text sentence by sentence too, so TTS starts on the first clause."""
    chunker = SentenceChunker()
    return [*chunker.push(text), *chunker.flush()] or [text]


async def _prepend(first: asyncio.Future[Any], rest: AsyncIterator[Any]) -> AsyncIterator[Any]:
    try:
        yield await first
    except StopAsyncIteration:
        return
    async for item in rest:
        yield item


def _json(value: JsonDict) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
