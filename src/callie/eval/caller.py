"""The simulated caller: an LLM (a different free model than the agent's) role-plays the scenario's persona.

The caller "hears" what Callie actually said (the text of the sentences that played, trimmed after a barge-in),
answers in one or two spoken sentences, and ends with [END] when its goal is done. Its words are then spoken by a
different TTS engine and voice than Callie's and sent through the audio pipeline.
"""

from __future__ import annotations

import re

from callie.eval.scenarios import Scenario
from callie.llm.base import ChatModel, JsonDict, ProviderError, TextDelta

END = "[END]"


def caller_system_prompt(scenario: Scenario) -> str:
    facts = "; ".join(f"{k}: {v}" for k, v in scenario.facts.items()) or "none"
    return (
        "You are role-playing a person who phones Brightside Dental and talks to its AI phone receptionist, "
        "Callie. Stay in character.\n"
        f"Who you are: {scenario.persona}\n"
        f"Your goal: {scenario.goal}\n"
        f"Facts about you (use exactly these; never make up other names or numbers): {facts}\n"
        "How to reply: one or two short sentences, the way people talk on the phone. No lists, no stage "
        "directions, no quotation marks, no emojis. Say phone numbers digit by digit with pauses, like "
        '"five five five, two one four, eight eight three nine". If Callie reads details back and they are '
        "correct, confirm with a short yes. When your goal is done, or Callie says goodbye, say a short goodbye "
        f"and end your message with {END}. Never say that you are an AI or that this is a test."
    )


class Caller:
    def __init__(self, scenario: Scenario, llm: ChatModel | None, *, max_turns: int = 8) -> None:
        self.scenario = scenario
        self.llm = llm
        self.max_turns = max_turns
        self.turns = 0
        self.messages: list[JsonDict] = [{"role": "system", "content": caller_system_prompt(scenario)}]
        self.script = list(scenario.script)
        self.errors: list[str] = []

    def _prompt(self) -> list[JsonDict]:
        """The conversation as a transcript in one message: small models keep their role far better this way."""
        lines = [_transcript_line(m) for m in self.messages[1:]]
        if not self.messages[1:] or self.messages[-1]["role"] != "user":
            lines.append("Receptionist (Callie): (silence)")
        transcript = "\n".join(lines)
        return [
            self.messages[0],
            {
                "role": "user",
                "content": f"The phone call so far:\n{transcript}\n\nYou are the caller. Reply with only the "
                "words you say next (one or two short sentences).",
            },
        ]

    def heard(self, agent_text: str) -> None:
        if agent_text.strip():
            self.messages.append({"role": "user", "content": agent_text.strip()})

    def said(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    async def next_line(self) -> tuple[str, bool]:
        """(text to say, ends the call)."""
        self.turns += 1
        if self.turns == 1 and self.scenario.opening:
            self.said(self.scenario.opening)
            return self.scenario.opening, False
        if self.turns > self.max_turns:
            return "Okay, thanks, bye.", True
        if self.llm is None:
            if not self.script:
                return "Thanks, goodbye.", True
            line = self.script.pop(0)
            self.said(line)
            return line.replace(END, "").strip(), END in line or not self.script
        text = ""
        try:
            async for event in self.llm.stream(self._prompt(), [], max_tokens=1500, temperature=0.7):
                if isinstance(event, TextDelta):
                    text += event.text
        except ProviderError as exc:
            self.errors.append(str(exc)[:200])
            return "Sorry, I have to go. Bye.", True
        ended = END in text
        text = clean_caller_text(text)
        self.said(text + (f" {END}" if ended else ""))
        return text or "Okay.", ended


def _transcript_line(message: JsonDict) -> str:
    who = "Receptionist (Callie)" if message["role"] == "user" else "You"
    return f"{who}: {message['content']}"


def clean_caller_text(text: str) -> str:
    text = text.replace(END, " ")
    text = re.sub(r"\*[^*]*\*|\([^)]*\)|\[[^\]]*\]", " ", text)  # stage directions
    text = re.sub(r"^\s*(caller|me|customer)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.replace('"', "").replace("“", "").replace("”", "")
    return " ".join(text.split())
