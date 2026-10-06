"""Minimal personal data: phone numbers and names are masked in transcripts, tool logs, call records and results.

Masking keeps what support staff need to recognize a call ("J*** D**", "***-***-4567") and nothing more.
Names the caller gave to a tool (book, take a message) are registered per call, so they are masked wherever they
appear later in that call's transcript, including in the agent's own read-backs.
"""

from __future__ import annotations

import re

_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)")
_LONG_DIGITS_RE = re.compile(r"(?<![\w-])(?:\d[\s.-]?){6,14}\d(?![\w-])")
_SPOKEN_DIGITS_RE = re.compile(
    r"\b(?:(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|double \w+)[\s,.-]+){6,}"
    r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)\b",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_INTRO_NAME_RE = re.compile(
    r"\b(my name is|my name's|this is|i'm|i am|name is|it's|under)\s+([A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+)?)"
)
_NOT_NAMES = {"Callie", "Brightside", "Dental", "Calling", "Just", "Sure", "Yes", "Fine", "Good", "Sorry", "Not"}


def digits_from_speech(text: str) -> str:
    """'five five five, one two three, four five six seven' -> '5551234567'; digits pass through."""
    tokens = re.findall(r"double \w+|triple \w+|[a-z]+|\d", text.lower())
    out = []
    for token in tokens:
        if token.startswith(("double ", "triple ")):
            word = token.split()[1]
            digit = _DIGIT_WORDS.get(word, word if word.isdigit() else "")
            out.append(digit * (2 if token.startswith("double") else 3))
        elif token.isdigit():
            out.append(token)
        elif token in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[token])
    return "".join(out)


def normalize_phone(raw: str | None) -> str | None:
    """Best-effort E.164 for North American numbers; None when there are not enough digits."""
    if not raw:
        return None
    digits = digits_from_speech(raw)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return "+1" + digits
    if raw.strip().startswith("+") and 8 <= len(digits) <= 15:
        return "+" + digits
    return None


def mask_phone(phone: str | None) -> str:
    if not phone:
        return ""
    digits = digits_from_speech(phone)
    return f"***-***-{digits[-4:]}" if len(digits) >= 4 else "***"


def mask_name(name: str | None) -> str:
    if not name:
        return ""
    return " ".join(part[0] + "*" * max(len(part) - 1, 2) for part in name.split() if part)


class Masker:
    """Masks phones, emails and the names seen in this call."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def register_name(self, name: str | None) -> None:
        if not name:
            return
        for part in name.replace(",", " ").split():
            cleaned = part.strip(".'’")
            if len(cleaned) >= 2 and cleaned[0].isalpha():
                self.names.add(cleaned)

    def mask(self, text: str) -> str:
        if not text:
            return text
        text = _EMAIL_RE.sub("[email]", text)
        text = _PHONE_RE.sub(lambda m: mask_phone(m.group(0)), text)
        text = _LONG_DIGITS_RE.sub(lambda m: mask_phone(m.group(0)), text)
        text = _SPOKEN_DIGITS_RE.sub(lambda m: mask_phone(digits_from_speech(m.group(0))), text)
        for match in _INTRO_NAME_RE.finditer(text):
            for part in match.group(2).split():
                if part not in _NOT_NAMES:
                    self.names.add(part)
        for name in sorted(self.names, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(name)}\b", mask_name(name), text, flags=re.IGNORECASE)
        return text

    def mask_value(self, value: object) -> object:
        """Mask strings inside tool arguments/results (dicts, lists) recursively."""
        if isinstance(value, str):
            return self.mask(value)
        if isinstance(value, dict):
            return {
                key: (mask_phone(str(item)) if "phone" in key and item else self.mask_value(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.mask_value(item) for item in value]
        return value
