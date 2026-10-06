"""Turns a token stream into speakable chunks, so TTS can start on the first sentence.

The first chunk of an answer may end at a comma once it has a few words ("Sure, let me check that for you,"),
because the time to the first audio is what the caller hears as latency; later chunks end at sentence
boundaries. Abbreviations ("Dr.", "a.m."), clock times ("1:30"), prices ("$1.50") and decimals do not split.
Markdown and emoji that a chat-tuned model may emit are removed: they are not speakable.
"""

from __future__ import annotations

import re

_ABBREVIATIONS = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "st",
    "ave",
    "no",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "a.m",
    "p.m",
    "approx",
    "jr",
    "sr",
}
_BOUNDARY = re.compile(r"([.!?]+)([\"')\]]*)(\s+|$)")
_MARKDOWN = re.compile(r"(\*\*|__|`|^#+\s*|^\s*[-*•]\s+|^\s*\d+\.\s+)", re.MULTILINE)
_EMOJI = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff]")


def clean_for_speech(text: str) -> str:
    text = _MARKDOWN.sub("", text)
    text = _EMOJI.sub("", text)
    text = text.replace("&", " and ").replace("—", ", ").replace("–", " to ")
    return re.sub(r"\s+", " ", text)


class SentenceChunker:
    def __init__(self, first_chunk_min_words: int = 6, max_words: int = 28) -> None:
        self.first_chunk_min_words = first_chunk_min_words
        self.max_words = max_words
        self._buffer = ""
        self._emitted = 0

    def push(self, delta: str) -> list[str]:
        self._buffer += delta
        chunks: list[str] = []
        while True:
            chunk = self._take()
            if chunk is None:
                break
            chunks.append(chunk)
        return chunks

    def flush(self) -> list[str]:
        rest = clean_for_speech(self._buffer).strip()
        self._buffer = ""
        if rest and re.search(r"\w", rest):
            self._emitted += 1
            return [rest]
        return []

    def _take(self) -> str | None:
        text = self._buffer
        boundary = self._sentence_end(text)
        if boundary == -1:
            return None  # a "." at the very end may still be "Dr." + " Patel": wait for more text
        words = text.split()
        if self._emitted == 0 and len(words) >= self.first_chunk_min_words:
            # The first chunk may end at an earlier comma or colon: the first audio is what the caller waits for.
            for comma in re.finditer(r"[,;:]\s", text):
                if boundary is not None and comma.start() >= boundary:
                    break
                head = text[: comma.start()]
                minimum = 2 if comma.group(0).startswith(":") else self.first_chunk_min_words - 2
                if len(head.split()) >= minimum and not re.search(r"\d$", head):
                    return self._emit(comma.end())
        if boundary is not None:
            return self._emit(boundary)
        if len(words) > self.max_words:
            split_at: int | None = None
            for found in re.finditer(r"[,;]\s", text):
                split_at = found.end()
                if len(text[: found.start()].split()) >= self.max_words // 2:
                    break
            if split_at is not None:
                return self._emit(split_at)
        return None

    def _sentence_end(self, text: str) -> int | None:
        """End index of the first complete sentence; None if there is none; -1 if it is undecidable yet."""
        for match in _BOUNDARY.finditer(text):
            if not match.group(3) and match.end() == len(text):
                return -1
            before = text[: match.start()]
            last_word = re.split(r"\s+", before.strip())[-1].lower().rstrip(".") if before.strip() else ""
            if match.group(1) == "." and last_word in _ABBREVIATIONS:
                continue
            if match.group(1) == "." and len(last_word) == 1 and last_word.isalpha():
                continue  # initials: "J. Smith"
            return match.end()
        return None

    def _emit(self, end: int) -> str | None:
        chunk = clean_for_speech(self._buffer[:end]).strip()
        self._buffer = self._buffer[end:]
        if not chunk or not re.search(r"\w", chunk):
            return None if not self._buffer else self._take()
        self._emitted += 1
        return chunk
