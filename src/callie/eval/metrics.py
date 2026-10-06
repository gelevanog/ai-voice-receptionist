"""Metrics: word error rate with a transparent text normalizer, and latency percentiles."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

_NUMBER_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "first": "1st",
    "second": "2nd",
    "third": "3rd",
    "fourth": "4th",
    "fifth": "5th",
    "sixth": "6th",
    "seventh": "7th",
    "eighth": "8th",
    "ninth": "9th",
    "tenth": "10th",
    "eleventh": "11th",
    "twelfth": "12th",
    "thirteenth": "13th",
    "fourteenth": "14th",
    "fifteenth": "15th",
    "sixteenth": "16th",
    "seventeenth": "17th",
    "eighteenth": "18th",
    "nineteenth": "19th",
    "twentieth": "20th",
    "thirtieth": "30th",
}
_SPELLING = {
    "ok": "okay",
    "mm": "mhm",
    "mmhmm": "mhm",
    "mhmm": "mhm",
    "mmhm": "mhm",
    "uhhuh": "mhm",
    "hmm": "mhm",
    "alright": "all right",
    "x-rays": "xrays",
    "x-ray": "xray",
    "checkup": "check up",
    "check-up": "check up",
    "dr": "doctor",
    "st": "street",
    "thanks": "thanks",
    "gonna": "going to",
    "wanna": "want to",
}
_FILLERS = {"um", "uh", "er", "erm", "ah", "hmm"}


def normalize_for_wer(text: str) -> list[str]:
    """Lowercase, drop punctuation and fillers, unify spellings and spoken numbers.

    Both the reference (what the simulated caller said) and the hypothesis (the STT output) go through this,
    so "3:30 p.m." and "three thirty PM" compare equal. The rules are listed here so the WER is reproducible.
    """
    text = text.lower().replace("’", "'")
    text = re.sub(r"\b([ap])\.\s?m\.?", r"\1m", text)
    text = re.sub(r"(\d)\s*(am|pm)\b", r"\1 \2", text)
    text = re.sub(r"(\d):(\d\d)", r"\1 \2", text)
    text = text.replace("$", " dollars ").replace("%", " percent ")
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = text.replace("-", " ")
    words = []
    for raw in text.split():
        word = raw.strip("'")
        if not word or word in _FILLERS:
            continue
        word = _SPELLING.get(word, word)
        word = _NUMBER_WORDS.get(word, word)
        words.extend(word.split())
    # "twenty five" -> "25", "1 30" stays (clock time)
    merged: list[str] = []
    for word in words:
        if merged and merged[-1] in {"20", "30", "40", "50"} and word.isdigit() and len(word) == 1:
            merged[-1] = str(int(merged[-1]) + int(word))
        elif (
            merged
            and merged[-1] in {"20", "30"}
            and word.endswith(("st", "nd", "rd", "th"))
            and word[:-2].isdigit()
            and len(word[:-2]) == 1
        ):
            merged[-1] = f"{int(merged[-1]) + int(word[:-2])}{word[-2:]}"
        else:
            merged.append(word)
    return [w.replace("'", "") for w in merged]


@dataclass(frozen=True)
class WerCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.reference_words if self.reference_words else float(self.errors > 0)


def wer_counts(reference: str, hypothesis: str) -> WerCounts:
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    # Levenshtein over words with backtracking to split the error types.
    rows, cols = len(ref) + 1, len(hyp) + 1
    dist = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dist[i][0] = i
    for j in range(cols):
        dist[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dist[i][j] = min(dist[i - 1][j] + 1, dist[i][j - 1] + 1, dist[i - 1][j - 1] + cost)
    i, j = len(ref), len(hyp)
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            subs += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return WerCounts(subs, dels, ins, len(ref))


def wer(reference: str, hypothesis: str) -> float:
    return wer_counts(reference, hypothesis).wer


def corpus_wer(pairs: Sequence[tuple[str, str]]) -> float:
    """Total errors over total reference words (not the mean of per-utterance WERs)."""
    counts = [wer_counts(r, h) for r, h in pairs]
    words = sum(c.reference_words for c in counts)
    return sum(c.errors for c in counts) / words if words else 0.0


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 100]); None for no data."""
    data = sorted(v for v in values if v is not None and not math.isnan(v))
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * q / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    return data[low] + (data[high] - data[low]) * (rank - low)


def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    data = [v for v in values if v is not None]
    return {
        "n": len(data),
        "p50": round(p, 3) if (p := percentile(data, 50)) is not None else None,
        "p95": round(p, 3) if (p := percentile(data, 95)) is not None else None,
        "mean": round(sum(data) / len(data), 3) if data else None,
        "max": round(max(data), 3) if data else None,
    }
