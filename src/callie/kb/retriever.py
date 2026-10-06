"""FAQ retrieval: BM25 over the sections of the clinic's Markdown knowledge base.

A dozen short sections do not need embeddings: BM25 with light stemming and a few phone-language synonyms
("open" -> hours, "cost" -> price) finds the right section, runs in microseconds, needs no model download
and is deterministic, which keeps the tests and the fake model exact. Below `min_score` the tool says it has
no answer, and the agent offers to take a message instead of guessing.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "please",
        "so",
        "that",
        "the",
        "there",
        "this",
        "to",
        "us",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "i'd",
        "i'm",
        "id",
        "im",
        "about",
        "any",
        "tell",
        "know",
        "want",
        "like",
        "just",
    ]
)
SYNONYMS = {
    "open": "hours",
    "close": "hours",
    "closed": "hours",
    "opening": "hours",
    "time": "hours",
    "sunday": "hours",
    "saturday": "hours",
    "weekend": "hours",
    "lunch": "hours",
    "cost": "price",
    "costs": "price",
    "much": "price",
    "charge": "price",
    "fee": "price",
    "expensive": "price",
    "pricing": "price",
    "prices": "price",
    "pay": "payment",
    "card": "payment",
    "cash": "payment",
    "financing": "payment",
    "plan": "payment",
    "address": "location",
    "where": "location",
    "located": "location",
    "directions": "location",
    "bus": "location",
    "park": "parking",
    "car": "parking",
    "garage": "parking",
    "insurer": "insurance",
    "covered": "insurance",
    "coverage": "insurance",
    "delta": "insurance",
    "cigna": "insurance",
    "aetna": "insurance",
    "metlife": "insurance",
    "guardian": "insurance",
    "ppo": "insurance",
    "hmo": "insurance",
    "medicaid": "insurance",
    "kid": "children",
    "kids": "children",
    "child": "children",
    "son": "children",
    "daughter": "children",
    "spanish": "languages",
    "language": "languages",
    "wheelchair": "accessibility",
    "elevator": "accessibility",
    "braces": "orthodontics",
    "cancel": "cancellation",
    "late": "cancellation",
    "miss": "cancellation",
    "text": "reminders",
    "sms": "reminders",
    "reminder": "reminders",
    "whiten": "whitening",
    "bleach": "whitening",
    "new": "new",
    "first": "new",
    "bring": "new",
}


def _stem(token: str) -> str:
    for suffix in ("ings", "ing", "ies", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in re.findall(r"[a-z0-9']+", text.lower()):
        if raw in STOPWORDS:
            continue
        tokens.append(_stem(raw))
        if raw in SYNONYMS:
            tokens.append(_stem(SYNONYMS[raw]))
    return tokens


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


def split_markdown(markdown: str) -> list[Passage]:
    passages = []
    for block in re.split(r"^## ", markdown, flags=re.MULTILINE)[1:]:
        title, _, body = block.partition("\n")
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        passages.append(Passage(id=slug, title=title.strip(), text=" ".join(body.split())))
    return passages


class KnowledgeBase:
    def __init__(self, passages: list[Passage], *, k1: float = 1.4, b: float = 0.6, min_score: float = 2.0) -> None:
        self.passages = passages
        self.k1, self.b, self.min_score = k1, b, min_score
        # The title counts twice: it is the best summary of what a section answers.
        self._docs = [tokenize(f"{p.title} {p.title} {p.text}") for p in passages]
        self._lengths = [len(d) for d in self._docs]
        self._avg = sum(self._lengths) / max(len(self._docs), 1)
        self._tf = [Counter(d) for d in self._docs]
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(set(doc))
        n = len(self._docs)
        self._idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    @classmethod
    def from_markdown(cls, markdown: str) -> KnowledgeBase:
        return cls(split_markdown(markdown))

    def search(self, query: str, k: int = 2) -> list[Hit]:
        terms = tokenize(query)
        scores = []
        for index, tf in enumerate(self._tf):
            score = 0.0
            for term in set(terms):
                if term not in tf:
                    continue
                freq = tf[term]
                norm = freq + self.k1 * (1 - self.b + self.b * self._lengths[index] / self._avg)
                score += self._idf[term] * freq * (self.k1 + 1) / norm
            scores.append(score)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [Hit(self.passages[i], round(scores[i], 3)) for i in ranked[:k] if scores[i] >= self.min_score]
