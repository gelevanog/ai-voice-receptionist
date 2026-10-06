"""Deterministic conversation rules that run before (and after) the model.

These are the decisions that must not depend on a model's mood: whether the caller just confirmed a read-back,
whether "mm-hm" was a backchannel or an interruption, whether to escalate (emergency, a request for a person,
anger, repeated misunderstanding), and whether a generated sentence strays into medical advice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_FILLERS = r"(?:um+|uh+|er+|erm|hmm+|oh|well|so|okay so|ah)"


def _clean(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(rf"^\s*(?:{_FILLERS}\s+)+", "", text)
    return " ".join(text.split())


class Reply(StrEnum):
    YES = "yes"  # a plain confirmation and nothing else
    YES_PLUS = "yes_plus"  # starts with yes but adds something ("yes, but make it 3 instead")
    NO = "no"
    OTHER = "other"


_YES_HEAD = (
    r"(?:yes|yeah|yea|yep|yup|ya|correct|that's correct|that is correct|that's right|that is right|right|sure|"
    r"perfect|sounds good|sounds great|exactly|absolutely|definitely|of course|please do|go ahead|go for it|"
    r"that works|that's fine|that's good|that's great|that's perfect|ok|okay|alright|all right|uh huh|mm hmm|"
    r"mhm|yes please|yes it is|it is|you got it|confirmed|book it|do it|let's do it|great|"
    r"that sounds (?:good|great|perfect|right|fine)|sounds perfect|that's (?:correct|right) yes|yes definitely|"
    r"yes that's (?:right|correct|perfect|it)|that's it|exactly right|spot on|works for me|that works for me)"
)
_YES_TAIL = (
    r"(?:\s+(?:yes|yeah|please|thanks|thank you|thank you so much|that's (?:right|correct|perfect|great|fine)|"
    r"correct|perfect|great|sounds good|go ahead|book it|do it|please book it|that works|that's it|"
    r"thanks a lot|so much|you're welcome|all good|exactly|yep|right|sure|absolutely))*"
)
_YES_FULL = re.compile(rf"^(?:{_YES_HEAD})(?:{_YES_TAIL})$")
_YES_START = re.compile(rf"^(?:{_YES_HEAD})\b")
_NO = re.compile(
    r"^(?:no|nope|nah|not quite|not really|that's wrong|that is wrong|wrong|incorrect|that's not right|"
    r"actually no|wait|hold on|hang on|not exactly|no no|cancel that|scratch that|never mind|nevermind|"
    r"actually|change)\b"
)


def classify_reply(text: str) -> Reply:
    cleaned = _clean(text)
    if not cleaned:
        return Reply.OTHER
    if _NO.match(cleaned):
        return Reply.NO
    if _YES_FULL.match(cleaned):
        return Reply.YES
    first_clause = _clean(re.split(r"[,.!?;]", text.strip(), maxsplit=1)[0])
    if _YES_START.match(cleaned) or _YES_FULL.match(first_clause):
        return Reply.YES_PLUS
    return Reply.OTHER


_BACKCHANNELS = {
    "mm hmm",
    "mhm",
    "mm",
    "mmm",
    "uh huh",
    "hmm",
    "yeah",
    "yep",
    "yes",
    "okay",
    "ok",
    "right",
    "sure",
    "got it",
    "i see",
    "alright",
    "all right",
    "uh",
    "um",
    "ah",
    "oh",
    "oh okay",
    "okay okay",
    "yeah yeah",
    "right right",
    "mm hm",
    "m hm",
    "great",
    "cool",
    "nice",
    "sounds good",
}


def is_backchannel(text: str) -> bool:
    """A short acknowledgement while the agent speaks; it should not stop the agent."""
    cleaned = _clean(text) or text.lower().strip(" .,!?")
    if not cleaned:
        return True  # breath, cough or noise the STT turned into nothing
    if len(cleaned.split()) > 3:
        return False
    if cleaned in _BACKCHANNELS or all(word in {"mm", "hmm", "uh", "huh", "yeah", "okay"} for word in cleaned.split()):
        return True
    # Speech recognizers spell "mm-hmm" many ways: "M.H.M.", "and mhm", "Mm hm", "uh huh".
    letters = re.sub(r"[^a-z]", "", cleaned)
    return bool(re.fullmatch(r"(and|so)?(m+h*m+|h?m+h+m+|u+h+h+u+h+|mh+m+|hm+)", letters))


class Escalation(StrEnum):
    EMERGENCY = "emergency"
    HUMAN_REQUESTED = "human_requested"
    ANGRY = "angry_caller"
    MISUNDERSTANDING = "repeated_misunderstanding"


_EMERGENCY = re.compile(
    r"\b(can'?t breathe|cannot breathe|trouble breathing|hard to breathe|difficulty breathing|"
    r"can'?t swallow|trouble swallowing|"
    r"(?:face|cheek|jaw|neck|eye)\s+(?:is\s+)?(?:\w+\s+)?(?:swollen|swelling|swelled)|"
    r"swelling (?:in|of|on) (?:my|his|her|the) (?:face|cheek|jaw|neck|eye)|swollen (?:face|cheek|jaw|neck)|"
    r"won'?t stop bleeding|bleeding (?:won'?t|doesn'?t|does not|will not) stop|bleeding a lot|heavy bleeding|"
    r"lots of blood|"
    r"broke (?:my|his|her) jaw|broken jaw|jaw (?:is )?broken|hit (?:in|on) the (?:face|mouth|jaw)|car accident|"
    r"fell (?:on|and hit) (?:my|his|her) (?:face|mouth)|knocked out|chest pain|high fever|passed out|unconscious|"
    r"(?:this is|it's|it is|i have|we have|i think it's|i think this is) an? (?:dental |medical |real )?emergency)\b"
)
_HUMAN = re.compile(
    r"\b(speak|talk|transfer me|connect me|put me through)\b.{0,30}\b(person|human|someone|somebody|receptionist|"
    r"real person|staff|manager|front desk|dentist|doctor|dr\.? patel|operator)\b|\b(human|real person|operator|"
    r"representative|a person please|agent please)\b"
)
_ANGER_STRONG = re.compile(
    r"\b(ridiculous|unacceptable|outrageous|furious|pissed|damn|hell|crap|bullshit|shit|fuck\w*|stupid|idiot|"
    r"useless|terrible|worst|sick of|fed up|waste of (?:my )?time|this is insane|are you kidding)\b"
)
_ANGER_SOFT = re.compile(r"\b(annoyed|frustrated|upset|angry|mad|unhappy|disappointed|complain|complaint)\b")
_GOODBYE = re.compile(
    r"^(?:(?:no|nope|nah)\s+)?(?:(?:that's|that is|that'll be|that will be)\s+(?:all|it|everything)|"
    r"(?:nothing|no)\s+(?:else|more)|i'm (?:all set|good|done)|we're (?:good|done)|"
    r"(?:ok(?:ay)?\s+)?(?:thanks|thank you)(?:\s+so much|\s+very much)?\s*"
    r"(?:bye|goodbye|have a (?:good|nice|great) (?:day|one))|"
    r"(?:ok(?:ay)?\s+)?(?:bye|goodbye|bye bye|see you|talk to you later|have a (?:good|nice|great) (?:day|one)))\b"
)
_CONFUSION = re.compile(
    r"\b(what\?|pardon|sorry what|say that again|repeat that|i don'?t understand|didn'?t understand|"
    r"that'?s not what i (?:said|meant)|you'?re not listening|no i said|i already (?:said|told you)|"
    r"you don'?t understand|are you even listening)\b"
)


@dataclass
class TurnSignals:
    escalation: Escalation | None = None
    anger: int = 0
    goodbye: bool = False
    confused: bool = False


def analyze_turn(text: str) -> TurnSignals:
    cleaned = _clean(text)
    signals = TurnSignals()
    if _EMERGENCY.search(cleaned):
        signals.escalation = Escalation.EMERGENCY
        return signals
    if _HUMAN.search(cleaned):
        signals.escalation = Escalation.HUMAN_REQUESTED
    signals.anger = 2 * len(_ANGER_STRONG.findall(cleaned)) + len(_ANGER_SOFT.findall(cleaned))
    if text.count("!") >= 2:
        signals.anger += 1
    signals.goodbye = bool(_GOODBYE.match(cleaned)) or (
        "?" not in text
        and bool(
            re.search(r"\b(bye|goodbye|have a (?:good|nice|great|wonderful) (?:day|one|evening|afternoon))\b", cleaned)
        )
        and not re.search(r"\b(but|also|one more|another|question|before you go)\b", cleaned)
    )
    signals.confused = bool(_CONFUSION.search(text.lower())) or not cleaned
    return signals


@dataclass
class EscalationState:
    """Running per-call counters; `update` returns an escalation once a threshold is crossed."""

    anger_score: int = 0
    confusion_streak: int = 0
    failures: int = 0
    anger_threshold: int = 3
    confusion_threshold: int = 2

    def update(self, signals: TurnSignals) -> Escalation | None:
        if signals.escalation is not None:
            return signals.escalation
        self.anger_score += signals.anger
        self.confusion_streak = self.confusion_streak + 1 if signals.confused else 0
        if self.anger_score >= self.anger_threshold:
            return Escalation.ANGRY
        if self.confusion_streak >= self.confusion_threshold or self.failures >= 2:
            return Escalation.MISUNDERSTANDING
        return None


# --- output guard ------------------------------------------------------------------------------------------------

_MEDICAL_ADVICE = re.compile(
    r"\b(\d+\s?(?:mg|milligrams?|ml)|ibuprofen|advil|motrin|acetaminophen|tylenol|paracetamol|aspirin|naproxen|"
    r"aleve|antibiotics?|amoxicillin|penicillin|clindamycin|orajel|benzocaine|painkillers?|pain relievers?|"
    r"take (?:two|2|one|1) (?:pills?|tablets?)|rinse with salt water|salt water rinse|clove oil|ice pack|"
    r"you (?:probably|likely|might) have an? (?:infection|abscess|cavity)|it sounds like an? (?:infection|abscess))\b",
    re.IGNORECASE,
)
_LEGAL_FINANCIAL = re.compile(
    r"\b(you should sue|legal advice|file a lawsuit|invest(?:ment)? advice|you should invest|tax deduct\w*)\b",
    re.IGNORECASE,
)
SAFE_REPLACEMENT = (
    "I'm not able to give medical advice, but I can book you in with Dr. Patel, "
    "and if it's severe or you have swelling or trouble breathing, please call 911."
)


def violates_advice_policy(sentence: str) -> bool:
    return bool(_MEDICAL_ADVICE.search(sentence) or _LEGAL_FINANCIAL.search(sentence))
