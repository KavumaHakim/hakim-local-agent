"""Deciding what is worth remembering, without a model where possible.

Two producers feed the same `Candidate` type:

  * `explicit()` - the user said "remember that ...". Stored immediately, at
    high confidence, with no queue and no model. Section 20 is explicit about
    this: an explicit instruction must not wait for a background pass, because
    the user will reasonably assume it took effect.

  * `heuristic()` - pattern matching over a turn. Catches the clear cases
    ("I prefer X", "I always use Y") and, just as importantly, *rejects* the
    obvious noise so the queue does not fill with greetings. What it is unsure
    about it does not guess at; that is what the auxiliary model is for.

The rejection half is doing more work than the extraction half. A memory system
that stores everything is worse than none: it fills the prompt with
"user said hello" and pushes out the one preference that mattered.
"""

from __future__ import annotations

import re

from memory.types import Candidate, MemoryType

# Openers that mean the user is *instructing* the agent to remember. Matched at
# the start of a message only - "I should remember to buy milk" is not one.
_EXPLICIT_REMEMBER = re.compile(
    r"^\s*(?:please\s+)?(?:remember|note|keep in mind|don't forget|do not forget)"
    r"(?:\s+that)?[:,]?\s+(?P<content>.+)",
    re.IGNORECASE | re.DOTALL,
)

_EXPLICIT_FORGET = re.compile(
    r"^\s*(?:please\s+)?forget"
    r"(?:\s+(?:that|about|everything about))?[:,]?\s+(?P<content>.+)",
    re.IGNORECASE | re.DOTALL,
)

_WHAT_DO_YOU_REMEMBER = re.compile(
    r"^\s*what\s+(?:do\s+you\s+)?(?:remember|know)\s+about\s+(?P<subject>.+?)\s*\??$",
    re.IGNORECASE,
)

# Durable preference statements. The verb list is short on purpose: "I like
# this song" is a preference in the linguistic sense and useless here, so the
# patterns require a habitual or configuring verb.
# The optional hedge group matters: "I might always use X" has to *match* so
# that the hedge check below can downgrade it to an INTENTION. Without it the
# pattern simply misses, and a hedged statement is silently dropped instead of
# being stored as the uncertain thing it is.
_HEDGE_WORDS = r"(?:might|may|could|would|will|probably|possibly|maybe|now)"

_PREFERENCE = re.compile(
    r"\bI\s+(?:" + _HEDGE_WORDS + r"\s+)?"
    r"(?:always|usually|normally|generally|typically)\s+(?P<verb>\w+)"
    r"\s+(?P<object>.{3,120})",
    re.IGNORECASE,
)

_PREFERENCE_DIRECT = re.compile(
    r"\bI\s+(?:" + _HEDGE_WORDS + r"\s+)?(?:prefer|favour|favor)"
    r"\s+(?P<object>.{3,120})",
    re.IGNORECASE,
)

# "I am using X", "I've switched to X", "my setup is X" - current state, which
# is a FACT and is the kind of thing a newer statement should supersede.
_CURRENT_STATE = re.compile(
    r"\b(?:I(?:'m| am)\s+(?:now\s+)?(?:using|running|on)|"
    r"I(?:'ve| have)\s+switched\s+to|my\s+(?:setup|machine|editor|shell)\s+is)"
    r"\s+(?P<object>.{2,120})",
    re.IGNORECASE,
)

# Hedged language. Anything matching this is at most an INTENTION, whatever
# else it looks like: "I might always use X tomorrow" is not a preference.
_HEDGED = re.compile(
    r"\b(?:might|maybe|perhaps|thinking of|considering|may\s+try|planning to|"
    r"going to try|possibly|not sure|probably)\b",
    re.IGNORECASE,
)

# Explicitly time-boxed. "today", "this week", "for now" - true, but not
# durable, so it becomes TEMPORARY and decays in days.
_TIME_BOXED = re.compile(
    r"\b(?:today|tonight|this (?:morning|afternoon|evening|week)|"
    r"right now|for now|at the moment|temporarily|just testing|"
    r"currently testing)\b",
    re.IGNORECASE,
)

# Messages that are never worth a memory, however they are phrased.
_NOISE = re.compile(
    r"^\s*(?:hi|hey|hello|yo|thanks|thank you|ta|cheers|ok|okay|k|sure|"
    r"yes|no|yep|nope|got it|nice|cool|great|lol|please|sorry|"
    r"good (?:morning|afternoon|evening|night)|bye|goodbye)"
    r"[\s!.,?]*$",
    re.IGNORECASE,
)

# A turn shorter than this carries nothing worth a database row.
MIN_MEANINGFUL_CHARS = 12
# ...and one longer than this is a document paste, not a fact about the user.
MAX_CANDIDATE_CHARS = 400


def is_noise(text: str) -> bool:
    """Whether a message is certainly not worth remembering."""
    stripped = (text or "").strip()
    if len(stripped) < MIN_MEANINGFUL_CHARS:
        return True
    return bool(_NOISE.match(stripped))


def explicit_request(text: str) -> str | None:
    """The content of an explicit 'remember that ...', or None."""
    match = _EXPLICIT_REMEMBER.match(text or "")
    if not match:
        return None
    content = " ".join(match.group("content").split())
    return content[:MAX_CANDIDATE_CHARS] or None


def forget_request(text: str) -> str | None:
    """The subject of an explicit 'forget ...', or None."""
    match = _EXPLICIT_FORGET.match(text or "")
    if not match:
        return None
    content = " ".join(match.group("content").split())
    return content[:MAX_CANDIDATE_CHARS] or None


def recall_request(text: str) -> str | None:
    """The subject of 'what do you remember about X', or None."""
    match = _WHAT_DO_YOU_REMEMBER.match(text or "")
    if not match:
        return None
    return " ".join(match.group("subject").split())[:MAX_CANDIDATE_CHARS] or None


def explicit(content: str, conversation_id: int | None = None) -> Candidate:
    """A memory the user asked for directly.

    High confidence because the user stated it, and high importance because
    they thought it worth an instruction. The type is left as FACT rather than
    guessed at - the auxiliary model refines it later if it ever runs, and a
    wrong guess here would be visible to the user immediately.
    """
    return Candidate(
        content=content.strip()[:MAX_CANDIDATE_CHARS],
        type=MemoryType.FACT,
        importance=0.8,
        confidence=0.95,
        source_conversation=conversation_id,
        origin="explicit",
    )


def heuristic(text: str, conversation_id: int | None = None) -> list[Candidate]:
    """Candidates a rule is confident about. Empty when unsure.

    Order matters: hedging and time-boxing are checked before the preference
    patterns, because "I might always use X" must not become a preference.
    """
    text = (text or "").strip()
    if is_noise(text) or len(text) > 4000:
        return []

    hedged = bool(_HEDGED.search(text))
    time_boxed = bool(_TIME_BOXED.search(text))

    candidates: list[Candidate] = []

    for pattern in (_PREFERENCE_DIRECT, _PREFERENCE):
        match = pattern.search(text)
        if not match:
            continue
        content = _sentence_around(text, match.start())
        if not content:
            continue
        if hedged:
            candidates.append(
                Candidate(
                    content=content,
                    type=MemoryType.INTENTION,
                    importance=0.3,
                    confidence=0.35,
                    source_conversation=conversation_id,
                )
            )
        elif time_boxed:
            candidates.append(
                Candidate(
                    content=content,
                    type=MemoryType.TEMPORARY,
                    importance=0.25,
                    confidence=0.5,
                    source_conversation=conversation_id,
                )
            )
        else:
            candidates.append(
                Candidate(
                    content=content,
                    type=MemoryType.PREFERENCE,
                    importance=0.75,
                    confidence=0.7,
                    source_conversation=conversation_id,
                )
            )
        break

    state = _CURRENT_STATE.search(text)
    if state and not candidates:
        content = _sentence_around(text, state.start())
        if content:
            if hedged:
                kind, importance, confidence = MemoryType.INTENTION, 0.3, 0.35
            elif time_boxed:
                kind, importance, confidence = MemoryType.TEMPORARY, 0.25, 0.5
            else:
                kind, importance, confidence = MemoryType.FACT, 0.65, 0.7
            candidates.append(
                Candidate(
                    content=content,
                    type=kind,
                    importance=importance,
                    confidence=confidence,
                    source_conversation=conversation_id,
                )
            )

    return candidates


def document_event(name: str, conversation_id: int | None = None) -> Candidate:
    """An episodic memory that a document entered the picture.

    The event only - never the document's text. Document content belongs to
    the RAG index, and copying it here would duplicate the one thing that is
    already searchable (section 4).
    """
    return Candidate(
        content=f"User added the document {name} to the searchable collection.",
        type=MemoryType.EVENT,
        importance=0.6,
        confidence=1.0,
        subject=name,
        source_conversation=conversation_id,
        origin="heuristic",
    )


def _sentence_around(text: str, position: int) -> str:
    """The sentence containing `position`, trimmed to a storable length."""
    start = max(
        text.rfind(". ", 0, position),
        text.rfind("\n", 0, position),
        text.rfind("! ", 0, position),
        text.rfind("? ", 0, position),
    )
    start = 0 if start < 0 else start + 1

    ends = [
        index
        for index in (
            text.find(". ", position),
            text.find("\n", position),
            text.find("! ", position),
            text.find("? ", position),
        )
        if index != -1
    ]
    end = min(ends) + 1 if ends else len(text)

    sentence = " ".join(text[start:end].split()).strip(" .")
    if len(sentence) < MIN_MEANINGFUL_CHARS:
        return ""
    return sentence[:MAX_CANDIDATE_CHARS]
