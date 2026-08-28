"""Ranking memories. Pure arithmetic - no model, no network, no I/O beyond SQL.

This is section 12 of the design, and the reason it is its own module is that
it is the part most likely to be tuned. Scoring lives in one function with the
weights named as constants, so changing how memory behaves is an edit here
rather than a hunt through the retrieval path.

The score is a product rather than a weighted sum, deliberately:

    score = similarity x importance x confidence x recency x type x usage

A sum lets one strong term carry a memory that is wrong on every other axis -
a vaguely similar, long-stale, low-confidence guess can out-rank an exact
match. A product means every factor has a veto, which is the behaviour wanted
from something that will be pasted into a prompt as if it were true.

Recency is an exponential decay with a per-type half-life, so a PREFERENCE from
last year still ranks while a TEMPORARY note from last week does not. That is
memory decay (section 13) and it is arithmetic, not a background job.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from memory.types import (
    HALF_LIFE_DAYS,
    TYPE_WEIGHT,
    Memory,
    MemoryStatus,
    ScoredMemory,
)

# An archived memory is still true and still findable, but should not compete
# with active ones for a place in the prompt.
ARCHIVED_PENALTY = 0.45
# A superseded memory is history. It surfaces only when a query is really
# about the past, which in practice means a very strong similarity.
SUPERSEDED_PENALTY = 0.25

# How much being used repeatedly can lift a memory. Capped hard: usage is
# evidence, but a memory that keeps being retrieved and keeps being useless
# must not climb forever.
MAX_USAGE_BONUS = 0.25
USAGE_SATURATES_AT = 10

# Below this, a memory is not worth the prompt tokens. Applied after the full
# score, not to raw similarity, so a weak-but-certain preference survives and a
# strong-but-stale guess does not.
DEFAULT_SCORE_FLOOR = 0.10

# The raw-similarity gate, applied *before* scoring, and the number that stops
# every question dragging the whole store into the prompt.
#
# It exists because BGE-small has a high noise floor: two unrelated English
# sentences score around 0.4-0.55, not near zero. Measured here against five
# stored memories:
#
#     relevant queries    best hit 0.575 - 0.664
#     irrelevant queries  best hit 0.395 - 0.549
#
# so 0.55 separates them. Without this gate "what is photosynthesis?" retrieves
# every memory in the store at a composite score above the floor, because a
# product of middling factors is still comfortably positive.
#
# Calibrated for BAAI/bge-small-en-v1.5. A different embedding model has a
# different noise floor and this must be re-measured - MEMORY_MIN_SIMILARITY
# exists for exactly that.
#
# One known overlap: a meta-question like "what do you remember about me?"
# scores 0.493 and is cut. That is deliberate - it is a request to *list*
# memories, not to search for one, and `recall` with no query answers it.
DEFAULT_MIN_SIMILARITY = 0.55


def recency_factor(memory: Memory, *, now: datetime | None = None) -> float:
    """Exponential decay on age, with a half-life set by the memory's type.

    Measured from `updated_at`, not `created_at`: re-confirming a memory should
    make it fresh again, which is exactly what re-storing it does.
    """
    now = now or datetime.now(timezone.utc)
    stamp = _parse(memory.updated_at) or _parse(memory.created_at)
    if stamp is None:
        return 0.5  # unreadable timestamp: neither fresh nor stale

    age_days = max(0.0, (now - stamp).total_seconds() / 86400.0)
    half_life = HALF_LIFE_DAYS.get(memory.type, 180.0)
    if half_life <= 0:
        return 0.0
    return float(math.pow(0.5, age_days / half_life))


def usage_factor(memory: Memory) -> float:
    """A small multiplier for memories that keep proving useful."""
    if memory.access_count <= 0:
        return 1.0
    saturated = min(memory.access_count, USAGE_SATURATES_AT) / USAGE_SATURATES_AT
    return 1.0 + MAX_USAGE_BONUS * saturated


def status_factor(memory: Memory) -> float:
    if memory.status is MemoryStatus.ACTIVE:
        return 1.0
    if memory.status is MemoryStatus.ARCHIVED:
        return ARCHIVED_PENALTY
    if memory.status is MemoryStatus.SUPERSEDED:
        return SUPERSEDED_PENALTY
    return 0.0  # deleted


def score(
    memory: Memory, similarity: float, *, now: datetime | None = None
) -> ScoredMemory:
    """Combine every signal into one number, keeping the parts."""
    recency = recency_factor(memory, now=now)
    # Similarity can come back slightly negative from a cosine; a negative
    # factor in a product would flip the sign of everything after it.
    clamped = max(0.0, float(similarity))

    value = (
        clamped
        * max(0.0, memory.importance)
        * max(0.0, memory.confidence)
        * recency
        * TYPE_WEIGHT.get(memory.type, 0.5)
        * usage_factor(memory)
        * status_factor(memory)
    )
    return ScoredMemory(
        memory=memory, score=value, similarity=clamped, recency=recency
    )


def rank(
    hits: list[tuple[Memory, float]],
    *,
    limit: int = 5,
    floor: float = DEFAULT_SCORE_FLOOR,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    now: datetime | None = None,
) -> list[ScoredMemory]:
    """Score, filter and order a set of candidate memories.

    Two filters, and they do different jobs.

    `min_similarity` runs first, on raw similarity, and answers "is this
    memory about the question at all?". It is what makes an unrelated question
    retrieve nothing.

    `floor` runs on the composite score and answers "is this worth the prompt
    tokens?" - it drops a memory that is on-topic but stale, unconfirmed or
    trivial. A memory has to pass both.
    """
    scored = [
        score(memory, similarity, now=now)
        for memory, similarity in hits
        if similarity >= min_similarity
    ]
    kept = [item for item in scored if item.score >= floor]
    kept.sort(key=lambda item: item.score, reverse=True)
    return kept[: max(0, limit)]


def _parse(stamp: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    # Rows written before timestamps carried a zone read as naive; treating
    # them as UTC is right, because that is what _now() has always written.
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
