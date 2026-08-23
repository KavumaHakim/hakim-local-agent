"""Finding duplicates and contradictions, and resolving the easy ones in code.

Section 14 says the model is a last resort here, and the split this module
draws is:

  * **Near-identical** (cosine above `MERGE_THRESHOLD`) - merged in code. The
    survivor is the more important, more confident, more recent one; the other
    becomes SUPERSEDED and points at it. No language understanding is needed to
    decide that "User prefers vim" and "user prefers vim." are one memory.

  * **Contradictory current state** - resolved in code by recency. Two FACTs
    with the same subject that are strongly similar but not identical are the
    "user uses Qwen" / "user switched to Mistral" case. The newer one wins and
    supersedes the older, which keeps exactly one current answer and keeps the
    old one as history (section 15).

  * **Related but genuinely different** - left for the model. "User prefers X"
    and "User uses X because of Y" want to become one sentence, and no
    threshold can write that sentence.

Everything except the third bullet runs with no model loaded at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from memory.store import MemoryStore
from memory.types import Memory, MemoryStatus, MemoryType

# Cosine at or above this means "the same memory, differently typed". Set high
# because a false merge silently destroys information, while a missed one just
# leaves two rows that the model may tidy later.
MERGE_THRESHOLD = 0.94

# Similar enough to be about the same thing, different enough to be a real
# statement. This band is where conflicts and model-worthy merges live.
RELATED_THRESHOLD = 0.80

# Only these types are treated as "current state" that a newer memory can
# supersede. An EVENT is never superseded - two things can both have happened.
SUPERSEDABLE = (MemoryType.FACT, MemoryType.PREFERENCE)


@dataclass(frozen=True)
class Pair:
    """Two memories that look related, and how much."""

    first: Memory
    second: Memory
    similarity: float

    @property
    def newer(self) -> Memory:
        return (
            self.first
            if self.first.updated_at >= self.second.updated_at
            else self.second
        )

    @property
    def older(self) -> Memory:
        return self.second if self.newer is self.first else self.first


@dataclass
class Outcome:
    """What a consolidation pass did."""

    merged: int = 0
    superseded: int = 0
    # Pairs code could not resolve, handed to the model when one is available.
    needs_model: list[Pair] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.needs_model is None:
            self.needs_model = []

    def as_dict(self) -> dict:
        return {
            "merged": self.merged,
            "superseded": self.superseded,
            "needs_model": len(self.needs_model),
        }


def find_pairs(
    memories: list[Memory], vectors: np.ndarray, *, threshold: float = RELATED_THRESHOLD
) -> list[Pair]:
    """Every pair above `threshold`, strongest first.

    One matrix multiply over unit vectors, so the whole comparison is a single
    numpy call rather than a loop. At a few thousand memories that is
    milliseconds; there is no need for anything cleverer.
    """
    if len(memories) < 2 or vectors.shape[0] != len(memories):
        return []

    similarity = vectors @ vectors.T
    # Only the upper triangle: the matrix is symmetric and the diagonal is
    # every memory matching itself.
    rows, columns = np.triu_indices(len(memories), k=1)
    scores = similarity[rows, columns]

    keep = np.where(scores >= threshold)[0]
    pairs = [
        Pair(memories[int(rows[i])], memories[int(columns[i])], float(scores[i]))
        for i in keep
    ]
    pairs.sort(key=lambda pair: pair.similarity, reverse=True)
    return pairs


def consolidate(
    store: MemoryStore,
    memories: list[Memory],
    vectors: np.ndarray,
    *,
    merge_threshold: float = MERGE_THRESHOLD,
    related_threshold: float = RELATED_THRESHOLD,
) -> Outcome:
    """Resolve what code can, and report what it cannot.

    Nothing here calls a model. The pairs it gives up on are returned so the
    processor can batch them into a single auxiliary-model session, rather than
    each one triggering its own model switch.
    """
    outcome = Outcome()
    if len(memories) < 2:
        return outcome

    pairs = find_pairs(memories, vectors, threshold=related_threshold)
    # A memory can only be resolved once per pass; after it is superseded, any
    # other pair mentioning it is stale.
    settled: set[int] = set()

    for pair in pairs:
        if pair.first.id in settled or pair.second.id in settled:
            continue

        if pair.similarity >= merge_threshold:
            survivor, absorbed = _pick_survivor(pair)
            _absorb(store, survivor, absorbed)
            settled.add(absorbed.id)
            outcome.merged += 1
            continue

        if _is_conflict(pair):
            newer, older = pair.newer, pair.older
            store.supersede(older.id, newer.id)
            settled.add(older.id)
            outcome.superseded += 1
            continue

        outcome.needs_model.append(pair)

    return outcome


def _pick_survivor(pair: Pair) -> tuple[Memory, Memory]:
    """Which of two near-identical memories to keep.

    Importance first, then confidence, then recency. Longer content breaks a
    remaining tie, on the grounds that the fuller sentence is the one worth
    keeping.
    """
    def rank(memory: Memory) -> tuple:
        return (
            memory.importance,
            memory.confidence,
            memory.updated_at,
            len(memory.content),
        )

    if rank(pair.first) >= rank(pair.second):
        return pair.first, pair.second
    return pair.second, pair.first


def _absorb(store: MemoryStore, survivor: Memory, absorbed: Memory) -> None:
    """Fold one memory into another, keeping the stronger signals.

    The absorbed row is marked SUPERSEDED rather than deleted so its history
    and its links survive, and so an accidental merge is recoverable.
    """
    store.update(
        survivor.id,
        importance=max(survivor.importance, absorbed.importance),
        # Two independent statements of the same thing is real evidence, so
        # confidence rises above either on its own - but never to certainty.
        confidence=min(1.0, max(survivor.confidence, absorbed.confidence) + 0.05),
        subject=survivor.subject or absorbed.subject,
    )
    store.update(absorbed.id, status=MemoryStatus.SUPERSEDED, superseded_by=survivor.id)
    store.link(survivor.id, absorbed.id, "merged_into")


def _is_conflict(pair: Pair) -> bool:
    """Whether these two are competing claims about the same current state.

    Both must be a superseding type, both ACTIVE, and they must not be the
    same age - two facts written in the same breath are not a contradiction,
    they are one thought split over two sentences.
    """
    if pair.first.type not in SUPERSEDABLE or pair.second.type not in SUPERSEDABLE:
        return False
    if pair.first.status is not MemoryStatus.ACTIVE:
        return False
    if pair.second.status is not MemoryStatus.ACTIVE:
        return False
    if pair.first.type is not pair.second.type:
        return False
    return pair.newer.updated_at > pair.older.updated_at


def merged_content(pair: Pair) -> str:
    """A deterministic merge for a pair the model will not get to.

    Used when consolidation is asked for and no auxiliary model is available.
    Joining with a semicolon keeps both statements intact and readable, which
    is the honest fallback: it does not pretend to have understood them.
    """
    first = pair.newer.content.rstrip(" .")
    second = pair.older.content.rstrip(" .")
    if second.lower() in first.lower():
        return first
    return f"{first}; {second}"
