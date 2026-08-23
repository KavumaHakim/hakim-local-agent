"""The vocabulary of the memory system.

Kept in one small module so the store, the retriever, the processor and the
tools all agree on what a memory *is* without importing each other.

Two enums carry most of the design.

`MemoryType` is what a memory claims to be, and it is not decoration: it drives
retrieval weight, whether a memory can be superseded by a newer one, and
whether it decays. A PREFERENCE outranks an EVENT when both match a query,
because "what setup do I normally use" is asking about preferences.

`MemoryStatus` is why deletion is rare here. A memory that turns out to be
stale becomes SUPERSEDED and keeps pointing at the one that replaced it, so
"what did I use before?" is still answerable. Only an explicit "forget" marks
something DELETED, and even that keeps the row until a purge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MemoryType(str, Enum):
    """What kind of claim a memory makes."""

    # A durable statement about the world or the user's setup.
    FACT = "fact"
    # A durable statement about what the user wants. Weighted highest: it is
    # what "how do I normally do X" is asking for.
    PREFERENCE = "preference"
    # Something that happened, with a time. The episodic half of the system.
    EVENT = "event"
    # Something the user plans to do. Deliberately never promoted to FACT by
    # the system: "I might try X tomorrow" is not "the user uses X".
    INTENTION = "intention"
    # True now, not true for long. Decays fast and is never used to supersede.
    TEMPORARY = "temporary"
    # Extracted but not trusted. Retrievable, but ranked far down.
    UNCERTAIN = "uncertain"
    # Recognised as not worth keeping. Stored only so the extractor does not
    # keep re-proposing the same line; never retrieved.
    IRRELEVANT = "irrelevant"

    @property
    def retrievable(self) -> bool:
        return self is not MemoryType.IRRELEVANT

    @property
    def durable(self) -> bool:
        """Whether this kind of memory is meant to outlive the day."""
        return self in (MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.EVENT)


class MemoryStatus(str, Enum):
    """Where a memory sits in its life."""

    ACTIVE = "active"
    # Still true, but not worth ranking highly any more.
    ARCHIVED = "archived"
    # Replaced by a newer memory, which `superseded_by` names. Kept, because
    # "what did I use before?" is a real question.
    SUPERSEDED = "superseded"
    # The user asked for it to go.
    DELETED = "deleted"

    @property
    def retrievable(self) -> bool:
        return self in (MemoryStatus.ACTIVE, MemoryStatus.ARCHIVED)


class JobKind(str, Enum):
    """The work the auxiliary model does when it is loaded.

    All four are batched into one model session; that is the whole reason the
    queue exists rather than each being done inline.
    """

    EXTRACT = "extract"
    CLASSIFY = "classify"
    CONSOLIDATE = "consolidate"
    SUMMARIZE = "summarize"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    # Nothing was worth keeping. Distinct from DONE so the counters can say
    # "looked at 40 turns, kept 3" honestly.
    DISCARDED = "discarded"


# Retrieval weight per type. Applied on top of semantic similarity, so a
# weak-but-preference match can outrank a strong-but-temporary one.
TYPE_WEIGHT: dict[MemoryType, float] = {
    MemoryType.PREFERENCE: 1.0,
    MemoryType.FACT: 0.95,
    MemoryType.EVENT: 0.8,
    MemoryType.INTENTION: 0.6,
    MemoryType.UNCERTAIN: 0.4,
    MemoryType.TEMPORARY: 0.35,
    MemoryType.IRRELEVANT: 0.0,
}

# Half-life in days, per type: how long before recency alone halves a memory's
# score. A preference barely decays; a temporary note is nearly gone in a week.
HALF_LIFE_DAYS: dict[MemoryType, float] = {
    MemoryType.PREFERENCE: 720.0,
    MemoryType.FACT: 540.0,
    MemoryType.EVENT: 180.0,
    MemoryType.INTENTION: 30.0,
    MemoryType.UNCERTAIN: 30.0,
    MemoryType.TEMPORARY: 5.0,
    MemoryType.IRRELEVANT: 1.0,
}


@dataclass(frozen=True)
class Memory:
    """One stored memory, exactly as the database holds it."""

    id: int
    type: MemoryType
    content: str
    importance: float
    confidence: float
    created_at: str
    updated_at: str
    last_accessed: str
    access_count: int
    status: MemoryStatus
    source_conversation: int | None = None
    # Row in the memory vector file, or None when it has not been embedded yet.
    embedding_row: int | None = None
    # Set on a SUPERSEDED memory: the id of the memory that replaced it.
    superseded_by: int | None = None
    # Free-text grouping, e.g. a document name for a document event. Used to
    # answer "forget everything about X" without a semantic search.
    subject: str = ""

    def as_dict(self) -> dict:
        """The shape the tools and the API hand back."""
        payload = {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "importance": round(self.importance, 3),
            "confidence": round(self.confidence, 3),
            "status": self.status.value,
            "created_at": self.created_at,
        }
        if self.subject:
            payload["subject"] = self.subject
        if self.superseded_by is not None:
            payload["superseded_by"] = self.superseded_by
        return payload


@dataclass(frozen=True)
class ScoredMemory:
    """A memory and why retrieval chose it.

    The components are kept rather than folded into one number so the API and
    the tests can show *why* something ranked where it did - a scoring bug is
    otherwise invisible until the answers are quietly wrong.
    """

    memory: Memory
    score: float
    similarity: float
    recency: float

    def as_dict(self) -> dict:
        payload = self.memory.as_dict()
        payload["score"] = round(self.score, 4)
        payload["similarity"] = round(self.similarity, 4)
        return payload


@dataclass(frozen=True)
class Candidate:
    """A proposed memory, before anything has decided it is worth keeping.

    Produced by the deterministic extractor and by the auxiliary model. The
    two paths converge here so the store only has one thing to accept.
    """

    content: str
    type: MemoryType = MemoryType.UNCERTAIN
    importance: float = 0.5
    confidence: float = 0.5
    subject: str = ""
    source_conversation: int | None = None
    # What produced it: "explicit", "heuristic" or "model". Kept for the API's
    # counters, so "the model found this" and "a rule found this" stay apart.
    origin: str = "heuristic"


@dataclass(frozen=True)
class Job:
    """One queued unit of work for the auxiliary model."""

    id: int
    kind: JobKind
    status: JobStatus
    payload: dict
    conversation_id: int | None
    created_at: str
    attempts: int = 0
    error: str = ""


@dataclass
class WorkingMemory:
    """The current task's scratch space. Deliberately not persisted.

    Section 1 of the design: working memory lives in the active context and
    does not automatically become long-term memory. Making it a plain in-memory
    object rather than a table is what enforces that - there is no code path
    that could quietly write it to disk.
    """

    task: str = ""
    active_tools: list[str] = field(default_factory=list)
    active_files: list[str] = field(default_factory=list)
    current_document: str = ""
    notes: dict[str, str] = field(default_factory=dict)

    def clear(self) -> None:
        self.task = ""
        self.active_tools.clear()
        self.active_files.clear()
        self.current_document = ""
        self.notes.clear()

    def summary(self) -> str:
        """A short line for the prompt, or empty when there is nothing to say."""
        parts = []
        if self.task:
            parts.append(f"Current task: {self.task}")
        if self.current_document:
            parts.append(f"Working on: {self.current_document}")
        if self.active_files:
            parts.append("Files in play: " + ", ".join(self.active_files[:5]))
        return "\n".join(parts)
