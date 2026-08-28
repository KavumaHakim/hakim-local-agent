"""The one object the rest of the application talks to about memory.

Everything below it does one job - the store persists, the vectors embed,
retrieval ranks, extraction proposes, consolidation tidies, the processor runs
the model. This decides the order, and owns the two policies that would
otherwise be scattered:

**When a model is needed, and when it is not.** Storing, retrieving, ranking,
decaying, deduplicating and forgetting all happen here with no model at all.
The only path that reaches `MemoryProcessor` is a queued job, and the only
thing that starts one is `maybe_process`, which the API's idle sweeper calls.

**What gets queued, and what does not.** `observe_turn` is called after every
turn and is deliberately cheap: it rejects noise with a regex, stores anything
explicit immediately, and only queues work when a turn actually looks like it
contains something. A queue that filled on every "thanks" would force a model
switch every few minutes for nothing.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from memory import consolidation, extraction, retrieval
from memory.context import ContextBuilder
from memory.store import MemoryStore
from memory.types import (
    Candidate,
    JobKind,
    Memory,
    MemoryStatus,
    MemoryType,
    ScoredMemory,
    WorkingMemory,
)

VECTOR_DIRNAME = "memory"


class MemoryError_(Exception):
    """A memory operation was refused."""


# Named so callers can `except memory.manager.MemoryOperationError` without
# colliding with the builtin MemoryError, which means something very different.
MemoryOperationError = MemoryError_


class MemoryManager:
    """Persistent memory: store it, find it, tidy it, and know when to think."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        store_dir: str | Path | None = None,
        embedder=None,
        dimension: int = 384,
        top_k: int = 5,
        score_floor: float = retrieval.DEFAULT_SCORE_FLOOR,
        min_similarity: float = retrieval.DEFAULT_MIN_SIMILARITY,
        context_tokens: int = 3000,
        summarize_after: int = 24,
        extract_every: int = 4,
        queue_high_water: int = 6,
    ) -> None:
        self.store = MemoryStore(db_path)
        self.top_k = max(1, int(top_k))
        self.score_floor = float(score_floor)
        self.min_similarity = float(min_similarity)
        self.summarize_after = max(4, int(summarize_after))
        self.extract_every = max(1, int(extract_every))
        self.queue_high_water = max(1, int(queue_high_water))
        self.builder = ContextBuilder(budget_tokens=context_tokens)
        self.working = WorkingMemory()

        self._dimension = dimension
        self._store_dir = Path(store_dir or Path(db_path).parent / VECTOR_DIRNAME)
        self._embedder = embedder
        self._vectors = None
        self._vectors_lock = threading.Lock()
        # Set by `attach_processor` once the model manager exists. Absent is a
        # supported state, not an error: everything except LLM-backed
        # extraction works without it.
        self.processor = None

        # A crash mid-batch leaves jobs RUNNING and nothing would pick them up.
        self.store.reset_running_jobs()

    # --- wiring ---

    @property
    def vectors(self):
        """The memory vector index, built on first use."""
        if self._vectors is None:
            with self._vectors_lock:
                if self._vectors is None:
                    from memory.vectors import MemoryVectors

                    self._vectors = MemoryVectors(
                        self._store_dir,
                        self.store,
                        embedder=self._embedder,
                        dimension=self._dimension,
                    )
        return self._vectors

    @property
    def embeddings_available(self) -> bool:
        try:
            return self.vectors.available
        except Exception:  # noqa: BLE001 - a missing dependency is not a crash
            return False

    def attach_processor(self, processor) -> None:
        """Give the manager an auxiliary model to work with. Optional."""
        self.processor = processor

    # --- explicit user control (section 20/21) ---

    def remember(
        self,
        content: str,
        *,
        type: str | MemoryType = MemoryType.FACT,
        importance: float = 0.8,
        confidence: float = 0.95,
        subject: str = "",
        conversation_id: int | None = None,
    ) -> dict[str, Any]:
        """Store something now. No queue, no model, no waiting.

        An explicit instruction has to take effect immediately - telling the
        user "I'll remember that" while the memory sits in a queue behind a
        model switch would be a lie by the time they asked about it.
        """
        kind = _as_type(type)
        candidate = Candidate(
            content=content,
            type=kind,
            importance=importance,
            confidence=confidence,
            subject=subject,
            source_conversation=conversation_id,
            origin="explicit",
        )
        try:
            memory_id, created = self.store.add(candidate)
        except ValueError as exc:
            raise MemoryOperationError(str(exc)) from None

        self._embed_quietly()
        # The model may sharpen the type later, but only if it happens to run.
        # Nothing depends on it.
        if self.processor is not None and kind is MemoryType.FACT:
            self.store.enqueue(
                JobKind.CLASSIFY, {"memory_id": memory_id}, conversation_id
            )

        memory = self.store.get(memory_id)
        return {
            "success": True,
            "stored": created,
            "reinforced": not created,
            "memory": memory.as_dict() if memory else {"id": memory_id},
        }

    def recall(
        self, query: str = "", *, limit: int | None = None
    ) -> dict[str, Any]:
        """Retrieve memories. Semantic when possible, listed when not."""
        limit = max(1, int(limit or self.top_k))
        query = (query or "").strip()

        if not query:
            memories = self.store.list_memories(limit=limit)
            return {
                "success": True,
                "query": "",
                "count": len(memories),
                "memories": [memory.as_dict() for memory in memories],
            }

        scored = self.search(query, limit=limit)
        if scored:
            self.store.touch([item.memory.id for item in scored])
        return {
            "success": True,
            "query": query,
            "count": len(scored),
            "memories": [item.as_dict() for item in scored],
            **(
                {}
                if scored
                else {"note": "Nothing stored is relevant to that."}
            ),
        }

    def search(self, query: str, *, limit: int | None = None) -> list[ScoredMemory]:
        """Rank memories against a query. Never loads a chat model.

        Falls back to substring search when embeddings are unavailable, which
        keeps the tool working rather than failing when sentence-transformers
        is not installed.
        """
        limit = max(1, int(limit or self.top_k))
        hits: list[tuple[Memory, float]] = []

        if self.embeddings_available:
            try:
                # Ask for more than needed: ranking reorders, and the floor
                # will cut what does not deserve to be there.
                hits = self.vectors.search(query, top_k=limit * 4)
            except Exception:  # noqa: BLE001 - fall through to text search
                hits = []

        if not hits:
            found = self.store.search_text(query, limit=limit * 4)
            # A substring hit has no cosine to report. It is given a value just
            # above the similarity gate rather than a fixed number: the gate is
            # calibrated for the embedding model's noise floor, and a literal
            # text match should never be filtered by it. Ranking still applies
            # importance, confidence and recency on top.
            hits = [(memory, max(0.6, self.min_similarity)) for memory in found]

        return retrieval.rank(
            hits,
            limit=limit,
            floor=self.score_floor,
            min_similarity=self.min_similarity,
        )

    def update(self, memory_id: int, **changes) -> dict[str, Any]:
        """Change a stored memory."""
        if "type" in changes and changes["type"] is not None:
            changes["type"] = _as_type(changes["type"])
        if "status" in changes and changes["status"] is not None:
            changes["status"] = _as_status(changes["status"])

        try:
            changed = self.store.update(memory_id, **changes)
        except ValueError as exc:
            raise MemoryOperationError(str(exc)) from None
        if not changed:
            raise MemoryOperationError(f"No memory with id {memory_id}.")

        self._embed_quietly()
        memory = self.store.get(memory_id)
        return {
            "success": True,
            "memory": memory.as_dict() if memory else {"id": memory_id},
        }

    def forget(
        self, target: str | int, *, hard: bool = False
    ) -> dict[str, Any]:
        """Forget one memory by id, or everything about a subject.

        A bare word is treated as a subject *and* as a search: "forget
        everything about Biology.pdf" should work whether that name was stored
        as a subject or only mentioned in the content.
        """
        if isinstance(target, int) or (
            isinstance(target, str) and target.strip().isdigit()
        ):
            memory_id = int(target)
            memory = self.store.get(memory_id)
            if memory is None:
                raise MemoryOperationError(f"No memory with id {memory_id}.")
            self.store.forget(memory_id, hard=hard)
            return {"success": True, "forgotten": 1, "memories": [memory.as_dict()]}

        text = str(target).strip()
        if not text:
            raise MemoryOperationError("Say what to forget.")

        removed = self.store.forget_subject(text, hard=hard)
        gone: list[dict] = []

        # Then anything whose content matches, which is what "forget that I
        # prefer X" actually means.
        for item in self.search(text, limit=10):
            if item.similarity < 0.55:
                continue
            if self.store.forget(item.memory.id, hard=hard):
                gone.append(item.memory.as_dict())
                removed += 1

        if not removed:
            raise MemoryOperationError(f"Nothing stored about {text!r}.")
        return {"success": True, "forgotten": removed, "memories": gone}

    # --- the turn hooks ---

    def observe_turn(
        self,
        *,
        prompt: str,
        answer: str = "",
        conversation_id: int | None = None,
        message_count: int = 0,
    ) -> dict[str, Any]:
        """Called after a turn. Cheap, deterministic, and usually does nothing.

        This is the guard that keeps the model switch rare. It runs a regex or
        two, stores anything explicit, and only queues a job when the turn
        looks like it might contain something worth the auxiliary model's time.
        """
        outcome: dict[str, Any] = {"stored": 0, "queued": 0}
        prompt = (prompt or "").strip()
        if not prompt:
            return outcome

        # 1. An explicit instruction is honoured now, not queued.
        explicit = extraction.explicit_request(prompt)
        if explicit:
            self.remember(explicit, conversation_id=conversation_id)
            outcome["stored"] += 1
            outcome["explicit"] = explicit
            return outcome

        # 2. Obvious noise never reaches the queue.
        if extraction.is_noise(prompt):
            return outcome

        # 3. Rules that are certain store directly - no model needed for
        #    "I always use X".
        for candidate in extraction.heuristic(prompt, conversation_id):
            try:
                _, created = self.store.add(candidate)
            except ValueError:
                continue
            if created:
                outcome["stored"] += 1

        # 4. Anything else worth a look is queued for the model, but only
        #    every few turns: extraction over one turn in isolation is both
        #    expensive and worse than over a stretch of them.
        if self.processor is not None and message_count and (
            message_count % self.extract_every == 0
        ):
            self.store.enqueue(
                JobKind.EXTRACT,
                {"turns": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": (answer or "")[:1200]},
                ]},
                conversation_id,
            )
            outcome["queued"] += 1

        if outcome["stored"]:
            self._embed_quietly()
        return outcome

    def note_document(self, name: str, conversation_id: int | None = None) -> None:
        """Record that a document joined the collection.

        The event only. The document's text belongs to the RAG index, and
        copying it here would duplicate the one thing already searchable.
        """
        try:
            self.store.add(extraction.document_event(name, conversation_id))
        except ValueError:
            return
        self._embed_quietly()

    def queue_summary(
        self, conversation_id: int, messages: list[dict[str, Any]]
    ) -> bool:
        """Queue a summarisation when a conversation has outgrown its window."""
        if self.processor is None or len(messages) < self.summarize_after:
            return False

        existing = self.store.get_summary(conversation_id)
        covered = int(existing["covers_up_to"]) if existing else 0

        # Only the part not already summarised, and not the recent window the
        # context builder is still sending verbatim.
        older = messages[: -self.builder.min_recent_messages] or []
        fresh = [
            message for message in older if int(message.get("id", 0)) > covered
        ]
        if len(fresh) < self.extract_every:
            return False

        self.store.enqueue(
            JobKind.SUMMARIZE,
            {
                "turns": [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in fresh
                ],
                "covers_up_to": max(int(m.get("id", 0)) for m in fresh),
            },
            conversation_id,
        )
        return True

    def summary_for(self, conversation_id: int | None) -> str:
        if conversation_id is None:
            return ""
        stored = self.store.get_summary(conversation_id)
        return str(stored["summary"]) if stored else ""

    # --- consolidation (deterministic first) ---

    def consolidate(self, *, limit: int = 200) -> dict[str, Any]:
        """Tidy the store. Merges what code can; queues the rest for the model."""
        memories = self.store.list_memories(limit=limit)
        if len(memories) < 2:
            return {"success": True, "merged": 0, "superseded": 0, "queued": 0}

        if not self.embeddings_available:
            return {
                "success": True,
                "merged": 0,
                "superseded": 0,
                "queued": 0,
                "note": (
                    "Consolidation needs embeddings to tell memories apart. "
                    "Install the document-search dependencies to enable it."
                ),
            }

        self.vectors.sync()
        memories = [m for m in self.store.list_memories(limit=limit)
                    if m.embedding_row is not None]
        stored_vectors = self.vectors.vectors_for(memories)
        if stored_vectors is None:
            return {"success": True, "merged": 0, "superseded": 0, "queued": 0}

        outcome = consolidation.consolidate(self.store, memories, stored_vectors)

        queued = 0
        if self.processor is not None:
            for pair in outcome.needs_model[:10]:
                self.store.enqueue(
                    JobKind.CONSOLIDATE,
                    {"first_id": pair.first.id, "second_id": pair.second.id},
                )
                queued += 1

        return {"success": True, **outcome.as_dict(), "queued": queued}

    # --- processing ---

    def should_process(self, *, busy: bool = False) -> tuple[bool, str]:
        """Whether a model switch is worth making right now.

        Deliberately conservative. Responsiveness beats memory: if anyone is
        mid-turn, the answer is always no.
        """
        if busy:
            return False, "a turn is running"
        if self.processor is None:
            return False, "no auxiliary model is configured"
        if self.processor.running:
            return False, "a batch is already running"

        pending = self.store.pending_count()
        if pending == 0:
            return False, "nothing queued"
        if pending < self.queue_high_water:
            return False, f"only {pending} job(s) queued; waiting for a batch"
        return True, f"{pending} jobs queued"

    def maybe_process(
        self, *, busy: Callable[[], bool] | None = None, force: bool = False
    ) -> dict[str, Any]:
        """Run a batch if the triggers say so. Called by the idle sweeper."""
        is_busy = bool(busy()) if busy is not None else False
        if not force:
            ready, why = self.should_process(busy=is_busy)
            if not ready:
                return {"ran": False, "reason": why}
        if self.processor is None:
            return {"ran": False, "reason": "no auxiliary model is configured"}
        return self.processor.run_batch(
            reason="forced" if force else "idle", busy=busy
        )

    # --- reporting ---

    def stats(self) -> dict[str, Any]:
        counts = self.store.counts()
        available, why = (
            self.processor.available() if self.processor else (False, "not configured")
        )
        return {
            "success": True,
            **counts,
            "embeddings": self.embeddings_available,
            "processor": {
                "configured": self.processor is not None,
                "available": available,
                "reason": why,
                "running": bool(self.processor and self.processor.running),
                "last_run": dict(self.processor.last_run) if self.processor else {},
            },
        }

    def build_context(
        self,
        *,
        system_prompt: str,
        history: list[dict[str, Any]],
        query: str = "",
        conversation_id: int | None = None,
    ):
        """The full context for one turn: summary, memories, recent messages."""
        memories = self.search(query, limit=self.top_k) if query.strip() else []
        return self.builder.build(
            system_prompt=system_prompt,
            history=history,
            memories=memories,
            summary=self.summary_for(conversation_id),
            working=self.working,
        )

    # --- internals ---

    def _embed_quietly(self) -> None:
        """Keep the vector index in step, without ever failing a write.

        A memory that is stored but not yet embedded is still listed, still
        searchable by substring, and gets its vector on the next sync. Losing
        the write because the embedder was unavailable would be far worse.
        """
        if not self.embeddings_available:
            return
        try:
            self.vectors.sync()
        except Exception:  # noqa: BLE001
            pass


# --- the process-wide instance --------------------------------------------
#
# The tool registry is rebuilt for every turn. A MemoryManager is not something
# to rebuild with it: it owns the vector index and the reference to the
# auxiliary-model processor, and a fresh one per turn would re-open both and
# lose the "is a batch running" flag that stops two batches fighting over the
# single model slot.

_shared: MemoryManager | None = None
_shared_key: tuple = ()
_shared_lock = threading.Lock()


def shared_manager(
    db_path: str | Path,
    *,
    store_dir: str | Path | None = None,
    dimension: int = 384,
    top_k: int = 5,
    score_floor: float = retrieval.DEFAULT_SCORE_FLOOR,
    min_similarity: float = retrieval.DEFAULT_MIN_SIMILARITY,
    context_tokens: int = 3000,
    summarize_after: int = 24,
    extract_every: int = 4,
    queue_high_water: int = 6,
) -> MemoryManager:
    """The process-wide memory manager, built on first use.

    Rebuilt only when the settings that define the store change, so flipping a
    tool switch does not silently detach the processor.
    """
    global _shared, _shared_key

    key = (
        str(db_path), str(store_dir or ""), int(dimension), int(top_k),
        float(score_floor), float(min_similarity), int(context_tokens),
        int(summarize_after),
        int(extract_every), int(queue_high_water),
    )
    with _shared_lock:
        if _shared is not None and _shared_key == key:
            return _shared
        previous = _shared
        _shared = MemoryManager(
            db_path,
            store_dir=store_dir,
            dimension=dimension,
            top_k=top_k,
            score_floor=score_floor,
            min_similarity=min_similarity,
            context_tokens=context_tokens,
            summarize_after=summarize_after,
            extract_every=extract_every,
            queue_high_water=queue_high_water,
        )
        # A processor is attached by the runtime once, at startup; carrying it
        # across a settings change saves re-attaching it from every call site.
        if previous is not None and previous.processor is not None:
            _shared.attach_processor(previous.processor)
        _shared_key = key
        return _shared


def reset_shared() -> None:
    """Drop the process-wide manager. For tests."""
    global _shared, _shared_key
    with _shared_lock:
        _shared = None
        _shared_key = ()


def _as_type(value: str | MemoryType) -> MemoryType:
    if isinstance(value, MemoryType):
        return value
    try:
        return MemoryType(str(value).strip().lower())
    except ValueError:
        raise MemoryOperationError(
            f"{value!r} is not a memory type. Use one of: "
            f"{', '.join(item.value for item in MemoryType)}."
        ) from None


def _as_status(value: str | MemoryStatus) -> MemoryStatus:
    if isinstance(value, MemoryStatus):
        return value
    try:
        return MemoryStatus(str(value).strip().lower())
    except ValueError:
        raise MemoryOperationError(
            f"{value!r} is not a memory status. Use one of: "
            f"{', '.join(item.value for item in MemoryStatus)}."
        ) from None
