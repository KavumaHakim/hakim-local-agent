"""Embeddings for memories, reusing the document system's machinery.

There is no second embedding stack here. `rag.index.VectorIndex` is the same
flat float32 file the document index uses, and `rag.embeddings.shared_embedder`
is the same BGE worker process - so a memory search and a document search in
the same turn load one model between them, not two.

The index lives in its own file because the two collections answer different
questions and are deleted on different schedules. Sharing one file would mean a
document rebuild had to know about memories.

**No LLM is involved in anything in this module.** BGE is an embedding model in
its own short-lived process; Mistral and the auxiliary model are untouched.
That is what lets memory retrieval run while Mistral is loaded and answering,
which is section 11 of the design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from memory.store import MemoryStore
from memory.types import Memory

VECTORS_FILE = "memory.f32"


class MemoryVectors:
    """The memory embedding index: build it, keep it in step, search it."""

    def __init__(
        self,
        store_dir: str | Path,
        store: MemoryStore,
        *,
        embedder=None,
        dimension: int = 384,
    ) -> None:
        from rag.index import VectorIndex

        self.store_dir = Path(store_dir).expanduser()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.index = VectorIndex(self.store_dir / VECTORS_FILE, dimension)
        self._embedder = embedder

    @property
    def embedder(self):
        """The shared BGE worker, built on first use."""
        if self._embedder is None:
            from rag.embeddings import shared_embedder

            self._embedder = shared_embedder()
        return self._embedder

    @property
    def available(self) -> bool:
        """Whether embedding is possible at all.

        False when sentence-transformers is not installed, which is a
        supported state: the store falls back to substring search and the rest
        of the memory system carries on.
        """
        try:
            import numpy  # noqa: F401

            from rag import embeddings  # noqa: F401
        except ImportError:
            return False
        return True

    # --- keeping the index in step ---

    def sync(self, limit: int = 256) -> int:
        """Embed any memories that do not have a vector yet.

        Returns how many were embedded. Called after writes and before a
        search, so an explicitly remembered fact is findable immediately
        rather than after the next background pass.
        """
        pending = self.store.needing_embedding(limit=limit)
        if not pending:
            return 0

        vectors = self.embedder.encode_passages([item.content for item in pending])
        rows = self._allocate(len(pending))

        from rag.index import VectorIndexError

        try:
            self.index.write(rows, vectors)
        except VectorIndexError:
            # The rows were claimed but never written; hand them back so they
            # are not lost.
            self._release(rows)
            raise

        self.store.set_embedding_rows(
            [(item.id, row) for item, row in zip(pending, rows)]
        )
        return len(pending)

    def _allocate(self, count: int) -> list[int]:
        """Claim `count` rows, reusing freed ones first.

        The boundary is read before the free list is drained, for the same
        reason as the document index: taking rows off the free list lowers the
        highest known row, and appending from there would hand out a row that
        was just claimed.
        """
        boundary = self.store.max_embedding_row() + 1
        reused = self.store.take_free_rows(count)
        fresh = list(range(boundary, boundary + count - len(reused)))
        return reused + fresh

    def _release(self, rows: Sequence[int]) -> None:
        with self.store._connect() as connection:  # noqa: SLF001 - same package
            connection.executemany(
                "INSERT OR IGNORE INTO memory_free_rows (embedding_row) VALUES (?)",
                [(int(row),) for row in rows],
            )

    # --- searching ---

    def search(self, query: str, *, top_k: int = 20) -> list[tuple[Memory, float]]:
        """The `top_k` closest memories to `query`, as (memory, similarity).

        Similarity only - importance, confidence and recency are applied by
        `memory.retrieval`, which is where all the ranking lives.
        """
        if not query.strip() or top_k <= 0:
            return []

        self.sync()
        if self.index.capacity() == 0:
            return []

        from rag.index import VectorIndexError

        free = self.store.free_row_set()
        try:
            vector = self.embedder.encode_query(query)
            hits = self.index.search(
                vector, top_k=top_k + min(len(free), 100), skip=free
            )
        except VectorIndexError:
            # A corrupt or out-of-step index must not take the memory system
            # down with it; the caller falls back to substring search.
            return []

        found = self.store.by_rows([row for row, _ in hits])
        results: list[tuple[Memory, float]] = []
        for row, similarity in hits:
            memory = found.get(row)
            if memory is None:
                continue
            results.append((memory, float(similarity)))
            if len(results) >= top_k:
                break
        return results

    def embed_one(self, text: str) -> np.ndarray:
        """One query vector, for callers doing their own comparison."""
        return self.embedder.encode_query(text)

    def vectors_for(self, memories: Sequence[Memory]) -> np.ndarray | None:
        """Stored vectors for `memories`, or None if any is missing.

        Used by consolidation, which compares memories against each other
        rather than against a query - and must not re-embed to do it.
        """
        rows = [item.embedding_row for item in memories]
        if not rows or any(row is None for row in rows):
            return None
        if self.index.capacity() == 0:
            return None

        data = self.index._open()  # noqa: SLF001 - same subsystem
        try:
            capacity = self.index.capacity()
            if any(row >= capacity for row in rows):
                return None
            return np.asarray(data[list(rows)], dtype=np.float32)
        finally:
            del data

    def clear(self) -> None:
        """Drop every vector. The store's rows are cleared by the caller."""
        from rag.index import VectorIndexError

        try:
            self.index.clear()
        except VectorIndexError:
            pass
