"""The vector index: one flat file of float32, searched with numpy.

**Why not FAISS.** FAISS earns its keep when an exhaustive scan is too slow,
which starts somewhere around a million vectors. A personal document
collection is three to four orders of magnitude short of that: 20,000 chunks
at 384 dimensions is 29 MB, and one matrix multiply against it takes a few
milliseconds on this CPU. Against that, faiss-cpu is a binary wheel to install,
a second file format to keep in step with the metadata, and an index that has
to be rebuilt to delete anything. numpy is already a dependency, and the
brute-force scan is exact - there is no recall lost to an approximation.

**Memory.** The file is opened as a memory map and read in blocks, so the
resident cost of a search is one block (about 6 MB), not the whole index. That
is what makes "do not load the entire document collection into RAM" true rather
than aspirational.

**Deletes.** Rows are never moved, because moving one would invalidate every
`vector_row` in the metadata store. A deleted chunk's row goes on a free list
and is overwritten by the next chunk added. `compact()` reclaims the file only
when asked, during a rebuild.
"""

from __future__ import annotations

import errno
from pathlib import Path

import numpy as np

BYTES_PER_FLOAT = 4

# Rows per block during a search. 4096 x 384 x 4 bytes is about 6 MB, which is
# small enough to stay out of the way of a 5 GB model and large enough that the
# per-block overhead does not show.
SEARCH_BLOCK_ROWS = 4096

# The on-disk dtype, stated explicitly: little-endian float32, so an index
# written on one machine reads the same on another.
DTYPE = np.dtype("<f4")


# Not called IndexError: shadowing the builtin in a module named `index` is
# exactly the kind of thing that turns a small bug into a confusing one.
class VectorIndexError(Exception):
    """The vector file is missing, unreadable, or inconsistent."""


class VectorIndex:
    """A growable array of unit-length vectors, stored in one file."""

    def __init__(self, path: str | Path, dimension: int) -> None:
        self.path = Path(path)
        self.dimension = int(dimension)
        if self.dimension <= 0:
            raise VectorIndexError(f"Dimension must be positive, got {dimension}.")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --- shape ---

    @property
    def row_bytes(self) -> int:
        return self.dimension * BYTES_PER_FLOAT

    def capacity(self) -> int:
        """How many rows the file currently holds."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return 0
        except OSError as exc:
            raise VectorIndexError(f"Could not read the vector index: {exc}") from None

        rows, remainder = divmod(size, self.row_bytes)
        if remainder:
            raise VectorIndexError(
                f"The vector index is corrupt: {size} bytes is not a whole "
                f"number of {self.dimension}-dimensional vectors. Rebuild it."
            )
        return rows

    def verify(self, expected_rows: int) -> None:
        """Check the file against what the metadata store believes.

        Called before a search rather than after a failure, so a truncated
        index is reported as something to rebuild instead of returning
        confidently wrong neighbours.
        """
        rows = self.capacity()
        if rows < expected_rows:
            raise VectorIndexError(
                f"The vector index holds {rows} vectors but the metadata "
                f"expects at least {expected_rows}. It is out of step - "
                f"rebuild the index."
            )

    # --- writing ---

    def write(self, rows: list[int], vectors: np.ndarray) -> None:
        """Write `vectors` at the given row numbers, growing the file if needed.

        Rows may be anywhere: reused free rows are scattered, appended ones are
        contiguous. Both are handled by seeking.
        """
        if len(rows) != len(vectors):
            raise VectorIndexError(
                f"{len(rows)} row numbers for {len(vectors)} vectors."
            )
        if not rows:
            return
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise VectorIndexError(
                f"Expected vectors of width {self.dimension}, got "
                f"{vectors.shape}."
            )

        payload = np.ascontiguousarray(vectors, dtype=DTYPE)
        needed = (max(rows) + 1) * self.row_bytes

        try:
            if not self.path.exists():
                self.path.touch()
            with self.path.open("r+b") as handle:
                # Growing by seeking past the end leaves a hole that reads as
                # zeros. Any row in it is unreachable anyway - nothing points
                # at a row the metadata store has not allocated.
                handle.seek(0, 2)
                if handle.tell() < needed:
                    handle.truncate(needed)

                # Consecutive rows are written in one go. Appends are the
                # common case and they are always consecutive, so this turns a
                # per-chunk write into a per-batch one.
                start = 0
                for position in range(1, len(rows) + 1):
                    contiguous = (
                        position < len(rows) and rows[position] == rows[position - 1] + 1
                    )
                    if contiguous:
                        continue
                    handle.seek(rows[start] * self.row_bytes)
                    handle.write(payload[start:position].tobytes())
                    start = position
                handle.flush()
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise VectorIndexError(
                    "The disk is full, so the index could not be written. Free "
                    "some space and index again."
                ) from None
            raise VectorIndexError(f"Could not write the vector index: {exc}") from None

    def compact(self, keep: list[int]) -> dict[int, int]:
        """Rewrite the file keeping only `keep`, in order.

        Returns the old row -> new row mapping so the metadata store can be
        updated in the same transaction. Used by `rebuild`, never during normal
        operation: this is the one operation that invalidates row numbers.
        """
        if not keep:
            self.clear()
            return {}

        source = self._open()
        temporary = self.path.with_name(self.path.name + ".compact")
        mapping: dict[int, int] = {}
        try:
            with temporary.open("wb") as handle:
                for position, old in enumerate(keep):
                    handle.write(source[old].astype(DTYPE, copy=False).tobytes())
                    mapping[old] = position
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            if exc.errno == errno.ENOSPC:
                raise VectorIndexError(
                    "The disk is full, so the index could not be compacted."
                ) from None
            raise VectorIndexError(f"Could not compact the index: {exc}") from None
        finally:
            del source

        try:
            temporary.replace(self.path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise VectorIndexError(f"Could not replace the index: {exc}") from None
        return mapping

    def clear(self) -> None:
        """Delete the vector file. The metadata store is cleared separately."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            raise VectorIndexError(f"Could not remove the index: {exc}") from None

    # --- searching ---

    def search(
        self,
        query: np.ndarray,
        *,
        top_k: int,
        skip: frozenset[int] | set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """The `top_k` closest rows to `query`, as (row, score) by score.

        Both sides are unit-length, so the dot product *is* cosine similarity
        and the score lands in [-1, 1]. `skip` holds free rows, whose contents
        are whatever a deleted chunk left behind.
        """
        rows = self.capacity()
        if rows == 0 or top_k <= 0:
            return []

        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dimension:
            raise VectorIndexError(
                f"Query has {vector.shape[0]} dimensions, index has "
                f"{self.dimension}. The index was probably built with a "
                f"different embedding model - rebuild it."
            )

        skip = skip or frozenset()
        data = self._open()

        best_scores = np.empty(0, dtype=np.float32)
        best_rows = np.empty(0, dtype=np.int64)
        try:
            for start in range(0, rows, SEARCH_BLOCK_ROWS):
                stop = min(start + SEARCH_BLOCK_ROWS, rows)
                # np.asarray materialises just this block; the rest of the file
                # stays on disk.
                block = np.asarray(data[start:stop], dtype=np.float32)
                scores = block @ vector

                if skip:
                    local = [row - start for row in skip if start <= row < stop]
                    if local:
                        scores[local] = -np.inf

                block_rows = np.arange(start, stop, dtype=np.int64)
                # Keep only a running top_k, so peak memory is the block plus
                # k, never one score per chunk in the collection.
                scores = np.concatenate([best_scores, scores])
                block_rows = np.concatenate([best_rows, block_rows])
                if scores.size > top_k:
                    keep = np.argpartition(-scores, top_k - 1)[:top_k]
                    scores, block_rows = scores[keep], block_rows[keep]
                best_scores, best_rows = scores, block_rows
        finally:
            del data

        order = np.argsort(-best_scores, kind="stable")
        return [
            (int(best_rows[position]), float(best_scores[position]))
            for position in order
            if np.isfinite(best_scores[position])
        ]

    def score_rows(
        self, query: np.ndarray, rows: list[int] | set[int]
    ) -> dict[int, float]:
        """Cosine similarity for specific rows, rather than the closest ones.

        What keyword search needs: it finds chunks by their words and has no
        score on any scale the rest of the system uses, so the rows come back
        here to be measured the same way a semantic hit was. Reading a handful
        of rows out of a memmap is cheaper than a scan, and the alternative -
        reporting a BM25 score in a field documented as cosine similarity -
        would put two different measurements in one column.
        """
        wanted = sorted({int(row) for row in rows})
        if not wanted:
            return {}

        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dimension:
            raise VectorIndexError(
                f"Query has {vector.shape[0]} dimensions, index has "
                f"{self.dimension}. The index was probably built with a "
                f"different embedding model - rebuild it."
            )

        capacity = self.capacity()
        wanted = [row for row in wanted if 0 <= row < capacity]
        if not wanted:
            return {}

        data = self._open()
        try:
            block = np.asarray(data[wanted], dtype=np.float32)
            scores = block @ vector
        finally:
            del data
        return {row: float(score) for row, score in zip(wanted, scores)}

    def _open(self) -> np.memmap:
        """Map the vector file for reading."""
        rows = self.capacity()
        if rows == 0:
            raise VectorIndexError("The index is empty.")
        try:
            return np.memmap(
                self.path, dtype=DTYPE, mode="r", shape=(rows, self.dimension)
            )
        except (OSError, ValueError) as exc:
            raise VectorIndexError(
                f"Could not open the vector index: {exc}. Rebuild it."
            ) from None
