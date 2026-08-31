"""Chunk text and document metadata, kept out of the vector index.

SQLite rather than the `metadata.json` + `documents.json` pair a RAG tutorial
would reach for, for three reasons that all matter on this machine:

  * A JSON store has to be read whole and written whole. Re-indexing one file
    in a collection of five hundred would rewrite every record, and hold all of
    them in memory to do it. The requirement is to load only the metadata a
    search actually needs, and a query for five chunks should read five rows.
  * Deleting a document has to remove its chunks and free its vector rows
    together or the index goes out of step. SQLite gives that a transaction.
  * The project already keeps its state in SQLite, with one connection per
    operation because background threads are involved. This follows that.

The vector row number is the join between the two halves: `chunks.vector_row`
is an offset into the flat file that `rag.index` owns.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# Bumped when the tables change in a way an existing store cannot be read
# with. The manager turns a mismatch into "rebuild the index", never into a
# silent misread.
SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL,
    suffix      TEXT    NOT NULL,
    size_bytes  INTEGER NOT NULL,
    mtime_ns    INTEGER NOT NULL,
    sha256      TEXT    NOT NULL,
    pages       INTEGER,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    indexed_at  TEXT    NOT NULL,
    -- Provenance the document tools have to report honestly: what kind of file
    -- it was, how much text came out, and whether any of that text was read by
    -- OCR rather than taken from the file itself.
    kind        TEXT    NOT NULL DEFAULT '',
    characters  INTEGER NOT NULL DEFAULT 0,
    ocr_used    INTEGER NOT NULL DEFAULT 0,
    ocr_pages   TEXT    NOT NULL DEFAULT '',
    note        TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    page        INTEGER,
    section     TEXT,
    text        TEXT    NOT NULL,
    vector_row  INTEGER NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_vector  ON chunks(vector_row);

-- Rows in the vector file that no chunk owns any more. Reused before the file
-- is grown, so deleting and re-indexing does not leak disk.
CREATE TABLE IF NOT EXISTS free_rows (
    vector_row INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# The keyword half of search, kept apart because it is optional: FTS5 is
# compiled into every SQLite Python ships on the platforms this runs on, but
# "every" is not "guaranteed", and a missing module must degrade to
# vector-only search rather than stopping the store from opening.
#
# `content='chunks'` makes it an external-content index: FTS5 stores the
# postings and reads the text back from `chunks`, so the book is on disk once
# rather than twice. The triggers are what keep the two in step, and they have
# to exist because the writer inserts into `chunks` directly.
SEARCH_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_search_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunk_search(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_search_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunk_search(chunk_search, rowid, text)
        VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_search_update AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunk_search(chunk_search, rowid, text)
        VALUES ('delete', old.id, old.text);
    INSERT INTO chunk_search(rowid, text) VALUES (new.id, new.text);
END;
"""

# Set once the FTS index has been filled from chunks that were indexed before
# it existed. Backfilling is a one-statement rebuild and needs no model, which
# is the whole reason this did not have to bump SCHEMA_VERSION: adding keyword
# search to an existing index costs nothing, where re-embedding it would cost
# hours.
SEARCH_BUILT_KEY = "keyword_index_built"


@dataclass(frozen=True)
class Document:
    """One indexed file."""

    id: int
    path: str
    name: str
    suffix: str
    size_bytes: int
    mtime_ns: int
    sha256: str
    pages: int | None
    chunk_count: int
    indexed_at: str
    # "pdf", "docx" or "text".
    kind: str = ""
    characters: int = 0
    ocr_used: bool = False
    ocr_pages: tuple[int, ...] = ()
    # Anything the caller should know about how this was read - a scan that
    # could not be OCR'd, pages OCR returned nothing for.
    note: str = ""


@dataclass(frozen=True)
class ChunkRecord:
    """One stored chunk, as returned by a search."""

    id: int
    document_id: int
    document: str
    path: str
    ordinal: int
    page: int | None
    section: str | None
    text: str
    vector_row: int

    @property
    def chunk_id(self) -> str:
        """Stable identifier: which document, and where in it."""
        return f"{self.document_id}#{self.ordinal}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetadataStore:
    """Documents, chunk text, and the free list, in one SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)
        # False when SQLite was built without FTS5. Searching then falls back
        # to vectors alone, which is what it did before this existed.
        self.keyword_search = self._prepare_keyword_index()

    def _prepare_keyword_index(self) -> bool:
        """Create the FTS index if it is missing, and fill it if it is empty.

        The fill is for indexes built before keyword search existed: their
        chunk text is already in SQLite, so the whole migration is one FTS5
        'rebuild' and no embedding model at all.
        """
        try:
            with self._connect() as connection:
                connection.executescript(SEARCH_SCHEMA)
        except sqlite3.OperationalError:
            # No FTS5 in this build of SQLite. Not an error worth stopping
            # for - it costs recall, not correctness.
            return False

        try:
            with self._connect() as connection:
                built = connection.execute(
                    "SELECT value FROM index_meta WHERE key = ?",
                    (SEARCH_BUILT_KEY,),
                ).fetchone()
                if built is None:
                    connection.execute(
                        "INSERT INTO chunk_search(chunk_search) VALUES ('rebuild')"
                    )
                    connection.execute(
                        "INSERT INTO index_meta (key, value) VALUES (?, ?)"
                        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (SEARCH_BUILT_KEY, "1"),
                    )
        except sqlite3.OperationalError:
            return False
        return True

    def rebuild_keyword_index(self) -> None:
        """Re-derive the FTS index from the chunk text. Cheap, and no model."""
        if not self.keyword_search:
            return
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO chunk_search(chunk_search) VALUES ('rebuild')"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """One connection per operation, as the rest of the project does.

        A shared connection is not safe across the threads FastAPI runs sync
        routes on, and the cost of opening a local SQLite file is negligible
        next to embedding anything.
        """
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        # Off by default in SQLite, and the chunks -> documents cascade is what
        # makes deleting a document leave nothing behind.
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    # --- index-wide settings ---

    def settings(self) -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT key, value FROM index_meta").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def save_settings(self, values: dict[str, str]) -> None:
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO index_meta (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [(key, str(value)) for key, value in values.items()],
            )

    # --- documents ---

    def get_document(self, path: str) -> Document | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE path = ?", (path,)
            ).fetchone()
        return _document(row) if row else None

    def get_document_by_id(self, document_id: int) -> Document | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return _document(row) if row else None

    def list_documents(self) -> list[Document]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return [_document(row) for row in rows]

    def counts(self) -> tuple[int, int]:
        """(documents, chunks)."""
        with self._connect() as connection:
            documents = connection.execute(
                "SELECT COUNT(*) AS n FROM documents"
            ).fetchone()["n"]
            chunks = connection.execute(
                "SELECT COUNT(*) AS n FROM chunks"
            ).fetchone()["n"]
        return int(documents), int(chunks)

    # --- allocation ---

    def take_free_rows(self, count: int) -> list[int]:
        """Claim up to `count` reusable vector rows, removing them from the list."""
        if count <= 0:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT vector_row FROM free_rows ORDER BY vector_row LIMIT ?",
                (count,),
            ).fetchall()
            claimed = [int(row["vector_row"]) for row in rows]
            if claimed:
                connection.executemany(
                    "DELETE FROM free_rows WHERE vector_row = ?",
                    [(row,) for row in claimed],
                )
        return claimed

    def free_row_set(self) -> set[int]:
        """Every free row, for masking during a search.

        One integer per deleted-and-not-yet-reused chunk. It stays small
        because the next ingest reuses them.
        """
        with self._connect() as connection:
            rows = connection.execute("SELECT vector_row FROM free_rows").fetchall()
        return {int(row["vector_row"]) for row in rows}

    def max_vector_row(self) -> int:
        """Highest row in use or on the free list; -1 when there are none.

        Both tables are consulted because a freed row is still allocated as far
        as the file is concerned - handing it out twice would corrupt the index.
        """
        with self._connect() as connection:
            used = connection.execute(
                "SELECT MAX(vector_row) AS m FROM chunks"
            ).fetchone()["m"]
            free = connection.execute(
                "SELECT MAX(vector_row) AS m FROM free_rows"
            ).fetchone()["m"]
        return max(used if used is not None else -1, free if free is not None else -1)

    # --- writing ---

    def replace_document(
        self,
        *,
        path: str,
        name: str,
        suffix: str,
        size_bytes: int,
        mtime_ns: int,
        sha256: str,
        pages: int | None,
        chunks: list[tuple[int, int | None, str | None, str, int]],
        kind: str = "",
        characters: int = 0,
        ocr_used: bool = False,
        ocr_pages: tuple[int, ...] | list[int] = (),
        note: str = "",
    ) -> tuple[int, list[int]]:
        """Store a document and its chunks, replacing any earlier version.

        `chunks` is (ordinal, page, section, text, vector_row). Returns the
        document id and the vector rows the previous version released, which
        the caller has already accounted for.

        One transaction: a half-replaced document would leave chunks pointing
        at vectors that describe the old text.
        """
        freed: list[int] = []
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM documents WHERE path = ?", (path,)
            ).fetchone()

            if existing is not None:
                document_id = int(existing["id"])
                rows = connection.execute(
                    "SELECT vector_row FROM chunks WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
                freed = [int(row["vector_row"]) for row in rows]
                connection.execute(
                    "DELETE FROM chunks WHERE document_id = ?", (document_id,)
                )
                connection.execute(
                    "UPDATE documents SET name = ?, suffix = ?, size_bytes = ?,"
                    " mtime_ns = ?, sha256 = ?, pages = ?, chunk_count = ?,"
                    " indexed_at = ?, kind = ?, characters = ?, ocr_used = ?,"
                    " ocr_pages = ?, note = ? WHERE id = ?",
                    (
                        name, suffix, size_bytes, mtime_ns, sha256, pages,
                        len(chunks), _now(), kind, characters, int(ocr_used),
                        _pack_pages(ocr_pages), note, document_id,
                    ),
                )
            else:
                cursor = connection.execute(
                    "INSERT INTO documents (path, name, suffix, size_bytes,"
                    " mtime_ns, sha256, pages, chunk_count, indexed_at,"
                    " kind, characters, ocr_used, ocr_pages, note)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        path, name, suffix, size_bytes, mtime_ns, sha256, pages,
                        len(chunks), _now(), kind, characters, int(ocr_used),
                        _pack_pages(ocr_pages), note,
                    ),
                )
                document_id = int(cursor.lastrowid)

            connection.executemany(
                "INSERT INTO chunks"
                " (document_id, ordinal, page, section, text, vector_row)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (document_id, ordinal, page, section, text, vector_row)
                    for ordinal, page, section, text, vector_row in chunks
                ],
            )

        return document_id, freed

    def touch_document(self, document_id: int, *, mtime_ns: int, size_bytes: int) -> None:
        """Record that a file is unchanged after a cheap re-check.

        Saves hashing it again next time when only the timestamp moved.
        """
        with self._connect() as connection:
            connection.execute(
                "UPDATE documents SET mtime_ns = ?, size_bytes = ? WHERE id = ?",
                (mtime_ns, size_bytes, document_id),
            )

    def release_rows(self, rows: list[int]) -> None:
        """Put vector rows back on the free list."""
        if not rows:
            return
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO free_rows (vector_row) VALUES (?)",
                [(int(row),) for row in rows],
            )

    def delete_document(self, document_id: int) -> tuple[str, list[int]]:
        """Remove a document and its chunks. Returns (name, freed rows)."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT name FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if row is None:
                raise KeyError(document_id)
            name = str(row["name"])

            rows = connection.execute(
                "SELECT vector_row FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchall()
            freed = [int(entry["vector_row"]) for entry in rows]

            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            connection.executemany(
                "INSERT OR IGNORE INTO free_rows (vector_row) VALUES (?)",
                [(entry,) for entry in freed],
            )
        return name, freed

    # --- reading ---

    def search_text(self, query: str, limit: int) -> list[int]:
        """Vector rows whose chunk text best matches `query`, best first.

        BM25, which is what FTS5 ranks with, and it is good at exactly what
        embeddings are worst at: a term that has to appear rather than be
        alluded to. "E2", a chapter number, a formula, a surname. The
        embedding model has no idea those are special; an inverted index does
        not need to know.

        Rows rather than scores, because the caller fuses this with the vector
        ranking by position. BM25 scores and cosine similarities are not on
        any common scale and pretending otherwise - by normalising them into
        one number - would invent a precision neither has.
        """
        if not self.keyword_search:
            return []
        expression = _fts_query(query)
        if not expression:
            return []
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT c.vector_row AS vector_row"
                    " FROM chunk_search s"
                    " JOIN chunks c ON c.id = s.rowid"
                    " WHERE chunk_search MATCH ?"
                    " ORDER BY bm25(chunk_search) LIMIT ?",
                    (expression, int(limit)),
                ).fetchall()
        except sqlite3.OperationalError:
            # A query FTS5 could not parse. The terms are quoted before they
            # get here, so this should not happen - and if it does, losing the
            # keyword half is better than losing the search.
            return []
        return [int(row["vector_row"]) for row in rows]

    def chunks_by_rows(self, rows: list[int]) -> dict[int, ChunkRecord]:
        """Fetch the chunks behind a set of search hits, keyed by vector row.

        Only the handful of rows a search returned are read; the rest of the
        collection is never touched.
        """
        if not rows:
            return {}
        placeholders = ",".join("?" for _ in rows)
        with self._connect() as connection:
            found = connection.execute(
                "SELECT c.*, d.name AS document, d.path AS path"
                " FROM chunks c JOIN documents d ON d.id = c.document_id"
                f" WHERE c.vector_row IN ({placeholders})",
                [int(row) for row in rows],
            ).fetchall()
        return {int(row["vector_row"]): _chunk(row) for row in found}

    def all_chunk_rows(self) -> list[tuple[int, int]]:
        """(chunk id, vector row) for everything, ordered. Used by rebuild."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, vector_row FROM chunks ORDER BY vector_row"
            ).fetchall()
        return [(int(row["id"]), int(row["vector_row"])) for row in rows]

    def remap_rows(self, mapping: dict[int, int]) -> None:
        """Apply a compaction's old-row -> new-row mapping, and clear the free list."""
        with self._connect() as connection:
            # Two passes through a temporary offset, because a direct update
            # can collide with a row number that has not moved yet and
            # vector_row is UNIQUE.
            offset = max(mapping.values(), default=0) + 1_000_000
            connection.executemany(
                "UPDATE chunks SET vector_row = ? WHERE vector_row = ?",
                [(new + offset, old) for old, new in mapping.items()],
            )
            connection.execute("UPDATE chunks SET vector_row = vector_row - ?", (offset,))
            connection.execute("DELETE FROM free_rows")

    def clear(self) -> None:
        """Empty the store. The vector file is cleared by the caller."""
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
            connection.execute("DELETE FROM free_rows")


# Anything that is not a letter, digit or underscore separates terms. FTS5
# has its own query language - AND, OR, NOT, NEAR, prefix *, column filters -
# and a question typed by a person is not written in it.
_TERM = re.compile(r"[^\W_]+", re.UNICODE)


def _fts_query(query: str) -> str:
    """Turn a human question into an FTS5 expression that cannot misfire.

    Every term is extracted and quoted, then joined with OR. Quoting is what
    makes this safe: a query containing NEAR, a bare hyphen or an unbalanced
    quote would otherwise be a syntax error at best and a different search at
    worst, and the person typing it meant none of that.

    OR rather than AND because BM25 already rewards a chunk that matches more
    of the terms. Requiring all of them would make a five-word question find
    nothing, which is the failure this whole feature exists to fix.
    """
    terms = [term for term in _TERM.findall(query) if len(term) > 1]
    if not terms:
        return ""
    # A long question contributes nothing after the first few dozen terms and
    # makes the query slower to plan.
    return " OR ".join(f'"{term}"' for term in terms[:32])


def _pack_pages(pages: tuple[int, ...] | list[int]) -> str:
    """Page numbers as a comma-separated string.

    A tiny list that is only ever read back whole, so it does not earn a table
    of its own the way chunks do.
    """
    return ",".join(str(int(page)) for page in pages)


def _unpack_pages(raw: str) -> tuple[int, ...]:
    if not raw:
        return ()
    numbers = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            numbers.append(int(part))
    return tuple(numbers)


def _column(row: sqlite3.Row, name: str, default):
    """Read a column that may be absent from an older store.

    `sqlite3.Row` raises IndexError rather than returning None for a name it
    does not have, and a store written before these columns existed still has
    to be readable enough to report that it needs rebuilding.
    """
    try:
        value = row[name]
    except (IndexError, KeyError):
        return default
    return default if value is None else value


def _document(row: sqlite3.Row) -> Document:
    return Document(
        id=int(row["id"]),
        path=str(row["path"]),
        name=str(row["name"]),
        suffix=str(row["suffix"]),
        size_bytes=int(row["size_bytes"]),
        mtime_ns=int(row["mtime_ns"]),
        sha256=str(row["sha256"]),
        pages=int(row["pages"]) if row["pages"] is not None else None,
        chunk_count=int(row["chunk_count"]),
        indexed_at=str(row["indexed_at"]),
        kind=str(_column(row, "kind", "")),
        characters=int(_column(row, "characters", 0)),
        ocr_used=bool(_column(row, "ocr_used", 0)),
        ocr_pages=_unpack_pages(str(_column(row, "ocr_pages", ""))),
        note=str(_column(row, "note", "")),
    )


def _chunk(row: sqlite3.Row) -> ChunkRecord:
    section = _column(row, "section", None)
    return ChunkRecord(
        id=int(row["id"]),
        document_id=int(row["document_id"]),
        document=str(row["document"]),
        path=str(row["path"]),
        ordinal=int(row["ordinal"]),
        page=int(row["page"]) if row["page"] is not None else None,
        section=str(section) if section else None,
        text=str(row["text"]),
        vector_row=int(row["vector_row"]),
    )
