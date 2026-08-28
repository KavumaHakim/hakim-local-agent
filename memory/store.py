"""Memories, their links, the job queue and conversation summaries.

Everything lives in the database the project already keeps - the same file as
the chat history - in its own tables, following the same connection-per-
operation rule the rest of the project uses because turns run on a worker
thread while requests are served on others.

**The legacy table.** The first version of this feature stored keyed facts in a
`memories(key, value, ...)` table. That shape cannot hold a type, a confidence
or an embedding, so this owns `memory_items` instead. On first open, the old
rows are imported as FACT memories and the old table is *renamed*, not dropped:
an upgrade should never be the thing that loses someone's data, and a rename is
reversible by hand if the import gets it wrong.

**Nothing in this module calls a model.** Every operation here is a query, a
timestamp or an arithmetic update. That is section 5 of the design and it is
the reason the memory system stays usable when no model is loaded at all.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from memory.types import (
    Candidate,
    Job,
    JobKind,
    JobStatus,
    Memory,
    MemoryStatus,
    MemoryType,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    type                TEXT    NOT NULL,
    content             TEXT    NOT NULL,
    importance          REAL    NOT NULL DEFAULT 0.5,
    confidence          REAL    NOT NULL DEFAULT 0.5,
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL,
    last_accessed       TEXT    NOT NULL,
    access_count        INTEGER NOT NULL DEFAULT 0,
    status              TEXT    NOT NULL DEFAULT 'active',
    source_conversation INTEGER,
    embedding_row       INTEGER UNIQUE,
    superseded_by       INTEGER REFERENCES memory_items(id) ON DELETE SET NULL,
    subject             TEXT    NOT NULL DEFAULT '',
    -- Lower-cased, whitespace-collapsed content. A UNIQUE index on this is
    -- what makes "store the same sentence twice" free to detect, with no
    -- embedding and no model involved.
    normalised          TEXT    NOT NULL DEFAULT ''
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_normalised
    ON memory_items(normalised) WHERE status != 'deleted';
CREATE INDEX IF NOT EXISTS idx_memory_status  ON memory_items(status);
CREATE INDEX IF NOT EXISTS idx_memory_subject ON memory_items(subject);
CREATE INDEX IF NOT EXISTS idx_memory_type    ON memory_items(type);

-- Rows in the memory vector file that no memory owns any more.
CREATE TABLE IF NOT EXISTS memory_free_rows (
    embedding_row INTEGER PRIMARY KEY
);

-- Why two memories are related: 'supersedes', 'merged_into', 'about'.
CREATE TABLE IF NOT EXISTS memory_links (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    relation  TEXT    NOT NULL,
    created_at TEXT   NOT NULL,
    UNIQUE(source_id, target_id, relation)
);

CREATE TABLE IF NOT EXISTS memory_jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'pending',
    payload         TEXT    NOT NULL DEFAULT '{}',
    conversation_id INTEGER,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    error           TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON memory_jobs(status, id);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    conversation_id  INTEGER PRIMARY KEY,
    summary          TEXT    NOT NULL,
    -- The highest message id the summary covers. Everything after it is still
    -- in the recent window and must not be summarised twice.
    covers_up_to     INTEGER NOT NULL DEFAULT 0,
    message_count    INTEGER NOT NULL DEFAULT 0,
    updated_at       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Retried this many times before a job is left FAILED. A job that fails twice
# is failing for a reason a third attempt will not fix.
MAX_JOB_ATTEMPTS = 3

MAX_CONTENT_LENGTH = 2000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalise(content: str) -> str:
    """The duplicate key for a memory: lower-cased, whitespace-collapsed.

    Punctuation is kept. "User prefers vim" and "User prefers vim." are not
    treated as the same row here - catching that pair is the embedding's job,
    and doing it with string surgery would eventually merge two things that
    only looked alike.
    """
    return " ".join(content.lower().split())


class MemoryStore:
    """Every persistent piece of the memory system, in SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)
        self._migrate_legacy()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    # --- upgrading from the keyed store ---

    def _migrate_legacy(self) -> None:
        """Import `memories(key, value)` rows, then rename the old table.

        Runs at most once: it is skipped as soon as `memories_legacy` exists.
        """
        with self._connect() as connection:
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if "memories" not in tables or "memories_legacy" in tables:
                return

            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(memories)")
            }
            if not {"key", "value"} <= columns:
                return  # not the shape we are upgrading from

            for row in connection.execute(
                "SELECT key, value, created_at, updated_at FROM memories"
            ).fetchall():
                content = f"{row['key']}: {row['value']}"
                connection.execute(
                    "INSERT OR IGNORE INTO memory_items"
                    " (type, content, importance, confidence, created_at,"
                    "  updated_at, last_accessed, access_count, status,"
                    "  subject, normalised)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'active', ?, ?)",
                    (
                        # Imported as FACT rather than PREFERENCE: the old
                        # store had no type, and guessing "preference" would
                        # invent a confidence the original data never had.
                        MemoryType.FACT.value,
                        content[:MAX_CONTENT_LENGTH],
                        0.6,
                        0.7,
                        row["created_at"] or _now(),
                        row["updated_at"] or _now(),
                        row["updated_at"] or _now(),
                        str(row["key"])[:80],
                        normalise(content)[:MAX_CONTENT_LENGTH],
                    ),
                )
            connection.execute("ALTER TABLE memories RENAME TO memories_legacy")

    # --- settings ---

    def settings(self) -> dict[str, str]:
        with self._connect() as connection:
            return {
                row["key"]: row["value"]
                for row in connection.execute("SELECT key, value FROM memory_meta")
            }

    def save_settings(self, values: dict[str, str]) -> None:
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO memory_meta (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [(key, str(value)) for key, value in values.items()],
            )

    # --- writing memories ---

    def add(self, candidate: Candidate) -> tuple[int, bool]:
        """Store a memory. Returns (id, created).

        An exact repeat of something already stored is not an error and not a
        second row: the existing memory is reinforced instead, which is both
        cheaper and more truthful than keeping two identical sentences.
        """
        content = (candidate.content or "").strip()
        if not content:
            raise ValueError("A memory needs some content.")
        if len(content) > MAX_CONTENT_LENGTH:
            raise ValueError(
                f"That memory is {len(content)} characters, over the "
                f"{MAX_CONTENT_LENGTH} limit. Store a fact, not a transcript."
            )

        key = normalise(content)
        stamp = _now()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id, importance, confidence FROM memory_items"
                " WHERE normalised = ? AND status != 'deleted'",
                (key,),
            ).fetchone()

            if existing is not None:
                # Seeing the same thing again is evidence, so confidence rises
                # towards 1 rather than being overwritten by the new guess.
                confidence = max(
                    float(existing["confidence"]),
                    min(1.0, float(existing["confidence"]) + 0.1),
                    candidate.confidence,
                )
                importance = max(float(existing["importance"]), candidate.importance)
                connection.execute(
                    "UPDATE memory_items SET confidence = ?, importance = ?,"
                    " updated_at = ?, status = 'active' WHERE id = ?",
                    (confidence, importance, stamp, existing["id"]),
                )
                return int(existing["id"]), False

            cursor = connection.execute(
                "INSERT INTO memory_items"
                " (type, content, importance, confidence, created_at, updated_at,"
                "  last_accessed, access_count, status, source_conversation,"
                "  subject, normalised)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'active', ?, ?, ?)",
                (
                    candidate.type.value,
                    content,
                    _clamp(candidate.importance),
                    _clamp(candidate.confidence),
                    stamp,
                    stamp,
                    stamp,
                    candidate.source_conversation,
                    candidate.subject[:80],
                    key,
                ),
            )
            return int(cursor.lastrowid), True

    def update(
        self,
        memory_id: int,
        *,
        content: str | None = None,
        type: MemoryType | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        status: MemoryStatus | None = None,
        subject: str | None = None,
        superseded_by: int | None = None,
    ) -> bool:
        """Change one memory. Only the fields given are touched."""
        sets: list[str] = []
        values: list = []
        if content is not None:
            cleaned = content.strip()
            if not cleaned:
                raise ValueError("A memory needs some content.")
            if len(cleaned) > MAX_CONTENT_LENGTH:
                raise ValueError(f"Content is over the {MAX_CONTENT_LENGTH} limit.")
            sets += ["content = ?", "normalised = ?"]
            values += [cleaned, normalise(cleaned)]
            # The text changed, so the stored vector no longer describes it.
            # Clearing the row is what makes the re-embed happen on next sync.
            sets.append("embedding_row = NULL")
        if type is not None:
            sets.append("type = ?")
            values.append(type.value)
        if importance is not None:
            sets.append("importance = ?")
            values.append(_clamp(importance))
        if confidence is not None:
            sets.append("confidence = ?")
            values.append(_clamp(confidence))
        if status is not None:
            sets.append("status = ?")
            values.append(status.value)
        if subject is not None:
            sets.append("subject = ?")
            values.append(subject[:80])
        if superseded_by is not None:
            sets.append("superseded_by = ?")
            values.append(superseded_by)

        if not sets:
            return False

        sets.append("updated_at = ?")
        values.append(_now())
        values.append(memory_id)

        with self._connect() as connection:
            if content is not None:
                freed = connection.execute(
                    "SELECT embedding_row FROM memory_items WHERE id = ?",
                    (memory_id,),
                ).fetchone()
                if freed is not None and freed["embedding_row"] is not None:
                    connection.execute(
                        "INSERT OR IGNORE INTO memory_free_rows (embedding_row)"
                        " VALUES (?)",
                        (int(freed["embedding_row"]),),
                    )
            cursor = connection.execute(
                f"UPDATE memory_items SET {', '.join(sets)} WHERE id = ?", values
            )
            return cursor.rowcount > 0

    def supersede(self, old_id: int, new_id: int) -> None:
        """Mark `old_id` replaced by `new_id`, keeping both.

        The heart of conflict handling: the current answer is the new memory,
        and the old one stays retrievable as history rather than being deleted
        and losing "what did I use before?".
        """
        with self._connect() as connection:
            connection.execute(
                "UPDATE memory_items SET status = ?, superseded_by = ?,"
                " updated_at = ? WHERE id = ?",
                (MemoryStatus.SUPERSEDED.value, new_id, _now(), old_id),
            )
            connection.execute(
                "INSERT OR IGNORE INTO memory_links"
                " (source_id, target_id, relation, created_at) VALUES (?, ?, ?, ?)",
                (new_id, old_id, "supersedes", _now()),
            )

    def link(self, source_id: int, target_id: int, relation: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO memory_links"
                " (source_id, target_id, relation, created_at) VALUES (?, ?, ?, ?)",
                (source_id, target_id, relation, _now()),
            )

    def links_for(self, memory_id: int) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source_id, target_id, relation FROM memory_links"
                " WHERE source_id = ? OR target_id = ?",
                (memory_id, memory_id),
            ).fetchall()
        return [dict(row) for row in rows]

    # --- reading ---

    def get(self, memory_id: int) -> Memory | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_items WHERE id = ?", (memory_id,)
            ).fetchone()
        return _memory(row) if row else None

    def by_rows(self, rows: Sequence[int]) -> dict[int, Memory]:
        """Memories keyed by embedding row, for turning search hits into text."""
        if not rows:
            return {}
        placeholders = ",".join("?" for _ in rows)
        with self._connect() as connection:
            found = connection.execute(
                f"SELECT * FROM memory_items WHERE embedding_row IN ({placeholders})",
                [int(row) for row in rows],
            ).fetchall()
        return {int(row["embedding_row"]): _memory(row) for row in found}

    def list_memories(
        self,
        *,
        statuses: Sequence[MemoryStatus] = (MemoryStatus.ACTIVE,),
        types: Sequence[MemoryType] | None = None,
        subject: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        clauses = []
        values: list = []
        if statuses:
            clauses.append(
                f"status IN ({','.join('?' for _ in statuses)})"
            )
            values += [status.value for status in statuses]
        if types:
            clauses.append(f"type IN ({','.join('?' for _ in types)})")
            values += [item.value for item in types]
        if subject:
            clauses.append("subject = ?")
            values.append(subject)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values += [max(1, int(limit)), max(0, int(offset))]
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM memory_items {where}"
                " ORDER BY importance DESC, updated_at DESC LIMIT ? OFFSET ?",
                values,
            ).fetchall()
        return [_memory(row) for row in rows]

    def search_text(self, query: str, limit: int = 20) -> list[Memory]:
        """Substring search. The fallback when no embedder is available."""
        pattern = f"%{query.strip().lower()}%"
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_items"
                " WHERE status IN ('active', 'archived')"
                "   AND (normalised LIKE ? OR subject LIKE ?)"
                " ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (pattern, pattern, max(1, int(limit))),
            ).fetchall()
        return [_memory(row) for row in rows]

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            by_status = {
                row["status"]: int(row["n"])
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS n FROM memory_items GROUP BY status"
                )
            }
            by_type = {
                row["type"]: int(row["n"])
                for row in connection.execute(
                    "SELECT type, COUNT(*) AS n FROM memory_items"
                    " WHERE status = 'active' GROUP BY type"
                )
            }
            pending = int(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM memory_jobs WHERE status = 'pending'"
                ).fetchone()["n"]
            )
        return {
            "total": sum(by_status.values()),
            "active": by_status.get("active", 0),
            "archived": by_status.get("archived", 0),
            "superseded": by_status.get("superseded", 0),
            "deleted": by_status.get("deleted", 0),
            "pending_jobs": pending,
            **{f"type_{name}": count for name, count in by_type.items()},
        }

    # --- access bookkeeping ---

    def touch(self, memory_ids: Sequence[int]) -> None:
        """Record that these memories were retrieved.

        Access count feeds the ranking, so a memory that keeps proving useful
        keeps surfacing. Deliberately one statement for the whole batch: this
        runs on the retrieval path, which is in front of the user.
        """
        if not memory_ids:
            return
        stamp = _now()
        with self._connect() as connection:
            connection.executemany(
                "UPDATE memory_items SET access_count = access_count + 1,"
                " last_accessed = ? WHERE id = ?",
                [(stamp, int(memory_id)) for memory_id in memory_ids],
            )

    # --- forgetting ---

    def forget(self, memory_id: int, *, hard: bool = False) -> bool:
        """Archive a memory, or delete the row outright.

        Default is a status change. A memory the user asked to forget should
        stop being retrieved, but "forget X" is also something people say by
        mistake, and a row that still exists can be put back.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT embedding_row FROM memory_items WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return False
            if row["embedding_row"] is not None:
                connection.execute(
                    "INSERT OR IGNORE INTO memory_free_rows (embedding_row)"
                    " VALUES (?)",
                    (int(row["embedding_row"]),),
                )
            if hard:
                connection.execute(
                    "DELETE FROM memory_items WHERE id = ?", (memory_id,)
                )
            else:
                connection.execute(
                    "UPDATE memory_items SET status = ?, embedding_row = NULL,"
                    " updated_at = ? WHERE id = ?",
                    (MemoryStatus.DELETED.value, _now(), memory_id),
                )
            return True

    def forget_subject(self, subject: str, *, hard: bool = False) -> int:
        """Forget everything filed under one subject. Returns how many."""
        subject = subject.strip()
        if not subject:
            return 0
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM memory_items WHERE subject = ?"
                " AND status != 'deleted'",
                (subject[:80],),
            ).fetchall()
        removed = 0
        for row in rows:
            if self.forget(int(row["id"]), hard=hard):
                removed += 1
        return removed

    # --- embedding rows ---

    def needing_embedding(self, limit: int = 256) -> list[Memory]:
        """Active memories with no vector yet."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_items WHERE embedding_row IS NULL"
                " AND status IN ('active', 'archived', 'superseded')"
                " ORDER BY id LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [_memory(row) for row in rows]

    def set_embedding_rows(self, pairs: Sequence[tuple[int, int]]) -> None:
        if not pairs:
            return
        with self._connect() as connection:
            connection.executemany(
                "UPDATE memory_items SET embedding_row = ? WHERE id = ?",
                [(int(row), int(memory_id)) for memory_id, row in pairs],
            )

    def take_free_rows(self, count: int) -> list[int]:
        if count <= 0:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT embedding_row FROM memory_free_rows"
                " ORDER BY embedding_row LIMIT ?",
                (count,),
            ).fetchall()
            claimed = [int(row["embedding_row"]) for row in rows]
            if claimed:
                connection.executemany(
                    "DELETE FROM memory_free_rows WHERE embedding_row = ?",
                    [(row,) for row in claimed],
                )
        return claimed

    def free_row_set(self) -> set[int]:
        with self._connect() as connection:
            return {
                int(row["embedding_row"])
                for row in connection.execute(
                    "SELECT embedding_row FROM memory_free_rows"
                )
            }

    def max_embedding_row(self) -> int:
        """Highest row in use or free; -1 when there are none.

        Both are consulted for the same reason the document index does it:
        a freed row is still allocated as far as the vector file is concerned,
        and handing it out twice would corrupt the index.
        """
        with self._connect() as connection:
            used = connection.execute(
                "SELECT MAX(embedding_row) AS m FROM memory_items"
            ).fetchone()["m"]
            free = connection.execute(
                "SELECT MAX(embedding_row) AS m FROM memory_free_rows"
            ).fetchone()["m"]
        return max(used if used is not None else -1, free if free is not None else -1)

    # --- the job queue ---

    def enqueue(
        self, kind: JobKind, payload: dict, conversation_id: int | None = None
    ) -> int:
        stamp = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO memory_jobs"
                " (kind, status, payload, conversation_id, created_at, updated_at)"
                " VALUES (?, 'pending', ?, ?, ?, ?)",
                (
                    kind.value,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    conversation_id,
                    stamp,
                    stamp,
                ),
            )
            return int(cursor.lastrowid)

    def pending_jobs(self, limit: int = 50) -> list[Job]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_jobs WHERE status = 'pending'"
                " ORDER BY id LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [_job(row) for row in rows]

    def pending_count(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM memory_jobs WHERE status = 'pending'"
                ).fetchone()["n"]
            )

    def finish_job(self, job_id: int, status: JobStatus, error: str = "") -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE memory_jobs SET status = ?, error = ?, updated_at = ?,"
                " attempts = attempts + 1 WHERE id = ?",
                (status.value, error[:500], _now(), job_id),
            )

    def retry_or_fail(self, job_id: int, error: str) -> None:
        """Put a job back unless it has already used up its attempts."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT attempts FROM memory_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            attempts = int(row["attempts"]) + 1 if row else MAX_JOB_ATTEMPTS
            status = (
                JobStatus.PENDING.value
                if attempts < MAX_JOB_ATTEMPTS
                else JobStatus.FAILED.value
            )
            connection.execute(
                "UPDATE memory_jobs SET status = ?, attempts = ?, error = ?,"
                " updated_at = ? WHERE id = ?",
                (status, attempts, error[:500], _now(), job_id),
            )

    def defer_jobs(self, job_ids: Sequence[int]) -> None:
        """Put jobs back on the queue without counting an attempt.

        Deliberately not `finish_job(..., PENDING)`: that increments
        `attempts`, and a job deferred three times because the user happened to
        be typing would exhaust its retries without ever having run. Being
        postponed is not a failure.
        """
        if not job_ids:
            return
        with self._connect() as connection:
            connection.executemany(
                "UPDATE memory_jobs SET status = 'pending', updated_at = ?"
                " WHERE id = ?",
                [(_now(), int(job_id)) for job_id in job_ids],
            )

    def reset_running_jobs(self) -> int:
        """Return jobs left RUNNING by a crash to the queue.

        Called at startup. Without it, a process killed mid-batch would strand
        its jobs in RUNNING and nothing would ever pick them up again.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE memory_jobs SET status = 'pending', updated_at = ?"
                " WHERE status = 'running'",
                (_now(),),
            )
            return cursor.rowcount

    def mark_running(self, job_ids: Sequence[int]) -> None:
        if not job_ids:
            return
        with self._connect() as connection:
            connection.executemany(
                "UPDATE memory_jobs SET status = 'running', updated_at = ?"
                " WHERE id = ?",
                [(_now(), int(job_id)) for job_id in job_ids],
            )

    def clear_finished_jobs(self, keep: int = 200) -> int:
        """Trim the job table, keeping the most recent finished rows."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memory_jobs WHERE status IN"
                " ('done', 'failed', 'discarded') AND id NOT IN ("
                "   SELECT id FROM memory_jobs WHERE status IN"
                "   ('done', 'failed', 'discarded') ORDER BY id DESC LIMIT ?"
                ")",
                (max(0, int(keep)),),
            )
            return cursor.rowcount

    # --- conversation summaries ---

    def save_summary(
        self, conversation_id: int, summary: str, covers_up_to: int, message_count: int
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversation_summaries"
                " (conversation_id, summary, covers_up_to, message_count, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(conversation_id) DO UPDATE SET"
                "   summary = excluded.summary,"
                "   covers_up_to = excluded.covers_up_to,"
                "   message_count = excluded.message_count,"
                "   updated_at = excluded.updated_at",
                (conversation_id, summary, covers_up_to, message_count, _now()),
            )

    def get_summary(self, conversation_id: int) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_summaries WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def purge(self) -> None:
        """Empty everything. Used by tests and by an explicit reset."""
        with self._connect() as connection:
            for table in (
                "memory_links", "memory_items", "memory_free_rows",
                "memory_jobs", "conversation_summaries",
            ):
                connection.execute(f"DELETE FROM {table}")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=int(row["id"]),
        type=MemoryType(row["type"]),
        content=str(row["content"]),
        importance=float(row["importance"]),
        confidence=float(row["confidence"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        last_accessed=str(row["last_accessed"]),
        access_count=int(row["access_count"]),
        status=MemoryStatus(row["status"]),
        source_conversation=(
            int(row["source_conversation"])
            if row["source_conversation"] is not None
            else None
        ),
        embedding_row=(
            int(row["embedding_row"]) if row["embedding_row"] is not None else None
        ),
        superseded_by=(
            int(row["superseded_by"]) if row["superseded_by"] is not None else None
        ),
        subject=str(row["subject"] or ""),
    )


def _job(row: sqlite3.Row) -> Job:
    try:
        payload = json.loads(row["payload"])
    except (ValueError, TypeError):
        payload = {}
    return Job(
        id=int(row["id"]),
        kind=JobKind(row["kind"]),
        status=JobStatus(row["status"]),
        payload=payload if isinstance(payload, dict) else {},
        conversation_id=(
            int(row["conversation_id"])
            if row["conversation_id"] is not None
            else None
        ),
        created_at=str(row["created_at"]),
        attempts=int(row["attempts"]),
        error=str(row["error"] or ""),
    )
