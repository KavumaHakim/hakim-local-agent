"""Long-term facts the agent can keep between conversations.

Deliberately separate from chat history. History is a transcript; this is a
small set of durable facts, keyed and overwritable. Dumping whole conversations
in here would defeat the point - the value is that it stays short enough to be
worth reading.

It lives in the same SQLite file as the history, in its own table, and follows
the same connection-per-operation rule: Streamlit reruns on background threads
and a shared sqlite3 connection is not safe across them.

Note on scope: recall is a substring search, not embeddings. On a machine that
generates at well under a token per second, adding an embedding model to search
a few dozen rows would cost far more than it saves.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

MAX_KEY_LENGTH = 80
MAX_VALUE_LENGTH = 2000
DEFAULT_LIMIT = 20


@dataclass(frozen=True)
class Memory:
    key: str
    value: str
    created_at: str
    updated_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    """Keyed facts, stored in SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    # --- operations ---

    def remember(self, key: str, value: str) -> dict:
        key = _clean_key(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("value must be a non-empty string.")
        value = value.strip()
        if len(value) > MAX_VALUE_LENGTH:
            raise ValueError(
                f"value is too long ({len(value)} characters, limit "
                f"{MAX_VALUE_LENGTH}). Store a fact, not a transcript."
            )

        stamp = _now()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at FROM memories WHERE key = ?", (key,)
            ).fetchone()
            created = existing["created_at"] if existing else stamp
            connection.execute(
                "INSERT INTO memories (key, value, created_at, updated_at)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
                " updated_at = excluded.updated_at",
                (key, value, created, stamp),
            )
        return {"success": True, "key": key, "replaced": existing is not None}

    def recall(self, query: str = "", limit: int = DEFAULT_LIMIT) -> dict:
        limit = max(1, min(int(limit), 100))
        query = (query or "").strip()

        with self._connect() as connection:
            if query:
                pattern = f"%{query}%"
                rows = connection.execute(
                    "SELECT * FROM memories WHERE key LIKE ? OR value LIKE ?"
                    " ORDER BY updated_at DESC LIMIT ?",
                    (pattern, pattern, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

        found = [
            {"key": row["key"], "value": row["value"], "updated_at": row["updated_at"]}
            for row in rows
        ]
        return {"success": True, "query": query, "count": len(found), "memories": found}

    def forget(self, key: str) -> dict:
        key = _clean_key(key)
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE key = ?", (key,))
            removed = cursor.rowcount > 0
        if not removed:
            return {"success": False, "error": f"Nothing stored under {key!r}."}
        return {"success": True, "key": key, "forgotten": True}

    def get(self, key: str) -> Memory | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE key = ?", (_clean_key(key),)
            ).fetchone()
        if row is None:
            return None
        return Memory(
            key=row["key"],
            value=row["value"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def count(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute("SELECT COUNT(*) AS n FROM memories")
                .fetchone()["n"]
            )

    def purge(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM memories")


def _clean_key(key: str) -> str:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("key must be a non-empty string.")
    cleaned = " ".join(key.strip().lower().split())
    if len(cleaned) > MAX_KEY_LENGTH:
        raise ValueError(f"key is too long (limit {MAX_KEY_LENGTH}).")
    return cleaned
