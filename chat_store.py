"""Chat history, stored in SQLite.

Deliberately small: two tables, no ORM, no migrations framework. SQLite is in
the standard library, the file sits next to the project, and the whole thing
is a few hundred rows of local conversation.

A connection is opened per operation rather than held open. Streamlit reruns
the script on background threads, and a shared sqlite3 connection is not safe
across them; opening per call sidesteps that entirely and costs nothing at
this size.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    model_key   TEXT,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    -- Tool calls are display metadata, not conversation state, so they live
    -- as JSON rather than earning their own table.
    tools           TEXT,
    elapsed         REAL,
    model_key       TEXT,
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, id);
"""

TITLE_LENGTH = 60


@dataclass(frozen=True)
class Conversation:
    id: int
    title: str
    model_key: str | None
    created_at: str
    updated_at: str
    message_count: int = 0


@dataclass(frozen=True)
class StoredMessage:
    id: int
    role: str
    content: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    elapsed: float | None = None
    model_key: str | None = None
    created_at: str = ""

    def as_ui_dict(self) -> dict[str, Any]:
        """Shape the Streamlit page already renders."""
        entry: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tools:
            entry["tools"] = self.tools
        if self.elapsed:
            entry["elapsed"] = self.elapsed
        return entry


def _now() -> str:
    """UTC timestamp, microsecond resolution.

    Full precision matters: conversations are ordered by updated_at, and at
    second resolution several updates land in the same tick and the ordering
    silently falls back to insertion order.
    """
    return datetime.now(timezone.utc).isoformat()


def make_title(text: str) -> str:
    """Derive a conversation title from its first message."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "New conversation"
    if len(cleaned) <= TITLE_LENGTH:
        return cleaned
    return cleaned[: TITLE_LENGTH - 1].rstrip() + "…"


class ChatStore:
    """Conversation history for the agent."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        # Not on by default in sqlite, and the messages cascade depends on it.
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    # --- conversations ---

    def create_conversation(
        self, title: str = "New conversation", model_key: str | None = None
    ) -> int:
        stamp = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO conversations (title, model_key, created_at, updated_at)"
                " VALUES (?, ?, ?, ?)",
                (title, model_key, stamp, stamp),
            )
            return int(cursor.lastrowid)

    def rename_conversation(self, conversation_id: int, title: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), conversation_id),
            )

    def delete_conversation(self, conversation_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
            connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )

    def list_conversations(self, limit: int = 30) -> list[Conversation]:
        """Most recently updated first."""
        with self._connect() as connection:
            # Ties on updated_at are not rare: the Windows clock can advance in
            # ~15 ms steps, so two operations often share a timestamp. Falling
            # back to c.id would order by creation, putting a newer but idle
            # conversation above the one just touched - so break the tie on the
            # newest message instead, which is what "recently used" means.
            # A conversation with no messages has MAX(m.id) NULL, and SQLite
            # sorts NULL last under DESC, which is the intent.
            rows = connection.execute(
                "SELECT c.*, COUNT(m.id) AS message_count, MAX(m.id) AS last_message"
                " FROM conversations c"
                " LEFT JOIN messages m ON m.conversation_id = c.id"
                " GROUP BY c.id"
                " ORDER BY c.updated_at DESC, last_message DESC, c.id DESC"
                " LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            Conversation(
                id=row["id"],
                title=row["title"],
                model_key=row["model_key"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
            )
            for row in rows
        ]

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT c.*, COUNT(m.id) AS message_count"
                " FROM conversations c"
                " LEFT JOIN messages m ON m.conversation_id = c.id"
                " WHERE c.id = ? GROUP BY c.id",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return Conversation(
            id=row["id"],
            title=row["title"],
            model_key=row["model_key"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=row["message_count"],
        )

    # --- messages ---

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        elapsed: float | None = None,
        model_key: str | None = None,
    ) -> int:
        stamp = _now()
        payload = json.dumps(tools) if tools else None
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO messages"
                " (conversation_id, role, content, tools, elapsed, model_key,"
                "  created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, role, content, payload, elapsed, model_key, stamp),
            )
            # Keeps list_conversations ordered by real activity.
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (stamp, conversation_id),
            )
            return int(cursor.lastrowid)

    def get_messages(self, conversation_id: int) -> list[StoredMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()

        messages: list[StoredMessage] = []
        for row in rows:
            try:
                tools = json.loads(row["tools"]) if row["tools"] else []
            except ValueError:
                tools = []
            messages.append(
                StoredMessage(
                    id=row["id"],
                    role=row["role"],
                    content=row["content"],
                    tools=tools,
                    elapsed=row["elapsed"],
                    model_key=row["model_key"],
                    created_at=row["created_at"],
                )
            )
        return messages

    def truncate_from(self, conversation_id: int, message_id: int) -> int:
        """Delete a message and everything after it. Returns how many went.

        The rewind behind editing a question: the old question, the answer it
        got, and anything that followed all have to go, because they were a
        reply to something that is no longer what was asked. Keeping them
        would leave a transcript that reads as a conversation nobody had.

        Ids are monotonic within a conversation, so "after" is "greater id" -
        the one place in this file where that is true without qualification,
        because unlike a queued turn's history there is nothing in flight to
        confuse the order.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM messages WHERE id = ? AND conversation_id = ?",
                (message_id, conversation_id),
            ).fetchone()
            if row is None:
                return 0
            cursor = connection.execute(
                "DELETE FROM messages WHERE conversation_id = ? AND id >= ?",
                (conversation_id, message_id),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_now(), conversation_id),
            )
            return int(cursor.rowcount)

    def message_count(self, conversation_id: int) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["n"])

    def purge(self) -> None:
        """Delete every conversation. Used by the UI's 'clear history'."""
        with self._connect() as connection:
            connection.execute("DELETE FROM messages")
            connection.execute("DELETE FROM conversations")
