"""Tools for the agent's long-term memory.

Off by default (config.memory_tool_enabled, env AGENT_ENABLE_MEMORY=1).

Three operations: remember, recall, forget. `forget` exists despite the
project's general rule against destructive operations, because this is the
agent's own store rather than the user's data, and a memory that turns out to
be wrong is worse than no memory at all. The blast radius is one row.

Memories are NOT injected into the system prompt. That would look convenient
and cost real time: prompt tokens run at a few per second on this machine, so a
growing preamble would tax every single turn, including the ones that need no
memory at all. The agent calls `recall` when it has a reason to.
"""

from __future__ import annotations

from typing import Any

from memory_store import MemoryStore
from tools.base import Tool, ToolError


class MemoryToolError(ToolError):
    """A memory operation was rejected."""


class MemoryTools:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def remember(self, key: str, value: str) -> dict[str, Any]:
        try:
            return self._store.remember(key, value)
        except ValueError as exc:
            raise MemoryToolError(str(exc)) from None

    def recall(self, query: str = "", limit: int = 20) -> dict[str, Any]:
        try:
            return self._store.recall(query, limit)
        except (ValueError, TypeError) as exc:
            raise MemoryToolError(str(exc)) from None

    def forget(self, key: str) -> dict[str, Any]:
        try:
            return self._store.forget(key)
        except ValueError as exc:
            raise MemoryToolError(str(exc)) from None

    def tools(self) -> list[Tool]:
        return [
            Tool(
                name="remember",
                category="memory",
                description=(
                    "Store a durable fact under a short key, so it survives "
                    "into later conversations. Storing the same key again "
                    "replaces it. Keep it to a fact or a decision, not a "
                    "transcript."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Short identifier, e.g. 'preferred editor'.",
                        },
                        "value": {
                            "type": "string",
                            "description": "The fact to remember.",
                        },
                    },
                    "required": ["key", "value"],
                },
                run=self.remember,
            ),
            Tool(
                name="recall",
                category="memory",
                description=(
                    "Look up stored facts. Give a query to search keys and "
                    "values, or omit it to list the most recent."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Text to search for. Omit to list recent memories.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default 20).",
                        },
                    },
                    "required": [],
                },
                run=self.recall,
            ),
            Tool(
                name="forget",
                category="memory",
                description=(
                    "Delete one stored fact by its key. Use when something "
                    "remembered turns out to be wrong."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The key to remove.",
                        }
                    },
                    "required": ["key"],
                },
                run=self.forget,
            ),
        ]


def build_memory_tools(db_path) -> list[Tool]:
    return MemoryTools(MemoryStore(db_path)).tools()
