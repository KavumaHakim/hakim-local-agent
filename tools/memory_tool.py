"""Tools for the agent's long-term memory.

Off by default (config.memory_tool_enabled, env AGENT_ENABLE_MEMORY=1).

Five operations, matching the design: remember, recall, search_memory,
update_memory and forget_memory. `recall` and `search_memory` look alike on
purpose - `recall` is the everyday one and returns memories ready to quote,
while `search_memory` exposes scores and lets the model widen the net when it
is checking whether something is known at all.

**None of these load a chat model.** Retrieval is embeddings and arithmetic, so
the agent can call them mid-turn with Mistral resident and nothing gets
unloaded. `remember` writes immediately for the same reason: telling the user
"noted" while the memory waits behind a model switch would be a lie.

Memories are still NOT injected into the system prompt by this module. The
context builder does that, with a budget, once per turn - see memory/context.py.
The tools exist for when the agent has a reason to look something up itself.
"""

from __future__ import annotations

from typing import Any

from memory.manager import MemoryManager, MemoryOperationError
from tools.base import Tool, ToolError

# Retrieval hands back at most this much text in one call. Memories are short,
# so this is generous - but a store with a thousand rows must not be able to
# empty itself into a prompt.
MAX_RESULTS = 25


class MemoryToolError(ToolError):
    """A memory operation was rejected."""


class MemoryTools:
    """The model-facing half of the memory system."""

    def __init__(self, manager: MemoryManager, *, conversation_id: int | None = None) -> None:
        self._memory = manager
        self._conversation_id = conversation_id

    # --- operations ---

    def remember(
        self,
        content: str,
        type: str = "fact",
        importance: float | None = None,
        subject: str = "",
    ) -> dict[str, Any]:
        try:
            return self._memory.remember(
                content,
                type=type,
                importance=0.8 if importance is None else float(importance),
                subject=subject,
                conversation_id=self._conversation_id,
            )
        except MemoryOperationError as exc:
            raise MemoryToolError(str(exc)) from None

    def recall(self, query: str = "", limit: int = 5) -> dict[str, Any]:
        try:
            return self._memory.recall(query, limit=_bounded(limit))
        except MemoryOperationError as exc:
            raise MemoryToolError(str(exc)) from None

    def search_memory(self, query: str, limit: int = 10) -> dict[str, Any]:
        if not query or not query.strip():
            raise MemoryToolError("Give something to search for.")
        found = self._memory.search(query, limit=_bounded(limit))
        if found:
            self._memory.store.touch([item.memory.id for item in found])
        return {
            "success": True,
            "query": query.strip(),
            "count": len(found),
            "memories": [item.as_dict() for item in found],
        }

    def update_memory(
        self,
        memory_id: int,
        content: str = "",
        type: str = "",
        importance: float | None = None,
        status: str = "",
    ) -> dict[str, Any]:
        changes: dict[str, Any] = {}
        if content.strip():
            changes["content"] = content.strip()
        if type.strip():
            changes["type"] = type.strip()
        if status.strip():
            changes["status"] = status.strip()
        if importance is not None:
            changes["importance"] = float(importance)
        if not changes:
            raise MemoryToolError(
                "Give at least one of content, type, importance or status."
            )
        try:
            return self._memory.update(int(memory_id), **changes)
        except MemoryOperationError as exc:
            raise MemoryToolError(str(exc)) from None

    def forget_memory(self, target: str) -> dict[str, Any]:
        try:
            return self._memory.forget(target)
        except MemoryOperationError as exc:
            raise MemoryToolError(str(exc)) from None

    # --- definitions ---

    def tools(self) -> list[Tool]:
        return [
            Tool(
                name="remember",
                category="memory",
                description=(
                    "Store one durable fact about the user so it survives into "
                    "later conversations. Use it when the user states a lasting "
                    "preference, a decision, or something about their setup - "
                    "and always when they say 'remember that ...'. "
                    "Write it as a short third-person sentence: 'User prefers "
                    "X'. One fact per call. "
                    "Do NOT store greetings, small talk, passing details, "
                    "one-off calculations, or anything true only today."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The fact, as one short sentence.",
                        },
                        "type": {
                            "type": "string",
                            "description": (
                                "preference (a lasting choice), fact (lasting "
                                "state), event (something that happened), "
                                "intention (something they might do - use this "
                                "when they are unsure), temporary (true today "
                                "only). Defaults to fact."
                            ),
                        },
                        "importance": {
                            "type": "number",
                            "description": "0-1, how much this should outrank others.",
                        },
                        "subject": {
                            "type": "string",
                            "description": (
                                "Optional grouping, e.g. a file or project name, "
                                "so it can be forgotten as a group later."
                            ),
                        },
                    },
                    "required": ["content"],
                },
                run=self.remember,
            ),
            Tool(
                name="recall",
                category="memory",
                description=(
                    "Look up what is remembered about the user. Give a query to "
                    "search by meaning, or omit it to list the most important. "
                    "Use it when the user refers to their preferences, their "
                    "usual setup, or something discussed in an earlier "
                    "conversation - and to answer 'what do you remember about "
                    "me'. Returns nothing when nothing relevant is stored, "
                    "which is a real answer: say so rather than guessing."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "What to look for. Omit to list the most "
                                "important memories."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum memories to return (default 5).",
                        },
                    },
                    "required": [],
                },
                run=self.recall,
            ),
            Tool(
                name="search_memory",
                category="memory",
                description=(
                    "Search memories by meaning and see their relevance scores. "
                    "Use this instead of recall when checking whether something "
                    "is known at all, or when recall returned too little."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default 10).",
                        },
                    },
                    "required": ["query"],
                },
                run=self.search_memory,
            ),
            Tool(
                name="update_memory",
                category="memory",
                description=(
                    "Correct a stored memory, by its id from recall or "
                    "search_memory. Use it when something remembered is wrong "
                    "or out of date but should not be forgotten entirely."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "integer",
                            "description": "The id of the memory to change.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Replacement text.",
                        },
                        "type": {
                            "type": "string",
                            "description": "New type, if it was misclassified.",
                        },
                        "importance": {
                            "type": "number",
                            "description": "New importance, 0-1.",
                        },
                        "status": {
                            "type": "string",
                            "description": (
                                "active, or archived to stop it being retrieved "
                                "without deleting it."
                            ),
                        },
                    },
                    "required": ["memory_id"],
                },
                run=self.update_memory,
            ),
            Tool(
                name="forget_memory",
                category="memory",
                description=(
                    "Forget what is remembered about something. Give a memory "
                    "id, or a description like 'my editor preference' to forget "
                    "everything about it. Use it whenever the user asks you to "
                    "forget something - actually call this, do not just say you "
                    "will."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": (
                                "A memory id, or a description of what to forget."
                            ),
                        }
                    },
                    "required": ["target"],
                },
                run=self.forget_memory,
            ),
        ]


def _bounded(limit: Any) -> int:
    try:
        return max(1, min(int(limit), MAX_RESULTS))
    except (TypeError, ValueError):
        return 5


def build_memory_tools(
    manager: MemoryManager, *, conversation_id: int | None = None
) -> list[Tool]:
    """The memory tools, bound to a manager."""
    return MemoryTools(manager, conversation_id=conversation_id).tools()
