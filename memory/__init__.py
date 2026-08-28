"""Persistent memory for the agent.

    working memory   in the active context, never persisted
    episodic         events: "user added Biology.pdf"
    semantic         durable facts and preferences
    summaries        what was dropped from a long conversation

`MemoryManager` is the only object the rest of the application uses. Retrieval,
ranking, decay and deduplication are ordinary code and need no model; the
auxiliary model is used only for language work, and only in batches.
"""

from memory.manager import MemoryManager, MemoryOperationError
from memory.types import Memory, MemoryStatus, MemoryType

__all__ = [
    "Memory",
    "MemoryManager",
    "MemoryOperationError",
    "MemoryStatus",
    "MemoryType",
]
