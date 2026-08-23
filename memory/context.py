"""Assembling one turn's context, under a budget.

The agent used to send the system prompt plus every message it had. This
replaces that with a layered build (section 17), newest and most specific
first:

    system instructions      always
    working memory           the current task, if there is one
    conversation summary     what was dropped from the window
    relevant memories        retrieved, ranked, capped
    recent messages          verbatim, as many as fit

**Budgets are in characters, converted from tokens.** Counting real tokens
would mean loading the model's tokeniser, and the whole point of this layer is
to run before the model is involved. `CHARS_PER_TOKEN` is the same conservative
figure the document chunker uses, so an over-estimate makes the context smaller
rather than letting it overflow.

**The recent window is never sacrificed.** Memories and summaries are allocated
their share first and truncated hard; whatever is left goes to actual messages.
A context where retrieved memories crowded out the question being asked would
be worse than no memory at all, which is why the caps are proportions of the
budget rather than "whatever fits".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory.types import ScoredMemory, WorkingMemory

# Matches rag.chunker: conservative, so estimates run high and the real
# context comes out under budget.
CHARS_PER_TOKEN = 3.5

# Shares of the total budget. They deliberately do not sum to 1: the remainder
# is headroom for the system prompt and the model's own reply.
MEMORY_SHARE = 0.15
SUMMARY_SHARE = 0.12
RECENT_SHARE = 0.55


def to_chars(tokens: int) -> int:
    return max(0, int(tokens * CHARS_PER_TOKEN))


@dataclass
class BuiltContext:
    """The messages to send, and an account of what went into them."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    memories_used: list[ScoredMemory] = field(default_factory=list)
    summary_used: bool = False
    messages_kept: int = 0
    messages_dropped: int = 0
    characters: int = 0

    def report(self) -> dict[str, Any]:
        """What the UI and the tests are told. Not sent to the model."""
        return {
            "memories": [item.as_dict() for item in self.memories_used],
            "summary_used": self.summary_used,
            "messages_kept": self.messages_kept,
            "messages_dropped": self.messages_dropped,
            "characters": self.characters,
            "estimated_tokens": int(self.characters / CHARS_PER_TOKEN),
        }


class ContextBuilder:
    """Turns history, memories and a summary into one message list."""

    def __init__(
        self,
        *,
        budget_tokens: int = 3000,
        memory_share: float = MEMORY_SHARE,
        summary_share: float = SUMMARY_SHARE,
        recent_share: float = RECENT_SHARE,
        min_recent_messages: int = 4,
    ) -> None:
        self.budget = to_chars(budget_tokens)
        self.memory_share = memory_share
        self.summary_share = summary_share
        self.recent_share = recent_share
        # However tight the budget, this many recent messages always survive:
        # a turn that dropped the user's own question would be incoherent.
        self.min_recent_messages = max(1, int(min_recent_messages))

    def build(
        self,
        *,
        system_prompt: str,
        history: list[dict[str, Any]],
        memories: list[ScoredMemory] | None = None,
        summary: str = "",
        working: WorkingMemory | None = None,
    ) -> BuiltContext:
        """Assemble the context for one turn."""
        built = BuiltContext()
        preamble: list[str] = [system_prompt]

        if working is not None:
            note = working.summary()
            if note:
                preamble.append(note)

        if summary:
            allowance = int(self.budget * self.summary_share)
            clipped = _clip(summary, allowance)
            if clipped:
                preamble.append(f"Summary of earlier conversation:\n{clipped}")
                built.summary_used = True

        chosen = self._fit_memories(memories or [])
        if chosen:
            built.memories_used = chosen
            preamble.append(_render_memories(chosen))

        built.messages.append({"role": "system", "content": "\n\n".join(preamble)})

        kept, dropped = self._fit_history(history)
        built.messages.extend(kept)
        built.messages_kept = len(kept)
        built.messages_dropped = dropped
        built.characters = sum(
            len(str(message.get("content", ""))) for message in built.messages
        )
        return built

    def _fit_memories(self, memories: list[ScoredMemory]) -> list[ScoredMemory]:
        """Take memories in rank order until their share is spent."""
        allowance = int(self.budget * self.memory_share)
        if allowance <= 0:
            return []

        chosen: list[ScoredMemory] = []
        used = 0
        for item in memories:
            cost = len(item.memory.content) + 4  # the bullet and newline
            if chosen and used + cost > allowance:
                break
            chosen.append(item)
            used += cost
        return chosen

    def _fit_history(
        self, history: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        """The most recent messages that fit, cut at a safe boundary.

        Walking backwards and then correcting the start is what keeps the
        result valid: a `tool` message whose assistant `tool_calls` were
        dropped is rejected by the chat template, so the window is only ever
        allowed to begin at a user message.
        """
        if not history:
            return [], 0

        allowance = int(self.budget * self.recent_share)
        kept: list[dict[str, Any]] = []
        used = 0

        for message in reversed(history):
            cost = len(str(message.get("content", ""))) + 16
            if len(kept) >= self.min_recent_messages and used + cost > allowance:
                break
            kept.append(message)
            used += cost

        kept.reverse()
        start = 0
        for index, message in enumerate(kept):
            if message.get("role") == "user":
                start = index
                break
        else:
            # No user message survived, so the window is only tool traffic.
            # Sending that alone would be incoherent; send nothing instead and
            # let the caller's own message stand.
            return [], len(history)

        kept = kept[start:]
        return kept, len(history) - len(kept)


def _render_memories(memories: list[ScoredMemory]) -> str:
    """The memory block, as the model sees it.

    Typed and hedged deliberately. "What you remember about the user" invites
    the model to treat every line as certain; naming the type and marking the
    weak ones lets it say "you mentioned you might..." instead of asserting it.
    """
    lines = ["What you remember about this user (may be incomplete):"]
    for item in memories:
        memory = item.memory
        marker = f"[{memory.type.value}]"
        if memory.confidence < 0.5:
            marker += " (unconfirmed)"
        lines.append(f"- {marker} {memory.content}")
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    # Cut at a sentence end when there is one nearby, so the summary does not
    # trail off mid-clause.
    window = text[:limit]
    cut = window.rfind(". ")
    if cut > limit * 0.6:
        return window[: cut + 1]
    return window.rstrip() + "..."
