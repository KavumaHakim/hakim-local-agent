"""A scripted stand-in for QwenClient, so the loop can be tested offline."""

from __future__ import annotations

import json
from typing import Any, Iterable


def text_message(content: str) -> dict[str, Any]:
    """An ordinary assistant reply with no tool calls."""
    return {"role": "assistant", "content": content}


def tool_call_message(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """An assistant reply requesting one or more tools.

    Mirrors what llama.cpp returns with --jinja: arguments arrive as a
    JSON-encoded string.
    """
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for index, (name, arguments) in enumerate(calls)
        ],
    }


class FakeQwenClient:
    """Returns pre-scripted messages and records what it was sent."""

    def __init__(self, responses: list[dict[str, Any]], *, repeat_last: bool = False):
        self._responses = list(responses)
        self._repeat_last = repeat_last
        self.calls: list[list[dict[str, Any]]] = []
        self.tools_seen: list[list[dict[str, Any]] | None] = []

    def chat(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append([dict(m) for m in messages])
        self.tools_seen.append(tools)

        if not self._responses:
            raise AssertionError("FakeQwenClient ran out of scripted responses")
        if len(self._responses) == 1 and self._repeat_last:
            return self._responses[0]
        return self._responses.pop(0)
