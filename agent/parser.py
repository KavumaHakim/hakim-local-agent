"""Interpretation of assistant messages returned by the Qwen server.

Design note (verified against llama-server build 10373):
llama.cpp runs with --jinja enabled by default. It applies the model's own
chat template and parses the model's native tool-call syntax server-side,
so a tool call arrives as a standard OpenAI `tool_calls` array:

    {"role": "assistant",
     "content": null,
     "reasoning_content": "...",
     "tool_calls": [{"id": "...", "type": "function",
                     "function": {"name": "...", "arguments": "{\\"k\\": 1}"}}]}

We therefore do NOT invent a custom protocol. We read the shape the server
already produces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class ParseError(Exception):
    """An assistant message could not be interpreted."""


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantTurn:
    """A normalized view of one assistant message."""

    content: str
    reasoning: str
    tool_calls: list[ToolCall]
    raw: dict[str, Any]

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


def parse_assistant_message(message: dict[str, Any]) -> AssistantTurn:
    """Normalize a raw assistant message into an AssistantTurn."""
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise ParseError(f"Expected string content, got {type(content).__name__}")

    reasoning = message.get("reasoning_content") or ""
    if not isinstance(reasoning, str):
        reasoning = ""

    return AssistantTurn(
        content=content,
        reasoning=reasoning,
        tool_calls=parse_tool_calls(message),
        raw=message,
    )


def parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Read the OpenAI-style `tool_calls` array, if present."""
    raw_calls = message.get("tool_calls")
    if not raw_calls:
        return []
    if not isinstance(raw_calls, list):
        raise ParseError(f"tool_calls was not a list: {raw_calls!r}")

    calls: list[ToolCall] = []
    for index, entry in enumerate(raw_calls):
        if not isinstance(entry, dict):
            raise ParseError(f"tool_calls[{index}] was not an object")

        function = entry.get("function")
        if not isinstance(function, dict):
            raise ParseError(f"tool_calls[{index}] had no function object")

        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ParseError(f"tool_calls[{index}] had no function name")

        # llama.cpp sends arguments as a JSON-encoded string.
        raw_args = function.get("arguments", "{}")
        if isinstance(raw_args, dict):
            arguments = raw_args
        elif isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args or "{}")
            except json.JSONDecodeError as exc:
                raise ParseError(
                    f"tool_calls[{index}] arguments were not valid JSON: {raw_args!r}"
                ) from exc
        else:
            raise ParseError(f"tool_calls[{index}] had unusable arguments")

        if not isinstance(arguments, dict):
            raise ParseError(f"tool_calls[{index}] arguments were not an object")

        calls.append(
            ToolCall(
                id=str(entry.get("id") or f"call_{index}"),
                name=name,
                arguments=arguments,
            )
        )
    return calls


# TODO(tool-calling): fallback for a server started with --no-jinja.
# In that mode llama.cpp does not parse tool calls and Qwen3 emits raw
# <tool_call>{...}</tool_call> blocks inside `content`. Only implement this
# after confirming we actually need to run without --jinja - do not guess at
# the tag format from documentation alone.

# TODO(tool-calling): validate arguments against each tool's JSON schema
# before execution, and return a structured error to the model on mismatch
# instead of raising.
