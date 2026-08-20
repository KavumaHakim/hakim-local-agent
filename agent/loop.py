"""The agent loop: orchestration only, no HTTP and no tool implementations.

One user turn runs as:

    user message
      -> Qwen
      -> if the reply has no tool calls, that is the final answer
      -> otherwise run each tool call, append its result, and go round again
      -> bounded by config.max_iterations
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from agent.parser import AssistantTurn, ParseError, ToolCall, parse_assistant_message
from agent.prompts import SYSTEM_PROMPT
from config import Config
from models.qwen import ChatClient, TokenCallback
from tools.base import ToolRegistry, ToolResult


class AgentError(Exception):
    """The agent could not complete the turn."""


class IterationLimitError(AgentError):
    """The model kept calling tools past the configured limit."""


@dataclass(frozen=True)
class ToolEvent:
    """Reported to the CLI so it can show progress. Not sent to the model."""

    call: ToolCall
    result: ToolResult


# The CLI passes one of these in to display tool activity as it happens.
Observer = Callable[[ToolEvent], None]


class Agent:
    """Holds conversation state and drives the reasoning/action cycle."""

    def __init__(
        self,
        client: ChatClient,
        config: Config,
        tools: ToolRegistry,
    ) -> None:
        self._client = client
        self._config = config
        self._tools = tools
        self._history: list[dict[str, Any]] = []

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def clear(self) -> None:
        """Reset the conversation. The system prompt is not stored in history."""
        self._history.clear()

    def load_history(self, messages: list[dict[str, Any]]) -> None:
        """Replace the conversation with `messages`.

        Used when a settings change rebuilds the agent and the existing
        conversation should carry over.
        """
        self._history = list(messages)

    def send(
        self,
        user_input: str,
        observer: Observer | None = None,
        on_token: TokenCallback | None = None,
    ) -> AssistantTurn:
        """Run one user turn to completion and return the final assistant turn.

        If `on_token` is given and the client supports streaming, content
        fragments are delivered as they arrive. Tool-call rounds stream too,
        but produce no content, so the caller sees tool events instead.
        """
        self._history.append({"role": "user", "content": user_input})

        definitions = self._tools.get_tool_definitions()

        for _ in range(max(1, self._config.max_iterations)):
            message = self._chat(definitions, on_token)

            try:
                turn = parse_assistant_message(message)
            except ParseError as exc:
                raise AgentError(f"Could not read the model's reply: {exc}") from exc

            self._history.append(self._assistant_entry(message, turn))

            if not turn.wants_tools:
                self._trim_history()
                return turn

            for call in turn.tool_calls:
                result = self._tools.execute(call.name, call.arguments)
                self._history.append(
                    {
                        "role": "tool",
                        # Ties the result back to the call the model made.
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result.content,
                    }
                )
                if observer is not None:
                    observer(ToolEvent(call=call, result=result))

            self._trim_history()

        raise IterationLimitError(
            f"Stopped after {self._config.max_iterations} tool rounds without a "
            f"final answer. Raise AGENT_MAX_ITERATIONS, or rephrase the request "
            f"so it needs fewer steps."
        )

    # --- internals ---

    def _chat(
        self,
        definitions: list[dict[str, Any]],
        on_token: TokenCallback | None,
    ) -> dict[str, Any]:
        """One model round, streaming when the caller asked for it."""
        if on_token is not None:
            stream = getattr(self._client, "chat_stream", None)
            if callable(stream):
                return stream(
                    self._messages(), tools=definitions, on_token=on_token
                )
        return self._client.chat(self._messages(), tools=definitions)

    def _messages(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": SYSTEM_PROMPT}, *self._history]

    @staticmethod
    def _assistant_entry(message: dict[str, Any], turn: AssistantTurn) -> dict[str, Any]:
        """Store the assistant message without its reasoning trace.

        Thinking is per-turn and must not be replayed to the model, but the
        tool_calls array has to be preserved exactly or the tool results that
        follow have nothing to attach to.
        """
        entry: dict[str, Any] = {"role": "assistant", "content": turn.content}
        if message.get("tool_calls"):
            entry["tool_calls"] = message["tool_calls"]
        return entry

    def _trim_history(self) -> None:
        """Drop the oldest turns once history grows past the configured limit.

        Trimming only cuts immediately before a user message, so a tool result
        is never left behind without the assistant tool call it belongs to -
        which the chat template would reject.
        """
        limit = self._config.max_history_messages
        if limit <= 0 or len(self._history) <= limit:
            return

        excess = len(self._history) - limit
        for index in range(excess, len(self._history)):
            if self._history[index].get("role") == "user":
                del self._history[:index]
                return
        # No safe cut point found; keep the history as it is.


def summarize_arguments(arguments: dict[str, Any], limit: int = 80) -> str:
    """One-line rendering of tool arguments for the CLI."""
    try:
        text = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(arguments)
    return text if len(text) <= limit else text[: limit - 1] + "…"
