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


# The smallest a tool result may be cut to. Below this the result says
# nothing useful and the note explaining the cut does not fit either, so the
# model is better told plainly that the thing did not fit.
MIN_RESULT_CHARS = 400


class TurnStopped(AgentError):
    """Someone asked for this turn to stop, and it did.

    Not a failure - it is the loop doing as it was told - but it travels the
    same path as one, because in both cases the turn ends without an answer
    and the caller has to hear about it before it can report anything.
    """


@dataclass(frozen=True)
class ToolEvent:
    """Reported to the CLI so it can show progress. Not sent to the model."""

    call: ToolCall
    result: ToolResult


# The CLI passes one of these in to display tool activity as it happens.
Observer = Callable[[ToolEvent], None]

# Asked at every checkpoint: "has someone asked for this turn to stop?"
#
# A predicate rather than a flag the loop owns, because the answer belongs to
# whoever is running the turn - the queue, in the API's case - and the loop
# should not have to know how that is stored.
StopCheck = Callable[[], bool]


class Agent:
    """Holds conversation state and drives the reasoning/action cycle."""

    def __init__(
        self,
        client: ChatClient,
        config: Config,
        tools: ToolRegistry,
        memory: Any = None,
        conversation_id: int | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._tools = tools
        self._history: list[dict[str, Any]] = []
        # Optional. With no memory manager the loop behaves exactly as it did
        # before: system prompt plus history, trimmed by message count.
        self._memory = memory
        self._conversation_id = conversation_id
        # What the last context build put in front of the model. Reported to
        # the UI so retrieved memories are visible rather than mysterious.
        self.context_report: dict[str, Any] = {}

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
        on_reasoning: TokenCallback | None = None,
        should_stop: StopCheck | None = None,
    ) -> AssistantTurn:
        """Run one user turn to completion and return the final assistant turn.

        If `on_token` is given and the client supports streaming, content
        fragments are delivered as they arrive. Tool-call rounds stream too,
        but produce no content, so the caller sees tool events instead.

        `on_reasoning` receives the model's thinking when it produces any. It
        is never added to history: `_assistant_entry` stores the answer and the
        tool calls only, and that has to stay true however the trace is
        displayed.

        `should_stop` is asked at every checkpoint, and `TurnStopped` is raised
        the first time it says yes. Checkpoints are the only honest way to do
        this: a Python thread cannot be interrupted from outside, so stopping
        means the loop noticing between one piece of work and the next. In
        practice that is per streamed token, which on this hardware is at most
        a second or so - except while the model is reading the prompt, where
        nothing comes back and the wait is however long that takes.
        """
        self._history.append({"role": "user", "content": user_input})

        definitions = self._tools.get_tool_definitions()

        for _ in range(max(1, self._config.max_iterations)):
            self._check_stop(should_stop)
            # Trim before sending, not after. Trimming afterwards leaves the
            # history within budget but says nothing about the request just
            # made - the newly appended message is exactly what pushes it over,
            # and that request is the one that fails.
            self._trim_history()
            message = self._chat(definitions, on_token, on_reasoning, should_stop)
            # The client returns what it had when it stopped, so the check has
            # to be here rather than inside it: a partial message is not an
            # answer, and parsing it as one would put a truncated tool call
            # into the history.
            self._check_stop(should_stop)

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
                        # Capped against the model's context. Uncapped, a
                        # page of OCR or a large file read overflows the
                        # window and loses the whole turn rather than part of
                        # one result.
                        "content": result.content_within(
                            self._result_budget()
                        ),
                    }
                )
                if observer is not None:
                    observer(ToolEvent(call=call, result=result))
                # Between tools, not only between rounds: a round of four
                # tool calls is minutes of work, and stopping after the first
                # is the point.
                self._check_stop(should_stop)

            self._trim_history()

        raise IterationLimitError(
            f"Stopped after {self._config.max_iterations} tool rounds without a "
            f"final answer. Raise AGENT_MAX_ITERATIONS, or rephrase the request "
            f"so it needs fewer steps."
        )

    # --- internals ---

    def _result_budget(self) -> int:
        """How much of the context this tool result may take, right now.

        The per-result cap, or whatever is left of the history budget if that
        is smaller. Both are needed and neither is enough alone: the per-result
        cap bounds one page of OCR, and this bounds the turn where the model
        lists four directories looking for a file. Each of those four is
        within its own cap and their sum is three times the window.

        `_trim_history` cannot help here. It only cuts immediately before a
        user message, and inside one turn there is exactly one of those, at
        the very front - so it finds nothing safe to drop and returns, leaving
        the request oversized rather than trimmed.

        Never below a floor: a result cut to nothing tells the model less than
        an honest "this did not fit", and the truncation note itself needs
        room to be read.
        """
        per_result = self._config.max_tool_result_chars
        budget = self._config.max_history_chars
        if budget <= 0:
            return per_result

        remaining = budget - self._history_chars()
        return max(MIN_RESULT_CHARS, min(per_result, remaining))

    @staticmethod
    def _check_stop(should_stop: StopCheck | None) -> None:
        """Raise if a stop has been asked for. The only place that decides."""
        if should_stop is not None and should_stop():
            raise TurnStopped("Stopped on request.")

    def _chat(
        self,
        definitions: list[dict[str, Any]],
        on_token: TokenCallback | None,
        on_reasoning: TokenCallback | None = None,
        should_stop: StopCheck | None = None,
    ) -> dict[str, Any]:
        """One model round, streaming when the caller asked for it."""
        if on_token is not None or on_reasoning is not None:
            stream = getattr(self._client, "chat_stream", None)
            if callable(stream):
                return stream(
                    self._messages(),
                    tools=definitions,
                    on_token=on_token,
                    on_reasoning=on_reasoning,
                    should_stop=should_stop,
                )
        # Not streaming: there is nothing to interrupt, because the whole
        # answer arrives in one blocking call. The checkpoint either side of
        # this is all a non-streaming client can offer.
        return self._client.chat(self._messages(), tools=definitions)

    def _messages(self) -> list[dict[str, Any]]:
        """The context for one model round.

        With no memory manager this is the original behaviour: the system
        prompt and the whole (already trimmed) history. With one, the context
        builder assembles the summary, the retrieved memories and a recent
        window that fits a token budget - see memory/context.py.

        The query used for retrieval is the most recent user message rather
        than the whole history: retrieving against a transcript matches
        whatever was talked about most, not what is being asked now.
        """
        if self._memory is None:
            return [{"role": "system", "content": SYSTEM_PROMPT}, *self._history]

        built = self._memory.build_context(
            system_prompt=SYSTEM_PROMPT,
            history=self._history,
            query=self._latest_user_message(),
            conversation_id=self._conversation_id,
        )
        self.context_report = built.report()
        return built.messages

    def _latest_user_message(self) -> str:
        for message in reversed(self._history):
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

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
        budget = self._config.max_history_chars

        while self._history:
            too_many = limit > 0 and len(self._history) > limit
            too_large = budget > 0 and self._history_chars() > budget
            if not (too_many or too_large):
                return

            # Search from 1, never 0: a cut of zero would make no progress and
            # this would spin. It also means the most recent user message can
            # never be dropped, which is the one thing the turn cannot lose.
            cut = next(
                (
                    index
                    for index in range(1, len(self._history))
                    if self._history[index].get("role") == "user"
                ),
                None,
            )
            if cut is None:
                # Nothing safe left to drop. Cutting anywhere else would orphan
                # a tool result from the assistant message that called it, and
                # the chat template rejects that outright.
                return
            del self._history[:cut]

    def _history_chars(self) -> int:
        """Roughly what the conversation costs, for the size limit.

        Characters rather than tokens, because counting tokens means loading a
        tokeniser and this runs on every round of every turn.

        `tool_calls` counts as well as `content`. An assistant message that
        asks for a tool has an empty content and a JSON array of calls that is
        several hundred characters, all of it sent to the model - so counting
        content alone under-reports every tool round, which is precisely the
        kind of turn that runs out of context.
        """
        total = 0
        for message in self._history:
            total += len(str(message.get("content") or ""))
            calls = message.get("tool_calls")
            if calls:
                try:
                    total += len(json.dumps(calls, ensure_ascii=False, default=str))
                except (TypeError, ValueError):
                    total += 200  # unserialisable, but it still costs something
        return total


def summarize_arguments(arguments: dict[str, Any], limit: int = 80) -> str:
    """One-line rendering of tool arguments for the CLI."""
    try:
        text = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(arguments)
    return text if len(text) <= limit else text[: limit - 1] + "…"
