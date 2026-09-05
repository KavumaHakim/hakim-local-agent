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

# Deliberately duplicated from memory/context.py rather than imported. That
# module is stdlib-only itself, but reaching it runs `memory/__init__.py`,
# which imports the manager, which imports numpy - and numpy lives in
# requirements-rag.txt, not requirements.txt. Importing it here would make a
# base install fail at `import agent.loop`, which is to say everywhere,
# including the CLI. One float is a smaller price than that.
#
# Conservative on purpose, and the same figure the chunker uses, so the
# estimate runs high and a context sized against it comes out under budget.
CHARS_PER_TOKEN = 3.5
from models.qwen import ChatClient, TokenCallback
from tools.base import ToolRegistry, ToolResult
from tools.lens import LOAD_TOOLS, ToolLens
from tools.skills import LOAD_SKILL, NEEDS_TOOLS


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
        results: Any = None,
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
        # How much this turn had to leave out. Kept on the agent rather than
        # returned from the trimmer, because trimming happens several times
        # in a turn and the total is what anyone wants to know.
        self._dropped = 0
        self._truncated_results = 0
        # What the server said about its own throughput this turn. Reported to
        # the UI; never sent to the model.
        self.stats: dict[str, Any] = {}
        # Off by default. When on, the roster reaches the model as a short
        # index and opens a group at a time - see tools/lens.py. It lives on
        # the Agent rather than the registry because what has been opened is a
        # property of *this conversation*, and the registry is shared.
        self._lens = ToolLens(tools) if config.lazy_tools else None
        # Where a result too big for the window is kept so the model can page
        # through it. None when the registry has no `read_result` - without
        # the tool, storing the text would be writing to disk for nobody.
        self._results = results

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

        # Counted per turn, not per round: what the interface reports is what
        # this turn cost, and a round is an implementation detail nobody
        # asked about.
        self._dropped = 0
        self._truncated_results = 0
        self.stats = {}

        if self._lens is not None:
            self._lens.consider(user_input)

        for _ in range(max(1, self._config.max_iterations)):
            self._check_stop(should_stop)
            # Rebuilt every round rather than once per turn: `load_tools`
            # changes what the next request may see, and a set computed before
            # the loop would not include what the model just asked for.
            definitions = self._definitions()
            # Trim before sending, not after. Trimming afterwards leaves the
            # history within budget but says nothing about the request just
            # made - the newly appended message is exactly what pushes it over,
            # and that request is the one that fails.
            self._trim_history()
            message = self._chat(definitions, on_token, on_reasoning, should_stop)
            self._absorb_stats(message)
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
                result = self._execute(call)
                whole = result.content
                content = result.content_within(self._result_budget())
                # Counted here rather than inferred later: once the result is
                # a JSON string in the history, the only way to tell a cut one
                # from a tool that happens to return a "truncated" field is to
                # compare against the uncut text, which is what this does.
                if len(content) < len(whole):
                    self._truncated_results += 1
                    content = self._offload(whole, content)
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
                        "content": content,
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

    def _definitions(self) -> list[dict[str, Any]]:
        """The tool schemas this round sends."""
        if self._lens is None:
            return self._tools.get_tool_definitions()
        return self._lens.definitions()

    def _execute(self, call: Any) -> ToolResult:
        """Run one tool call.

        `load_tools` is the lens's own, and is answered here rather than from
        the registry: the registry is shared between conversations and what a
        lens has opened is not.
        """
        if self._lens is not None and call.name == LOAD_TOOLS:
            return ToolResult(name=call.name, payload=self._lens.load(call.arguments))
        result = self._tools.execute(call.name, call.arguments)
        if call.name == LOAD_SKILL:
            return self._open_skill_tools(result)
        return result

    def _open_skill_tools(self, result: ToolResult) -> ToolResult:
        """Open the tool groups a skill said its instructions need.

        A skill that explains how to plot something is no use to a model that
        cannot see the python tool, and the round trip it would otherwise
        spend on `load_tools` discovering that is one the skill already knew
        about. The prefix-cache miss is the same one opening the group costs
        whenever it happens; this only moves it earlier.

        The declared list never reaches the model. What it is told is what
        *actually* opened, which is not the same thing: a group can be already
        open, switched off, or named wrongly in the skill file, and a model
        told about a tool whose schema is not coming will try to call it.

        When the lens is off every tool is in every request already, so there
        is nothing to open - the key is dropped and the result is otherwise
        unchanged.
        """
        payload = result.payload
        if not isinstance(payload, dict) or NEEDS_TOOLS not in payload:
            return result

        payload = dict(payload)
        wanted = payload.pop(NEEDS_TOOLS) or []
        if self._lens is None:
            return ToolResult(name=result.name, payload=payload)

        opened = sorted(self._lens.open_categories_by_name([str(w) for w in wanted]))
        if opened:
            payload["loaded_tools"] = opened
            # Appended to the existing note rather than added beside it: two
            # instruction fields in one result get read as two topics, and
            # this is a clause of the first one.
            payload["note"] = (
                f"{payload.get('note', '')} The tools these instructions need "
                f"({', '.join(opened)}) are loaded, and available from your "
                f"next message."
            ).strip()
        return ToolResult(name=result.name, payload=payload)

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
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self._history]
            # Reported on this path too. Memory is off by default, so leaving
            # the report empty here would mean the interface had nothing to
            # say about the context for most turns anybody actually runs.
            characters = sum(
                len(str(message.get("content") or "")) for message in messages
            )
            self.context_report = self._with_turn_costs(
                {
                    "memories": [],
                    "summary_used": False,
                    "messages_kept": len(self._history),
                    "messages_dropped": self._dropped,
                    "characters": characters,
                    "estimated_tokens": int(characters / CHARS_PER_TOKEN),
                }
            )
            return messages

        built = self._memory.build_context(
            system_prompt=SYSTEM_PROMPT,
            history=self._history,
            query=self._latest_user_message(),
            conversation_id=self._conversation_id,
        )
        report = built.report()
        # The builder counts its own messages and knows nothing about the
        # trimmer that ran before it, so the two dropped counts are added
        # rather than one overwriting the other.
        report["messages_dropped"] = report.get("messages_dropped", 0) + self._dropped
        self.context_report = self._with_turn_costs(report)
        return built.messages

    def _with_turn_costs(self, report: dict[str, Any]) -> dict[str, Any]:
        """Add what the report cannot know from the messages alone.

        The window it has to fit in, what the tool schemas cost on top of the
        conversation, and whether any tool result had to be cut. Without the
        limit the token figure is a number with nothing to compare it to,
        which is the state the interface was in before.
        """
        report["context_limit"] = self._config.model_context
        report["tool_tokens"] = self._definition_tokens()
        report["truncated_results"] = self._truncated_results
        report["total_estimated_tokens"] = (
            report.get("estimated_tokens", 0) + report["tool_tokens"]
        )
        return report

    def _offload(self, whole: str, cut: str) -> str:
        """Keep the whole result, and tell the model where it went.

        A cut result already says how much is missing. This turns that dead
        end into an address: the same first page, plus an id the model can
        page through with `read_result`.

        The tool's group is opened here rather than being permanently on,
        because `read_result` is useless until something has been set aside
        and its schema is not free. Opening is monotonic, so once a
        conversation has produced one large result the tool stays available -
        which is right, since a conversation that produced one usually
        produces more.

        Falls back to the cut text unchanged if anything goes wrong. Losing
        the tail is what already happened; failing the turn would be worse.
        """
        if self._results is None:
            return cut
        stored = self._results.save(whole)
        if stored is None:
            return cut
        if self._lens is not None:
            self._lens.open_categories_by_name(["results"])
        note = (
            f"\n\n[The whole result is {stored.characters:,} characters "
            f"({stored.lines:,} lines) and did not fit. It is kept as "
            f"{stored.id!r} - call read_result with that id, an offset and a "
            f"limit to read any part of it. Do not answer from the excerpt "
            f"above if the part you need is not in it.]"
        )
        return cut + note

    def _absorb_stats(self, message: dict[str, Any]) -> None:
        """Fold one round's throughput numbers into the turn's.

        A turn with tool calls is several generations, and what someone wants
        to know is what the turn cost. Token counts add up; the rate does not,
        so it is re-derived from the totals at the end - a mean weighted by
        how much each round actually produced, rather than the mean of the
        rates, which would let a two-token round count as much as a long one.
        """
        stats = message.pop("stats", None)
        if not isinstance(stats, dict) or not stats:
            return

        for key in ("output_tokens", "prompt_tokens"):
            value = stats.get(key)
            if isinstance(value, (int, float)):
                self.stats[key] = self.stats.get(key, 0) + value

        # Seconds, accumulated, so the rate can be rebuilt across rounds.
        rate = stats.get("tokens_per_second")
        produced = stats.get("output_tokens")
        if isinstance(rate, (int, float)) and rate > 0 and produced:
            self.stats["_seconds"] = self.stats.get("_seconds", 0.0) + produced / rate

        total, seconds = self.stats.get("output_tokens"), self.stats.get("_seconds")
        if total and seconds:
            self.stats["tokens_per_second"] = round(total / seconds, 2)

    def _definition_tokens(self) -> int:
        """Roughly what this round's tool schemas cost.

        The same character-based estimate the context builder uses, for the
        same reason: it runs before the model is involved, so there is no
        tokeniser to ask.
        """
        try:
            blob = json.dumps(self._definitions())
        except (TypeError, ValueError):
            return 0
        return int(len(blob) / CHARS_PER_TOKEN)

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
            self._dropped += cut

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
