"""Process-wide objects, and the function that actually runs a turn.

Streamlit held these in `@st.cache_resource`, which meant one per Streamlit
process. Here they hang off the FastAPI app's lifespan, which means one per
uvicorn worker - so the server must run with a single worker. Two workers
would mean two ModelManagers, each convinced it owns `llama-server.exe`, and
they would fight over the same ports.

The agent itself is deliberately *not* kept between turns. It is rebuilt from
the stored conversation each time, which costs nothing (history is a list of
dicts), survives a restart, and removes any need for session affinity.
llama.cpp's prefix cache still hits, because the messages it is sent are
byte-identical to last time.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from agent.loop import Agent, AgentError, IterationLimitError, ToolEvent
from agent.router import TaskRouter
from chat_store import ChatStore
from config import Config, load_config
from models.manager import ModelManager, ModelManagerError
from models.qwen import QwenClient, QwenError
from tools.registry import build_default_registry

from api.turns import Turn, TurnQueue


# The tool switches the UI may flip, and the Config field each one sets.
#
# Enabling a tool and enabling its sharp end are separate switches, exactly as
# they are separate environment variables: `depends_on` is what makes the UI
# nest the second under the first rather than offering it on its own.
@dataclasses.dataclass(frozen=True)
class ToolFlag:
    id: str
    field: str
    label: str
    depends_on: str | None = None


TOOL_FLAGS: tuple[ToolFlag, ...] = (
    ToolFlag("file_writes", "file_writes_enabled", "File writes"),
    ToolFlag("python", "python_tool_enabled", "Python"),
    ToolFlag(
        "python_unrestricted",
        "python_unrestricted",
        "Python — unrestricted scripts",
        depends_on="python",
    ),
    ToolFlag("terminal", "shell_tool_enabled", "Terminal"),
    ToolFlag("http", "http_tool_enabled", "HTTP"),
    ToolFlag(
        "http_writes",
        "http_allow_writes",
        "HTTP — state-changing methods",
        depends_on="http",
    ),
    ToolFlag("git", "git_tool_enabled", "Git"),
    ToolFlag(
        "git_writes", "git_allow_writes", "Git — commit and branch", depends_on="git"
    ),
    ToolFlag("memory", "memory_tool_enabled", "Memory"),
    ToolFlag("ocr", "ocr_enabled", "OCR"),
)

FLAGS_BY_ID = {flag.id: flag for flag in TOOL_FLAGS}


class Runtime:
    """The objects a request needs, built once for the process."""

    def __init__(
        self, config: Config | None = None, manager: ModelManager | None = None
    ) -> None:
        self.config = config or load_config()
        # Injectable so tests can supply the manager harness the model tests
        # already use, rather than one that probes real ports.
        self.manager = manager if manager is not None else ModelManager()
        self.store = ChatStore(self.config.db_path)
        self.queue = TurnQueue(self.run_turn)
        # Tool switches flipped from the UI, applied on top of the environment.
        # Held in memory only: a restart returns to whatever the environment
        # says, so the env vars stay the durable answer to "what is on here"
        # and a switch cannot quietly become permanent.
        self._overrides: dict[str, bool] = {}

    # --- tool switches ---

    @property
    def overrides(self) -> dict[str, bool]:
        return dict(self._overrides)

    def set_override(self, flag_id: str, enabled: bool) -> None:
        """Turn one tool switch on or off for this process."""
        flag = FLAGS_BY_ID[flag_id]
        self._overrides[flag.field] = enabled
        if not enabled:
            # Turning off a tool turns off its sharp end too, so the pair can
            # never end up in the state "unrestricted Python, no Python tool".
            for other in TOOL_FLAGS:
                if other.depends_on == flag_id:
                    self._overrides[other.field] = False

    def effective_config(self) -> Config:
        """The config with the UI's switches applied."""
        if not self._overrides:
            return self.config
        return dataclasses.replace(self.config, **self._overrides)

    # --- introspection used by several routes ---

    def registry_for(self, config: Config) -> tuple[Any, list[Any]]:
        """Build the tool registry for a config.

        Rebuilt rather than cached because the registry is derived entirely
        from the config, and a settings change must not leave a stale one
        behind offering tools that are no longer enabled.
        """
        return build_default_registry(config)

    def turn_config(self, *, qwen_url: str, enable_thinking: bool) -> Config:
        """The frozen config for one turn, with per-turn settings applied.

        Built from `effective_config` so a tool switched on in the UI is
        available to the very next turn, with no restart.
        """
        return dataclasses.replace(
            self.effective_config(),
            qwen_url=qwen_url,
            enable_thinking=enable_thinking,
        )

    # --- seams ---
    #
    # The two places a turn touches something that needs a live llama-server.
    # They are methods rather than direct calls so tests can override them and
    # exercise the queue, the event stream and the persistence without one -
    # a test that quietly depends on whether a server happens to be listening
    # passes for the wrong reason and breaks the moment one is.

    def make_client(self, config: Config) -> Any:
        """The chat client for one turn."""
        return QwenClient(config)

    def ensure_model(self, key: str) -> str:
        """Make `key` resident, returning its base URL."""
        return self.manager.ensure(key)

    def escalated(self, conversation_id: int) -> bool:
        """Whether this conversation has already needed the strong model.

        Derived from the stored messages rather than kept in a session, so it
        survives a restart and needs no extra column. The router's no-downgrade
        rule depends on it: once a conversation has paid to load the strong
        model, going back would pay the switch cost twice to free RAM that is
        already spent.
        """
        strong = self.manager.router_strong
        return any(
            message.model_key == strong
            for message in self.store.get_messages(conversation_id)
        )

    # --- the turn itself ---

    def run_turn(self, turn: Turn) -> None:
        """Run one turn to completion, reporting progress as events.

        Called on the queue's worker thread. Every failure is reported as an
        event rather than raised, because by this point the HTTP response has
        long since started and there is no status code left to set.
        """
        request = turn.request
        started = time.time()
        # Bound before the try so the error paths can report what was actually
        # attempted, rather than digging them back out of locals().
        target = request.model_key
        calls: list[dict[str, Any]] = []

        try:
            stored = self.store.get_messages(request.conversation_id)
            # Everything before this turn's own user message. Comparing ids is
            # exact even when several turns for one conversation are queued.
            history = [
                {"role": message.role, "content": message.content}
                for message in stored
                if message.id < request.user_message_id
            ]

            target = self._route(turn, stored)
            spec = self.manager.get_spec(target)


            # Loading is the long silence at the front of a cold turn - up to
            # 130 s for the 8B - so it gets its own event.
            turn.emit("model", key=target, label=spec.label, state="loading")
            self.ensure_model(target)
            status = self.manager.status(target)
            turn.emit(
                "model",
                key=target,
                label=spec.label,
                state="ready",
                warning=status.warning or "",
            )

            config = self.turn_config(
                qwen_url=spec.url, enable_thinking=request.enable_thinking
            )
            registry, _ = self.registry_for(config)
            agent = Agent(self.make_client(config), config, registry)
            agent.load_history(history)

            def on_token(text: str) -> None:
                turn.emit("token", text=text)

            def on_reasoning(text: str) -> None:
                # Streamed on its own channel and never stored: the agent's
                # history keeps the answer and the tool calls only, and
                # replaying a thinking trace to the model is not something the
                # chat template expects.
                turn.emit("reasoning", text=text)

            def on_tool(event: ToolEvent) -> None:
                entry = {
                    "name": event.call.name,
                    "ok": event.result.ok,
                    "summary": event.result.summary(60),
                }
                calls.append(entry)
                # A tool round produces no prose, so the client clears whatever
                # it has streamed on seeing this - otherwise two rounds of text
                # are glued together.
                turn.emit("tool", **entry)

            turn.emit("start", model_key=target)
            result = agent.send(
                request.prompt,
                observer=on_tool,
                on_token=on_token,
                on_reasoning=on_reasoning,
            )
            elapsed = round(time.time() - started, 1)

            message_id = self.store.add_message(
                request.conversation_id,
                "assistant",
                result.content,
                tools=calls,
                elapsed=elapsed,
                model_key=target,
            )
            turn.emit(
                "done",
                message_id=message_id,
                content=result.content,
                tools=calls,
                elapsed=elapsed,
                model_key=target,
            )

        except IterationLimitError as exc:
            # The clearest signal the small model is out of its depth. The
            # client decides whether to retry on the strong one; the server
            # does not silently burn another few minutes deciding for it.
            turn.emit(
                "error",
                kind="iteration_limit",
                message=str(exc),
                tools=calls,
                can_escalate=target != self.manager.router_strong,
            )
        except ModelManagerError as exc:
            turn.emit("error", kind="model", message=str(exc), tools=calls)
        except (QwenError, AgentError) as exc:
            turn.emit("error", kind="agent", message=str(exc), tools=calls)

    def _route(self, turn: Turn, stored: list[Any]) -> str:
        """Pick the model for this turn, announcing a change if there is one.

        Routing happens before anything is loaded, so a correct guess costs
        nothing at all. The decision's reason is reported because a model
        changing underneath you without explanation is alarming.
        """
        request = turn.request
        if not request.auto_route:
            return request.model_key

        strong = self.manager.router_strong
        router = TaskRouter(self.manager.router_fast, strong, enabled=True)
        decision = router.choose(
            request.prompt,
            current_key=request.model_key,
            escalated=any(message.model_key == strong for message in stored),
        )
        if decision.key != request.model_key:
            turn.emit(
                "route",
                key=decision.key,
                label=self.manager.get_spec(decision.key).label,
                reason=decision.reason,
            )
        return decision.key


def open_conversation(
    runtime: Runtime, conversation_id: int | None, prompt: str, model_key: str | None
) -> int:
    """Return the conversation to append to, creating one if needed.

    Conversations are created on the first message rather than on page load,
    so an idle session never litters the database.
    """
    from chat_store import make_title

    if conversation_id is not None:
        return conversation_id
    return runtime.store.create_conversation(
        title=make_title(prompt), model_key=model_key
    )
