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
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable

from agent.loop import Agent, AgentError, IterationLimitError, ToolEvent, TurnStopped
from agent.router import TaskRouter
from chat_store import ChatStore
from config import Config, load_config
from models.connectivity import Connectivity
from models.manager import ModelManager, ModelManagerError, ModelSpec
from models.qwen import QwenClient, QwenError
from models.remote import RemoteClient
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
    ToolFlag(
        "memory_processing",
        "memory_processing_enabled",
        "Memory - background processing",
        depends_on="memory",
    ),
    ToolFlag("documents", "rag_enabled", "Document search"),
    ToolFlag("ocr", "ocr_enabled", "OCR"),
)

FLAGS_BY_ID = {flag.id: flag for flag in TOOL_FLAGS}

# How many past workspaces the picker offers back.
RECENT_WORKSPACES = 8


class WorkspaceError(ValueError):
    """A folder that cannot be the workspace, and why."""


def _system_roots() -> list[Path]:
    """Directories that must never become the jail.

    Not a security boundary - anyone who can reach this API can already ask
    for a shell - but a guard against the two mistakes that turn the jail into
    no jail at all: pointing it at a drive root, or at the operating system.
    """
    roots: list[Path] = []
    for name in ("SystemRoot", "windir", "ProgramFiles", "ProgramFiles(x86)"):
        value = os.environ.get(name, "").strip()
        if value:
            try:
                roots.append(Path(value).resolve())
            except OSError:
                continue
    if os.name != "nt":
        roots.extend(
            Path(p) for p in ("/bin", "/boot", "/dev", "/etc", "/proc", "/sys", "/usr")
        )
    return roots


def resolve_workspace(raw: str | Path) -> Path:
    """Turn a folder someone typed or clicked into a usable workspace root.

    Resolved before it is checked, exactly as `WorkspaceFiles` resolves the
    paths the model gives it: a check against an unresolved path says nothing
    about where it actually lands.
    """
    text = str(raw).strip().strip('"')
    if not text:
        raise WorkspaceError("Give a folder.")

    try:
        path = Path(text).expanduser().resolve()
    except OSError as exc:
        raise WorkspaceError(f"That is not a usable path: {exc}") from None

    if not path.exists():
        raise WorkspaceError(f"There is no such folder: {path}")
    if not path.is_dir():
        raise WorkspaceError(f"That is a file, not a folder: {path}")
    if path.parent == path:
        raise WorkspaceError(
            f"{path} is the root of a drive. A workspace there is no jail at "
            f"all - pick the folder you actually want the agent working in."
        )
    for root in _system_roots():
        if path == root or root in path.parents or path in root.parents:
            raise WorkspaceError(
                f"{path} holds part of the operating system. Pick a folder of "
                f"your own instead."
            )

    # A folder that cannot be listed is not a workspace; finding that out now
    # is better than every tool call failing later.
    try:
        next(path.iterdir(), None)
    except OSError as exc:
        raise WorkspaceError(f"That folder cannot be read: {exc}") from None

    return path

# How much of a tool call to keep for display.
#
# The model's own limits are much larger - a Python snippet may be 100,000
# characters and a file read 200,000 - and all of it would otherwise be
# serialised into the message row and sent to the browser on every reload. This
# is what gets shown and stored; it is not what the model saw, so anything cut
# says so rather than trailing off and looking complete.
TOOL_DISPLAY_LIMIT = 8_000


def _clip(text: str, limit: int = TOOL_DISPLAY_LIMIT) -> tuple[str, bool]:
    """Cut `text` to `limit`, reporting whether anything was removed."""
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def stopped_text(partial: str) -> str:
    """What a stopped turn stores as its assistant message.

    The marker is not decoration. A partial answer with nothing to say so
    reads, on reload, as an answer that simply ended - which is the same
    reason a clipped tool result says it was clipped rather than trailing
    off and looking complete.

    Asterisks rather than underscores because that is the italic the UI's
    markdown subset actually renders - underscores would show as themselves,
    which is a scruffier way of saying the same thing.
    """
    marker = "*(stopped before finishing)*"
    return f"{partial}\n\n{marker}" if partial else marker


def tool_entry(event: ToolEvent) -> dict[str, Any]:
    """What the UI is told about one tool call.

    Carries the arguments and the whole result payload, not just the one-line
    summary: "it ran read_text_file" is not enough to check its work, which is
    the entire reason for looking.
    """
    arguments, arguments_clipped = _clip(
        json.dumps(event.call.arguments, ensure_ascii=False, indent=2, default=str)
    )
    output, output_clipped = _clip(
        json.dumps(event.result.payload, ensure_ascii=False, indent=2, default=str)
    )
    return {
        "name": event.call.name,
        "ok": event.result.ok,
        "summary": event.result.summary(60),
        "arguments": arguments,
        "output": output,
        "clipped": arguments_clipped or output_clipped,
    }


@dataclasses.dataclass(frozen=True)
class ModelChoice:
    """Which model will answer a turn, and how that was decided.

    Worked out before the turn is queued rather than inside it, because a
    hosted model needs the user's agreement first and the agent loop cannot
    stop halfway through to ask for it.
    """

    key: str
    reason: str = ""
    # True when the auto-router picked this rather than the user.
    routed: bool = False
    # Set when a hosted model was wanted and the network was down.
    fell_back_from: str | None = None


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
        # Model downloads, into the same folder discovery watches. Built here
        # rather than per-request so progress survives between polls.
        self._downloads = None
        self.queue = TurnQueue(self.run_turn)
        self.connectivity = Connectivity()
        # Built lazily: memory pulls in numpy, and a runtime for a CLI-only
        # or memory-off setup should not pay for that at startup.
        self._memory = None
        self._memory_lock = threading.Lock()
        # Settings changed from the UI, applied on top of the environment.
        # Held in memory only: a restart returns to whatever the environment
        # says, so the env vars stay the durable answer to "what is on here"
        # and a switch cannot quietly become permanent.
        #
        # Mostly booleans - the tool switches - but not only: the OCR backend
        # is a string and the workspace is a Path.
        self._overrides: dict[str, Any] = {}
        # Workspaces used in this process, most recent first, starting with
        # the one the environment chose. In memory for the same reason the
        # switches are: nothing here should outlive the process quietly.
        self._recent_workspaces: list[Path] = [self.config.workspace]
        self._voice = None
        # How the shutdown route ends the process, once it has answered.
        # SIGINT rather than anything harder, so uvicorn runs the same
        # lifespan shutdown Ctrl+C does - the one that unloads models and
        # the embedding worker. Python delivers the handler to the main
        # thread whichever thread raises it, which is what lets a request
        # thread do this. Injectable so a test can watch it without killing
        # the test runner.
        self.exit_process: Callable[[], None] = lambda: signal.raise_signal(
            signal.SIGINT
        )

    @property
    def voice(self):
        """The Piper voice, started on first use and swept when idle.

        Lazy for the same reason as `downloads`: a session where nobody
        presses the speaker button should not pay for the import, let alone
        the 175 MB.
        """
        if self._voice is None:
            from speech.piper import Voice

            config = self.effective_config()
            self._voice = Voice(
                voice=config.piper_voice,
                idle_seconds=config.piper_idle_seconds,
            )
        return self._voice

    def sweep_voice(self) -> bool:
        """Give the voice's memory back when nobody is listening.

        Only touches a voice that was actually started - asking for one here
        would load a model on a machine that has never used it, which is the
        opposite of what a sweep is for.
        """
        if self._voice is None:
            return False
        return self._voice.unload_if_idle()

    @property
    def downloads(self):
        """Model downloads, pointed at the folder discovery already watches.

        Lazy because it needs the manager's models folder, and because a run
        that never opens the model browser should not pay for the import.
        """
        if self._downloads is None:
            from models.hub import Downloads

            self._downloads = Downloads(self.manager.models_dir)
        return self._downloads

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

    def set_ocr_backend(self, backend: str) -> None:
        """Choose the OCR reader for this process.

        The tool switches are booleans and share one mechanism; this is a
        string, so it gets its own setter rather than bending that one. Like
        the switches it lives in memory only - a restart returns to whatever
        OCR_BACKEND says.
        """
        if backend not in ("tesseract", "model"):
            raise ValueError(f"Unknown OCR backend {backend!r}.")
        self._overrides["ocr_backend"] = backend

    # --- workspace ---
    #
    # The workspace is the jail every file-touching tool resolves against, and
    # until now it could only be chosen before startup with AGENT_WORKSPACE.
    # Moving it from the UI is the same loosening the tool switches were: the
    # jail itself is unchanged - one directory, resolved paths, no escape - but
    # which directory it is has become a click rather than a restart.
    #
    # Like the switches, it lives in memory only, so the environment stays the
    # durable answer to "what can this thing reach".

    @property
    def workspace(self) -> Path:
        """The workspace the next turn will use."""
        return self.effective_config().workspace

    @property
    def default_workspace(self) -> Path:
        """The workspace the environment chose, and a restart returns to."""
        return self.config.workspace

    @property
    def recent_workspaces(self) -> list[Path]:
        return list(self._recent_workspaces)

    def set_workspace(self, raw: str | Path) -> Path:
        """Point the file tools at another folder for this process.

        Returns the resolved path, which is rarely what was typed: `~`, a
        relative path and a trailing separator all survive the trip.
        """
        path = resolve_workspace(raw)
        self._overrides["workspace"] = path
        self._remember(path)
        return path

    def reset_workspace(self) -> Path:
        """Go back to what AGENT_WORKSPACE says."""
        self._overrides.pop("workspace", None)
        self._remember(self.config.workspace)
        return self.config.workspace

    def _remember(self, path: Path) -> None:
        self._recent_workspaces = [path] + [
            other for other in self._recent_workspaces if other != path
        ][: RECENT_WORKSPACES - 1]

    def effective_config(self) -> Config:
        """The config with the UI's switches applied."""
        if not self._overrides:
            return self.config
        return dataclasses.replace(self.config, **self._overrides)

    # --- memory ---

    def memory(self):
        """The process-wide memory manager, with its processor attached.

        Returns None when the optional dependencies are missing, which is a
        supported state: the agent simply runs without memory rather than
        failing to start.
        """
        if self._memory is not None:
            return self._memory
        with self._memory_lock:
            if self._memory is not None:
                return self._memory
            config = self.config
            try:
                from memory.manager import shared_manager
                from memory.processor import MemoryProcessor
            except ImportError:
                return None

            manager = shared_manager(
                config.db_path,
                store_dir=config.memory_store,
                dimension=config.rag_dimension,
                top_k=config.memory_top_k,
                score_floor=config.memory_score_floor,
            min_similarity=config.memory_min_similarity,
                context_tokens=config.memory_context_tokens,
                summarize_after=config.memory_summarize_after,
                extract_every=config.memory_extract_every,
                queue_high_water=config.memory_queue_high_water,
            )
            # The processor drives THIS manager - the same object that owns
            # llama-server and enforces one chat model at a time. Building a
            # second loader here is exactly what the design forbids.
            manager.attach_processor(
                MemoryProcessor(
                    manager.store,
                    manager.vectors,
                    manager=self.manager,
                    config=config,
                    aux_key=config.memory_aux_model,
                    batch_size=config.memory_batch_size,
                )
            )
            self._memory = manager
            return self._memory

    def sweep_memory(self) -> dict:
        """Run a memory batch if the triggers allow. Called by the sweeper.

        `queue.busy` is passed rather than read once, so a batch that starts
        just as a turn arrives puts its remaining jobs back instead of making
        the user wait for a model switch in each direction.
        """
        if not self.effective_config().memory_processing_enabled:
            return {"ran": False, "reason": "background processing is off"}
        manager = self.memory()
        if manager is None:
            return {"ran": False, "reason": "memory is unavailable"}
        return manager.maybe_process(busy=self.queue.busy)

    # --- introspection used by several routes ---

    def registry_for(
        self, config: Config, *, approve: Any = None
    ) -> tuple[Any, list[Any]]:
        """Build the tool registry for a config.

        Rebuilt rather than cached because the registry is derived entirely
        from the config, and a settings change must not leave a stale one
        behind offering tools that are no longer enabled.

        `approve` is how a tool asks a person before doing something that
        changes state. It is per-turn, because the question has to reach
        whoever is watching *that* turn's stream - which is the other reason
        the registry cannot be cached.
        """
        return build_default_registry(config, approve=approve)

    def turn_config(
        self, *, qwen_url: str, enable_thinking: bool, context: int = 4096
    ) -> Config:
        """The frozen config for one turn, with per-turn settings applied.

        Built from `effective_config` so a tool switched on in the UI is
        available to the very next turn, with no restart.
        """
        return dataclasses.replace(
            self.effective_config(),
            qwen_url=qwen_url,
            enable_thinking=enable_thinking,
            model_context=context,
        )

    # --- seams ---
    #
    # The two places a turn touches something that needs a live llama-server.
    # They are methods rather than direct calls so tests can override them and
    # exercise the queue, the event stream and the persistence without one -
    # a test that quietly depends on whether a server happens to be listening
    # passes for the wrong reason and breaks the moment one is.

    def make_client(self, config: Config, spec: ModelSpec) -> Any:
        """The chat client for one turn.

        The only place that decides whether a turn stays on this machine.
        """
        if spec.remote:
            return RemoteClient(
                spec,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
            )
        return QwenClient(config)

    def decide_model(
        self,
        prompt: str,
        requested: str | None,
        *,
        auto_route: bool,
        conversation_id: int | None,
    ) -> ModelChoice:
        """Pick the model for a turn, before anything is queued or stored.

        Routing used to happen inside the turn. It moved out here because a
        hosted model has to be agreed to first, and by the time the worker is
        running there is no way to ask.
        """
        chosen = requested or self.manager.active_key() or self.manager.default_key
        reason = ""
        routed = False

        if auto_route:
            strong = self.manager.router_strong
            stored = (
                self.store.get_messages(conversation_id)
                if conversation_id is not None
                else []
            )
            router = TaskRouter(self.manager.router_fast, strong, enabled=True)
            decision = router.choose(
                prompt,
                current_key=chosen,
                escalated=any(message.model_key == strong for message in stored),
            )
            routed = decision.key != chosen
            chosen = decision.key
            reason = decision.reason

        spec = self.manager.get_spec(chosen)
        if spec.remote and not self.connectivity.online():
            # Greyed out in the UI, but the network can drop between the page
            # loading and the message being sent, so it is checked again here.
            local = self.manager.default_key
            return ModelChoice(
                key=local,
                reason=(
                    f"{spec.label} needs the internet and there is none, so "
                    f"this ran on {self.manager.get_spec(local).label} instead."
                ),
                routed=routed,
                fell_back_from=chosen,
            )

        return ModelChoice(key=chosen, reason=reason, routed=routed)

    def ensure_model(self, key: str) -> str:
        """Make `key` resident, returning its base URL."""
        return self.manager.ensure(key)

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
            # The conversation as it stands, minus this turn's own question -
            # the agent appends that itself - and minus any question queued
            # behind this one.
            #
            # Not simply "everything with a lower id". Rows are not written in
            # conversation order: a queued turn's question is stored the moment
            # it is accepted, which is before the answer to the turn ahead of
            # it exists. Selecting by id alone therefore drops exactly the
            # answer the user is most likely replying to. Ids still settle the
            # other direction, because a question queued behind this one has no
            # answer yet and would have the model replying to the wrong thing.
            history = [
                {"role": message.role, "content": message.content}
                for message in stored
                if message.id != request.user_message_id
                and not (
                    message.role == "user"
                    and message.id > request.user_message_id
                )
            ]

            # Already decided, and agreed to, before this was queued.
            spec = self.manager.get_spec(target)


            # Loading is the long silence at the front of a cold turn - up to
            # 130 s for the 8B - so it gets its own event.
            turn.emit(
                "model",
                key=target,
                label=spec.label,
                state="loading",
                provider=spec.provider,
                remote=spec.remote,
            )
            self.ensure_model(target)
            status = self.manager.status(target)
            turn.emit(
                "model",
                key=target,
                label=spec.label,
                state="ready",
                provider=spec.provider,
                remote=spec.remote,
                warning=status.warning or "",
            )

            config = self.turn_config(
                qwen_url=spec.url,
                enable_thinking=request.enable_thinking,
                # So a tool result is capped against the window it has to fit,
                # not against a guess.
                context=spec.context,
            )
            registry, _ = self.registry_for(
                config,
                approve=lambda command, reason: turn.ask(
                    command, reason, timeout=self.config.approval_timeout
                ),
            )
            # Memory is attached only when its tools are on. The context
            # builder is what injects retrieved memories, so switching memory
            # off has to switch the injection off too - otherwise the roster
            # would say "no memory" while the prompt was full of it.
            memory = self.memory() if config.memory_tool_enabled else None
            agent = Agent(
                self.make_client(config, spec),
                config,
                registry,
                memory=memory,
                conversation_id=request.conversation_id,
            )
            agent.load_history(history)

            # Kept so a stopped turn can save what it had. The client
            # assembles the same text, but a stop means never getting its
            # return value - and on a machine this slow, throwing away two
            # minutes of generated prose because the last word was missing
            # would be the wrong trade.
            streamed: list[str] = []

            def on_token(text: str) -> None:
                streamed.append(text)
                turn.emit("token", text=text)

            def on_reasoning(text: str) -> None:
                # Streamed on its own channel and never stored: the agent's
                # history keeps the answer and the tool calls only, and
                # replaying a thinking trace to the model is not something the
                # chat template expects.
                turn.emit("reasoning", text=text)

            def on_tool(event: ToolEvent) -> None:
                entry = tool_entry(event)
                calls.append(entry)
                # A tool round produces no prose, so the client clears whatever
                # it has streamed on seeing this - otherwise two rounds of text
                # are glued together. The saved copy is cleared for the same
                # reason and at the same moment, so the two cannot disagree.
                streamed.clear()
                turn.emit("tool", **entry)

            turn.emit("start", model_key=target)
            result = agent.send(
                request.prompt,
                observer=on_tool,
                on_token=on_token,
                on_reasoning=on_reasoning,
                should_stop=turn.is_stopped,
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
            # Cheap and deterministic: a regex or two, an immediate write
            # for anything explicit, and at most a queued job. No model is
            # loaded or unloaded here - that happens later, when idle.
            memory_note = {}
            if memory is not None:
                try:
                    memory_note = memory.observe_turn(
                        prompt=request.prompt,
                        answer=result.content,
                        conversation_id=request.conversation_id,
                        message_count=len(stored) + 2,
                    )
                    memory.queue_summary(
                        request.conversation_id,
                        [
                            {"id": m.id, "role": m.role, "content": m.content}
                            for m in stored
                        ],
                    )
                except Exception:  # noqa: BLE001 - memory must not fail a turn
                    memory_note = {}

            turn.emit(
                "done",
                message_id=message_id,
                content=result.content,
                tools=calls,
                elapsed=elapsed,
                model_key=target,
                memory=memory_note,
                context=agent.context_report,
            )

        except TurnStopped:
            # Asked for, not a failure - so it is reported as its own thing
            # rather than as an error someone has to dismiss.
            #
            # Whatever prose had arrived is saved, marked so the transcript
            # does not show a truncated answer as a finished one. Nothing is
            # stored when nothing was generated: an empty assistant message
            # would be a row that says only "this happened".
            partial = "".join(streamed).strip()
            elapsed = round(time.time() - started, 1)
            message_id = None
            content = ""
            if partial or calls:
                content = stopped_text(partial)
                message_id = self.store.add_message(
                    request.conversation_id,
                    "assistant",
                    content,
                    tools=calls,
                    elapsed=elapsed,
                    model_key=target,
                )
            # The event carries what was *stored*, marker and all, rather than
            # the raw partial. Otherwise the client would have to append the
            # same note itself, and the wording would live in two places and
            # drift apart.
            turn.emit(
                "stopped",
                state="running",
                message_id=message_id,
                content=content,
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



def open_conversation(
    runtime: Runtime, conversation_id: int | None, prompt: str, model_key: str | None
) -> int:
    """Return the conversation to append to, creating one if needed.

    Conversations are created on the first message rather than on page load,
    so an idle session never litters the database.

    An existing conversation with nothing in it is retitled from this prompt.
    That happens when the first question was edited: the old one is gone, and
    a title still quoting it would name a question the conversation no longer
    contains.
    """
    from chat_store import make_title

    if conversation_id is not None:
        if runtime.store.message_count(conversation_id) == 0:
            runtime.store.rename_conversation(conversation_id, make_title(prompt))
        return conversation_id
    return runtime.store.create_conversation(
        title=make_title(prompt), model_key=model_key
    )
