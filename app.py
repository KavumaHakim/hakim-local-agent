"""Streamlit chat interface for the local agent.

    streamlit run app.py

A front end only: it reuses the same Agent, ToolRegistry and QwenClient as the
CLI, so the workspace jail and the disabled-by-default tools apply here exactly
as they do in the terminal.

Tokens stream as they arrive. On CPU a turn takes minutes, so without
streaming the page would simply sit blank.
"""

from __future__ import annotations

import dataclasses
import time

import streamlit as st

import ui_commands
import ui_style
from agent.loop import Agent, AgentError, IterationLimitError, ToolEvent
from chat_store import ChatStore, make_title
from agent.router import TaskRouter
from config import load_config
from models.manager import ModelManager, ModelManagerError, ModelState
from models.qwen import QwenClient, QwenError
from tools.registry import build_default_registry

st.set_page_config(
    page_title="Hakim AI System",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.markdown(ui_style.CSS, unsafe_allow_html=True)

USER_AVATAR = "🧑"
AGENT_AVATAR = "🤖"

SUGGESTIONS = [
    "Calculate sqrt(144) + 25^2",
    "List the files in the workspace root",
    "Read requirements.txt and tell me what it depends on",
    "What is 17 * 43 - 209?",
]


# --------------------------------------------------------------------------
# service wiring
# --------------------------------------------------------------------------


@st.cache_resource
def get_manager() -> ModelManager:
    """One manager per Streamlit process: it owns the llama-server children."""
    return ModelManager()


@st.cache_resource
def get_store() -> ChatStore:
    """Conversation history, shared by every session in this process."""
    return ChatStore(load_config().db_path)


def save_turn(role: str, content: str, **extra) -> None:
    """Append one message to the open conversation, opening one if needed."""
    store = get_store()
    if st.session_state.conversation_id is None:
        st.session_state.conversation_id = store.create_conversation(
            title=make_title(content) if role == "user" else "New conversation",
            model_key=st.session_state.get("model_key"),
        )
    store.add_message(
        st.session_state.conversation_id,
        role,
        content,
        model_key=st.session_state.get("model_key"),
        **extra,
    )


def load_conversation(conversation_id: int) -> None:
    """Replace the visible chat and the agent's memory with a saved one."""
    store = get_store()
    stored = store.get_messages(conversation_id)
    st.session_state.conversation_id = conversation_id
    st.session_state.messages = [m.as_ui_dict() for m in stored]
    # The agent keeps its own transcript, and it must match what is on screen
    # or the model will answer against a conversation the user cannot see.
    service = st.session_state.get("service")
    if service:
        service["agent"].load_history(
            [{"role": m.role, "content": m.content} for m in stored]
        )


def build_service(enable_thinking: bool, url: str) -> dict:
    """Build the client, registry and agent for the current settings."""
    config = load_config()
    # Config is frozen, so a settings change produces a new one rather than
    # mutating the object the agent already holds.
    config = dataclasses.replace(
        config, enable_thinking=enable_thinking, qwen_url=url
    )
    client = QwenClient(config)
    registry, disabled = build_default_registry(config)
    return {
        "config": config,
        "client": client,
        "registry": registry,
        "disabled": disabled,
        "agent": Agent(client, config, registry),
    }


def get_service(enable_thinking: bool, url: str) -> dict:
    """Return the cached service, rebuilding it if the settings changed."""
    service = st.session_state.get("service")
    stale = (
        service is None
        or service["config"].enable_thinking != enable_thinking
        or service["config"].qwen_url != url
    )
    if stale:
        history = service["agent"].history if service else []
        service = build_service(enable_thinking, url)
        service["agent"].load_history(history)  # carry the conversation over
        st.session_state.service = service
    return service


st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("thinking", False)
st.session_state.setdefault("auto_route", False)
# Set once a conversation has needed the strong model, so it stays there.
st.session_state.setdefault("escalated", False)
# None until the first message, so idle sessions do not litter the database.
st.session_state.setdefault("conversation_id", None)


def run_command(text: str, manager: ModelManager, service: dict, current: str) -> str:
    """Handle a /command typed into the chat box. Returns the reply markdown."""
    parts = text.split(maxsplit=1)
    name = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if name == "/help":
        return ui_commands.help_text()

    if name == "/tools":
        lines = ["**Tools**", ""]
        for category, tools in service["registry"].categories().items():
            names = ", ".join(f"`{t.name}`" for t in tools)
            lines.append(f"- **{category}** — {names}")
        for item in service["disabled"]:
            lines.append(f"- **{item.category}** — disabled: {item.reason}")
        return "\n".join(lines)

    if name == "/models":
        lines = ["**Models**", ""]
        for status in manager.statuses():
            marks = []
            if status.spec.key == current:
                marks.append("selected")
            if status.state is ModelState.READY:
                marks.append("loaded")
            if not status.spec.available:
                marks.append("file missing")
            suffix = f" — {', '.join(marks)}" if marks else ""
            lines.append(
                f"- `{status.spec.key}` {status.spec.label} "
                f":{status.spec.port}{suffix}"
            )
        return "\n".join(lines)

    if name == "/model":
        if not argument:
            return "Give a model key, e.g. `/model tiny`. Use `/models` to see them."
        try:
            spec = manager.get_spec(argument)
        except ModelManagerError as exc:
            return str(exc)
        st.session_state.model_key = argument
        # An explicit choice overrides the router's no-downgrade rule.
        st.session_state.escalated = argument == manager.router_strong
        return f"Selected **{spec.label}**. It loads on your next message."

    if name == "/unload":
        if manager.stop(current):
            return "Model unloaded."
        return "That server was not started by the agent, so it was left alone."

    if name == "/auto":
        st.session_state.auto_route = not st.session_state.auto_route
        if st.session_state.auto_route:
            return (
                "Automatic routing **on**. Simple prompts go to "
                f"`{manager.router_fast}`, involved ones to "
                f"`{manager.router_strong}`."
            )
        return "Automatic routing **off**."

    if name == "/clear":
        service["agent"].clear()
        st.session_state.messages = []
        st.session_state.escalated = False
        st.session_state.conversation_id = None
        return ""

    return f"Unknown command `{name}`. Type `/help`."


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="chat-header"><div class="mark">🤖</div>'
        "<div><h1>Hakim AI System</h1>"
        '<div class="sub">Local models · llama.cpp</div></div></div>',
        unsafe_allow_html=True,
    )

    manager = get_manager()
    specs = manager.specs()
    states = {s.spec.key: s for s in manager.statuses()}
    router = TaskRouter(
        manager.router_fast,
        manager.router_strong,
        enabled=st.session_state.auto_route,
    )

    def _label(key: str) -> str:
        spec = next(s for s in specs if s.key == key)
        if not spec.available:
            return f"{spec.label}  (file missing)"
        mark = " ●" if states[key].state is ModelState.READY else ""
        return f"{spec.label}{mark}"

    keys = [spec.key for spec in specs]
    current = st.session_state.get("model_key") or manager.active_key() or manager.default_key
    chosen = st.selectbox(
        "Model",
        keys,
        index=keys.index(current) if current in keys else 0,
        format_func=_label,
        help="Only one model is held in RAM at a time; switching unloads the other.",
    )
    st.session_state.model_key = chosen

    spec = next(s for s in specs if s.key == chosen)
    st.caption(spec.description)

    # Settings are set once and then ignored, so they fold away.
    with st.expander("Settings"):
        auto = st.toggle(
            "Auto-route by task",
            value=st.session_state.auto_route,
            help=(
                f"Send simple prompts to "
                f"{manager.get_spec(manager.router_fast).label} and involved "
                f"ones to {manager.get_spec(manager.router_strong).label}. "
                "Never routes back down within a conversation, because "
                "reloading costs more than the RAM is worth."
            ),
        )
        st.session_state.auto_route = auto
        router.enabled = auto

        thinking = st.toggle(
            "Extended thinking",
            value=st.session_state.thinking,
            help=(
                "Qwen3 reasons before answering. Much slower on CPU, and the "
                "reasoning itself is never displayed."
            ),
        )
        st.session_state.thinking = thinking

    # Loading a model can take minutes on this machine, so it happens on an
    # explicit click rather than silently on every rerun.
    ready = states[chosen].state is ModelState.READY
    if not ready:
        if st.button(
            f"Load {spec.label}",
            use_container_width=True,
            disabled=not spec.available,
        ):
            with st.spinner(f"Loading {spec.label}… this can take a few minutes."):
                try:
                    manager.ensure(chosen)
                except ModelManagerError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()
        if not spec.available:
            st.caption(f"Not found: `{spec.path.name}`")

    service = get_service(thinking, spec.url)
    config = service["config"]

    st.markdown(
        ui_style.status(ready, config.qwen_url),
        unsafe_allow_html=True,
    )

    if ready and st.button("Unload model", use_container_width=True):
        manager.stop(chosen)
        st.rerun()

    if states[chosen].error:
        st.caption(states[chosen].error)
    if states[chosen].warning:
        st.warning(states[chosen].warning, icon="⚠️")

    # Give back RAM after a long pause without needing a background thread.
    for released in manager.unload_idle():
        st.toast(f"Unloaded {released} after idle timeout")

    # The tool list and the reasons behind the disabled ones are reference
    # material: worth having, not worth a screenful every time.
    active_tools = service["registry"].names()
    with st.expander(f"Tools ({len(active_tools)})"):
        for category, tools in service["registry"].categories().items():
            names = " · ".join(tool.name for tool in tools)
            st.markdown(
                f'<div class="tool-item"><span class="cat">{category}</span> '
                f"{names}</div>",
                unsafe_allow_html=True,
            )
        if service["disabled"]:
            st.markdown(
                '<div class="side-label">Disabled</div>', unsafe_allow_html=True
            )
            for item in service["disabled"]:
                st.markdown(
                    f'<div class="tool-off"><b>{item.category}</b> — '
                    f"{item.reason}</div>",
                    unsafe_allow_html=True,
                )
        st.markdown(
            '<div class="side-label">Workspace</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="path">{config.workspace}</div>', unsafe_allow_html=True
        )

    if st.button("New conversation", use_container_width=True):
        service["agent"].clear()
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.session_state.escalated = False
        st.rerun()

    store = get_store()
    conversations = store.list_conversations(limit=20)
    # Open when there is something to pick up, shut when the list is empty.
    with st.expander(f"History ({len(conversations)})", expanded=bool(conversations)):
        if not conversations:
            st.markdown(
                '<div class="hist-empty">Nothing saved yet.</div>',
                unsafe_allow_html=True,
            )
        for conversation in conversations:
            active = conversation.id == st.session_state.conversation_id
            row, remove = st.columns([5, 1])
            if row.button(
                ("● " if active else "") + conversation.title,
                key=f"conv{conversation.id}",
                use_container_width=True,
                help=(
                    f"{conversation.message_count} messages · "
                    f"{conversation.updated_at}"
                ),
            ):
                load_conversation(conversation.id)
                st.rerun()
            if remove.button("✕", key=f"del{conversation.id}", help="Delete"):
                store.delete_conversation(conversation.id)
                if active:
                    st.session_state.messages = []
                    st.session_state.conversation_id = None
                    service["agent"].clear()
                st.rerun()


# --------------------------------------------------------------------------
# conversation
# --------------------------------------------------------------------------

st.markdown(
    ui_style.header(f"{spec.label} · tools · nothing leaves this machine"),
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        '<div class="empty-hero"><h2>What can I help with?</h2>'
        "<p>Ask a question, or give the agent a task it can use its tools for.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    for index, suggestion in enumerate(SUGGESTIONS):
        column = left if index % 2 == 0 else right
        if column.button(suggestion, key=f"sug{index}", use_container_width=True):
            st.session_state.pending = suggestion
            st.rerun()

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else AGENT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("tools"):
            st.markdown(ui_style.tool_pills(message["tools"]), unsafe_allow_html=True)
        st.markdown(message["content"])
        if message.get("elapsed"):
            st.markdown(
                f'<div class="turn-meta">{message["elapsed"]}s</div>',
                unsafe_allow_html=True,
            )


typed = st.chat_input("Message Hakim…  (type / for commands)")
# The dropdown attaches to the box above; zero height so it takes no space.
# st.iframe grants the frame same-origin access, which the script needs to
# reach the chat textarea. The HTML is built from a fixed command list here,
# never from user input.
st.iframe(ui_commands.palette_script(), height=1)

prompt = typed or st.session_state.pending
st.session_state.pending = None

if prompt and prompt.strip().startswith("/"):
    reply = run_command(prompt.strip(), manager, service, chosen)
    if reply:
        st.session_state.messages.append(
            {"role": "user", "content": prompt.strip()}
        )
        st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_turn("user", prompt)
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # Pick the model for this turn before loading anything, so a correct
    # guess costs nothing.
    target = chosen
    if st.session_state.auto_route:
        decision = router.choose(
            prompt,
            current_key=chosen,
            escalated=st.session_state.escalated,
        )
        target = decision.key
        if target != chosen:
            st.session_state.model_key = target
            st.info(
                f"Switching to {manager.get_spec(target).label} — {decision.reason}"
            )
        if target == manager.router_strong:
            st.session_state.escalated = True
        spec = manager.get_spec(target)
        service = get_service(thinking, spec.url)

    try:
        with st.spinner(f"Starting {spec.label}…"):
            manager.ensure(target)
    except ModelManagerError as exc:
        st.error(str(exc))
        st.stop()

    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        tool_area = st.empty()
        answer_area = st.empty()
        meta_area = st.empty()

        # Until the first token lands there is nothing to show, and on this
        # hardware that gap is minutes long.
        answer_area.markdown(ui_style.TYPING, unsafe_allow_html=True)

        buffer: list[str] = []
        calls: list[dict] = []
        started = time.time()

        def on_token(text: str) -> None:
            buffer.append(text)
            answer_area.markdown("".join(buffer) + "▌")

        def on_tool(event: ToolEvent) -> None:
            calls.append(
                {
                    "name": event.call.name,
                    "ok": event.result.ok,
                    "summary": event.result.summary(60),
                }
            )
            tool_area.markdown(ui_style.tool_pills(calls), unsafe_allow_html=True)
            # A tool round produces no prose, so clear the buffer to avoid
            # gluing two rounds of text together.
            buffer.clear()
            answer_area.markdown(ui_style.TYPING, unsafe_allow_html=True)

        try:
            turn = service["agent"].send(prompt, observer=on_tool, on_token=on_token)
            elapsed = round(time.time() - started, 1)
            answer_area.markdown(turn.content)
            meta_area.markdown(
                f'<div class="turn-meta">{elapsed}s</div>', unsafe_allow_html=True
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": turn.content,
                    "tools": calls,
                    "elapsed": elapsed,
                }
            )
            save_turn("assistant", turn.content, tools=calls, elapsed=elapsed)
        except IterationLimitError as exc:
            # The turn ran out of tool rounds. That is the clearest signal the
            # small model is out of its depth, so offer the strong one rather
            # than silently burning another few minutes on a retry.
            answer_area.empty()
            meta_area.empty()
            st.warning(str(exc))
            if (
                st.session_state.auto_route
                and target != manager.router_strong
                and not st.session_state.escalated
            ):
                st.session_state.escalated = True
                st.session_state.model_key = manager.router_strong
                st.session_state.pending = prompt
                st.info(
                    f"Retrying on "
                    f"{manager.get_spec(manager.router_strong).label}."
                )
                st.rerun()
            st.session_state.messages.append(
                {"role": "assistant", "content": f"**Stopped:** {exc}", "tools": calls}
            )
        except (QwenError, AgentError) as exc:
            answer_area.empty()
            meta_area.empty()
            st.error(str(exc))
            st.session_state.messages.append(
                {"role": "assistant", "content": f"**Error:** {exc}", "tools": calls}
            )

    st.rerun()
