"""CLI entry point for the local agent.

Usage:
    python main.py
"""

from __future__ import annotations

import sys

import dataclasses

from agent.loop import Agent, AgentError, ToolEvent
from config import load_config
from models.manager import (
    ModelManager,
    ModelManagerError,
    ModelState,
    available_ram_mb,
)
from models.qwen import QwenClient, QwenError
from tools.registry import build_default_registry

HELP = """Commands:
  /help          show this message
  /tools         list the tools the agent can use
  /models        list models and which one is loaded
  /model <key>   switch to a model (unloads the current one)
  /primary <key> set the model everything defaults to (remembered)
  /rescan        re-read the models folder after copying a file in
  /unload        unload the current model and free its RAM
  /clear         reset the conversation
  /quit          quit  (/exit also works)
"""


def _print_tools(agent: Agent, disabled) -> None:
    print("\nAvailable tools:")
    for category, tools in agent.tools.categories().items():
        print(f"  {category}")
        for tool in tools:
            params = tool.parameters.get("properties", {})
            required = set(tool.parameters.get("required", []))
            signature = ", ".join(
                name if name in required else f"{name}?" for name in params
            )
            print(f"    - {tool.name}({signature})")

    if disabled:
        print("\nDisabled:")
        for item in disabled:
            print(f"  {item.category}: {item.reason}")
    print()


def _on_tool_event(event: ToolEvent) -> None:
    """Show concise tool activity. Never prints reasoning."""
    status = "" if event.result.ok else " FAILED"
    print(f"  [tool: {event.call.name}]{status} {event.result.summary()}")


def _print_models(manager: ModelManager, current: str) -> None:
    print("\nModels:")
    for status in manager.statuses():
        spec = status.spec
        marks = []
        if spec.key == manager.default_key:
            marks.append("primary")
        if spec.key == current:
            marks.append("selected")
        if status.state is ModelState.READY:
            marks.append("loaded (external)" if status.adopted else "loaded")
        if manager.is_discovered(spec.key):
            marks.append("found on disk")
        if manager.is_hidden(spec.key):
            marks.append("hidden")
        if not spec.available:
            marks.append("FILE MISSING")
        suffix = f"   [{', '.join(marks)}]" if marks else ""
        print(f"  {spec.key:<24} {spec.label:<26} :{spec.port}{suffix}")

    print(f"\nModels folder: {manager.models_dir}")
    print("Drop a .gguf in there and run /rescan to pick it up.")
    print("Switch with /model <key>, set the default with /primary <key>.")
    print("Only one model stays in RAM.\n")


def _choose_primary(manager: ModelManager) -> bool:
    """Ask which model to use, on first launch only.

    Shown when the models folder holds more than one usable chat model and
    nobody has chosen between them. One model is not a choice, so a
    single-model install is never interrupted to make one.

    Answering writes the choice to data/models.local.json, so it is asked once
    rather than every start. Declining is allowed - the default stands, and the
    question comes back next time.
    """
    candidates = [
        status.spec
        for status in manager.statuses()
        if status.spec.role == "chat" and status.spec.available
        and not status.spec.remote
    ]
    if not candidates:
        return False

    print("\nWhich model should be the primary?")
    print("Everything defaults to it. You can change it later with /primary.\n")
    for position, spec in enumerate(candidates, start=1):
        marks = []
        if manager.is_discovered(spec.key):
            marks.append("found in the models folder")
        size = _model_size_mb(spec)
        if size:
            marks.append(f"{size:,} MB")
        marks.append(f"needs {spec.min_free_mb:,} MB free")
        print(f"  {position}. {spec.label}  ({', '.join(marks)})")

    free = available_ram_mb()
    if free is not None:
        print(f"\n  This machine has {free:,} MB free right now.")

    try:
        answer = input("\nNumber, or Enter to decide later: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not answer:
        return False

    try:
        spec = candidates[int(answer) - 1]
    except (ValueError, IndexError):
        print("That was not one of the numbers; leaving it for now.\n")
        return False

    try:
        manager.set_primary(spec.key)
    except ModelManagerError as exc:
        print(f"{exc}\n", file=sys.stderr)
        return False
    print(f"\nPrimary model set to {spec.label}. Change it with /primary.\n")
    return True


def _model_size_mb(spec) -> int:
    try:
        return int(spec.path.stat().st_size / (1024 * 1024))
    except (OSError, AttributeError):
        return 0


def main() -> int:
    try:
        config = load_config()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    try:
        manager = ModelManager()
    except ModelManagerError as exc:
        print(f"Model registry error: {exc}", file=sys.stderr)
        return 1

    if manager.setup_required:
        _choose_primary(manager)

    state: dict = {"key": manager.active_key() or manager.default_key}

    def rebuild() -> None:
        """Point the agent at the selected model, carrying history across."""
        spec = manager.get_spec(state["key"])
        cfg = dataclasses.replace(config, qwen_url=spec.url)
        history = state["agent"].history if state.get("agent") else []
        client = QwenClient(cfg)
        registry, disabled = build_default_registry(cfg)
        agent = Agent(client, cfg, registry)
        agent.load_history(history)
        state.update(
            config=cfg,
            client=client,
            tools=registry,
            disabled=disabled,
            agent=agent,
        )

    rebuild()
    selected = manager.get_spec(state["key"])

    print("Hakim AI System - llama.cpp")
    print(f"  model:     {selected.label} ({selected.url})")
    print(f"  workspace: {config.workspace}")
    print(f"  tools:     {', '.join(state['tools'].names())}")
    print("Type /help for commands, /quit to exit.")

    if not state["client"].health():
        print(
            f"\nNote: {selected.label} is not loaded yet. It starts on your "
            f"first message, or run /model {state['key']} to load it now."
        )
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0

        if not user_input:
            continue

        if user_input in ("/quit", "/exit"):
            print("bye")
            return 0
        if user_input == "/help":
            print(HELP)
            continue
        if user_input == "/tools":
            _print_tools(state["agent"], state["disabled"])
            continue
        if user_input == "/models" or user_input == "/model":
            _print_models(manager, state["key"])
            continue
        if user_input.startswith("/model "):
            key = user_input.split(maxsplit=1)[1].strip()
            try:
                spec = manager.get_spec(key)
                print(f"Loading {spec.label}... this can take a few minutes.")
                manager.ensure(key)
            except ModelManagerError as exc:
                print(f"{exc}\n", file=sys.stderr)
                continue
            state["key"] = key
            rebuild()
            print(f"Now using {spec.label} on {spec.url}\n")
            continue
        if user_input.startswith("/primary"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print(f"Primary is {manager.default_key}. Usage: /primary <key>\n")
                continue
            try:
                manager.set_primary(parts[1].strip())
            except ModelManagerError as exc:
                print(f"{exc}\n", file=sys.stderr)
                continue
            print(f"Primary model is now {manager.default_key}.\n")
            continue
        if user_input == "/rescan":
            try:
                added = manager.rescan()
            except ModelManagerError as exc:
                print(f"{exc}\n", file=sys.stderr)
                continue
            if added:
                print(f"Found: {', '.join(added)}\n")
            else:
                print(f"Nothing new in {manager.models_dir}.\n")
            continue
        if user_input == "/unload":
            if manager.stop(state["key"]):
                print("(model unloaded)\n")
            else:
                print("(that server was not started by the agent)\n")
            continue
        if user_input == "/clear":
            state["agent"].clear()
            print("(conversation cleared)\n")
            continue
        if user_input.startswith("/"):
            print(f"Unknown command {user_input}. Type /help.\n")
            continue

        try:
            manager.ensure(state["key"])
        except ModelManagerError as exc:
            print(f"\n{exc}\n", file=sys.stderr)
            continue

        try:
            turn = state["agent"].send(user_input, observer=_on_tool_event)
        except QwenError as exc:
            print(f"\nModel error: {exc}\n", file=sys.stderr)
            continue
        except AgentError as exc:
            print(f"\nAgent error: {exc}\n", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            print("\n(cancelled)\n")
            continue

        # Only the final answer is shown; reasoning traces are never printed.
        print(f"\nAgent: {turn.content.strip()}\n")


if __name__ == "__main__":
    raise SystemExit(main())
