"""CLI entry point for the local agent.

Usage:
    python main.py
"""

from __future__ import annotations

import sys

import dataclasses

from agent.loop import Agent, AgentError, ToolEvent
from config import load_config
from models.manager import ModelManager, ModelManagerError, ModelState
from models.qwen import QwenClient, QwenError
from tools.registry import build_default_registry

HELP = """Commands:
  /help          show this message
  /tools         list the tools the agent can use
  /models        list models and which one is loaded
  /model <key>   switch to a model (unloads the current one)
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
        if spec.key == current:
            marks.append("selected")
        if status.state is ModelState.READY:
            marks.append("loaded (external)" if status.adopted else "loaded")
        if not spec.available:
            marks.append("FILE MISSING")
        suffix = f"   [{', '.join(marks)}]" if marks else ""
        print(f"  {spec.key:<10} {spec.label:<18} :{spec.port}{suffix}")
    print("\nSwitch with /model <key>. Only one model stays in RAM.\n")


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
