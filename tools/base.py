"""The common tool abstraction.

A Tool bundles the four things both sides of the loop need: name, description,
parameter schema, and the local function that runs it. Tools return structured
dicts (``{"success": True, ...}`` / ``{"success": False, "error": ...}``); the
registry serialises them for the model.

The registry is the trust boundary. Every argument dict it receives came from
the model and is treated as untrusted input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

# A tool takes keyword arguments and returns a structured result.
ToolFunc = Callable[..., dict[str, Any]]


class ToolError(Exception):
    """A tool could not run, or ran and failed.

    The message reaches the model so it can correct itself: keep it short
    and factual.
    """


class UnknownToolError(ToolError):
    """The model asked for a tool that is not registered."""


class InvalidArgumentsError(ToolError):
    """The model's arguments did not match the tool's schema."""


@dataclass(frozen=True)
class Tool:
    name: str
    # Grouping shown by /tools: calculator, filesystem, python, ocr.
    category: str
    description: str
    # JSON Schema object describing the tool's parameters.
    parameters: dict[str, Any]
    run: ToolFunc

    def definition(self) -> dict[str, Any]:
        """The OpenAI-compatible tool definition sent to llama.cpp."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool call."""

    name: str
    payload: dict[str, Any]

    @property
    def ok(self) -> bool:
        return bool(self.payload.get("success"))

    @property
    def content(self) -> str:
        """JSON text appended to the conversation as the tool message."""
        try:
            return json.dumps(self.payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return json.dumps({"success": False, "error": "Unserialisable result"})

    def summary(self, limit: int = 100) -> str:
        """One-line rendering for the CLI."""
        if not self.ok:
            text = str(self.payload.get("error", "failed"))
        else:
            interesting = [k for k in self.payload if k != "success"]
            text = ", ".join(
                f"{k}={_short(self.payload[k])}" for k in interesting
            ) or "ok"
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"


def _short(value: Any, limit: int = 60) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def failure(message: str) -> dict[str, Any]:
    """The standard failure payload."""
    return {"success": False, "error": message}


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


class ToolRegistry:
    """Holds the available tools and executes calls against them."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def list_tools(self) -> list[Tool]:
        """Every registered tool, ordered by category then name."""
        return sorted(self._tools.values(), key=lambda t: (t.category, t.name))

    def categories(self) -> dict[str, list[Tool]]:
        """Tools grouped by category, for /tools."""
        grouped: dict[str, list[Tool]] = {}
        for tool in self.list_tools():
            grouped.setdefault(tool.category, []).append(tool)
        return grouped

    def get_tool(self, name: str) -> Tool:
        """Resolve a tool name, or raise UnknownToolError."""
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(
                f"No tool named {name!r}. Available tools: "
                f"{', '.join(self.names()) or 'none'}."
            )
        return tool

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Tool definitions in the format llama.cpp expects."""
        return [tool.definition() for tool in self.list_tools()]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a tool call. Never raises for model-caused problems.

        Unknown names, bad arguments and tool failures all come back as a
        ToolResult with success=False, so the model can read the error and
        try again.
        """
        try:
            tool = self.get_tool(name)
            validated = _validate_arguments(tool, arguments)
            payload = tool.run(**validated)
        except ToolError as exc:
            return ToolResult(name=name, payload=failure(str(exc)))
        except Exception as exc:  # a bug in the tool itself
            return ToolResult(
                name=name, payload=failure(f"{type(exc).__name__}: {exc}")
            )

        if not isinstance(payload, dict):
            payload = {"success": True, "result": payload}
        payload.setdefault("success", True)
        return ToolResult(name=name, payload=payload)


def _validate_arguments(tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
    """Check the model's arguments against the tool's schema.

    Deliberately minimal: required keys, unknown keys, and top-level types.
    Enough to keep bad calls out of the tool functions without pulling in a
    JSON Schema dependency.
    """
    if not isinstance(arguments, dict):
        raise InvalidArgumentsError("Arguments must be a JSON object.")

    properties = tool.parameters.get("properties", {})
    required = tool.parameters.get("required", [])

    # Report missing and unknown keys together: a misspelled argument produces
    # both at once, and the model needs to see both to fix it in one go.
    problems: list[str] = []
    missing = [key for key in required if key not in arguments]
    if missing:
        problems.append(f"Missing required argument(s): {', '.join(missing)}.")

    unknown = [key for key in arguments if key not in properties]
    if unknown:
        problems.append(f"Unknown argument(s): {', '.join(unknown)}.")

    if problems:
        problems.append(f"Expected: {', '.join(properties) or 'none'}.")
        raise InvalidArgumentsError(" ".join(problems))

    for key, value in arguments.items():
        expected = properties[key].get("type")
        allowed = _JSON_TYPES.get(expected) if isinstance(expected, str) else None
        if allowed is None:
            continue
        # bool is a subclass of int; don't let True through as a number.
        if isinstance(value, bool) and expected in ("number", "integer"):
            raise InvalidArgumentsError(f"Argument {key!r} must be a {expected}.")
        if not isinstance(value, allowed):
            raise InvalidArgumentsError(
                f"Argument {key!r} must be a {expected}, got "
                f"{type(value).__name__}."
            )

    return dict(arguments)
