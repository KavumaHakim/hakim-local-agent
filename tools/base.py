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


# Characters set aside for the truncation note, so adding it cannot push a
# trimmed result back over the limit.
_NOTE_RESERVE = 260


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
        return self.content_within(0)

    def content_within(self, limit: int) -> str:
        """The same JSON, cut to `limit` characters. 0 means no limit.

        Needed because a tool result goes straight into the model's context and
        some of them are unbounded: a page of OCR, or a file read. On a model
        with a 4,096-token window - about 9,800 characters once the system
        prompt and the tool schemas are paid for - one dense page overflows it
        and the whole turn fails.

        The cut is made inside the payload's longest string rather than on the
        serialised text, so the result stays valid JSON. And it is *announced*:
        a truncated field is replaced with a marker saying how much is missing,
        because a model handed the first half of a page with no indication
        will summarise it as though it were the whole thing.
        """
        payload = self.payload
        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return json.dumps({"success": False, "error": "Unserialisable result"})

        if limit <= 0 or len(text) <= limit:
            return text

        payload = dict(payload)
        # Longest first: cutting the biggest field usually gets under the
        # limit in one step and leaves the small metadata fields intact.
        fields = sorted(
            (key for key, value in payload.items() if isinstance(value, str)),
            key=lambda key: len(payload[key]),
            reverse=True,
        )

        for key in fields:
            if len(text) <= limit:
                break
            original = payload[key]

            # How many characters everything *except* this field costs, with
            # room set aside for the note that is about to be added. Measuring
            # it rather than subtracting an estimate is what stops the note
            # pushing the result back over the limit - which, when the first
            # attempt at this got it wrong, threw the text away entirely.
            probe = dict(payload)
            probe[key] = ""
            probe.pop("truncated", None)
            try:
                fixed = len(json.dumps(probe, ensure_ascii=False, default=str))
            except (TypeError, ValueError):
                break

            keep = max(0, min(len(original), limit - fixed - _NOTE_RESERVE))
            if keep >= len(original):
                continue

            payload[key] = original[:keep]
            payload["truncated"] = (
                f"{key!r} was cut: {len(original) - keep:,} of "
                f"{len(original):,} characters are not shown, because the "
                f"whole result does not fit this model's context. Say so "
                f"rather than treating this as complete."
            )
            try:
                text = json.dumps(payload, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                break

        # Lists next, because not every big result is a big string. A
        # directory listing is hundreds of small dicts under one key, and
        # cutting strings does nothing to it - which is how `list_directory`
        # used to walk straight past this limit and overflow the window.
        lists = sorted(
            (key for key, value in payload.items() if isinstance(value, list)),
            key=lambda key: len(payload[key]),
            reverse=True,
        )

        for key in lists:
            if len(text) <= limit:
                break
            original = payload[key]
            if not original:
                continue

            probe = dict(payload)
            probe[key] = []
            probe.pop("truncated", None)
            try:
                fixed = len(json.dumps(probe, ensure_ascii=False, default=str))
            except (TypeError, ValueError):
                break

            room = limit - fixed - _NOTE_RESERVE
            keep = _entries_that_fit(original, room)
            if keep >= len(original):
                continue

            payload[key] = original[:keep]
            payload["truncated"] = (
                f"{key!r} was cut: {len(original) - keep:,} of "
                f"{len(original):,} entries are not shown, because the whole "
                f"result does not fit this model's context. Say so rather "
                f"than treating this as complete."
            )
            try:
                text = json.dumps(payload, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                break

        # The backstop. Everything above cuts a *field*, which cannot help a
        # payload whose bulk is somewhere those passes do not reach - a nested
        # structure, or thousands of numbers. Returning an oversized result
        # here is what overflows the model's window, so this never does: it
        # gives up on the content and says so, in valid JSON.
        if len(text) > limit:
            return json.dumps(
                {
                    "success": bool(self.payload.get("success", self.ok)),
                    "truncated": (
                        f"The result was {len(text):,} characters and could "
                        f"not be cut to the {limit:,} this model's context "
                        f"allows, so none of it is shown. Ask for less of it "
                        f"- a narrower path, or one page rather than a whole "
                        f"document."
                    ),
                },
                ensure_ascii=False,
            )

        return text

    @staticmethod
    def _fits(entries: list, room: int) -> bool:
        try:
            return len(json.dumps(entries, ensure_ascii=False, default=str)) <= room
        except (TypeError, ValueError):
            return False

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


def _entries_that_fit(entries: list, room: int) -> int:
    """How many leading entries of `entries` serialise within `room`.

    A binary search rather than a loop adding one at a time: a listing of a
    node_modules directory is tens of thousands of entries, and measuring the
    cost of each prefix in turn is quadratic in the thing already known to be
    too big.
    """
    if room <= 2:  # not even "[]"
        return 0

    low, high = 0, len(entries)
    while low < high:
        middle = (low + high + 1) // 2
        try:
            cost = len(json.dumps(entries[:middle], ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            return low
        if cost <= room:
            low = middle
        else:
            high = middle - 1
    return low


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
