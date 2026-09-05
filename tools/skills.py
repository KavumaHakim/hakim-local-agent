"""Instruction packs the model can ask for, one at a time.

A skill is a markdown file with a name and a description: how to do something
in this project, written once, kept out of the way until it is wanted. The
document search tool answers "what does my file say"; a skill answers "how do
we do this here", which is not a thing to retrieve from a corpus.

**Delivered as a tool result, never injected into the prompt.** That is the
whole design, and it is a consequence of the hardware rather than taste. The
obvious implementation is to detect a relevant skill and prepend it to the
system prompt - but the prompt prefix is what llama.cpp's cache is keyed on,
and changing it re-reads the entire conversation. Measured on this machine
that is around 200 seconds. A tool call appends to the *end* instead, which
the cache does not mind, and costs one extra round the model chose to spend.
The same reasoning gave the tool lens its monotonic rule; see tools/lens.py.

So the shape is exactly `load_tools`: an index of what exists in the tool's
description, and the body only when asked for. A model that never needs one
pays for the index and nothing else, and a project with no skills at all pays
for nothing, because the tool is not registered.

A skill may also name the tool groups its instructions need, and loading it
opens them - see `tools:` below. Instructions that say "plot this with
matplotlib" are no use to a model that cannot see the python tool, and making
it spend a second round trip on `load_tools` to discover that is a round trip
the skill already knew about. The prefix-cache miss is the same one the lens
would have paid anyway, moved earlier.

No YAML parser. The frontmatter is three keys, the project has no YAML
dependency, and adding one to read `name:` would be the wrong trade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tools.base import Tool, ToolError

LOAD_SKILL = "load_skill"

# The key `load_skill` uses to hand its declared tool groups to the loop.
# Private between the two of them: the loop always removes it, so the model
# never sees it.
NEEDS_TOOLS = "_needs_tools"

# A skill body reaches the model whole, so it has to fit beside everything
# else in the window. Long enough for real instructions, short enough that
# two of them do not fill a 4,096-token context on their own.
MAX_SKILL_CHARS = 6_000

# Names are used as identifiers and shown in the index, so they are kept to
# something a model can repeat back without quoting.
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,48}$")

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

# A tool group named in `tools:`. Same shape as a category in the registry.
_CATEGORY = re.compile(r"^[a-z][a-z0-9_:-]{0,32}$")


def _parse_tools(value: str) -> tuple[str, ...]:
    """The tool groups named in one frontmatter line.

    `tools: python, filesystem` and `tools: [python, filesystem]` both mean
    the same thing, because both are what someone writes. Whitespace
    separates too, so a stray missing comma is not a silent failure.

    Names are *not* checked against the registry here: this runs when the
    library is built, and which categories exist depends on which optional
    tools are switched on. An unknown or switched-off name simply opens
    nothing, and the model is told what actually opened rather than what was
    asked for.
    """
    seen: list[str] = []
    for piece in re.split(r"[,\s]+", value.strip().strip("[]")):
        name = piece.strip().strip("\"'").lower()
        if _CATEGORY.match(name) and name not in seen:
            seen.append(name)
    return tuple(seen)


class SkillError(ToolError):
    """The skill could not be found or read."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path
    # Tool groups the instructions assume. Opened when the skill is loaded.
    tools: tuple[str, ...] = ()

    @property
    def truncated(self) -> bool:
        return len(self.body) >= MAX_SKILL_CHARS


def parse_skill(text: str, fallback_name: str) -> tuple[str, str, str, tuple[str, ...]]:
    """Split a SKILL.md into name, description, body and tool groups.

    Frontmatter is optional. Without it the folder name is the name and the
    first non-empty line is the description, so a plain markdown file dropped
    into the folder still works - which is what someone will do first.
    """
    name, description, tools = fallback_name, "", ()
    body = text

    match = _FRONTMATTER.match(text)
    if match:
        body = text[match.end():]
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip().strip("\"'")
            if key == "name" and value:
                name = value
            elif key == "description" and value:
                description = value
            elif key == "tools" and value:
                tools = _parse_tools(value)

    body = body.strip()
    if not description:
        for line in body.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                description = stripped
                break

    return name.strip().lower(), description, body, tools


def discover(directory: Path) -> list[Skill]:
    """Every skill under `directory`, by name.

    Two layouts, because both are things people do: a folder per skill with a
    SKILL.md inside it, which is what makes a skill a git-shaped unit that can
    carry other files later, and a bare `<name>.md` for someone who just wants
    to write one down.
    """
    found: dict[str, Skill] = {}
    if not directory.is_dir():
        return []

    candidates: list[tuple[Path, str]] = []
    try:
        for entry in sorted(directory.iterdir()):
            if entry.is_dir():
                skill_file = entry / "SKILL.md"
                if skill_file.is_file():
                    candidates.append((skill_file, entry.name))
            elif entry.suffix.lower() == ".md" and entry.name.lower() != "readme.md":
                candidates.append((entry, entry.stem))
    except OSError:
        return []

    for path, fallback in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        name, description, body, tools = parse_skill(text, fallback)
        if not _NAME.match(name) or not body:
            # A nameless or empty file is a note somebody left, not a skill.
            continue
        found.setdefault(
            name,
            Skill(
                name=name,
                description=description[:200],
                body=body[:MAX_SKILL_CHARS],
                path=path,
                tools=tools,
            ),
        )

    return [found[name] for name in sorted(found)]


class SkillLibrary:
    """The skills on disk, read once when the registry is built."""

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._skills = {skill.name: skill for skill in discover(self._dir)}

    def __len__(self) -> int:
        return len(self._skills)

    @property
    def names(self) -> list[str]:
        return sorted(self._skills)

    def load(self, name: str) -> dict[str, object]:
        """The body of one skill, as a tool result."""
        key = (name or "").strip().lower()
        skill = self._skills.get(key)
        if skill is None:
            raise SkillError(
                f"There is no skill called {name!r}. Available: "
                f"{', '.join(self.names) or 'none'}."
            )
        return {
            "success": True,
            "skill": skill.name,
            "instructions": skill.body,
            # Said rather than left implicit: a model that has just been handed
            # instructions should act on them, not summarise them back.
            "note": (
                "Follow these for the rest of this conversation. Do not repeat "
                "them to the user unless asked."
            ),
            # Consumed by the agent loop, which opens these in the lens and
            # replaces the key with what actually opened - see
            # `Agent._open_skill_tools`. It never reaches the model under this
            # name, because a list of groups the model cannot see yet is worse
            # than no list at all.
            **({NEEDS_TOOLS: list(skill.tools)} if skill.tools else {}),
            **({"truncated": True} if skill.truncated else {}),
        }

    def tool(self) -> Tool:
        listing = "\n".join(
            f"- {skill.name}: {skill.description}"
            for skill in (self._skills[name] for name in self.names)
        )
        return Tool(
            name=LOAD_SKILL,
            category="skills",
            description=(
                "Load written instructions for a particular kind of task. Call "
                "this when one of the below matches what you have been asked "
                "to do, before starting it.\n" + listing
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": self.names,
                        "description": "Which one to load.",
                    }
                },
                "required": ["name"],
            },
            run=lambda name: self.load(name),
        )
