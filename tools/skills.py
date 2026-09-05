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

No YAML parser. The frontmatter is two keys, the project has no YAML
dependency, and adding one to read `name:` would be the wrong trade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tools.base import Tool, ToolError

LOAD_SKILL = "load_skill"

# A skill body reaches the model whole, so it has to fit beside everything
# else in the window. Long enough for real instructions, short enough that
# two of them do not fill a 4,096-token context on their own.
MAX_SKILL_CHARS = 6_000

# Names are used as identifiers and shown in the index, so they are kept to
# something a model can repeat back without quoting.
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,48}$")

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class SkillError(ToolError):
    """The skill could not be found or read."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path

    @property
    def truncated(self) -> bool:
        return len(self.body) >= MAX_SKILL_CHARS


def parse_skill(text: str, fallback_name: str) -> tuple[str, str, str]:
    """Split a SKILL.md into name, description and body.

    Frontmatter is optional. Without it the folder name is the name and the
    first non-empty line is the description, so a plain markdown file dropped
    into the folder still works - which is what someone will do first.
    """
    name, description = fallback_name, ""
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

    body = body.strip()
    if not description:
        for line in body.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                description = stripped
                break

    return name.strip().lower(), description, body


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
        name, description, body = parse_skill(text, fallback)
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
