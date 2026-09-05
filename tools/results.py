"""Tool results too large for the context, kept where the model can reach them.

A tool result goes straight into the model's window, and some of them have no
natural size: a file read, a page of OCR, a directory of ten thousand entries.
On a 4,096-token model that window is about 9,800 characters once the system
prompt and the tool schemas are paid for, so a result is capped at a quarter of
it - roughly 3,300 characters.

Until now the rest was simply gone. The cut was announced, which is better than
silence, but "12,000 of 12,400 lines are not shown" is a dead end: the model
cannot ask for the part it needs, so it either answers from the first page or
gives up, and neither is what the file said.

This keeps the whole thing on disk and hands back a handle. The model sees the
first page as before, plus an id, and `read_result` fetches any window of the
rest. Nothing is lost; it is just not all in the context at once, which is the
distinction the window actually requires.

**Not in the workspace.** The obvious place is a folder beside the files the
agent is reading, and `read_text_file` would then need no companion. But the
workspace belongs to the person, writing there is a side effect they did not
ask for, and it would happen even with file writes switched off - which is
exactly the setting that says "do not put things in my folders". So these live
under the application's own data directory, and reaching them is the one thing
`read_result` does.
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolError

# How much of a stored result one `read_result` call may return. The same
# order as a tool result's own cap: this exists to page through something too
# big for the window, so a page has to fit in the window.
DEFAULT_WINDOW = 3_000
MAX_WINDOW = 20_000

# Ids are short because the model has to type one back, and a UUID is 36
# characters of nothing. Random rather than sequential so two conversations
# writing at once cannot collide on a counter.
_ID = re.compile(r"^r[0-9a-f]{6}$")

# Disk is not free, and this machine has run out of it. Oldest go first.
MAX_STORED_FILES = 200
MAX_STORED_BYTES = 64 * 1024 * 1024


class ResultStoreError(ToolError):
    """The stored result could not be read."""


@dataclass(frozen=True)
class Stored:
    """What was kept, and what the model is told about it."""

    id: str
    characters: int
    lines: int


class ResultStore:
    """Full tool results on disk, addressed by a short id."""

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)

    def _path(self, result_id: str) -> Path:
        # Validated rather than sanitised: the id is generated here, so
        # anything not matching the shape was invented by the model, and the
        # right answer to an invented id is to say so - not to go looking for
        # whatever `../../.ssh/config` resolves to.
        if not _ID.match(result_id or ""):
            raise ResultStoreError(
                f"{result_id!r} is not a result id. They look like 'r3f9a1' "
                f"and are given to you when a result is too large to show."
            )
        return self._dir / f"{result_id}.json"

    def save(self, text: str) -> Stored | None:
        """Keep `text`, returning its handle. None if it could not be kept.

        Never raises. Offloading is an improvement on truncation, and a disk
        that will not take it leaves the caller exactly where it was before -
        with a cut result and a note saying so.
        """
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._prune()
            result_id = f"r{secrets.token_hex(3)}"
            (self._dir / f"{result_id}.json").write_text(text, encoding="utf-8")
            return Stored(
                id=result_id,
                characters=len(text),
                lines=text.count("\n") + 1,
            )
        except OSError:
            return None

    def read(
        self, result_id: str, offset: int = 0, limit: int = DEFAULT_WINDOW
    ) -> dict[str, Any]:
        """A window onto a stored result."""
        path = self._path(result_id)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            raise ResultStoreError(
                f"There is no stored result {result_id!r}. They do not survive "
                f"a restart, and the oldest are removed as new ones arrive."
            ) from None
        except OSError as exc:
            raise ResultStoreError(f"Could not read {result_id}: {exc}") from None

        total = len(text)
        start = max(0, int(offset or 0))
        window = max(1, min(int(limit or DEFAULT_WINDOW), MAX_WINDOW))
        chunk = text[start : start + window]
        end = start + len(chunk)

        return {
            "success": True,
            "id": result_id,
            "offset": start,
            "returned": len(chunk),
            "total_characters": total,
            # Said plainly, because the model has to decide whether to ask
            # again and "there is more" is the only fact that decides it.
            "more": end < total,
            "next_offset": end if end < total else None,
            "text": chunk,
        }

    def _prune(self) -> None:
        """Keep the store small. Oldest first, by count and by bytes."""
        try:
            files = sorted(
                self._dir.glob("r*.json"), key=lambda p: p.stat().st_mtime
            )
        except OSError:
            return

        total = 0
        for path in reversed(files):
            try:
                total += path.stat().st_size
            except OSError:
                continue

        while files and (len(files) >= MAX_STORED_FILES or total > MAX_STORED_BYTES):
            oldest = files.pop(0)
            try:
                total -= oldest.stat().st_size
                oldest.unlink()
            except OSError:
                continue


def build_result_tool(store: ResultStore) -> Tool:
    # The schema says `id` because that is what reads naturally to a model,
    # and the store says `result_id` because `id` shadows a builtin. The
    # registry calls run(**arguments), so something has to bridge the two.
    def read(id: str, offset: int = 0, limit: int = DEFAULT_WINDOW):  # noqa: A002
        return store.read(id, offset=offset, limit=limit)

    return Tool(
        name="read_result",
        category="results",
        description=(
            "Read part of a tool result that was too large to show in full. "
            "You are given an id when that happens. Ask for the part you need "
            "with offset and limit rather than reading the whole thing: it was "
            "set aside precisely because it does not fit."
        ),
        parameters={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The id from the truncated result, e.g. r3f9a1.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Character to start from. 0 is the beginning.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"How many characters to return. Default {DEFAULT_WINDOW}, "
                        f"maximum {MAX_WINDOW}."
                    ),
                },
            },
            "required": ["id"],
        },
        run=read,
    )
