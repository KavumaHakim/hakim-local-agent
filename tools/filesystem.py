"""Filesystem tools, confined to a configured workspace.

Reading and listing are always available. Writing is off unless enabled, and
even then only creating files and directories exists - there is no delete,
rename, move or chmod path anywhere in this module.

Every model-supplied path is resolved first and then checked against the
workspace root, which is what makes the check hold against '..' segments,
absolute paths and symlinks that point outside.

One extra guard applies to writing. The workspace defaults to the project
directory, so without it the agent could rewrite tools/base.py - the very code
that enforces every other tool's limits - or its own registry, and silently
disable the lot. PROTECTED_TOP_LEVEL keeps the agent's own source out of reach
even when the workspace contains it. A workspace pointed somewhere else
entirely is still the safer arrangement.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolError

# The agent's own code and state, relative to the project root. Writing to any
# of these could disable the guards that make the other tools safe.
PROTECTED_TOP_LEVEL = frozenset(
    {
        "agent", "models", "tools", "tests", ".git", ".venv",
        # The HTTP layer and the front end it serves. `api` matters as much as
        # the rest: it builds the registry the model is offered, so writing
        # there would reach every other guard. `web` is what the browser runs.
        "api", "web", ".claude",
        "main.py", "config.py", "chat_store.py", "memory_store.py",
        "models.json", "requirements.txt", "README.md",
    }
)

# Refused wherever they appear, whatever the workspace is.
ALWAYS_PROTECTED_PARTS = frozenset({".git", ".venv"})

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FilesystemToolError(ToolError):
    """A filesystem operation was rejected or failed."""


class WorkspaceFiles:
    """Filesystem access limited to a single root directory."""

    def __init__(
        self,
        root: Path,
        *,
        max_read_bytes: int = 200_000,
        max_write_bytes: int = 200_000,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._max_read_bytes = max_read_bytes
        self._max_write_bytes = max_write_bytes

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, path: str) -> Path:
        """Resolve a model-supplied path, confined to the workspace.

        Public because other tools (OCR) must reuse exactly this check rather
        than rolling their own.
        """
        return self._resolve(path)

    def _resolve(self, path: str) -> Path:
        if not isinstance(path, str) or not path.strip():
            raise FilesystemToolError("Path must be a non-empty string.")

        candidate = Path(path.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = self._root / candidate

        try:
            # resolve() collapses '..' and follows symlinks, so the containment
            # check below cannot be fooled by either.
            resolved = candidate.resolve()
        except OSError as exc:
            raise FilesystemToolError(f"Invalid path: {exc}") from None

        if resolved != self._root and self._root not in resolved.parents:
            raise FilesystemToolError(
                f"Access denied: path is outside the workspace ({self._root})."
            )
        return resolved

    def _display(self, path: Path) -> str:
        if path == self._root:
            return "."
        return str(path.relative_to(self._root)).replace("\\", "/")

    def list_directory(self, path: str = ".") -> dict[str, Any]:
        target = self._resolve(path)
        if not target.exists():
            raise FilesystemToolError(f"Path does not exist: {self._display(target)}")
        if not target.is_dir():
            raise FilesystemToolError(f"Not a directory: {self._display(target)}")

        files: list[dict[str, Any]] = []
        try:
            for child in target.iterdir():
                try:
                    is_dir = child.is_dir()
                    entry: dict[str, Any] = {"name": child.name, "type": "dir" if is_dir else "file"}
                    if not is_dir:
                        entry["size_bytes"] = child.stat().st_size
                except OSError:
                    # Broken link or a permission problem on one entry: skip it
                    # rather than failing the whole listing.
                    continue
                files.append(entry)
        except PermissionError:
            raise FilesystemToolError(
                f"Permission denied: {self._display(target)}"
            ) from None

        files.sort(key=lambda e: (e["type"] != "dir", e["name"].lower()))
        return {"success": True, "path": self._display(target), "files": files}

    def read_text_file(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.exists():
            raise FilesystemToolError(f"File does not exist: {self._display(target)}")
        if target.is_dir():
            raise FilesystemToolError(
                f"{self._display(target)} is a directory, not a file."
            )

        try:
            size = target.stat().st_size
        except OSError as exc:
            raise FilesystemToolError(f"Could not stat file: {exc}") from None

        if size > self._max_read_bytes:
            raise FilesystemToolError(
                f"File is {size} bytes, over the {self._max_read_bytes} byte "
                f"limit. Read a smaller file."
            )

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except PermissionError:
            raise FilesystemToolError(
                f"Permission denied: {self._display(target)}"
            ) from None
        except OSError as exc:
            raise FilesystemToolError(f"Could not read file: {exc}") from None

        return {
            "success": True,
            "path": self._display(target),
            "size_bytes": size,
            "content": content,
        }

    # --- writing ---

    def _check_writable(self, target: Path) -> None:
        """Refuse writes to the agent's own code and state.

        The containment check has already run; this is the separate question
        of whether a path *inside* the workspace should be written to at all.
        """
        if any(part in ALWAYS_PROTECTED_PARTS for part in target.parts):
            raise FilesystemToolError(
                f"Refusing to write inside {'/'.join(ALWAYS_PROTECTED_PARTS)}."
            )

        try:
            relative = target.relative_to(PROJECT_ROOT)
        except ValueError:
            return  # outside the agent's own directory, so not its source

        top = relative.parts[0] if relative.parts else ""
        if top in PROTECTED_TOP_LEVEL:
            raise FilesystemToolError(
                f"{top!r} is part of the agent's own source and is not "
                f"writable. Write somewhere else, or point AGENT_WORKSPACE at "
                f"a directory that is not the project."
            )

    def write_text_file(
        self, path: str, content: str, overwrite: bool = False
    ) -> dict[str, Any]:
        """Create a text file, or replace one when overwrite is set."""
        if not isinstance(content, str):
            raise FilesystemToolError("content must be a string.")
        encoded = content.encode("utf-8")
        if len(encoded) > self._max_write_bytes:
            raise FilesystemToolError(
                f"Content is {len(encoded)} bytes, over the "
                f"{self._max_write_bytes} byte limit."
            )

        target = self._resolve(path)
        self._check_writable(target)

        if target.is_dir():
            raise FilesystemToolError(
                f"{self._display(target)} is a directory, not a file."
            )

        existed = target.exists()
        if existed and not overwrite:
            # Clobbering by accident is the main risk here, so replacing an
            # existing file has to be asked for.
            raise FilesystemToolError(
                f"{self._display(target)} already exists. Pass overwrite=true "
                f"to replace it."
            )

        if not target.parent.exists():
            raise FilesystemToolError(
                f"The directory {self._display(target.parent)} does not exist. "
                f"Create it first."
            )

        # Write beside the target and rename over it, so a failure part-way
        # cannot leave a half-written file where a good one used to be.
        temporary = target.with_name(target.name + ".agent-tmp")
        try:
            temporary.write_bytes(encoded)
            os.replace(temporary, target)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise FilesystemToolError(f"Could not write the file: {exc}") from None

        return {
            "success": True,
            "path": self._display(target),
            "bytes_written": len(encoded),
            "replaced": existed,
        }

    def create_directory(self, path: str) -> dict[str, Any]:
        """Create a directory, including any missing parents."""
        target = self._resolve(path)
        self._check_writable(target)

        if target.is_file():
            raise FilesystemToolError(
                f"{self._display(target)} is a file, not a directory."
            )
        existed = target.is_dir()
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise FilesystemToolError(
                f"Could not create the directory: {exc}"
            ) from None

        return {
            "success": True,
            "path": self._display(target),
            "created": not existed,
        }

    def write_tools(self) -> list[Tool]:
        """The write tools. Registered only when writing is enabled."""
        return [
            Tool(
                name="write_text_file",
                category="filesystem",
                description=(
                    "Create a text file in the workspace, or replace one by "
                    "passing overwrite=true. Paths are relative to the "
                    "workspace root. The parent directory must already exist. "
                    "The agent's own source files cannot be written."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File to write, relative to the workspace root.",
                        },
                        "content": {
                            "type": "string",
                            "description": "The full text to write.",
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": "Replace the file if it already exists.",
                        },
                    },
                    "required": ["path", "content"],
                },
                run=self.write_text_file,
            ),
            Tool(
                name="create_directory",
                category="filesystem",
                description=(
                    "Create a directory inside the workspace, including any "
                    "missing parent directories."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory to create, relative to the workspace root.",
                        }
                    },
                    "required": ["path"],
                },
                run=self.create_directory,
            ),
        ]

    def tools(self) -> list[Tool]:
        """Build the Tool objects bound to this workspace."""
        return [
            Tool(
                name="list_directory",
                category="filesystem",
                description=(
                    "List files and sub-directories inside the workspace. "
                    "Paths are relative to the workspace root; use '.' for the "
                    "root. Paths outside the workspace are refused."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory to list. Defaults to '.'.",
                        }
                    },
                    "required": [],
                },
                run=self.list_directory,
            ),
            Tool(
                name="read_text_file",
                category="filesystem",
                description=(
                    "Read a text file inside the workspace. Paths are relative "
                    "to the workspace root. Paths outside the workspace are "
                    "refused and oversized files are rejected."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File to read, relative to the workspace root.",
                        }
                    },
                    "required": ["path"],
                },
                run=self.read_text_file,
            ),
        ]


# Deletion, rename and move stay out of the model's reach deliberately. They
# are the operations with no undo, and nothing so far has needed them.
