"""Git inspection, and optionally committing, inside the workspace.

Off by default (config.git_tool_enabled, env AGENT_ENABLE_GIT_TOOL=1).

This overlaps the terminal tool on purpose. `run_command` can already run
read-only git verbs, but it returns raw text the model has to parse, and it
requires enabling terminal access to everything else on that allowlist. This
returns structured results - a list of changed paths, a list of commit objects -
and can be enabled on its own.

The boundary:

* git runs with `shell=False` and a scrubbed environment, exactly as the
  terminal tool does, so nothing is interpreted and no credentials of yours
  ride along.
* Reading is always available. **Writing is a separate opt-in** and covers only
  `commit` and creating a branch.
* Nothing here can reach a remote. No push, fetch or pull exists, so the agent
  cannot publish anything or pull code in.
* Nothing here can destroy uncommitted work. No reset, checkout, clean, stash,
  rebase or merge. A commit can be undone; a `reset --hard` cannot, and
  `checkout` silently discards changes, which is why both are absent rather
  than merely discouraged.

`GIT_TERMINAL_PROMPT=0` is set so a repository that wants credentials fails
instead of hanging forever on a prompt nobody can answer.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolError
from tools.shell_tool import _child_environment

MAX_OUTPUT_CHARS = 6000
MAX_MESSAGE_LENGTH = 500
# ASCII unit separator: safe inside a commit field, unlike commas or tabs.
FIELD = "\x1f"


class GitToolError(ToolError):
    """A git operation was refused or failed."""


class GitRepository:
    """Runs git against the workspace."""

    def __init__(
        self,
        workspace: Path,
        *,
        timeout: float = 30.0,
        allow_writes: bool = False,
    ) -> None:
        self._root = Path(workspace).expanduser().resolve()
        self._timeout = timeout
        self._allow_writes = allow_writes

    # --- plumbing ---

    def _git(self, *arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=_child_environment(),
                shell=False,
            )
        except FileNotFoundError:
            raise GitToolError("git is not installed, or not on PATH.") from None
        except subprocess.TimeoutExpired:
            raise GitToolError(
                f"git exceeded the {self._timeout:g}s timeout."
            ) from None
        except OSError as exc:
            raise GitToolError(f"Could not run git: {exc}") from None

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            if "not a git repository" in detail.lower():
                raise GitToolError(
                    f"{self._root} is not a git repository."
                )
            raise GitToolError(f"git failed: {detail[:400]}")
        return completed.stdout or ""

    def is_repository(self) -> bool:
        try:
            self._git("rev-parse", "--git-dir")
        except GitToolError:
            return False
        return True

    def _require_writes(self, what: str) -> None:
        if not self._allow_writes:
            raise GitToolError(
                f"{what} changes the repository and is off. Set "
                f"AGENT_GIT_ALLOW_WRITES=1 to permit it."
            )

    # --- reading ---

    def status(self) -> dict[str, Any]:
        """Branch, and the working tree as a list of changed paths."""
        branch = self._git("rev-parse", "--abbrev-ref", "HEAD").strip()
        raw = self._git("status", "--porcelain=v1")

        changes: list[dict[str, str]] = []
        for line in raw.splitlines():
            if len(line) < 4:
                continue
            index, worktree, path = line[0], line[1], line[3:]
            changes.append(
                {
                    "path": path,
                    "staged": _describe(index),
                    "unstaged": _describe(worktree),
                }
            )

        return {
            "success": True,
            "branch": branch,
            "clean": not changes,
            "changes": changes,
            "change_count": len(changes),
        }

    def log(self, count: int = 10) -> dict[str, Any]:
        count = max(1, min(int(count), 100))
        raw = self._git(
            "log",
            f"-{count}",
            f"--format=%h{FIELD}%an{FIELD}%ad{FIELD}%s",
            "--date=short",
        )
        commits = []
        for line in raw.splitlines():
            parts = line.split(FIELD)
            if len(parts) == 4:
                commits.append(
                    {
                        "hash": parts[0],
                        "author": parts[1],
                        "date": parts[2],
                        "subject": parts[3],
                    }
                )
        return {"success": True, "commits": commits, "count": len(commits)}

    def diff(self, path: str | None = None, staged: bool = False) -> dict[str, Any]:
        arguments = ["diff"]
        if staged:
            arguments.append("--staged")
        if path:
            _check_pathspec(path)
            arguments += ["--", path]

        patch = self._git(*arguments)
        summary = self._git(*(arguments[:-2] if path else arguments), "--stat") \
            if not path else self._git(*arguments, "--stat")
        truncated = len(patch) > MAX_OUTPUT_CHARS
        return {
            "success": True,
            "staged": staged,
            "summary": summary.strip()[:MAX_OUTPUT_CHARS],
            "patch": patch[:MAX_OUTPUT_CHARS],
            "truncated": truncated,
        }

    def branches(self) -> dict[str, Any]:
        current = self._git("rev-parse", "--abbrev-ref", "HEAD").strip()
        raw = self._git("branch", "--format=%(refname:short)")
        names = [line.strip() for line in raw.splitlines() if line.strip()]
        return {"success": True, "current": current, "branches": names}

    # --- writing ---

    def commit(self, message: str, paths: list[str] | None = None) -> dict[str, Any]:
        self._require_writes("commit")

        if not isinstance(message, str) or not message.strip():
            raise GitToolError("A commit message is required.")
        if len(message) > MAX_MESSAGE_LENGTH:
            raise GitToolError(
                f"Commit message is too long (limit {MAX_MESSAGE_LENGTH})."
            )

        if paths:
            if not isinstance(paths, list):
                raise GitToolError("paths must be a list of strings.")
            for path in paths:
                _check_pathspec(path)
            self._git("add", "--", *paths)
        else:
            # Everything already tracked and modified. Deliberately not `-A`:
            # that would sweep in untracked files nobody asked to commit.
            self._git("add", "--update")

        staged = self._git("diff", "--staged", "--name-only").strip()
        if not staged:
            raise GitToolError(
                "Nothing is staged, so there is nothing to commit."
            )

        self._git("commit", "-m", message.strip())
        head = self._git("rev-parse", "--short", "HEAD").strip()
        return {
            "success": True,
            "commit": head,
            "message": message.strip(),
            "files": staged.splitlines(),
        }

    def create_branch(self, name: str) -> dict[str, Any]:
        self._require_writes("creating a branch")
        if not isinstance(name, str) or not name.strip():
            raise GitToolError("A branch name is required.")
        clean = name.strip()
        if clean.startswith("-") or any(c.isspace() for c in clean):
            raise GitToolError(f"{clean!r} is not a usable branch name.")

        # Creates without switching, so nothing in the working tree moves.
        self._git("branch", clean)
        return {"success": True, "branch": clean, "switched": False}

    # --- tools ---

    def tools(self) -> list[Tool]:
        found: list[Tool] = [
            Tool(
                name="git_status",
                category="git",
                description=(
                    "Current branch and the list of changed files in the "
                    "workspace repository."
                ),
                parameters={"type": "object", "properties": {}, "required": []},
                run=self.status,
            ),
            Tool(
                name="git_log",
                category="git",
                description="Recent commits: hash, author, date and subject.",
                parameters={
                    "type": "object",
                    "properties": {
                        "count": {
                            "type": "integer",
                            "description": "How many commits to return (default 10, max 100).",
                        }
                    },
                    "required": [],
                },
                run=self.log,
            ),
            Tool(
                name="git_diff",
                category="git",
                description=(
                    "Changes in the working tree, as a summary and a patch. "
                    "Pass staged=true for what is staged instead."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Limit the diff to one path.",
                        },
                        "staged": {
                            "type": "boolean",
                            "description": "Show staged changes instead of unstaged.",
                        },
                    },
                    "required": [],
                },
                run=self.diff,
            ),
            Tool(
                name="git_branches",
                category="git",
                description="List local branches and which one is checked out.",
                parameters={"type": "object", "properties": {}, "required": []},
                run=self.branches,
            ),
        ]

        if self._allow_writes:
            found += [
                Tool(
                    name="git_commit",
                    category="git",
                    description=(
                        "Commit changes in the workspace repository. Stages the "
                        "given paths, or all tracked modified files if none are "
                        "given. Cannot push."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The commit message.",
                            },
                            "paths": {
                                "type": "array",
                                "description": "Specific paths to stage.",
                            },
                        },
                        "required": ["message"],
                    },
                    run=self.commit,
                ),
                Tool(
                    name="git_create_branch",
                    category="git",
                    description=(
                        "Create a branch without switching to it, so nothing "
                        "in the working tree changes."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The new branch name.",
                            }
                        },
                        "required": ["name"],
                    },
                    run=self.create_branch,
                ),
            ]
        return found


def _describe(code: str) -> str:
    return {
        " ": "unchanged",
        "M": "modified",
        "A": "added",
        "D": "deleted",
        "R": "renamed",
        "C": "copied",
        "U": "conflicted",
        "?": "untracked",
        "!": "ignored",
    }.get(code, code)


def _check_pathspec(path: str) -> None:
    """Keep a pathspec from reaching outside the repository."""
    if not isinstance(path, str) or not path.strip():
        raise GitToolError("Path must be a non-empty string.")
    candidate = path.strip().replace("\\", "/")
    if candidate.startswith("-"):
        raise GitToolError(f"{path!r} looks like an option, not a path.")
    if candidate.startswith("/") or (len(path) > 1 and path[1] == ":"):
        raise GitToolError("Absolute paths are not allowed.")
    if any(part == ".." for part in candidate.split("/")):
        raise GitToolError("Paths may not climb out of the repository.")


def build_git_tools(
    workspace: Path, *, timeout: float, allow_writes: bool
) -> list[Tool]:
    return GitRepository(
        workspace, timeout=timeout, allow_writes=allow_writes
    ).tools()
