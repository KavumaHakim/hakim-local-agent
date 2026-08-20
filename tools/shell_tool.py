"""Terminal commands, restricted to an allowlist.

SECURITY NOTE - READ BEFORE ENABLING
------------------------------------
Disabled by default (config.shell_tool_enabled, env AGENT_ENABLE_SHELL_TOOL=1).

Handing a language model a shell is the single most dangerous thing in this
project, so this is built as an allowlist of specific commands rather than a
filter over arbitrary ones. Denylists leak; allowlists fail closed.

The design, in order of how much it actually protects you:

1. **No shell.** The command is tokenised and handed straight to CreateProcess.
   Nothing ever reaches cmd.exe or a POSIX shell, so `;`, `&&`, `||`, `|`,
   `>`, backticks and `$(...)` are not metacharacters - they are literal
   argument text. This removes command chaining as a category rather than
   trying to spot it.
2. **Executable allowlist.** Only the programs in COMMANDS may run, matched on
   the bare name. A path separator in the command name is refused outright, so
   `.\evil.exe` and `C:\Windows\System32\cmd.exe` never resolve.
3. **Sub-command allowlist.** Programs that can both read and write - git being
   the obvious one - are restricted to their read-only verbs.
4. **Dangerous option screening.** Some options turn a safe program into an
   arbitrary one. `git -c core.pager='sh -c ...' log` executes whatever you
   like, and `git --exec-path` relocates the binaries git calls. Those are
   rejected explicitly, because no sub-command allowlist would catch them.
5. **Workspace confinement.** The working directory is the workspace, and
   arguments may not contain absolute paths or `..` segments.
6. **A scrubbed environment.** The child gets a minimal PATH-and-essentials
   environment rather than inheriting yours, so API keys and tokens sitting in
   your shell are not handed to a subprocess on the model's say-so.
7. **Timeout and output cap.**

What this is not: a sandbox. Every allowed command still runs with your user
account's privileges. The protection here is that the set of reachable
programs is small, read-only, and chosen on purpose - not that a hostile
command would be contained if one got through.

Widening COMMANDS or AGENT_SHELL_EXTRA is exactly as safe as the programs you
add. Adding an interpreter (python, node, powershell) hands over arbitrary
execution and undoes everything above.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolError

MAX_COMMAND_LENGTH = 500


class ShellToolError(ToolError):
    """The command was refused, or could not be run."""


@dataclass(frozen=True)
class CommandRule:
    """What one allowed program may do."""

    name: str
    # Allowed first non-flag token. None means the program takes no sub-command.
    subcommands: frozenset[str] | None = None
    # Options refused before the sub-command, which change what gets executed.
    banned_options: frozenset[str] = field(default_factory=frozenset)
    # When set, every argument must appear here. Used for programs that are
    # only safe in one or two exact forms.
    allowed_args: frozenset[str] | None = None
    description: str = ""


# Read-only git verbs. Deliberately excludes anything that writes: no commit,
# checkout, reset, clean, push, pull, fetch, rebase, merge, stash, apply.
# `config` is excluded too - it writes as readily as it reads.
GIT_SUBCOMMANDS = frozenset(
    {
        "status", "log", "diff", "show", "branch", "tag", "remote",
        "describe", "blame", "shortlog", "ls-files", "ls-tree",
        "rev-parse", "count-objects", "whatchanged", "grep",
    }
)

# Options that make git run something of the caller's choosing, whatever the
# sub-command is.
GIT_BANNED_OPTIONS = frozenset(
    {"-c", "--exec-path", "--upload-pack", "--receive-pack", "-C",
     "--git-dir", "--work-tree", "--namespace"}
)

COMMANDS: dict[str, CommandRule] = {
    "git": CommandRule(
        name="git",
        subcommands=GIT_SUBCOMMANDS,
        banned_options=GIT_BANNED_OPTIONS,
        description="Inspect the repository. Read-only verbs only.",
    ),
    "where": CommandRule(
        name="where",
        description="Locate an executable on PATH (Windows).",
    ),
    "pip": CommandRule(
        name="pip",
        subcommands=frozenset({"list", "show", "--version"}),
        description="Inspect installed packages.",
    ),
    "python": CommandRule(
        name="python",
        # An interpreter is only safe in forms that cannot execute anything.
        allowed_args=frozenset({"--version", "-V"}),
        description="Report the Python version. Nothing else.",
    ),
}


def _strip_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


def tokenize(command: str) -> list[str]:
    """Split a command line into argv, without shell semantics.

    Uses POSIX rules so quoted arguments containing spaces survive intact
    (`--pretty="%h %s"` is one token, not two broken ones). POSIX rules also
    treat a backslash as an escape, which would silently turn `sub\\file.txt`
    into `subfile.txt` - so backslashes are refused up front with a pointer to
    forward slashes, which Windows accepts anyway.
    """
    if not isinstance(command, str) or not command.strip():
        raise ShellToolError("Command must be a non-empty string.")
    if len(command) > MAX_COMMAND_LENGTH:
        raise ShellToolError(
            f"Command is too long ({len(command)} characters, limit "
            f"{MAX_COMMAND_LENGTH})."
        )
    if "\n" in command or "\r" in command:
        raise ShellToolError("Only a single command line is allowed.")
    if "\\" in command:
        raise ShellToolError(
            "Use forward slashes in paths, not backslashes. Windows accepts "
            "them and they cannot be mistaken for escape characters."
        )

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ShellToolError(f"Could not parse the command: {exc}") from None

    tokens = [t for t in tokens if t]
    if not tokens:
        raise ShellToolError("Command must be a non-empty string.")
    return tokens


def _check_argument(token: str) -> None:
    """Keep arguments inside the workspace."""
    if token.startswith("-"):
        return  # an option, not a path

    candidate = token.replace("\\", "/")
    if candidate.startswith("/") or (len(token) > 1 and token[1] == ":"):
        raise ShellToolError(
            f"Absolute paths are not allowed in arguments: {token!r}. "
            f"Use a path relative to the workspace."
        )
    if any(part == ".." for part in candidate.split("/")):
        raise ShellToolError(
            f"Paths may not climb out of the workspace: {token!r}."
        )


def validate(command: str, allowed: dict[str, CommandRule]) -> list[str]:
    """Return argv if the command is permitted, else raise."""
    tokens = tokenize(command)
    program = tokens[0]

    # A bare name only. Anything with a separator is an attempt to reach a
    # binary that is not the allowlisted one.
    if "/" in program or "\\" in program:
        raise ShellToolError(
            f"Give the command by name, not by path: {program!r}."
        )

    rule = allowed.get(program.lower())
    if rule is None:
        raise ShellToolError(
            f"{program!r} is not an allowed command. Available: "
            f"{', '.join(sorted(allowed))}."
        )

    arguments = tokens[1:]

    for token in arguments:
        _check_argument(token)

    if rule.allowed_args is not None:
        for token in arguments:
            if token not in rule.allowed_args:
                raise ShellToolError(
                    f"{program} only accepts "
                    f"{', '.join(sorted(rule.allowed_args))} in this tool."
                )

    if rule.banned_options:
        for token in arguments:
            # Match -c and --exec-path=... alike.
            head = token.split("=", 1)[0]
            if head in rule.banned_options:
                raise ShellToolError(
                    f"The {head!r} option is not allowed with {program}: it can "
                    f"change which programs {program} runs."
                )
            if not token.startswith("-"):
                break  # options come before the sub-command

    if rule.subcommands is not None:
        subcommand = next((t for t in arguments if not t.startswith("-")), None)
        if subcommand is None:
            # `git` alone, or only options: harmless, prints usage.
            if not arguments:
                return tokens
            raise ShellToolError(
                f"{program} needs one of: {', '.join(sorted(rule.subcommands))}."
            )
        if subcommand.lower() not in rule.subcommands:
            raise ShellToolError(
                f"{program} {subcommand!r} is not allowed here. Allowed: "
                f"{', '.join(sorted(rule.subcommands))}."
            )

    return tokens


def _child_environment() -> dict[str, str]:
    """A minimal environment, so secrets in yours are not passed along."""
    keep = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
            "TEMP", "TMP", "HOME", "USERPROFILE", "LANG", "LC_ALL")
    child = {name: os.environ[name] for name in keep if name in os.environ}
    child["GIT_TERMINAL_PROMPT"] = "0"   # never block waiting for credentials
    child["GIT_PAGER"] = "cat"
    child["PAGER"] = "cat"
    return child


class ShellRunner:
    """Runs allowlisted commands inside the workspace."""

    def __init__(
        self,
        workspace: Path,
        *,
        timeout: float = 30.0,
        max_output_chars: int = 4000,
        extra_commands: tuple[str, ...] = (),
    ) -> None:
        self._workspace = Path(workspace).expanduser().resolve()
        self._timeout = timeout
        self._max_output = max_output_chars

        self._allowed = dict(COMMANDS)
        for name in extra_commands:
            name = name.strip().lower()
            if name and name not in self._allowed:
                # Opted in by the operator, so no sub-command restrictions are
                # invented for it. Documented as your responsibility.
                self._allowed[name] = CommandRule(
                    name=name, description="Added via AGENT_SHELL_EXTRA."
                )

    @property
    def allowed_commands(self) -> list[str]:
        return sorted(self._allowed)

    def run(self, command: str) -> dict[str, Any]:
        argv = validate(command, self._allowed)

        try:
            completed = subprocess.run(
                argv,
                cwd=self._workspace,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=_child_environment(),
                # The absent shell=True is the point of this module.
                shell=False,
            )
        except subprocess.TimeoutExpired:
            raise ShellToolError(
                f"Command exceeded the {self._timeout:g}s timeout and was stopped."
            ) from None
        except FileNotFoundError:
            raise ShellToolError(
                f"{argv[0]!r} is allowed but not installed, or not on PATH."
            ) from None
        except OSError as exc:
            raise ShellToolError(f"Could not run the command: {exc}") from None

        stdout = _truncate(completed.stdout or "", self._max_output)
        stderr = _truncate(completed.stderr or "", self._max_output)

        # A non-zero exit is information for the model, not a tool failure:
        # `git diff --quiet` uses exit codes to answer a question.
        return {
            "success": True,
            "command": " ".join(argv),
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "output": stdout or stderr or "(no output)",
        }


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated at {limit} characters]"


def build_shell_tool(
    workspace: Path,
    *,
    timeout: float,
    max_output_chars: int,
    extra_commands: tuple[str, ...] = (),
) -> Tool:
    runner = ShellRunner(
        workspace,
        timeout=timeout,
        max_output_chars=max_output_chars,
        extra_commands=extra_commands,
    )

    return Tool(
        name="run_command",
        category="terminal",
        description=(
            "Run one read-only terminal command in the workspace and return its "
            "output. Allowed: "
            + ", ".join(runner.allowed_commands)
            + ". git is limited to read-only verbs (status, log, diff, show, "
            "branch, remote, ls-files, rev-parse and similar). "
            "One command only - pipes, redirection and chaining with ; or && "
            "are not interpreted. No absolute paths."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "The command line, e.g. 'git status --short' or "
                        "'git log --oneline -10'."
                    ),
                }
            },
            "required": ["command"],
        },
        run=runner.run,
    )
