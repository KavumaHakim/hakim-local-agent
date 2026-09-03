"""Terminal commands, restricted to an allowlist, some of them behind consent.

SECURITY NOTE - READ BEFORE ENABLING
------------------------------------
Disabled by default (config.shell_tool_enabled, env AGENT_ENABLE_SHELL_TOOL=1).

Handing a language model a shell is the single most dangerous thing in this
project, so this is built as an allowlist of specific commands rather than a
filter over arbitrary ones. Denylists leak; allowlists fail closed.

There are three tiers, and the middle one is new:

* **Free.** Reads something and changes nothing: `git log`, `ls`, `head`,
  `npm list`. These run the moment the model asks.
* **Approved.** Changes a file, installs a package, runs a build, or reaches
  the network: `git commit`, `pip install`, `npm run`, `curl`, `mkdir`. The
  turn blocks, the command line is shown verbatim to a person, and it runs
  only if they say yes. Silence declines - nobody watching means nobody
  agreed - and so does stopping the turn.
* **Never.** Programs that run whatever they are handed: `bash`, `sh`, `cmd`,
  `powershell`, `perl`, `xargs`, `sudo`, and `python -c`. These are refused
  with or without approval, and that is deliberate. A prompt is only a control
  if a person can read the command and judge it; nobody can meaningfully audit
  an arbitrary program pasted into `-c` on every call, so the prompt would
  degrade into a rubber stamp. The Python tool exists behind its own switch
  for running Python on purpose.

**What approval is, and is not.** It means a human saw the exact command line
before it ran. It does not mean the command is contained: an approved
`npm run build` executes whatever package.json says, and `make` runs whatever
the Makefile says. The workspace confinement, the scrubbed environment and the
absence of a shell all still apply, but the thing you approved still runs with
your account's privileges. Read the command, not the verb.

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
   the obvious one - have their verbs split: the read-only ones run freely,
   the writing ones need approval, and anything in neither set is refused.
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
programs is small, chosen on purpose, and split so that the ones which change
something have to be agreed to - not that a hostile command would be contained
if one got through.

Widening COMMANDS or AGENT_SHELL_EXTRA is exactly as safe as the programs you
add, and an extra added that way is free rather than approved: `ShellRunner`
invents no rules for a name the operator opted into. Adding an interpreter
(node with -e, powershell) hands over arbitrary execution and undoes
everything above.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tools.base import Tool, ToolError

MAX_COMMAND_LENGTH = 500

# Asked before a command that changes something runs: the command line and
# why it is being asked about, in; whether to go ahead, out. Blocking, and
# expected to have its own timeout - the turn is waiting on it.
ApprovalCheck = Callable[[str, str], bool]


class ShellToolError(ToolError):
    """The command was refused, or could not be run."""


@dataclass(frozen=True)
class Verdict:
    """What validation decided about one command line."""

    argv: list[str]
    # True when a person has to say yes before this runs.
    needs_approval: bool = False
    # Shown in the prompt. Says what the command will do, not that it is risky.
    reason: str = ""


@dataclass(frozen=True)
class CommandRule:
    """What one allowed program may do."""

    name: str
    # Allowed first non-flag token. None means the program takes no sub-command.
    subcommands: frozenset[str] | None = None
    # Sub-commands that are allowed but only after someone approves them.
    # Checked as part of the allowlist: a verb in neither set is refused.
    approval_subcommands: frozenset[str] = field(default_factory=frozenset)
    # Options refused because they change what gets executed.
    banned_options: frozenset[str] = field(default_factory=frozenset)
    # Where those options are dangerous. Git's are only dangerous *before* the
    # sub-command - `git log -c` is a diff format, not `git -c` - so scanning
    # stops at the first positional token. Find is the other way round: its
    # options come after the path, and `find . -exec` was read as safe until
    # this existed.
    banned_options_everywhere: bool = False
    # When set, every argument must appear here. Used for programs that are
    # only safe in one or two exact forms.
    allowed_args: frozenset[str] | None = None
    # True when *every* use of this program needs approval, whatever the verb.
    approval: bool = False
    description: str = ""
    # Why a person is being asked. Written for someone deciding in a hurry.
    approval_reason: str = ""


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

# Git verbs that change the repository or reach the network. Allowed, but a
# person sees the command line first. `config` is here rather than banned
# because setting a value is a normal thing to want and an obvious thing to
# read before agreeing to.
GIT_WRITE_SUBCOMMANDS = frozenset(
    {
        "add", "commit", "checkout", "switch", "restore", "merge", "rebase",
        "cherry-pick", "revert", "reset", "clean", "mv", "rm", "stash",
        "pull", "push", "fetch", "clone", "init", "config", "apply",
        "format-patch", "am",
    }
)

# Version flags, which every one of these programs answers harmlessly.
VERSIONS = frozenset({"--version", "-v", "-V", "version", "--help", "-h"})


def _version_only(name: str, what: str) -> CommandRule:
    """A program allowed only to say what version it is."""
    return CommandRule(
        name=name, allowed_args=VERSIONS, description=f"Report the {what} version."
    )


COMMANDS: dict[str, CommandRule] = {
    # --- inspect the repository ------------------------------------------
    "git": CommandRule(
        name="git",
        subcommands=GIT_SUBCOMMANDS,
        approval_subcommands=GIT_WRITE_SUBCOMMANDS,
        banned_options=GIT_BANNED_OPTIONS,
        description="Inspect the repository; writing verbs need approval.",
        approval_reason="changes the repository or contacts a remote",
    ),
    # --- look around ------------------------------------------------------
    "ls": CommandRule(name="ls", description="List a directory."),
    "dir": CommandRule(name="dir", description="List a directory (Windows)."),
    "tree": CommandRule(name="tree", description="Show the directory tree."),
    "pwd": CommandRule(name="pwd", description="Print the working directory."),
    "where": CommandRule(
        name="where", description="Locate an executable on PATH (Windows)."
    ),
    "which": CommandRule(name="which", description="Locate an executable on PATH."),
    "stat": CommandRule(name="stat", description="File metadata."),
    "file": CommandRule(name="file", description="Guess a file's type."),
    "du": CommandRule(name="du", description="Disk usage of a path."),
    "df": CommandRule(name="df", description="Free space per filesystem."),
    # --- read and slice text ---------------------------------------------
    # No pipes exist here, so each of these is only ever one step. They earn
    # their place on files too large to read whole.
    "head": CommandRule(name="head", description="First lines of a file."),
    "tail": CommandRule(name="tail", description="Last lines of a file."),
    "wc": CommandRule(name="wc", description="Count lines, words, bytes."),
    "sort": CommandRule(name="sort", description="Sort lines of a file."),
    "uniq": CommandRule(name="uniq", description="Collapse repeated lines."),
    "nl": CommandRule(name="nl", description="Number the lines of a file."),
    "diff": CommandRule(name="diff", description="Compare two files."),
    "grep": CommandRule(
        name="grep",
        # -f reads a pattern file, which is fine; there is no --exec here.
        description="Search files for a pattern.",
    ),
    "findstr": CommandRule(
        name="findstr", description="Search files for a pattern (Windows)."
    ),
    "rg": CommandRule(name="rg", description="Search files with ripgrep."),
    # `find` is the odd one: -exec and -delete turn it into a general
    # execution and deletion tool, so both are refused outright rather than
    # sent for approval. Everything else about it is a directory listing.
    "find": CommandRule(
        name="find",
        banned_options=frozenset({"-exec", "-execdir", "-delete", "-ok", "-okdir"}),
        # find's options follow the path it searches, so the scan must not
        # stop at the first positional token the way git's does.
        banned_options_everywhere=True,
        description="Find files by name, size or age.",
    ),
    # --- ask the machine about itself -------------------------------------
    "whoami": CommandRule(name="whoami", description="The current user."),
    "hostname": CommandRule(name="hostname", description="This machine's name."),
    "date": CommandRule(name="date", description="The current date and time."),
    "uname": CommandRule(name="uname", description="The operating system."),
    "env": CommandRule(
        name="env",
        # The child environment is already scrubbed, so this shows the
        # scrubbed one - which is the honest answer to "what will my command
        # see", and not a way to read your API keys.
        description="Show the (already scrubbed) environment a command gets.",
    ),
    # --- toolchains, reporting only ---------------------------------------
    "node": _version_only("node", "Node"),
    "deno": _version_only("deno", "Deno"),
    "java": _version_only("java", "Java"),
    "go": CommandRule(
        name="go",
        subcommands=frozenset({"version", "env", "list", "vet", "doc"}),
        approval_subcommands=frozenset({"build", "run", "test", "get", "install"}),
        description="Inspect a Go module; building and running need approval.",
        approval_reason="compiles or runs Go code, or downloads modules",
    ),
    "cargo": CommandRule(
        name="cargo",
        subcommands=frozenset({"--version", "tree", "metadata", "search"}),
        approval_subcommands=frozenset(
            {"build", "run", "test", "check", "clippy", "add", "update", "install"}
        ),
        description="Inspect a Rust crate; building and running need approval.",
        approval_reason="compiles or runs Rust code, or changes dependencies",
    ),
    "rustc": _version_only("rustc", "Rust compiler"),
    "dotnet": CommandRule(
        name="dotnet",
        subcommands=frozenset({"--version", "--info", "--list-sdks"}),
        approval_subcommands=frozenset({"build", "run", "test", "restore", "add"}),
        description="Inspect the .NET SDK; building and running need approval.",
        approval_reason="compiles or runs .NET code, or restores packages",
    ),
    "pip": CommandRule(
        name="pip",
        subcommands=frozenset({"list", "show", "--version", "freeze", "check"}),
        approval_subcommands=frozenset({"install", "uninstall", "download"}),
        description="Inspect packages; installing needs approval.",
        approval_reason="downloads and installs packages into the environment",
    ),
    "npm": CommandRule(
        name="npm",
        subcommands=frozenset({"--version", "list", "ls", "outdated", "why"}),
        # `run` executes whatever package.json says, which is arbitrary code
        # under a friendly name - so it is approval-gated rather than free,
        # and the prompt shows the script name.
        approval_subcommands=frozenset(
            {"install", "i", "uninstall", "update", "ci", "run", "exec", "audit"}
        ),
        description="Inspect npm packages; installing and running need approval.",
        approval_reason="installs packages or runs a package.json script",
    ),
    "make": CommandRule(
        name="make",
        approval=True,
        description="Run a Makefile target.",
        approval_reason="runs whatever the Makefile says, which can be anything",
    ),
    "docker": CommandRule(
        name="docker",
        subcommands=frozenset({"ps", "images", "version", "info", "logs", "inspect"}),
        approval_subcommands=frozenset(
            {"run", "build", "start", "stop", "rm", "rmi", "exec", "pull", "compose"}
        ),
        description="Inspect containers; starting and building need approval.",
        approval_reason="starts, builds or removes containers",
    ),
    "python": CommandRule(
        name="python",
        # Still only the version. An approval prompt on `python -c ...` would
        # be a rubber stamp: it asks a person to audit an arbitrary program on
        # every call, which is not a control that works. The Python tool
        # exists behind its own switch for running Python on purpose.
        allowed_args=frozenset({"--version", "-V"}),
        description="Report the Python version. Nothing else.",
    ),
    # --- writes to the workspace ------------------------------------------
    "mkdir": CommandRule(
        name="mkdir",
        approval=True,
        description="Create a directory.",
        approval_reason="creates a directory in the workspace",
    ),
    "touch": CommandRule(
        name="touch",
        approval=True,
        description="Create an empty file, or update its timestamp.",
        approval_reason="creates or touches a file in the workspace",
    ),
    "cp": CommandRule(
        name="cp",
        approval=True,
        description="Copy a file.",
        approval_reason="copies a file, overwriting the destination if it exists",
    ),
    "mv": CommandRule(
        name="mv",
        approval=True,
        description="Move or rename a file.",
        approval_reason="moves or renames a file, overwriting the destination",
    ),
    # --- reaches the network ----------------------------------------------
    "curl": CommandRule(
        name="curl",
        # -o and -O write wherever they are pointed; the argument check keeps
        # that inside the workspace, and approval covers the rest.
        approval=True,
        description="Fetch a URL.",
        approval_reason="sends a request off this machine, and may save the reply",
    ),
    "wget": CommandRule(
        name="wget",
        approval=True,
        description="Download a URL.",
        approval_reason="sends a request off this machine and saves the reply",
    ),
    "ping": CommandRule(
        name="ping",
        approval=True,
        description="Check whether a host answers.",
        approval_reason="contacts another machine on the network",
    ),
}

# Programs that run whatever they are handed, and are therefore never allowed
# whatever the arguments look like. Approval does not help: the whole point of
# a prompt is that a person can read the command and judge it, and nobody can
# judge an arbitrary program pasted into `-c` on every call. Kept as a named
# set so the refusal can say why rather than just "not on the list".
NEVER_ALLOWED = {
    "sh", "bash", "zsh", "fish", "dash", "ksh",
    "cmd", "cmd.exe", "powershell", "pwsh", "powershell.exe",
    "perl", "ruby", "php", "lua", "irb", "osascript", "wscript", "cscript",
    "eval", "exec", "source", "sudo", "su", "runas", "start", "xargs",
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


def validate(command: str, allowed: dict[str, CommandRule]) -> Verdict:
    """Decide whether a command may run, and whether to ask first.

    Raises for anything refused. Returns a verdict for anything allowed,
    including the ones that still need a person to agree.
    """
    tokens = tokenize(command)
    program = tokens[0]

    # A bare name only. Anything with a separator is an attempt to reach a
    # binary that is not the allowlisted one.
    if "/" in program or "\\" in program:
        raise ShellToolError(
            f"Give the command by name, not by path: {program!r}."
        )

    if program.lower() in NEVER_ALLOWED:
        raise ShellToolError(
            f"{program!r} is never allowed here, with or without approval: it "
            f"runs whatever it is given, so no one could review a call to it "
            f"meaningfully. Ask for the specific command instead."
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
            if not token.startswith("-") and not rule.banned_options_everywhere:
                break  # for this program, options come before the sub-command

    # Whole programs that always ask, whatever they are asked to do.
    if rule.approval:
        return Verdict(tokens, True, rule.approval_reason or "changes something")

    if rule.subcommands is not None:
        subcommand = next((t for t in arguments if not t.startswith("-")), None)
        if subcommand is None:
            # No verb at all, only options - `git`, `git --version`,
            # `npm --help`. These print usage or a version string and do
            # nothing else, and the options that would make that untrue have
            # already been refused above. Refusing them was friction with
            # nothing behind it: `git --version` used to come back as
            # "git needs one of: blame, branch, ...".
            return Verdict(tokens)

        verb = subcommand.lower()
        if verb in rule.approval_subcommands:
            return Verdict(
                tokens,
                True,
                rule.approval_reason or f"runs {program} {verb}",
            )
        if verb not in rule.subcommands:
            # Both sets are the allowlist; naming only the free half would
            # read as though the approval-gated verbs were forbidden.
            permitted = sorted(rule.subcommands | rule.approval_subcommands)
            raise ShellToolError(
                f"{program} {subcommand!r} is not allowed here. Allowed: "
                f"{', '.join(permitted)}."
            )

    return Verdict(tokens)


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
        approve: ApprovalCheck | None = None,
    ) -> None:
        self._workspace = Path(workspace).expanduser().resolve()
        self._timeout = timeout
        self._max_output = max_output_chars
        # Asked before anything the rules mark as needing a person. None means
        # nobody is listening, and those commands are refused rather than run.
        self._approve = approve

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
        verdict = validate(command, self._allowed)
        argv = verdict.argv

        if verdict.needs_approval:
            asked = " ".join(argv)
            if self._approve is None:
                # No one to ask - the CLI, a test, a script. Refusing is the
                # only honest answer: running it would mean the gate silently
                # does nothing wherever there is no interface attached.
                raise ShellToolError(
                    f"{asked!r} needs approval before it can run, and there is "
                    f"nobody to ask in this context. Run it yourself, or use "
                    f"the web interface where the prompt can be shown."
                )
            if not self._approve(asked, verdict.reason):
                # A refusal, not a failure: the model should say so and carry
                # on, not treat it as an error to work around.
                # Worded against the failure actually seen: a 2B model was
                # declined and then told the user "the directory was created
                # successfully". So this says what did not happen, and what
                # to tell the user, rather than only reporting an error code.
                return {
                    "success": False,
                    "error": (
                        f"DECLINED BY THE USER. {asked!r} did NOT run and "
                        f"nothing was changed. Tell the user their request "
                        f"was not carried out because they declined the "
                        f"command. Do not claim it succeeded, and do not try "
                        f"to run it another way."
                    ),
                    "command": asked,
                    "ran": False,
                    "declined": True,
                }

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
    approve: ApprovalCheck | None = None,
) -> Tool:
    runner = ShellRunner(
        workspace,
        timeout=timeout,
        max_output_chars=max_output_chars,
        extra_commands=extra_commands,
        approve=approve,
    )

    return Tool(
        name="run_command",
        category="terminal",
        description=(
            "Run one terminal command in the workspace and return its output. "
            "Allowed: "
            + ", ".join(runner.allowed_commands)
            + ". Commands that only read run straight away. Commands that "
            "change files, install packages, run a build or reach the network "
            "are shown to the user for approval first, so expect a pause - and "
            "if one is declined, say so rather than looking for another way to "
            "do it. One command only: pipes, redirection and chaining with ; "
            "or && are not interpreted. No absolute paths."
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
