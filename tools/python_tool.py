"""Restricted Python execution for computational work.

SECURITY NOTE - READ BEFORE ENABLING
------------------------------------
This tool is DISABLED by default (config.python_tool_enabled, env
AGENT_ENABLE_PYTHON_TOOL=1).

What it actually does:
  * screens the model's code with `ast` first, rejecting imports, dunder names
    and attributes, and calls to open/eval/exec/compile/__import__/getattr and
    friends;
  * runs the code in a SEPARATE process (`python -I -S`) with a stripped
    __builtins__, a temporary working directory, a wall-clock timeout and an
    output cap.

What it is NOT: a sandbox. CPython was never designed to contain hostile code
in-process, and a determined escape from a restricted-builtins namespace is a
known class of trick. The separate process and the AST screen raise the cost of
an escape and stop the obvious attempts; they do not make it safe to run code
from an untrusted source.

So: leave it off unless you need it, and if you turn it on, understand that you
are trusting the model's output roughly as much as you trust code you paste
into a terminal yourself. Real isolation means a container, a VM or a seccomp
jail, and that is a deliberate follow-up rather than something faked here.

RUNNING SCRIPT FILES
--------------------
`run_python_file` runs a .py file from the workspace. By default it applies the
same screen and the same stripped child as an inline snippet - being on disk
changes nothing about what the code may do.

AGENT_PYTHON_UNRESTRICTED=1 drops all of that and runs plain CPython with
imports, the venv's packages and the filesystem. There is no honest middle
ground: a script that cannot import anything is not much of a script, and
pretending a filtered interpreter is safe would be worse than saying plainly
that this one is not. It is a second, separate opt-in on top of enabling the
tool at all, and it refuses to run when the workspace is the project directory
- otherwise a script could simply rewrite the agent's own source and remove
every other guard.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolError
from tools.shell_tool import ApprovalCheck

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAX_CODE_LENGTH = 10_000
# Files get a larger allowance: the 10k limit is sized for a snippet the model
# typed inline, and a real script easily exceeds it. The AST screen is what
# does the work either way, and it copes fine with a larger tree.
MAX_FILE_CODE_LENGTH = 100_000


def _is_within(candidate: Path, root: Path) -> bool:
    candidate = Path(candidate).resolve()
    root = Path(root).resolve()
    return candidate == root or root in candidate.parents

# Names the model may not call. The child also lacks them, but rejecting here
# gives a clear error instead of a confusing NameError.
_BANNED_NAMES = frozenset(
    {
        "__import__", "open", "eval", "exec", "compile", "input",
        "globals", "locals", "vars", "getattr", "setattr", "delattr",
        "dir", "help", "breakpoint", "exit", "quit", "memoryview",
        "os", "sys", "subprocess", "socket", "requests", "pathlib",
        "shutil", "importlib", "ctypes", "builtins",
    }
)

# Modules the child pre-imports and exposes by name.
_AVAILABLE_MODULES = (
    "math", "statistics", "random", "itertools", "functools",
    "decimal", "fractions", "re", "json", "datetime", "collections",
)


class PythonToolError(ToolError):
    """The code was rejected, or the runner failed."""


# The child program. It never imports anything the parent did not choose, and
# it writes its result as a single JSON line on the real stdout after the
# user's own prints have been captured into a buffer.
_RUNNER = r'''
import io, json, sys
from contextlib import redirect_stdout

MODULES = %(modules)r

safe_builtins = {}
import builtins as _b
_ALLOWED = (
    "abs all any ascii bin bool bytes callable chr complex dict divmod "
    "enumerate filter float format frozenset hash hex int isinstance "
    "issubclass iter len list map max min next oct ord pow print range "
    "repr reversed round set slice sorted str sum tuple type zip "
    "True False None Exception ValueError TypeError ZeroDivisionError "
    "IndexError KeyError StopIteration ArithmeticError OverflowError "
    "RuntimeError NotImplementedError AssertionError"
).split()
for _name in _ALLOWED:
    if hasattr(_b, _name):
        safe_builtins[_name] = getattr(_b, _name)

namespace = {"__builtins__": safe_builtins, "__name__": "__main__"}
for _m in MODULES:
    namespace[_m] = __import__(_m)

code = sys.stdin.read()
buffer = io.StringIO()
real_stdout = sys.stdout
try:
    with redirect_stdout(buffer):
        exec(compile(code, "<model_code>", "exec"), namespace)
except BaseException as exc:
    json.dump(
        {"ok": False, "error": "%%s: %%s" %% (type(exc).__name__, exc),
         "output": buffer.getvalue()},
        real_stdout,
    )
else:
    json.dump({"ok": True, "output": buffer.getvalue()}, real_stdout)
'''


def _screen(code: str, limit: int = MAX_CODE_LENGTH) -> None:
    """Reject obviously dangerous code before spawning anything."""
    if not isinstance(code, str) or not code.strip():
        raise PythonToolError("Code must be a non-empty string.")
    if len(code) > limit:
        raise PythonToolError(
            f"Code is too long ({len(code)} characters, limit {limit})."
        )

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise PythonToolError(f"Syntax error on line {exc.lineno}: {exc.msg}.") from None

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise PythonToolError(
                "Imports are not allowed. These modules are already available: "
                + ", ".join(_AVAILABLE_MODULES)
                + "."
            )
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise PythonToolError(
                f"Access to the private attribute {node.attr!r} is not allowed."
            )
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise PythonToolError(f"The name {node.id!r} is not allowed.")
            if node.id in _BANNED_NAMES:
                raise PythonToolError(f"{node.id!r} is not available in this tool.")


def run_python(
    code: str,
    *,
    timeout: float,
    max_output_chars: int,
    length_limit: int = MAX_CODE_LENGTH,
) -> dict[str, Any]:
    """Screen, then run the code in a separate restricted interpreter."""
    _screen(code, length_limit)

    program = _RUNNER % {"modules": list(_AVAILABLE_MODULES)}

    # A throwaway cwd, so a relative path in the code cannot land in the project.
    with tempfile.TemporaryDirectory() as workdir:
        try:
            completed = subprocess.run(
                # -I isolates from PYTHONPATH and user site-packages;
                # -S skips site.py.
                [sys.executable, "-I", "-S", "-c", program],
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            raise PythonToolError(
                f"Execution exceeded the {timeout:g}s timeout and was stopped."
            ) from None
        except OSError as exc:
            raise PythonToolError(f"Could not start the interpreter: {exc}") from None

    if completed.returncode != 0 and not completed.stdout.strip():
        detail = (completed.stderr or "").strip()[:400] or "no output"
        raise PythonToolError(f"Interpreter exited with an error: {detail}")

    try:
        envelope = json.loads(completed.stdout)
    except ValueError:
        raise PythonToolError("The runner returned unreadable output.") from None

    output = _truncate(envelope.get("output", ""), max_output_chars)

    if not envelope.get("ok"):
        return {
            "success": False,
            "error": envelope.get("error", "Execution failed."),
            "output": output,
        }
    if not output.strip():
        output = "(the code produced no output - use print() to return a value)"
    return {"success": True, "output": output}


def run_python_file(
    path: str,
    *,
    workspace,
    timeout: float,
    max_output_chars: int,
    unrestricted: bool = False,
) -> dict[str, Any]:
    """Run a .py file from the workspace.

    Two modes, and the difference is the whole point:

    * restricted (default) - the file goes through exactly the same AST screen
      and stripped-builtins child as `run_python`. Safe in the same limited
      sense, and equally unable to import anything.
    * unrestricted - plain CPython, with imports, the venv's packages and the
      filesystem. That is arbitrary code execution and is labelled as such;
      nothing here contains it.
    """
    try:
        target = workspace.resolve(path)
    except ToolError as exc:
        raise PythonToolError(str(exc)) from None

    if not target.is_file():
        raise PythonToolError(f"No such file in the workspace: {path}")
    if target.suffix.lower() != ".py":
        raise PythonToolError(f"{path} is not a .py file.")

    try:
        source = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise PythonToolError(f"Could not read {path}: {exc}") from None

    if not unrestricted:
        # Same rules as an inline snippet: the file being on disk changes
        # nothing about what it is allowed to do.
        return run_python(
            source,
            timeout=timeout,
            max_output_chars=max_output_chars,
            length_limit=MAX_FILE_CODE_LENGTH,
        )

    # An unrestricted script can do anything this account can, including
    # rewriting tools/base.py and switching off every other guard. Refusing to
    # run one against the agent's own directory is what keeps the rest of the
    # security model meaningful.
    if _is_within(workspace.root, PROJECT_ROOT):
        raise PythonToolError(
            "Unrestricted Python will not run while the workspace is the "
            "agent's own project directory: a script could rewrite the tools "
            "that enforce every other limit. Point AGENT_WORKSPACE at a "
            "different directory."
        )

    try:
        completed = subprocess.run(
            # -I still keeps PYTHONPATH and the user site directory out; the
            # venv's own packages remain available, which is the point.
            [sys.executable, "-I", str(target)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workspace.root,
        )
    except subprocess.TimeoutExpired:
        raise PythonToolError(
            f"{path} exceeded the {timeout:g}s timeout and was stopped."
        ) from None
    except OSError as exc:
        raise PythonToolError(f"Could not start the interpreter: {exc}") from None

    stdout = _truncate(completed.stdout or "", max_output_chars)
    stderr = _truncate(completed.stderr or "", max_output_chars)

    # A non-zero exit is information for the model, not a tool failure.
    return {
        "success": True,
        "path": path,
        "exit_code": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "output": stdout or stderr or "(no output)",
    }


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated at {limit} characters]"


def build_python_file_tool(
    *,
    workspace,
    timeout: float,
    max_output_chars: int,
    unrestricted: bool,
    approve: ApprovalCheck | None = None,
) -> Tool:
    """The file runner, asked about first when it is the unrestricted one.

    Running an arbitrary script is the most powerful thing in this project -
    more so than any command the terminal tool will run - and the terminal
    tool asks before creating a directory. Leaving this one silent while
    `mkdir` prompts had the gate exactly the wrong way round.

    Only the unrestricted form asks. The restricted one cannot import, open a
    file or reach the network, so there is nothing for a person to weigh.
    """

    def _run(path: str) -> dict[str, Any]:
        if unrestricted:
            asked = f"run_python_file {path}"
            if approve is None:
                raise PythonToolError(
                    f"{asked} needs approval before it can run, and there is "
                    f"nobody to ask in this context. Run the script yourself, "
                    f"or use the web interface where the prompt can be shown."
                )
            if not approve(
                asked,
                "runs a Python script with no restrictions - imports, the "
                "filesystem and the network are all available to it",
            ):
                return {
                    "success": False,
                    "error": (
                        f"Not approved: {asked} was declined, or the request "
                        f"timed out. Do not try to run it another way."
                    ),
                    "path": path,
                    "declined": True,
                }
        return run_python_file(
            path,
            workspace=workspace,
            timeout=timeout,
            max_output_chars=max_output_chars,
            unrestricted=unrestricted,
        )

    if unrestricted:
        limits = (
            "Full Python: imports, installed packages and the filesystem are "
            "all available, and the working directory is the workspace."
        )
    else:
        limits = (
            "The file runs under the same restrictions as run_python: no "
            "imports, no file access, no network. Available modules: "
            + ", ".join(_AVAILABLE_MODULES)
            + "."
        )

    return Tool(
        name="run_python_file",
        category="python",
        description=(
            "Run a .py file that already exists in the workspace and return "
            "what it prints. Give a path relative to the workspace root. "
            + limits
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The .py file to run, relative to the workspace root.",
                }
            },
            "required": ["path"],
        },
        run=_run,
    )


def build_python_tool(*, timeout: float, max_output_chars: int) -> Tool:
    def _run(code: str) -> dict[str, Any]:
        return run_python(code, timeout=timeout, max_output_chars=max_output_chars)

    return Tool(
        name="run_python",
        category="python",
        description=(
            "Run a short Python snippet for data processing or multi-step "
            "computation, and return whatever it prints. Use print() to return "
            "results. Available modules: "
            + ", ".join(_AVAILABLE_MODULES)
            + ". No imports, no file access, no network, no shell. "
            "For a single arithmetic expression use the calculate tool instead."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to run. Use print() for output.",
                }
            },
            "required": ["code"],
        },
        run=_run,
    )
