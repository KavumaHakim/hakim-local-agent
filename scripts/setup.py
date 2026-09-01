"""Set up a fresh clone: virtualenv, dependencies, front end, configuration.

One Python script rather than a .bat and a .sh that drift apart. Python is the
one prerequisite this project cannot avoid, so it is also the one thing a setup
script may assume is present; `setup.bat` and `setup.sh` exist only to find an
interpreter and hand over to this.

What it does NOT do is download a model or install llama.cpp. Both are large,
both belong to other projects with their own instructions, and a setup script
that quietly pulls gigabytes over someone's connection is not being helpful.
It checks for them and says exactly what is missing.

Everything here is safe to run twice.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
WEIGHTS = ROOT / "weights"

MIN_PYTHON = (3, 11)

# Set by main() once the arguments are known.
QUIET = False


# --- output ---------------------------------------------------------------
#
# Deliberately plain ASCII. This runs in cmd.exe as often as in a terminal
# that can render anything nicer, and a UnicodeEncodeError from a tick mark
# would be an absurd way for a setup script to fail.


def say(message: str = "") -> None:
    print(message, flush=True)


def step(message: str) -> None:
    say(f"\n==> {message}")


def ok(message: str) -> None:
    say(f"  [ok] {message}")


def warn(message: str) -> None:
    say(f"  [!]  {message}")


def fail(message: str) -> None:
    say(f"  [X]  {message}")


def run(command: list[str], *, cwd: Path | None = None) -> bool:
    """Run a command, showing its output only when it fails."""
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd or ROOT),
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        fail(f"could not run {command[0]}: {exc}")
        return False

    if completed.returncode != 0:
        fail(" ".join(command))
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        for line in tail[-15:]:
            say(f"       {line}")
        if not tail:
            say(f"       exit code {completed.returncode}, and it said nothing.")
        return False
    return True


# --- the pieces -----------------------------------------------------------


def venv_python() -> Path:
    """The interpreter inside the virtualenv, on either platform."""
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def check_python() -> bool:
    step("Checking Python")
    if sys.version_info < MIN_PYTHON:
        fail(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is too old. "
            f"This needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer."
        )
        return False
    ok(f"Python {sys.version.split()[0]} at {sys.executable}")
    return True


def make_venv() -> bool:
    step("Creating the virtualenv at .venv")
    if venv_python().is_file():
        ok("already there")
        return True
    try:
        venv.EnvBuilder(with_pip=True).create(VENV)
    except Exception as exc:  # noqa: BLE001 - report whatever venv complains of
        fail(f"could not create {VENV}: {exc}")
        if os.name != "nt":
            warn("On Debian and Ubuntu this usually means: apt install python3-venv")
        return False
    ok(str(VENV))
    return True


def install_python_deps(*, with_rag: bool) -> bool:
    step("Installing Python dependencies")
    python = str(venv_python())
    if not run([python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"]):
        return False
    if not run([python, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"]):
        return False
    ok("requirements.txt")

    # The test dependencies, because this script offers to run the tests and
    # the README tells people to. Small - one HTTP client - so it is not worth
    # a flag to skip.
    if (ROOT / "requirements-dev.txt").is_file():
        if not run(
            [python, "-m", "pip", "install", "-r", "requirements-dev.txt", "--quiet"]
        ):
            return False
        ok("requirements-dev.txt")

    if with_rag:
        say("       document search pulls in torch; this is the slow part")
        if not run(
            [python, "-m", "pip", "install", "-r", "requirements-rag.txt", "--quiet"]
        ):
            return False
        ok("requirements-rag.txt")
    else:
        warn("skipped document search (torch). Add it later with --with-rag")
    return True


def install_web(*, build: bool) -> bool:
    step("Installing the front end")
    npm = shutil.which("npm")
    if npm is None:
        warn("npm not found, so the web UI was skipped.")
        warn("Install Node.js 20 or newer, then re-run this script.")
        warn("The terminal client (python main.py) works without it.")
        return True

    # Not --silent: npm says nothing on success anyway, and on failure the
    # quiet flag hides the one thing worth having. `run` only shows output
    # when a command fails, so this costs nothing when it works.
    if not run([npm, "--prefix", "web", "install", "--no-fund", "--no-audit"]):
        return False
    ok("npm install")

    if build:
        if not run([npm, "--prefix", "web", "run", "build"]):
            return False
        ok("npm run build - the API will serve the built UI at /")
    return True


def check_llama_server() -> bool:
    """Is there a llama-server to run models with?"""
    step("Looking for llama-server")
    registry = ROOT / "models.json"
    configured = ""
    if registry.is_file():
        try:
            configured = json.loads(registry.read_text(encoding="utf-8")).get(
                "server_exe", ""
            )
        except (ValueError, OSError):
            configured = ""

    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = (registry.parent / candidate).resolve()
        if candidate.is_file():
            ok(f"models.json points at {candidate}")
            return True

    found = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if found:
        ok(f"found on PATH at {found}")
        return True

    warn("llama-server was not found, so no local model can start yet.")
    warn("It is part of llama.cpp, and this project does not bundle it:")
    warn("    https://github.com/ggml-org/llama.cpp/releases")
    warn("Put it on PATH, or set \"server_exe\" in models.json to its path.")
    return False


def check_weights() -> bool:
    step("Looking for model weights")
    WEIGHTS.mkdir(exist_ok=True)
    found = sorted(WEIGHTS.glob("*.gguf"))
    if found:
        for path in found:
            ok(f"{path.name}  ({path.stat().st_size / 1e9:.1f} GB)")
        return True

    warn(f"no .gguf files in {WEIGHTS}")
    warn("Any GGUF works - it is sized from its own header when discovered.")
    warn("A small instruct model is the right place to start on 8 GB of RAM.")
    return False


def make_env_file() -> None:
    step("Checking .env")
    target = ROOT / ".env"
    if target.is_file():
        ok(".env is already there, leaving it alone")
        return
    target.write_text(
        "# API keys for hosted models. Git-ignored, and only read at startup.\n"
        "# Leave this empty to stay entirely local - nothing here is required.\n"
        "#\n"
        "# The name of each key is set in models.json, under api_key_env.\n"
        "# GEMINI_API_KEY=\n"
        "# CEREBRAS_API_KEY=\n",
        encoding="utf-8",
    )
    ok("wrote a commented .env - no keys needed to run locally")


def verify() -> bool:
    step("Verifying the install")
    python = str(venv_python())
    if not run([python, "-c", "import fastapi, uvicorn, requests"]):
        fail("the API's own dependencies are not importable")
        return False
    ok("api imports")

    if not run([python, "-m", "unittest", "tests.test_tools", "-q"]):
        fail("the tool tests did not pass")
        return False
    ok("tests pass")
    return True


def summary(*, llama: bool, weights: bool) -> None:
    step("Done")
    activate = (
        r".venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate"
    )
    launcher = "start.bat" if os.name == "nt" else "./start.sh"

    if llama and weights:
        say("  Everything needed to run locally is in place.")
    else:
        say("  The project is installed, but a local model cannot start yet:")
        if not llama:
            say("    - llama-server is missing (see above)")
        if not weights:
            say("    - there are no .gguf files in weights/ (see above)")
        say("  Fix those and re-run this script to check.")

    say("")
    say("  Start everything:")
    say(f"      {launcher}")
    say("")
    say("  Or by hand, in two terminals:")
    say(f"      {activate}")
    say("      python -m uvicorn api.main:app --host 127.0.0.1 --port 8000")
    say("      npm --prefix web run dev")
    say("")
    say("  Terminal client instead of the web UI:")
    say(f"      {activate}")
    say("      python main.py")
    say("")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set up a fresh clone of the Hakim AI System."
    )
    parser.add_argument(
        "--with-rag",
        action="store_true",
        help="also install document search (torch, ~2 GB)",
    )
    parser.add_argument(
        "--build-web",
        action="store_true",
        help="build the front end so the API serves it, instead of running Vite",
    )
    parser.add_argument(
        "--skip-tests", action="store_true", help="do not run the verification tests"
    )
    arguments = parser.parse_args()

    say("Hakim AI System - setup")
    say(f"  project: {ROOT}")

    if not check_python():
        return 1
    if not make_venv():
        return 1
    if not install_python_deps(with_rag=arguments.with_rag):
        return 1
    if not install_web(build=arguments.build_web):
        return 1

    llama = check_llama_server()
    weights = check_weights()
    make_env_file()

    if not arguments.skip_tests and not verify():
        return 1

    summary(llama=llama, weights=weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
