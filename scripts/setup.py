"""Set up a fresh clone: a guided walkthrough, or a single non-interactive run.

One Python script rather than a .bat and a .sh that drift apart. Python is the
one prerequisite this project cannot avoid, so it is also the one thing a setup
script may assume is present; `setup.bat` and `setup.sh` exist only to find an
interpreter and hand over to this.

Run with a terminal attached it walks through the choices - what to install,
whether to fetch llama.cpp - and shows progress while it works. Run from a
script, a pipe or CI it asks nothing, takes the defaults, and prints plain
lines with no cursor tricks. `--yes` forces that second mode explicitly.

What it does NOT do is download a model. Which one to run depends on your RAM,
your language and what you want the agent for, and several gigabytes is not
something a setup script should pull on someone's behalf. It fetches llama.cpp,
because that choice is not personal: there is one right build for a machine.

Everything here is safe to run twice.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import venv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ui  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
WEIGHTS = ROOT / "weights"

MIN_PYTHON = (3, 11)


def run(command: list[str], *, label: str, cwd: Path | None = None) -> bool:
    """Run a command behind a spinner, showing its output only if it fails."""
    result: dict[str, object] = {}

    def work() -> None:
        try:
            result["completed"] = subprocess.run(
                command,
                cwd=str(cwd or ROOT),
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            result["error"] = exc

    with ui.Spinner(label) as spinner:
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        worker.join()
        seconds = spinner.elapsed

    if "error" in result:
        ui.fail(f"could not run {command[0]}: {result['error']}")
        return False

    completed = result["completed"]
    if completed.returncode != 0:
        ui.fail(f"{label} failed")
        ui.note(" ".join(command))
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        for line in tail[-15:]:
            ui.note(line)
        if not tail:
            ui.note(f"exit code {completed.returncode}, and it said nothing.")
        return False

    ui.ok(f"{label} {ui.DIM}({seconds:.0f}s){ui.RESET}")
    return True


# --- the pieces -----------------------------------------------------------


def venv_python() -> Path:
    """The interpreter inside the virtualenv, on either platform."""
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def check_python() -> bool:
    if sys.version_info < MIN_PYTHON:
        ui.fail(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is too old. "
            f"This needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer."
        )
        return False
    ui.ok(f"Python {sys.version.split()[0]}")
    ui.note(sys.executable)
    return True


def make_venv() -> bool:
    if venv_python().is_file():
        ui.ok("already there")
        ui.note(str(VENV))
        return True
    with ui.Spinner("creating .venv"):
        try:
            venv.EnvBuilder(with_pip=True).create(VENV)
        except Exception as exc:  # noqa: BLE001 - report whatever venv says
            ui.fail(f"could not create {VENV}: {exc}")
            if os.name != "nt":
                ui.note("On Debian and Ubuntu: sudo apt install python3-venv")
            return False
    ui.ok(str(VENV))
    return True


def install_python_deps(*, with_rag: bool) -> bool:
    python = str(venv_python())
    if not run(
        [python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
        label="upgrading pip",
    ):
        return False
    if not run(
        [python, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
        label="requirements.txt",
    ):
        return False

    if (ROOT / "requirements-dev.txt").is_file():
        if not run(
            [python, "-m", "pip", "install", "-r", "requirements-dev.txt", "--quiet"],
            label="requirements-dev.txt",
        ):
            return False

    if with_rag:
        ui.note("torch is the largest download here; several minutes is normal")
        if not run(
            [python, "-m", "pip", "install", "-r", "requirements-rag.txt", "--quiet"],
            label="document search (torch)",
        ):
            return False
    return True


def install_web(*, build: bool) -> bool:
    npm = shutil.which("npm")
    if npm is None:
        ui.warn("npm not found, so the web UI was skipped.")
        ui.note("Install Node.js 20 or newer, then run this again.")
        ui.note("The terminal client (python main.py) works without it.")
        return True

    # Not --silent: npm says nothing on success anyway, and the quiet flag
    # hides the one thing worth having when it fails.
    if not run(
        [npm, "--prefix", "web", "install", "--no-fund", "--no-audit"],
        label="npm install",
    ):
        return False

    if build:
        if not run([npm, "--prefix", "web", "run", "build"], label="npm run build"):
            return False
        ui.note("the API will serve the built UI at /")
    return True


def find_llama_server() -> Path | None:
    """An existing llama-server: configured, vendored, or on PATH."""
    registry = ROOT / "models.json"
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
                return candidate

    vendored = ROOT / "vendor" / "llama"
    if vendored.is_dir():
        for name in ("llama-server.exe", "llama-server"):
            for found in sorted(vendored.rglob(name)):
                if found.is_file():
                    return found

    found = shutil.which("llama-server") or shutil.which("llama-server.exe")
    return Path(found) if found else None


def _download_progress(label: str, total: int):
    """Hand get_llama a progress bar without it having to know about ui."""
    return ui.Progress(label, total)


def check_llama_server(*, download: bool) -> bool:
    existing = find_llama_server()
    if existing is not None:
        ui.ok("already here")
        ui.note(str(existing))
        return True

    if not download:
        ui.warn("not installed, and it was not asked for.")
        ui.note("https://github.com/ggml-org/llama.cpp/releases")
        return False

    try:
        import get_llama

        server = get_llama.install(on_progress=_download_progress)
    except Exception as exc:  # noqa: BLE001 - report whatever went wrong
        ui.fail(f"could not fetch llama.cpp: {exc}")
        ui.note("Download one by hand and put it on PATH:")
        ui.note("https://github.com/ggml-org/llama.cpp/releases")
        return False

    ui.ok(str(server))
    return True


def check_weights() -> bool:
    WEIGHTS.mkdir(exist_ok=True)
    found = sorted(WEIGHTS.glob("*.gguf"))
    if found:
        for path in found:
            ui.ok(f"{path.name} {ui.DIM}({path.stat().st_size / 1e9:.1f} GB){ui.RESET}")
        return True

    ui.warn("no .gguf files yet - this is the one thing you supply")
    ui.note("Any GGUF works; it is sized from its own header when discovered.")
    ui.note("On 8 GB of RAM, a 2-3B instruct model at Q4_K_M is the place to start.")
    ui.note("See 'Choosing a model for your hardware' in the README.")
    return False


def make_env_file() -> None:
    """Put a .env in place, from the committed example.

    Copied rather than generated, so there is one description of the settings
    instead of two that drift. `.env.example` carries names and no values, so
    copying it commits nothing and enables nothing.
    """
    target = ROOT / ".env"
    if target.is_file():
        ui.ok(".env is already there, leaving it alone")
        return

    example = ROOT / ".env.example"
    if not example.is_file():
        ui.warn("no .env.example to copy; hosted models will need one by hand")
        return

    shutil.copyfile(example, target)
    ui.ok("copied .env.example to .env")
    ui.note("no keys needed to run locally")


def verify() -> bool:
    python = str(venv_python())
    if not run([python, "-c", "import fastapi, uvicorn, requests"], label="api imports"):
        return False
    if not run([python, "-m", "unittest", "tests.test_tools", "-q"], label="tests"):
        return False
    return True


# --- the walkthrough ------------------------------------------------------


def plan(arguments) -> dict:
    """Work out what to do, asking when there is somebody to ask."""
    scripted = arguments.yes or not ui.interactive()

    choices = {
        "rag": arguments.with_rag,
        "web_build": arguments.build_web,
        "llama": not arguments.no_llama,
        "tests": not arguments.skip_tests,
    }

    if scripted:
        return choices

    ui.say(
        f"\n  This installs into {ui.BOLD}.venv{ui.RESET} inside the project "
        f"and touches nothing else."
    )

    options = [
        {
            "label": "Download llama.cpp (about 18 MB)",
            "note": "The engine that runs the models. Skip if you already have one.",
            "on": choices["llama"],
            "key": "llama",
        },
        {
            "label": "Document search (torch, about 2 GB)",
            "note": "Semantic search over your own files. Slow to install; optional.",
            "on": choices["rag"],
            "key": "rag",
        },
        {
            "label": "Build the web UI for production",
            "note": "Otherwise it runs in development mode, which is what most want.",
            "on": choices["web_build"],
            "key": "web_build",
        },
        {
            "label": "Run the tests at the end",
            "note": "About a minute, and it is how you know the install is sound.",
            "on": choices["tests"],
            "key": "tests",
        },
    ]
    for item in ui.toggle("Optional pieces", options):
        choices[item["key"]] = item["on"]
    return choices


def summary(*, llama: bool, weights: bool) -> None:
    launcher = "start.bat" if os.name == "nt" else "./start.sh"
    activate = (
        r".venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate"
    )

    ui.rule()
    if llama and weights:
        ui.say(f"\n  {ui.GREEN}{ui.BOLD}Ready.{ui.RESET} Start it with:\n")
        ui.say(f"      {ui.BOLD}{launcher}{ui.RESET}")
    elif llama and not weights:
        ui.say(f"\n  {ui.BOLD}Installed. One thing left: a model.{ui.RESET}\n")
        ui.say(f"  Drop any .gguf into {ui.BOLD}weights/{ui.RESET} and it is picked up")
        ui.say("  automatically - no configuration, it is sized from its own header.")
        ui.say(f"\n  {ui.DIM}https://huggingface.co/models?library=gguf{ui.RESET}")
        ui.say(f"\n  Then: {ui.BOLD}{launcher}{ui.RESET}")
    else:
        ui.say(f"\n  {ui.BOLD}Installed, but a local model cannot start yet.{ui.RESET}")
        if not llama:
            ui.say("    - llama-server is missing (see above)")
        if not weights:
            ui.say("    - there are no .gguf files in weights/")
        ui.say("\n  Fix those and run this again to check.")

    ui.say(f"\n  {ui.DIM}By hand, in two terminals:{ui.RESET}")
    ui.note(activate)
    ui.note("python -m uvicorn api.main:app --host 127.0.0.1 --port 8000")
    ui.note("npm --prefix web run dev")
    ui.say(f"\n  {ui.DIM}Terminal client instead of the web UI:{ui.RESET}")
    ui.note("python main.py")
    ui.say()


def step_names(choices: dict) -> list[str]:
    names = [
        "Checking Python",
        "Creating the virtualenv",
        "Python dependencies",
        "Front end",
        "llama.cpp" if choices["llama"] else "Looking for llama.cpp",
        "Model weights",
        "Configuration",
    ]
    if choices["tests"]:
        names.append("Verifying")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set up a fresh clone of the Hakim AI System."
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="ask nothing; take the defaults and the flags below",
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
    parser.add_argument(
        "--no-llama",
        action="store_true",
        help="do not download llama.cpp (about 18 MB from GitHub)",
    )
    arguments = parser.parse_args()

    ui.say()
    ui.say(f"  {ui.BOLD}Hakim AI System{ui.RESET} {ui.DIM}- setup{ui.RESET}")
    ui.note(str(ROOT))

    try:
        choices = plan(arguments)
    except (EOFError, KeyboardInterrupt):
        ui.say("\n  Stopped. Nothing was changed.")
        return 1

    steps = ui.Steps(step_names(choices))
    steps.show()
    position = 0

    def advance() -> None:
        nonlocal position
        steps.start(position)
        position += 1

    llama = False
    weights = False

    try:
        advance()
        if not check_python():
            return 1

        advance()
        if not make_venv():
            return 1

        advance()
        if not install_python_deps(with_rag=choices["rag"]):
            return 1

        advance()
        if not install_web(build=choices["web_build"]):
            return 1

        advance()
        llama = check_llama_server(download=choices["llama"])

        advance()
        weights = check_weights()

        advance()
        make_env_file()

        if choices["tests"]:
            advance()
            if not verify():
                return 1
    except (EOFError, KeyboardInterrupt):
        ui.say("\n  Stopped part-way. Run this again to carry on where it left off.")
        return 1

    summary(llama=llama, weights=weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
