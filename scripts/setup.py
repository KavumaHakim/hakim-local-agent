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
SCRIPTS = ROOT / "scripts"
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
        ui.ok("already made, reusing it")
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


def remember_server(path: Path) -> None:
    """Write a llama-server path into the machine's own preferences.

    Not into models.json: that file is in version control, and a path from
    somebody's laptop is wrong on every other machine. data/models.local.json
    is git-ignored and already the place the app keeps per-machine choices.
    """
    try:
        sys.path.insert(0, str(ROOT))
        from models.preferences import ModelPreferences

        preferences = ModelPreferences.load(ROOT / "data")
        preferences.set_server_exe(str(path))
    except Exception as exc:  # noqa: BLE001 - remembering is a convenience
        ui.warn(f"could not save the path ({exc}); it will still work this time")


def verify_server(path: Path) -> str:
    """Check a path really is a llama-server before believing it."""
    if not path.exists():
        raise ValueError(f"there is nothing at {path}")
    if path.is_dir():
        # People paste the folder more often than the binary.
        for name in ("llama-server.exe", "llama-server"):
            for found in sorted(path.rglob(name)):
                if found.is_file():
                    path = found
                    break
            else:
                continue
            break
        else:
            raise ValueError(f"no llama-server anywhere under {path}")
    if not path.is_file():
        raise ValueError(f"{path} is not a file")

    try:
        import get_llama

        version = get_llama.verify(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{path.name} would not run: {exc}") from None
    return version


def ask_for_server() -> Path | None:
    """Let someone point at a llama-server they already have."""
    while True:
        typed = ui.ask_text("Where is it? (blank to give up)")
        if not typed:
            return None
        candidate = Path(typed.strip().strip('"').strip("'")).expanduser()
        try:
            verify_server(candidate)
        except ValueError as exc:
            ui.warn(str(exc))
            continue

        if candidate.is_dir():
            for name in ("llama-server.exe", "llama-server"):
                for found in sorted(candidate.rglob(name)):
                    if found.is_file():
                        candidate = found
                        break
                else:
                    continue
                break
        return candidate


def check_llama_server(*, download: bool) -> bool:
    existing = find_llama_server()
    if existing is not None:
        ui.ok("you already have one")
        ui.note(str(existing))
        return True

    if not download:
        # Still worth asking where it is - "do not download" is not the same
        # as "do not help".
        if ui.interactive():
            ui.warn("I could not find llama.cpp anywhere.")
            chosen = ask_for_server()
            if chosen is not None:
                remember_server(chosen)
                ui.ok(str(chosen))
                ui.note("remembered in data/models.local.json")
                return True
        ui.warn("not installed, and it was not asked for.")
        ui.note("https://github.com/ggml-org/llama.cpp/releases")
        return False

    if ui.interactive():
        ui.warn("I could not find llama.cpp on this machine.")
        pick = ui.choose(
            "What would you like to do?",
            [
                ("Download it for me", "About 18 MB, straight from the project."),
                ("I already have it", "Tell me where, and I will remember."),
                ("Skip for now", "Nothing local will run until it is sorted."),
            ],
        )
        if pick == 1:
            chosen = ask_for_server()
            if chosen is not None:
                remember_server(chosen)
                ui.ok(str(chosen))
                ui.note("remembered in data/models.local.json")
                return True
            ui.note("nothing given, so nothing was changed")
            return False
        if pick == 2:
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

    ui.warn("nothing here yet - a model is the one thing you choose yourself")
    ui.note("Any .gguf works. Drop one in and it is measured automatically.")
    ui.note("On 8 GB of RAM, start with a 2-3B instruct model at Q4_K_M.")
    ui.note("The README explains why, under 'Choosing a model'.")
    return False


def check_speech(*, download: bool) -> dict:
    """Set up dictation and reading aloud, or say what is missing.

    Three pieces that fail independently, so they are reported independently:
    a whisper.cpp build and a speech model for listening, and a Piper voice
    for talking. Having one says nothing about having the others, and a person
    who ends up with two of the three should be told which one to go and get.

    Neither feature is required, and neither is switched on anywhere - the
    microphone and the speaker are simply not drawn when what they need is
    absent. So a failure here is a note, never a stopped install.
    """
    # The project root, because this runs from scripts/ and under whichever
    # Python launched it - not necessarily the venv, and possibly before the
    # venv exists at all. Both modules are stdlib-only for exactly this
    # reason: asking them where things are must not require the install to
    # have finished.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from speech.piper import find_voice
        from speech.whisper import find_model, find_whisper
    except ImportError as exc:
        # Never fatal. Neither feature is required, and a setup that died here
        # would have died on the optional part after doing all the real work.
        ui.warn(f"could not check the voice features: {exc}")
        return {}

    found = {
        "whisper": bool(find_whisper()),
        "model": bool(find_model()),
        "voice": bool(find_voice()),
    }
    if all(found.values()):
        ui.ok("dictation and reading aloud are both ready")
        return found

    if not download:
        _speech_notes(found)
        return found

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    try:
        import get_speech
    except ImportError as exc:
        ui.warn(f"could not load the fetcher: {exc}")
        _speech_notes(found)
        return found

    # 220 MB between them, so it is worth saying before it starts rather than
    # after somebody has watched a bar for four minutes.
    ui.note("About 220 MB: an 8 MB build, a 148 MB model, a 63 MB voice.")

    for key, label, fetch in (
        ("whisper", "whisper.cpp", get_speech.install_whisper),
        ("model", "the speech model", get_speech.install_model),
        ("voice", "the voice", get_speech.install_voice),
    ):
        if found[key]:
            continue
        try:
            fetch()
            found[key] = True
            ui.ok(label)
        except Exception as exc:  # noqa: BLE001 - never fatal, always reported
            # One failing piece must not take the other two with it: a machine
            # with no whisper build can still read replies aloud.
            ui.warn(f"{label}: {exc}")

    _speech_notes(found)
    return found


def _speech_notes(found: dict) -> None:
    """Say which half works and what the other one needs."""
    listening = found["whisper"] and found["model"]
    if listening:
        ui.ok("dictation is ready")
    else:
        missing = "a whisper.cpp build" if not found["whisper"] else "a speech model"
        ui.note(f"Dictation needs {missing}: python scripts/get_speech.py")

    if found["voice"]:
        ui.ok("reading aloud is ready")
    else:
        ui.note(
            "Reading aloud needs a Piper voice: "
            "python scripts/get_speech.py --what voice"
        )


def make_env_file() -> None:
    """Put a .env in place, from the committed example.

    Copied rather than generated, so there is one description of the settings
    instead of two that drift. `.env.example` carries names and no values, so
    copying it commits nothing and enables nothing.
    """
    target = ROOT / ".env"
    if target.is_file():
        ui.ok("you already have a .env - left exactly as it was")
        return

    example = ROOT / ".env.example"
    if not example.is_file():
        ui.warn("no .env.example to copy; hosted models will need one by hand")
        return

    shutil.copyfile(example, target)
    ui.ok("made a .env from the example")
    ui.note("no API keys needed - everything runs locally without them")


# The hosted models this project ships an entry for, and the variable each
# one's key lives in. Read from models.json rather than hard-coded, so adding
# a provider there is enough to have setup ask about it.
def hosted_providers() -> list[tuple[str, str]]:
    """(label, environment variable) for each hosted model with a key name."""
    registry = ROOT / "models.json"
    try:
        raw = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    found = []
    for entry in raw.get("models", []):
        variable = entry.get("api_key_env")
        if variable:
            found.append((entry.get("label") or entry.get("key", variable), variable))
    return found


def existing_keys() -> set[str]:
    """Variables already answered, in the environment or in .env."""
    answered = {name for name in os.environ if os.environ.get(name)}
    target = ROOT / ".env"
    if target.is_file():
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if value.strip():
                answered.add(name.strip())
    return answered


def write_key(variable: str, value: str) -> None:
    """Put one key into .env, replacing any commented placeholder for it."""
    target = ROOT / ".env"
    lines = (
        target.read_text(encoding="utf-8").splitlines()
        if target.is_file()
        else []
    )

    replaced = False
    for index, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip()
        if stripped.startswith(f"{variable}="):
            lines[index] = f"{variable}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{variable}={value}")

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ask_for_keys(*, ask: bool) -> None:
    """Offer to save hosted-model API keys.

    Entirely optional: the agent is fully local without any of them, which is
    said out loud because a setup script asking for API keys otherwise reads
    like something is required that is not.

    Keys are read without echo and never printed back - not even partially.
    They go into .env, which is git-ignored.
    """
    providers = hosted_providers()
    if not providers or not ask or not ui.interactive():
        return

    already = existing_keys()
    wanted = [(label, name) for label, name in providers if name not in already]
    if not wanted:
        ui.ok("hosted model keys are already set")
        return

    ui.say(
        f"\n  {ui.DIM}These are optional. Everything works locally without "
        f"them;{ui.RESET}"
    )
    ui.say(
        f"  {ui.DIM}a key just makes that provider's model selectable too."
        f"{ui.RESET}"
    )

    if not ui.confirm("Add a hosted model API key?", default=False):
        ui.note("skipped - you can add them to .env whenever you like")
        return

    for label, variable in wanted:
        value = ui.ask_secret(f"{label} ({variable})")
        if not value:
            ui.note(f"{variable} left empty")
            continue
        write_key(variable, value)
        # Deliberately not echoed, not even the last few characters.
        ui.ok(f"{variable} saved to .env")


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
        "speech": arguments.with_speech,
        "tests": not arguments.skip_tests,
    }

    if scripted:
        return choices

    ui.say(
        f"\n  Everything goes in {ui.BOLD}.venv{ui.RESET} inside this folder. "
        f"Nothing else on your\n  machine is touched, and deleting the folder "
        f"undoes all of it."
    )

    options = [
        {
            "label": "Get llama.cpp for me",
            "note": "18 MB. It is the engine that actually runs your models.",
            "on": choices["llama"],
            "key": "llama",
        },
        {
            "label": "Let me talk to it, and hear it back",
            "note": "220 MB. Dictate a message, and read any answer aloud.",
            "on": choices["speech"],
            "key": "speech",
        },
        {
            "label": "Let it search my documents",
            "note": "2 GB and slow to install. You can add this later.",
            "on": choices["rag"],
            "key": "rag",
        },
        {
            "label": "Build the web interface for production",
            "note": "Most people want the development mode instead. Leave this off.",
            "on": choices["web_build"],
            "key": "web_build",
        },
        {
            "label": "Check it works when you are done",
            "note": "Runs the tests. About a minute, and worth it.",
            "on": choices["tests"],
            "key": "tests",
        },
    ]
    for item in ui.toggle("What would you like?", options):
        choices[item["key"]] = item["on"]
    return choices


def _size(path: Path) -> str:
    try:
        return f"{path.stat().st_size / 1e9:.1f} GB"
    except OSError:
        return "?"


def report(*, llama: bool, weights: bool, speech: dict, choices: dict) -> None:
    """Say where everything went, what was done, and how to start it.

    Printed in full every time rather than only when something is missing. A
    setup script that finishes silently leaves someone with a working install
    and no idea what it did to their machine or what to type next, which is
    most of the reason people distrust them.
    """
    launcher = "start.bat" if os.name == "nt" else "./start.sh"
    activate = (
        r".venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate"
    )

    ui.say()
    ui.rule("where everything is")
    ui.say()
    ui.field("project", str(ROOT))
    ui.field("python", str(venv_python()))

    server = find_llama_server()
    ui.field("llama.cpp", str(server) if server else "not installed yet")

    models = sorted(WEIGHTS.glob("*.gguf"))
    if models:
        ui.field("models", f"{WEIGHTS}")
        for path in models:
            ui.say(f"   {' ' * 16} {ui.DIM}- {path.name} ({_size(path)}){ui.RESET}")
    else:
        ui.field("models", f"{WEIGHTS} {ui.DIM}(empty){ui.RESET}")

    if speech.get("whisper") or speech.get("model"):
        ui.field("dictation", str(ROOT / "whisper"))
        ui.say(
            f"   {' ' * 16} {ui.DIM}the build is in vendor/whisper{ui.RESET}"
        )
    if speech.get("voice"):
        voices = sorted((ROOT / "tts").glob("*.onnx"))
        ui.field("voice", str(ROOT / "tts"))
        for path in voices:
            ui.say(f"   {' ' * 16} {ui.DIM}- {path.stem}{ui.RESET}")

    ui.field("settings", str(ROOT / ".env"))
    ui.field("your data", str(ROOT / "data"))
    ui.say(
        f"   {' ' * 16} {ui.DIM}- chat_history.db, and anything the agent "
        f"remembers{ui.RESET}"
    )
    ui.say(
        f"   {' ' * 16} {ui.DIM}- models.local.json, your own model choices"
        f"{ui.RESET}"
    )
    ui.field("workspace", str(ROOT))
    ui.say(
        f"   {' ' * 16} {ui.DIM}the only folder the file tools may touch; "
        f"changeable in the UI{ui.RESET}"
    )

    ui.say()
    ui.rule("what was set up")
    ui.say()
    web = (ROOT / "web" / "node_modules").is_dir()
    built = (ROOT / "web" / "dist").is_dir()
    keys = [name for _, name in hosted_providers() if name in existing_keys()]

    lines = [
        ("Python packages", "installed into .venv, nothing system-wide"),
        (
            "Document search",
            "installed" if choices.get("rag") else "skipped - add it with --with-rag",
        ),
        (
            "Web interface",
            ("built for production" if built else "ready for development mode")
            if web
            else "skipped - npm was not found",
        ),
        (
            "llama.cpp",
            "ready" if llama else "still needed - nothing local runs without it",
        ),
        (
            "A model",
            f"{len(models)} found" if models else "still needed - your choice to make",
        ),
        (
            "Dictation",
            "ready - press the microphone"
            if speech.get("whisper") and speech.get("model")
            else "skipped - add it with scripts/get_speech.py",
        ),
        (
            "Reading aloud",
            "ready - press the speaker on any answer"
            if speech.get("voice")
            else "skipped - add a voice with scripts/get_speech.py --what voice",
        ),
        (
            "Hosted models",
            f"{len(keys)} key(s) saved" if keys else "none - everything runs locally",
        ),
    ]
    for name, state in lines:
        ui.field(name, state)

    ui.say()
    ui.rule("how to start it")
    ui.say()

    if llama and weights:
        ui.say(f"   {ui.GREEN}Everything it needs is in place.{ui.RESET}")
    elif llama and not weights:
        ui.say(f"   {ui.BOLD}One thing left: a model.{ui.RESET}")
        ui.say(
            f"   Put any {ui.BOLD}.gguf{ui.RESET} into {ui.BOLD}weights/{ui.RESET} "
            f"and it is found on its own -"
        )
        ui.say("   nothing to configure, it is measured from the file itself.")
        ui.say(
            f"   {ui.DIM}The README's 'Choosing a model' section has the "
            f"arithmetic.{ui.RESET}"
        )
    else:
        ui.say(f"   {ui.BOLD}Not runnable yet:{ui.RESET}")
        if not llama:
            ui.say("     llama.cpp is missing - run this again to fetch it")
        if not weights:
            ui.say("     there is no .gguf in weights/")

    ui.say()
    ui.say(f"   {ui.ACCENT}{ui.BOLD}{launcher}{ui.RESET}   starts both servers "
           f"and opens the browser")
    ui.say()
    ui.say(f"   {ui.DIM}The web UI is at {ui.RESET}http://127.0.0.1:5173")
    ui.say(f"   {ui.DIM}The API is at    {ui.RESET}http://127.0.0.1:8000"
           f"{ui.DIM}  (docs at /docs){ui.RESET}")
    ui.say(
        f"   {ui.DIM}Both bind to 127.0.0.1 only, and are meant to stay that "
        f"way.{ui.RESET}"
    )
    ui.say()
    ui.say(f"   {ui.DIM}By hand instead, in two terminals:{ui.RESET}")
    ui.note(activate)
    ui.note("python -m uvicorn api.main:app --host 127.0.0.1 --port 8000")
    ui.note("npm --prefix web run dev")
    ui.say()
    ui.say(f"   {ui.DIM}Or without a browser at all:{ui.RESET}")
    ui.note("python main.py")

    ui.say()
    ui.rule("worth knowing")
    ui.say()
    ui.say(
        f"   {ui.DIM}Models load when first used, not at startup, and unload "
        f"when idle.{ui.RESET}"
    )
    ui.say(
        f"   {ui.DIM}The risky tools - shell, Python, file writes - are all off "
        f"until you{ui.RESET}"
    )
    ui.say(
        f"   {ui.DIM}switch them on in the sidebar, and each explains what it "
        f"allows.{ui.RESET}"
    )
    ui.say(
        f"   {ui.DIM}Nothing leaves this machine unless you pick a hosted "
        f"model, and it{ui.RESET}"
    )
    ui.say(f"   {ui.DIM}asks first when it would.{ui.RESET}")
    ui.say()
    ui.say(f"   {ui.DIM}Run this script again any time - it is safe to repeat "
           f"and will{ui.RESET}")
    ui.say(f"   {ui.DIM}only do what is still missing.{ui.RESET}")
    ui.say()


def step_names(choices: dict) -> list[str]:
    names = [
        "Checking your Python",
        "Making a private environment",
        "Installing the Python side",
        "Installing the web interface",
        "Fetching llama.cpp" if choices["llama"] else "Looking for llama.cpp",
        "Looking for a model",
        "Setting up the voice" if choices["speech"] else "Checking the voice",
        "Writing your configuration",
    ]
    if choices["tests"]:
        names.append("Making sure it all works")
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
        "--with-speech",
        action="store_true",
        help="also fetch dictation and a voice (about 220 MB)",
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

    ui.banner("HAKIM", "a local agent that runs on your own machine")
    ui.say()
    ui.say(f"  {ui.DIM}{ROOT}{ui.RESET}")

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
    speech: dict = {}

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
        speech = check_speech(download=choices["speech"])

        advance()
        make_env_file()
        ask_for_keys(ask=not arguments.yes)

        if choices["tests"]:
            advance()
            if not verify():
                return 1
    except (EOFError, KeyboardInterrupt):
        ui.say("\n  Stopped part-way. Run this again to carry on where it left off.")
        return 1

    report(llama=llama, weights=weights, speech=speech, choices=choices)
    # --yes means unattended, so it must not hold at the end either: someone
    # scripting this in a terminal would otherwise wait forever on a keypress
    # they never asked to be prompted for.
    if not arguments.yes:
        ui.pause("All done - press Enter to close")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
