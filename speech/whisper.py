"""Speech to text by shelling out to whisper.cpp.

**No resident server, deliberately.** Every other model in this project runs as
a long-lived `llama-server`, so the obvious thing would be to run
`whisper-server` the same way. Measured here with `ggml-base.en.bin`, it should
not be:

    2 s of audio      4.9 s
    10 s of audio     5.2 s
    30 s of audio     4.1 s

The wall clock barely moves with the length of the clip, because almost all of
it is loading a 148 MB model - whisper decodes in 30-second windows, so a short
clip and a long one are the same single window of work. A resident server would
save about four seconds a clip and hold roughly 200 MB for as long as it lived.
On a machine with 8 GB, where a chat model is already the thing competing for
RAM, four seconds is the cheaper side of that trade.

It also means there is nothing to supervise, nothing to reconcile after a
crash, and no second idle timeout to reason about.

**Whisper invents speech when there is none.** Two seconds of digital silence
transcribes as " you"; a synthetic tone becomes " (dramatic music)". This is
whisper doing what it was trained to do, not a bug, and it matters here because
the output goes straight into somebody's message box - press record, say
nothing, and you would find a word you did not say. `clean_transcript` strips
the bracketed annotations, and `-sns` asks whisper to suppress non-speech
tokens in the first place.

VERIFIED against whisper.cpp's Windows x64 build and `ggml-base.en.bin`: the
invocation below returns the transcript alone on stdout, with the backend and
audio-decoding chatter on stderr, which is what makes parsing this safe.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the setup script would put a whisper.cpp build, checked before PATH for
# the same reason `find_server` checks `vendor/llama` first: somebody who put a
# build there meant to use it.
VENDOR = PROJECT_ROOT / "vendor" / "whisper"

# Folders searched for a `ggml-*.bin` model, in order. `whisper/` is where one
# belongs; `weights/` is checked too because it is the folder this project
# tells people to drop models into, so it is where the first one lands.
MODEL_DIRS = ("whisper", "weights", "models")

# Whisper models are `ggml-<name>.bin`. The GGUF discovery in models/ ignores
# them - it only reads `.gguf` - so the two never collide in `weights/`.
MODEL_NAME = re.compile(r"^ggml-.*\.bin$", re.IGNORECASE)

# Preferred smallest-first, because on this hardware the difference between
# base and small is seconds per clip and the accuracy gain is slight for
# dictation. An explicit path in the config beats all of it.
MODEL_PREFERENCE = ("tiny", "base", "small", "medium", "large")

DEFAULT_LANGUAGE = "en"

# Generous against the measured four seconds, because the first run after a
# reboot reads 148 MB off a cold disk. A clip that takes longer than this has
# gone wrong rather than being slow.
DEFAULT_TIMEOUT = 180.0

# Whisper's own annotations for things that are not speech. It emits these
# readily on silence, breathing, or background noise, and they are never
# something the person said.
_ANNOTATION = re.compile(r"[\[(\*][^\])\*]{0,60}[\])\*]")

# Left after the annotations go: whisper marks a silent clip like this.
_BLANK = re.compile(r"^\s*(blank_?audio|silence|inaudible)\s*$", re.IGNORECASE)


class WhisperError(Exception):
    """Whisper is missing, or could not transcribe the audio."""


@dataclass(frozen=True)
class WhisperInfo:
    """What was found on this machine."""

    path: str
    model: str

    @property
    def model_name(self) -> str:
        return Path(self.model).stem.replace("ggml-", "") if self.model else ""

    @property
    def summary(self) -> str:
        return f"whisper.cpp at {self.path}, model {self.model_name or 'missing'}"


def find_whisper(configured: str = "") -> str:
    """Locate `whisper-cli`, or return "".

    Order: what the config says, then the vendored build, then PATH.
    """
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate)
        found = shutil.which(configured)
        return found or ""

    for name in ("whisper-cli.exe", "whisper-cli", "main.exe", "main"):
        candidate = VENDOR / name
        if candidate.is_file():
            return str(candidate)

    for name in ("whisper-cli", "whisper"):
        found = shutil.which(name)
        if found:
            return found
    return ""


def find_model(configured: str = "") -> str:
    """Locate a whisper model file, or return "".

    A configured path wins outright. Otherwise the folders in `MODEL_DIRS` are
    searched in order, and within a folder the smallest useful model wins -
    on this hardware the seconds matter more than the last few points of
    accuracy for a dictated sentence.
    """
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate)
        return ""

    for folder in MODEL_DIRS:
        directory = PROJECT_ROOT / folder
        if not directory.is_dir():
            continue
        found = [path for path in sorted(directory.iterdir()) if _is_model(path)]
        if found:
            return str(_preferred(found))
    return ""


def _is_model(path: Path) -> bool:
    try:
        return path.is_file() and bool(MODEL_NAME.match(path.name))
    except OSError:
        return False


def _preferred(paths: list[Path]) -> Path:
    """The smallest model in `paths` by the usual whisper size names."""

    def rank(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        for index, size in enumerate(MODEL_PREFERENCE):
            if size in name:
                return (index, name)
        # An unrecognised name sorts last rather than first: it is more likely
        # to be a large or fine-tuned model than a tiny one.
        return (len(MODEL_PREFERENCE), name)

    return min(paths, key=rank)


def probe(command: str = "", model: str = "") -> WhisperInfo | None:
    """What is available, or None when speech to text cannot run.

    Returns None rather than raising, because "not installed" is a supported
    state: the UI hides the microphone instead of offering a button that fails
    the moment it is pressed.
    """
    path = find_whisper(command)
    if not path:
        return None
    found_model = find_model(model)
    if not found_model:
        return None
    return WhisperInfo(path=path, model=found_model)


def transcribe(
    audio: Path | str,
    *,
    command: str = "",
    model: str = "",
    language: str = DEFAULT_LANGUAGE,
    threads: int = 4,
    timeout: float = DEFAULT_TIMEOUT,
    runner=subprocess.run,
) -> str:
    """Turn an audio file into text.

    The path is made absolute first. That is the lesson `tools/tesseract.py`
    records in prose: a relative path resolved twice fails in a way no stub
    ever reproduces.
    """
    path = Path(audio).expanduser().resolve()
    if not path.is_file():
        raise WhisperError(f"No audio file at {path}.")

    binary = find_whisper(command)
    if not binary:
        raise WhisperError(
            "whisper-cli was not found. Put a whisper.cpp build in "
            f"{VENDOR}, or on PATH."
        )
    weights = find_model(model)
    if not weights:
        raise WhisperError(
            "No whisper model was found. Put a ggml-*.bin file in "
            f"{PROJECT_ROOT / 'whisper'} or {PROJECT_ROOT / 'weights'}."
        )

    arguments = [
        binary,
        "-m", weights,
        "-f", str(path),
        # Text only: no timestamps, no progress, nothing but what was said.
        "-nt",
        "-np",
        # Ask whisper not to emit its non-speech tokens. `clean_transcript`
        # still runs, because this reduces them rather than removing them.
        "-sns",
        "-l", language or DEFAULT_LANGUAGE,
        "-t", str(int(threads)),
    ]

    try:
        result = runner(
            arguments,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        raise WhisperError(
            f"Transcription took longer than {timeout:.0f}s and was stopped."
        ) from None
    except (OSError, subprocess.SubprocessError) as exc:
        raise WhisperError(f"Could not run whisper-cli: {exc}") from None

    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        raise WhisperError(
            f"whisper-cli failed: {detail[-1] if detail else 'no output'}"
        )

    return clean_transcript(result.stdout or "")


def clean_transcript(raw: str) -> str:
    """Strip whisper's non-speech annotations and tidy the whitespace.

    Whisper transcribes two seconds of digital silence as " you" and a tone as
    " (dramatic music)". The annotations are removable; the invented word is
    not, and is why the caller puts this in a box to be read before it is sent
    rather than sending it.
    """
    lines = []
    for line in raw.splitlines():
        without = _ANNOTATION.sub(" ", line)
        without = " ".join(without.split())
        if not without or _BLANK.match(without):
            continue
        lines.append(without)
    return " ".join(lines).strip()
