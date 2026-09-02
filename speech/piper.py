"""Owning the Piper voice process: start it late, stop it early.

`Voice` is the only thing that knows the voice model exists. It starts the
worker on first use, keeps it while replies keep being read aloud, and shuts it
down once it has been idle - so a machine sitting at a chat prompt with nobody
listening holds no voice at all.

The idle shutdown mirrors `Embedder.unload_if_idle` and `ModelManager.
unload_idle`, and is driven by the same sweeper in `api/main.py`. That is
deliberate: there is one answer in this project to "a model is resident and
nobody is using it", and this is it.

**This is the opposite of what speech-to-text does, on purpose.** Whisper is a
subprocess per clip that holds nothing between times, because almost all of its
cost is loading a model and clips are occasional. Piper is kept warm because
its numbers say something different - measured here with
`en_US-lessac-medium`:

                        cold                 warm
    21 words            8.8 s                1.37 s
    42 words           10.2 s                2.52 s
    84 words           15.5 s                5.21 s

About 7.3 s of fixed cost against 0.07 s a word, and six of those seconds are
loading the voice. Warm, it produces speech at 0.22x realtime - four and a half
times faster than anybody can listen to it - so the wait is the load and
nothing else. Paying it before every spoken reply would be seven seconds of
silence for a one-line answer; paying it once a session costs 175 MB, measured,
and the sweeper gives that back.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folders searched for a voice, in order. `TTS/` is where the ones on this
# machine were put; `tts/` is the same folder on a case-sensitive filesystem,
# and both are checked so a repository cloned onto Linux still finds them.
#
# Note there is deliberately no `tts` Python package: a data directory sharing
# a name with an importable one shadows it, which is the trap `models/` versus
# `weights/` already records. Text-to-speech lives here, beside whisper.
VOICE_DIRS = ("tts", "TTS", "voices")

# A Piper voice is an .onnx beside an .onnx.json. The .json is not optional -
# it carries the phoneme map and the sample rate - so a voice without one is
# not a voice.
VOICE_SUFFIX = ".onnx"

# Loading is an onnxruntime session over a 63 MB model, measured at 6.1 s here
# and slower on a cold page cache. Generous rather than snappy, so a healthy
# worker is never reported as broken.
START_TIMEOUT = 180.0

# Warm, 168 words took 23 s. This is the bound on a whole answer read aloud,
# not on a sentence, so it is generous - but bounded, so a wedged worker
# surfaces as an error instead of a hang.
REQUEST_TIMEOUT = 300.0

# How much of the worker's stderr to keep for error messages.
STDERR_LINES = 40


class VoiceError(Exception):
    """The voice could not be loaded, or could not speak."""


def find_voice(configured: str = "") -> str:
    """Locate a Piper voice, or return "".

    A configured path wins outright. Otherwise the folders above are searched
    in order and the first voice with a config beside it is taken.
    """
    if configured:
        candidate = Path(configured).expanduser()
        if _is_voice(candidate):
            return str(candidate)
        return ""

    for folder in VOICE_DIRS:
        directory = PROJECT_ROOT / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if _is_voice(path):
                return str(path)
    return ""


def _is_voice(path: Path) -> bool:
    """An .onnx with its .json beside it."""
    try:
        if not path.is_file() or path.suffix.lower() != VOICE_SUFFIX:
            return False
        return path.with_suffix(path.suffix + ".json").is_file()
    except OSError:
        return False


def available(configured: str = "") -> bool:
    """Whether reading aloud can work at all.

    Both halves are needed and they fail differently: the package may be
    missing because nobody installed it, and the voice because nobody
    downloaded one.
    """
    try:
        import piper  # noqa: F401
    except ImportError:
        return False
    return bool(find_voice(configured))


class Voice:
    """A lazily-started, idle-stopped Piper voice in a child process."""

    def __init__(
        self,
        *,
        voice: str = "",
        idle_seconds: float = 300.0,
        python_executable: str | None = None,
    ) -> None:
        self.configured = voice
        self.idle_seconds = max(0.0, float(idle_seconds))
        self._python = python_executable or sys.executable

        self._process: subprocess.Popen | None = None
        self._replies: queue.Queue[str | None] = queue.Queue()
        self._stderr: list[str] = []
        self._rate = 0
        self._name = ""
        self._last_used = 0.0
        # One request at a time: the protocol is a single pipe with no request
        # ids, so two callers interleaving would read each other's replies.
        self._lock = threading.RLock()

    # --- state ---

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    @property
    def sample_rate(self) -> int:
        return self._rate

    @property
    def name(self) -> str:
        """The voice's short name, once it has been loaded."""
        if self._name:
            return self._name
        found = find_voice(self.configured)
        return Path(found).stem if found else ""

    def idle_for(self) -> float:
        """Seconds since the last request, or 0 when not loaded."""
        with self._lock:
            if not self.loaded or not self._last_used:
                return 0.0
            return time.time() - self._last_used

    # --- lifecycle ---

    def ensure_loaded(self) -> None:
        with self._lock:
            if self.loaded:
                return
            self._start()

    def unload(self) -> bool:
        """Stop the worker and give its 175 MB back. Safe to call any time."""
        with self._lock:
            process = self._process
            self._process = None
            if process is None:
                return False

            try:
                if process.poll() is None and process.stdin is not None:
                    # Ask first. A clean exit closes the onnxruntime session
                    # rather than leaving the model file locked, which is the
                    # sort of thing Windows remembers.
                    try:
                        process.stdin.write('{"op": "shutdown"}\n')
                        process.stdin.flush()
                    except (OSError, ValueError):
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
            except OSError:
                pass
            finally:
                for stream in (process.stdin, process.stdout, process.stderr):
                    try:
                        if stream is not None:
                            stream.close()
                    except OSError:
                        pass

            self._drain_replies()
            self._last_used = 0.0
            return True

    def unload_if_idle(self) -> bool:
        """Stop the worker when it has been unused for `idle_seconds`."""
        if self.idle_seconds <= 0:
            return False
        with self._lock:
            if not self.loaded or self.idle_for() < self.idle_seconds:
                return False
            return self.unload()

    # --- speaking ---

    def speak(self, text: str) -> bytes:
        """Turn text into a WAV. Starts the worker if it is not running."""
        cleaned = (text or "").strip()
        if not cleaned:
            raise VoiceError("There was nothing to say.")

        with self._lock:
            self.ensure_loaded()
            reply = self._request({"op": "speak", "text": cleaned})
            self._last_used = time.time()

        try:
            return base64.b64decode(reply["wav"])
        except (KeyError, ValueError) as exc:
            raise VoiceError(f"The voice returned unreadable audio: {exc}") from None

    # --- the process ---

    def _start(self) -> None:
        found = find_voice(self.configured)
        if not found:
            raise VoiceError(
                "No Piper voice was found. Put an .onnx and its .onnx.json in "
                f"{PROJECT_ROOT / 'tts'}."
            )

        environment = dict(os.environ)
        environment["PIPER_VOICE"] = found
        # Unbuffered, or replies sit in the child's pipe buffer and every
        # request looks like a timeout.
        environment["PYTHONUNBUFFERED"] = "1"

        try:
            process = subprocess.Popen(
                [self._python, "-m", "speech.voice_worker"],
                cwd=str(PROJECT_ROOT),
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise VoiceError(f"Could not start the voice worker: {exc}") from None

        self._process = process
        self._replies = queue.Queue()
        self._stderr = []

        threading.Thread(
            target=self._read_replies, args=(process,), daemon=True,
            name="piper-stdout",
        ).start()
        threading.Thread(
            target=self._read_stderr, args=(process,), daemon=True,
            name="piper-stderr",
        ).start()

        # The worker announces itself once the voice is loaded, so the first
        # reply is the handshake rather than something asked for.
        try:
            reply = self._await_reply(START_TIMEOUT)
        except VoiceError:
            self.unload()
            raise
        self._rate = int(reply.get("rate", 0))
        self._name = str(reply.get("voice", ""))
        self._last_used = time.time()

    def _read_replies(self, process: subprocess.Popen) -> None:
        """Pump the worker's stdout into a queue. None marks the end."""
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    self._replies.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._replies.put(None)

    def _read_stderr(self, process: subprocess.Popen) -> None:
        """Keep the tail of the worker's stderr, for error messages."""
        try:
            if process.stderr is not None:
                for line in process.stderr:
                    self._stderr.append(line.rstrip())
                    del self._stderr[:-STDERR_LINES]
        except (OSError, ValueError):
            pass

    def _request(self, payload: dict, timeout: float = REQUEST_TIMEOUT) -> dict:
        process = self._process
        if process is None or process.stdin is None:
            raise VoiceError("The voice worker is not running.")

        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
        except (OSError, ValueError):
            raise VoiceError(
                f"The voice worker stopped accepting work.{self._diagnostics()}"
            ) from None
        return self._await_reply(timeout)

    def _await_reply(self, timeout: float) -> dict:
        try:
            line = self._replies.get(timeout=timeout)
        except queue.Empty:
            self.unload()
            raise VoiceError(
                f"The voice worker did not answer within {timeout:.0f}s. "
                f"It has been stopped; try again."
            ) from None

        if line is None:
            self.unload()
            raise VoiceError(f"The voice worker exited.{self._diagnostics()}")

        try:
            reply = json.loads(line)
        except ValueError:
            raise VoiceError(
                f"Unreadable reply from the voice worker: {line[:200]!r}"
            ) from None

        if not reply.get("ok"):
            raise VoiceError(str(reply.get("error", "The voice failed.")))
        return reply

    def _drain_replies(self) -> None:
        while True:
            try:
                self._replies.get_nowait()
            except queue.Empty:
                return

    def _diagnostics(self) -> str:
        """The tail of the worker's stderr, when there is any."""
        tail = [line for line in self._stderr if line.strip()]
        if not tail:
            return ""
        return "\n\nThe worker said:\n" + "\n".join(tail[-12:])
