"""The Piper voice, running in its own process.

Run as `python -m speech.voice_worker`. Not imported by the application: the
parent (`speech.piper.Voice`) spawns it and talks to it over stdin/stdout.

**Why a separate process**, and it is the same answer `rag/worker.py` gives.
Loading the voice in the API worker costs 175 MB measured, and `del voice`
would not give it back - freeing the weights does not free the onnxruntime
allocations behind them. A child process returns every byte the moment it
exits, which on 8 GB is the difference that matters.

**Why keep it loaded at all**, which is the opposite of what whisper does.
Measured on this machine with `en_US-lessac-medium`:

    cold, per utterance     ~7.3 s before any sound, plus 0.07 s a word
    warm                    ~1.4 s for a sentence, then 0.22x realtime

Six of those seven seconds are loading the voice. Paying it once at the start
of a session is worth 175 MB; paying it before every spoken reply is seven
seconds of silence for a one-line answer. So it loads late, stays while it is
being used, and stops when it is not - the same bargain the embedder strikes.

**Protocol.** One JSON object per line, both directions.

    -> {"op": "ping"}
    <- {"ok": true, "rate": 22050, "voice": "en_US-lessac-medium"}

    -> {"op": "speak", "text": "..."}
    <- {"ok": true, "rate": 22050, "samples": 108800, "wav": "<base64>"}

    <- {"ok": false, "error": "..."}

Audio comes back as a complete base64 WAV rather than raw samples, because the
browser plays a WAV and nothing in between needs to understand it.

**stdout is the protocol channel and nothing else may touch it.** onnxruntime
and espeak-ng both write warnings, and a single stray line would desynchronise
the stream. So the real stdout is taken aside at startup and `sys.stdout` is
pointed at stderr, where anything a library prints becomes diagnostics rather
than corruption.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import wave

# The real stdout, taken aside by `_claim_stdout()` when this runs as a
# worker. It stays None on import, because the parent imports this module for
# its constants and must keep its own stdout intact.
_CHANNEL = None

# Longer than anybody dictates a reply into a message box, and long enough for
# a whole answer read aloud. Past this the text is almost certainly a mistake -
# a whole document pasted in - and synthesising it would take minutes.
MAX_CHARACTERS = 8000


def _claim_stdout() -> None:
    """Take stdout for the protocol and point everything else at stderr."""
    global _CHANNEL
    _CHANNEL = sys.stdout
    sys.stdout = sys.stderr


def _reply(payload: dict) -> None:
    assert _CHANNEL is not None, "_claim_stdout() has not run"
    _CHANNEL.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _CHANNEL.flush()


def _fail(message: str) -> None:
    _reply({"ok": False, "error": message})


def to_wav(frames: bytes, rate: int) -> bytes:
    """Wrap raw 16-bit mono samples in a RIFF container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return buffer.getvalue()


def main() -> int:
    _claim_stdout()

    voice_path = os.environ.get("PIPER_VOICE", "")
    if not voice_path:
        _fail("PIPER_VOICE was not set.")
        return 1

    try:
        from piper import PiperVoice
    except ImportError as exc:
        _fail(f"piper-tts is not installed: {exc}")
        return 1

    try:
        voice = PiperVoice.load(voice_path)
    except Exception as exc:  # noqa: BLE001 - any load failure is reported, not raised
        _fail(f"Could not load {voice_path}: {exc}")
        return 1

    rate = int(voice.config.sample_rate)
    name = os.path.splitext(os.path.basename(voice_path))[0]
    _reply({"ok": True, "rate": rate, "voice": name})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            _fail("That was not JSON.")
            continue

        operation = request.get("op")
        if operation == "shutdown":
            return 0
        if operation == "ping":
            _reply({"ok": True, "rate": rate, "voice": name})
            continue
        if operation != "speak":
            _fail(f"Unknown op {operation!r}.")
            continue

        text = str(request.get("text") or "").strip()
        if not text:
            _fail("There was nothing to say.")
            continue
        if len(text) > MAX_CHARACTERS:
            _fail(
                f"{len(text):,} characters is past the {MAX_CHARACTERS:,} this "
                f"reads in one go."
            )
            continue

        try:
            frames = b"".join(
                chunk.audio_int16_bytes for chunk in voice.synthesize(text)
            )
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            _fail(f"Could not synthesise that: {exc}")
            continue

        audio = to_wav(frames, rate)
        _reply(
            {
                "ok": True,
                "rate": rate,
                "samples": len(frames) // 2,
                "wav": base64.b64encode(audio).decode("ascii"),
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
