"""Dictation and reading aloud: a clip in, or a reply out.

Two directions, two entirely different installs, and they are reported
separately because having one says nothing about having the other.

**Dictation.** The transcript is not sent anywhere. It goes back to the browser
and lands in the message box, where it can be read and corrected before
anything is done with it. That is not politeness - whisper invents words when
it hears no speech, so a dictated message that went straight to the agent would
sometimes carry a sentence nobody said. `speech/whisper.py` has the
measurements.

Nothing here trusts the client: the extension must be one whisper.cpp reads,
the size cap is enforced while reading rather than after, and the clip is
written to a temporary file that is deleted whether or not transcription
worked. Clips are not kept - a recording of somebody's voice is the most
personal thing this application handles, and the only reason to keep one would
be to debug this route.

**Reading aloud.** The WAV is returned as the response body and never written
to disk, so it exists for as long as the request does. The voice behind it is
kept warm between utterances, which is the opposite of what dictation does and
is argued from measurements in `speech/piper.py`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import Response

from api.deps import get_runtime
from api.runtime import Runtime
from api.schemas import SpeakRequest, SpeechOut, SpeechStatusOut
from speech import WhisperError, probe, transcribe
from speech.piper import VoiceError, available as voice_available, find_voice

router = APIRouter(tags=["speech"])

# What whisper.cpp's own help output lists as readable. The browser sends WAV
# because it is the one format every browser can produce without a library,
# but a file dropped in from elsewhere may be any of these.
ALLOWED_SUFFIXES = frozenset({".wav", ".mp3", ".ogg", ".flac"})

# 25 MB is about 13 minutes of the 16 kHz mono PCM the browser sends, which is
# far more than anybody dictates into a message box in one go.
MAX_BYTES = 25 * 1024 * 1024

CHUNK = 64 * 1024


@router.get("/speech", response_model=SpeechStatusOut)
def speech_status(runtime: Runtime = Depends(get_runtime)):
    """Whether dictation and reading aloud can run at all.

    The UI asks this once and hides whichever button has nothing behind it,
    rather than offering one that fails the moment it is pressed. The two
    halves are independent installs and are reported separately: a whisper
    build and a Piper voice have nothing to do with each other.
    """
    info = probe()
    if info is None:
        listening = {
            "available": False,
            "detail": (
                "Speech to text needs a whisper.cpp build in vendor/whisper "
                "and a ggml-*.bin model in whisper/ or weights/."
            ),
        }
    else:
        listening = {
            "available": True,
            "model": info.model_name,
            "detail": info.summary,
        }

    configured = runtime.effective_config().piper_voice
    if voice_available(configured):
        found = find_voice(configured)
        speaking = {
            "voice_available": True,
            "voice": Path(found).stem,
            "voice_detail": f"Piper voice {Path(found).stem} at {found}",
        }
    else:
        speaking = {
            "voice_available": False,
            "voice_detail": (
                "Reading aloud needs `pip install piper-tts` and a voice - an "
                ".onnx with its .onnx.json beside it - in tts/."
            ),
        }

    return SpeechStatusOut(**listening, **speaking)


@router.post("/speech/speak")
def speak(body: SpeakRequest, runtime: Runtime = Depends(get_runtime)):
    """Read text aloud and return the audio.

    Returns the WAV itself rather than a path or a base64 field: the browser
    hands it straight to an `<audio>` element, and nothing in between needs to
    understand it. Nothing is written to disk - the audio exists for as long as
    the response does.

    The first call after an idle period pays about seven seconds to load the
    voice; the ones after it take under half a second. `speech/piper.py` has
    the measurements and why it is kept warm at all.
    """
    try:
        audio = runtime.voice.speak(body.text)
    except VoiceError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            # Same text, same audio, and a reply does not change once it is
            # written - so re-pressing the button costs nothing.
            "Cache-Control": "private, max-age=600",
            "Content-Length": str(len(audio)),
        },
    )


@router.post("/speech/transcribe", response_model=SpeechOut)
async def transcribe_clip(
    file: UploadFile = File(...),
    runtime: Runtime = Depends(get_runtime),
):
    """Transcribe one recorded clip and return the text."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"{suffix or 'That file'} is not audio whisper.cpp reads. "
            f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}.",
        )

    # A named temporary file rather than the workspace. The agent has no reason
    # to read this, and a recording of somebody speaking should not be left
    # lying in a folder that tools can list.
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    target = Path(handle.name)
    written = 0
    try:
        with handle:
            while chunk := await file.read(CHUNK):
                written += len(chunk)
                if written > MAX_BYTES:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"Longer than the {MAX_BYTES // (1024 * 1024)} MB this "
                        f"accepts. Record in shorter pieces.",
                    )
                handle.write(chunk)

        if written == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "That clip was empty.")

        config = runtime.effective_config()
        try:
            text = transcribe(
                target,
                command=getattr(config, "whisper_cmd", ""),
                model=getattr(config, "whisper_model", ""),
                threads=getattr(config, "whisper_threads", 4),
            )
        except WhisperError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)
            ) from None
    finally:
        # Whether it worked or not. The clip has served its whole purpose by
        # the time this returns.
        target.unlink(missing_ok=True)

    return SpeechOut(text=text, bytes_received=written)
