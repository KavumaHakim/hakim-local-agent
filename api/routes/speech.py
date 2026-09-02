"""Dictation: a recorded clip in, the words in it out.

The transcript is **not** sent anywhere. It goes back to the browser and lands
in the message box, where it can be read and corrected before anything is done
with it. That is not politeness - whisper invents words when it hears no
speech, so a dictated message that went straight to the agent would sometimes
carry a sentence nobody said. `speech/whisper.py` documents the measurements.

Nothing here trusts the client: the extension must be one whisper.cpp reads,
the size cap is enforced while reading rather than after, and the clip is
written to a temporary file that is deleted whether or not transcription
worked. Clips are not kept - a recording of somebody's voice is the most
personal thing this application handles, and the only reason to keep one would
be to debug this route.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.deps import get_runtime
from api.runtime import Runtime
from api.schemas import SpeechOut, SpeechStatusOut
from speech import WhisperError, probe, transcribe

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
def speech_status():
    """Whether dictation can run at all.

    The UI asks this once and hides the microphone if the answer is no, rather
    than offering a button that fails the moment it is pressed.
    """
    info = probe()
    if info is None:
        return SpeechStatusOut(
            available=False,
            detail=(
                "Speech to text needs a whisper.cpp build in vendor/whisper "
                "and a ggml-*.bin model in whisper/ or weights/."
            ),
        )
    return SpeechStatusOut(available=True, model=info.model_name, detail=info.summary)


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
