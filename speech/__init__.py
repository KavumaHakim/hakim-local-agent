"""Speech to text, for dictating a message instead of typing it."""

from speech.whisper import (
    WhisperError,
    WhisperInfo,
    clean_transcript,
    find_model,
    find_whisper,
    probe,
    transcribe,
)

__all__ = [
    "WhisperError",
    "WhisperInfo",
    "clean_transcript",
    "find_model",
    "find_whisper",
    "probe",
    "transcribe",
]
