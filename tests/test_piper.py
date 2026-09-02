"""Reading a reply aloud. No Piper install and no audio required.

The worker process is replaced by a fake one that speaks the protocol, because
what is worth testing here is the lifecycle - lazy start, idle stop, one
request at a time, what happens when the worker dies - and not that Piper
synthesises speech, which is Piper's business.

The one thing that cannot be faked is recorded in prose in `speech/piper.py`:
the measurements that say why the voice is kept warm at all when whisper,
twenty lines away, deliberately is not.
"""

from __future__ import annotations

import base64
import io
import json
import pathlib
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

from speech import piper
from speech.piper import Voice, VoiceError, find_voice
from speech.voice_worker import to_wav


def a_wav(seconds: float = 0.1, rate: int = 22050) -> bytes:
    return to_wav(b"\0\0" * int(rate * seconds), rate)


class FakeProcess:
    """A worker that speaks the protocol without loading anything."""

    def __init__(self, *, rate: int = 22050, dies_after: int | None = None):
        self.rate = rate
        self.dies_after = dies_after
        self.requests: list[dict] = []
        self.shutdown = False
        self.killed = False
        self._alive = True
        self.stdin = self
        self.stdout = None
        self.stderr = None
        self.replies: list[str] = [
            json.dumps({"ok": True, "rate": rate, "voice": "test-voice"})
        ]

    # --- the stdin half ---

    def write(self, line: str) -> None:
        request = json.loads(line)
        self.requests.append(request)
        if request.get("op") == "shutdown":
            self.shutdown = True
            self._alive = False
            return
        if self.dies_after is not None and len(self.requests) > self.dies_after:
            self._alive = False
            return
        self.replies.append(
            json.dumps(
                {
                    "ok": True,
                    "rate": self.rate,
                    "samples": 100,
                    "wav": base64.b64encode(a_wav()).decode("ascii"),
                }
            )
        )

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    # --- the process half ---

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        self._alive = False
        return 0

    def kill(self):
        self.killed = True
        self._alive = False


class VoiceHarness(Voice):
    """A Voice whose worker is a FakeProcess rather than a subprocess."""

    def __init__(self, **kwargs):
        self.spawned = 0
        self.process_kwargs = kwargs.pop("process_kwargs", {})
        super().__init__(**kwargs)

    def _start(self) -> None:
        self.spawned += 1
        process = FakeProcess(**self.process_kwargs)
        self._process = process
        self._stderr = []
        # The real one reads replies on a thread; here they are already there.
        import queue

        self._replies = queue.Queue()
        self._pump = process
        reply = self._take_reply()
        self._rate = int(reply.get("rate", 0))
        self._name = str(reply.get("voice", ""))
        self._last_used = time.time()

    def _take_reply(self) -> dict:
        return json.loads(self._pump.replies.pop(0))

    def _await_reply(self, timeout: float) -> dict:
        if not self._pump.replies:
            self.unload()
            raise VoiceError("The voice worker exited.")
        reply = self._take_reply()
        if not reply.get("ok"):
            raise VoiceError(str(reply.get("error", "The voice failed.")))
        return reply


class FindingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        root = mock.patch.object(piper, "PROJECT_ROOT", self.tmp)
        root.start()
        self.addCleanup(root.stop)

    def make_voice(self, relative: str, *, config: bool = True) -> Path:
        path = self.tmp / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"onnx")
        if config:
            path.with_suffix(path.suffix + ".json").write_text("{}", encoding="utf-8")
        return path

    def test_nothing_installed_is_an_empty_string(self):
        self.assertEqual(find_voice(), "")

    def test_a_voice_in_the_tts_folder_is_found(self):
        made = self.make_voice("tts/en_US-lessac-medium.onnx")
        self.assertEqual(find_voice(), str(made))

    def test_an_onnx_with_no_config_beside_it_is_not_a_voice(self):
        """The .json carries the phoneme map and the sample rate. Without it
        Piper cannot load the model, so offering it would be a trap."""
        self.make_voice("tts/lonely.onnx", config=False)
        self.assertEqual(find_voice(), "")

    def test_the_uppercase_folder_is_searched_too(self):
        """Windows is case-insensitive and Linux is not. A repository whose
        voices were put in TTS/ on Windows still has to work when cloned."""
        made = self.make_voice("TTS/voice.onnx")
        found = find_voice()
        self.assertTrue(found)
        # Compared as a file, not as a string: on Windows the search finds it
        # under the first spelling it tries, which is the same file by another
        # name. On Linux only one of the two spellings exists at all.
        self.assertTrue(pathlib.Path(found).samefile(made))

    def test_a_configured_path_wins(self):
        self.make_voice("tts/found.onnx")
        mine = self.make_voice("elsewhere/mine.onnx")
        self.assertEqual(find_voice(str(mine)), str(mine))

    def test_a_configured_path_that_is_gone_is_not_honoured(self):
        self.assertEqual(find_voice(str(self.tmp / "ghost.onnx")), "")


class LifecycleTests(unittest.TestCase):
    """Start late, stop early - the bargain that makes 175 MB acceptable."""

    def test_nothing_starts_until_something_is_said(self):
        voice = VoiceHarness()
        self.assertFalse(voice.loaded)
        self.assertEqual(voice.spawned, 0)

    def test_speaking_starts_the_worker_once(self):
        voice = VoiceHarness()
        voice.speak("first")
        voice.speak("second")
        self.assertEqual(voice.spawned, 1, "the voice was reloaded between calls")
        self.assertTrue(voice.loaded)

    def test_the_audio_comes_back_as_a_wav(self):
        voice = VoiceHarness()
        audio = voice.speak("hello")
        with wave.open(io.BytesIO(audio)) as handle:
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getsampwidth(), 2)
            self.assertEqual(handle.getframerate(), 22050)

    def test_an_empty_string_never_reaches_the_worker(self):
        voice = VoiceHarness()
        with self.assertRaises(VoiceError):
            voice.speak("   ")
        self.assertEqual(voice.spawned, 0)

    def test_unloading_asks_before_killing(self):
        """A clean exit closes the onnxruntime session rather than leaving the
        model file locked, which is the sort of thing Windows remembers."""
        voice = VoiceHarness()
        voice.speak("hello")
        process = voice._process
        self.assertTrue(voice.unload())
        self.assertTrue(process.shutdown)
        self.assertFalse(process.killed)

    def test_unloading_twice_is_harmless(self):
        voice = VoiceHarness()
        voice.speak("hello")
        self.assertTrue(voice.unload())
        self.assertFalse(voice.unload())

    def test_it_starts_again_after_being_swept(self):
        voice = VoiceHarness()
        voice.speak("before")
        voice.unload()
        voice.speak("after")
        self.assertEqual(voice.spawned, 2)
        self.assertTrue(voice.loaded)


class IdleTests(unittest.TestCase):
    def test_a_busy_voice_is_not_swept(self):
        voice = VoiceHarness(idle_seconds=60)
        voice.speak("hello")
        self.assertFalse(voice.unload_if_idle())
        self.assertTrue(voice.loaded)

    def test_an_idle_voice_gives_its_memory_back(self):
        voice = VoiceHarness(idle_seconds=0.05)
        voice.speak("hello")
        time.sleep(0.1)
        self.assertTrue(voice.unload_if_idle())
        self.assertFalse(voice.loaded)

    def test_zero_seconds_means_never_sweep(self):
        """A deliberate "keep it", not an accidental "drop it immediately"."""
        voice = VoiceHarness(idle_seconds=0)
        voice.speak("hello")
        time.sleep(0.05)
        self.assertFalse(voice.unload_if_idle())
        self.assertTrue(voice.loaded)

    def test_a_voice_that_never_started_is_not_swept(self):
        voice = VoiceHarness(idle_seconds=0.01)
        self.assertFalse(voice.unload_if_idle())

    def test_idle_time_is_zero_while_unloaded(self):
        self.assertEqual(VoiceHarness().idle_for(), 0.0)


class FailureTests(unittest.TestCase):
    def test_a_worker_that_dies_is_reported_not_hung(self):
        voice = VoiceHarness(process_kwargs={"dies_after": 0})
        with self.assertRaises(VoiceError):
            voice.speak("hello")
        self.assertFalse(voice.loaded)

    def test_a_dead_worker_does_not_stop_the_next_attempt(self):
        voice = VoiceHarness(process_kwargs={"dies_after": 0})
        with self.assertRaises(VoiceError):
            voice.speak("first")
        voice.process_kwargs = {}
        self.assertTrue(voice.speak("second"))


class MarkupTests(unittest.TestCase):
    """The worker's WAV wrapper, which the browser has to be able to play."""

    def test_the_wav_header_says_what_the_audio_is(self):
        audio = to_wav(b"\0\0" * 2205, 22050)
        self.assertEqual(audio[:4], b"RIFF")
        self.assertEqual(audio[8:12], b"WAVE")
        with wave.open(io.BytesIO(audio)) as handle:
            self.assertEqual(handle.getframerate(), 22050)
            self.assertEqual(handle.getnframes(), 2205)


if __name__ == "__main__":
    unittest.main()
