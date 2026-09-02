"""Speech to text. No whisper binary and no audio required.

The subprocess is replaced by a fake runner, because what is worth testing
here is the argument list, the failure paths and what comes back out of the
transcript - not that whisper.cpp works, which is whisper.cpp's business.

The one thing that cannot be faked is checked in prose in `speech/whisper.py`
instead: the real binary puts the transcript on stdout and its chatter on
stderr, which is what makes parsing this safe at all.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from speech import whisper as speech
from speech.whisper import WhisperError, clean_transcript, find_model, find_whisper


class FakeResult:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class FakeRunner:
    """Stands in for subprocess.run and remembers what it was asked to do."""

    def __init__(self, result=None, raises=None):
        self.result = result or FakeResult(stdout="hello there\n")
        self.raises = raises
        self.calls: list[list[str]] = []

    def __call__(self, arguments, **kwargs):
        self.calls.append(list(arguments))
        self.kwargs = kwargs
        if self.raises is not None:
            raise self.raises
        return self.result


class FindingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)

        # Nothing here may see the real project: a developer machine has a
        # build in vendor/whisper and a model in weights/, and a test that
        # passes only on that machine is not a test.
        root = mock.patch.object(speech, "PROJECT_ROOT", self.tmp)
        root.start()
        self.addCleanup(root.stop)
        vendor = mock.patch.object(speech, "VENDOR", self.tmp / "vendor" / "whisper")
        vendor.start()
        self.addCleanup(vendor.stop)
        nothing_on_path = mock.patch.object(speech.shutil, "which", lambda name: None)
        nothing_on_path.start()
        self.addCleanup(nothing_on_path.stop)

    def make(self, relative: str) -> Path:
        path = self.tmp / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return path

    # --- the binary ---

    def test_nothing_installed_is_an_empty_string_not_an_error(self):
        """Not installed is a supported state: the UI hides the microphone
        rather than offering a button that fails when pressed."""
        self.assertEqual(find_whisper(), "")

    def test_the_vendored_build_is_found(self):
        made = self.make("vendor/whisper/whisper-cli.exe")
        self.assertEqual(find_whisper(), str(made))

    def test_a_configured_path_wins_over_the_vendored_build(self):
        self.make("vendor/whisper/whisper-cli.exe")
        mine = self.make("elsewhere/whisper-cli.exe")
        self.assertEqual(find_whisper(str(mine)), str(mine))

    def test_a_configured_path_that_is_gone_is_not_honoured(self):
        """Better to fall back to the search than to report a path that was
        right last month as though it were still there."""
        self.assertEqual(find_whisper(str(self.tmp / "ghost.exe")), "")

    # --- the model ---

    def test_no_model_is_an_empty_string(self):
        self.make("vendor/whisper/whisper-cli.exe")
        self.assertEqual(find_model(), "")

    def test_a_model_in_the_whisper_folder_is_found(self):
        made = self.make("whisper/ggml-base.en.bin")
        self.assertEqual(find_model(), str(made))

    def test_a_model_dropped_into_weights_is_found_too(self):
        """weights/ is the folder this project tells people to put models in,
        so it is where the first whisper model lands whatever the docs say."""
        made = self.make("weights/ggml-base.en.bin")
        self.assertEqual(find_model(), str(made))

    def test_the_whisper_folder_is_preferred_over_weights(self):
        proper = self.make("whisper/ggml-small.en.bin")
        self.make("weights/ggml-base.en.bin")
        self.assertEqual(find_model(), str(proper))

    def test_the_smallest_model_in_a_folder_wins(self):
        """Seconds a clip matter more here than the last points of accuracy on
        a dictated sentence."""
        self.make("whisper/ggml-medium.en.bin")
        small = self.make("whisper/ggml-base.en.bin")
        self.make("whisper/ggml-large-v3.bin")
        self.assertEqual(find_model(), str(small))

    def test_a_gguf_in_weights_is_not_mistaken_for_a_whisper_model(self):
        """weights/ is full of GGUFs. Only `ggml-*.bin` is whisper's."""
        self.make("weights/Qwen3-8B-Q4_K_M.gguf")
        self.assertEqual(find_model(), "")

    def test_probe_reports_nothing_when_either_half_is_missing(self):
        self.make("vendor/whisper/whisper-cli.exe")
        self.assertIsNone(speech.probe())  # binary but no model

        self.make("whisper/ggml-base.en.bin")
        info = speech.probe()
        self.assertIsNotNone(info)
        self.assertEqual(info.model_name, "base.en")


class TranscribeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)

        self.binary = self.tmp / "whisper-cli.exe"
        self.binary.write_bytes(b"x")
        self.model = self.tmp / "ggml-base.en.bin"
        self.model.write_bytes(b"x")
        self.audio = self.tmp / "clip.wav"
        self.audio.write_bytes(b"RIFF")

    def run_with(self, runner):
        return speech.transcribe(
            self.audio,
            command=str(self.binary),
            model=str(self.model),
            runner=runner,
        )

    def test_the_transcript_comes_back(self):
        self.assertEqual(self.run_with(FakeRunner()), "hello there")

    def test_the_command_asks_for_text_and_nothing_else(self):
        runner = FakeRunner()
        self.run_with(runner)
        command = runner.calls[0]

        self.assertEqual(command[0], str(self.binary))
        self.assertEqual(command[command.index("-m") + 1], str(self.model))
        # No timestamps, no progress, no non-speech tokens: everything that
        # would otherwise have to be parsed back out of the transcript.
        self.assertIn("-nt", command)
        self.assertIn("-np", command)
        self.assertIn("-sns", command)

    def test_the_audio_path_is_made_absolute(self):
        """The lesson tools/tesseract.py records in prose: a relative path gets
        resolved twice and the binary reports only that it failed."""
        runner = FakeRunner()
        self.run_with(runner)
        given = runner.calls[0][runner.calls[0].index("-f") + 1]
        self.assertTrue(Path(given).is_absolute())

    def test_a_missing_clip_says_so_before_anything_is_run(self):
        runner = FakeRunner()
        with self.assertRaises(WhisperError):
            speech.transcribe(
                self.tmp / "ghost.wav",
                command=str(self.binary),
                model=str(self.model),
                runner=runner,
            )
        self.assertEqual(runner.calls, [])

    def test_a_missing_binary_names_where_it_was_looked_for(self):
        with mock.patch.object(speech.shutil, "which", lambda name: None):
            with mock.patch.object(speech, "VENDOR", self.tmp / "nowhere"):
                with self.assertRaises(WhisperError) as caught:
                    speech.transcribe(self.audio, runner=FakeRunner())
        self.assertIn("nowhere", str(caught.exception))

    def test_a_timeout_is_reported_as_one(self):
        runner = FakeRunner(raises=subprocess.TimeoutExpired("whisper", 5))
        with self.assertRaises(WhisperError) as caught:
            self.run_with(runner)
        self.assertIn("longer than", str(caught.exception))

    def test_a_failure_carries_the_last_line_of_the_error(self):
        runner = FakeRunner(
            FakeResult(stderr="loading model\nerror: bad model file\n", returncode=1)
        )
        with self.assertRaises(WhisperError) as caught:
            self.run_with(runner)
        self.assertIn("bad model file", str(caught.exception))


class CleaningTests(unittest.TestCase):
    """Whisper's annotations, which are never something a person said."""

    def test_real_speech_is_left_alone(self):
        self.assertEqual(
            clean_transcript(" Read requirements.txt and tell me what it needs.\n"),
            "Read requirements.txt and tell me what it needs.",
        )

    def test_bracketed_annotations_go(self):
        self.assertEqual(clean_transcript(" [BLANK_AUDIO]\n"), "")
        self.assertEqual(clean_transcript(" (dramatic music)\n"), "")
        self.assertEqual(clean_transcript(" *coughs*\n"), "")

    def test_an_annotation_beside_real_speech_leaves_the_speech(self):
        self.assertEqual(
            clean_transcript(" (clears throat) open the file\n"), "open the file"
        )

    def test_several_lines_become_one_message(self):
        """Whisper breaks on its 30-second windows, which has nothing to do
        with where the sentences are."""
        self.assertEqual(
            clean_transcript(" read the file\n and tell me what it says\n"),
            "read the file and tell me what it says",
        )

    def test_whitespace_is_tidied(self):
        self.assertEqual(clean_transcript("   hello    there  \n\n"), "hello there")

    def test_a_word_whisper_invented_cannot_be_cleaned_away(self):
        """Two seconds of silence transcribes as " you". There is no way to
        tell that from somebody actually saying "you", which is exactly why the
        transcript goes to the message box to be read rather than to the
        agent."""
        self.assertEqual(clean_transcript(" you\n"), "you")


if __name__ == "__main__":
    unittest.main()
