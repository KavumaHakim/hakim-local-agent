"""Fetching what dictation and reading aloud need. No network.

The asset names below are release b4938's real contents, copied verbatim, for
the same reason `test_get_llama.py` copies llama.cpp's: what is worth testing
is that the right one is picked out of a list where most of the entries would
download an accelerator runtime this machine cannot use.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import get_speech  # noqa: E402
from get_speech import FetchError, asset_for, find_cli  # noqa: E402

# Every asset of whisper.cpp release b4938, exactly as GitHub lists them.
ASSETS = [
    "whisper-b4938-xcframework.zip",
    "whisper-bin-Win32.zip",
    "whisper-bin-ubuntu-arm64.tar.gz",
    "whisper-bin-ubuntu-x64.tar.gz",
    "whisper-bin-x64.zip",
    "whisper-blas-bin-Win32.zip",
    "whisper-blas-bin-x64.zip",
    "whisper-cublas-11.8.0-bin-x64.zip",
    "whisper-cublas-12.4.0-bin-x64.zip",
]


class AssetChoiceTests(unittest.TestCase):
    def test_every_supported_platform_maps_to_a_real_asset(self):
        for (system, architecture), name in get_speech.ASSETS.items():
            with self.subTest(platform=f"{system}-{architecture}"):
                self.assertIn(name, ASSETS)

    def test_no_accelerator_build_is_ever_chosen(self):
        """blas and cublas need runtimes this machine does not have, and
        choosing one produces a binary that fails at the moment it is used."""
        for name in get_speech.ASSETS.values():
            self.assertNotIn("blas", name)
            self.assertNotIn("cublas", name)

    def test_the_xcframework_is_not_mistaken_for_a_build(self):
        """It is an Xcode framework, not a command line program."""
        self.assertNotIn(
            "whisper-b4938-xcframework.zip", set(get_speech.ASSETS.values())
        )

    def test_x64_and_x86_are_not_confused(self):
        self.assertEqual(asset_for("windows", "x64"), "whisper-bin-x64.zip")
        self.assertEqual(asset_for("windows", "x86"), "whisper-bin-Win32.zip")

    def test_macos_says_plainly_that_there_is_no_build(self):
        """whisper.cpp publishes only an xcframework. Saying so beats a
        confusing "not found" on a platform where the answer is "build it"."""
        with self.assertRaises(FetchError) as caught:
            asset_for("darwin", "arm64")
        message = str(caught.exception)
        self.assertIn("macOS", message)
        self.assertIn("brew", message)

    def test_this_machine_maps_to_something(self):
        system, architecture = get_speech.platform_tokens()
        if system == "darwin":
            self.skipTest("macOS has no binary release, which has its own test")
        self.assertIn(asset_for(system, architecture), ASSETS)


class NameTests(unittest.TestCase):
    def test_an_unknown_model_lists_the_known_ones(self):
        with self.assertRaises(FetchError) as caught:
            get_speech.install_model("enormous.en")
        self.assertIn("base.en", str(caught.exception))

    def test_an_unknown_voice_says_where_any_voice_comes_from(self):
        """The four named here are a convenience, not the whole set - so the
        error points at the repository rather than implying a closed list."""
        with self.assertRaises(FetchError) as caught:
            get_speech.install_voice("de_DE-thorsten-medium")
        message = str(caught.exception)
        self.assertIn("piper-voices", message)
        self.assertIn(".onnx.json", message)

    def test_every_named_voice_has_a_complete_path(self):
        for name, entry in get_speech.VOICES_AVAILABLE.items():
            with self.subTest(voice=name):
                language, locale, speaker, quality, size = entry
                self.assertTrue(name.startswith(locale))
                self.assertIn(speaker, name)
                self.assertIn(quality, name)
                self.assertGreater(size, 0)


class FindingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_nothing_there_is_none_not_an_error(self):
        self.assertIsNone(find_cli(self.tmp))

    def test_a_missing_directory_is_none(self):
        self.assertIsNone(find_cli(self.tmp / "never-made"))

    def test_the_cli_is_found_at_the_top(self):
        made = self.tmp / "whisper-cli.exe"
        made.write_bytes(b"x")
        self.assertEqual(find_cli(self.tmp), made)

    def test_the_cli_is_found_however_deep_the_archive_nested_it(self):
        """The Windows archive puts everything under Release/ and the Linux
        one does not, so the search cannot assume either shape."""
        nested = self.tmp / "Release" / "bin"
        nested.mkdir(parents=True)
        made = nested / "whisper-cli.exe"
        made.write_bytes(b"x")
        self.assertEqual(find_cli(self.tmp), made)


class InstallTests(unittest.TestCase):
    """What install_* does around the download, with the download stubbed."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        for name in ("VENDOR", "WHISPER_MODELS", "VOICES"):
            patch = mock.patch.object(get_speech, name, self.tmp / name.lower())
            patch.start()
            self.addCleanup(patch.stop)
        quiet = mock.patch.object(get_speech, "say", lambda *a, **k: None)
        quiet.start()
        self.addCleanup(quiet.stop)

    def test_an_existing_model_is_not_downloaded_again(self):
        get_speech.WHISPER_MODELS.mkdir(parents=True)
        (get_speech.WHISPER_MODELS / "ggml-base.en.bin").write_bytes(b"x")
        with mock.patch.object(get_speech, "download") as download:
            get_speech.install_model("base.en")
        download.assert_not_called()

    def test_a_voice_downloads_its_config_as_well(self):
        """The .json carries the phoneme map and the sample rate. A voice
        without one does not load, so it is not an optional second file."""
        urls = []

        def fake(url, target, size, **kwargs):
            urls.append(url)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")

        with mock.patch.object(get_speech, "download", fake):
            get_speech.install_voice("en_US-lessac-medium")

        self.assertEqual(len(urls), 2)
        self.assertTrue(urls[0].endswith(".onnx"))
        self.assertTrue(urls[1].endswith(".onnx.json"))
        self.assertTrue((get_speech.VOICES / "en_US-lessac-medium.onnx").is_file())
        self.assertTrue(
            (get_speech.VOICES / "en_US-lessac-medium.onnx.json").is_file()
        )

    def test_a_voice_with_no_config_beside_it_is_fetched_again(self):
        """Half a voice is not a voice, and `speech/piper.py` will not offer
        it - so finding only the .onnx has to mean "unfinished", not "done"."""
        get_speech.VOICES.mkdir(parents=True)
        (get_speech.VOICES / "en_US-lessac-medium.onnx").write_bytes(b"x")

        def fake(url, target, size, **kwargs):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")

        with mock.patch.object(get_speech, "download", side_effect=fake) as download:
            get_speech.install_voice("en_US-lessac-medium")
        self.assertTrue(download.called)


if __name__ == "__main__":
    unittest.main()
