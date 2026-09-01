"""The Tesseract OCR backend, and the switch between it and the model.

Tesseract is not installed on the machine this was written on, so every test
here runs against a **stub binary** this file writes: a small Python script
that behaves the way Tesseract does - takes the same arguments, writes to
stdout, exits non-zero with a message on failure.

That draws a clear line. What is verified is the part that is ours: discovery,
argument construction, the timeout, the failure paths, and the dispatch between
the two backends. What is *not* verified is whether Tesseract itself reads an
image well, which is Tesseract's business. The README says so too, rather than
implying a coverage that does not exist.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from config import Config
from tools.base import ToolRegistry
from tools.filesystem import WorkspaceFiles
from tools.ocr_tool import OcrClient, OcrError
from tools.registry import build_default_registry
from tools.tesseract import (
    TesseractBackend,
    TesseractError,
    find_tesseract,
    probe,
)


def write_stub(
    directory: Path,
    *,
    output: str = "stub text\nsecond line",
    exit_code: int = 0,
    stderr: str = "",
    version: str = "tesseract 5.3.3",
    languages: str = "List of available languages (2):\neng\nosd",
    sleep: float = 0.0,
) -> Path:
    """Write a fake `tesseract` that behaves like the real one.

    A .cmd shim on Windows so it is directly executable, wrapping a Python
    script that does the actual pretending.
    """
    script = directory / "stub_tesseract.py"
    script.write_text(
        "import sys, time, pathlib\n"
        f"time.sleep({sleep!r})\n"
        "args = sys.argv[1:]\n"
        "if '--version' in args:\n"
        f"    sys.stdout.write({version!r})\n"
        "    sys.exit(0)\n"
        "if '--list-langs' in args:\n"
        f"    sys.stdout.write({languages!r})\n"
        "    sys.exit(0)\n"
        # Real Tesseract opens the image, and resolves a relative path against
        # its own working directory. The stub must do the same, or a path bug
        # is invisible here and only appears against the real binary - which is
        # exactly what happened with the cwd/relative-path pair.
        "if args and not pathlib.Path(args[0]).is_file():\n"
        "    sys.stderr.write('Error, cannot read input file ' + args[0] +\n"
        "        ': No such file or directory\\nError during processing.')\n"
        "    sys.exit(1)\n"
        f"sys.stderr.write({stderr!r})\n"
        f"sys.stdout.write({output!r})\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )

    if os.name == "nt":
        shim = directory / "tesseract.cmd"
        shim.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8"
        )
    else:
        shim = directory / "tesseract"
        shim.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
        )
        shim.chmod(0o755)
    return shim


class StubCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.image = self.tmp / "note.png"
        # Not a real PNG; nothing here decodes it, and the stub does not care.
        self.image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 128)

    def tearDown(self):
        self._tmp.cleanup()

    # What configures the fake binary, against what configures the backend
    # talking to it. Keeping them apart stops a backend setting being passed
    # to the stub writer, which is silently wrong rather than obviously so.
    STUB_KEYS = ("output", "exit_code", "stderr", "version", "languages", "sleep")

    def backend(self, **overrides) -> TesseractBackend:
        stub = {key: overrides.pop(key) for key in list(overrides) if key in self.STUB_KEYS}
        return TesseractBackend(command=str(write_stub(self.tmp, **stub)), **overrides)


class DiscoveryTests(StubCase):
    def test_a_configured_path_is_used(self):
        stub = write_stub(self.tmp)
        self.assertEqual(find_tesseract(str(stub)), str(stub))

    def test_a_configured_path_that_does_not_exist_is_not_invented(self):
        self.assertEqual(find_tesseract(str(self.tmp / "nope.exe")), "")

    def test_nothing_configured_and_nothing_installed_returns_empty(self):
        # This machine has no Tesseract; if that ever changes the test would
        # be asserting something else, so it checks the contract not the box.
        found = find_tesseract("")
        self.assertTrue(found == "" or Path(found).exists())

    def test_probe_reports_version_and_languages(self):
        stub = write_stub(self.tmp)
        info = probe(str(stub))
        self.assertIsNotNone(info)
        self.assertIn("5.3.3", info.version)
        self.assertIn("eng", info.languages)
        # The header line is not a language.
        self.assertNotIn("List", " ".join(info.languages))

    def test_probe_of_something_missing_is_none_not_an_exception(self):
        self.assertIsNone(probe(str(self.tmp / "nope.exe")))

    def test_availability_reflects_whether_the_binary_is_there(self):
        self.assertTrue(self.backend().available())
        self.assertFalse(TesseractBackend(command="definitely-not-here").available())

    def test_the_missing_message_says_how_to_fix_it(self):
        message = TesseractBackend(command="nope").missing_message()
        self.assertIn("TESSERACT_CMD", message)
        self.assertIn("GLM-OCR", message)


class ReadingTests(StubCase):
    def test_text_comes_back(self):
        text = self.backend().read(self.image)
        self.assertIn("stub text", text)
        self.assertIn("second line", text)

    def test_the_arguments_are_the_ones_tesseract_expects(self):
        # "stdout" is Tesseract's keyword for the output file, not a path, and
        # getting it wrong writes a .txt beside the image instead.
        recorder = self.tmp / "args.txt"
        script = self.tmp / "recording.py"
        script.write_text(
            "import sys, pathlib\n"
            f"pathlib.Path({str(recorder)!r}).write_text(' '.join(sys.argv[1:]))\n"
            "sys.stdout.write('ok')\n",
            encoding="utf-8",
        )
        if os.name == "nt":
            shim = self.tmp / "rec.cmd"
            shim.write_text(
                f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8"
            )
        else:
            shim = self.tmp / "rec"
            shim.write_text(
                f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n',
                encoding="utf-8",
            )
            shim.chmod(0o755)

        TesseractBackend(command=str(shim), language="deu", psm=6).read(self.image)
        written = recorder.read_text(encoding="utf-8")
        self.assertIn("stdout", written)
        self.assertIn("-l deu", written)
        self.assertIn("--psm 6", written)

    def test_a_per_call_language_overrides_the_default(self):
        recorder = self.tmp / "args.txt"
        script = self.tmp / "recording.py"
        script.write_text(
            "import sys, pathlib\n"
            f"pathlib.Path({str(recorder)!r}).write_text(' '.join(sys.argv[1:]))\n"
            "sys.stdout.write('ok')\n",
            encoding="utf-8",
        )
        if os.name == "nt":
            shim = self.tmp / "rec.cmd"
            shim.write_text(
                f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8"
            )
        else:
            shim = self.tmp / "rec"
            shim.write_text(
                f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n',
                encoding="utf-8",
            )
            shim.chmod(0o755)

        TesseractBackend(command=str(shim), language="eng").read(
            self.image, language="fra"
        )
        self.assertIn("-l fra", recorder.read_text(encoding="utf-8"))

    def test_a_relative_path_survives_the_working_directory_move(self):
        """The bug the stub used to hide.

        `read` sets cwd to the image's own folder, so a relative path handed
        straight to Tesseract is resolved against it twice - "samples/a.png"
        becomes "samples/samples/a.png", and Tesseract reports only "Error
        during processing". A path relative to *this* process must still work.
        """
        # The path must carry a directory component: with a bare filename the
        # parent is "." and cwd does not actually move, so the double
        # resolution never happens and the test would pass either way.
        nested = self.tmp / "scans"
        nested.mkdir()
        (nested / "page.png").write_bytes(bytes(64))

        original = os.getcwd()
        os.chdir(self.tmp)
        try:
            text = self.backend().read(Path("scans") / "page.png")
        finally:
            os.chdir(original)
        self.assertIn("stub text", text)

    def test_a_missing_binary_is_a_clear_error(self):
        with self.assertRaises(TesseractError) as caught:
            TesseractBackend(command="not-installed").read(self.image)
        self.assertIn("not installed", str(caught.exception))

    def test_a_non_zero_exit_reports_what_tesseract_said(self):
        backend = self.backend(exit_code=1, stderr="Image file note.png not found")
        with self.assertRaises(TesseractError) as caught:
            backend.read(self.image)
        self.assertIn("not found", str(caught.exception))

    def test_a_missing_language_pack_says_how_to_fix_it(self):
        backend = self.backend(
            exit_code=1, stderr="Error: Failed loading language 'deu'"
        )
        with self.assertRaises(TesseractError) as caught:
            backend.read(self.image, language="deu")
        self.assertIn("language pack", str(caught.exception))
        self.assertIn("TESSERACT_LANG", str(caught.exception))

    def test_an_empty_result_suggests_the_other_backend(self):
        backend = self.backend(output="   \n  ")
        with self.assertRaises(TesseractError) as caught:
            backend.read(self.image)
        self.assertIn("no text", str(caught.exception))
        self.assertIn("GLM-OCR", str(caught.exception))

    def test_a_hung_binary_times_out_rather_than_hanging(self):
        backend = self.backend(sleep=5.0, timeout=0.5)
        with self.assertRaises(TesseractError) as caught:
            backend.read(self.image)
        self.assertIn("did not finish", str(caught.exception))


class BackendDispatchTests(StubCase):
    """The tool picks a backend, and says which one it used."""

    def client(self, backend: str, **config_overrides) -> OcrClient:
        settings = dict(
            workspace=self.tmp,
            ocr_enabled=True,
            ocr_backend=backend,
            tesseract_cmd=str(write_stub(self.tmp)),
        )
        settings.update(config_overrides)
        config = Config(**settings)
        return OcrClient(
            config, WorkspaceFiles(self.tmp), tesseract=TesseractBackend(
                command=config.tesseract_cmd
            )
        )

    def test_the_tesseract_backend_reads_without_any_server(self):
        result = self.client("tesseract").ocr_image("note.png")
        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "tesseract")
        self.assertIn("stub text", result["text"])

    def test_a_prompt_is_reported_as_ignored_rather_than_silently_dropped(self):
        result = self.client("tesseract").ocr_image("note.png", prompt="find the total")
        self.assertIn("note", result)
        self.assertIn("ignored", result["note"])

    def test_no_note_when_no_prompt_was_given(self):
        self.assertNotIn("note", self.client("tesseract").ocr_image("note.png"))

    def test_the_backend_property_reflects_the_config(self):
        self.assertEqual(self.client("tesseract").backend, "tesseract")
        self.assertEqual(self.client("model").backend, "model")

    def test_an_unknown_backend_name_falls_back_to_the_model(self):
        # Rather than failing at the moment someone attaches an image.
        self.assertEqual(self.client("nonsense").backend, "model")

    def test_readiness_reports_a_missing_tesseract(self):
        client = self.client("tesseract", tesseract_cmd="not-installed")
        ready, why = client.backend_ready()
        self.assertFalse(ready)
        self.assertIn("not installed", why)

    def test_readiness_is_true_when_the_binary_is_there(self):
        ready, why = self.client("tesseract").backend_ready()
        self.assertTrue(ready)
        self.assertEqual(why, "")

    def test_the_workspace_jail_applies_to_both_backends(self):
        with self.assertRaises(OcrError):
            self.client("tesseract").ocr_image("../outside.png")

    def test_the_size_limit_applies_to_both_backends(self):
        big = self.tmp / "big.png"
        big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 5000)
        client = self.client("tesseract", ocr_max_image_bytes=100)
        with self.assertRaises(OcrError) as caught:
            client.ocr_image("big.png")
        self.assertIn("limit", str(caught.exception))

    def test_an_unsupported_extension_is_refused_by_both_backends(self):
        (self.tmp / "notes.txt").write_text("hello", encoding="utf-8")
        with self.assertRaises(OcrError):
            self.client("tesseract").ocr_image("notes.txt")


class DescriptionTests(StubCase):
    """The model is told what it is actually getting."""

    def description(self, backend: str) -> str:
        config = Config(
            workspace=self.tmp,
            ocr_enabled=True,
            ocr_backend=backend,
            tesseract_cmd=str(write_stub(self.tmp)),
        )
        return OcrClient(config, WorkspaceFiles(self.tmp)).tool().description

    def test_the_tesseract_description_warns_it_ignores_instructions(self):
        text = self.description("tesseract").lower()
        self.assertIn("tesseract", text)
        self.assertIn("does not follow instructions", text)
        self.assertIn("screenshot", text)

    def test_the_model_description_offers_layout_and_tables(self):
        text = self.description("model").lower()
        self.assertIn("glm-ocr", text)
        self.assertIn("tables", text)
        self.assertIn("handwriting", text)

    def test_the_two_descriptions_differ(self):
        self.assertNotEqual(self.description("model"), self.description("tesseract"))


class RegistryTests(StubCase):
    def config(self, **overrides) -> Config:
        settings = dict(workspace=self.tmp)
        settings.update(overrides)
        return Config(**settings)

    def test_ocr_is_still_off_by_default(self):
        registry, disabled = build_default_registry(self.config())
        self.assertNotIn("ocr_image", registry)
        self.assertIn("ocr", [item.category for item in disabled])

    def test_the_disabled_reason_names_both_backends(self):
        _, disabled = build_default_registry(self.config())
        reason = next(item.reason for item in disabled if item.category == "ocr")
        self.assertIn("tesseract", reason.lower())
        self.assertIn("glm-ocr", reason.lower())
        self.assertIn("OCR_BACKEND", reason)

    def test_enabling_ocr_registers_the_tool_whichever_backend(self):
        for backend in ("model", "tesseract"):
            with self.subTest(backend=backend):
                registry, _ = build_default_registry(
                    self.config(ocr_enabled=True, ocr_backend=backend)
                )
                self.assertIn("ocr_image", registry)

    def test_tesseract_needs_no_model_server_to_be_registered(self):
        # The point of the second backend: OCR without 1.4 GB of weights.
        registry, _ = build_default_registry(
            self.config(
                ocr_enabled=True,
                ocr_backend="tesseract",
                tesseract_cmd=str(write_stub(self.tmp)),
            )
        )
        self.assertIn("ocr_image", registry)

    def test_the_tool_runs_through_the_registry(self):
        registry, _ = build_default_registry(
            self.config(
                ocr_enabled=True,
                ocr_backend="tesseract",
                tesseract_cmd=str(write_stub(self.tmp)),
            )
        )
        result = registry.execute("ocr_image", {"path": "note.png"})
        self.assertTrue(result.ok, result.payload)
        self.assertEqual(result.payload["backend"], "tesseract")


def _pillow_installed() -> bool:
    """The generated-image test draws with Pillow, which ships with the
    optional document-search dependencies rather than the base install."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(
    find_tesseract("") and _pillow_installed(),
    "needs Tesseract and Pillow, which are both optional",
)
class RealTesseractTests(unittest.TestCase):
    """Against a real Tesseract, when there is one.

    Skipped here - none is installed - so the stub tests above are what
    actually ran. This exists so that installing Tesseract turns the claim
    "it works" into something the suite checks rather than something the
    README asserts.
    """

    def test_it_reports_a_version_and_a_language(self):
        info = probe("")
        self.assertIsNotNone(info)
        self.assertTrue(info.version)
        self.assertTrue(info.languages)

    def test_it_reads_text_out_of_a_generated_image(self):
        from PIL import Image, ImageDraw  # noqa: PLC0415 - optional, test-only

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hello.png"
            image = Image.new("RGB", (320, 80), "white")
            ImageDraw.Draw(image).text((10, 30), "HELLO WORLD", fill="black")
            image.save(path)

            text = TesseractBackend().read(path)
            self.assertIn("HELLO", text.upper())
