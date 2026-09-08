"""Fetching a model into weights/.

Nothing here downloads anything. The transport is `get_llama.download`, which
has its own tests; what is worth testing is the part written around it, and
that part is mostly about the two ways this can go wrong quietly:

  * a 3 GB download interrupted at 2.9 GB leaving something that *looks* like
    a model, because weights/ is scanned by extension and anything found is
    offered as something to talk to, and
  * a size read from the wrong field of the Hugging Face listing, which would
    call a 3 GB file 135 bytes and then declare the real download truncated.

The catalog's repositories were resolved against the live API and every URL
answered a ranged request with 206. That is not re-checked here - a test suite
that fails because somebody's wifi is down is a test suite people learn to
ignore - but `test_the_starter_is_a_file_models_json_already_names` does guard
the half of it that can rot silently without anybody noticing.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import get_model  # noqa: E402


class CatalogTests(unittest.TestCase):
    """What is on offer, and whether it lands where the registry expects."""

    def test_the_starter_is_a_file_models_json_already_names(self):
        """The starter has to land on its curated registry entry, not be
        discovered. models.json's gemma entry carries a context and a RAM
        threshold somebody measured; a file arriving under any other name gets
        those inferred from its header instead, which is a worse answer than
        the one already written down."""
        registry = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
        named = {model.get("file") for model in registry["models"]}
        named |= {model.get("mmproj") for model in registry["models"]}

        starter = get_model.CATALOG[get_model.STARTER]["files"][0][0]
        self.assertIn(starter, named)

    def test_the_ocr_entry_fetches_both_halves(self):
        """The language model alone loads and then cannot see, which reads as
        a broken model rather than a missing file."""
        names = [name for name, _ in get_model.CATALOG["ocr"]["files"]]
        self.assertEqual(len(names), 2)
        self.assertTrue(any(name.startswith("mmproj-") for name in names))

    def test_every_entry_is_listed_in_the_order_it_is_offered(self):
        self.assertEqual(set(get_model.ORDER), set(get_model.CATALOG))

    def test_an_unknown_name_says_what_the_known_ones_are(self):
        with self.assertRaises(get_model.FetchError) as raised:
            get_model.entry("gemma-4-27b")
        self.assertIn("gemma", str(raised.exception))

    def test_a_projector_on_its_own_is_not_something_to_talk_to(self):
        with tempfile.TemporaryDirectory() as scratch:
            weights = Path(scratch)
            (weights / "mmproj-GLM-OCR-Q8_0.gguf").write_bytes(b"x")
            with mock.patch.object(get_model, "WEIGHTS", weights):
                self.assertFalse(get_model.anything_to_talk_to())
                (weights / "gemma-4-E2B-it-Q4_0.gguf").write_bytes(b"x")
                self.assertTrue(get_model.anything_to_talk_to())


class DownloadTests(unittest.TestCase):
    """Where the bytes land, and what happens when they stop arriving."""

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.weights = Path(self.scratch.name)
        self.addCleanup(self.scratch.cleanup)

        patch = mock.patch.object(get_model, "WEIGHTS", self.weights)
        patch.start()
        self.addCleanup(patch.stop)

        # Nothing in these tests may reach the network.
        sizes = mock.patch.object(get_model, "live_sizes", return_value={})
        sizes.start()
        self.addCleanup(sizes.stop)

    def fake_download(self, *, fail_on: str = ""):
        """Stand in for the transport, recording what it was asked for."""
        self.asked: list[tuple[str, str, int]] = []

        def download(url, target, size, on_progress=None):
            self.asked.append((url, target.name, size))
            if fail_on and fail_on in target.name:
                # Half a file, exactly as a dropped connection leaves it.
                target.write_bytes(b"x" * 3)
                raise get_model.FetchError("connection reset")
            target.write_bytes(b"x" * 10)

        return download

    def install(self, key, **kwargs):
        with redirect_stdout(io.StringIO()):
            return get_model.install(key, **kwargs)

    def test_a_finished_download_is_renamed_into_place(self):
        with mock.patch.object(get_model, "download", self.fake_download()):
            self.install("gemma")

        landed = sorted(path.name for path in self.weights.iterdir())
        self.assertEqual(landed, ["gemma-4-E2B-it-Q4_0.gguf"])

    def test_the_bytes_arrive_under_a_part_name_first(self):
        """The rename is the whole safeguard, so it is worth asserting that
        the transport never sees the real name."""
        with mock.patch.object(get_model, "download", self.fake_download()):
            self.install("gemma")

        _, written_to, _ = self.asked[0]
        self.assertTrue(written_to.endswith(".gguf.part"), written_to)

    def test_an_interrupted_download_leaves_no_gguf_behind(self):
        """weights/ is scanned by extension. A partial file under the real
        name would be offered as a model and fail at the first message, which
        is a far worse outcome than a model that is simply absent."""
        with mock.patch.object(
            get_model, "download", self.fake_download(fail_on="gemma")
        ):
            with self.assertRaises(get_model.FetchError):
                self.install("gemma")

        self.assertEqual(list(self.weights.glob("*.gguf")), [])
        self.assertTrue(list(self.weights.glob("*.part")))

    def test_the_leftover_part_is_kept_for_the_resume(self):
        """Deleting it would make every retry start from zero, which is the
        thing Range resume exists to avoid on a 3 GB file."""
        with mock.patch.object(
            get_model, "download", self.fake_download(fail_on="gemma")
        ):
            with self.assertRaises(get_model.FetchError):
                self.install("gemma")

        partial = self.weights / "gemma-4-E2B-it-Q4_0.gguf.part"
        self.assertEqual(partial.read_bytes(), b"x" * 3)

    def test_running_it_twice_downloads_nothing_the_second_time(self):
        with mock.patch.object(get_model, "download", self.fake_download()):
            self.install("gemma")
            self.install("gemma")
        self.assertEqual(len(self.asked), 1)

    def test_force_downloads_over_a_file_that_is_already_there(self):
        """replace(), not rename(): on Windows renaming onto an existing file
        raises, and --force is exactly the case where one exists."""
        with mock.patch.object(get_model, "download", self.fake_download()):
            self.install("gemma")
            self.install("gemma", force=True)
        self.assertEqual(len(self.asked), 2)

    def test_only_the_missing_half_of_a_pair_is_fetched(self):
        (self.weights / "GLM-OCR-Q8_0.gguf").write_bytes(b"x")
        with mock.patch.object(get_model, "download", self.fake_download()):
            self.install("ocr")

        fetched = [name for _, name, _ in self.asked]
        self.assertEqual(fetched, ["mmproj-GLM-OCR-Q8_0.gguf.part"])

    def test_it_refuses_before_starting_when_the_disk_cannot_hold_it(self):
        """Filling somebody's disk and then failing is the one outcome worse
        than not starting."""
        usage = mock.Mock(free=1_000_000)
        with mock.patch.object(get_model.shutil, "disk_usage", return_value=usage):
            with mock.patch.object(get_model, "download", self.fake_download()):
                with self.assertRaises(get_model.FetchError) as raised:
                    self.install("gemma")

        self.assertIn("disk", str(raised.exception).lower())
        self.assertEqual(list(self.weights.iterdir()), [])

    def test_a_live_size_overrides_the_recorded_one(self):
        """The recorded number is for the sentence printed beforehand. The
        live one decides whether the download finished, because a repository
        can be re-quantised under a name that never changes."""
        with mock.patch.object(
            get_model, "live_sizes", return_value={"gemma-4-E2B-it-Q4_0.gguf": 7}
        ):
            with mock.patch.object(get_model, "download", self.fake_download()):
                self.install("gemma")

        _, _, size = self.asked[0]
        self.assertEqual(size, 7)


class ListingTests(unittest.TestCase):
    """Reading the file listing Hugging Face returns."""

    def listing(self, payload):
        response = io.BytesIO(json.dumps(payload).encode())
        response.__enter__ = lambda self=response: self
        response.__exit__ = lambda *args: None
        with mock.patch.object(
            get_model.urllib.request, "urlopen", return_value=response
        ):
            return get_model.live_sizes("owner/repo")

    def test_it_reads_the_lfs_size_not_the_pointer_size(self):
        """An LFS-backed file reports the pointer's length at the top level -
        about 135 bytes - and the real one under `lfs`. Taking the top-level
        number would have every real model declared truncated on arrival."""
        sizes = self.listing(
            [{"path": "model.gguf", "size": 135, "lfs": {"size": 3_041_378_400}}]
        )
        self.assertEqual(sizes, {"model.gguf": 3_041_378_400})

    def test_a_plain_file_still_reports_its_size(self):
        sizes = self.listing([{"path": "small.gguf", "size": 4096}])
        self.assertEqual(sizes, {"small.gguf": 4096})

    def test_anything_that_is_not_a_gguf_is_ignored(self):
        sizes = self.listing(
            [{"path": "README.md", "size": 10}, {"path": "a.gguf", "size": 10}]
        )
        self.assertEqual(sizes, {"a.gguf": 10})

    def test_a_failure_falls_back_rather_than_stopping_the_download(self):
        """No network, a rate limit, a renamed repository: the recorded size
        is right far more often than it is wrong, so this is a fallback and
        never an error."""
        with mock.patch.object(
            get_model.urllib.request, "urlopen", side_effect=OSError("no route")
        ):
            self.assertEqual(get_model.live_sizes("owner/repo"), {})

    def test_a_reply_that_is_not_a_listing_is_not_trusted(self):
        self.assertEqual(self.listing({"error": "Repository not found"}), {})


if __name__ == "__main__":
    unittest.main()
