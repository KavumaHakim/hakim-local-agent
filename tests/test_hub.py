"""Searching for and downloading models. No network.

The Hugging Face responses here are the real shapes, copied from the live API,
because the interesting failures are in what comes back rather than in the
request. Downloads are exercised against a local HTTP server serving a few
kilobytes: the machinery is worth testing, and a real GGUF is gigabytes.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from models import hub
from models.hub import (
    Downloads,
    HubError,
    HubFile,
    _safe_name,
    estimate_ram_mb,
    files,
    search,
)

# One page of the real /api/models response, trimmed to the fields used.
SEARCH_RESPONSE = [
    {
        "id": "unsloth/Qwen3-1.7B-GGUF",
        "downloads": 48250,
        "likes": 42,
        "tags": ["gguf", "qwen3", "text-generation"],
    },
    {"modelId": "someone/Other-GGUF", "downloads": 10, "likes": 0},
    {"downloads": 5},  # no id at all; must be skipped rather than crash
]

# The real /tree/main shape: a mix of GGUF and everything else.
TREE_RESPONSE = [
    {"path": "README.md", "size": 4321, "type": "file"},
    {"path": "Qwen3-1.7B-BF16.gguf", "size": 3_450_000_000, "type": "file"},
    {"path": "Qwen3-1.7B-Q4_K_M.gguf", "size": 1_100_000_000, "type": "file"},
    {"path": "Qwen3-1.7B-IQ1_S.gguf", "size": 540_000_000, "type": "file"},
    {"path": "imatrix.dat", "size": 999, "type": "file"},
]


class Reply:
    """Stands in for a requests.Response."""

    def __init__(self, payload, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class SearchTests(unittest.TestCase):
    def test_results_carry_what_the_ui_shows(self):
        with mock.patch.object(hub.requests, "get", return_value=Reply(SEARCH_RESPONSE)):
            found = search("qwen3")
        self.assertEqual(found[0].id, "unsloth/Qwen3-1.7B-GGUF")
        self.assertEqual(found[0].downloads, 48250)

    def test_an_entry_with_no_id_is_skipped_not_fatal(self):
        with mock.patch.object(hub.requests, "get", return_value=Reply(SEARCH_RESPONSE)):
            found = search("qwen3")
        self.assertEqual(len(found), 2)

    def test_modelid_is_accepted_as_well_as_id(self):
        with mock.patch.object(hub.requests, "get", return_value=Reply(SEARCH_RESPONSE)):
            found = search("qwen3")
        self.assertIn("someone/Other-GGUF", [model.id for model in found])

    def test_an_empty_query_asks_nothing(self):
        with mock.patch.object(hub.requests, "get", side_effect=AssertionError("asked")):
            self.assertEqual(search("   "), [])

    def test_a_gated_repository_says_so_rather_than_asking_for_a_login(self):
        with mock.patch.object(hub.requests, "get", return_value=Reply([], 401)):
            with self.assertRaises(HubError) as caught:
                search("something")
        self.assertIn("login", str(caught.exception))

    def test_rate_limiting_is_reported_as_itself(self):
        with mock.patch.object(hub.requests, "get", return_value=Reply([], 429)):
            with self.assertRaises(HubError) as caught:
                search("something")
        self.assertIn("rate-limiting", str(caught.exception))


class FileListingTests(unittest.TestCase):
    def listing(self):
        with mock.patch.object(hub.requests, "get", return_value=Reply(TREE_RESPONSE)):
            return files("unsloth/Qwen3-1.7B-GGUF")

    def test_only_gguf_files_are_offered(self):
        names = [item.path for item in self.listing()]
        self.assertNotIn("README.md", names)
        self.assertNotIn("imatrix.dat", names)
        self.assertEqual(len(names), 3)

    def test_smallest_first(self):
        """A list that opens with a 30 GB BF16 file is one nobody reads to the
        end of, on the hardware this is built for."""
        sizes = [item.size_bytes for item in self.listing()]
        self.assertEqual(sizes, sorted(sizes))

    def test_the_quantisation_is_pulled_out_of_the_name(self):
        found = {item.path: item.quantisation for item in self.listing()}
        self.assertEqual(found["Qwen3-1.7B-Q4_K_M.gguf"], "Q4_K_M")
        self.assertEqual(found["Qwen3-1.7B-IQ1_S.gguf"], "IQ1_S")
        self.assertEqual(found["Qwen3-1.7B-BF16.gguf"], "BF16")

    def test_a_bad_repository_name_is_refused_before_any_request(self):
        with mock.patch.object(hub.requests, "get", side_effect=AssertionError("asked")):
            for bad in ("", "no-slash", "too/many/slashes"):
                with self.assertRaises(HubError):
                    files(bad)


class EstimateTests(unittest.TestCase):
    """The number that makes the listing worth having."""

    def test_a_small_model_is_roughly_its_weights_plus_overhead(self):
        # 1.1 GB file -> 0.8 x that, plus cache and headroom.
        self.assertAlmostEqual(estimate_ram_mb(1_100_000_000), 1289, delta=40)

    def test_a_large_model_is_penalised_for_not_fitting(self):
        """Past 3 GB a model stops fitting beside the operating system, and
        the plain formula stops describing it."""
        big = estimate_ram_mb(5_000_000_000)
        plain = int(5_000_000_000 / (1024 * 1024) * 0.8) + 200 + 250
        self.assertGreater(big, plain * 1.3)

    def test_the_estimate_rises_with_the_file(self):
        sizes = [estimate_ram_mb(n) for n in (500_000_000, 2_000_000_000, 8_000_000_000)]
        self.assertEqual(sizes, sorted(sizes))


class SafeNameTests(unittest.TestCase):
    def test_a_path_cannot_escape_the_models_folder(self):
        """The repository controls this string, and '../' is a valid one as
        far as their API is concerned."""
        self.assertEqual(_safe_name("../../.ssh/authorized_keys.gguf"), "authorized_keys.gguf")

    def test_only_gguf_is_accepted(self):
        for bad in ("model.bin", "notes.txt", "run.exe"):
            with self.assertRaises(HubError):
                _safe_name(bad)

    def test_an_ordinary_name_survives_intact(self):
        self.assertEqual(_safe_name("Qwen3-1.7B-Q4_K_M.gguf"), "Qwen3-1.7B-Q4_K_M.gguf")


class _Serve(BaseHTTPRequestHandler):
    """Serves a small body slowly enough to be cancelled."""

    BODY = b"GGUF" + b"x" * 200_000
    DELAY = 0.0

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.BODY)))
        self.end_headers()
        for start in range(0, len(self.BODY), 20_000):
            try:
                self.wfile.write(self.BODY[start : start + 20_000])
                self.wfile.flush()
            except OSError:
                return
            time.sleep(self.DELAY)


class DownloadTests(unittest.TestCase):
    """The machinery, against a local server rather than a real 2 GB file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        _Serve.DELAY = 0.0
        self.server = HTTPServer(("127.0.0.1", 0), _Serve)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        # Both: shutdown stops the loop, server_close releases the socket.
        # Without the second, every test in this class leaks one and the run
        # ends in a wall of ResourceWarnings.
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]

        patch = mock.patch.object(hub, "HOST", f"http://127.0.0.1:{self.port}")
        patch.start()
        self.addCleanup(patch.stop)

        self.downloads = Downloads(self.tmp)

    def wait(self, download, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline and download.state == "running":
            time.sleep(0.02)

    def test_a_download_lands_in_the_models_folder(self):
        download = self.downloads.start(
            "owner/repo", "model-Q4_K_M.gguf", size_bytes=len(_Serve.BODY)
        )
        self.wait(download)

        self.assertEqual(download.state, "done", download.error)
        landed = self.tmp / "model-Q4_K_M.gguf"
        self.assertTrue(landed.is_file())
        self.assertEqual(landed.stat().st_size, len(_Serve.BODY))

    def test_nothing_is_left_named_like_a_model_until_it_is_complete(self):
        """A half-written file that discovery picks up is a model that fails
        to load for reasons nothing explains."""
        _Serve.DELAY = 0.05
        download = self.downloads.start(
            "owner/repo", "slow.gguf", size_bytes=len(_Serve.BODY)
        )
        time.sleep(0.12)
        self.assertFalse((self.tmp / "slow.gguf").exists())
        self.assertTrue(list(self.tmp.glob("*.part")))

        self.wait(download)
        self.assertTrue((self.tmp / "slow.gguf").is_file())
        self.assertFalse(list(self.tmp.glob("*.part")))

    def test_cancelling_stops_it_and_leaves_nothing_behind(self):
        _Serve.DELAY = 0.08
        download = self.downloads.start(
            "owner/repo", "cancel-me.gguf", size_bytes=len(_Serve.BODY)
        )
        time.sleep(0.1)
        self.assertTrue(self.downloads.cancel(download.id))
        self.wait(download)

        self.assertEqual(download.state, "cancelled")
        self.assertFalse((self.tmp / "cancel-me.gguf").exists())
        self.assertEqual(list(self.tmp.glob("*.part")), [])

    def test_progress_is_reported_while_it_runs(self):
        download = self.downloads.start(
            "owner/repo", "watch.gguf", size_bytes=len(_Serve.BODY)
        )
        self.wait(download)
        reported = download.as_dict()
        self.assertEqual(reported["percent"], 100.0)
        self.assertEqual(reported["seen_bytes"], len(_Serve.BODY))

    def test_two_at_once_are_refused(self):
        _Serve.DELAY = 0.05
        first = self.downloads.start("owner/repo", "first.gguf", size_bytes=len(_Serve.BODY))
        with self.assertRaises(HubError) as caught:
            self.downloads.start("owner/repo", "second.gguf", size_bytes=1)
        self.assertIn("one at a time", str(caught.exception))
        self.wait(first)

    def test_a_file_already_present_is_refused(self):
        (self.tmp / "there.gguf").write_bytes(b"already")
        with self.assertRaises(HubError) as caught:
            self.downloads.start("owner/repo", "there.gguf", size_bytes=10)
        self.assertIn("already", str(caught.exception))

    def test_a_download_bigger_than_the_disk_is_refused_before_it_starts(self):
        """An hour in is the wrong moment to find out, and a full volume takes
        the machine with it."""
        with self.assertRaises(HubError) as caught:
            self.downloads.start(
                "owner/repo", "huge.gguf", size_bytes=500_000_000_000_000
            )
        self.assertIn("disk", str(caught.exception).lower())

    def test_a_truncated_download_fails_rather_than_being_kept(self):
        """Claiming more bytes than arrive is how a corrupt model gets saved
        and then blamed on llama.cpp."""
        download = self.downloads.start(
            "owner/repo", "short.gguf", size_bytes=len(_Serve.BODY) * 2
        )
        self.wait(download)
        self.assertEqual(download.state, "failed")
        self.assertIn("interrupted", download.error)
        self.assertFalse((self.tmp / "short.gguf").exists())


if __name__ == "__main__":
    unittest.main()
