"""Fetching a llama.cpp build: which asset, and what happens to it.

No network. The asset names below are the real contents of release b10731,
copied verbatim, because the thing worth testing is that the right one is
picked out of the twenty-seven actually published - most of which are
accelerator builds that would download hundreds of megabytes and then need a
runtime this machine does not have.
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import get_llama  # noqa: E402

# Every asset of llama.cpp release b10731, exactly as GitHub lists them.
ASSETS = [
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.3-x64.zip",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip",
    "llama-b10731-bin-android-arm64.tar.gz",
    "llama-b10731-bin-macos-arm64.tar.gz",
    "llama-b10731-bin-macos-x64.tar.gz",
    "llama-b10731-bin-ubuntu-arm64.tar.gz",
    "llama-b10731-bin-ubuntu-openvino-2026.3.1-x64.tar.gz",
    "llama-b10731-bin-ubuntu-rocm-7.14-x64.tar.gz",
    "llama-b10731-bin-ubuntu-s390x.tar.gz",
    "llama-b10731-bin-ubuntu-sycl-fp16-x64.tar.gz",
    "llama-b10731-bin-ubuntu-sycl-fp32-x64.tar.gz",
    "llama-b10731-bin-ubuntu-vulkan-arm64.tar.gz",
    "llama-b10731-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b10731-bin-ubuntu-x64.tar.gz",
    "llama-b10731-bin-win-cpu-arm64.zip",
    "llama-b10731-bin-win-cpu-x64.zip",
    "llama-b10731-bin-win-cuda-12.4-x64.zip",
    "llama-b10731-bin-win-cuda-13.3-x64.zip",
    "llama-b10731-bin-win-cuda-13.4-arm64.zip",
    "llama-b10731-bin-win-opencl-adreno-arm64.zip",
    "llama-b10731-bin-win-openvino-2026.3.1-x64.zip",
    "llama-b10731-bin-win-rocm-7.14-x64.zip",
    "llama-b10731-bin-win-sycl-x64.zip",
    "llama-b10731-bin-win-vulkan-x64.zip",
    "llama-b10731-ui.tar.gz",
    "llama-b10731-xcframework.zip",
]


def picked(system: str, architecture: str, backend: str = "cpu") -> list[str]:
    return [
        name for name in ASSETS if get_llama.wanted(name, system, architecture, backend)
    ]


class AssetChoiceTests(unittest.TestCase):
    def test_exactly_one_build_matches_each_platform(self):
        """Two matches would mean the choice is arbitrary; none would mean the
        setup script silently does nothing."""
        for system, architecture, expected in (
            ("win-cpu", "x64", "llama-b10731-bin-win-cpu-x64.zip"),
            ("win-cpu", "arm64", "llama-b10731-bin-win-cpu-arm64.zip"),
            ("ubuntu", "x64", "llama-b10731-bin-ubuntu-x64.tar.gz"),
            ("ubuntu", "arm64", "llama-b10731-bin-ubuntu-arm64.tar.gz"),
            ("macos", "x64", "llama-b10731-bin-macos-x64.tar.gz"),
            ("macos", "arm64", "llama-b10731-bin-macos-arm64.tar.gz"),
        ):
            with self.subTest(system=system, architecture=architecture):
                self.assertEqual(picked(system, architecture), [expected])

    def test_no_accelerator_build_is_ever_chosen(self):
        """Each needs a runtime that is not installed, and each is several
        times the size of the CPU build."""
        for system, architecture in (
            ("win-cpu", "x64"),
            ("ubuntu", "x64"),
            ("macos", "arm64"),
        ):
            for name in picked(system, architecture):
                lowered = name.lower()
                for token in get_llama.ACCELERATORS:
                    self.assertNotIn(token, lowered)

    def test_x64_does_not_match_arm64(self):
        """"arm64" ends in "64", so a careless suffix test picks the wrong
        binary and it fails only when someone runs it."""
        self.assertNotIn(
            "llama-b10731-bin-win-cpu-arm64.zip", picked("win-cpu", "x64")
        )
        self.assertNotIn(
            "llama-b10731-bin-win-cpu-x64.zip", picked("win-cpu", "arm64")
        )

    def test_the_odds_and_ends_are_not_builds(self):
        for name in ("llama-b10731-ui.tar.gz", "llama-b10731-xcframework.zip"):
            self.assertFalse(get_llama.wanted(name, "ubuntu", "x64"))
            self.assertFalse(get_llama.wanted(name, "win-cpu", "x64"))

    def test_the_cuda_runtime_bundles_are_not_builds(self):
        """They start with cudart- and contain no server at all."""
        for name in ASSETS:
            if name.startswith("cudart-"):
                self.assertFalse(get_llama.wanted(name, "win-cpu", "x64"))

    def test_a_musl_or_s390x_build_is_not_mistaken_for_x64(self):
        self.assertNotIn(
            "llama-b10731-bin-ubuntu-s390x.tar.gz", picked("ubuntu", "x64")
        )

    def test_this_machine_maps_to_a_real_asset(self):
        """Whatever is running the tests must be able to name a build."""
        system, architecture = get_llama.platform_tokens()
        self.assertEqual(len(picked(system, architecture)), 1)


class BackendTests(unittest.TestCase):
    """Fetching an accelerator build on purpose, without ever getting one by
    accident."""

    def test_the_vulkan_build_can_be_asked_for(self):
        self.assertEqual(
            picked("win-vulkan", "x64", "vulkan"),
            ["llama-b10731-bin-win-vulkan-x64.zip"],
        )
        self.assertEqual(
            picked("ubuntu-vulkan", "x64", "vulkan"),
            ["llama-b10731-bin-ubuntu-vulkan-x64.tar.gz"],
        )

    def test_asking_for_vulkan_does_not_also_match_cuda_or_rocm(self):
        """Only the one accelerator asked for is allowed back in."""
        for name in picked("win-vulkan", "x64", "vulkan"):
            for token in ("cuda", "rocm", "sycl", "openvino"):
                self.assertNotIn(token, name.lower())

    def test_the_cpu_build_never_matches_an_accelerator_one(self):
        """'ubuntu-vulkan-x64' contains 'ubuntu', so without the exclusions a
        request for the CPU build would match a 34 MB Vulkan archive."""
        self.assertEqual(
            picked("ubuntu", "x64", "cpu"), ["llama-b10731-bin-ubuntu-x64.tar.gz"]
        )

    def test_each_backend_gets_its_own_directory(self):
        """Both have to be present at once: the only way to know whether an
        integrated GPU helps is to measure it against the CPU build."""
        self.assertNotEqual(
            get_llama.vendor_dir("cpu"), get_llama.vendor_dir("vulkan")
        )
        self.assertEqual(get_llama.vendor_dir("cpu").name, "llama")

    def test_an_unknown_backend_is_refused(self):
        with self.assertRaises(get_llama.LlamaError):
            get_llama.platform_tokens("cuda")

    def test_macos_says_metal_is_already_included(self):
        with mock.patch.object(get_llama.platform, "system", return_value="Darwin"):
            with mock.patch.object(get_llama.platform, "machine", return_value="arm64"):
                with self.assertRaises(get_llama.LlamaError) as caught:
                    get_llama.platform_tokens("vulkan")
        self.assertIn("Metal", str(caught.exception))


class _Flaky(BaseHTTPRequestHandler):
    """Drops the first attempt half-way, then honours Range like a real CDN.

    Not a hypothetical failure: fetching the 35 MB Vulkan archive over a
    domestic connection died three times in a row exactly this way, at 2 MB,
    then 20 MB, then part-way again.
    """

    BODY = b"PK\x03\x04" + bytes(range(256)) * 400
    served = 0
    honour_range = True
    # Set when a test wants every attempt cut short rather than just the
    # first: the link that never finishes however patient you are.
    always_cut = False

    def log_message(self, *args):
        pass

    def do_GET(self):
        type(self).served += 1
        start = 0
        header = self.headers.get("Range")

        if header and self.honour_range:
            start = int(header.split("=", 1)[1].split("-", 1)[0])
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {start}-{len(self.BODY) - 1}/{len(self.BODY)}",
            )
        else:
            self.send_response(200)

        body = self.BODY[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        first = type(self).served == 1
        cut = len(body) // 2 if (first or type(self).always_cut) else len(body)
        try:
            self.wfile.write(body[:cut])
            self.wfile.flush()
        except OSError:
            pass
        if cut < len(body):
            self.close_connection = True


class DownloadResumeTests(unittest.TestCase):
    """Picking the archive back up instead of losing all of it.

    Both transports are exercised. `requests` is what actually runs through
    the virtualenv, but urllib is the fallback for a machine where nothing is
    installed yet - which is precisely the machine running the setup script,
    so it is the path least able to afford being broken.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        _Flaky.served = 0
        _Flaky.honour_range = True
        _Flaky.always_cut = False
        self.server = HTTPServer(("127.0.0.1", 0), _Flaky)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/build.zip"

        # No point waiting two seconds a try in a test.
        backoff = mock.patch.object(get_llama, "BACKOFF_SECONDS", 0.01)
        backoff.start()
        self.addCleanup(backoff.stop)
        quiet = mock.patch.object(get_llama, "say", lambda *a, **k: None)
        quiet.start()
        self.addCleanup(quiet.stop)

    def use_urllib(self):
        patch = mock.patch.object(get_llama, "_requests", lambda: None)
        patch.start()
        self.addCleanup(patch.stop)

    def test_a_dropped_connection_is_picked_back_up(self):
        target = self.tmp / "build.zip"
        get_llama.download(self.url, target, len(_Flaky.BODY))
        self.assertEqual(target.read_bytes(), _Flaky.BODY)

    def test_it_took_more_than_one_request(self):
        """Proof it resumed rather than getting lucky on the first try."""
        get_llama.download(self.url, self.tmp / "build.zip", len(_Flaky.BODY))
        self.assertGreater(_Flaky.served, 1)

    def test_the_urllib_fallback_resumes_too(self):
        self.use_urllib()
        target = self.tmp / "build.zip"
        get_llama.download(self.url, target, len(_Flaky.BODY))
        self.assertEqual(target.read_bytes(), _Flaky.BODY)
        self.assertGreater(_Flaky.served, 1)

    def test_a_server_that_ignores_the_range_starts_over_cleanly(self):
        """A 200 to a ranged request means the whole file is coming again.
        Appending it to what is already on disk would double the archive."""
        _Flaky.honour_range = False
        target = self.tmp / "build.zip"
        get_llama.download(self.url, target, len(_Flaky.BODY))
        self.assertEqual(target.read_bytes(), _Flaky.BODY)

    def test_giving_up_says_so_rather_than_leaving_a_short_file(self):
        """A truncated archive that is treated as complete unpacks into a
        broken toolchain, which is a worse failure than a failed download."""
        _Flaky.always_cut = True
        with self.assertRaises(get_llama.LlamaError) as caught:
            get_llama.download(
                self.url, self.tmp / "build.zip", len(_Flaky.BODY), attempts=2
            )
        self.assertIn("tries", str(caught.exception))


class UnpackTests(unittest.TestCase):
    """What happens to the archive once it is here."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def make_zip(self, members: dict[str, bytes]) -> Path:
        path = self.tmp / "bundle.zip"
        with zipfile.ZipFile(path, "w") as bundle:
            for name, data in members.items():
                bundle.writestr(name, data)
        return path

    def make_tar(self, members: dict[str, bytes]) -> Path:
        path = self.tmp / "bundle.tar.gz"
        with tarfile.open(path, "w:gz") as bundle:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))
        return path

    def test_a_zip_is_unpacked_and_the_server_found(self):
        archive = self.make_zip(
            {
                "build/bin/llama-server.exe": b"binary",
                "build/bin/ggml.dll": b"library",
            }
        )
        into = self.tmp / "out"
        get_llama.unpack(archive, into)

        server = get_llama.find_server(into)
        self.assertIsNotNone(server)
        self.assertEqual(server.name, "llama-server.exe")
        # The shared libraries have to land beside it or it will not start.
        self.assertTrue((server.parent / "ggml.dll").is_file())

    def test_a_tarball_is_unpacked_and_the_server_found(self):
        archive = self.make_tar(
            {"build/bin/llama-server": b"binary", "build/bin/libggml.so": b"library"}
        )
        into = self.tmp / "out"
        get_llama.unpack(archive, into)

        server = get_llama.find_server(into)
        self.assertIsNotNone(server)
        self.assertEqual(server.name, "llama-server")

    def test_a_zip_member_cannot_escape_the_target(self):
        """A crafted archive naming ../ would otherwise write outside the
        directory it is being unpacked into."""
        archive = self.make_zip({"../escaped.txt": b"nope"})
        with self.assertRaises(get_llama.LlamaError):
            get_llama.unpack(archive, self.tmp / "out")
        self.assertFalse((self.tmp / "escaped.txt").exists())

    def test_a_tar_member_cannot_escape_the_target(self):
        archive = self.make_tar({"../escaped.txt": b"nope"})
        with self.assertRaises(Exception):
            get_llama.unpack(archive, self.tmp / "out")
        self.assertFalse((self.tmp / "escaped.txt").exists())

    def test_an_archive_with_no_server_is_noticed(self):
        archive = self.make_zip({"build/bin/README.md": b"nothing here"})
        into = self.tmp / "out"
        get_llama.unpack(archive, into)
        self.assertIsNone(get_llama.find_server(into))

    @unittest.skipIf(os.name == "nt", "no executable bit on Windows")
    def test_the_executable_bit_is_set_on_posix(self):
        """Release tarballs do not always carry it, and without it the binary
        cannot be launched however correct the path is."""
        archive = self.make_tar({"build/bin/llama-server": b"binary"})
        into = self.tmp / "out"
        get_llama.unpack(archive, into)

        server = get_llama.find_server(into)
        get_llama.make_executable(server)
        self.assertTrue(os.access(server, os.X_OK))


if __name__ == "__main__":
    unittest.main()
