"""Fetch a llama.cpp release build, so setup leaves only the model to you.

Run by `scripts/setup.py`, and usable on its own to update later:

    python scripts/get_llama.py            # newest build
    python scripts/get_llama.py --build b10731
    python scripts/get_llama.py --list     # what is on offer for this machine

Three things about llama.cpp's releases shape everything here.

**The binary releases are prereleases.** `/releases/latest` returns a tag with
no binaries at all, so the release list has to be walked until one with assets
turns up. Asking for "latest" the obvious way silently finds nothing.

**There is a build per platform, per accelerator.** Only the plain CPU builds
are wanted: a CUDA or ROCm build is ten times the size and needs a runtime this
machine does not have. So accelerator builds are excluded by name rather than
hoped against.

**Windows ships .zip and everything else .tar.gz**, which is why both are
handled.

No checksums are published alongside these archives, so there is nothing to
verify them against. What is done instead: HTTPS to GitHub, the archive must
open as an archive, it must contain a llama-server, and that binary must answer
`--version` before this reports success. That is weaker than a signature and it
is said plainly rather than implied.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Where the binaries land. Inside the project so nothing is installed system
# wide, and git-ignored so a 60 MB toolchain never enters the repository.
VENDOR = ROOT / "vendor" / "llama"

RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases"

# Accelerator builds. Each needs a runtime that has to be installed separately,
# and each is several times the size of the CPU build for a machine that cannot
# use it.
ACCELERATORS = (
    "cuda",
    "cudart",
    "rocm",
    "hip",
    "vulkan",
    "sycl",
    "openvino",
    "opencl",
    "musa",
    "cann",
)

# Not a server build at all.
NOT_A_BUILD = ("xcframework", "android", "-ui.", "nightly-tag")

# How many releases to look through before giving up. Binary releases are
# frequent, so the first page is always enough in practice; the bound exists so
# a change upstream cannot turn this into an unbounded crawl.
MAX_RELEASES = 12


class LlamaError(Exception):
    """Something went wrong fetching or unpacking a build."""


def say(message: str = "") -> None:
    print(message, flush=True)


# --- choosing the right archive -------------------------------------------


def platform_tokens() -> tuple[str, str]:
    """The (platform, architecture) tokens that appear in an asset name."""
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64", "x64"):
        architecture = "x64"
    elif machine in ("arm64", "aarch64"):
        architecture = "arm64"
    else:
        raise LlamaError(
            f"No llama.cpp build is published for {machine}. Build it from "
            f"source: https://github.com/ggml-org/llama.cpp"
        )

    system = platform.system().lower()
    if system == "windows":
        return "win-cpu", architecture
    if system == "linux":
        return "ubuntu", architecture
    if system == "darwin":
        return "macos", architecture
    raise LlamaError(f"Unsupported system: {platform.system()}")


def wanted(name: str, system: str, architecture: str) -> bool:
    """Whether one asset is the plain CPU build for this machine."""
    lowered = name.lower()
    if not lowered.startswith("llama-") or "-bin-" not in lowered:
        return False
    if any(token in lowered for token in NOT_A_BUILD):
        return False
    if any(token in lowered for token in ACCELERATORS):
        return False
    if system not in lowered:
        return False
    # "x64" must not match inside "arm64"; compare the tail before the suffix.
    stem = lowered.split("-bin-", 1)[1]
    for suffix in (".zip", ".tar.gz", ".tgz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.endswith(architecture)


HEADERS = {
    "User-Agent": "hakim-local-agent-setup",
    "Accept": "application/vnd.github+json",
}


def _requests():
    """`requests` if it is importable, else None.

    It is a dependency of the project, so it is there whenever this is run
    through the virtualenv - which is how setup.py calls it. Preferred because
    urllib mishandles GitHub's chunked responses on some connections, failing
    with IncompleteRead part-way through a perfectly good reply. urllib
    remains the fallback so this still works before anything is installed.
    """
    try:
        import requests
    except ImportError:
        return None
    return requests


def fetch_json(url: str, params: str = "") -> object:
    session = _requests()
    if session is not None:
        try:
            response = session.get(url + params, headers=HEADERS, timeout=60)
        except Exception as exc:  # noqa: BLE001 - requests' own error tree
            raise LlamaError(f"Could not reach GitHub: {exc}") from None
        if response.status_code in (403, 429):
            raise LlamaError(
                "GitHub is rate-limiting this connection. Wait a few minutes, "
                "or download a build by hand from "
                "https://github.com/ggml-org/llama.cpp/releases"
            )
        if response.status_code >= 400:
            raise LlamaError(f"GitHub returned HTTP {response.status_code} for {url}")
        try:
            return response.json()
        except ValueError as exc:
            raise LlamaError(f"GitHub sent something that is not JSON: {exc}") from None

    request = urllib.request.Request(url + params, headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            raise LlamaError(
                "GitHub is rate-limiting this connection. Wait a few minutes, "
                "or download a build by hand from "
                "https://github.com/ggml-org/llama.cpp/releases"
            ) from None
        raise LlamaError(f"GitHub returned HTTP {exc.code} for {url}") from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise LlamaError(f"Could not reach GitHub: {exc}") from None


def find_asset(build: str | None) -> tuple[str, str, str, int]:
    """(tag, asset name, download url, size) for this machine.

    `build` pins a tag such as "b10731"; without it the newest release that
    actually carries binaries wins.
    """
    system, architecture = platform_tokens()

    if build:
        releases = [fetch_json(f"{RELEASES}/tags/{build}")]
    else:
        releases = fetch_json(RELEASES, f"?per_page={MAX_RELEASES}")
        if not isinstance(releases, list):
            raise LlamaError("GitHub returned something unexpected for the releases.")

    for release in releases:
        if not isinstance(release, dict):
            continue
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if wanted(name, system, architecture):
                return (
                    release.get("tag_name", "?"),
                    name,
                    asset["browser_download_url"],
                    int(asset.get("size", 0)),
                )

    raise LlamaError(
        f"No CPU build for {system}-{architecture} was found in the last "
        f"{len(releases)} release(s). Download one by hand from "
        f"https://github.com/ggml-org/llama.cpp/releases"
    )


# --- fetching and unpacking -----------------------------------------------


def download(url: str, target: Path, size: int) -> None:
    say(f"  downloading {size / 1e6:.1f} MB ...")
    session = _requests()

    if session is not None:
        try:
            with session.get(url, headers=HEADERS, stream=True, timeout=300) as response:
                if response.status_code >= 400:
                    raise LlamaError(f"Download returned HTTP {response.status_code}")
                with target.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1 << 20):
                        handle.write(block)
        except LlamaError:
            raise
        except Exception as exc:  # noqa: BLE001 - requests' own error tree
            raise LlamaError(f"Download failed: {exc}") from None
    else:
        request = urllib.request.Request(
            url, headers={"User-Agent": "hakim-local-agent-setup"}
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                with target.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1 << 20)
        except (urllib.error.URLError, OSError) as exc:
            raise LlamaError(f"Download failed: {exc}") from None

    if size and target.stat().st_size != size:
        raise LlamaError(
            f"Downloaded {target.stat().st_size} bytes, expected {size}. "
            f"The connection was probably interrupted; try again."
        )


def unpack(archive: Path, into: Path) -> None:
    """Extract an archive, refusing any member that escapes `into`.

    Both formats let a member name its own path, and a crafted archive can name
    one outside the directory it is being unpacked into. tarfile's data filter
    handles that where it exists; the zip side is checked by hand because
    zipfile has no equivalent.
    """
    into.mkdir(parents=True, exist_ok=True)

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                destination = (into / member).resolve()
                if into.resolve() not in destination.parents and destination != into.resolve():
                    raise LlamaError(f"Archive member escapes the target: {member}")
            bundle.extractall(into)
        return

    with tarfile.open(archive, "r:*") as bundle:
        try:
            bundle.extractall(into, filter="data")
        except TypeError:
            # Python without the extraction filter: check every member first.
            for member in bundle.getmembers():
                destination = (into / member.name).resolve()
                if into.resolve() not in destination.parents:
                    raise LlamaError(
                        f"Archive member escapes the target: {member.name}"
                    ) from None
            bundle.extractall(into)


def find_server(root: Path) -> Path | None:
    """The llama-server binary somewhere under `root`."""
    names = ("llama-server.exe", "llama-server")
    for name in names:
        for found in root.rglob(name):
            if found.is_file():
                return found
    return None


def make_executable(path: Path) -> None:
    """Set the executable bit on POSIX. Archives do not always carry it."""
    if os.name == "nt":
        return
    for sibling in path.parent.iterdir():
        if sibling.is_file() and not sibling.suffix:
            mode = sibling.stat().st_mode
            sibling.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def verify(server: Path) -> str:
    """Run it. An archive that unpacked is not the same as a binary that runs."""
    try:
        completed = subprocess.run(
            [str(server), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LlamaError(f"{server.name} would not run: {exc}") from None

    output = (completed.stderr or completed.stdout or "").strip()
    first = output.splitlines()[0] if output else ""
    # llama-server --version exits non-zero on some builds while still printing
    # the version, so the output is what is trusted, not the code.
    if not first:
        raise LlamaError(
            f"{server.name} ran but said nothing when asked for its version."
        )
    return first


# --- the whole job --------------------------------------------------------


def install(*, build: str | None = None, force: bool = False) -> Path | None:
    """Download and unpack llama.cpp. Returns the server path, or None."""
    existing = find_server(VENDOR)
    if existing and not force:
        say(f"  already there: {existing}")
        return existing

    tag, name, url, size = find_asset(build)
    say(f"  {name}  ({tag})")

    if VENDOR.exists() and force:
        shutil.rmtree(VENDOR, ignore_errors=True)

    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / name
        download(url, archive, size)
        say("  unpacking ...")
        unpack(archive, VENDOR)

    server = find_server(VENDOR)
    if server is None:
        raise LlamaError(
            f"The archive unpacked but contained no llama-server. "
            f"Look in {VENDOR} and report this."
        )

    make_executable(server)
    version = verify(server)
    say(f"  {version}")
    (VENDOR / "BUILD.txt").write_text(
        f"{tag}\n{name}\n{url}\n", encoding="utf-8"
    )
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a llama.cpp CPU build.")
    parser.add_argument("--build", help="pin a release tag, e.g. b10731")
    parser.add_argument(
        "--force", action="store_true", help="re-download even if one is present"
    )
    parser.add_argument(
        "--list", action="store_true", help="show the newest build for this machine"
    )
    arguments = parser.parse_args()

    try:
        if arguments.list:
            tag, name, url, size = find_asset(arguments.build)
            say(f"{tag}  {name}  {size / 1e6:.1f} MB")
            say(url)
            return 0

        server = install(build=arguments.build, force=arguments.force)
        say(f"llama-server: {server}")
    except LlamaError as exc:
        say(f"[X] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
