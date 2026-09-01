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
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Where the binaries land. Inside the project so nothing is installed system
# wide, and git-ignored so a 60 MB toolchain never enters the repository.
VENDOR = ROOT / "vendor" / "llama"

# Backends worth fetching. Each gets its own directory so both can be present
# at once, which is the whole point of having the second: on an integrated GPU
# the only way to know whether offloading helps is to measure it against the
# CPU build on the same machine.
#
# Only the plain CPU build is searched for automatically. Anything else has to
# be asked for and then pointed at deliberately, because an accelerator build
# that cannot reach its device is slower than the CPU one, not faster.
BACKENDS = {
    "cpu": {
        "windows": "win-cpu",
        "linux": "ubuntu",
        "darwin": "macos",
        "directory": "llama",
    },
    "vulkan": {
        "windows": "win-vulkan",
        "linux": "ubuntu-vulkan",
        # macOS has no Vulkan build: Metal is compiled into the ordinary one.
        "darwin": "",
        "directory": "llama-vulkan",
    },
}


def vendor_dir(backend: str) -> Path:
    return ROOT / "vendor" / BACKENDS[backend]["directory"]

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


def platform_tokens(backend: str = "cpu") -> tuple[str, str]:
    """The (platform, architecture) tokens that appear in an asset name."""
    if backend not in BACKENDS:
        raise LlamaError(
            f"Unknown backend {backend!r}. Known: {', '.join(sorted(BACKENDS))}."
        )
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
    token = BACKENDS[backend].get(system)
    if token is None:
        raise LlamaError(f"Unsupported system: {platform.system()}")
    if not token:
        raise LlamaError(
            f"There is no {backend} build for {platform.system()}. On macOS, "
            f"Metal is compiled into the ordinary build already."
        )
    return token, architecture


def wanted(name: str, system: str, architecture: str, backend: str = "cpu") -> bool:
    """Whether one asset is the build this machine asked for.

    Every accelerator except the one being fetched is excluded by name. That
    matters more than it looks: "ubuntu-vulkan-x64" and "ubuntu-x64" both
    contain "ubuntu", so without the exclusions a request for the CPU build
    would happily match a 34 MB Vulkan one.
    """
    lowered = name.lower()
    if not lowered.startswith("llama-") or "-bin-" not in lowered:
        return False
    if any(token in lowered for token in NOT_A_BUILD):
        return False
    unwanted = [token for token in ACCELERATORS if token != backend]
    if any(token in lowered for token in unwanted):
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


def find_asset(build: str | None, backend: str = "cpu") -> tuple[str, str, str, int]:
    """(tag, asset name, download url, size) for this machine.

    `build` pins a tag such as "b10731"; without it the newest release that
    actually carries binaries wins.
    """
    system, architecture = platform_tokens(backend)

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
            if wanted(name, system, architecture, backend):
                return (
                    release.get("tag_name", "?"),
                    name,
                    asset["browser_download_url"],
                    int(asset.get("size", 0)),
                )

    raise LlamaError(
        f"No {backend} build for {system}-{architecture} was found in the last "
        f"{len(releases)} release(s). Download one by hand from "
        f"https://github.com/ggml-org/llama.cpp/releases"
    )


# --- fetching and unpacking -----------------------------------------------


# How many times to pick a download back up before giving in, and how long to
# wait between tries. Measured need rather than caution: this archive failed
# three times in a row on a domestic connection, at 2 MB, then 20 MB, then
# part-way again, each time with the connection simply dropping.
ATTEMPTS = 5
BACKOFF_SECONDS = 2.0

# Per-attempt timeout. Generous because it covers a whole 35 MB transfer on a
# slow link, and a resumed attempt is shorter than the one before it.
DOWNLOAD_TIMEOUT = 300


def download(
    url: str, target: Path, size: int, on_progress=None, attempts: int = ATTEMPTS
) -> None:
    """Fetch `url` into `target`, picking up where it left off if cut off.

    A dropped connection part-way through a 35 MB archive used to lose all of
    it. With `Range` the bytes already on disk are kept and the rest is asked
    for, which is the difference between a download that eventually finishes on
    a bad link and one that never does.

    Two things the server may do have to be handled rather than assumed:
    answering 200 to a ranged request, which means it ignored the range and is
    sending the whole file again, and answering 416, which means there was
    nothing left to send.
    """
    bar = on_progress("downloading", size) if on_progress else None
    if bar is None:
        say(f"  downloading {size / 1e6:.1f} MB ...")

    session = _requests()
    done = 0
    last_error = ""

    for attempt in range(attempts):
        try:
            if session is not None:
                done = _fetch_requests(session, url, target, done, bar)
            else:
                done = _fetch_urllib(url, target, done, bar)

            if not size or done >= size:
                break
            # A clean end short of the total is still a truncated file.
            last_error = f"stopped at {done:,} of {size:,} bytes"
        except Exception as exc:  # noqa: BLE001 - any transport failure retries
            last_error = str(exc)
            done = target.stat().st_size if target.exists() else 0

        if attempt < attempts - 1:
            remaining = f"{(size - done) / 1e6:.1f} MB left" if size else ""
            say(f"  connection dropped; resuming {remaining}")
            time.sleep(BACKOFF_SECONDS)
    else:
        raise LlamaError(f"Download failed after {attempts} tries: {last_error}")

    if bar is not None:
        bar.done()

    if size and target.stat().st_size != size:
        raise LlamaError(
            f"Got {target.stat().st_size:,} bytes, expected {size:,}, after "
            f"{attempts} tries. Try again later, or download it by hand."
        )


def _range_headers(done: int) -> dict:
    headers = dict(HEADERS)
    if done:
        headers["Range"] = f"bytes={done}-"
    return headers


def _fetch_requests(session, url: str, target: Path, done: int, bar) -> int:
    with session.get(
        url, headers=_range_headers(done), stream=True, timeout=DOWNLOAD_TIMEOUT
    ) as response:
        if response.status_code == 416:
            return done  # nothing left to send
        if response.status_code >= 400:
            raise LlamaError(f"Download returned HTTP {response.status_code}")

        # 206 means the range was honoured; a plain 200 to a ranged request
        # means it was not, and the file is arriving from the beginning again.
        append = done > 0 and response.status_code == 206
        if not append:
            done = 0
            if bar is not None:
                bar.seen = 0

        with target.open("ab" if append else "wb") as handle:
            for block in response.iter_content(chunk_size=1 << 16):
                handle.write(block)
                done += len(block)
                if bar is not None:
                    bar.advance(len(block))
    return done


def _fetch_urllib(url: str, target: Path, done: int, bar) -> int:
    request = urllib.request.Request(url, headers=_range_headers(done))
    try:
        response = urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            return done
        raise

    with response:
        append = done > 0 and response.status == 206
        if not append:
            done = 0
            if bar is not None:
                bar.seen = 0
        with target.open("ab" if append else "wb") as handle:
            while True:
                block = response.read(1 << 16)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                if bar is not None:
                    bar.advance(len(block))
    return done


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


def install(
    *,
    build: str | None = None,
    force: bool = False,
    on_progress=None,
    backend: str = "cpu",
) -> Path | None:
    """Download and unpack llama.cpp. Returns the server path, or None."""
    target = vendor_dir(backend)

    existing = find_server(target)
    if existing and not force:
        say(f"  already there: {existing}")
        return existing

    tag, name, url, size = find_asset(build, backend)
    say(f"  {name}  ({tag})")

    if target.exists() and force:
        shutil.rmtree(target, ignore_errors=True)

    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / name
        download(url, archive, size, on_progress=on_progress)
        say("  unpacking ...")
        unpack(archive, target)

    server = find_server(target)
    if server is None:
        raise LlamaError(
            f"The archive unpacked but contained no llama-server. "
            f"Look in {target} and report this."
        )

    make_executable(server)
    version = verify(server)
    say(f"  {version}")
    (target / "BUILD.txt").write_text(
        f"{tag}\n{name}\n{url}\n", encoding="utf-8"
    )

    if backend != "cpu":
        say("")
        say(f"  This is the {backend} build, and nothing uses it yet.")
        say("  Only vendor/llama is searched automatically, so point at it")
        say("  deliberately - setup.py's 'I already have it', or:")
        say(f"      {server}")
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
    parser.add_argument(
        "--backend",
        default="cpu",
        choices=sorted(BACKENDS),
        help=(
            "which build to fetch. 'vulkan' is for benchmarking an integrated "
            "GPU; it lands beside the CPU build rather than replacing it, and "
            "must be pointed at deliberately."
        ),
    )
    arguments = parser.parse_args()

    try:
        if arguments.list:
            tag, name, url, size = find_asset(arguments.build, arguments.backend)
            say(f"{tag}  {name}  {size / 1e6:.1f} MB")
            say(url)
            return 0

        server = install(
            build=arguments.build, force=arguments.force, backend=arguments.backend
        )
        say(f"llama-server: {server}")
    except LlamaError as exc:
        say(f"[X] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
