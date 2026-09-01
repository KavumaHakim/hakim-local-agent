"""Finding and fetching GGUF models from Hugging Face.

The one thing setup deliberately does not do for you is choose a model, because
it depends on your RAM, your language and what you want the agent for. This is
the other half of that: a way to look without leaving the app, and to be told
whether a 4.8 GB file is going to fit *before* spending an hour downloading it.

Three things shape the design.

**The size is known before the bytes are.** The file listing gives an exact
byte count, so the same arithmetic `models.discovery` uses on a downloaded file
can be applied to one that has not been downloaded yet. Being told "this needs
about 6.2 GB free and you have 2.1" is worth more than any download speed.

**A download must not block a turn.** They take minutes to hours on a domestic
connection, so they run on their own thread and are polled, exactly as turns
are. The one-at-a-time rule is the same as the model manager's, for the same
reason: two large downloads on one connection finish no sooner and fill the
disk twice as fast.

**Nothing is trusted from the far end.** Only huggingface.co, only over HTTPS,
only paths ending in .gguf, and the saved name is rebuilt from the basename
rather than taken as given. The file lands under a temporary name and is
renamed into place only once it is complete, so an interrupted download can
never look like a working model.
"""

from __future__ import annotations

import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

HOST = "https://huggingface.co"
API = f"{HOST}/api/models"

# Sent so Hugging Face can see what is calling. No key: everything reachable
# here is public, and a gated repository is reported as gated rather than
# prompting for credentials this application has no business holding.
HEADERS = {"User-Agent": "hakim-local-agent"}

TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60

# Matches the estimate in models.discovery, which was measured rather than
# guessed: GLM-OCR's 906 MB file occupies 683 MB resident.
WEIGHT_RESIDENCY = 0.8
HEADROOM_MB = 250
LARGE_MODEL_MB = 3000
LARGE_MODEL_PADDING = 0.5
# A stand-in for the KV cache, which cannot be known without the GGUF header
# and therefore without the file. Mid-sized rather than zero: an estimate that
# ignores the cache is wrong in the direction that fills someone's RAM.
ASSUMED_CACHE_MB = 200

# Disk left over after the file has landed. A download that fills the volume
# to the last byte takes the machine down with it.
DISK_MARGIN_MB = 500


class HubError(Exception):
    """Searching or downloading failed, with something worth reading."""


def _safe_name(path: str) -> str:
    """The filename to save under, rebuilt rather than trusted.

    A repository controls these strings, and `../../.ssh/authorized_keys` is a
    valid one as far as their API is concerned.
    """
    name = Path(str(path)).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    if not cleaned.lower().endswith(".gguf"):
        raise HubError(f"Not a GGUF file: {path}")
    return cleaned


def estimate_ram_mb(file_bytes: int) -> int:
    """Free RAM this model would want, from its size alone.

    The same shape as `discovery.estimate_min_free_mb`, minus the parts that
    need the GGUF header. Deliberately an over-estimate: being wrong high
    shows a warning, and being wrong low fills someone's memory.
    """
    weights_mb = int(file_bytes / (1024 * 1024) * WEIGHT_RESIDENCY)
    total = weights_mb + ASSUMED_CACHE_MB + HEADROOM_MB
    if weights_mb >= LARGE_MODEL_MB:
        total += int(weights_mb * LARGE_MODEL_PADDING)
    return total


@dataclass(frozen=True)
class HubModel:
    """One repository in a search result."""

    id: str
    downloads: int = 0
    likes: int = 0
    gated: bool = False
    tags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "downloads": self.downloads,
            "likes": self.likes,
            "gated": self.gated,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class HubFile:
    """One GGUF inside a repository."""

    path: str
    size_bytes: int

    @property
    def quantisation(self) -> str:
        """The Q4_K_M-ish part of the name, for grouping in the UI."""
        found = re.search(
            r"(IQ\d[A-Z_]*|Q\d[A-Z0-9_]*|BF16|F16|F32)", self.path, re.IGNORECASE
        )
        return found.group(1).upper() if found else ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "quantisation": self.quantisation,
            "needs_ram_mb": estimate_ram_mb(self.size_bytes),
        }


def _get(url: str, **params: Any) -> Any:
    try:
        response = requests.get(url, params=params or None, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise HubError(f"Could not reach Hugging Face: {exc}") from None

    if response.status_code == 401:
        raise HubError(
            "That repository needs a Hugging Face login. Accept its terms on "
            "the website and download the file by hand into weights/."
        )
    if response.status_code == 404:
        raise HubError("No such model on Hugging Face.")
    if response.status_code == 429:
        raise HubError("Hugging Face is rate-limiting this connection. Wait a minute.")
    if response.status_code >= 400:
        raise HubError(f"Hugging Face returned HTTP {response.status_code}.")

    try:
        return response.json()
    except ValueError:
        raise HubError("Hugging Face sent something that is not JSON.") from None


def search(query: str, *, limit: int = 20) -> list[HubModel]:
    """Repositories matching `query` that carry GGUF files.

    Sorted by downloads rather than relevance: for a given model there are
    dozens of near-identical re-uploads, and the popular one is overwhelmingly
    the one that is complete, correctly converted and still there next month.
    """
    query = str(query or "").strip()
    if not query:
        return []

    raw = _get(
        API,
        search=query,
        filter="gguf",
        sort="downloads",
        direction=-1,
        limit=max(1, min(int(limit), 50)),
    )
    if not isinstance(raw, list):
        return []

    found = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        identifier = entry.get("id") or entry.get("modelId")
        if not identifier:
            continue
        found.append(
            HubModel(
                id=str(identifier),
                downloads=int(entry.get("downloads") or 0),
                likes=int(entry.get("likes") or 0),
                gated=bool(entry.get("gated")),
                tags=tuple(str(tag) for tag in (entry.get("tags") or [])),
            )
        )
    return found


def files(repo: str) -> list[HubFile]:
    """The GGUF files in one repository, smallest first.

    Smallest first because on the hardware this is built for, the smallest
    usable quantisation is nearly always the right answer, and a list that
    opens with a 30 GB BF16 file is a list nobody reads to the end of.
    """
    repo = str(repo or "").strip().strip("/")
    if not repo or repo.count("/") != 1:
        raise HubError(f"Not a repository name: {repo!r}. Expected owner/name.")

    raw = _get(f"{API}/{repo}/tree/main")
    if not isinstance(raw, list):
        return []

    found = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path") or "")
        if not path.lower().endswith(".gguf"):
            continue
        found.append(HubFile(path=path, size_bytes=int(entry.get("size") or 0)))

    found.sort(key=lambda item: item.size_bytes)
    return found


@dataclass
class Download:
    """One model being fetched, and how far it has got."""

    repo: str
    path: str
    target: Path
    total_bytes: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    seen_bytes: int = 0
    state: str = "running"  # running | done | failed | cancelled
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        elapsed = (self.finished or time.time()) - self.started
        rate = self.seen_bytes / elapsed if elapsed > 1 else 0.0
        remaining = max(0, self.total_bytes - self.seen_bytes)
        return {
            "id": self.id,
            "repo": self.repo,
            "path": self.path,
            "name": self.target.name,
            "state": self.state,
            "error": self.error,
            "seen_bytes": self.seen_bytes,
            "total_bytes": self.total_bytes,
            "percent": (
                round(100 * self.seen_bytes / self.total_bytes, 1)
                if self.total_bytes
                else 0.0
            ),
            "bytes_per_second": int(rate),
            "seconds_left": int(remaining / rate) if rate > 0 else None,
        }


class Downloads:
    """Fetches models in the background, one at a time.

    One at a time for the reason the model manager runs one server at a time:
    two large downloads over one connection finish no sooner together than in
    turn, and they fill the disk twice as fast while doing it.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()
        self._downloads: dict[str, Download] = {}
        self._cancelled: set[str] = set()
        self._thread: threading.Thread | None = None

    # --- inspection ---

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            items = sorted(self._downloads.values(), key=lambda d: d.started, reverse=True)
            return [item.as_dict() for item in items]

    def busy(self) -> bool:
        with self._lock:
            return any(d.state == "running" for d in self._downloads.values())

    def get(self, download_id: str) -> Download | None:
        with self._lock:
            return self._downloads.get(download_id)

    # --- the checks that happen before any bytes move ---

    def check(self, size_bytes: int, name: str) -> None:
        """Refuse a download that cannot work, before it starts.

        Disk is checked rather than discovered: filling the volume mid-download
        takes the whole machine with it, and the error when it happens names
        the wrong culprit.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / name).exists():
            raise HubError(f"{name} is already in weights/.")

        try:
            free = shutil.disk_usage(self.directory).free
        except OSError:
            return  # cannot tell; not a reason to refuse

        needed = size_bytes + DISK_MARGIN_MB * 1024 * 1024
        if free < needed:
            raise HubError(
                f"Not enough disk. {name} is {size_bytes / 1e9:.1f} GB and "
                f"there is {free / 1e9:.1f} GB free; {DISK_MARGIN_MB} MB is "
                f"kept spare so a full disk does not take the machine down."
            )

    # --- running one ---

    def start(self, repo: str, path: str, size_bytes: int = 0) -> Download:
        if self.busy():
            raise HubError(
                "A download is already running. They go one at a time - two "
                "over one connection finish no sooner and fill the disk twice "
                "as fast."
            )

        name = _safe_name(path)
        self.check(size_bytes, name)

        download = Download(
            repo=repo,
            path=path,
            target=self.directory / name,
            total_bytes=int(size_bytes),
        )
        with self._lock:
            self._downloads[download.id] = download

        self._thread = threading.Thread(
            target=self._run, args=(download,), name="model-download", daemon=True
        )
        self._thread.start()
        return download

    def cancel(self, download_id: str) -> bool:
        with self._lock:
            download = self._downloads.get(download_id)
            if download is None or download.state != "running":
                return False
            self._cancelled.add(download_id)
        return True

    def _run(self, download: Download) -> None:
        partial = download.target.with_suffix(download.target.suffix + ".part")
        url = f"{HOST}/{download.repo}/resolve/main/{download.path}"

        try:
            with requests.get(
                url, headers=HEADERS, stream=True, timeout=DOWNLOAD_TIMEOUT
            ) as response:
                if response.status_code in (401, 403):
                    raise HubError(
                        "That file needs a Hugging Face login. Accept the "
                        "repository's terms on the website, then download it "
                        "by hand into weights/."
                    )
                if response.status_code >= 400:
                    raise HubError(f"Download returned HTTP {response.status_code}.")

                declared = int(response.headers.get("Content-Length") or 0)
                if declared and not download.total_bytes:
                    download.total_bytes = declared

                # 64 KB rather than a megabyte. Cancellation is only checked
                # between blocks, and on the slow connection where somebody
                # actually wants to cancel, a 1 MB block is seven seconds of a
                # button that looks broken.
                with partial.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1 << 16):
                        with self._lock:
                            if download.id in self._cancelled:
                                raise _Cancelled()
                        handle.write(block)
                        download.seen_bytes += len(block)

            if download.total_bytes and download.seen_bytes != download.total_bytes:
                raise HubError(
                    f"Got {download.seen_bytes:,} bytes, expected "
                    f"{download.total_bytes:,}. The connection was interrupted."
                )

            # Renamed only now. A .part file cannot be mistaken for a model,
            # and discovery never sees a half-written one.
            partial.replace(download.target)
            download.state = "done"

        except _Cancelled:
            partial.unlink(missing_ok=True)
            download.state = "cancelled"
        except HubError as exc:
            partial.unlink(missing_ok=True)
            download.state = "failed"
            download.error = str(exc)
        except (requests.RequestException, OSError) as exc:
            partial.unlink(missing_ok=True)
            download.state = "failed"
            download.error = f"Download failed: {exc}"
        finally:
            download.finished = time.time()
            with self._lock:
                self._cancelled.discard(download.id)


class _Cancelled(Exception):
    """Someone pressed cancel. Not an error, so it is not reported as one."""
