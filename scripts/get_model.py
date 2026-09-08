"""Fetch a model to run, and the OCR pair that reads text off an image.

Setup used to stop short of this on the reasoning that choosing a model is
personal - it depends on your RAM, your language and what you want the agent
for. That is still true of the *fifth* model somebody adds. It is not true of
the first one: a fresh clone with no `.gguf` in `weights/` cannot answer a
single message, and "your choice to make" is a poor thing to hand someone who
has not yet seen the thing work. So there is a starter model, offered with its
size stated, and everything beyond it stays a choice.

    weights/gemma-4-E2B-it-Q4_0.gguf          3.0 GB   the starter
    weights/GLM-OCR-Q8_0.gguf                 0.9 GB   OCR, language half
    weights/mmproj-GLM-OCR-Q8_0.gguf          0.5 GB   OCR, vision half

Four things shape this file.

**The repositories are checked, not remembered.** Every entry below was
resolved against the Hugging Face API and its byte count read from the file
listing. The sizes recorded here are only for saying "this will cost 3.0 GB"
before anything starts; the authoritative number is fetched at download time,
because a repository can be re-quantised under a name that never changes, and a
hardcoded total that turns out to be stale is how a *good* download gets thrown
away as truncated.

**A partial file must never look like a model.** Downloads land as
`name.gguf.part` and are renamed into place only once complete. `weights/` is
scanned for `*.gguf` and anything found is offered as something to talk to, so
an interrupted 3 GB download keeping the real name would present itself as a
broken model rather than as an unfinished one. The `.part` survives, which is
what lets `Range` resume pick it up - a 3 GB file over a domestic connection is
exactly the case that needs it.

**GLM-OCR is a pair.** The language half alone loads and then cannot see, which
reads as a broken model rather than a missing file, so both halves are fetched
or neither is claimed.

**Nothing here is required to have run.** Any `.gguf` dropped into `weights/`
by hand works identically - it is measured from its own header. This is a
convenience, not a gate, and it says so when it fails.

The downloading, resuming and progress reporting are `get_llama.py`'s, imported
rather than copied, for the same reason `get_speech.py` imports them: that
module's `download` grew HTTP `Range` resume after a 35 MB archive failed three
times running, and a second copy would be a second place to fix it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Aliased at the boundary: it is the same fetcher, and calling a failed model
# download a LlamaError in an error message would be a small lie.
from get_llama import (  # noqa: E402
    LlamaError as FetchError,
    download,
    say,
)

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"

HOST = "https://huggingface.co"
API = f"{HOST}/api/models"

# No key. Everything here is a public repository, and a gated one is reported
# as gated rather than prompting for credentials a setup script has no business
# holding.
HEADERS = {"User-Agent": "hakim-local-agent"}
API_TIMEOUT = 30

# Disk is checked before a multi-gigabyte download rather than during it:
# filling somebody's disk and then failing is the one outcome worse than not
# starting.
DISK_MARGIN = 500_000_000


# Every entry was resolved against the Hugging Face API and its size read from
# the file listing. Four of the six filenames are ones `models.json` already
# names - gemma, mistral, reasoning and the OCR pair - so those land straight
# on a registry entry with the context and RAM threshold somebody measured.
#
# `fast` and `tiny` do NOT: the copies here were renamed by hand at some point
# (`Qwen3.5-2B-M-TS-...`) and upstream has no such name. They arrive under the
# upstream one and are picked up by discovery instead, which sizes them from
# their own GGUF header. That works - it is the same path any dropped-in file
# takes - but it is worth knowing why two of these six behave differently.
#
# `unsloth` publishes the chat models and `ggml-org` the OCR pair. Both are
# named in the README as good sources; neither is special, and any other
# well-made quantisation of the same model works the same way.
CATALOG: dict[str, dict] = {
    "gemma": {
        "label": "Gemma 4 E2B",
        "repo": "unsloth/gemma-4-E2B-it-GGUF",
        "files": (("gemma-4-E2B-it-Q4_0.gguf", 3_041_378_400),),
        "free_mb": 2627,
        "note": (
            "The starter. Fastest of the local models on 8 GB - measured at "
            "5.9-6.0 tok/s - and its chat template carries tool calls, so the "
            "agent is an agent rather than a chatbot."
        ),
    },
    "mistral": {
        "label": "Ministral 3B",
        "repo": "unsloth/Ministral-3-3B-Instruct-2512-GGUF",
        "files": (("Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", 2_146_497_824),),
        "free_mb": 1900,
        "note": "Mistral's native function-calling template. A close second.",
    },
    "fast": {
        "label": "Qwen3.5 2B",
        "repo": "unsloth/Qwen3.5-2B-GGUF",
        "files": (("Qwen3.5-2B-Q4_K_M.gguf", 1_280_835_840),),
        "free_mb": 1150,
        "note": "Quick answers, short tasks, simple tool calls.",
    },
    "tiny": {
        "label": "Qwen3.5 2B (Q3_K_S)",
        "repo": "unsloth/Qwen3.5-2B-GGUF",
        "files": (("Qwen3.5-2B-Q3_K_S.gguf", 1_030_947_072),),
        "free_mb": 900,
        "note": (
            "Smallest and quickest. A harsher quantisation than the others, so "
            "expect weaker instruction-following."
        ),
    },
    "reasoning": {
        "label": "Qwen3 8B",
        "repo": "unsloth/Qwen3-8B-GGUF",
        "files": (("Qwen3-8B-Q4_K_M.gguf", 5_027_784_512),),
        "free_mb": 6200,
        "note": (
            "Harder reasoning and longer tasks. It does not fit in 8 GB of RAM "
            "and pages from disk - minutes per answer, not seconds."
        ),
    },
    "ocr": {
        "label": "GLM-OCR",
        "repo": "ggml-org/GLM-OCR-GGUF",
        # Both halves, always. The language model alone loads and then cannot
        # see, which is indistinguishable from a broken model.
        "files": (
            ("GLM-OCR-Q8_0.gguf", 950_433_408),
            ("mmproj-GLM-OCR-Q8_0.gguf", 484_403_648),
        ),
        "free_mb": 1150,
        "note": (
            "Reads text off an image and keeps the layout - tables, columns, "
            "handwriting. Runs beside a chat model rather than instead of one. "
            "Tesseract is the faster default for plain text; this is what "
            "earns its cost when the page has structure worth preserving."
        ),
    },
}

# The one a fresh clone is offered. It is `models.json`'s default because it is
# the fastest thing here that can still call a tool.
STARTER = "gemma"

# The order for `--what all` and for the listing: the starter first, then the
# other chat models, then OCR. Not CATALOG order by accident - a listing whose
# order nobody chose reads like one nobody thought about.
ORDER = ("gemma", "mistral", "fast", "tiny", "reasoning", "ocr")


def entry(key: str) -> dict:
    """The catalog entry for `key`, or a message naming the real ones."""
    try:
        return CATALOG[key]
    except KeyError:
        known = ", ".join(ORDER)
        raise FetchError(
            f"Unknown model {key!r}. Known: {known}. Any .gguf from Hugging "
            f"Face works too - put it in weights/ and it is measured from its "
            f"own header."
        ) from None


def human(size: int) -> str:
    """A size somebody can read. Everything in the catalog today is measured
    in gigabytes, but "0.0 GB" for a small file reads like a bug."""
    if size < 1_000_000_000:
        return f"{size / 1e6:.0f} MB"
    return f"{size / 1e9:.1f} GB"


def target_for(name: str) -> Path:
    """Where a file lands. The basename only, never a path from the far end."""
    return WEIGHTS / Path(name).name


def have(key: str) -> bool:
    """True when every file this entry needs is already in weights/."""
    return all(target_for(name).is_file() for name, _ in entry(key)["files"])


def missing(key: str) -> list[tuple[str, int]]:
    """The files this entry still needs, with their recorded sizes."""
    return [
        (name, size)
        for name, size in entry(key)["files"]
        if not target_for(name).is_file()
    ]


def recorded_bytes(key: str, *, only_missing: bool = True) -> int:
    """What this entry costs to fetch, from the sizes recorded above."""
    files = missing(key) if only_missing else list(entry(key)["files"])
    return sum(size for _, size in files)


def live_sizes(repo: str) -> dict[str, int]:
    """Ask Hugging Face how big the files actually are, right now.

    The recorded sizes are for the sentence printed before a download starts.
    This is the one used to decide whether a download finished, because a
    repository can be re-quantised under a name that never changes, and a total
    stale by a few hundred bytes would condemn a perfectly good file.

    Returns {} on any failure - no network, a rate limit, a renamed repository.
    The caller then falls back to the recorded number, which is right far more
    often than it is wrong.
    """
    url = f"{API}/{repo}/tree/main?recursive=true"
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
            listing_json = json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as exc:  # noqa: F841
        return {}

    if not isinstance(listing_json, list):
        return {}

    sizes: dict[str, int] = {}
    for item in listing_json:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if not isinstance(path, str) or not path.endswith(".gguf"):
            continue
        # An LFS-backed file reports the pointer's size at the top level and
        # the real one under `lfs`. Every .gguf worth having is LFS-backed, so
        # taking the top-level number would call a 3 GB file 135 bytes.
        lfs = item.get("lfs")
        size = lfs.get("size") if isinstance(lfs, dict) else None
        if not isinstance(size, int):
            size = item.get("size")
        if isinstance(size, int) and size > 0:
            sizes[Path(path).name] = size
    return sizes


def check_disk(needed: int) -> None:
    """Refuse before the download rather than half way through it."""
    try:
        free = shutil.disk_usage(str(WEIGHTS)).free
    except OSError:
        # Not knowing is not a reason to stop; the download will say so itself.
        return
    if free < needed + DISK_MARGIN:
        raise FetchError(
            f"Not enough disk: {human(needed)} is needed and "
            f"{human(free)} is free where weights/ lives."
        )


def install(key: str, *, force: bool = False, on_progress=None) -> list[Path]:
    """Download one catalog entry into weights/. Safe to run twice.

    `on_progress(label, total)` is the hook `setup.py` uses to draw a bar; it
    returns the same object `get_llama.download` expects, and None means plain
    lines instead.
    """
    item = entry(key)
    WEIGHTS.mkdir(parents=True, exist_ok=True)

    wanted = list(item["files"]) if force else missing(key)
    if not wanted:
        for name, _ in item["files"]:
            say(f"  already there: {name}")
        return [target_for(name) for name, _ in item["files"]]

    sizes = live_sizes(item["repo"])
    planned = [(name, sizes.get(name, size)) for name, size in wanted]
    check_disk(sum(size for _, size in planned))

    written: list[Path] = []
    for name, size in planned:
        target = target_for(name)
        # The download lands beside the real name, not on it. weights/ is
        # scanned for *.gguf, so a half-finished file under the real name would
        # be offered as a model and fail the moment somebody talked to it.
        partial = target.with_name(target.name + ".part")
        say(f"  {name}  ({human(size)})")
        download(
            f"{HOST}/{item['repo']}/resolve/main/{name}",
            partial,
            size,
            on_progress=on_progress,
        )
        # replace(), not rename(): on Windows rename() onto an existing file
        # raises, and --force is exactly the case where one exists.
        partial.replace(target)
        written.append(target)

    return written


def install_starter(*, force: bool = False, on_progress=None) -> list[Path]:
    """The one a fresh clone is offered."""
    return install(STARTER, force=force, on_progress=on_progress)


def anything_to_talk_to() -> bool:
    """True when weights/ holds a model that is not a vision projector.

    An `mmproj-*.gguf` is half of a pair and not something to talk to, so a
    folder holding only the OCR projector is still a folder with nothing to run.
    """
    return any(not path.name.startswith("mmproj-") for path in WEIGHTS.glob("*.gguf"))


def listing() -> None:
    """Print the catalog, marking what is already here."""
    say("Models this can fetch into weights/:")
    say()
    for key in ORDER:
        item = CATALOG[key]
        total = sum(size for _, size in item["files"])
        mark = "*" if have(key) else " "
        starter = "   <- the starter" if key == STARTER else ""
        say(f" {mark} {key:<10} {item['label']:<22} {total / 1e9:>5.1f} GB{starter}")
        say(f"     {item['note']}")
        say(f"     wants about {item['free_mb']} MB of free RAM to run")
        say()
    say("* already downloaded")
    say()
    say("Any other .gguf works the same way: put it in weights/ and it is")
    say("picked up on the next scan, sized from its own header.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch a model to run, and the OCR pair.",
        epilog="With no arguments it fetches the starter model, Gemma 4 E2B.",
    )
    parser.add_argument(
        "--what",
        default=STARTER,
        choices=("all", *ORDER),
        help=f"which to fetch (default: {STARTER}, the starter model)",
    )
    parser.add_argument(
        "--list", action="store_true", help="show what is available and stop"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even when it is there"
    )
    arguments = parser.parse_args()

    if arguments.list:
        listing()
        return 0

    keys = ORDER if arguments.what == "all" else (arguments.what,)

    failures = 0
    for key in keys:
        say(f"{CATALOG[key]['label']}:")
        try:
            install(key, force=arguments.force)
        except FetchError as exc:
            # One failure must not take the rest with it: somebody fetching
            # everything on a bad connection should keep what did arrive.
            say(f"  [X] {exc}")
            failures += 1
        except KeyboardInterrupt:
            say("\nStopped. Run this again to carry on where it left off.")
            return 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
