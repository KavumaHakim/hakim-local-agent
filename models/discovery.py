"""Turning a folder of GGUF files into runnable model entries.

Drop a `.gguf` into `weights/` and it appears. That is the whole feature, and
almost all of the work is in the three things a hand-written registry entry
used to supply.

**Context.** Not the model's training context. Ministral is trained for
262,144 tokens and costs 79,872 bytes of KV cache per token, so running it as
trained would ask for 19.5 GB of cache on an 8 GB machine. `choose_context`
spends a fixed cache budget instead, which is why a dropped-in model starts
rather than dying in an allocator.

**RAM threshold.** Resident weights are about 0.8x the file - measured, from
GLM-OCR's 906 MB file using 683 MB - plus the KV cache, plus headroom. Large
models get more than the formula, because a model that cannot stay resident
pages from disk and the formula stops describing it.

**Ports.** Assigned from a pool, skipping anything already claimed by a
curated entry, the API, or the dev server.

Nothing here starts a process or reads a tensor. It reads file sizes and GGUF
headers, so a rescan of a folder is milliseconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from models.gguf import GgufInfo, read_metadata

# Ports handed out to discovered models. 8000 is the API and 5173 the dev
# server; both would look like a free port and fail confusingly.
PORT_POOL_START = 8090
PORT_POOL_END = 8199
RESERVED_PORTS = frozenset({8000, 5173})

# Resident weights as a fraction of file size. Measured on this machine:
# GLM-OCR's 906 MB file occupies 683 MB, a ratio of 0.754. Rounded up, because
# under-estimating means the guard lets a model start that then thrashes.
WEIGHT_RESIDENCY = 0.8

# Free RAM left over after weights and cache, for the rest of the system.
HEADROOM_MB = 250

# What one model may spend on KV cache. Chosen by working backwards from the
# hand-tuned entries in models.json: at 420 MB, GLM-OCR lands on 8192 and
# Ministral on 4096, which is exactly what those entries were measured to.
# It is the knob to turn when long contexts matter more than headroom.
DEFAULT_KV_BUDGET_MB = 420

# Contexts are chosen from this ladder rather than computed freely: a round
# number is easier to reason about, and llama.cpp's own defaults are powers of
# two. 8192 is the ceiling because beyond it the cache dominates on 8 GB.
CONTEXT_LADDER = (2048, 4096, 8192)
MIN_CONTEXT = 2048

# Past this much resident weight, a model cannot comfortably stay in RAM on a
# small machine and starts paging from disk. The linear estimate stops
# describing what it needs, so the threshold is padded proportionally - which
# is the same judgement models.json makes for the 8B in prose, and lands within
# 2% of the value that was measured for it by hand.
LARGE_MODEL_MB = 3000
LARGE_MODEL_PADDING = 0.5

# Filenames that are a vision projector rather than a model. The header is
# authoritative (architecture == clip), but the name is checked first because
# it costs nothing and is right almost always.
_PROJECTOR_NAME = re.compile(r"(^|[-_.])mmproj([-_.]|$)", re.IGNORECASE)

# Stripped from a filename to make a readable label.
_QUANT_SUFFIX = re.compile(
    r"[-_.]?(Q\d+_[KM0-9_A-Z]*|IQ\d+[_A-Z0-9]*|F16|F32|BF16|GGUF)$",
    re.IGNORECASE,
)
_SPLIT_PART = re.compile(r"-\d{5}-of-\d{5}$", re.IGNORECASE)


@dataclass
class Discovered:
    """One model found on disk, sized and ready to register."""

    key: str
    label: str
    file: Path
    port: int
    context: int
    threads: int
    min_free_mb: int
    role: str = "chat"
    mmproj: Path | None = None
    info: GgufInfo | None = None
    description: str = ""
    # Anything the user should know before choosing it - a context that had to
    # be cut, a header that could not be read.
    notes: list[str] = field(default_factory=list)

    @property
    def file_mb(self) -> int:
        try:
            return int(self.file.stat().st_size / (1024 * 1024))
        except OSError:
            return 0

    def as_entry(self) -> dict:
        """The registry-entry shape `load_registry` already understands."""
        entry = {
            "key": self.key,
            "label": self.label,
            "file": self.file.name,
            "port": self.port,
            "context": self.context,
            "threads": self.threads,
            "min_free_mb": self.min_free_mb,
            "description": self.description,
            "role": self.role,
            "discovered": True,
        }
        if self.mmproj is not None:
            entry["mmproj"] = self.mmproj.name
        return entry


def slugify(name: str) -> str:
    """A short, stable key from a filename.

    Stable matters more than pretty: the key is what a saved primary model
    refers to, so it must not change when the folder is rescanned.
    """
    stem = Path(name).stem
    stem = _SPLIT_PART.sub("", stem)
    stem = _QUANT_SUFFIX.sub("", stem)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower()
    return slug or "model"


def labelise(name: str) -> str:
    """A human-readable label from a filename."""
    stem = Path(name).stem
    stem = _SPLIT_PART.sub("", stem)
    stem = _QUANT_SUFFIX.sub("", stem)
    return re.sub(r"[-_]+", " ", stem).strip() or Path(name).stem


def is_projector(path: Path, info: GgufInfo | None) -> bool:
    """Whether a file is an mmproj projector rather than a model."""
    if info is not None and info.is_projector:
        return True
    if info is not None and info.architecture:
        # The header was readable and said something else; trust it over the
        # name, which is only a convention.
        return False
    return bool(_PROJECTOR_NAME.search(path.name))


def choose_context(
    info: GgufInfo | None, *, kv_budget_mb: int = DEFAULT_KV_BUDGET_MB
) -> tuple[int, str]:
    """The largest sensible context that fits the cache budget.

    Returns the context and a note when it had to be cut below what the model
    was trained for, so the UI can say why rather than silently under-running
    a long-context model.
    """
    if info is None or not info.kv_bytes_per_token:
        # No header, so no arithmetic is possible. 4096 is what every curated
        # entry here uses and is a safe floor.
        return 4096, ""

    trained = info.training_context or CONTEXT_LADDER[-1]
    budget_bytes = kv_budget_mb * 1024 * 1024

    best = MIN_CONTEXT
    for candidate in CONTEXT_LADDER:
        if candidate > trained:
            break
        if info.kv_bytes_per_token * candidate <= budget_bytes:
            best = candidate

    best = min(best, trained) if trained else best
    note = ""
    if trained and best < trained:
        note = (
            f"Context capped at {best:,} of the model's {trained:,}: the full "
            f"cache would need "
            f"{info.kv_cache_mb(trained):,} MB."
        )
    return best, note


def estimate_min_free_mb(
    file_bytes: int, info: GgufInfo | None, context: int
) -> int:
    """Free RAM to insist on before starting this model.

    Weights plus cache plus headroom. Deliberately an over-estimate: the cost
    of being wrong high is a warning the user can override, and the cost of
    being wrong low is a machine that swaps.
    """
    weights_mb = int(file_bytes / (1024 * 1024) * WEIGHT_RESIDENCY)
    cache_mb = info.kv_cache_mb(context) if info is not None else 0
    if not cache_mb:
        # Unknown cache: assume the same order as a mid-sized model here
        # rather than zero, which would make the guard useless.
        cache_mb = 200

    total = weights_mb + cache_mb + HEADROOM_MB
    if weights_mb >= LARGE_MODEL_MB:
        total += int(weights_mb * LARGE_MODEL_PADDING)
    return total


def allocate_port(taken: set[int]) -> int:
    """The next free port from the pool."""
    for port in range(PORT_POOL_START, PORT_POOL_END + 1):
        if port not in taken and port not in RESERVED_PORTS:
            taken.add(port)
            return port
    raise ValueError(
        f"No free port between {PORT_POOL_START} and {PORT_POOL_END}. "
        f"Remove some models, or give them explicit ports in models.json."
    )


def discover(
    models_dir: str | Path,
    *,
    taken_keys: Iterable[str] = (),
    taken_ports: Iterable[int] = (),
    known_files: Iterable[str] = (),
    threads: int = 4,
    kv_budget_mb: int = DEFAULT_KV_BUDGET_MB,
) -> list[Discovered]:
    """Every runnable model in `models_dir` that is not already registered.

    `known_files` are the filenames curated entries already claim. Those are
    skipped rather than duplicated: a hand-tuned entry is better than anything
    inferred here, and the whole point is that discovery adds to the registry
    without overriding what someone measured.
    """
    directory = Path(models_dir).expanduser()
    if not directory.is_dir():
        return []

    keys = {key.lower() for key in taken_keys}
    ports = set(taken_ports)
    claimed = {Path(name).name.lower() for name in known_files}

    try:
        files = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".gguf"
        )
    except OSError:
        return []

    headers: dict[Path, GgufInfo | None] = {path: read_metadata(path) for path in files}

    projectors = [path for path in files if is_projector(path, headers[path])]
    models = [path for path in files if path not in projectors]

    found: list[Discovered] = []
    for path in models:
        if path.name.lower() in claimed:
            continue

        info = headers[path]
        # A split model is many files; only the first carries the header and
        # llama.cpp loads the rest by itself.
        if _SPLIT_PART.search(path.stem) and not path.stem.endswith("-00001-of-00001"):
            if "-00001-of-" not in path.stem:
                continue

        key = _unique_key(slugify(path.name), keys)
        context, note = choose_context(info, kv_budget_mb=kv_budget_mb)
        try:
            size = path.stat().st_size
        except OSError:
            continue

        projector = _match_projector(path, projectors)
        notes = [note] if note else []
        if info is None:
            notes.append(
                "The GGUF header could not be read, so context and RAM are "
                "estimates from the file size alone."
            )

        found.append(
            Discovered(
                key=key,
                label=labelise(path.name),
                file=path,
                port=allocate_port(ports),
                context=context,
                threads=threads,
                min_free_mb=estimate_min_free_mb(size, info, context),
                # A model shipped with a projector is a vision backend. It is
                # given the same role GLM-OCR has, which keeps it out of the
                # one-at-a-time chat rotation and out of the model picker.
                role="ocr" if projector is not None else "chat",
                mmproj=projector,
                info=info,
                description=_describe(path, info, size),
                notes=notes,
            )
        )
    return found


def _unique_key(base: str, taken: set[str]) -> str:
    """A key not already in use, keeping the readable form where possible."""
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}-{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _match_projector(model: Path, projectors: list[Path]) -> Path | None:
    """Find the mmproj that belongs to `model`, if there is one.

    Matched on the stem, because the convention is `mmproj-<model>.gguf`. A
    folder with one model and one projector pairs them regardless of naming,
    which is what people actually have.
    """
    if not projectors:
        return None

    stem = model.stem.lower()
    for projector in projectors:
        name = projector.stem.lower()
        cleaned = _PROJECTOR_NAME.sub("", name).strip("-_.")
        if cleaned and (cleaned in stem or stem in cleaned):
            return projector
    return None


def _describe(path: Path, info: GgufInfo | None, size: int) -> str:
    """A one-line description, from what the header actually said."""
    megabytes = int(size / (1024 * 1024))
    parts = [f"Found in the models folder. {megabytes:,} MB on disk"]
    if info is not None and info.architecture:
        parts.append(f"{info.architecture} architecture")
    if info is not None and info.training_context:
        parts.append(f"trained for {info.training_context:,} tokens")
    return ", ".join(parts) + "."
