"""Reading a GGUF file's metadata header, without loading the model.

Plug-and-play needs this. Dropping a model in a folder is only half the
problem; the other half is knowing what it costs to run, and the honest answer
is in the file. A 4 GB model trained for a 262,144-token context will happily
start with that context and allocate tens of gigabytes of KV cache on a machine
with eight. Guessing from the file size cannot catch that, because the cache is
not made of weights.

So this reads the header - a few kilobytes at the front of the file, never the
tensors - and returns the numbers the sizing code needs:

    architecture, name, training context, layer count, KV head count,
    embedding width

from which `kv_bytes_per_token` follows exactly. That figure is what turns
"pick a context" from a guess into arithmetic.

The format is documented at ggml-org/ggml/docs/gguf.md. Only the header is
parsed, and every read is bounded: a truncated or hostile file produces None,
never an exception out of this module and never an unbounded allocation.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"

# Reading more than this from the header means the file is not what it claims.
# The largest real header seen here is well under a megabyte; tokeniser
# vocabularies are the bulk of it.
MAX_HEADER_BYTES = 64 * 1024 * 1024
# A single string or array longer than this is a corrupt length field, not
# data. Without this check a bad uint64 becomes a multi-gigabyte allocation.
MAX_STRING_BYTES = 16 * 1024 * 1024
MAX_ARRAY_ITEMS = 4_000_000

# GGUF value type ids.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_FIXED: dict[int, tuple[str, int]] = {
    _UINT8: ("<B", 1),
    _INT8: ("<b", 1),
    _UINT16: ("<H", 2),
    _INT16: ("<h", 2),
    _UINT32: ("<I", 4),
    _INT32: ("<i", 4),
    _FLOAT32: ("<f", 4),
    _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8),
    _INT64: ("<q", 8),
    _FLOAT64: ("<d", 8),
}

# Bytes per element in the KV cache. llama.cpp defaults to f16 for both K and
# V; quantising the cache (--cache-type-k q8_0) roughly halves this, which is
# the documented escape hatch when a context will not fit.
KV_BYTES_PER_ELEMENT = 2

# Architectures that are a vision projector rather than a language model.
# These are the `mmproj-*.gguf` files: they carry no chat capability and must
# never be offered as a model to talk to.
PROJECTOR_ARCHITECTURES = frozenset({"clip"})


@dataclass(frozen=True)
class GgufInfo:
    """What the header says about a model."""

    path: Path
    architecture: str = ""
    name: str = ""
    # The context the model was trained for. Not what it should be run at:
    # that is a RAM decision, made by models/discovery.py.
    training_context: int = 0
    block_count: int = 0
    head_count: int = 0
    head_count_kv: int = 0
    embedding_length: int = 0
    file_bytes: int = 0

    @property
    def is_projector(self) -> bool:
        """Whether this is an mmproj file rather than a model."""
        return self.architecture in PROJECTOR_ARCHITECTURES

    @property
    def head_dim(self) -> int:
        """Width of one attention head, or 0 when the header did not say."""
        if not self.head_count or not self.embedding_length:
            return 0
        return self.embedding_length // self.head_count

    @property
    def kv_bytes_per_token(self) -> int:
        """How much KV cache one token of context costs.

        Two caches (K and V), one entry per layer per KV head. This is the
        number the models.json comments worked out by hand for GLM-OCR;
        computing it means a new model does not need that done again.

        Returns 0 when the header lacked a field, which the caller must read as
        "unknown" and fall back to a conservative default rather than as "free".
        """
        if not (self.block_count and self.head_count_kv and self.head_dim):
            return 0
        return (
            2
            * self.block_count
            * self.head_count_kv
            * self.head_dim
            * KV_BYTES_PER_ELEMENT
        )

    def kv_cache_mb(self, context: int) -> int:
        """KV cache cost in MB for a given context length."""
        per_token = self.kv_bytes_per_token
        if not per_token:
            return 0
        return int(per_token * max(0, context) / (1024 * 1024))


def read_metadata(path: str | Path) -> GgufInfo | None:
    """Read a GGUF header. Returns None for anything that is not readable.

    Never raises: a file that is missing, truncated, not GGUF, or a version
    this does not understand all come back as None, and the caller falls back
    to sizing from the file size alone.
    """
    target = Path(path)
    try:
        size = target.stat().st_size
    except OSError:
        return None

    try:
        with target.open("rb") as handle:
            if handle.read(4) != MAGIC:
                return None

            version = _fixed(handle, "<I", 4)
            # v1 put counts in uint32 and is long obsolete; v4 does not exist
            # yet. Refusing is better than misreading a format that changed.
            if version not in (2, 3):
                return None

            tensor_count = _fixed(handle, "<Q", 8)
            kv_count = _fixed(handle, "<Q", 8)
            if tensor_count is None or kv_count is None:
                return None
            if kv_count > 100_000:
                return None

            fields: dict[str, Any] = {}
            complete = True
            for _ in range(int(kv_count)):
                if handle.tell() > MAX_HEADER_BYTES:
                    complete = False
                    break
                key = _string(handle)
                if key is None:
                    complete = False
                    break
                value_type = _fixed(handle, "<I", 4)
                if value_type is None:
                    complete = False
                    break
                value = _value(handle, int(value_type))
                if value is _FAILED:
                    complete = False
                    break
                # Long arrays are the tokeniser vocabulary and are of no
                # interest here; they are parsed to advance the file position
                # and then dropped rather than held.
                if not isinstance(value, list) or len(value) <= 16:
                    fields[key] = value
    except (OSError, struct.error, MemoryError, OverflowError):
        return None

    # A header that could not be walked to the end is not a header. Returning
    # the fields that happened to parse would be worse than returning nothing:
    # the caller cannot tell a genuinely absent field from a truncated read,
    # and would size a model from half a file. None sends it to the
    # conservative fallback, which is the honest answer.
    if not complete:
        return None

    architecture = str(fields.get("general.architecture", "") or "")
    if not architecture:
        # Every real GGUF declares this, and without it none of the per-
        # architecture keys below can even be named.
        return None
    prefix = architecture

    def number(suffix: str, *alternatives: str) -> int:
        for candidate in (f"{prefix}.{suffix}", *alternatives):
            value = fields.get(candidate)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        return 0

    head_count = number("attention.head_count")
    return GgufInfo(
        path=target,
        architecture=architecture,
        name=str(fields.get("general.name", "") or ""),
        training_context=number("context_length"),
        block_count=number("block_count"),
        head_count=head_count,
        # Multi-query and grouped-query models have fewer KV heads than
        # attention heads, and it is the KV count that sizes the cache. When
        # the field is absent the model is multi-head, so they are equal.
        head_count_kv=number("attention.head_count_kv") or head_count,
        embedding_length=number("embedding_length"),
        file_bytes=size,
    )


# --- header primitives ---

_FAILED = object()


def _read(handle: BinaryIO, count: int) -> bytes | None:
    data = handle.read(count)
    return data if len(data) == count else None


def _fixed(handle: BinaryIO, layout: str, size: int) -> Any:
    data = _read(handle, size)
    if data is None:
        return None
    return struct.unpack(layout, data)[0]


def _string(handle: BinaryIO) -> str | None:
    length = _fixed(handle, "<Q", 8)
    if length is None or length > MAX_STRING_BYTES:
        return None
    data = _read(handle, int(length))
    if data is None:
        return None
    return data.decode("utf-8", errors="replace")


def _value(handle: BinaryIO, value_type: int) -> Any:
    if value_type in _FIXED:
        layout, size = _FIXED[value_type]
        value = _fixed(handle, layout, size)
        return _FAILED if value is None else value

    if value_type == _STRING:
        value = _string(handle)
        return _FAILED if value is None else value

    if value_type == _ARRAY:
        item_type = _fixed(handle, "<I", 4)
        length = _fixed(handle, "<Q", 8)
        if item_type is None or length is None or length > MAX_ARRAY_ITEMS:
            return _FAILED
        items: list[Any] = []
        for _ in range(int(length)):
            item = _value(handle, int(item_type))
            if item is _FAILED:
                return _FAILED
            # Only the first few are kept; the rest are read to advance the
            # position. A 150,000-entry vocabulary must not be materialised.
            if len(items) < 16:
                items.append(item)
        return items

    # An unknown type means the rest of the header cannot be walked, because
    # the length of this value is unknown.
    return _FAILED
