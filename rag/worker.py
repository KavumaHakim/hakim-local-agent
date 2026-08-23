"""The embedding model, running in its own process.

Run as `python -m rag.worker`. Not imported by the application: the parent
(`rag.embeddings.Embedder`) spawns it and talks to it over stdin/stdout.

**Why a separate process.** The whole point is that the embedding model must
not sit in memory while the 8B model is answering. Loading it in-process would
mean importing torch into the API worker, and that import costs a few hundred
megabytes of runtime that Python never gives back - `del model` frees the
weights, not the runtime. A child process gives every byte back the moment it
exits, which on an 8 GB machine is the difference that matters. It also mirrors
what the application already does with llama-server: models are processes, and
they get started and stopped.

**Protocol.** One JSON object per line, both directions.

    -> {"op": "ping"}
    <- {"ok": true, "dim": 384, "max_seq": 512}

    -> {"op": "embed", "texts": [...], "prefix": ""}
    <- {"ok": true, "rows": 2, "dim": 384, "vectors": "<base64 float32>"}

    <- {"ok": false, "error": "..."}

Vectors come back as base64-encoded little-endian float32 rather than JSON
numbers: a batch of 32 is 48 KB packed against roughly 250 KB as text, and it
survives the round trip without float formatting getting involved.

**stdout is the protocol channel and nothing else may touch it.** transformers
and sentence-transformers both write progress and warnings, and a single stray
line would desynchronise the stream. So the real stdout is taken aside at
startup and `sys.stdout` is pointed at stderr, where anything a library prints
becomes diagnostics instead of corruption.
"""

from __future__ import annotations

import base64
import json
import os
import sys

# The real stdout, taken aside by `_claim_stdout()` when this runs as a
# worker. It stays None on import, because the parent imports this module for
# QUERY_PREFIX and must keep its own stdout intact.
_CHANNEL = None


def _claim_stdout() -> None:
    """Take stdout for the protocol and point everything else at stderr."""
    global _CHANNEL
    _CHANNEL = sys.stdout
    sys.stdout = sys.stderr


# BGE-small-en-v1.5's own recommendation for retrieval: queries carry this
# instruction, passages carry nothing. Asymmetric on purpose, and dropping it
# measurably costs recall.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _reply(payload: dict) -> None:
    assert _CHANNEL is not None, "_claim_stdout() has not run"
    _CHANNEL.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _CHANNEL.flush()


def _fail(message: str) -> None:
    _reply({"ok": False, "error": message})


def _configure_threads() -> int:
    """Cap CPU threads before torch is imported.

    torch reads these at import time, and it defaults to every core. This
    machine has two, and llama-server wants them as well - an embedding run
    that takes all of them makes the model it is meant to be helping crawl.
    """
    try:
        threads = max(1, int(os.environ.get("RAG_THREADS", "") or 2))
    except ValueError:
        threads = 2
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, str(threads))
    return threads


def _load(threads: int):
    """Import sentence-transformers and load the model.

    Both steps are slow enough to matter - the import alone has been measured
    at over a minute on a cold page cache - which is why the parent keeps this
    process alive between calls rather than paying it per query.
    """
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            f"sentence-transformers is not installed ({exc}). Install it with: "
            f"pip install sentence-transformers"
        ) from None

    torch.set_num_threads(threads)
    # Inference only. Autograd would allocate graph state for every batch and
    # none of it would ever be used.
    torch.set_grad_enabled(False)

    name = os.environ.get("RAG_MODEL", "").strip() or "BAAI/bge-small-en-v1.5"
    folder = os.environ.get("RAG_MODEL_DIR", "").strip() or None

    try:
        model = SentenceTransformer(name, cache_folder=folder, device="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the embedding model {name!r}: "
            f"{type(exc).__name__}: {exc}. If this machine is offline, the "
            f"model has to be downloaded once first - see the README."
        ) from None

    return model


def main() -> int:
    _claim_stdout()
    threads = _configure_threads()

    try:
        model = _load(threads)
    except RuntimeError as exc:
        # Report the failure on the protocol channel and stop. The parent turns
        # this into a tool error rather than a traceback.
        _fail(str(exc))
        return 1

    dimension = int(model.get_sentence_embedding_dimension())
    max_seq = int(model.max_seq_length)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as exc:
            _fail(f"Malformed request: {exc}")
            continue

        operation = request.get("op")
        if operation == "ping":
            _reply({"ok": True, "dim": dimension, "max_seq": max_seq})
            continue
        if operation == "shutdown":
            _reply({"ok": True})
            return 0
        if operation != "embed":
            _fail(f"Unknown op {operation!r}.")
            continue

        texts = request.get("texts")
        if not isinstance(texts, list) or not texts:
            _fail("embed needs a non-empty list of texts.")
            continue

        prefix = request.get("prefix") or ""
        prepared = [prefix + str(text) for text in texts]

        try:
            vectors = model.encode(
                prepared,
                batch_size=len(prepared),
                # Cosine similarity becomes a dot product, so the index does
                # not have to normalise on every search.
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            packed = base64.b64encode(
                vectors.astype("<f4", copy=False).tobytes()
            ).decode("ascii")
        except Exception as exc:
            # Out of memory is the realistic failure here, and it must come
            # back as a message rather than killing the process: the parent
            # can then fall back to a smaller batch.
            _fail(f"Embedding failed: {type(exc).__name__}: {exc}")
            continue

        _reply(
            {
                "ok": True,
                "rows": int(vectors.shape[0]),
                "dim": int(vectors.shape[1]),
                "vectors": packed,
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
