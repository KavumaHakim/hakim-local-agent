"""Owning the embedding worker process: start it late, stop it early.

`Embedder` is the only thing that knows the model exists. It starts the worker
on first use, keeps it for as long as work keeps arriving, and shuts it down
once it has been idle - so the common case, a machine sitting at a chat prompt
with no searching going on, holds no embedding model at all.

The idle shutdown mirrors `ModelManager.unload_idle`, and is driven by the same
sweeper thread in `api/main.py`. That is deliberate: there is one answer in this
project to "a model is resident and nobody is using it", and this is it.

Reading the worker's stdout goes through a reader thread rather than a blocking
`readline`. On Windows there is no way to poll a pipe with a timeout, and a
worker that dies during startup would otherwise hang the caller forever.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from rag.worker import QUERY_PREFIX

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# BGE-small-en-v1.5. Stated here as well as in config so the worker and the
# index agree on the width of a vector even when config is not involved.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIMENSION = 384

# Loading is an import of torch plus a model read. Both are slow on a cold page
# cache - the import alone has been measured at over a minute here - so the
# handshake gets a long deadline rather than a snappy one that would report a
# healthy worker as broken.
START_TIMEOUT = 420.0
# A batch of a few dozen short texts on two cores. Generous, but bounded, so a
# wedged worker surfaces as an error instead of a hang.
REQUEST_TIMEOUT = 600.0

# How much of the worker's stderr to keep for error messages.
STDERR_LINES = 40

# Reported to the caller so a long ingest can show progress.
ProgressCallback = Callable[[int, int], None]


class EmbeddingError(Exception):
    """The embedding model could not be loaded, or could not embed."""


def _kill_tree(process: subprocess.Popen) -> None:
    """Kill the worker and everything it started.

    `process.kill()` alone is not enough here, and the reason is easy to miss.
    A virtualenv's `Scripts\\python.exe` is often not the interpreter but a
    46 KB launcher that runs the real one as a *child* - uv builds venvs this
    way. Killing the launcher then leaves the real process, holding torch and
    the model, orphaned and running. That is the exact leak this whole
    out-of-process design exists to avoid, and it would only ever show up on
    the path that matters: a wedged worker being force-stopped.

    `taskkill /T` takes the whole tree. It matches how models/manager.py stops
    a stray llama-server, and falls back to `kill()` where taskkill is not the
    right tool.
    """
    if process.poll() is not None:
        return

    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            pass  # fall through to kill(), which is better than nothing

    try:
        process.kill()
        process.wait(timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass


class Embedder:
    """A lazily-started, idle-stopped embedding model in a child process."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        model_dir: str | Path | None = None,
        threads: int = 2,
        batch_size: int = 8,
        idle_seconds: float = 120.0,
        python_executable: str | None = None,
    ) -> None:
        self.model = model
        self.model_dir = str(model_dir) if model_dir else ""
        self.threads = max(1, int(threads))
        self.batch_size = max(1, int(batch_size))
        self.idle_seconds = max(0.0, float(idle_seconds))
        self._python = python_executable or sys.executable

        self._process: subprocess.Popen | None = None
        self._replies: queue.Queue[str | None] = queue.Queue()
        self._stderr: list[str] = []
        self._dimension = DEFAULT_DIMENSION
        self._last_used = 0.0
        # One request at a time: the protocol is a single pipe with no request
        # ids, so two callers interleaving would read each other's replies.
        self._lock = threading.RLock()

    # --- state ---

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    @property
    def dimension(self) -> int:
        """Vector width. The model's real value once it has been loaded."""
        return self._dimension

    def idle_for(self) -> float:
        """Seconds since the last request, or 0 when not loaded."""
        with self._lock:
            if not self.loaded or not self._last_used:
                return 0.0
            return time.time() - self._last_used

    # --- lifecycle ---

    def ensure_loaded(self) -> None:
        """Start the worker if it is not already running."""
        with self._lock:
            if self.loaded:
                return
            self._start()

    def unload(self) -> bool:
        """Stop the worker and give its memory back. Safe to call any time."""
        with self._lock:
            process = self._process
            self._process = None
            if process is None:
                return False

            try:
                if process.poll() is None and process.stdin is not None:
                    # Ask first: a clean exit closes the model's file handles,
                    # which matters on Windows where a killed process can leave
                    # the cache locked.
                    try:
                        process.stdin.write('{"op": "shutdown"}\n')
                        process.stdin.flush()
                    except (OSError, ValueError):
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        _kill_tree(process)
            except OSError:
                pass
            finally:
                for stream in (process.stdin, process.stdout, process.stderr):
                    try:
                        if stream is not None:
                            stream.close()
                    except OSError:
                        pass

            self._drain_replies()
            self._last_used = 0.0
            return True

    def unload_if_idle(self) -> bool:
        """Stop the worker when it has been unused for `idle_seconds`.

        Called by the same sweeper that unloads idle llama-servers.
        """
        if self.idle_seconds <= 0:
            return False
        with self._lock:
            if not self.loaded or self.idle_for() < self.idle_seconds:
                return False
            return self.unload()

    # --- embedding ---

    def encode_passages(
        self, texts: Iterable[str], *, progress: ProgressCallback | None = None
    ) -> np.ndarray:
        """Embed document chunks. No instruction prefix, per BGE's design."""
        return self._encode(list(texts), prefix="", progress=progress)

    def encode_query(self, text: str) -> np.ndarray:
        """Embed one search query, with the retrieval instruction BGE wants.

        Returns a 1-D vector, because every caller wants exactly one.
        """
        return self._encode([text], prefix=QUERY_PREFIX)[0]

    def _encode(
        self,
        texts: list[str],
        *,
        prefix: str,
        progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        if any(not isinstance(text, str) for text in texts):
            raise EmbeddingError("Every text to embed must be a string.")

        with self._lock:
            self.ensure_loaded()

            blocks: list[np.ndarray] = []
            done = 0
            # Batched so peak memory is a batch, not the whole corpus. A
            # 40,000-chunk ingest would otherwise build one enormous request.
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                blocks.append(self._embed_batch(batch, prefix))
                done += len(batch)
                if progress is not None:
                    try:
                        progress(done, len(texts))
                    except Exception:
                        pass  # a reporting callback must not fail the ingest

            self._last_used = time.time()
            return np.vstack(blocks)

    def _embed_batch(self, batch: list[str], prefix: str) -> np.ndarray:
        reply = self._request({"op": "embed", "texts": batch, "prefix": prefix})

        try:
            raw = base64.b64decode(reply["vectors"])
            rows, dim = int(reply["rows"]), int(reply["dim"])
        except (KeyError, ValueError, TypeError) as exc:
            raise EmbeddingError(f"Malformed reply from the embedder: {exc}") from None

        vectors = np.frombuffer(raw, dtype="<f4")
        if vectors.size != rows * dim:
            raise EmbeddingError(
                f"The embedder returned {vectors.size} floats for {rows}x{dim}."
            )
        if rows != len(batch):
            raise EmbeddingError(
                f"Asked for {len(batch)} vectors and got {rows}."
            )
        self._dimension = dim
        # frombuffer is a view over a read-only bytes object; copy so callers
        # get an array they can write to.
        return vectors.reshape(rows, dim).astype(np.float32, copy=True)

    # --- process plumbing ---

    def _start(self) -> None:
        environment = dict(os.environ)
        environment["RAG_MODEL"] = self.model
        environment["RAG_MODEL_DIR"] = self.model_dir
        environment["RAG_THREADS"] = str(self.threads)
        # Unbuffered, or replies sit in the child's pipe buffer and every
        # request looks like a timeout.
        environment["PYTHONUNBUFFERED"] = "1"
        # Hugging Face's cache links blobs into the snapshot directory with
        # symlinks, and creating one on Windows needs Developer Mode or an
        # elevated process. Without this the very first download dies with
        # "WinError 1314: A required privilege is not held by the client".
        # Copying instead costs a duplicate of a 130 MB model, once.
        environment.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

        try:
            process = subprocess.Popen(
                [self._python, "-m", "rag.worker"],
                cwd=str(PROJECT_ROOT),
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise EmbeddingError(f"Could not start the embedding worker: {exc}") from None

        self._process = process
        self._replies = queue.Queue()
        self._stderr = []

        threading.Thread(
            target=self._read_replies, args=(process,), daemon=True,
            name="rag-embedder-stdout",
        ).start()
        threading.Thread(
            target=self._read_stderr, args=(process,), daemon=True,
            name="rag-embedder-stderr",
        ).start()

        try:
            reply = self._request({"op": "ping"}, timeout=START_TIMEOUT)
        except EmbeddingError:
            self.unload()
            raise
        self._dimension = int(reply.get("dim", DEFAULT_DIMENSION))
        self._last_used = time.time()

    def _read_replies(self, process: subprocess.Popen) -> None:
        """Pump the worker's stdout into a queue. None marks the end."""
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    self._replies.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._replies.put(None)

    def _read_stderr(self, process: subprocess.Popen) -> None:
        """Keep the tail of the worker's stderr, for error messages."""
        try:
            if process.stderr is not None:
                for line in process.stderr:
                    self._stderr.append(line.rstrip())
                    del self._stderr[:-STDERR_LINES]
        except (OSError, ValueError):
            pass

    def _request(self, payload: dict, timeout: float = REQUEST_TIMEOUT) -> dict:
        process = self._process
        if process is None or process.stdin is None:
            raise EmbeddingError("The embedding worker is not running.")

        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
        except (OSError, ValueError):
            raise EmbeddingError(
                f"The embedding worker stopped accepting work.{self._diagnostics()}"
            ) from None

        try:
            line = self._replies.get(timeout=timeout)
        except queue.Empty:
            self.unload()
            raise EmbeddingError(
                f"The embedding worker did not answer within {timeout:.0f}s. "
                f"It has been stopped; try again."
            ) from None

        if line is None:
            self.unload()
            raise EmbeddingError(
                f"The embedding worker exited.{self._diagnostics()}"
            )

        try:
            reply = json.loads(line)
        except ValueError:
            raise EmbeddingError(
                f"Unreadable reply from the embedding worker: {line[:200]!r}"
            ) from None

        if not reply.get("ok"):
            raise EmbeddingError(str(reply.get("error", "The embedder failed.")))
        return reply

    def _drain_replies(self) -> None:
        while True:
            try:
                self._replies.get_nowait()
            except queue.Empty:
                return

    def _diagnostics(self) -> str:
        """The tail of the worker's stderr, when there is any."""
        tail = [line for line in self._stderr if line.strip()]
        if not tail:
            return ""
        return "\n\nThe worker said:\n" + "\n".join(tail[-12:])


# --- the process-wide instance -------------------------------------------
#
# The tool registry is rebuilt for every turn, and a RagManager is cheap enough
# to go with it. The worker is not: one per rebuilt registry would mean a new
# torch process per turn. So the worker is shared, and the managers that come
# and go all point at the same one.

_shared: Embedder | None = None
_shared_key: tuple = ()
_shared_lock = threading.Lock()


def shared_embedder(
    *,
    model: str = DEFAULT_MODEL,
    model_dir: str | Path | None = None,
    threads: int = 2,
    batch_size: int = 8,
    idle_seconds: float = 120.0,
) -> Embedder:
    """The process-wide embedder, built on first use.

    Rebuilt only when the settings that define the model change - which is why
    the key includes them rather than just returning whatever exists.
    """
    global _shared, _shared_key

    key = (model, str(model_dir or ""), int(threads), int(batch_size), float(idle_seconds))
    with _shared_lock:
        if _shared is not None and _shared_key == key:
            return _shared
        if _shared is not None:
            _shared.unload()
        _shared = Embedder(
            model=model,
            model_dir=model_dir,
            threads=threads,
            batch_size=batch_size,
            idle_seconds=idle_seconds,
        )
        _shared_key = key
        return _shared


def unload_shared() -> bool:
    """Stop the shared embedder, if there is one. Used at shutdown."""
    with _shared_lock:
        return _shared.unload() if _shared is not None else False


def sweep_shared() -> bool:
    """Stop the shared embedder if it has gone idle. Used by the sweeper."""
    with _shared_lock:
        return _shared.unload_if_idle() if _shared is not None else False
