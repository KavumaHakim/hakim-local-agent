"""The one object the rest of the application talks to about documents.

It owns the order of operations and nothing else: extraction, chunking,
embedding, the vector file and the metadata store each do their own job, and
this decides when.

Two rules shape the ingest path.

**Nothing is written until the slow part has succeeded.** A file is extracted,
chunked and embedded before a single row is allocated. Embedding is where an
ingest realistically fails - the model is missing, the machine is out of memory
- and doing it first means a failure leaves the index exactly as it was.

**A file that has not changed is not embedded again.** Size and modification
time are checked first because they are free; the hash is computed only when
they disagree, because reading a 40 MB PDF to hash it is not. Re-running
`index` over a folder therefore costs seconds, not minutes, and cannot produce
duplicate chunks: a document is replaced by path, in one transaction.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rag.chunker import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    chunk_document,
)
from rag.embeddings import DEFAULT_DIMENSION, EmbeddingError, Embedder
from rag.extract import (
    SUPPORTED_EXTENSIONS,
    ExtractionError,
    extract,
    is_supported,
)
from rag.index import VectorIndex, VectorIndexError
from rag.metadata import ChunkRecord, Document, MetadataStore, SCHEMA_VERSION

# File names inside the store directory.
VECTORS_FILE = "vectors.f32"
METADATA_FILE = "chunks.db"

# Read in blocks when hashing, so a large PDF is not loaded to check whether it
# changed.
HASH_BLOCK = 1024 * 1024

# Refuse to start an ingest with less headroom than this. The index itself is
# small; the risk is filling a disk that is already nearly full, which breaks
# far more than this feature.
MIN_FREE_DISK_MB = 200

# Directories never walked when indexing a folder. Indexing a virtualenv is
# always a mistake and would take hours.
SKIP_DIRECTORIES = frozenset(
    {
        ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
        ".pytest_cache", ".ruff_cache", "dist", "build", ".idea", ".vscode",
        "site-packages", ".next", ".cache",
    }
)

# Reported during a long ingest: (stage, done, total).
Progress = Callable[[str, int, int], None]


class RagError(Exception):
    """A document-search operation could not be completed."""


@dataclass(frozen=True)
class SearchHit:
    """One retrieved chunk and where it came from."""

    text: str
    score: float
    document: str
    path: str
    chunk_id: str
    page: int | None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "document": self.document,
            "path": self.path,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "text": self.text,
        }
        # Omitted rather than null for formats without pages: an explicit
        # "page": null in every result of a .md search is noise in a context
        # window that costs seconds per token.
        if self.page is not None:
            payload["page"] = self.page
        return payload


class RagManager:
    """Index documents, and search them.

    Cheap to construct: it opens a SQLite file and computes some paths. The
    embedding model is not touched until something actually needs it, and the
    embedder it uses is shared process-wide rather than owned here.
    """

    def __init__(
        self,
        store_dir: str | Path,
        *,
        embedder: Embedder | None = None,
        model: str = "BAAI/bge-small-en-v1.5",
        dimension: int = DEFAULT_DIMENSION,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        top_k: int = 5,
        min_score: float = 0.3,
        max_file_bytes: int = 20_000_000,
        ocr=None,
    ) -> None:
        self.store_dir = Path(store_dir).expanduser()
        self.model = model
        self.chunk_tokens = int(chunk_tokens)
        self.overlap_tokens = int(overlap_tokens)
        self.top_k = max(1, int(top_k))
        self.min_score = float(min_score)
        self.max_file_bytes = int(max_file_bytes)

        try:
            self.store_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RagError(f"Could not create {self.store_dir}: {exc}") from None

        self.store = MetadataStore(self.store_dir / METADATA_FILE)
        self.index = VectorIndex(self.store_dir / VECTORS_FILE, dimension)
        self._embedder = embedder
        # An OCR backend, or None. None is not a failure: it means scanned
        # pages are reported as needing OCR rather than silently skipped.
        self.ocr = ocr

    # --- the embedder ---

    @property
    def embedder(self) -> Embedder:
        """The shared embedding worker, built on first use."""
        if self._embedder is None:
            from rag.embeddings import shared_embedder

            self._embedder = shared_embedder(model=self.model)
        return self._embedder

    def unload(self) -> bool:
        """Stop the embedding model. The index stays on disk and searchable."""
        return self.embedder.unload() if self._embedder is not None else False

    # --- indexing ---

    def index_path(
        self,
        path: str | Path,
        *,
        recursive: bool = True,
        force: bool = False,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        """Index a file, or every supported file in a directory.

        Files that have not changed since they were last indexed are skipped
        unless `force` is set.
        """
        target = Path(path).expanduser()
        try:
            target = target.resolve()
        except OSError as exc:
            raise RagError(f"Invalid path: {exc}") from None

        if not target.exists():
            raise RagError(f"Nothing to index: {target} does not exist.")

        if target.is_dir():
            candidates = self._walk(target, recursive=recursive)
            if not candidates:
                raise RagError(
                    f"No supported files under {target}. Looked for: "
                    f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
                )
        elif not is_supported(target):
            raise RagError(
                f"{target.suffix or 'That file'} is not a type this indexes. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
            )
        else:
            candidates = [target]

        self._check_disk()
        self._check_compatible()

        indexed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []

        for position, candidate in enumerate(candidates, start=1):
            if progress is not None:
                progress(candidate.name, position, len(candidates))
            try:
                outcome = self._index_one(candidate, force=force)
            except RagError as exc:
                # One unreadable file must not abandon the other four hundred.
                failed.append({"document": candidate.name, "error": str(exc)})
                continue
            (skipped if outcome["skipped"] else indexed).append(outcome)

        if not indexed and not skipped and failed:
            raise RagError(
                "Nothing could be indexed. "
                + "; ".join(f"{item['document']}: {item['error']}" for item in failed[:3])
            )

        documents, chunks = self.store.counts()
        return {
            "success": True,
            "indexed": [
                {"document": item["document"], "chunks": item["chunks"]}
                for item in indexed
            ],
            "skipped": [item["document"] for item in skipped],
            "failed": failed,
            "documents_total": documents,
            "chunks_total": chunks,
        }

    def _index_one(self, path: Path, *, force: bool) -> dict[str, Any]:
        """Index a single file, skipping it when it is already up to date."""
        try:
            stat = path.stat()
        except OSError as exc:
            raise RagError(f"Could not read {path.name}: {exc}") from None

        existing = self.store.get_document(str(path))

        if existing is not None and not force:
            # Free check first. Only when it fails is the file read to hash it.
            if (
                existing.size_bytes == stat.st_size
                and existing.mtime_ns == stat.st_mtime_ns
            ):
                return {"document": path.name, "chunks": existing.chunk_count, "skipped": True}

            digest = self._hash(path)
            if digest == existing.sha256:
                # Touched but not changed - a copy, or a sync tool. Record the
                # new timestamp so the cheap check works next time.
                self.store.touch_document(
                    existing.id, mtime_ns=stat.st_mtime_ns, size_bytes=stat.st_size
                )
                return {"document": path.name, "chunks": existing.chunk_count, "skipped": True}
        else:
            digest = self._hash(path)

        try:
            document = extract(path, max_bytes=self.max_file_bytes, ocr=self.ocr)
        except ExtractionError as exc:
            raise RagError(str(exc)) from None

        chunks = chunk_document(
            document,
            chunk_tokens=self.chunk_tokens,
            overlap_tokens=self.overlap_tokens,
        )
        if not chunks:
            raise RagError(f"{path.name} produced no chunks worth indexing.")

        # The slow, failure-prone step, done before anything is written.
        vectors = self._embed_passages([chunk.text for chunk in chunks])

        rows = self._allocate(len(chunks))
        try:
            self.index.write(rows, vectors)
        except VectorIndexError as exc:
            # The rows were claimed but never used; hand them back so they are
            # not lost until the next rebuild.
            self.store.release_rows(rows)
            raise RagError(str(exc)) from None

        _, freed = self.store.replace_document(
            path=str(path),
            name=path.name,
            suffix=path.suffix.lower(),
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            sha256=digest,
            pages=document.page_count,
            chunks=[
                (chunk.ordinal, chunk.page, chunk.section, chunk.text, row)
                for chunk, row in zip(chunks, rows)
            ],
            kind=document.kind,
            characters=document.characters,
            ocr_used=document.ocr_used,
            ocr_pages=document.ocr_pages,
            note=document.ocr_note,
        )
        # The previous version's rows, now that the new one is safely stored.
        self.store.release_rows(freed)
        self._save_settings()

        return {
            "document": path.name,
            "chunks": len(chunks),
            "skipped": False,
            "ocr_used": document.ocr_used,
            "note": document.ocr_note,
        }

    # --- searching ---

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        """Find the chunks closest in meaning to `query`."""
        if not isinstance(query, str) or not query.strip():
            raise RagError("The search query must be a non-empty string.")
        query = query.strip()
        if len(query) > 2000:
            raise RagError(
                f"That query is {len(query)} characters. Search with a "
                f"question, not a document."
            )

        limit = max(1, min(int(top_k if top_k is not None else self.top_k), 50))
        threshold = float(min_score if min_score is not None else self.min_score)

        documents, total_chunks = self.store.counts()
        if total_chunks == 0:
            return {
                "success": True,
                "query": query,
                "count": 0,
                "results": [],
                "note": (
                    "Nothing has been indexed yet, so there is nothing to "
                    "search. Index a file or folder first."
                ),
            }

        self._check_compatible()

        free = self.store.free_row_set()
        try:
            self.index.verify(total_chunks)
            vector = self.embedder.encode_query(query)
            # A small margin over `limit`, so a hit whose chunk has somehow
            # gone still leaves enough results to fill the answer. Bounded
            # rather than `limit + len(free)`: the free list is normally tiny,
            # but after deleting a large collection it is not, and top_k is
            # what sizes the running result arrays.
            margin = min(len(free), 100)
            hits = self.index.search(vector, top_k=limit + margin, skip=free)
        except EmbeddingError as exc:
            raise RagError(str(exc)) from None
        except VectorIndexError as exc:
            raise RagError(str(exc)) from None

        records = self.store.chunks_by_rows([row for row, _ in hits])

        results: list[SearchHit] = []
        for row, score in hits:
            record = records.get(row)
            if record is None:
                continue
            if score < threshold:
                continue
            results.append(_hit(record, score))
            if len(results) >= limit:
                break

        payload: dict[str, Any] = {
            "success": True,
            "query": query,
            "count": len(results),
            "results": [hit.as_dict() for hit in results],
        }
        if not results:
            best = max((score for _, score in hits), default=0.0)
            payload["note"] = (
                f"No chunk scored above the {threshold:g} similarity "
                f"threshold (best was {best:.2f}) across {documents} "
                f"document(s). The answer may not be in the indexed files."
            )
        return payload

    # --- management ---

    def list_documents(self) -> dict[str, Any]:
        documents = self.store.list_documents()
        _, chunks = self.store.counts()
        return {
            "success": True,
            "count": len(documents),
            "chunks_total": chunks,
            "documents": [
                {
                    "id": document.id,
                    "document": document.name,
                    "path": document.path,
                    "chunks": document.chunk_count,
                    "pages": document.pages,
                    "size_bytes": document.size_bytes,
                    "indexed_at": document.indexed_at,
                }
                for document in documents
            ],
        }

    def remove(self, document: str | int) -> dict[str, Any]:
        """Remove one document from the index, by id, name or path."""
        target = self._resolve_document(document)
        name, freed = self.store.delete_document(target.id)
        return {
            "success": True,
            "document": name,
            "removed_chunks": len(freed),
        }

    def compact(self) -> dict[str, Any]:
        """Squeeze deleted rows out of the vector file, without re-embedding.

        Cheap: it moves vectors that already exist. Use it after deleting a lot
        of documents. `rebuild` is the one that reads the source files again.
        """
        free = self.store.free_row_set()
        pairs = self.store.all_chunk_rows()
        if not free:
            return {
                "success": True,
                "compacted": False,
                "note": "Nothing to reclaim - the index has no free rows.",
                "chunks": len(pairs),
            }

        try:
            mapping = self.index.compact([row for _, row in pairs])
        except VectorIndexError as exc:
            raise RagError(str(exc)) from None
        self.store.remap_rows(mapping)
        return {
            "success": True,
            "compacted": True,
            "reclaimed_rows": len(free),
            "chunks": len(pairs),
        }

    def rebuild(self, *, progress: Progress | None = None) -> dict[str, Any]:
        """Re-read and re-embed every indexed document from its source file.

        The expensive operation, and the right one after changing the
        embedding model or the chunk size - both of which make existing
        vectors incomparable with new ones. Documents whose source file has
        gone are dropped and reported.
        """
        documents = self.store.list_documents()
        if not documents:
            raise RagError("Nothing has been indexed, so there is nothing to rebuild.")

        paths = [Path(document.path) for document in documents]
        missing = [path.name for path in paths if not path.is_file()]

        self.store.clear()
        try:
            self.index.clear()
        except VectorIndexError as exc:
            raise RagError(str(exc)) from None
        self._save_settings()

        rebuilt: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        present = [path for path in paths if path.is_file()]
        for position, path in enumerate(present, start=1):
            if progress is not None:
                progress(path.name, position, len(present))
            try:
                rebuilt.append(self._index_one(path, force=True))
            except RagError as exc:
                failed.append({"document": path.name, "error": str(exc)})

        total_documents, total_chunks = self.store.counts()
        return {
            "success": True,
            "rebuilt": [
                {"document": item["document"], "chunks": item["chunks"]}
                for item in rebuilt
            ],
            "dropped": missing,
            "failed": failed,
            "documents_total": total_documents,
            "chunks_total": total_chunks,
        }

    def stats(self) -> dict[str, Any]:
        """What is indexed, and with what settings."""
        documents, chunks = self.store.counts()
        settings = self.store.settings()
        try:
            vector_bytes = self.index.path.stat().st_size
        except OSError:
            vector_bytes = 0
        return {
            "success": True,
            "documents": documents,
            "chunks": chunks,
            "model": settings.get("model", self.model),
            "dimension": int(settings.get("dimension", self.index.dimension)),
            "chunk_tokens": int(settings.get("chunk_tokens", self.chunk_tokens)),
            "overlap_tokens": int(settings.get("overlap_tokens", self.overlap_tokens)),
            "store": str(self.store_dir),
            "vector_bytes": vector_bytes,
            "embedder_loaded": self._embedder is not None and self._embedder.loaded,
        }

    # --- internals ---

    def _embed_passages(self, texts: list[str]) -> np.ndarray:
        try:
            vectors = self.embedder.encode_passages(texts)
        except EmbeddingError as exc:
            raise RagError(str(exc)) from None

        if vectors.shape[1] != self.index.dimension:
            raise RagError(
                f"The embedding model returned {vectors.shape[1]}-dimensional "
                f"vectors but the index is {self.index.dimension}-dimensional. "
                f"Rebuild the index, or set RAG_DIMENSION to match the model."
            )
        return vectors

    def _allocate(self, count: int) -> list[int]:
        """Claim `count` vector rows, reusing freed ones first.

        The boundary is read *before* the free list is drained: taking rows off
        the free list lowers the highest known row, and appending from there
        would hand out a row that was just claimed.
        """
        boundary = self.store.max_vector_row() + 1
        reused = self.store.take_free_rows(count)
        fresh = list(range(boundary, boundary + count - len(reused)))
        return reused + fresh

    def _walk(self, root: Path, *, recursive: bool) -> list[Path]:
        """Every supported file under `root`, in a stable order."""
        found: list[Path] = []
        if not recursive:
            for child in sorted(root.iterdir()):
                if child.is_file() and is_supported(child):
                    found.append(child)
            return found

        for current, directories, files in os.walk(root):
            # Pruned in place, which is what stops `walk` descending into them.
            directories[:] = sorted(
                name for name in directories
                if name not in SKIP_DIRECTORIES and not name.startswith(".")
            )
            for name in sorted(files):
                candidate = Path(current) / name
                if is_supported(candidate):
                    found.append(candidate)
        return found

    def _hash(self, path: Path) -> str:
        """SHA-256 of a file, read in blocks."""
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                while block := handle.read(HASH_BLOCK):
                    digest.update(block)
        except OSError as exc:
            raise RagError(f"Could not read {path.name}: {exc}") from None
        return digest.hexdigest()

    def _check_disk(self) -> None:
        try:
            free_mb = shutil.disk_usage(self.store_dir).free // (1024 * 1024)
        except OSError:
            return  # cannot tell; the write itself will report ENOSPC
        if free_mb < MIN_FREE_DISK_MB:
            raise RagError(
                f"Only {free_mb} MB free on the drive holding {self.store_dir}. "
                f"Indexing needs at least {MIN_FREE_DISK_MB} MB of headroom."
            )

    def _check_compatible(self) -> None:
        """Refuse to mix vectors from different models or widths.

        Vectors from two models share a coordinate space by coincidence only,
        so searching a mixed index returns confident nonsense. Caught here, in
        the one place both settings are known.
        """
        settings = self.store.settings()
        if not settings:
            return  # a fresh store adopts the current settings

        stored_model = settings.get("model", "")
        if stored_model and stored_model != self.model:
            raise RagError(
                f"The index was built with {stored_model!r} and the configured "
                f"model is {self.model!r}. Vectors from two models cannot be "
                f"compared - rebuild the index, or set RAG_MODEL back."
            )

        stored_dimension = int(settings.get("dimension", self.index.dimension))
        if stored_dimension != self.index.dimension:
            raise RagError(
                f"The index holds {stored_dimension}-dimensional vectors and "
                f"this is configured for {self.index.dimension}. Rebuild it."
            )

        stored_version = int(settings.get("schema_version", SCHEMA_VERSION))
        if stored_version != SCHEMA_VERSION:
            raise RagError(
                f"The index was written by an older version of this tool "
                f"(schema {stored_version}, this is {SCHEMA_VERSION}). "
                f"Rebuild it."
            )

    def _save_settings(self) -> None:
        self.store.save_settings(
            {
                "model": self.model,
                "dimension": str(self.index.dimension),
                "chunk_tokens": str(self.chunk_tokens),
                "overlap_tokens": str(self.overlap_tokens),
                "schema_version": str(SCHEMA_VERSION),
            }
        )

    def _resolve_document(self, document: str | int) -> Document:
        """Find a document by id, exact path, or file name."""
        if isinstance(document, bool):
            raise RagError("Give a document id, name or path.")

        if isinstance(document, int):
            found = self.store.get_document_by_id(document)
            if found is None:
                raise RagError(f"No indexed document with id {document}.")
            return found

        if not isinstance(document, str) or not document.strip():
            raise RagError("Give a document id, name or path.")
        text = document.strip()

        if text.isdigit():
            return self._resolve_document(int(text))

        exact = self.store.get_document(text)
        if exact is not None:
            return exact
        try:
            exact = self.store.get_document(str(Path(text).expanduser().resolve()))
        except OSError:
            exact = None
        if exact is not None:
            return exact

        matches = [
            candidate
            for candidate in self.store.list_documents()
            if candidate.name.lower() == text.lower()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RagError(
                f"{len(matches)} indexed documents are called {text!r}. Use the "
                f"id or the full path: "
                + ", ".join(f"{item.id} ({item.path})" for item in matches[:5])
            )
        raise RagError(f"{text!r} is not in the index.")


def _hit(record: ChunkRecord, score: float) -> SearchHit:
    return SearchHit(
        text=record.text,
        score=score,
        document=record.document,
        path=record.path,
        chunk_id=record.chunk_id,
        page=record.page,
    )
