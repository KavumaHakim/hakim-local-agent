"""Managing the document index over HTTP.

The model gets to search. Everything that *changes* the index - indexing,
removing, rebuilding - is here instead, because those are decisions a person
makes, and the same split already applies to every other tool in this project.

These routes are declared `def`, not `async def`, so FastAPI runs them on its
threadpool. Indexing is minutes of blocking CPU work, and on the event loop it
would stall the chat stream of a turn already in progress.

That still leaves both competing for two cores, which is why `_guard_busy`
refuses to start an ingest mid-turn: embedding a folder while the 8B model is
generating makes both slow and risks the memory ceiling. Searching is allowed
during a turn - it is one short embed, and refusing it would break the agent's
own tool call.

Nothing from `rag` is imported at module scope. This module is imported when
the app starts, and `rag.manager` pulls in numpy and the rest of the package -
which would make the project's largest optional dependency a hard requirement
for starting the API at all.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import get_runtime
from api.runtime import Runtime
from api.schemas import (
    RagDocumentsOut,
    RagIndexRequest,
    RagIndexResult,
    RagRemoveResult,
    RagSearchRequest,
    RagOutlineResult,
    RagSearchResult,
    RagStatsOut,
)

router = APIRouter(prefix="/rag", tags=["rag"])

MISSING_DEPENDENCIES = (
    "Document search needs its optional dependencies. Install them with: "
    "pip install -r requirements-rag.txt"
)


def _rag_errors() -> tuple[type[BaseException], ...]:
    """The manager's error type, as a tuple that is safe in an `except` clause.

    Empty when the optional dependencies are missing. That matters more than it
    looks: an `except` clause is evaluated *while* an exception is being
    handled, so writing `except _rag().RagError` would turn a clean 501 about a
    missing package into an ImportError traceback from inside the handler.
    `except ()` simply never matches.
    """
    try:
        from rag.manager import RagError
    except ImportError:
        return ()
    return (RagError,)


def _manager(runtime: Runtime):
    """A RagManager built from the current settings.

    Cheap - it opens a SQLite file. The embedding worker behind it is shared
    process-wide, so this does not start a second copy of the model.
    """
    try:
        from rag.embeddings import shared_embedder
        from rag.manager import RagError, RagManager
    except ImportError as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED, f"{MISSING_DEPENDENCIES} ({exc})"
        ) from None

    config = runtime.effective_config()
    try:
        return RagManager(
            config.rag_store,
            embedder=shared_embedder(
                model=config.rag_model,
                model_dir=config.rag_model_dir or None,
                threads=config.rag_threads,
                batch_size=config.rag_batch_size,
                idle_seconds=config.rag_idle_seconds,
            ),
            model=config.rag_model,
            dimension=config.rag_dimension,
            chunk_tokens=config.rag_chunk_tokens,
            overlap_tokens=config.rag_overlap_tokens,
            top_k=config.rag_top_k,
            min_score=config.rag_min_score,
            max_file_bytes=config.rag_max_file_bytes,
            hybrid=config.rag_hybrid,
            figures=config.rag_figures,
        )
    except RagError as exc:
        # The store directory could not be created, which is a server problem
        # rather than a bad request.
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)
        ) from None


def _run(runtime: Runtime, operation, *, missing: int = status.HTTP_400_BAD_REQUEST):
    """Call one manager method, mapping its failure to an HTTP status.

    Every route is "build the manager, call one thing, translate the error",
    and writing that out eight times invites the eight of them to drift.
    """
    manager = _manager(runtime)
    try:
        return operation(manager)
    except _rag_errors() as exc:
        raise HTTPException(missing, str(exc)) from None


def _guard_busy(runtime: Runtime) -> None:
    """Refuse work that would fight a running turn for the CPU."""
    if runtime.queue.busy():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A turn is running. Indexing would compete with it for both cores "
            "and for memory; try again when it finishes.",
        )


@router.get("/documents", response_model=RagDocumentsOut)
def list_documents(runtime: Runtime = Depends(get_runtime)):
    """Every indexed document."""
    return RagDocumentsOut(**_run(runtime, lambda m: m.list_documents()))


@router.get("/stats", response_model=RagStatsOut)
def stats(runtime: Runtime = Depends(get_runtime)):
    """Index size, the model it was built with, and whether that model is loaded."""
    return RagStatsOut(**_run(runtime, lambda m: m.stats()))


@router.post("/index", response_model=RagIndexResult)
def index(body: RagIndexRequest, runtime: Runtime = Depends(get_runtime)):
    """Index a file or a folder.

    Slow - minutes for a large folder on this hardware - and deliberately
    synchronous. A background job would need progress reporting, cancellation
    and a way to survive a restart, none of which is worth building before
    someone has actually waited too long for this.
    """
    _guard_busy(runtime)
    return RagIndexResult(
        **_run(
            runtime,
            lambda m: m.index_path(
                body.path, recursive=body.recursive, force=body.force
            ),
        )
    )


@router.post("/search", response_model=RagSearchResult)
def search(body: RagSearchRequest, runtime: Runtime = Depends(get_runtime)):
    """Search, the same call the agent's tool makes.

    By meaning and by exact wording at once, optionally narrowed to one
    document or section.
    """
    return RagSearchResult(
        **_run(
            runtime,
            lambda m: m.search(
                body.query,
                top_k=body.top_k,
                min_score=body.min_score,
                document=body.document,
                section=body.section,
            ),
        )
    )


@router.get("/documents/{document}/outline", response_model=RagOutlineResult)
def outline(document: str, runtime: Runtime = Depends(get_runtime)):
    """The sections of one document, in order.

    By name or id, because this is the call you make when you know what a
    document is called and not what number it was given. Answers the question
    retrieval cannot: what is in here, before you have thought of a question
    to ask about it.
    """
    # 404, like deleting one: an unknown document is not found, not a bad
    # request. The default here is 400 because most failures in this router
    # are "that path is not indexable", which is a different thing.
    return RagOutlineResult(
        **_run(
            runtime,
            lambda m: m.outline(document),
            missing=status.HTTP_404_NOT_FOUND,
        )
    )


@router.delete("/documents/{document_id}", response_model=RagRemoveResult)
def remove(document_id: int, runtime: Runtime = Depends(get_runtime)):
    """Remove one document and its chunks.

    By id rather than by name: two folders can hold a `notes.md`, and deleting
    the wrong one is not recoverable without re-indexing.
    """
    _guard_busy(runtime)
    return RagRemoveResult(
        **_run(
            runtime,
            lambda m: m.remove(document_id),
            missing=status.HTTP_404_NOT_FOUND,
        )
    )


@router.post("/rebuild", response_model=RagIndexResult)
def rebuild(runtime: Runtime = Depends(get_runtime)):
    """Re-read and re-embed every indexed document from its source file."""
    _guard_busy(runtime)
    outcome = _run(runtime, lambda m: m.rebuild())
    # `rebuilt` and `indexed` are the same shape; the response model names it
    # once so the client has one thing to read.
    return RagIndexResult(
        success=True,
        indexed=outcome["rebuilt"],
        skipped=outcome["dropped"],
        failed=outcome["failed"],
        documents_total=outcome["documents_total"],
        chunks_total=outcome["chunks_total"],
    )


@router.post("/unload", response_model=RagStatsOut)
def unload(runtime: Runtime = Depends(get_runtime)):
    """Stop the embedding model now, without waiting for the idle sweep.

    Here because the reason to want it is immediate: about to load the 8B and
    wanting every megabyte back first.
    """
    try:
        from rag import embeddings
    except ImportError as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED, f"{MISSING_DEPENDENCIES} ({exc})"
        ) from None

    embeddings.unload_shared()
    return RagStatsOut(**_run(runtime, lambda m: m.stats()))
