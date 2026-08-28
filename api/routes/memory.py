"""Inspecting and steering memory over HTTP.

The split is the same one the document routes draw: the model may read and
write individual memories through its tools, while the operations that move the
whole system - running a batch, consolidating, wiping - are a person's
decision and live here.

One route here can start a model. `POST /memory/process` forces a batch, which
stops the chat model and starts the auxiliary one. It is refused mid-turn for
the same reason indexing is: the two would fight over the single model slot,
and the user waiting for an answer should win.

Nothing from `memory` is imported at module scope, so the API still starts when
the optional dependencies are absent.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import get_runtime
from api.runtime import Runtime
from api.schemas import (
    MemoryConsolidateResult,
    MemoryListOut,
    MemoryOut,
    MemoryProcessResult,
    MemoryRememberRequest,
    MemorySearchRequest,
    MemoryStatsOut,
    MemoryUpdateRequest,
)

router = APIRouter(prefix="/memory", tags=["memory"])

UNAVAILABLE = (
    "Memory needs its optional dependencies. Install them with: "
    "pip install -r requirements-rag.txt"
)


def _memory(runtime: Runtime):
    """The process-wide memory manager, or a 501."""
    manager = runtime.memory()
    if manager is None:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, UNAVAILABLE)
    return manager


def _errors() -> tuple[type[BaseException], ...]:
    """The manager's error type, safe to use in an `except` clause.

    Empty when the dependencies are missing - an `except` clause is evaluated
    while an exception is being handled, so one that can itself raise would
    replace a clean 501 with an ImportError.
    """
    try:
        from memory.manager import MemoryOperationError
    except ImportError:
        return ()
    return (MemoryOperationError,)


def _run(runtime: Runtime, operation, *, code: int = status.HTTP_400_BAD_REQUEST):
    manager = _memory(runtime)
    try:
        return operation(manager)
    except _errors() as exc:
        raise HTTPException(code, str(exc)) from None


@router.get("", response_model=MemoryListOut)
def list_memories(
    query: str = "",
    limit: int = 25,
    runtime: Runtime = Depends(get_runtime),
):
    """What is remembered. With a query, ranked by relevance."""
    payload = _run(runtime, lambda m: m.recall(query, limit=limit))
    return MemoryListOut(
        query=payload.get("query", ""),
        count=payload.get("count", 0),
        memories=[MemoryOut(**item) for item in payload.get("memories", [])],
        note=payload.get("note", ""),
    )


@router.post("/search", response_model=MemoryListOut)
def search(body: MemorySearchRequest, runtime: Runtime = Depends(get_runtime)):
    """Semantic search with scores. Needs no language model."""
    found = _run(runtime, lambda m: m.search(body.query, limit=body.limit))
    return MemoryListOut(
        query=body.query,
        count=len(found),
        memories=[MemoryOut(**item.as_dict()) for item in found],
    )


@router.post("", response_model=MemoryOut)
def remember(body: MemoryRememberRequest, runtime: Runtime = Depends(get_runtime)):
    """Store a memory now, with no queue and no model."""
    payload = _run(
        runtime,
        lambda m: m.remember(
            body.content,
            type=body.type,
            importance=body.importance,
            subject=body.subject,
        ),
    )
    return MemoryOut(**payload["memory"])


@router.patch("/{memory_id}", response_model=MemoryOut)
def update(
    memory_id: int,
    body: MemoryUpdateRequest,
    runtime: Runtime = Depends(get_runtime),
):
    """Correct a stored memory."""
    changes = body.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Give at least one field to change."
        )
    payload = _run(
        runtime,
        lambda m: m.update(memory_id, **changes),
        code=status.HTTP_404_NOT_FOUND,
    )
    return MemoryOut(**payload["memory"])


@router.delete("/{target}")
def forget(target: str, runtime: Runtime = Depends(get_runtime)):
    """Forget one memory by id, or everything about a subject."""
    return _run(
        runtime,
        lambda m: m.forget(target),
        code=status.HTTP_404_NOT_FOUND,
    )


@router.get("/stats", response_model=MemoryStatsOut)
def stats(runtime: Runtime = Depends(get_runtime)):
    """Counts, whether embeddings work, and the processor's last run."""
    return MemoryStatsOut(**_run(runtime, lambda m: m.stats()))


@router.post("/consolidate", response_model=MemoryConsolidateResult)
def consolidate(runtime: Runtime = Depends(get_runtime)):
    """Merge duplicates and resolve conflicts.

    Deterministic: it compares stored vectors and applies the rules. Pairs it
    cannot resolve are queued for the auxiliary model rather than triggering a
    model switch here.
    """
    if runtime.queue.busy():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A turn is running. Consolidation competes with it for the CPU.",
        )
    return MemoryConsolidateResult(**_run(runtime, lambda m: m.consolidate()))


@router.post("/process", response_model=MemoryProcessResult)
def process(runtime: Runtime = Depends(get_runtime)):
    """Run the memory queue now.

    This is the one route that stops the chat model and starts the auxiliary
    one. It is refused mid-turn, and the auxiliary model is always stopped
    again before it returns.
    """
    if runtime.queue.busy():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A turn is running. Memory processing would unload the model that "
            "is answering it.",
        )
    manager = _memory(runtime)
    return MemoryProcessResult(**manager.maybe_process(busy=runtime.queue.busy, force=True))
