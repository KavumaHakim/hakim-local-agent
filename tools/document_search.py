"""Semantic search over the user's own documents.

Off by default (config.rag_enabled, env AGENT_ENABLE_RAG=1).

Two tools reach the model: `search_documents` and `list_documents`. Indexing,
removing and rebuilding do not - they are done from the CLI (`python -m rag`)
or the API. That is the same boundary every other tool in this project draws:
the model reads, a person decides what to change. It also keeps two more tool
definitions out of a prompt that is re-processed at a few tokens per second.

The description on `search_documents` is doing real work. The requirement is
that "according to my biology notes, explain photosynthesis" searches and
"what is 25 x 17?" does not, and on an 8B model that distinction is made
entirely by the sentence the model reads before deciding. So it names the case
it is for and the cases it is not, rather than describing the mechanism.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.manager import RagError, RagManager
from tools.base import Tool, ToolError

# Total characters of retrieved text handed back in one call. Five chunks of
# 1,750 characters is roughly 2,200 tokens, and prompt tokens cost real seconds
# here, so the budget is enforced by dropping the weakest hits rather than by
# cutting a chunk in half and leaving the model to guess at the rest.
DEFAULT_CONTEXT_BUDGET = 6000


class DocumentSearchError(ToolError):
    """A document search was rejected, or failed."""


class DocumentSearchTools:
    """The model-facing half of the RAG subsystem."""

    def __init__(
        self,
        manager: RagManager,
        *,
        context_budget: int = DEFAULT_CONTEXT_BUDGET,
    ) -> None:
        self._manager = manager
        self._budget = max(500, int(context_budget))

    # --- operations ---

    def search_documents(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        try:
            payload = self._manager.search(query, top_k=top_k)
        except RagError as exc:
            raise DocumentSearchError(str(exc)) from None

        results = payload.get("results", [])
        kept, dropped = self._fit(results)
        payload["results"] = kept
        payload["count"] = len(kept)
        if dropped:
            payload["truncated"] = (
                f"{dropped} lower-scoring result(s) were left out to keep the "
                f"retrieved text under {self._budget} characters."
            )
        return payload

    def list_documents(self) -> dict[str, Any]:
        try:
            return self._manager.list_documents()
        except RagError as exc:
            raise DocumentSearchError(str(exc)) from None

    def _fit(self, results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Keep the strongest hits that fit the context budget.

        Results arrive best-first, so this is a prefix - the model never has to
        wonder why a mid-ranked chunk is missing.
        """
        kept: list[dict[str, Any]] = []
        used = 0
        for result in results:
            length = len(result.get("text", ""))
            if kept and used + length > self._budget:
                break
            kept.append(result)
            used += length
        return kept, len(results) - len(kept)

    # --- definitions ---

    def tools(self) -> list[Tool]:
        return [
            Tool(
                name="search_documents",
                category="documents",
                description=(
                    "Search the user's indexed local documents (notes, PDFs, "
                    "files) by meaning and return the most relevant passages "
                    "with their document name and page. "
                    "Use this whenever the request refers to the user's own "
                    "material - 'my notes', 'the report', 'according to the "
                    "handbook', 'what does my document say about X' - or when "
                    "answering needs specific facts that would be in their "
                    "files rather than in general knowledge. "
                    "Do NOT use it for general knowledge, arithmetic, code you "
                    "can write yourself, or anything already answered in this "
                    "conversation. Searching returns nothing useful when the "
                    "topic was never indexed. "
                    "Quote or cite the document and page when you use a result."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "What to look for, as a question or a "
                                "description of the topic. Full sentences "
                                "retrieve better than keywords."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": (
                                "How many passages to return. Defaults to the "
                                "configured value; raise it only when one "
                                "passage is clearly not enough."
                            ),
                        },
                    },
                    "required": ["query"],
                },
                run=self.search_documents,
            ),
            Tool(
                name="list_documents",
                category="documents",
                description=(
                    "List which local documents have been indexed and are "
                    "searchable. Use it to answer 'what documents do you have' "
                    "or to check whether a file the user mentions is available "
                    "before searching for something that may not be there."
                ),
                parameters={"type": "object", "properties": {}, "required": []},
                run=self.list_documents,
            ),
        ]


def build_document_tools(
    store_dir: str | Path,
    *,
    model: str,
    model_dir: str | Path | None = None,
    dimension: int,
    chunk_tokens: int,
    overlap_tokens: int,
    top_k: int,
    min_score: float,
    max_file_bytes: int,
    context_budget: int = DEFAULT_CONTEXT_BUDGET,
    threads: int = 2,
    batch_size: int = 8,
    idle_seconds: float = 120.0,
) -> list[Tool]:
    """Build the document tools against the shared embedding worker.

    The manager is rebuilt with the registry on every turn, which is cheap. The
    embedder it points at is not rebuilt - `shared_embedder` returns the one
    process-wide worker, so a turn never starts a second copy of the model.
    """
    from rag.embeddings import shared_embedder

    embedder = shared_embedder(
        model=model,
        model_dir=model_dir,
        threads=threads,
        batch_size=batch_size,
        idle_seconds=idle_seconds,
    )
    manager = RagManager(
        store_dir,
        embedder=embedder,
        model=model,
        dimension=dimension,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        top_k=top_k,
        min_score=min_score,
        max_file_bytes=max_file_bytes,
    )
    return DocumentSearchTools(manager, context_budget=context_budget).tools()
