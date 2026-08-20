"""Web search / fetch tool (not implemented).

Interface only. No network calls, no scraping.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


def search(query: str, *, max_results: int = 5) -> list[SearchResult]:
    """Search the web.

    Raises:
        NotImplementedError: always, until a search backend is chosen.
    """
    raise NotImplementedError("Web search is not implemented yet.")


def fetch_page(url: str, *, timeout_seconds: float = 20.0) -> str:
    """Fetch a URL and return readable text.

    Raises:
        NotImplementedError: always, until fetching is implemented.
    """
    raise NotImplementedError("Web fetching is not implemented yet.")


# TODO(web): choose a search backend. A search API with a key (Brave, Tavily,
# SearXNG on localhost) is far more reliable than scraping a search engine's
# HTML, which breaks constantly and is usually against its terms.

# TODO(web): fetching needs HTML -> text extraction, a response size cap, and a
# redirect/scheme allow-list (http/https only, no file:// or localhost probing).

# TODO(security): treat fetched page content as untrusted data, never as
# instructions to the agent. This is the main prompt-injection surface.

# TODO(tools): JSON schemas for these once implemented.
