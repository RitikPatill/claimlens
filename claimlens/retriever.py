from __future__ import annotations

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException


def search(claim: str, max_results: int = 5) -> list[dict[str, str]]:
    """Fetch top web results for *claim* via DuckDuckGo.

    Returns up to *max_results* dicts: {"title": str, "url": str, "body": str}.

    Raises:
        RuntimeError: wrapping any DuckDuckGoSearchException so callers don't
            need to import the library exception directly.
    """
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(claim, max_results=max_results):
                body = r.get("body") or ""
                if not body:
                    continue
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "body": body,
                    }
                )
            return results
    except DuckDuckGoSearchException as exc:
        raise RuntimeError(f"DuckDuckGo search failed: {exc}") from exc
