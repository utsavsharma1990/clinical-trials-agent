"""Web search tool using Tavily API for clinical trial intelligence."""

from __future__ import annotations

import os

from tavily import TavilyClient

CLINICAL_DOMAINS = [
    "clinicaltrials.gov",
    "fda.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "who.int",
    "ema.europa.eu",
    "reuters.com",
    "statnews.com",
    "fiercebiotech.com",
    "biopharmadive.com",
    "nejm.org",
    "thelancet.com",
    "jama.jamanetwork.com",
    "nature.com",
    "bmj.com",
]


def _get_client() -> TavilyClient:
    """Instantiate Tavily client from environment."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY environment variable is not set")
    return TavilyClient(api_key=api_key)


def search_clinical_context(
    query: str,
    max_results: int = 5,
) -> list[dict]:
    """Search trusted clinical/pharma sources for context on a query.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, url, content, score, published_date.
    """
    client = _get_client()
    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            include_domains=CLINICAL_DOMAINS,
            max_results=max_results,
        )
        results = []
        for item in response.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "published_date": item.get("published_date", ""),
                }
            )
        return results
    except Exception as exc:
        return [{"error": f"Web search failed: {exc}", "url": "", "content": "", "title": ""}]


def search_regulatory_news(
    nct_id: str,
    sponsor: str | None = None,
) -> list[dict]:
    """Search for regulatory news and trial updates for a specific NCT ID.

    Combines the NCT ID with sponsor name and relevant regulatory terms for
    a targeted news search.

    Args:
        nct_id: The NCT identifier.
        sponsor: Optional sponsor name to improve result relevance.

    Returns:
        List of dicts with keys: title, url, content, score, published_date.
    """
    query_parts = [nct_id, "clinical trial results FDA"]
    if sponsor:
        query_parts.insert(1, sponsor)
    query = " ".join(query_parts)
    return search_clinical_context(query=query, max_results=5)
