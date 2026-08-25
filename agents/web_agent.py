"""Web Intelligence Agent — retrieves context from web search and PubMed."""

from __future__ import annotations

import time

from agents.state import ClinicalTrialsAgentState
from tools.pubmed_search import search_pubmed_for_trial
from tools.web_search import search_clinical_context, search_regulatory_news

# Keywords that signal the user wants published outcome data
_RESULTS_KEYWORDS = {
    "results", "outcome", "efficacy", "survival", "response rate",
    "hazard ratio", "published", "paper", "study results",
    "phase 3 results", "data", "findings", "safety", "adverse",
}


def _should_search_pubmed(query: str, intent: str, nct_ids: list[str]) -> bool:
    """Return True when PubMed should be queried for this request."""
    if not nct_ids:
        return False
    # Always trigger for FDAAA checks and hybrid queries
    if intent in ("fdaaa_check", "hybrid"):
        return True
    # For NCT lookups, only trigger when the user explicitly wants outcomes
    if intent == "nct_lookup":
        q_lower = query.lower()
        return any(kw in q_lower for kw in _RESULTS_KEYWORDS)
    return False


def _build_search_query(
    user_query: str,
    nct_ids: list[str],
    ct_results: list[dict],
) -> str:
    """Construct a targeted search query from user question and available context."""
    parts = [user_query[:150]]

    for nct_id in nct_ids[:2]:
        if nct_id not in user_query:
            parts.append(nct_id)

    # Add sponsor name if available from CT results for better precision
    for result in ct_results[:1]:
        sponsor = result.get("sponsor_name", "")
        if sponsor and sponsor not in user_query and len(sponsor) < 50:
            parts.append(sponsor)

    return " ".join(parts)


def run_web_agent(state: ClinicalTrialsAgentState) -> dict:
    """Search the web for clinical trial context, news, and regulatory information.

    Reads: user_query, extracted_nct_ids, ct_api_results, query_intent
    Writes: web_search_results, retrieval_sources, agent_trace
    """
    start_ts = time.time()
    user_query = state.get("user_query", "")
    nct_ids = state.get("extracted_nct_ids", [])
    ct_results = state.get("ct_api_results", [])
    query_intent = state.get("query_intent", "")

    web_results: list[dict] = []
    retrieval_sources: list[str] = list(state.get("retrieval_sources", []))
    trace_entries: list[dict] = []

    # ── Primary web search ────────────────────────────────────────────────────
    search_query = _build_search_query(user_query, nct_ids, ct_results)
    search_start = time.time()

    try:
        primary_results = search_clinical_context(query=search_query, max_results=5)
        search_latency = int((time.time() - search_start) * 1000)

        valid_primary = [r for r in primary_results if not r.get("error") and r.get("url")]
        web_results.extend(valid_primary)

        trace_entries.append(
            {
                "agent": "web_search",
                "action": "primary_search",
                "result": f"{len(valid_primary)} results",
                "query": search_query[:80],
                "latency_ms": search_latency,
            }
        )
    except Exception as exc:
        trace_entries.append(
            {
                "agent": "web_search",
                "action": "primary_search",
                "result": f"error: {exc}",
                "latency_ms": int((time.time() - search_start) * 1000),
            }
        )

    # ── Regulatory news search for specific NCT IDs ──────────────────────────
    if nct_ids and query_intent in ("fdaaa_check", "nct_lookup", "hybrid"):
        for nct_id in nct_ids[:2]:
            reg_start = time.time()
            sponsor = ""
            for r in ct_results:
                if r.get("nct_id", "").upper() == nct_id.upper():
                    sponsor = r.get("sponsor_name", "")
                    break

            try:
                reg_results = search_regulatory_news(nct_id=nct_id, sponsor=sponsor or None)
                reg_latency = int((time.time() - reg_start) * 1000)

                valid_reg = [r for r in reg_results if not r.get("error") and r.get("url")]
                web_results.extend(valid_reg)

                trace_entries.append(
                    {
                        "agent": "web_search",
                        "action": f"regulatory_news:{nct_id}",
                        "result": f"{len(valid_reg)} results",
                        "latency_ms": reg_latency,
                    }
                )
            except Exception as exc:
                trace_entries.append(
                    {
                        "agent": "web_search",
                        "action": f"regulatory_news:{nct_id}",
                        "result": f"error: {exc}",
                        "latency_ms": int((time.time() - reg_start) * 1000),
                    }
                )

    # ── Deduplicate by URL ────────────────────────────────────────────────────
    seen_urls: set[str] = set()
    deduped_results: list[dict] = []
    for item in web_results:
        url = item.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped_results.append(item)

    # Extend retrieval_sources with new web URLs
    for item in deduped_results:
        url = item.get("url", "")
        if url and url not in retrieval_sources:
            retrieval_sources.append(url)

    total_latency = int((time.time() - start_ts) * 1000)
    trace_entries.append(
        {
            "agent": "web_search",
            "action": "complete",
            "result": f"{len(deduped_results)} unique results",
            "latency_ms": total_latency,
        }
    )

    # ── PubMed published results ──────────────────────────────────────────────
    pubmed_results: list[dict] = []
    pubmed_triggered = _should_search_pubmed(user_query, query_intent, nct_ids)

    if pubmed_triggered:
        pubmed_start = time.time()
        for nct_id in nct_ids[:2]:  # max 2 to respect NCBI rate limits
            # Get trial metadata from CT results for better search
            trial_title: str | None = None
            sponsor: str | None = None
            for r in ct_results:
                if r.get("nct_id", "").upper() == nct_id.upper():
                    trial_title = r.get("brief_title") or r.get("official_title")
                    sponsor = r.get("sponsor_name")
                    break

            papers = search_pubmed_for_trial(
                nct_id=nct_id,
                trial_title=trial_title,
                sponsor=sponsor,
            )
            pubmed_results.extend(papers)

            # Add PubMed URLs to retrieval sources
            for paper in papers:
                url = paper.get("pubmed_url", "")
                if url and url not in retrieval_sources:
                    retrieval_sources.append(url)

        pubmed_latency = int((time.time() - pubmed_start) * 1000)
        trace_entries.append(
            {
                "agent": "web_agent",
                "action": "pubmed_search",
                "nct_ids": nct_ids[:2],
                "papers_found": len(pubmed_results),
                "latency_ms": pubmed_latency,
            }
        )

    return {
        "web_search_results": deduped_results,
        "retrieval_sources": retrieval_sources,
        "pubmed_results": pubmed_results,
        "pubmed_papers_found": len(pubmed_results),
        "pubmed_triggered": pubmed_triggered,
        "agent_trace": state.get("agent_trace", []) + trace_entries,
    }
