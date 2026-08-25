"""Shared LangGraph state definition for the Clinical Trials Agent."""

from typing import TypedDict


class ClinicalTrialsAgentState(TypedDict):
    """Complete state passed between all agents in the pipeline."""

    # ── Input ─────────────────────────────────────────────────────────────────
    user_query: str
    conversation_history: list[dict]  # {role: str, content: str} pairs

    # ── Router output ─────────────────────────────────────────────────────────
    query_intent: str        # nct_lookup | search_trials | fdaaa_check | web_research | hybrid
    requires_ct_api: bool
    requires_web_search: bool
    extracted_nct_ids: list[str]
    search_params: dict      # condition, sponsor, date_range, status, phase

    # ── Retrieval outputs ─────────────────────────────────────────────────────
    ct_api_results: list[dict]       # raw ClinicalTrials.gov responses
    web_search_results: list[dict]   # raw web results with urls + snippets
    retrieval_sources: list[str]     # all source URLs for citation
    pubmed_results: list[dict]       # raw PubMed paper dicts
    pubmed_papers_found: int         # count for display in UI
    pubmed_triggered: bool           # whether PubMed was called this query

    # ── Synthesis output ──────────────────────────────────────────────────────
    synthesized_answer: str
    answer_confidence: float         # 0.0 to 1.0

    # ── Quality check outputs ─────────────────────────────────────────────────
    quality_scores: dict     # faithfulness, completeness, source_coverage, hallucination_risk
    quality_passed: bool
    quality_feedback: str    # what to improve if failed
    revision_count: int      # retry counter, max 2

    # ── Final output ──────────────────────────────────────────────────────────
    final_answer: str
    citations: list[dict]    # {title, url, source_type}
    agent_trace: list[dict]  # {agent, action, result, latency_ms}
    metrics_summary: dict    # all scores for UI display
    error: str


def make_initial_state(
    query: str,
    history: list[dict] | None = None,
) -> ClinicalTrialsAgentState:
    """Return a clean initial state for a new query."""
    return ClinicalTrialsAgentState(
        user_query=query,
        conversation_history=history or [],
        query_intent="",
        requires_ct_api=False,
        requires_web_search=False,
        extracted_nct_ids=[],
        search_params={},
        ct_api_results=[],
        web_search_results=[],
        retrieval_sources=[],
        pubmed_results=[],
        pubmed_papers_found=0,
        pubmed_triggered=False,
        synthesized_answer="",
        answer_confidence=0.0,
        quality_scores={},
        quality_passed=False,
        quality_feedback="",
        revision_count=0,
        final_answer="",
        citations=[],
        agent_trace=[],
        metrics_summary={},
        error="",
    )
