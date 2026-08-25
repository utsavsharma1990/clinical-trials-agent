"""Quality Check Agent — evaluates answer accuracy and routes retries."""

from __future__ import annotations

import time

from agents.state import ClinicalTrialsAgentState
from tools.metrics import (
    aggregate_quality_scores,
    compute_completeness,
    compute_faithfulness,
    compute_hallucination_risk,
    compute_source_coverage,
)

MAX_REVISIONS = 2


def _build_source_texts(
    ct_results: list[dict],
    web_results: list[dict],
) -> list[str]:
    """Collect all source text for faithfulness checking."""
    texts: list[str] = []
    for study in ct_results:
        if study.get("_formatted_summary"):
            texts.append(study["_formatted_summary"])
        elif not study.get("error"):
            summary = study.get("brief_summary", "")
            if summary:
                texts.append(summary)
        fdaaa = study.get("fdaaa_status_data")
        if fdaaa:
            texts.append(str(fdaaa))
    for item in web_results:
        content = item.get("content", "")
        if content and not item.get("error"):
            texts.append(content[:800])
    return texts


def _generate_feedback(
    scores: dict,
    answer: str,
    citations: list[dict],
) -> str:
    """Produce specific actionable feedback for the synthesis agent."""
    issues: list[str] = []

    if scores["faithfulness"] < 0.85:
        issues.append(
            f"FAITHFULNESS ({scores['faithfulness']:.2f} < 0.85): Some factual claims in "
            f"the answer could not be traced to retrieved sources. Remove or hedge any "
            f"specific statistics, dates, or outcomes that are not explicitly stated in "
            f"the retrieved ClinicalTrials.gov data or web sources."
        )

    if scores["completeness"] < 0.80:
        issues.append(
            f"COMPLETENESS ({scores['completeness']:.2f} < 0.80): The answer did not "
            f"address all parts of the question. Review the question carefully and ensure "
            f"every information need is either answered or explicitly flagged as unavailable."
        )

    if scores["hallucination_risk"] < 0.80:
        issues.append(
            f"HALLUCINATION RISK ({scores['hallucination_risk']:.2f} < 0.80): The answer "
            f"contains statements that may not be supported by retrieved data. "
            f"Check: (1) Any NCT IDs mentioned must appear in retrieved results. "
            f"(2) Specific percentages or p-values must come from source documents. "
            f"(3) Avoid definitive statements about future trial outcomes."
        )

    if scores["source_coverage"] < 0.75:
        ct_count = sum(
            1 for c in citations if "clinicaltrials.gov" in c.get("url", "").lower()
        )
        web_count = sum(
            1 for c in citations if "clinicaltrials.gov" not in c.get("url", "").lower()
        )
        issues.append(
            f"SOURCE COVERAGE ({scores['source_coverage']:.2f} < 0.75): "
            f"The required source types are not present (CT.gov: {ct_count}, web: {web_count}). "
            f"Ensure you cite from the appropriate source types for this query intent."
        )

    if not issues:
        return "Minor quality issues detected. Ensure all claims are properly sourced."

    return "\n\n".join(issues)


def run_quality_agent(state: ClinicalTrialsAgentState) -> dict:
    """Evaluate answer quality and decide whether to pass or trigger a revision.

    Reads: synthesized_answer, user_query, ct_api_results, web_search_results,
           citations, query_intent, revision_count
    Writes: quality_scores, quality_passed, quality_feedback, revision_count,
            metrics_summary, agent_trace
    """
    start_ts = time.time()
    answer = state.get("synthesized_answer", "")
    question = state.get("user_query", "")
    ct_results = state.get("ct_api_results", [])
    web_results = state.get("web_search_results", [])
    citations = state.get("citations", [])
    query_intent = state.get("query_intent", "")
    revision_count = state.get("revision_count", 0)

    source_texts = _build_source_texts(ct_results, web_results)

    # ── Compute all four metrics ──────────────────────────────────────────────
    f_start = time.time()
    faithfulness = compute_faithfulness(answer, source_texts)
    f_latency = int((time.time() - f_start) * 1000)

    c_start = time.time()
    completeness = compute_completeness(answer, question)
    c_latency = int((time.time() - c_start) * 1000)

    sc_start = time.time()
    source_coverage = compute_source_coverage(citations, query_intent)
    sc_latency = int((time.time() - sc_start) * 1000)

    h_start = time.time()
    hallucination_risk = compute_hallucination_risk(answer, ct_results)
    h_latency = int((time.time() - h_start) * 1000)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    scores = aggregate_quality_scores(
        faithfulness=faithfulness,
        completeness=completeness,
        source_coverage=source_coverage,
        hallucination_risk=hallucination_risk,
    )

    passed = scores["passed"]
    feedback = ""
    new_revision_count = revision_count

    if not passed:
        if revision_count >= MAX_REVISIONS:
            # Max retries reached — pass with warning
            passed = True
            feedback = (
                f"Max revisions ({MAX_REVISIONS}) reached. Answer presented with quality warnings. "
                f"Overall score: {scores['overall_score']:.2f} (Grade {scores['grade']}). "
                f"Failing metrics: "
                + _generate_feedback(scores, answer, citations)
            )
        else:
            new_revision_count = revision_count + 1
            feedback = _generate_feedback(scores, answer, citations)

    total_latency = int((time.time() - start_ts) * 1000)

    trace_entries = [
        {
            "agent": "quality_check",
            "action": "evaluate",
            "faithfulness": faithfulness,
            "completeness": completeness,
            "source_coverage": source_coverage,
            "hallucination_risk": hallucination_risk,
            "overall_score": scores["overall_score"],
            "grade": scores["grade"],
            "passed": passed,
            "revision_count": new_revision_count,
            "latency_breakdown_ms": {
                "faithfulness": f_latency,
                "completeness": c_latency,
                "source_coverage": sc_latency,
                "hallucination_risk": h_latency,
            },
            "latency_ms": total_latency,
        }
    ]

    metrics_summary = {
        **scores,
        "revision_count": new_revision_count,
        "quality_feedback": feedback,
    }

    return {
        "quality_scores": scores,
        "quality_passed": passed,
        "quality_feedback": feedback,
        "revision_count": new_revision_count,
        "metrics_summary": metrics_summary,
        "agent_trace": state.get("agent_trace", []) + trace_entries,
    }
