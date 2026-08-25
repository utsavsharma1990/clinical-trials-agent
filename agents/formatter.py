"""Response Formatter Agent — produces the final clean markdown answer."""

from __future__ import annotations

import time

from agents.state import ClinicalTrialsAgentState


def _grade_badge(grade: str, overall_score: float) -> str:
    """Return a confidence badge string based on quality grade."""
    if grade == "A":
        return "✅ **High confidence answer** (Quality Grade A)"
    elif grade == "B":
        return "⚡ **Good confidence answer** (Quality Grade B)"
    elif grade == "C":
        return "⚠️ **Moderate confidence — verify key details** (Quality Grade C)"
    else:
        return "🔴 **Low confidence — treat as preliminary** (Quality Grade F)"


def _format_citations(citations: list[dict]) -> str:
    """Format citations as a numbered markdown list."""
    if not citations:
        return ""

    lines = ["\n---\n**Sources**\n"]
    seen: set[str] = set()
    idx = 1

    for cite in citations:
        url = cite.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        title = cite.get("title", url[:60])
        source_type = cite.get("source_type", "Source")
        lines.append(f"{idx}. [{title}]({url}) — *{source_type}*")
        idx += 1

    return "\n".join(lines) if idx > 1 else ""


def run_formatter(state: ClinicalTrialsAgentState) -> dict:
    """Format the synthesized answer into clean, production-ready markdown.

    Reads: synthesized_answer, citations, quality_scores, agent_trace,
           revision_count, metrics_summary
    Writes: final_answer, metrics_summary, agent_trace
    """
    start_ts = time.time()
    answer = state.get("synthesized_answer", "")
    citations = state.get("citations", [])
    quality_scores = state.get("quality_scores", {})
    revision_count = state.get("revision_count", 0)
    quality_feedback = state.get("quality_feedback", "")

    grade = quality_scores.get("grade", "C")
    overall_score = quality_scores.get("overall_score", 0.0)

    # ── Assemble final answer ─────────────────────────────────────────────────
    sections: list[str] = []

    # Main answer body
    sections.append(answer.strip())

    # Quality warning if max revisions reached
    if revision_count >= 2 and not quality_scores.get("passed", True):
        sections.append(
            "\n> ⚠️ **Quality Notice**: This answer reached the maximum revision limit. "
            "Some claims may not be fully verifiable from retrieved sources. "
            "Cross-reference with ClinicalTrials.gov directly for critical decisions."
        )
    elif quality_feedback and "Max revisions" in quality_feedback:
        sections.append(
            "\n> ⚠️ **Quality Notice**: Answer presented after maximum revisions. "
            "Verify key details against primary sources."
        )

    # Citations
    citation_block = _format_citations(citations)
    if citation_block:
        sections.append(citation_block)

    # Confidence badge
    badge = _grade_badge(grade, overall_score)
    sections.append(f"\n---\n{badge}")

    final_answer = "\n\n".join(sections)

    # ── Build metrics_summary for Streamlit sidebar ───────────────────────────
    metrics_summary = {
        "faithfulness": quality_scores.get("faithfulness", 0.0),
        "completeness": quality_scores.get("completeness", 0.0),
        "source_coverage": quality_scores.get("source_coverage", 0.0),
        "hallucination_risk": quality_scores.get("hallucination_risk", 0.0),
        "overall_score": overall_score,
        "grade": grade,
        "passed": quality_scores.get("passed", False),
        "revision_count": revision_count,
        "citations_count": len(
            {c.get("url") for c in citations if c.get("url")}
        ),
        "thresholds": quality_scores.get(
            "thresholds",
            {
                "faithfulness": 0.85,
                "completeness": 0.80,
                "source_coverage": 0.75,
                "hallucination_risk": 0.80,
            },
        ),
    }

    latency_ms = int((time.time() - start_ts) * 1000)
    new_trace_entry = {
        "agent": "formatter",
        "action": "format_response",
        "grade": grade,
        "overall_score": overall_score,
        "citations_count": metrics_summary["citations_count"],
        "latency_ms": latency_ms,
    }

    return {
        "final_answer": final_answer,
        "metrics_summary": metrics_summary,
        "agent_trace": state.get("agent_trace", []) + [new_trace_entry],
    }
