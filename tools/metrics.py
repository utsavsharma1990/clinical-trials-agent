"""Quality metric computation for the Clinical Trials Agent.

Implements custom scoring logic (no RAGAS) using LLM-assisted claim extraction
for faithfulness and completeness, plus rule-based logic for source coverage
and hallucination risk.
"""

from __future__ import annotations

import json
import os
import re

from langchain_anthropic import ChatAnthropic

from prompts.templates import COMPLETENESS_EVAL_PROMPT, QUALITY_EVAL_PROMPT

# Thresholds
FAITHFULNESS_THRESHOLD = 0.85
COMPLETENESS_THRESHOLD = 0.80
SOURCE_COVERAGE_THRESHOLD = 0.75
HALLUCINATION_RISK_THRESHOLD = 0.80

# Grade weights
WEIGHTS = {
    "faithfulness": 0.40,
    "completeness": 0.30,
    "hallucination_risk": 0.20,
    "source_coverage": 0.10,
}


_LLM = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response text, handling markdown code fences."""
    text = text.strip()
    # Remove markdown code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}


def compute_faithfulness(answer: str, sources: list[str]) -> float:
    """Score what proportion of factual claims in the answer are traceable to sources.

    Uses an LLM to extract discrete verifiable claims, then checks each claim
    against the concatenated source text.

    Args:
        answer: The synthesized answer to evaluate.
        sources: List of source text strings (CT.gov summaries + web snippets).

    Returns:
        Faithfulness score 0.0-1.0. Returns 1.0 if no claims found.
    """
    if not answer.strip():
        return 0.0
    if not sources:
        return 0.3  # Penalise: claims without any sources

    source_text = "\n\n".join(
        f"[Source {i+1}]: {s[:800]}" for i, s in enumerate(sources[:8])
    )

    user_prompt = f"""Answer to evaluate:
{answer[:2000]}

Source documents:
{source_text[:4000]}"""

    try:
        response = _LLM.invoke([
            {"role": "system", "content": QUALITY_EVAL_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        result = _extract_json(response.content)
        score = result.get("faithfulness_score")
        if score is not None:
            return max(0.0, min(1.0, float(score)))
        # Fallback: compute from claim data
        total = result.get("total_claims", 0)
        supported = result.get("supported_claims", 0)
        if total > 0:
            return supported / total
        return 1.0  # No specific claims = trivially faithful
    except Exception:
        return 0.75  # Conservative fallback on error


def compute_completeness(answer: str, question: str) -> float:
    """Score whether the answer addresses all parts of the question.

    Uses an LLM to decompose the question into sub-questions and checks
    whether each is addressed in the answer.

    Args:
        answer: The synthesized answer to evaluate.
        question: The original user question.

    Returns:
        Completeness score 0.0-1.0. Returns 1.0 if no sub-questions found.
    """
    if not answer.strip() or not question.strip():
        return 0.0

    user_prompt = f"""Question: {question}

Answer: {answer[:2500]}"""

    try:
        response = _LLM.invoke([
            {"role": "system", "content": COMPLETENESS_EVAL_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        result = _extract_json(response.content)
        score = result.get("completeness_score")
        if score is not None:
            return max(0.0, min(1.0, float(score)))
        total = result.get("total_sub_questions", 0)
        addressed = result.get("addressed_count", 0)
        if total > 0:
            return addressed / total
        return 1.0
    except Exception:
        return 0.75


def compute_source_coverage(
    citations: list[dict],
    query_intent: str,
) -> float:
    """Score whether the correct source types are present for the query intent.

    Different intents require different source types:
    - nct_lookup / fdaaa_check: requires ClinicalTrials.gov source
    - web_research: requires at least 2 web sources
    - hybrid: requires both

    Args:
        citations: List of citation dicts with url and source_type keys.
        query_intent: The classified intent string.

    Returns:
        Source coverage score 0.0-1.0.
    """
    if not citations:
        return 0.0

    ct_sources = [
        c for c in citations if "clinicaltrials.gov" in c.get("url", "").lower()
    ]
    pubmed_sources = [
        c
        for c in citations
        if c.get("source_type") == "PubMed"
        or "pubmed.ncbi.nlm.nih.gov" in c.get("url", "").lower()
    ]
    # Web sources exclude both CT.gov and PubMed
    web_sources = [
        c
        for c in citations
        if "clinicaltrials.gov" not in c.get("url", "").lower()
        and "pubmed.ncbi.nlm.nih.gov" not in c.get("url", "").lower()
        and c.get("source_type") != "PubMed"
    ]

    has_ct = len(ct_sources) > 0
    has_pubmed = len(pubmed_sources) > 0
    has_web = len(web_sources) >= 2

    intent = query_intent.lower()

    if intent == "nct_lookup":
        # PubMed present = premium score (published efficacy data)
        if has_pubmed:
            return 1.0
        if has_ct:
            return 0.75  # Good but missing published outcomes
        return 0.3

    if intent == "fdaaa_check":
        # CT.gov is primary for FDAAA; PubMed adds results context
        if has_ct and has_pubmed:
            return 1.0
        if has_ct:
            return 0.85  # CT.gov is the main FDAAA source
        return 0.2

    if intent == "search_trials":
        return 1.0 if has_ct else 0.3

    if intent == "web_research":
        if has_web:
            return 1.0
        elif len(web_sources) == 1:
            return 0.5
        return 0.0

    if intent == "hybrid":
        if has_ct and has_web:
            return 1.0
        if has_ct or has_web:
            return 0.5
        return 0.0

    return 0.5  # Unknown intent


def compute_hallucination_risk(
    answer: str,
    ct_results: list[dict],
) -> float:
    """Score the inverse hallucination risk of the answer.

    Returns 1.0 (low risk) when the answer avoids suspicious patterns.
    Returns 0.0 (high risk) when the answer contains unverifiable specifics.

    Checks:
    - NCT IDs mentioned in answer that were not retrieved
    - Specific percentages or p-values not found in source data
    - Definitive future outcome statements ("will be approved", "will show")

    Args:
        answer: The synthesized answer to evaluate.
        ct_results: List of CT.gov study dicts actually retrieved.

    Returns:
        Score 0.0-1.0 where 1.0 = low hallucination risk.
    """
    if not answer.strip():
        return 1.0

    risk_points = 0.0
    max_risk = 0.0

    # Check 1: NCT IDs mentioned but not retrieved
    nct_pattern = re.compile(r"NCT\d{8}", re.IGNORECASE)
    mentioned_nct_ids = set(nct_pattern.findall(answer.upper()))
    retrieved_nct_ids = {
        str(r.get("nct_id", "")).upper()
        for r in ct_results
        if not r.get("error")
    }
    unverified_nct_ids = mentioned_nct_ids - retrieved_nct_ids
    if mentioned_nct_ids:
        max_risk += 0.4
        if unverified_nct_ids:
            risk_points += 0.4 * (len(unverified_nct_ids) / len(mentioned_nct_ids))

    # Check 2: Specific efficacy percentages or p-values
    stat_pattern = re.compile(
        r"\b(\d{1,3}\.?\d*%|\bp\s*[<=>]\s*0\.\d+|hazard ratio\s+\d|OR\s+\d\.\d)"
    )
    stats_found = stat_pattern.findall(answer)
    if stats_found:
        max_risk += 0.3
        # Check if any stat appears in the source data
        source_texts = " ".join(
            str(r.get("brief_summary", "")) + " " + str(r.get("_formatted_summary", ""))
            for r in ct_results
        )
        unverified_stats = [
            s for s in stats_found if s not in source_texts
        ]
        if unverified_stats:
            unverified_ratio = len(unverified_stats) / len(stats_found)
            risk_points += 0.3 * unverified_ratio

    # Check 3: Definitive future outcome claims
    future_pattern = re.compile(
        r"\b(will (be approved|show|demonstrate|prove|receive approval|gain clearance)"
        r"|is (certain|guaranteed) to|definitive(ly)? (proves|shows))\b",
        re.IGNORECASE,
    )
    if future_pattern.search(answer):
        max_risk += 0.3
        risk_points += 0.3

    # Check 4: Fabricated clinical trial details (very specific numbers without context)
    made_up_pattern = re.compile(
        r"\benrolled\s+exactly\s+\d+|\b\d{4,}\s+participants\s+from\s+\d+\s+countries\b",
        re.IGNORECASE,
    )
    if made_up_pattern.search(answer) and not ct_results:
        max_risk += 0.2
        risk_points += 0.2

    if max_risk == 0.0:
        return 1.0  # No risk indicators found

    # Inverse risk: 1.0 = safe, 0.0 = high risk
    risk_ratio = risk_points / max_risk
    return max(0.0, min(1.0, 1.0 - risk_ratio))


def aggregate_quality_scores(
    faithfulness: float,
    completeness: float,
    source_coverage: float,
    hallucination_risk: float,
) -> dict:
    """Aggregate individual quality scores into an overall assessment.

    Args:
        faithfulness: Score 0-1, threshold 0.85.
        completeness: Score 0-1, threshold 0.80.
        source_coverage: Score 0-1, threshold 0.75.
        hallucination_risk: Score 0-1 (1=low risk), threshold 0.80.

    Returns:
        Dict with individual scores, overall_score, passed bool, and grade.
    """
    overall = (
        faithfulness * WEIGHTS["faithfulness"]
        + completeness * WEIGHTS["completeness"]
        + hallucination_risk * WEIGHTS["hallucination_risk"]
        + source_coverage * WEIGHTS["source_coverage"]
    )

    passed = (
        faithfulness >= FAITHFULNESS_THRESHOLD
        and completeness >= COMPLETENESS_THRESHOLD
        and source_coverage >= SOURCE_COVERAGE_THRESHOLD
        and hallucination_risk >= HALLUCINATION_RISK_THRESHOLD
    )

    if overall >= 0.90:
        grade = "A"
    elif overall >= 0.75:
        grade = "B"
    elif overall >= 0.60:
        grade = "C"
    else:
        grade = "F"

    return {
        "faithfulness": round(faithfulness, 3),
        "completeness": round(completeness, 3),
        "source_coverage": round(source_coverage, 3),
        "hallucination_risk": round(hallucination_risk, 3),
        "overall_score": round(overall, 3),
        "passed": passed,
        "grade": grade,
        "thresholds": {
            "faithfulness": FAITHFULNESS_THRESHOLD,
            "completeness": COMPLETENESS_THRESHOLD,
            "source_coverage": SOURCE_COVERAGE_THRESHOLD,
            "hallucination_risk": HALLUCINATION_RISK_THRESHOLD,
        },
    }
