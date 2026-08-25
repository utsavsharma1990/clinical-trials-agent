"""Answer Synthesis Agent — generates the final answer from retrieved context."""

from __future__ import annotations

import json
import os
import re
import time

from langchain_anthropic import ChatAnthropic

from agents.state import ClinicalTrialsAgentState
from prompts.templates import SYNTHESIS_REVISION_PROMPT, SYNTHESIS_SYSTEM_PROMPT
from tools.clinical_trials_api import format_study_summary

# Per-intent config: model, output cap, context char limit, CT/web result caps
_INTENT_CFG: dict[str, dict] = {
    "nct_lookup":    {"model": "claude-haiku-4-5-20251001", "max_tokens": 700,  "ctx_chars": 2500, "max_ct": 3, "max_web": 0, "summary_chars": 200},
    "fdaaa_check":   {"model": "claude-haiku-4-5-20251001", "max_tokens": 600,  "ctx_chars": 2000, "max_ct": 3, "max_web": 0, "summary_chars": 150},
    "search_trials": {"model": "claude-haiku-4-5-20251001", "max_tokens": 1200, "ctx_chars": 4000, "max_ct": 5, "max_web": 2, "summary_chars": 200},
    "web_research":  {"model": "claude-sonnet-4-6",         "max_tokens": 1800, "ctx_chars": 5000, "max_ct": 3, "max_web": 4, "summary_chars": 300},
    "hybrid":        {"model": "claude-sonnet-4-6",         "max_tokens": 1800, "ctx_chars": 5000, "max_ct": 4, "max_web": 3, "summary_chars": 250},
}
_DEFAULT_CFG = {"model": "claude-sonnet-4-6", "max_tokens": 1500, "ctx_chars": 4000, "max_ct": 5, "max_web": 3, "summary_chars": 250}


_LLM_CACHE: dict[str, ChatAnthropic] = {}


def _get_llm(intent: str) -> ChatAnthropic:
    cfg = _INTENT_CFG.get(intent, _DEFAULT_CFG)
    key = f"{cfg['model']}:{cfg['max_tokens']}"
    if key not in _LLM_CACHE:
        _LLM_CACHE[key] = ChatAnthropic(
            model=cfg["model"],
            temperature=0.1,
            max_tokens=cfg["max_tokens"],
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    return _LLM_CACHE[key]


def _build_context_block(
    ct_results: list[dict],
    web_results: list[dict],
    intent: str = "",
    pubmed_results: list[dict] | None = None,
) -> str:
    """Build the context block, sized to the query intent to minimise tokens.

    Context order: CT.gov → PubMed (if present) → Web search results.
    PubMed is placed between registry facts and web news because it is
    the authoritative source for published efficacy/safety outcomes.
    """
    from tools.pubmed_search import format_pubmed_for_synthesis

    cfg = _INTENT_CFG.get(intent, _DEFAULT_CFG)
    max_ct       = cfg["max_ct"]
    max_web      = cfg["max_web"]
    summary_chars = cfg["summary_chars"]

    parts: list[str] = []

    if ct_results:
        parts.append("=== ClinicalTrials.gov Registry Data ===")
        for study in ct_results[:max_ct]:
            if study.get("_formatted_summary"):
                # Trim pre-formatted summaries too
                parts.append(study["_formatted_summary"][:summary_chars * 3])
            elif not study.get("error"):
                raw = format_study_summary(study)
                # Truncate the brief_summary line within the formatted string
                lines = []
                for line in raw.splitlines():
                    if line.startswith("Summary:") and len(line) > summary_chars + 9:
                        line = line[: summary_chars + 9] + "…"
                    lines.append(line)
                parts.append("\n".join(lines))
            fdaaa = study.get("fdaaa_status_data")
            if fdaaa:
                parts.append(
                    f"FDAAA Status for {fdaaa.get('nct_id', '')}:\n"
                    f"  - Applicable Trial: {fdaaa.get('is_applicable_trial', False)}\n"
                    f"  - Results Due: {fdaaa.get('results_due_date', 'N/A')}\n"
                    f"  - Has Results: {fdaaa.get('has_results', False)}\n"
                    f"  - Status: {fdaaa.get('fdaaa_status', 'Unknown')}\n"
                    f"  - Days Overdue: {fdaaa.get('days_overdue', 'N/A')}\n"
                    f"  - Reason: {fdaaa.get('reason', '')}"
                )
        parts.append("")

    # ── PubMed: peer-reviewed published data ─────────────────────────────────
    if pubmed_results:
        nct_id = ""
        if ct_results:
            nct_id = ct_results[0].get("nct_id", "")
        parts.append("=== PEER-REVIEWED PUBLISHED DATA ===")
        parts.append(format_pubmed_for_synthesis(pubmed_results, nct_id=nct_id))
        parts.append("")

    if max_web > 0 and web_results:
        parts.append("=== Web Search Results ===")
        for item in web_results[:max_web]:
            if item.get("error"):
                continue
            title = item.get("title", "No title")
            url = item.get("url", "")
            content = item.get("content", "")[:400]
            pub_date = item.get("published_date", "")
            parts.append(f"[{title}]({url}){' — ' + pub_date if pub_date else ''}")
            parts.append(content)
            parts.append("")

    return "\n".join(parts)


def _extract_confidence(answer_text: str) -> tuple[str, float]:
    """Extract confidence JSON from end of answer and return clean answer + confidence."""
    confidence = 0.75  # Default

    pattern = re.compile(r"```json\s*\{[^}]*\"confidence\"\s*:\s*([\d.]+)[^}]*\}\s*```", re.DOTALL)
    match = pattern.search(answer_text)
    if match:
        try:
            confidence = float(match.group(1))
            answer_text = answer_text[: match.start()].rstrip()
        except (ValueError, IndexError):
            pass

    # Also try inline JSON block
    inline_pattern = re.compile(r'\{"confidence":\s*([\d.]+)\}')
    inline_match = inline_pattern.search(answer_text)
    if inline_match and not match:
        try:
            confidence = float(inline_match.group(1))
            answer_text = answer_text[: inline_match.start()].rstrip()
        except (ValueError, IndexError):
            pass

    return answer_text, max(0.0, min(1.0, confidence))


def _build_citations(
    ct_results: list[dict],
    web_results: list[dict],
    retrieval_sources: list[str],
    pubmed_results: list[dict] | None = None,
) -> list[dict]:
    """Build the citations list from retrieved sources."""
    citations: list[dict] = []
    seen: set[str] = set()

    # ClinicalTrials.gov citations
    for study in ct_results:
        if study.get("error"):
            continue
        nct_id = study.get("nct_id", "")
        if not nct_id:
            continue
        url = f"https://clinicaltrials.gov/study/{nct_id}"
        if url not in seen:
            seen.add(url)
            title = study.get("brief_title") or study.get("official_title", nct_id)
            citations.append(
                {
                    "title": f"{nct_id}: {title[:80]}",
                    "url": url,
                    "source_type": "ClinicalTrials.gov",
                }
            )

    # PubMed citations — source_type "PubMed" keeps them distinct in the UI
    for paper in (pubmed_results or []):
        url = paper.get("pubmed_url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        title = paper.get("title", url[:60])
        journal = paper.get("journal", "")
        citations.append(
            {
                "title": f"{title[:80]} — {journal}" if journal else title[:80],
                "url": url,
                "source_type": "PubMed",
            }
        )

    # Web citations
    for item in web_results:
        if item.get("error"):
            continue
        url = item.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        citations.append(
            {
                "title": item.get("title", url[:60]),
                "url": url,
                "source_type": "Web Source",
            }
        )

    # Any remaining retrieval sources not already cited
    for url in retrieval_sources:
        if url not in seen:
            seen.add(url)
            citations.append(
                {
                    "title": url[:80],
                    "url": url,
                    "source_type": (
                        "ClinicalTrials.gov"
                        if "clinicaltrials.gov" in url
                        else "Web Source"
                    ),
                }
            )

    return citations


def run_synthesis_agent(state: ClinicalTrialsAgentState) -> dict:
    """Generate a synthesized answer from all retrieved context.

    On revision runs, incorporates quality feedback into the prompt.

    Reads: user_query, ct_api_results, web_search_results, pubmed_results,
           query_intent, quality_feedback, revision_count, retrieval_sources
    Writes: synthesized_answer, answer_confidence, citations, agent_trace
    """
    start_ts = time.time()
    user_query = state.get("user_query", "")
    ct_results = state.get("ct_api_results", [])
    web_results = state.get("web_search_results", [])
    pubmed_results = state.get("pubmed_results", [])
    query_intent = state.get("query_intent", "")
    quality_feedback = state.get("quality_feedback", "")
    revision_count = state.get("revision_count", 0)
    retrieval_sources = state.get("retrieval_sources", [])

    context_block = _build_context_block(
        ct_results, web_results, intent=query_intent, pubmed_results=pubmed_results
    )

    # Choose system prompt based on whether this is a revision
    if revision_count > 0 and quality_feedback:
        system_prompt = SYNTHESIS_REVISION_PROMPT.replace("{feedback}", quality_feedback)
    else:
        system_prompt = SYNTHESIS_SYSTEM_PROMPT

    # Handle empty context
    if not context_block.strip():
        context_note = (
            "\n\n[NOTE: No data was retrieved from ClinicalTrials.gov or web search. "
            "Please indicate clearly that you cannot answer this question due to "
            "insufficient retrieved data.]"
        )
    else:
        context_note = ""

    ctx_chars = _INTENT_CFG.get(query_intent, _DEFAULT_CFG)["ctx_chars"]
    user_message = (
        f"Query Intent: {query_intent}\n\n"
        f"User Question: {user_query}\n\n"
        f"Retrieved Context:\n{context_block[:ctx_chars]}{context_note}\n\n"
        f"Please answer the question using only the retrieved context above. "
        f"Cite your sources inline as [Source: URL]. "
        f"End your response with a confidence JSON block."
    )

    llm = _get_llm(query_intent)
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )
    raw_answer = response.content

    synthesized_answer, confidence = _extract_confidence(raw_answer)
    citations = _build_citations(ct_results, web_results, retrieval_sources, pubmed_results)

    latency_ms = int((time.time() - start_ts) * 1000)
    new_trace_entry = {
        "agent": "synthesis",
        "action": "generate_answer",
        "revision_count": revision_count,
        "confidence": confidence,
        "latency_ms": latency_ms,
    }

    return {
        "synthesized_answer": synthesized_answer,
        "answer_confidence": confidence,
        "citations": citations,
        "agent_trace": state.get("agent_trace", []) + [new_trace_entry],
    }
