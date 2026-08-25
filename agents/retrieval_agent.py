"""Clinical Trials Retrieval Agent — fetches data from ClinicalTrials.gov API."""

from __future__ import annotations

import time

from agents.state import ClinicalTrialsAgentState
from tools.clinical_trials_api import (
    compute_fdaaa_status,
    format_study_summary,
    get_study_by_nct_id,
    search_studies,
)


def run_retrieval_agent(state: ClinicalTrialsAgentState) -> dict:
    """Fetch clinical trial data from ClinicalTrials.gov.

    Reads: extracted_nct_ids, search_params, query_intent
    Writes: ct_api_results, retrieval_sources, agent_trace
    """
    start_ts = time.time()
    query_intent = state.get("query_intent", "")
    nct_ids = state.get("extracted_nct_ids", [])
    search_params = state.get("search_params", {})

    ct_api_results: list[dict] = []
    retrieval_sources: list[str] = list(state.get("retrieval_sources", []))
    trace_entries: list[dict] = []

    # ── Phase 1: Fetch specific NCT IDs ───────────────────────────────────────
    for nct_id in nct_ids:
        fetch_start = time.time()
        study = get_study_by_nct_id(nct_id)
        fetch_latency = int((time.time() - fetch_start) * 1000)

        if study.get("error"):
            trace_entries.append(
                {
                    "agent": "retrieval",
                    "action": f"fetch_nct:{nct_id}",
                    "result": f"error: {study['error']}",
                    "latency_ms": fetch_latency,
                }
            )
        else:
            trace_entries.append(
                {
                    "agent": "retrieval",
                    "action": f"fetch_nct:{nct_id}",
                    "result": f"status={study.get('overall_status', '?')}, "
                              f"phase={study.get('phase', '?')}",
                    "latency_ms": fetch_latency,
                }
            )
            ct_url = f"https://clinicaltrials.gov/study/{nct_id}"
            if ct_url not in retrieval_sources:
                retrieval_sources.append(ct_url)

        ct_api_results.append(study)

        # ── FDAAA enrichment ──────────────────────────────────────────────────
        if query_intent in ("fdaaa_check", "hybrid") and not study.get("error"):
            fdaaa_start = time.time()
            fdaaa_data = compute_fdaaa_status(study)
            fdaaa_latency = int((time.time() - fdaaa_start) * 1000)

            # Merge FDAAA data into the study dict
            study["fdaaa_status_data"] = fdaaa_data

            trace_entries.append(
                {
                    "agent": "retrieval",
                    "action": f"fdaaa_check:{nct_id}",
                    "result": fdaaa_data.get("fdaaa_status", "unknown"),
                    "latency_ms": fdaaa_latency,
                }
            )

    # ── Phase 2: Search by parameters (if no specific NCT IDs or intent is search) ──
    should_search = (
        search_params
        and (
            query_intent in ("search_trials", "hybrid")
            or (not nct_ids and query_intent != "fdaaa_check")
        )
    )

    if should_search:
        search_start = time.time()
        try:
            results = search_studies(
                condition=search_params.get("condition"),
                sponsor=search_params.get("sponsor"),
                status=search_params.get("status"),
                phase=search_params.get("phase"),
                start_date_from=search_params.get("start_date_from"),
                start_date_to=search_params.get("start_date_to"),
                max_results=10,
            )
            search_latency = int((time.time() - search_start) * 1000)

            non_error = [r for r in results if not r.get("error")]
            ct_api_results.extend(non_error)

            for r in non_error:
                nct_id = r.get("nct_id", "")
                if nct_id:
                    ct_url = f"https://clinicaltrials.gov/study/{nct_id}"
                    if ct_url not in retrieval_sources:
                        retrieval_sources.append(ct_url)

            trace_entries.append(
                {
                    "agent": "retrieval",
                    "action": "search_studies",
                    "result": f"found {len(non_error)} studies",
                    "params": search_params,
                    "latency_ms": search_latency,
                }
            )
        except Exception as exc:
            trace_entries.append(
                {
                    "agent": "retrieval",
                    "action": "search_studies",
                    "result": f"error: {exc}",
                    "latency_ms": int((time.time() - search_start) * 1000),
                }
            )
            ct_api_results.append({"error": f"Search failed: {type(exc).__name__}: {exc}"})

    # ── Build formatted context strings ───────────────────────────────────────
    for study in ct_api_results:
        if not study.get("_formatted_summary"):
            study["_formatted_summary"] = format_study_summary(study)

    total_latency = int((time.time() - start_ts) * 1000)
    trace_entries.append(
        {
            "agent": "retrieval",
            "action": "complete",
            "result": f"{len(ct_api_results)} studies total",
            "latency_ms": total_latency,
        }
    )

    return {
        "ct_api_results": ct_api_results,
        "retrieval_sources": retrieval_sources,
        "agent_trace": state.get("agent_trace", []) + trace_entries,
    }
