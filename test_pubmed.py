"""
PubMed integration smoke tests — 4 queries from the spec.

Run:  python test_pubmed.py
"""

from __future__ import annotations

import os
import sys

# ── Bootstrap env (mirrors app.py secrets bridge) ───────────────────────────
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

# ── Test 1: Direct PubMed tool calls ─────────────────────────────────────────
print("=" * 60)
print("TEST 1 — search_by_nct_id: NCT02978625 (KEYNOTE-189)")
print("=" * 60)
from tools.pubmed_search import (
    fetch_abstracts,
    format_pubmed_for_synthesis,
    search_by_nct_id,
    search_pubmed_for_trial,
)

pmids = search_by_nct_id("NCT02978625", max_results=5)
print(f"  PMIDs found: {pmids}")

if pmids:
    papers = fetch_abstracts(pmids[:2])
    print(f"  Abstracts fetched: {len(papers)}")
    for p in papers:
        print(f"    • {p.get('title', 'N/A')[:70]}")
        print(f"      Journal: {p.get('journal', 'N/A')}")
        print(f"      URL:     {p.get('pubmed_url', 'N/A')}")
else:
    print("  ⚠ No PMIDs — NCBI may be slow or NCT ID not indexed under [si]")


# ── Test 2: Orchestrator with trial title fallback ────────────────────────────
print()
print("=" * 60)
print("TEST 2 — search_pubmed_for_trial: NCT02978625")
print("=" * 60)
papers = search_pubmed_for_trial(
    nct_id="NCT02978625",
    trial_title="KEYNOTE-189: A Study of Pembrolizumab (MK-3475) in Combination With Platinum-based Doublet Chemotherapy",
    sponsor="Merck",
    max_results=5,
)
print(f"  Papers returned: {len(papers)}")
for p in papers:
    print(f"    • [{p.get('pub_date','?')}] {p.get('title','N/A')[:65]}")

formatted = format_pubmed_for_synthesis(papers, nct_id="NCT02978625")
print()
print("  Formatted block:")
print(formatted[:600])


# ── Test 3: NCT ID not in PubMed → graceful empty return ────────────────────
print()
print("=" * 60)
print("TEST 3 — no papers for fake NCT: NCT99999999")
print("=" * 60)
empty = search_pubmed_for_trial("NCT99999999")
print(f"  Papers: {len(empty)}  (expected 0)")
empty_fmt = format_pubmed_for_synthesis(empty, nct_id="NCT99999999")
print(f"  Formatted: {empty_fmt}")


# ── Test 4: Full pipeline via graph (nct_lookup + outcomes keyword) ───────────
print()
print("=" * 60)
print("TEST 4 — Full pipeline: NCT02978625 results query")
print("=" * 60)
try:
    from graph.pipeline import stream_agent

    query = "What are the published efficacy results for NCT02978625?"
    print(f"  Query: {query}")
    print()

    final_state = None
    pubmed_triggered = False
    papers_found = 0

    for event in stream_agent(query, thread_id="test-pubmed-001"):
        if event.get("type") == "agent_update":
            agent_name = event.get("agent", "")
            print(f"  [{agent_name}] {event.get('status', '')}")
        elif event.get("type") == "done":
            final_state = event.get("state", {})

    if final_state:
        pubmed_triggered = final_state.get("pubmed_triggered", False)
        papers_found = final_state.get("pubmed_papers_found", 0)
        citations = final_state.get("citations", [])
        pubmed_citations = [c for c in citations if c.get("source_type") == "PubMed"]

        print()
        print(f"  ✓ pubmed_triggered:    {pubmed_triggered}")
        print(f"  ✓ pubmed_papers_found: {papers_found}")
        print(f"  ✓ PubMed citations:    {len(pubmed_citations)}")
        if pubmed_citations:
            for c in pubmed_citations:
                print(f"    📄 {c['title'][:60]}")
        metrics = final_state.get("metrics_summary", {})
        print(f"  ✓ Quality grade:       {metrics.get('grade','?')} ({metrics.get('overall_score','?')})")
        answer = final_state.get("synthesized_answer", "")
        print()
        print("  Answer (first 400 chars):")
        print("  " + answer[:400].replace("\n", "\n  "))
    else:
        print("  ⚠ No final state returned")

except Exception as exc:
    print(f"  ✗ Pipeline test failed: {type(exc).__name__}: {exc}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("All tests complete.")
print("=" * 60)
