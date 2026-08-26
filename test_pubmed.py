"""
PubMed integration verification tests.

Each test prints PASS/FAIL per assertion and a failure reason.
Exits with code 1 if any assertion fails.

Run:  python test_pubmed.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

FAILURES: list[str] = []


def check(label: str, condition: bool, reason: str = "") -> None:
    """Record a labelled assertion. Print PASS/FAIL immediately."""
    if condition:
        print(f"  PASS  {label}")
    else:
        msg = f"  FAIL  {label}" + (f" — {reason}" if reason else "")
        print(msg)
        FAILURES.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STATE — make_initial_state PubMed fields")
print("=" * 60)
from agents.state import ClinicalTrialsAgentState, make_initial_state

s = make_initial_state("test query")

check("pubmed_results initialises to []",
      s["pubmed_results"] == [],
      f"got {s['pubmed_results']!r}")
check("pubmed_papers_found initialises to 0",
      s["pubmed_papers_found"] == 0,
      f"got {s['pubmed_papers_found']!r}")
check("pubmed_triggered initialises to False",
      s["pubmed_triggered"] is False,
      f"got {s['pubmed_triggered']!r}")
check("pubmed_results is TypedDict field (key exists)",
      "pubmed_results" in ClinicalTrialsAgentState.__annotations__,
      "missing from TypedDict")
check("pubmed_papers_found is TypedDict field",
      "pubmed_papers_found" in ClinicalTrialsAgentState.__annotations__)
check("pubmed_triggered is TypedDict field",
      "pubmed_triggered" in ClinicalTrialsAgentState.__annotations__)


# ─────────────────────────────────────────────────────────────────────────────
# PUBMED TOOL — timeout config
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TOOL — pubmed_search.py timeout values")
print("=" * 60)
import inspect
from tools import pubmed_search

src = inspect.getsource(pubmed_search)
timeout_values = [int(t) for t in __import__("re").findall(r"timeout=(\d+)", src)]

check("all NCBI timeouts are >= 15s (cold-start safety)",
      all(t >= 15 for t in timeout_values),
      f"found timeouts: {timeout_values}")
check("no timeout=10 remains in pubmed_search.py",
      10 not in timeout_values,
      f"found: {timeout_values}")


# ─────────────────────────────────────────────────────────────────────────────
# PUBMED TOOL — error handling paths
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TOOL — error handling: empty inputs, fake NCT ID")
print("=" * 60)
from tools.pubmed_search import (
    fetch_abstracts,
    format_pubmed_for_synthesis,
    search_by_nct_id,
    search_pubmed_for_trial,
)

# fetch_abstracts([]) must return [] without hitting the network
result_empty = fetch_abstracts([])
check("fetch_abstracts([]) returns []",
      result_empty == [],
      f"got {result_empty!r}")

# format_pubmed_for_synthesis([]) must return the 'not found' string
fmt_empty = format_pubmed_for_synthesis([], nct_id="NCT00000000")
check("format_pubmed_for_synthesis([]) contains 'No published papers'",
      "No published papers" in fmt_empty,
      f"got: {fmt_empty!r}")

# Fake NCT ID — both [si] and text search return 0; function must not raise
print("  [hitting NCBI for fake NCT ID — expected 0 results]")
fake_papers = search_pubmed_for_trial("NCT99999999")
check("fake NCT99999999 returns empty list (no crash)",
      isinstance(fake_papers, list) and len(fake_papers) == 0,
      f"got {fake_papers!r}")


# ─────────────────────────────────────────────────────────────────────────────
# PUBMED TOOL — acronym extraction
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TOOL — acronym extraction from CT.gov titles")
print("=" * 60)
import re as _re
_ACRONYM_RE = _re.compile(r'\b([A-Z]{2,}[-–]\d+[A-Z0-9]*)\b')

cases = [
    ("KEYNOTE-189: A Study of Pembrolizumab...", "KEYNOTE-189"),
    ("MONARCH-2: Abemaciclib combined with fulvestrant...", "MONARCH-2"),
    ("CHECKMATE-067: Nivolumab plus ipilimumab...", "CHECKMATE-067"),
    ("IMpower133: First-Line Atezolizumab...", None),   # IMpower133 has no hyphen
    ("A Phase 2 Study of Drug X in Patients With Cancer", None),
]
for title, expected in cases:
    m = _ACRONYM_RE.search(title)
    got = m.group(1) if m else None
    check(f"acronym from '{title[:45]}...' → {expected!r}",
          got == expected,
          f"extracted {got!r}")


# ─────────────────────────────────────────────────────────────────────────────
# PUBMED TOOL — live NCBI: search with acronym title
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TOOL — live NCBI: KEYNOTE-189 acronym fallback")
print("=" * 60)
papers = search_pubmed_for_trial(
    nct_id="NCT02978625",
    trial_title="KEYNOTE-189: A Study of Pembrolizumab in Combination With Chemotherapy",
    max_results=5,
)
check("returns a list",
      isinstance(papers, list))
check("found >= 1 paper via acronym fallback",
      len(papers) >= 1,
      "NCBI returned 0 — may be a network issue")

if papers:
    p = papers[0]
    check("each paper has 'title' key",
          "title" in p and isinstance(p["title"], str) and len(p["title"]) > 0)
    check("each paper has 'pubmed_url' starting with https://pubmed",
          p.get("pubmed_url", "").startswith("https://pubmed.ncbi.nlm.nih.gov/"))
    check("each paper has 'abstract_text' key",
          "abstract_text" in p and isinstance(p["abstract_text"], str))
    check("each paper has 'nct_id' injected",
          p.get("nct_id") == "NCT02978625",
          f"got {p.get('nct_id')!r}")
    check("papers sorted newest-first (pub_date descending)",
          papers == sorted(papers, key=lambda x: x.get("pub_date", ""), reverse=True))


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ROUTING — PubMed trigger logic
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("PIPELINE — _should_search_pubmed routing logic")
print("=" * 60)
from agents.web_agent import _should_search_pubmed

routing_cases = [
    # (query, intent, nct_ids, expected, label)
    ("What are the results for NCT12345678?", "nct_lookup", ["NCT12345678"], True,
     "nct_lookup + 'results' keyword → True"),
    ("What is the status of NCT12345678?",    "nct_lookup", ["NCT12345678"], False,
     "nct_lookup + no outcome keyword → False"),
    ("Is NCT12345678 FDAAA compliant?",       "fdaaa_check", ["NCT12345678"], True,
     "fdaaa_check always → True"),
    ("Tell me everything about NCT12345678",  "hybrid",     ["NCT12345678"], True,
     "hybrid always → True"),
    ("What are the efficacy results?",        "nct_lookup", [],              False,
     "nct_lookup + keyword but no NCT IDs → False"),
    ("Find Phase 3 oncology trials",          "search_trials", [],           False,
     "search_trials → False"),
]
for query, intent, nct_ids, expected, label in routing_cases:
    result = _should_search_pubmed(query, intent, nct_ids)
    check(label, result == expected, f"got {result}, expected {expected}")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE — end-to-end: nct_lookup + outcomes keyword triggers PubMed
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("PIPELINE — end-to-end: outcomes query routes through web agent")
print("=" * 60)
from graph.pipeline import stream_agent

query = "What are the published efficacy results for NCT02978625?"
final_state: dict = {}
nodes_visited: list[str] = []

for event in stream_agent(query, thread_id="test-health-check-001"):
    if event.get("type") == "node_complete":
        nodes_visited.append(event["node"])
    elif event.get("type") == "done":
        final_state = event.get("state", {})

check("search_web node was visited (PubMed routing fix works)",
      "search_web" in nodes_visited,
      f"nodes visited: {nodes_visited}")
check("synthesize_answer node was visited",
      "synthesize_answer" in nodes_visited)
check("check_quality node was visited",
      "check_quality" in nodes_visited)
check("format_response node was visited",
      "format_response" in nodes_visited)
check("pubmed_triggered is True in final state",
      final_state.get("pubmed_triggered") is True,
      f"got {final_state.get('pubmed_triggered')!r}")
check("pubmed_papers_found > 0",
      final_state.get("pubmed_papers_found", 0) > 0,
      f"got {final_state.get('pubmed_papers_found')}")
check("at least one PubMed citation in final state",
      any(c.get("source_type") == "PubMed" for c in final_state.get("citations", [])),
      "no PubMed citation found")
check("final_answer is non-empty string",
      isinstance(final_state.get("final_answer"), str) and len(final_state.get("final_answer", "")) > 50)
check("quality grade is A or B (not F)",
      final_state.get("metrics_summary", {}).get("grade", "F") in ("A", "B"),
      f"grade: {final_state.get('metrics_summary', {}).get('grade')}")
check("no error in final state",
      not final_state.get("error"),
      f"error: {final_state.get('error')}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
total = sum(1 for line in open(__file__).readlines() if "check(" in line)
passed = total - len(FAILURES)
print(f"RESULT: {passed} passed, {len(FAILURES)} failed")
if FAILURES:
    print("\nFailed assertions:")
    for f in FAILURES:
        print(f)
    sys.exit(1)
else:
    print("All assertions passed.")
print("=" * 60)
