"""Query Router Agent — classifies intent and extracts structured parameters."""

from __future__ import annotations

import json
import os
import re
import time

from langchain_anthropic import ChatAnthropic

from agents.state import ClinicalTrialsAgentState
from prompts.templates import ROUTER_SYSTEM_PROMPT


def _extract_nct_ids(text: str) -> list[str]:
    """Extract all NCT IDs from text using regex."""
    pattern = re.compile(r"NCT\d{8}", re.IGNORECASE)
    matches = pattern.findall(text)
    return [m.upper() for m in matches]


_LLM = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def _clean_search_params(params: dict) -> dict:
    """Remove null/None values from search_params dict."""
    return {k: v for k, v in params.items() if v is not None and v != "null"}


def run_router(state: ClinicalTrialsAgentState) -> dict:
    """Classify the user query and extract routing parameters.

    Reads: user_query, conversation_history
    Writes: query_intent, requires_ct_api, requires_web_search,
            extracted_nct_ids, search_params, agent_trace
    """
    start_ts = time.time()
    query = state["user_query"]
    history = state.get("conversation_history", [])

    # Build context-aware message including recent history
    context_messages = []
    if history:
        recent = history[-4:]  # Last 2 turns
        context_parts = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]
            context_parts.append(f"{role}: {content}")
        context_block = "\n".join(context_parts)
        context_messages = [
            {
                "role": "user",
                "content": (
                    f"Recent conversation context:\n{context_block}\n\n"
                    f"Current question: {query}"
                ),
            }
        ]
    else:
        context_messages = [{"role": "user", "content": query}]

    # Always run regex extraction as a fallback / cross-check
    regex_nct_ids = _extract_nct_ids(query)

    try:
        response = _LLM.invoke(
            [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}] + context_messages
        )
        raw = response.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)

        parsed = json.loads(raw)

        query_intent = parsed.get("query_intent", "hybrid")
        requires_ct_api = parsed.get("requires_ct_api", True)
        requires_web_search = parsed.get("requires_web_search", False)

        # Merge LLM-extracted NCT IDs with regex results
        llm_nct_ids = [
            nid.upper()
            for nid in parsed.get("extracted_nct_ids", [])
            if re.match(r"NCT\d{8}", nid, re.IGNORECASE)
        ]
        all_nct_ids = list(dict.fromkeys(llm_nct_ids + regex_nct_ids))

        search_params = _clean_search_params(parsed.get("search_params", {}))

        # Sanity override: if we have NCT IDs and intent is web_research, upgrade to hybrid
        if all_nct_ids and query_intent == "web_research":
            query_intent = "hybrid"
            requires_ct_api = True

        # Sanity override: hybrid needs both
        if query_intent == "hybrid":
            requires_ct_api = True
            requires_web_search = True

    except (json.JSONDecodeError, KeyError, TypeError):
        # Graceful degradation: fall back to heuristic routing
        query_lower = query.lower()
        all_nct_ids = regex_nct_ids

        if regex_nct_ids and any(
            w in query_lower for w in ("fdaaa", "results", "compliant", "overdue", "submitted")
        ):
            query_intent = "fdaaa_check"
            requires_ct_api = True
            requires_web_search = False
        elif regex_nct_ids and any(w in query_lower for w in ("news", "everything", "results")):
            query_intent = "hybrid"
            requires_ct_api = True
            requires_web_search = True
        elif regex_nct_ids:
            query_intent = "nct_lookup"
            requires_ct_api = True
            requires_web_search = False
        elif any(w in query_lower for w in ("find", "list", "search", "which", "what trials")):
            query_intent = "search_trials"
            requires_ct_api = True
            requires_web_search = False
        elif any(w in query_lower for w in ("news", "results", "approved", "fda", "outcome")):
            query_intent = "web_research"
            requires_ct_api = False
            requires_web_search = True
        else:
            query_intent = "hybrid"
            requires_ct_api = True
            requires_web_search = True

        search_params = {}

    latency_ms = int((time.time() - start_ts) * 1000)

    new_trace_entry = {
        "agent": "router",
        "action": "classify_query",
        "result": query_intent,
        "nct_ids_found": all_nct_ids,
        "latency_ms": latency_ms,
    }

    return {
        "query_intent": query_intent,
        "requires_ct_api": requires_ct_api,
        "requires_web_search": requires_web_search,
        "extracted_nct_ids": all_nct_ids,
        "search_params": search_params,
        "agent_trace": state.get("agent_trace", []) + [new_trace_entry],
    }
