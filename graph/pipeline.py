"""LangGraph pipeline assembly for the Clinical Trials Agent.

Compiles a stateful graph with 6 agents, conditional routing,
a quality-check retry loop (max 2 revisions), and MemorySaver checkpointing.
"""

from __future__ import annotations

import traceback
import uuid as _uuid
from typing import Generator

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.formatter import run_formatter
from agents.quality_agent import run_quality_agent
from agents.retrieval_agent import run_retrieval_agent
from agents.router import run_router
from agents.state import ClinicalTrialsAgentState, make_initial_state
from agents.synthesis_agent import run_synthesis_agent
from agents.web_agent import run_web_agent


# ── Node wrappers with error handling ─────────────────────────────────────────

def _node(fn):
    """Wrap an agent function to catch exceptions and route to handle_error."""
    def wrapper(state: ClinicalTrialsAgentState) -> dict:
        try:
            return fn(state)
        except Exception as exc:
            tb = traceback.format_exc()
            agent_name = fn.__name__.replace("run_", "")
            return {
                "error": f"[{agent_name}] {type(exc).__name__}: {exc}",
                "agent_trace": state.get("agent_trace", [])
                + [
                    {
                        "agent": agent_name,
                        "action": "error",
                        "result": str(exc),
                        "traceback": tb[-500:],
                        "latency_ms": 0,
                    }
                ],
            }
    wrapper.__name__ = fn.__name__
    return wrapper


def _handle_error(state: ClinicalTrialsAgentState) -> dict:
    """Produce a graceful error response when any node fails."""
    import re as _re
    error_msg = state.get("error", "An unknown error occurred.")
    # Strip URLs and tokens before showing to the user
    safe_msg = _re.sub(r"https?://\S+", "<URL redacted>", error_msg)
    safe_msg = _re.sub(r"sk-ant-[A-Za-z0-9_-]{10,}", "***", safe_msg)
    safe_msg = safe_msg[:300]
    final_answer = (
        f"⚠️ **I encountered an error processing your request.**\n\n"
        f"Error details: `{safe_msg}`\n\n"
        f"Please try again. If the issue persists, try rephrasing your question "
        f"or check that your API keys are configured correctly."
    )
    return {
        "final_answer": final_answer,
        "quality_passed": False,
        "metrics_summary": {
            "grade": "F",
            "overall_score": 0.0,
            "error": error_msg,
        },
        "agent_trace": state.get("agent_trace", []) + [{
            "agent": "handle_error",
            "action": "error",
            "result": safe_msg,
            "latency_ms": 0,
        }],
    }


# ── Routing functions ──────────────────────────────────────────────────────────

def _route_after_router(state: ClinicalTrialsAgentState) -> str:
    if state.get("error"):
        return "handle_error"
    if state.get("requires_ct_api", False):
        return "retrieve_clinical_trials"
    if state.get("requires_web_search", False):
        return "search_web"
    return "synthesize_answer"


def _route_after_retrieval(state: ClinicalTrialsAgentState) -> str:
    if state.get("error"):
        return "handle_error"
    if state.get("requires_web_search", False):
        return "search_web"
    # Also invoke the web agent when PubMed should be queried — e.g. an nct_lookup
    # that asks about published outcomes.  The web agent handles both Tavily and PubMed.
    from agents.web_agent import _should_search_pubmed
    if _should_search_pubmed(
        state.get("user_query", ""),
        state.get("query_intent", ""),
        state.get("extracted_nct_ids", []),
    ):
        return "search_web"
    return "synthesize_answer"


def _route_after_web(state: ClinicalTrialsAgentState) -> str:
    if state.get("error"):
        return "handle_error"
    return "synthesize_answer"


def _route_after_synthesis(state: ClinicalTrialsAgentState) -> str:
    if state.get("error"):
        return "handle_error"
    return "check_quality"


def _route_after_quality(state: ClinicalTrialsAgentState) -> str:
    if state.get("error"):
        return "handle_error"
    if state.get("quality_passed", False):
        return "format_response"
    revision_count = state.get("revision_count", 0)
    if revision_count < 2:
        return "synthesize_answer"
    # Max revisions reached — pass through to formatter anyway
    return "format_response"


# ── Build the graph ────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(ClinicalTrialsAgentState)

    # Register nodes
    graph.add_node("route_query", _node(run_router))
    graph.add_node("retrieve_clinical_trials", _node(run_retrieval_agent))
    graph.add_node("search_web", _node(run_web_agent))
    graph.add_node("synthesize_answer", _node(run_synthesis_agent))
    graph.add_node("check_quality", _node(run_quality_agent))
    graph.add_node("format_response", _node(run_formatter))
    graph.add_node("handle_error", _handle_error)

    # Entry point
    graph.add_edge(START, "route_query")

    # After router: conditional dispatch to retrieval, web, or direct synthesis
    graph.add_conditional_edges(
        "route_query",
        _route_after_router,
        {
            "retrieve_clinical_trials": "retrieve_clinical_trials",
            "search_web": "search_web",
            "synthesize_answer": "synthesize_answer",
            "handle_error": "handle_error",
        },
    )

    # After retrieval: optionally go to web, then synthesis
    graph.add_conditional_edges(
        "retrieve_clinical_trials",
        _route_after_retrieval,
        {
            "search_web": "search_web",
            "synthesize_answer": "synthesize_answer",
            "handle_error": "handle_error",
        },
    )

    # After web search: always go to synthesis
    graph.add_conditional_edges(
        "search_web",
        _route_after_web,
        {
            "synthesize_answer": "synthesize_answer",
            "handle_error": "handle_error",
        },
    )

    # After synthesis: always go to quality check
    graph.add_conditional_edges(
        "synthesize_answer",
        _route_after_synthesis,
        {
            "check_quality": "check_quality",
            "handle_error": "handle_error",
        },
    )

    # ── Quality check retry loop ───────────────────────────────────────────────
    # Pass → formatter; Fail + retries remaining → back to synthesis; Max retries → formatter
    graph.add_conditional_edges(
        "check_quality",
        _route_after_quality,
        {
            "format_response": "format_response",
            "synthesize_answer": "synthesize_answer",
            "handle_error": "handle_error",
        },
    )

    # Terminal edges
    graph.add_edge("format_response", END)
    graph.add_edge("handle_error", END)

    return graph


# ── Compile with MemorySaver for conversation continuity ─────────────────────

_checkpointer = MemorySaver()
_graph = _build_graph()
clinical_trials_agent = _graph.compile(checkpointer=_checkpointer)


# ── Public run function ───────────────────────────────────────────────────────

def stream_agent(
    query: str,
    thread_id: str,
    history: list[dict] | None = None,
) -> Generator[dict, None, None]:
    """Stream pipeline events as the graph executes.

    Yields dicts with 'type' key:
      {"type": "node_complete", "node": str, "data": dict}  — after each node
      {"type": "error",         "error": str}               — on exception
      {"type": "done",          "state": dict}              — final state
    """
    initial_state = make_initial_state(query=query, history=history)
    # Use a per-query ID so each invocation starts from a clean state.
    # MemorySaver resumes from the last checkpoint when thread_id is reused,
    # which causes ct_api_results/web_search_results from prior queries to bleed in.
    per_query_id = f"{thread_id}:{_uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": per_query_id}}

    try:
        for event in clinical_trials_agent.stream(
            initial_state, config=config, stream_mode="updates"
        ):
            for node_name, node_data in event.items():
                yield {"type": "node_complete", "node": node_name, "data": node_data or {}}
        snapshot = clinical_trials_agent.get_state(config)
        yield {"type": "done", "state": dict(snapshot.values)}
    except Exception as exc:
        yield {"type": "error", "error": str(exc)}
        yield {"type": "done", "state": {}}


def run_agent(
    query: str,
    thread_id: str,
    history: list[dict] | None = None,
) -> ClinicalTrialsAgentState:
    """Run the clinical trials agent pipeline for a single query.

    Uses a stable thread_id per session so the LangGraph MemorySaver can
    checkpoint between invocations. Each call passes the full refreshed state
    so prior processing fields do not bleed across queries.

    Args:
        query: The user's natural language question.
        thread_id: Stable session identifier (UUID) for LangGraph memory.
        history: Conversation history as list of {role, content} dicts.

    Returns:
        The final ClinicalTrialsAgentState after the pipeline completes.
    """
    initial_state = make_initial_state(query=query, history=history)
    per_query_id = f"{thread_id}:{_uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": per_query_id}}
    result = clinical_trials_agent.invoke(initial_state, config=config)
    return result
