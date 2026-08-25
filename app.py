"""Streamlit Chat UI for the Clinical Trials Intelligence Agent.

Run with: python -m streamlit run app.py
"""

from __future__ import annotations

import os
import time
import uuid

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Bridge Streamlit Cloud secrets → env vars ─────────────────────────────────
# When deployed on Streamlit Cloud, keys live in st.secrets (not .env).
# Injecting them into os.environ means all downstream code using os.getenv()
# works identically whether running locally or on the cloud.
for _secret_key in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY", "MAX_QUERIES_PER_SESSION"):
    try:
        if _secret_key in st.secrets and not os.environ.get(_secret_key):
            os.environ[_secret_key] = str(st.secrets[_secret_key])
    except Exception:
        pass

from graph.pipeline import stream_agent  # noqa: E402

# ── Rate-limit config ─────────────────────────────────────────────────────────
_MAX_QUERIES = int(os.getenv("MAX_QUERIES_PER_SESSION", "20"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Trials Intelligence Agent · Utsav Sharma",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts & base ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Top identity bar ────────────────────────────────────── */
.identity-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    border-radius: 10px;
    margin-bottom: 6px;
}
.identity-bar .name-block { color: #ffffff; }
.identity-bar .name-block strong { font-size: 1.05em; letter-spacing: 0.3px; }
.identity-bar .name-block span { font-size: 0.82em; color: #a8d8ea; margin-left: 8px; }
.identity-bar .links a {
    display: inline-flex; align-items: center; gap: 5px;
    color: #a8d8ea; text-decoration: none; font-size: 0.85em;
    margin-left: 14px; padding: 4px 10px;
    border: 1px solid #a8d8ea44; border-radius: 20px;
    transition: all 0.2s;
}
.identity-bar .links a:hover { background: #a8d8ea22; color: #ffffff; }

/* ── Hero section ────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #1a4a5c 100%);
    border-radius: 12px;
    padding: 28px 32px 22px;
    margin-bottom: 20px;
    border: 1px solid #2c5364;
}
.hero h1 { color: #ffffff; font-size: 1.9em; font-weight: 700; margin: 0 0 6px; }
.hero p { color: #a8d8ea; font-size: 0.92em; margin: 0 0 16px; line-height: 1.6; }
.tech-badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 0.75em; font-weight: 600;
    margin-right: 6px; margin-bottom: 4px;
}
.badge-langgraph  { background: #6c3483; color: #e8daef; }
.badge-claude     { background: #1a3350; color: #aed6f1; }
.badge-ct         { background: #1e8449; color: #a9dfbf; }
.badge-tavily     { background: #784212; color: #fdebd0; }
.badge-langchain  { background: #1a3a4a; color: #85c1e9; }

/* ── Sample query pills ──────────────────────────────────── */
.query-pill-grid {
    display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 20px;
}

/* ── Metric cards ────────────────────────────────────────── */
.metric-card {
    background: #1a1a2e; border-radius: 10px;
    padding: 12px 14px; margin-bottom: 8px;
    border-left: 4px solid #ccc;
}
.metric-card.pass { border-left-color: #27ae60; }
.metric-card.warn { border-left-color: #f39c12; }
.metric-card.fail { border-left-color: #e74c3c; }
.metric-label { font-size: 0.78em; color: #a8a8b3; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 1.4em; font-weight: 700; margin: 2px 0; }
.metric-value.pass { color: #27ae60; }
.metric-value.warn { color: #f39c12; }
.metric-value.fail { color: #e74c3c; }
.metric-threshold { font-size: 0.72em; color: #666; }

/* ── Grade badge ─────────────────────────────────────────── */
.grade-badge {
    display: inline-block; width: 44px; height: 44px; line-height: 44px;
    text-align: center; border-radius: 50%; font-weight: 700; font-size: 1.3em;
}
.grade-A { background: #1e5631; color: #a9dfbf; }
.grade-B { background: #1a3a1a; color: #a9dfbf; border: 2px solid #27ae60; }
.grade-C { background: #7d6608; color: #fdebd0; }
.grade-F { background: #641e16; color: #fadbd8; }

/* ── Trace step ──────────────────────────────────────────── */
.trace-step {
    background: #12192a; border-radius: 8px;
    padding: 8px 12px; margin-bottom: 6px;
    border-left: 3px solid #2c5364; font-size: 0.82em;
}
.trace-agent { color: #a8d8ea; font-weight: 600; }
.trace-action { color: #85c1e9; font-family: monospace; }
.trace-latency { color: #666; float: right; }

/* ── Source chip ─────────────────────────────────────────── */
.source-chip {
    display: flex; align-items: center; gap: 8px;
    background: #12192a; border-radius: 8px;
    padding: 7px 10px; margin-bottom: 5px; font-size: 0.8em;
}
.source-chip a { color: #a8d8ea; text-decoration: none; flex: 1; }
.source-chip a:hover { color: #ffffff; text-decoration: underline; }
.source-type-ct  { color: #27ae60; font-size: 0.7em; font-weight: 600; }
.source-type-web { color: #2980b9; font-size: 0.7em; font-weight: 600; }

/* ── Trial info card ─────────────────────────────────────── */
.trial-cards-wrapper { margin: 0 0 14px; }
.trial-card {
    background: #12192a;
    border: 1px solid #2c5364;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.trial-card-header {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 6px;
}
.trial-nct-id {
    font-size: 0.95em; font-weight: 700;
    color: #a8d8ea; text-decoration: none; font-family: monospace;
}
.trial-nct-id:hover { color: #fff; text-decoration: underline; }
.trial-status-badge {
    font-size: 0.68em; font-weight: 700;
    padding: 2px 9px; border-radius: 12px; letter-spacing: 0.5px;
}
.status-recruiting    { background: #1e5631; color: #a9dfbf; }
.status-completed     { background: #1a3a5c; color: #aed6f1; }
.status-active        { background: #6e2f02; color: #fad7a0; }
.status-terminated    { background: #641e16; color: #fadbd8; }
.status-withdrawn     { background: #2c3e50; color: #95a5a6; }
.status-not-yet       { background: #0e3d38; color: #a2d9ce; }
.status-default       { background: #2c3e50; color: #bbb; }
.trial-title {
    font-size: 0.88em; font-weight: 600;
    color: #e0e0e0; margin-bottom: 10px; line-height: 1.4;
}
.trial-meta-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px 12px;
}
.trial-meta-item { display: flex; flex-direction: column; }
.trial-meta-label { font-size: 0.68em; color: #555; text-transform: uppercase; letter-spacing: 0.4px; }
.trial-meta-value { font-size: 0.82em; color: #ccc; font-weight: 500; margin-top: 1px; }
.trial-meta-value.has-results { color: #27ae60; }
.trial-meta-value.no-results  { color: #666; }

/* ── Footer ──────────────────────────────────────────────── */
.footer {
    margin-top: 40px; padding: 16px 0 4px;
    border-top: 1px solid #2c3e50; text-align: center;
    color: #555; font-size: 0.78em; line-height: 1.8;
}
.footer a { color: #a8d8ea; text-decoration: none; }
.footer a:hover { text-decoration: underline; }

/* ── Misc tweaks ─────────────────────────────────────────── */
.stChatMessage { border-radius: 12px !important; }
div[data-testid="stChatInput"] > div { border-radius: 24px !important; }

</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_QUERIES = [
    ("🔍", "What is the status of NCT04280705?"),
    ("⚖️", "Is NCT02993237 FDAAA compliant?"),
    ("🧪", "Find Phase 3 oncology trials started in 2023"),
    ("💊", "What trials is Novartis running for cardiovascular disease?"),
    ("📰", "What happened with the Lilly MONARCH 2 trial results?"),
]

_AGENT_LABELS = {
    "route_query":              "Classifying query",
    "retrieve_clinical_trials": "Fetching from ClinicalTrials.gov",
    "search_web":               "Searching web sources",
    "synthesize_answer":        "Synthesizing answer",
    "check_quality":            "Evaluating quality",
    "format_response":          "Formatting response",
    "handle_error":             "Handling error",
}

_AGENT_ICONS = {
    "route_query":              "🔍",
    "retrieve_clinical_trials": "🏥",
    "search_web":               "🌐",
    "synthesize_answer":        "🧠",
    "check_quality":            "✅",
    "format_response":          "📝",
    "handle_error":             "⚠️",
}

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "last_pubmed_results" not in st.session_state:
    st.session_state.last_pubmed_results = []
if "last_pubmed_triggered" not in st.session_state:
    st.session_state.last_pubmed_triggered = False


# ── UI helpers ────────────────────────────────────────────────────────────────

def _score_class(score: float, threshold: float) -> str:
    if score >= threshold:
        return "pass"
    elif score >= threshold - 0.10:
        return "warn"
    return "fail"


def _metric_card(label: str, score: float, threshold: float, desc: str) -> str:
    cls = _score_class(score, threshold)
    icon = "✓" if cls == "pass" else ("~" if cls == "warn" else "✗")
    bar_pct = int(score * 100)
    bar_colors = {"pass": "#27ae60", "warn": "#f39c12", "fail": "#e74c3c"}
    color = bar_colors[cls]
    return f"""
<div class="metric-card {cls}">
  <div class="metric-label">{label} &nbsp;<span style="font-size:0.9em;">{icon}</span></div>
  <div class="metric-value {cls}">{score:.0%}</div>
  <div style="background:#2a2a3e;border-radius:4px;height:5px;margin:4px 0;">
    <div style="background:{color};border-radius:4px;height:5px;width:{bar_pct}%;"></div>
  </div>
  <div class="metric-threshold">threshold: {threshold:.0%} · {desc}</div>
</div>"""


def _status_css_class(status: str) -> str:
    s = status.lower().replace("_", " ")
    if "recruiting" in s and "not yet" not in s:
        return "status-recruiting"
    if "completed" in s:
        return "status-completed"
    if "active" in s:
        return "status-active"
    if "terminat" in s:
        return "status-terminated"
    if "withdrawn" in s:
        return "status-withdrawn"
    if "not yet" in s:
        return "status-not-yet"
    return "status-default"


def _render_trial_cards(ct_results: list[dict]) -> str:
    """Return HTML for structured trial info cards (max 3)."""
    cards = []
    for study in ct_results[:3]:
        if study.get("error"):
            continue
        nct_id = study.get("nct_id", "")
        url = f"https://clinicaltrials.gov/study/{nct_id}"
        title = (study.get("brief_title") or study.get("official_title", ""))[:100]
        status = study.get("overall_status", "Unknown")
        status_cls = _status_css_class(status)
        phase = study.get("phase", "N/A")
        sponsor = (study.get("sponsor_name") or "N/A")[:40]
        enrollment = study.get("enrollment_count")
        enr_str = f"{int(enrollment):,}" if isinstance(enrollment, (int, float)) and enrollment else "N/A"
        start = (study.get("start_date") or "N/A")[:7]
        completion = (study.get("primary_completion_date") or "N/A")[:7]
        locations = study.get("locations_count", 0)
        has_results = study.get("has_results", False)
        res_cls = "has-results" if has_results else "no-results"
        res_label = "✓ Available" if has_results else "✗ None"

        cards.append(f"""
<div class="trial-card">
  <div class="trial-card-header">
    <a class="trial-nct-id" href="{url}" target="_blank">{nct_id}</a>
    <span class="trial-status-badge {status_cls}">{status.upper()}</span>
  </div>
  <div class="trial-title">{title}</div>
  <div class="trial-meta-grid">
    <div class="trial-meta-item">
      <span class="trial-meta-label">Phase</span>
      <span class="trial-meta-value">{phase}</span>
    </div>
    <div class="trial-meta-item">
      <span class="trial-meta-label">Sponsor</span>
      <span class="trial-meta-value">{sponsor}</span>
    </div>
    <div class="trial-meta-item">
      <span class="trial-meta-label">Enrollment</span>
      <span class="trial-meta-value">{enr_str}</span>
    </div>
    <div class="trial-meta-item">
      <span class="trial-meta-label">Start</span>
      <span class="trial-meta-value">{start}</span>
    </div>
    <div class="trial-meta-item">
      <span class="trial-meta-label">Primary Completion</span>
      <span class="trial-meta-value">{completion}</span>
    </div>
    <div class="trial-meta-item">
      <span class="trial-meta-label">Sites</span>
      <span class="trial-meta-value">{locations}</span>
    </div>
    <div class="trial-meta-item">
      <span class="trial-meta-label">Results</span>
      <span class="trial-meta-value {res_cls}">{res_label}</span>
    </div>
  </div>
</div>""")

    if not cards:
        return ""
    return '<div class="trial-cards-wrapper">' + "".join(cards) + "</div>"


def _build_export_md(query: str, answer: str, state: dict) -> str:
    """Build a markdown document for download."""
    citations = state.get("citations") or []
    quality = state.get("quality_scores") or state.get("metrics_summary") or {}

    lines = [
        "# Clinical Trials Query",
        "",
        f"**Question:** {query}",
        "",
        "---",
        "",
        "## Answer",
        "",
        answer,
        "",
    ]

    if citations:
        lines += ["## Sources", ""]
        seen: set[str] = set()
        for c in citations:
            url = c.get("url", "")
            if url and url not in seen:
                seen.add(url)
                lines.append(f"- [{c.get('title', url)}]({url})")
        lines.append("")

    grade = quality.get("grade")
    if grade:
        lines += [
            "## Quality Assessment",
            "",
            f"- **Grade:** {grade}",
            f"- **Overall Score:** {quality.get('overall_score', 0):.0%}",
            f"- **Faithfulness:** {quality.get('faithfulness', 0):.0%}",
            f"- **Completeness:** {quality.get('completeness', 0):.0%}",
            f"- **Source Coverage:** {quality.get('source_coverage', 0):.0%}",
            f"- **Hallucination Risk:** {quality.get('hallucination_risk', 0):.0%}",
            "",
        ]

    lines += [
        "---",
        "*Generated by [Clinical Trials Intelligence Agent](https://github.com/utsavsharma1990)*  ",
        "*Built by [Utsav Sharma](https://www.linkedin.com/in/utsav001/)*",
    ]
    return "\n".join(lines)


def _word_streamer(text: str):
    """Yield words one by one with a small delay for visual streaming effect."""
    if not text:
        return
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        if i % 10 == 9:
            time.sleep(0.012)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🔬 Agent Trace & Metrics")

        if st.session_state.last_metrics is None:
            st.info("Run a query to see quality scores and agent trace.")
            st.markdown("---")
            st.markdown(
                "**About this agent**\n\n"
                "Built by [Utsav Sharma](https://www.linkedin.com/in/utsav001/) "
                "· Data Engineering Manager at Norstella Citeline.\n\n"
                "6 LangGraph agents collaborate on every query — routing, "
                "retrieval, web search, synthesis, quality checking, and formatting."
            )
            return

        metrics = st.session_state.last_metrics
        grade = metrics.get("grade", "?")
        overall = metrics.get("overall_score", 0.0)
        revisions = metrics.get("revision_count", 0)

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(
                f'<div class="grade-badge grade-{grade}">{grade}</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(f"**Overall: {overall:.0%}**")
            st.caption(
                f"{'✓ Passed' if metrics.get('passed') else '⚠ Below threshold'}"
                + (f" · {revisions} revision(s)" if revisions else "")
            )

        st.markdown("---")

        thresholds = metrics.get("thresholds", {
            "faithfulness": 0.85, "completeness": 0.80,
            "hallucination_risk": 0.80, "source_coverage": 0.75,
        })
        metric_defs = [
            ("faithfulness",       "Faithfulness",       "Claims traceable to sources"),
            ("completeness",       "Completeness",       "All question parts answered"),
            ("hallucination_risk", "Hallucination Risk", "Unverified specifics"),
            ("source_coverage",   "Source Coverage",    "Correct source types present"),
        ]
        cards_html = ""
        for key, label, desc in metric_defs:
            score = metrics.get(key, 0.0)
            threshold = thresholds.get(key, 0.80)
            cards_html += _metric_card(label, score, threshold, desc)
        st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown("---")

        trace = st.session_state.last_trace
        with st.expander(f"🔀 Agent Trace ({len(trace)} steps)", expanded=False):
            if trace:
                for step in trace:
                    agent  = step.get("agent", "?")
                    action = step.get("action", "?")
                    result = str(step.get("result", ""))[:70]
                    ms     = step.get("latency_ms", 0)
                    st.markdown(
                        f'<div class="trace-step">'
                        f'<span class="trace-agent">{agent}</span> '
                        f'→ <span class="trace-action">{action}</span>'
                        f'<span class="trace-latency">{ms}ms</span><br>'
                        f'<span style="color:#888;font-size:0.9em;">{result}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No trace available.")

        st.markdown("---")

        citations = st.session_state.last_citations
        with st.expander(
            f"📚 Sources ({len({c.get('url') for c in citations if c.get('url')})})",
            expanded=False,
        ):
            if citations:
                seen: set[str] = set()
                for cite in citations:
                    url = cite.get("url", "")
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    title = cite.get("title", url)[:55]
                    stype = cite.get("source_type", "")
                    is_ct = stype == "ClinicalTrials.gov" or "clinicaltrials.gov" in url.lower()
                    is_pm = stype == "PubMed" or "pubmed.ncbi.nlm.nih.gov" in url.lower()
                    if is_ct:
                        icon, type_cls, type_label = "🔬", "source-type-ct", "ClinicalTrials.gov"
                    elif is_pm:
                        icon, type_cls, type_label = "📄", "source-type-pubmed", "PubMed"
                    else:
                        icon, type_cls, type_label = "🌐", "source-type-web", "Web"
                    st.markdown(
                        f'<div class="source-chip">'
                        f'{icon} <a href="{url}" target="_blank">{title}</a>'
                        f'<span class="{type_cls}">{type_label}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No sources retrieved.")

        st.markdown("---")

        # ── PubMed papers section ─────────────────────────────────────────────
        pubmed_triggered = st.session_state.last_pubmed_triggered
        pubmed_papers    = st.session_state.last_pubmed_results
        n_papers = len(pubmed_papers)

        if pubmed_triggered and n_papers > 0:
            with st.expander(f"📄 PubMed Papers ({n_papers} found)", expanded=False):
                for paper in pubmed_papers:
                    title  = (paper.get("title") or "")[:60]
                    title_disp = f"{title}…" if len(paper.get("title", "")) > 60 else title
                    journal   = paper.get("journal", "")
                    pub_date  = paper.get("pub_date", "")
                    pub_url   = paper.get("pubmed_url", "")
                    st.markdown(
                        f'<div class="source-chip">'
                        f'📄 <a href="{pub_url}" target="_blank">{title_disp}</a>'
                        f'<span class="source-type-pubmed">{journal} {pub_date}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        elif pubmed_triggered and n_papers == 0:
            st.caption("📄 PubMed searched — no papers found for this NCT ID")
        elif st.session_state.last_metrics is not None:
            # Only show "not triggered" message when there's been a query
            st.caption("📄 PubMed not triggered (query did not request outcomes)")

        st.markdown("---")

        # Query usage meter
        used = st.session_state.query_count
        remaining = max(0, _MAX_QUERIES - used)
        pct = int((used / _MAX_QUERIES) * 100) if _MAX_QUERIES else 0
        bar_color = "#27ae60" if pct < 70 else ("#f39c12" if pct < 90 else "#e74c3c")
        st.markdown(
            f"""<div style="margin-bottom:8px;">
              <div style="font-size:0.78em;color:#a8a8b3;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">
                Session Queries
              </div>
              <div style="font-size:1em;font-weight:600;color:#e0e0e0;">
                {used} / {_MAX_QUERIES} used &nbsp;
                <span style="font-size:0.8em;color:#888;">({remaining} remaining)</span>
              </div>
              <div style="background:#2a2a3e;border-radius:4px;height:5px;margin-top:5px;">
                <div style="background:{bar_color};border-radius:4px;height:5px;width:{pct}%;transition:width .3s;"></div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.caption(
            f"Session `{st.session_state.thread_id[:8]}…` · {len(st.session_state.messages)} messages"
        )
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_metrics = None
            st.session_state.last_trace = []
            st.session_state.last_citations = []
            st.session_state.last_pubmed_results = []
            st.session_state.last_pubmed_triggered = False
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.query_count = 0
            st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
def render_main():
    # Identity bar
    st.markdown("""
<div class="identity-bar">
  <div class="name-block">
    <strong>Utsav Sharma</strong>
    <span>Data Engineering Manager · Norstella Citeline</span>
  </div>
  <div class="links">
    <a href="https://www.linkedin.com/in/utsav001/" target="_blank">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136
        1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85
        3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0
        1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452z"/>
      </svg>
      LinkedIn
    </a>
    <a href="https://github.com/utsavsharma1990" target="_blank">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729
        1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931
        0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138
        3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479
        5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z"/>
      </svg>
      GitHub
    </a>
  </div>
</div>
""", unsafe_allow_html=True)

    # Hero section
    st.markdown("""
<div class="hero">
  <h1>🧬 Clinical Trials Intelligence Agent</h1>
  <p>
    A multi-agent AI system that answers any question about clinical trials — from registry lookups
    and FDAAA compliance checks to sponsor pipeline analysis and regulatory news.<br>
    Built to automate the manual trial lookups I do daily at Norstella Citeline.
  </p>
  <span class="tech-badge badge-langgraph">⛓ LangGraph</span>
  <span class="tech-badge badge-claude">✦ Claude</span>
  <span class="tech-badge badge-ct">🔬 ClinicalTrials.gov v2</span>
  <span class="tech-badge badge-tavily">🌐 Tavily Search</span>
  <span class="tech-badge badge-langchain">🦜 LangChain</span>
</div>
""", unsafe_allow_html=True)

    # Sample queries on first load
    if not st.session_state.messages:
        st.markdown("**Try a sample query:**")
        cols = st.columns(len(SAMPLE_QUERIES))
        for i, (icon, query) in enumerate(SAMPLE_QUERIES):
            with cols[i]:
                label = f"{icon} {query[:32]}{'…' if len(query) > 32 else ''}"
                if st.button(label, key=f"sample_{i}", use_container_width=True):
                    st.session_state["pending_query"] = query
                    st.rerun()
        st.markdown("")

    # Chat history replay
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # Replay trial cards for assistant messages that had CT data
            ct_results = msg.get("ct_results", [])
            if ct_results:
                st.markdown(_render_trial_cards(ct_results), unsafe_allow_html=True)
            st.markdown(msg["content"])

    # Pending sample query
    if "pending_query" in st.session_state:
        pending = st.session_state.pop("pending_query")
        _process_query(pending)

    # Chat input — disabled when session limit is reached
    if st.session_state.query_count >= _MAX_QUERIES:
        st.warning(
            f"You've reached the **{_MAX_QUERIES}-query session limit**. "
            "Refresh the page to start a new session.",
            icon="🚦",
        )
    elif prompt := st.chat_input("Ask anything about clinical trials — NCT IDs, FDAAA, sponsors, outcomes…"):
        _process_query(prompt)

    # Footer
    st.markdown("""
<div class="footer">
  Built by <a href="https://www.linkedin.com/in/utsav001/" target="_blank">Utsav Sharma</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/utsavsharma1990" target="_blank">GitHub</a>
  &nbsp;·&nbsp;
  Data sources: <a href="https://clinicaltrials.gov" target="_blank">ClinicalTrials.gov</a>,
  FDA, PubMed, STAT News
  &nbsp;·&nbsp;
  6 LangGraph agents · Quality-controlled with 4-metric scoring
</div>
""", unsafe_allow_html=True)


def _process_query(query: str):
    """Run a query through the pipeline with live progress and streaming output."""
    # Guard — should not be called when at limit, but be defensive
    if st.session_state.query_count >= _MAX_QUERIES:
        st.warning(f"Session limit of {_MAX_QUERIES} queries reached. Refresh to start a new session.")
        return
    st.session_state.query_count += 1

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-5:-1]  # last 2 turns only
    ]

    with st.chat_message("assistant"):
        final_state: dict | None = None
        error_msg: str | None = None

        # ── Live agent progress ────────────────────────────────────────────────
        with st.status("Starting pipeline…", expanded=True) as sw:
            try:
                for event in stream_agent(
                    query=query,
                    thread_id=st.session_state.thread_id,
                    history=history,
                ):
                    etype = event["type"]
                    if etype == "node_complete":
                        node = event["node"]
                        icon = _AGENT_ICONS.get(node, "▸")
                        label = _AGENT_LABELS.get(node, node)
                        sw.update(label=f"{icon} {label}…")
                        sw.write(f"{icon} {label}")
                    elif etype == "done":
                        final_state = event["state"]
                    elif etype == "error":
                        error_msg = event["error"]

                sw.update(label="✅ Complete", state="complete", expanded=False)
            except Exception as exc:
                error_msg = str(exc)
                sw.update(label="❌ Error", state="error", expanded=True)

        # ── Trial cards ────────────────────────────────────────────────────────
        ct_results = (final_state or {}).get("ct_api_results") or []
        valid_ct = [s for s in ct_results if not s.get("error")]
        if valid_ct:
            st.markdown(_render_trial_cards(valid_ct), unsafe_allow_html=True)

        # ── Answer ─────────────────────────────────────────────────────────────
        final_answer = ""
        if final_state:
            final_answer = (
                final_state.get("final_answer")
                or final_state.get("synthesized_answer")
                or ""
            )
        if not final_answer:
            final_answer = (
                f"⚠️ **Error**: {error_msg}\n\nPlease check your API keys and try again."
                if error_msg
                else "No answer was generated. Please try again."
            )

        # Stream the answer word-by-word for visual effect
        st.write_stream(_word_streamer(final_answer))

        # ── Export button ──────────────────────────────────────────────────────
        export_md = _build_export_md(query, final_answer, final_state or {})
        st.download_button(
            label="📥 Download as Markdown",
            data=export_md,
            file_name="clinical_trials_answer.md",
            mime="text/markdown",
            key=f"dl_{len(st.session_state.messages)}",
        )

    # Store to chat history (including ct_results for replay)
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "ct_results": valid_ct,
    })

    if final_state:
        st.session_state.last_metrics        = final_state.get("metrics_summary") or {}
        st.session_state.last_trace          = final_state.get("agent_trace") or []
        st.session_state.last_citations      = final_state.get("citations") or []
        st.session_state.last_pubmed_results = final_state.get("pubmed_results") or []
        st.session_state.last_pubmed_triggered = final_state.get("pubmed_triggered", False)

    st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────
render_sidebar()
render_main()
