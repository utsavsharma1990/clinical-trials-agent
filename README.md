# Clinical Trials Intelligence Agent

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-7C3AED)](https://github.com/langchain-ai/langgraph)
[![Claude](https://img.shields.io/badge/Claude-Haiku%20%7C%20Sonnet-D97706?logo=anthropic&logoColor=white)](https://anthropic.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![ClinicalTrials.gov](https://img.shields.io/badge/ClinicalTrials.gov-v2%20API-0066CC)](https://clinicaltrials.gov/data-api/api)
[![Tavily](https://img.shields.io/badge/Tavily-Advanced%20Search-F97316)](https://tavily.com)

A **production-quality multi-agent AI system** that answers any question about clinical trials — trial status, FDAAA compliance, sponsor pipelines, regulatory news, and published efficacy data — by orchestrating six specialized LLM agents across ClinicalTrials.gov, PubMed, and the open web, with a built-in quality-check retry loop that scores every answer before showing it to the user.

Built as a portfolio project demonstrating production-grade agentic system design: stateful graph orchestration, intent-aware token optimization, multi-source data fusion (registry + peer-reviewed literature + web), LLM-graded quality assurance, streaming UI, and safe API key handling.

---

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clinical-trials-agent.streamlit.app/)

> Deploy your own copy in 5 minutes: [jump to deployment](#deployment)

---

## What It Can Answer

| Category | Example Query |
|---|---|
| **Trial status** | `What is the current status of NCT04280705?` |
| **FDAAA compliance** | `Is NCT02993237 FDAAA compliant? When were results due?` |
| **Sponsor pipeline** | `What Phase 3 trials is Eli Lilly running for breast cancer?` |
| **Condition search** | `Find recruiting Phase 3 oncology trials started after 2022` |
| **Published outcomes** | `What are the published efficacy results for NCT02978625?` |
| **Regulatory news** | `What happened with the Pfizer BNT162b2 trial outcomes?` |
| **Deep dive** | `Tell me everything about NCT01668784 including its results` |
| **Follow-up** | `What about its FDAAA compliance?` *(uses session context)* |

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Agent 1 — Query Router                              │
│  Claude Haiku · Intent classification · NCT ID       │
│  extraction · routing flags · search params          │
└─────────────────────┬────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
┌─────────────────┐   ┌─────────────────────────────┐
│  Agent 2        │   │  Agent 3                    │
│  CT Retrieval   │──▶│  Web Intelligence           │
│  Claude Haiku   │   │  Claude Sonnet / Haiku      │
│  CT.gov v2 API  │   │  Tavily advanced search     │
│  FDAAA checker  │   │  PubMed / NCBI E-utilities  │
└────────┬────────┘   └─────────────┬───────────────┘
         │                          │
         └────────────┬─────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │  Agent 4 — Synthesis  │◀────────────────────┐
          │  Claude Haiku/Sonnet  │                      │ retry with
          │  Intent-aware model   │                      │ LLM feedback
          │  + token sizing       │                      │ (max 2×)
          └──────────┬────────────┘                      │
                     │                                    │
                     ▼                                    │
          ┌───────────────────────┐   fail + feedback     │
          │  Agent 5 — Quality    │──────────────────────┘
          │  Claude Haiku         │
          │  4-axis scoring       │
          │  Faithfulness ·       │
          │  Completeness ·       │
          │  Hallucination risk · │
          │  Source coverage      │
          └──────────┬────────────┘
                     │ pass
                     ▼
          ┌───────────────────────┐
          │  Agent 6 — Formatter  │
          │  Rule-based           │
          │  Citations · grade    │
          │  badge · markdown     │
          └──────────┬────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  Streamlit UI   │
            │  Trial cards    │
            │  Quality scores │
            │  Agent timeline │
            │  Export to MD   │
            └─────────────────┘
```

### Key design decisions

**Intent-aware model selection.** The router classifies each query into one of five intents (`nct_lookup`, `fdaaa_check`, `search_trials`, `web_research`, `hybrid`). Each intent maps to the cheapest model that can handle it — NCT lookups use Claude Haiku at 700 tokens; hybrid synthesis uses Claude Sonnet at 1800 tokens. This cuts per-query cost by ~60% versus using a single model.

**Quality-check retry loop.** No answer reaches the user without passing four scoring dimensions. Failed answers are revised up to twice with the exact quality feedback injected into the prompt — not a simple retry. This mirrors how human analysts cross-check their work.

**Stateful graph with isolation.** LangGraph's `MemorySaver` checkpointer tracks session state. Each query gets a unique `thread_id` (scoped to the session) so the graph retains conversation context without prior query data bleeding into new results.

**Per-query TTL cache.** ClinicalTrials.gov responses are cached for one hour using an in-process MD5-keyed dict. Repeat lookups on the same NCT ID within a session are instant and free.

---

## Features

- **Six-agent LangGraph pipeline** with conditional routing, quality-check retry loop, and error recovery
- **Intent-aware model selection** — Haiku for simple lookups, Sonnet for hybrid/research; per-intent token caps
- **FDAAA compliance checker** — applicable trial determination, 12-month results window, days-overdue calculation
- **PubMed integration** — NCBI E-utilities search for peer-reviewed abstracts linked to NCT IDs; trial acronym extraction (KEYNOTE-189, MONARCH-2, etc.) as fallback; PubMed results prioritised over web sources in synthesis
- **Three-source data fusion** — ClinicalTrials.gov registry data + PubMed published abstracts + Tavily web search, each cited separately
- **Streaming responses** — word-by-word output with live agent progress timeline
- **Structured trial cards** — clickable NCT ID links, status badges, 7-cell metadata grid
- **Four-dimensional quality scoring** — faithfulness, completeness, hallucination risk, source coverage (PubMed = premium score)
- **One-hour TTL cache** for ClinicalTrials.gov API calls
- **Export to Markdown** — download any answer with full citations
- **Session rate limiting** — configurable query cap with sidebar progress bar
- **Safe API key handling** — `.env` for local dev, `st.secrets` bridge for Streamlit Cloud; no keys in source

---

## Quick Start (Local)

### Prerequisites

- Python 3.10+
- [Anthropic API key](https://console.anthropic.com/) (Claude Haiku + Sonnet)
- [Tavily API key](https://tavily.com/) (free tier: 1,000 searches/month)

### Install

```bash
git clone https://github.com/utsavsharma1990/clinical-trials-agent.git
cd clinical-trials-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-api03-...your-key-here...
TAVILY_API_KEY=tvly-...your-key-here...
MAX_QUERIES_PER_SESSION=20
```

### Run

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Deployment

### Streamlit Community Cloud (recommended, free)

1. **Push to GitHub** (private repo is fine):

```bash
git add .
git commit -m "initial commit"
git remote add origin https://github.com/utsavsharma1990/clinical-trials-agent.git
git push -u origin main
```

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → connect repo → set main file to `app.py`

3. Click **Advanced settings → Secrets** and paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-api03-..."
TAVILY_API_KEY    = "tvly-..."
MAX_QUERIES_PER_SESSION = "20"
```

4. Click **Deploy** — you get a public `https://YOUR_APP.streamlit.app` URL.

> See [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) for the full template.

### Other platforms

| Platform | Notes |
|---|---|
| **Railway / Render** | Add `ANTHROPIC_API_KEY` + `TAVILY_API_KEY` as env vars; start command: `streamlit run app.py --server.port $PORT` |
| **Docker** | `docker build -t clinical-trials-agent . && docker run -p 8501:8501 --env-file .env clinical-trials-agent` |
| **Azure Container Apps** | Set env vars in the container spec; expose port 8501 |

---

## Agent Details

| Agent | Model | Role | Reads | Writes |
|---|---|---|---|---|
| **Query Router** | Haiku | Intent classification, NCT ID extraction, routing flags | `user_query`, `conversation_history` | `query_intent`, `extracted_nct_ids`, `search_params`, `requires_ct_api`, `requires_web_search` |
| **CT Retrieval** | — | ClinicalTrials.gov v2 fetch, FDAAA computation, TTL cache | `extracted_nct_ids`, `search_params`, `query_intent` | `ct_api_results`, `fdaaa_status_data`, `retrieval_sources` |
| **Web Intelligence** | Haiku/Sonnet | Tavily web search + PubMed abstract retrieval via NCBI E-utilities; triggered for outcome/results queries | `user_query`, `query_intent`, `ct_api_results` | `web_search_results`, `pubmed_results`, `pubmed_papers_found`, `pubmed_triggered` |
| **Synthesis** | Haiku or Sonnet | Generate sourced answer from CT.gov + PubMed + web; revision on quality feedback; peer-reviewed data prioritised | `ct_api_results`, `web_search_results`, `pubmed_results`, `quality_feedback` | `synthesized_answer`, `answer_confidence`, `citations` |
| **Quality Check** | Haiku | 4-axis LLM scoring; generates targeted revision feedback; PubMed sources count as premium for source coverage | `synthesized_answer`, `ct_api_results`, `user_query`, `citations` | `quality_scores`, `quality_passed`, `quality_feedback`, `revision_count` |
| **Formatter** | — | Inject citations (CT.gov / PubMed / Web labelled separately), quality badge, markdown structure | `synthesized_answer`, `citations`, `metrics_summary` | `final_answer` |

---

## Quality Control System

Every answer passes through four independent scoring dimensions before reaching the user. If any dimension fails its threshold, the quality agent generates **targeted feedback** and routes back to the synthesis agent (up to 2 retries).

| Metric | What It Measures | Threshold | Weight |
|---|---|---|---|
| **Faithfulness** | Proportion of factual claims verifiable in retrieved sources | ≥ 85% | 40% |
| **Completeness** | Whether all sub-questions implied by the query were addressed | ≥ 80% | 30% |
| **Hallucination Risk** | Inverse of: unverified NCT IDs, statistics not in sources, definitive future claims | ≥ 80% | 20% |
| **Source Coverage** | Correct source types present for the query intent; PubMed = premium score for outcome queries | ≥ 75% | 10% |

```
Grade A  (≥ 90%)  ✅  High confidence — safe to act on
Grade B  (≥ 75%)  ⚡  Good confidence — verify critical figures
Grade C  (≥ 60%)  ⚠️  Moderate — treat as preliminary
Grade F  (< 60%)  🔴  Low confidence — do not rely on without verification
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Agent orchestration | [LangGraph 0.2+](https://github.com/langchain-ai/langgraph) | Stateful graph with conditional routing, retry loops, and MemorySaver checkpointing |
| LLM provider | [Claude (Anthropic)](https://anthropic.com) — Haiku + Sonnet | Best-in-class instruction following; intent-matched model sizing |
| LLM framework | [LangChain 0.3+](https://github.com/langchain-ai/langchain) | `ChatAnthropic` abstraction, message formatting |
| Trial data | [ClinicalTrials.gov v2 REST API](https://clinicaltrials.gov/data-api/api) | Authoritative structured registry; no authentication required |
| Published literature | [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25499/) (PubMed) | Peer-reviewed abstracts linked to NCT IDs; free, no API key required |
| Web search | [Tavily](https://tavily.com) advanced mode | Deep pharma/regulatory domain coverage; returns content, not just links |
| UI | [Streamlit 1.35+](https://streamlit.io) | Rapid deployment; `st.status`, `st.write_stream`, `st.download_button` |
| XML parsing | [lxml 4.9+](https://lxml.de/) | Fast, standards-compliant PubMed efetch XML parsing |
| SSL compatibility | [truststore](https://github.com/sethmlarson/truststore) | Handles corporate CA injection transparently |

---

## Project Structure

```
clinical-trials-agent/
├── app.py                    # Streamlit UI — chat interface, trial cards, export
├── graph/
│   └── pipeline.py           # LangGraph StateGraph assembly, stream_agent(), run_agent()
├── agents/
│   ├── state.py              # ClinicalTrialsAgentState TypedDict, make_initial_state()
│   ├── router.py             # Agent 1 — intent classification
│   ├── retrieval_agent.py    # Agent 2 — ClinicalTrials.gov + FDAAA
│   ├── web_agent.py          # Agent 3 — Tavily web search
│   ├── synthesis_agent.py    # Agent 4 — answer generation
│   ├── quality_agent.py      # Agent 5 — 4-axis scoring + feedback
│   └── formatter.py          # Agent 6 — citation injection + markdown
├── tools/
│   ├── clinical_trials_api.py  # CT.gov v2 client, FDAAA logic, TTL cache
│   ├── pubmed_search.py        # NCBI E-utilities client — esearch + efetch, acronym fallback
│   ├── web_search.py           # Tavily advanced search — primary + regulatory news
│   └── metrics.py              # Faithfulness, completeness, hallucination, coverage scorers
├── prompts/
│   └── templates.py            # All system prompts and eval templates
├── .env.example                # API key template (no real keys)
├── .streamlit/
│   └── secrets.toml.example    # Streamlit Cloud secrets template
└── requirements.txt
```

---

## Limitations

- **Data freshness** — ClinicalTrials.gov data reflects the last sponsor update; some registries lag by weeks.
- **Results data** — the registry's `resultsSection` is sparse; PubMed integration covers published abstracts, but not all trials have indexed papers (depends on authors registering the NCT ID in paper metadata or using a recognisable trial acronym like KEYNOTE-189).
- **PubMed search precision** — NCT ID text search returns 0 results in NCBI's index; the system falls back to trial acronym extraction, then first-5-word keyword search. Generic CT.gov titles without a trial acronym yield broad, less-targeted results.
- **FDAAA edge cases** — trials with partial dates, phased designs, or pre-2007 grandfathering should be verified by a regulatory professional. This tool is informational, not a legal compliance determination.
- **Web search scope** — limited to Tavily's index; breaking news (< 24h) may not appear.
- **No EMA/WHO/ISRCTN data** — only ClinicalTrials.gov (US registry) is queried directly.
- **Rate limits** — CT.gov enforces rate limits; queries returning > 100 studies may be slow. NCBI E-utilities allows up to 3 requests/second without an API key.
- **Quality scores are probabilistic** — LLM-based scoring is not deterministic; treat scores as signals, not ground truth.
- **Session memory only** — conversation context is not persisted across browser sessions.

---

## Future Roadmap

### Near-term (next 3 months)

- [ ] **Multi-registry support** — add EMA EudraCT, WHO ICTRP, and ISRCTN as retrieval sources alongside CT.gov
- [ ] **PubMed full-text links** — surface DOI + open-access PDF links from efetch so users can read the full paper, not just the abstract
- [ ] **PDF report export** — generate a structured one-page PDF summary per trial using `reportlab` or `weasyprint`
- [ ] **Async pipeline execution** — run CT retrieval and web search concurrently with `asyncio` to cut latency by ~40%
- [ ] **Evaluation harness** — golden Q&A dataset of 50 clinical trial questions with ground-truth answers for offline quality regression testing

### Medium-term (3–6 months)

- [ ] **Trial comparison mode** — multi-trial analysis: compare two NCT IDs head-to-head on endpoints, enrollment, status
- [ ] **Email alerts** — subscribe to a trial or sponsor; get notified when status, results, or completion date changes
- [ ] **FastAPI REST endpoint** — expose `POST /query` so the agent can be called programmatically without the Streamlit UI
- [ ] **Persistent conversation history** — store sessions in SQLite so users can resume previous conversations
- [ ] **Structured data extraction** — extract primary endpoints, efficacy outcomes, and adverse events into a typed schema from results sections
- [ ] **Cost tracking dashboard** — per-query Anthropic token spend visible in the sidebar

### Long-term (6–12 months)

- [ ] **Domain-adapted retrieval** — fine-tune an embedding model on clinical trial abstracts for semantic search over a local vector store of recent trials
- [ ] **Sponsor intelligence module** — aggregate all trials by sponsor into a pipeline health score: phase distribution, success rate, average enrollment time
- [ ] **Regulatory timeline prediction** — given trial phase and indication, predict likely FDA review timeline using historical precedent
- [ ] **Integration with Citeline Trialtrove** — connect to proprietary clinical intelligence databases for licensed trial data not available on public registries
- [ ] **Multi-tenant deployment** — per-user API key management, role-based access (read-only vs. full), and audit logging for enterprise use

---

## Author

**Utsav Sharma** — Data Engineering Manager at Norstella Citeline, building pharma intelligence tools.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Utsav%20Sharma-0A66C2?logo=linkedin)](https://linkedin.com/in/utsav-sharma)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?logo=github)](https://github.com/utsavsharma1990)

---

## License

MIT — see [LICENSE](LICENSE) for details.
