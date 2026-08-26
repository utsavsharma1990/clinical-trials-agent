"""PubMed search tool using NCBI E-utilities API.

No authentication required. Adds published peer-reviewed abstracts to the
pipeline for NCT ID queries about results, efficacy, or safety.

NCBI E-utilities base: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET

import requests

_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
_TOOL = "clinical-trials-agent"
_EMAIL = "agent@clinical-trials-tool"  # NCBI strongly recommends identifying callers


def search_by_nct_id(nct_id: str, max_results: int = 5) -> list[str]:
    """Find PubMed paper PMIDs linked to an NCT ID.

    Strategy A — NCT ID secondary-identifier field [si]:
      Searches the field where authors register NCT IDs in paper metadata.
    Strategy B — plain text fallback if A returns nothing.

    Returns a deduplicated list of PMID strings (may be empty).
    Never raises an exception.
    """
    pmids: list[str] = []
    base_params = {
        "db": "pubmed",
        "retmode": "json",
        "retmax": max_results,
        "tool": _TOOL,
        "email": _EMAIL,
    }

    # ── Strategy A: secondary-identifier field ────────────────────────────────
    try:
        t0 = time.time()
        resp = requests.get(
            f"{_BASE_URL}esearch.fcgi",
            params={**base_params, "term": f"{nct_id}[si]"},
            timeout=15,
        )
        resp.raise_for_status()
        ids_a = resp.json().get("esearchresult", {}).get("idlist", [])
        elapsed = int((time.time() - t0) * 1000)
        print(f"[PubMed] esearch[si] for {nct_id}: {len(ids_a)} results ({elapsed}ms)")
        pmids.extend(ids_a)
    except requests.exceptions.Timeout:
        print(f"[PubMed] API timeout for {nct_id} [si] search")
    except requests.exceptions.ConnectionError:
        print(f"[PubMed] API unreachable for {nct_id} [si] search")
    except Exception as exc:
        print(f"[PubMed] Error in [si] search for {nct_id}: {exc}")

    # ── Strategy B: plain text fallback ──────────────────────────────────────
    if not pmids:
        try:
            t0 = time.time()
            resp = requests.get(
                f"{_BASE_URL}esearch.fcgi",
                params={**base_params, "term": nct_id},
                timeout=15,
            )
            resp.raise_for_status()
            ids_b = resp.json().get("esearchresult", {}).get("idlist", [])
            elapsed = int((time.time() - t0) * 1000)
            print(f"[PubMed] esearch[text] for {nct_id}: {len(ids_b)} results ({elapsed}ms)")
            pmids.extend(ids_b)
        except requests.exceptions.Timeout:
            print(f"[PubMed] API timeout for {nct_id} text search")
        except requests.exceptions.ConnectionError:
            print(f"[PubMed] API unreachable for {nct_id} text search")
        except Exception as exc:
            print(f"[PubMed] Error in text search for {nct_id}: {exc}")

    return list(dict.fromkeys(pmids))  # deduplicate preserving order


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Retrieve full abstract data for a list of PMIDs via efetch XML.

    Returns a list of dicts with keys:
      pmid, title, authors, journal, pub_date,
      abstract_text, doi, pubmed_url

    Never raises — skips unparseable articles and logs warnings.
    """
    if not pmids:
        return []

    try:
        t0 = time.time()
        resp = requests.get(
            f"{_BASE_URL}efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "abstract",
                "retmode": "xml",
                "tool": _TOOL,
                "email": _EMAIL,
            },
            timeout=15,
        )
        resp.raise_for_status()
        elapsed = int((time.time() - t0) * 1000)
        print(f"[PubMed] efetch for {len(pmids)} PMIDs: {elapsed}ms")
    except requests.exceptions.Timeout:
        print("PubMed API timeout")
        return []
    except requests.exceptions.ConnectionError:
        print("PubMed API unreachable")
        return []
    except Exception as exc:
        print(f"[PubMed] efetch error: {exc}")
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        print(f"PubMed XML parse error: {exc}")
        return []

    papers: list[dict] = []
    for article in root.findall(".//PubmedArticle"):
        try:
            citation = article.find("MedlineCitation")
            if citation is None:
                continue

            pmid_el = citation.find("PMID")
            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""

            art = citation.find("Article")
            if art is None:
                continue

            # Title — some have nested tags (<i>, <sub> etc.), use itertext
            title_el = art.find("ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""

            # Abstract — join multiple labelled sections
            abstract_parts: list[str] = []
            for ab_el in art.findall(".//AbstractText"):
                text = "".join(ab_el.itertext()).strip()
                if not text:
                    continue
                label = ab_el.get("Label", "")
                abstract_parts.append(f"{label}: {text}" if label else text)
            abstract_text = " ".join(abstract_parts) if abstract_parts else "Abstract not available"

            # Authors
            author_els = art.findall(".//Author")
            if author_els:
                first_ln = author_els[0].find("LastName")
                first_name = first_ln.text.strip() if first_ln is not None and first_ln.text else "Unknown"
                authors = f"{first_name} et al." if len(author_els) > 1 else first_name
            else:
                authors = "Unknown"

            # Journal title
            journal_el = art.find(".//Journal/Title")
            journal = journal_el.text.strip() if journal_el is not None and journal_el.text else ""

            # Publication date
            pub_year_el  = art.find(".//PubDate/Year")
            pub_month_el = art.find(".//PubDate/Month")
            pub_year  = pub_year_el.text.strip()  if pub_year_el  is not None and pub_year_el.text  else ""
            pub_month = pub_month_el.text.strip() if pub_month_el is not None and pub_month_el.text else ""
            pub_date = f"{pub_year} {pub_month}".strip()

            # DOI
            doi = ""
            pubmed_data = article.find("PubmedData")
            if pubmed_data is not None:
                for aid in pubmed_data.findall(".//ArticleId"):
                    if aid.get("IdType") == "doi" and aid.text:
                        doi = aid.text.strip()
                        break

            papers.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "authors": authors,
                    "journal": journal,
                    "pub_date": pub_date,
                    "abstract_text": abstract_text,
                    "doi": doi,
                    "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )
        except Exception as exc:
            print(f"[PubMed] Error parsing article: {exc}")
            continue

    return papers


def search_pubmed_for_trial(
    nct_id: str,
    trial_title: str | None = None,
    sponsor: str | None = None,
    max_results: int = 5,
) -> list[dict]:
    """Main entry point — full search orchestrator.

    Step 1: NCT ID search via search_by_nct_id
    Step 2: If results found → fetch_abstracts
    Step 3: If no results AND trial_title given → title-based fallback
    Step 4: Return papers sorted by pub_date descending

    All papers get a 'nct_id' key injected for downstream use.
    Never raises.
    """
    pmids = search_by_nct_id(nct_id, max_results=max_results)
    papers: list[dict] = []

    if pmids:
        papers = fetch_abstracts(pmids)
    elif trial_title:
        # Title-based fallback — use keyword search rather than exact phrase match.
        # Many CT.gov titles are long and won't match paper titles exactly.
        # Strategy: extract a trial acronym (e.g. KEYNOTE-189, MONARCH-2) if present;
        # otherwise use the first 4-5 words as free keywords.
        _ACRONYM_RE = re.compile(r'\b([A-Z]{2,}[-–]\d+[A-Z0-9]*)\b')
        acr_match = _ACRONYM_RE.search(trial_title)
        if acr_match:
            search_term = acr_match.group(1)  # e.g. "KEYNOTE-189"
        else:
            # Take first 5 meaningful words (skip "A", "An", "The" etc.)
            words = [w for w in trial_title.split() if len(w) > 3][:5]
            search_term = " ".join(words)

        try:
            t0 = time.time()
            resp = requests.get(
                f"{_BASE_URL}esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": search_term,
                    "retmode": "json",
                    "retmax": 5,
                    "tool": _TOOL,
                    "email": _EMAIL,
                },
                timeout=15,
            )
            resp.raise_for_status()
            fallback_pmids = resp.json().get("esearchresult", {}).get("idlist", [])
            elapsed = int((time.time() - t0) * 1000)
            print(
                f"[PubMed] title fallback for '{search_term}': "
                f"{len(fallback_pmids)} results ({elapsed}ms)"
            )
            if fallback_pmids:
                papers = fetch_abstracts(fallback_pmids)
        except requests.exceptions.Timeout:
            print(f"[PubMed] API timeout for title fallback: {trial_title}")
        except requests.exceptions.ConnectionError:
            print(f"[PubMed] API unreachable for title fallback")
        except Exception as exc:
            print(f"[PubMed] Error in title fallback: {exc}")

    # Inject nct_id into each paper dict for downstream use
    for paper in papers:
        paper["nct_id"] = nct_id

    # Sort by pub_date descending (year first, then month)
    def _sort_key(p: dict) -> str:
        return p.get("pub_date", "") or ""

    papers.sort(key=_sort_key, reverse=True)
    return papers


def format_pubmed_for_synthesis(papers: list[dict], nct_id: str = "") -> str:
    """Format PubMed results as a clean string for LLM synthesis context.

    If no papers found, returns a single-line 'not found' note.
    """
    # Infer nct_id from papers if not passed explicitly
    if not nct_id and papers:
        nct_id = papers[0].get("nct_id", "")

    if not papers:
        return f"PUBMED: No published papers found for {nct_id} in PubMed."

    lines: list[str] = [
        f"PUBMED PUBLISHED RESULTS ({len(papers)} paper{'s' if len(papers) != 1 else ''} found for {nct_id}):",
        "",
    ]
    for i, paper in enumerate(papers, 1):
        lines += [
            f"---",
            f"[PubMed Source {i}]",
            f"Title: {paper.get('title', 'N/A')}",
            f"Authors: {paper.get('authors', 'N/A')}",
            f"Journal: {paper.get('journal', 'N/A')} ({paper.get('pub_date', 'N/A')})",
            f"URL: {paper.get('pubmed_url', '')}",
            "",
            "Abstract:",
            paper.get("abstract_text", "Abstract not available"),
            "---",
            "",
        ]

    return "\n".join(lines)
