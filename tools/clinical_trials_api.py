"""ClinicalTrials.gov v2 API wrappers.

Base URL: https://clinicaltrials.gov/api/v2/studies
No authentication required.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime, timedelta
from typing import Any

import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_TIMEOUT = 10

# ── Simple in-process TTL cache ───────────────────────────────────────────────
_STUDY_CACHE: dict[str, dict] = {}
_SEARCH_CACHE: dict[str, dict] = {}
_CACHE_TTL = 3600  # 1 hour


def _cache_get(store: dict, key: str):
    entry = store.get(key)
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL:
        return entry["data"]
    return None


def _cache_set(store: dict, key: str, data) -> None:
    store[key] = {"data": data, "ts": time.time()}


def _safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Traverse a nested dict safely."""
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is None:
            return default
    return current


def _parse_date(date_struct: dict | None) -> str | None:
    """Extract date string from ClinicalTrials.gov dateStruct."""
    if not date_struct:
        return None
    return date_struct.get("date")


def _extract_study_fields(raw: dict) -> dict:
    """Parse the ClinicalTrials.gov v2 response into a clean flat dict."""
    proto = raw.get("protocolSection", {})
    results_section = raw.get("resultsSection", {})

    id_mod = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design_mod = proto.get("designModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    cond_mod = proto.get("conditionsModule", {})
    desc_mod = proto.get("descriptionModule", {})
    oversight_mod = proto.get("oversightModule", {})
    contacts_mod = proto.get("contactsLocationsModule", {})

    phases: list[str] = design_mod.get("phases", [])
    phase_str = ", ".join(phases) if phases else "N/A"

    locations: list[dict] = contacts_mod.get("locations", [])

    results_first_submit: str | None = status_mod.get(
        "resultsFirstSubmitDate"
    ) or _safe_get(results_section, "moreInfoModule", "submittedQCDate")

    return {
        "nct_id": id_mod.get("nctId", ""),
        "official_title": id_mod.get("officialTitle", ""),
        "brief_title": id_mod.get("briefTitle", ""),
        "overall_status": status_mod.get("overallStatus", ""),
        "start_date": _parse_date(status_mod.get("startDateStruct")),
        "primary_completion_date": _parse_date(
            status_mod.get("primaryCompletionDateStruct")
        ),
        "completion_date": _parse_date(status_mod.get("completionDateStruct")),
        "phase": phase_str,
        "sponsor_name": _safe_get(sponsor_mod, "leadSponsor", "name", default=""),
        "conditions": cond_mod.get("conditions", []),
        "brief_summary": desc_mod.get("briefSummary", ""),
        "is_fda_regulated_drug": oversight_mod.get("isFdaRegulatedDrug", False),
        "is_fda_regulated_device": oversight_mod.get("isFdaRegulatedDevice", False),
        "results_first_submit_date": results_first_submit,
        "has_results": bool(results_section) and bool(results_first_submit),
        "enrollment_count": _safe_get(design_mod, "enrollmentInfo", "count", default=0),
        "study_type": design_mod.get("studyType", ""),
        "locations_count": len(locations),
    }


def get_study_by_nct_id(nct_id: str) -> dict:
    """Fetch a single study by NCT ID from ClinicalTrials.gov v2 API.

    Args:
        nct_id: The NCT identifier (e.g. "NCT04280705").

    Returns:
        Clean study dict with standardised fields, or error dict on failure.
    """
    cache_key = nct_id.upper()
    cached = _cache_get(_STUDY_CACHE, cache_key)
    if cached is not None:
        return cached

    url = f"{BASE_URL}/{nct_id}"
    try:
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 404:
            return {"nct_id": nct_id, "error": "Study not found"}
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        result = _extract_study_fields(resp.json())
        _cache_set(_STUDY_CACHE, cache_key, result)
        return result
    except requests.exceptions.Timeout:
        return {"nct_id": nct_id, "error": "Request timed out"}
    except requests.exceptions.RequestException as exc:
        return {"nct_id": nct_id, "error": f"API error: {exc}"}


def compute_fdaaa_status(study_dict: dict) -> dict:
    """Determine FDAAA 801 compliance status for a study.

    FDAAA 801 requires applicable clinical trials to submit results within
    12 months of primary completion date.

    Args:
        study_dict: Clean study dict from get_study_by_nct_id.

    Returns:
        Dict with FDAAA compliance details.
    """
    nct_id = study_dict.get("nct_id", "")

    if study_dict.get("error"):
        return {
            "nct_id": nct_id,
            "is_applicable_trial": False,
            "results_due_date": None,
            "has_results": False,
            "fdaaa_status": "Not Applicable",
            "days_overdue": None,
            "reason": "Study data unavailable",
        }

    is_fda_drug = study_dict.get("is_fda_regulated_drug", False)
    is_fda_device = study_dict.get("is_fda_regulated_device", False)
    study_type = study_dict.get("study_type", "")
    phase = study_dict.get("phase", "")
    has_results = study_dict.get("has_results", False)
    primary_completion_str = study_dict.get("primary_completion_date")
    results_submit_str = study_dict.get("results_first_submit_date")
    start_date_str = study_dict.get("start_date")

    # Check if study started before FDAAA enactment (September 27, 2007)
    if start_date_str:
        try:
            start_dt = datetime.strptime(start_date_str[:10], "%Y-%m-%d").date()
            if start_dt < date(2007, 9, 27):
                return {
                    "nct_id": nct_id,
                    "is_applicable_trial": False,
                    "results_due_date": None,
                    "has_results": has_results,
                    "fdaaa_status": "Not Applicable",
                    "days_overdue": None,
                    "reason": "Study initiated before FDAAA enactment (2007-09-27)",
                }
        except ValueError:
            pass

    # Determine if applicable trial (case-insensitive comparison for API values)
    is_interventional = "interventional" in study_type.lower()
    # Phase exclusions: Early Phase 1 only (Phase 1, 2, 3, 4 are all applicable)
    phase_upper = phase.upper()
    is_excluded_phase = (
        "EARLY_PHASE1" in phase_upper or "EARLY PHASE 1" in phase_upper
    ) and "PHASE2" not in phase_upper and "PHASE 2" not in phase_upper
    is_applicable = (
        (is_fda_drug or is_fda_device)
        and is_interventional
        and not is_excluded_phase
    )

    if not is_applicable:
        reasons = []
        if not (is_fda_drug or is_fda_device):
            reasons.append("not FDA-regulated drug or device")
        if not is_interventional:
            reasons.append(f"study type is {study_type}")
        if is_excluded_phase:
            reasons.append(f"phase is {phase}")
        return {
            "nct_id": nct_id,
            "is_applicable_trial": False,
            "results_due_date": None,
            "has_results": has_results,
            "fdaaa_status": "Not Applicable",
            "days_overdue": None,
            "reason": "; ".join(reasons) if reasons else "Not an applicable clinical trial",
        }

    # Compute results due date
    if not primary_completion_str:
        return {
            "nct_id": nct_id,
            "is_applicable_trial": True,
            "results_due_date": None,
            "has_results": has_results,
            "fdaaa_status": "Not Yet Due",
            "days_overdue": None,
            "reason": "Primary completion date not set",
        }

    try:
        pcd = datetime.strptime(primary_completion_str[:7], "%Y-%m").date()
        due_date = date(pcd.year + 1, pcd.month, 1)
    except ValueError:
        try:
            pcd = datetime.strptime(primary_completion_str[:10], "%Y-%m-%d").date()
            due_date = date(pcd.year + 1, pcd.month, 1)
        except ValueError:
            return {
                "nct_id": nct_id,
                "is_applicable_trial": True,
                "results_due_date": None,
                "has_results": has_results,
                "fdaaa_status": "Not Yet Due",
                "days_overdue": None,
                "reason": f"Could not parse primary completion date: {primary_completion_str}",
            }

    today = date.today()
    due_date_str = due_date.isoformat()

    if due_date > today:
        return {
            "nct_id": nct_id,
            "is_applicable_trial": True,
            "results_due_date": due_date_str,
            "has_results": has_results,
            "fdaaa_status": "Not Yet Due",
            "days_overdue": None,
            "reason": f"Results due {due_date_str}; today is {today.isoformat()}",
        }

    days_overdue = (today - due_date).days

    # Check if results were submitted on time
    if has_results and results_submit_str:
        try:
            submit_date = datetime.strptime(results_submit_str[:10], "%Y-%m-%d").date()
            if submit_date <= due_date:
                return {
                    "nct_id": nct_id,
                    "is_applicable_trial": True,
                    "results_due_date": due_date_str,
                    "has_results": True,
                    "fdaaa_status": "Compliant",
                    "days_overdue": None,
                    "reason": f"Results submitted {results_submit_str}, due {due_date_str}",
                }
            else:
                late_days = (submit_date - due_date).days
                return {
                    "nct_id": nct_id,
                    "is_applicable_trial": True,
                    "results_due_date": due_date_str,
                    "has_results": True,
                    "fdaaa_status": "Compliant (Late)",
                    "days_overdue": None,
                    "reason": f"Results submitted {late_days} days late on {results_submit_str}",
                }
        except ValueError:
            pass

    if has_results:
        return {
            "nct_id": nct_id,
            "is_applicable_trial": True,
            "results_due_date": due_date_str,
            "has_results": True,
            "fdaaa_status": "Compliant",
            "days_overdue": None,
            "reason": "Results submitted (submission date unverified)",
        }

    return {
        "nct_id": nct_id,
        "is_applicable_trial": True,
        "results_due_date": due_date_str,
        "has_results": False,
        "fdaaa_status": "Overdue",
        "days_overdue": days_overdue,
        "reason": f"Results due {due_date_str}; no results submitted; {days_overdue} days overdue",
    }


def search_studies(
    condition: str | None = None,
    sponsor: str | None = None,
    status: str | None = None,
    phase: str | None = None,
    start_date_from: str | None = None,
    start_date_to: str | None = None,
    max_results: int = 10,
) -> list[dict]:
    """Search ClinicalTrials.gov for studies matching criteria.

    Args:
        condition: Disease or condition to search for.
        sponsor: Sponsor name or term.
        status: Overall status filter (RECRUITING, COMPLETED, etc.).
        phase: Trial phase filter (PHASE3, etc.).
        start_date_from: Earliest start date (YYYY-MM-DD).
        start_date_to: Latest start date (YYYY-MM-DD).
        max_results: Maximum number of studies to return.

    Returns:
        List of clean study dicts.
    """
    params: dict[str, Any] = {"pageSize": min(max_results, 100), "format": "json"}

    if condition:
        params["query.cond"] = condition
    if sponsor:
        params["query.spons"] = sponsor
    if status:
        params["filter.overallStatus"] = status
    if phase:
        params["filter.advanced"] = f"AREA[Phase]{phase}"

    if start_date_from or start_date_to:
        date_filter = ""
        if start_date_from:
            date_filter += f"AREA[StartDate]RANGE[{start_date_from}, "
        else:
            date_filter += "AREA[StartDate]RANGE[MIN, "
        if start_date_to:
            date_filter += f"{start_date_to}]"
        else:
            date_filter += "MAX]"

        existing = params.get("filter.advanced", "")
        params["filter.advanced"] = (
            f"{existing} AND {date_filter}" if existing else date_filter
        )

    cache_key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    cached = _cache_get(_SEARCH_CACHE, cache_key)
    if cached is not None:
        return cached

    try:
        resp = requests.get(BASE_URL, params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.get(BASE_URL, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        studies = data.get("studies", [])
        results = [_extract_study_fields(s) for s in studies]
        _cache_set(_SEARCH_CACHE, cache_key, results)
        return results
    except requests.exceptions.Timeout:
        return [{"error": "Search request timed out"}]
    except requests.exceptions.RequestException as exc:
        return [{"error": f"Search API error: {exc}"}]


def get_studies_by_sponsor(
    sponsor_name: str,
    status: str | None = None,
) -> list[dict]:
    """Fetch studies by sponsor name.

    Args:
        sponsor_name: The sponsor or lead organization name.
        status: Optional status filter.

    Returns:
        List of clean study dicts.
    """
    return search_studies(sponsor=sponsor_name, status=status, max_results=20)


def format_study_summary(study_dict: dict) -> str:
    """Format a study dict into a clean human-readable context string for LLM prompts.

    Args:
        study_dict: Clean study dict from get_study_by_nct_id or search_studies.

    Returns:
        Formatted markdown-ish string suitable for LLM context.
    """
    if study_dict.get("error"):
        return f"[{study_dict.get('nct_id', 'Unknown')}]: Error — {study_dict['error']}"

    nct_id = study_dict.get("nct_id", "N/A")
    conditions = ", ".join(study_dict.get("conditions", [])) or "N/A"
    status = study_dict.get("overall_status", "N/A")
    phase = study_dict.get("phase", "N/A")
    sponsor = study_dict.get("sponsor_name", "N/A")
    title = study_dict.get("brief_title") or study_dict.get("official_title", "N/A")
    enrollment = study_dict.get("enrollment_count", "N/A")
    start = study_dict.get("start_date", "N/A")
    primary_completion = study_dict.get("primary_completion_date", "N/A")
    completion = study_dict.get("completion_date", "N/A")
    has_results = study_dict.get("has_results", False)
    study_type = study_dict.get("study_type", "N/A")
    is_fda_drug = study_dict.get("is_fda_regulated_drug", False)
    is_fda_device = study_dict.get("is_fda_regulated_device", False)
    locations = study_dict.get("locations_count", 0)
    brief_summary = study_dict.get("brief_summary", "")
    if brief_summary and len(brief_summary) > 400:
        brief_summary = brief_summary[:400] + "..."

    fda_reg = []
    if is_fda_drug:
        fda_reg.append("FDA-Regulated Drug")
    if is_fda_device:
        fda_reg.append("FDA-Regulated Device")
    fda_str = ", ".join(fda_reg) if fda_reg else "Not FDA-regulated"

    lines = [
        f"--- Study: {nct_id} ---",
        f"Title: {title}",
        f"Status: {status}",
        f"Type: {study_type} | Phase: {phase}",
        f"Sponsor: {sponsor}",
        f"Conditions: {conditions}",
        f"Enrollment: {enrollment} participants",
        f"Start Date: {start}",
        f"Primary Completion: {primary_completion}",
        f"Study Completion: {completion}",
        f"Has Results: {'Yes' if has_results else 'No'}",
        f"FDA Regulation: {fda_str}",
        f"Locations: {locations} sites",
        f"ClinicalTrials.gov URL: https://clinicaltrials.gov/study/{nct_id}",
    ]
    if brief_summary:
        lines.append(f"Summary: {brief_summary}")

    return "\n".join(lines)
