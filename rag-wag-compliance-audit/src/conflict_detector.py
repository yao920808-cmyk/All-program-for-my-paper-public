"""Conflict and effective-date checks for the RAG-WAG experiment."""

from __future__ import annotations

import re
from typing import Dict, Any, List

from .evidence_ranker import authority_score


def _doc_text(evidence: List[Dict[str, Any]]) -> str:
    parts = []
    for ev in evidence:
        md = ev.get("metadata", {}) or {}
        parts.append(str(ev.get("title", "")))
        parts.append(str(ev.get("text", "")))
        parts.append(str(md.get("effective_date", "")))
        parts.append(str(md.get("effective_date_note", "")))
    return "\n".join(parts).lower()


def detect_conflicts(qa_row: Dict[str, Any], rag_evidence: List[Dict[str, Any]], wag_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    conflicts = []

    q = str(qa_row.get("question", "")).lower()
    rag_text = _doc_text(rag_evidence)
    wag_text = _doc_text(wag_evidence)

    explicit_conflict_context = (
        any(t in q for t in ["conflict", "outdated", "old policy", "old internal", "external guidance", "external standard"])
        or ("internal" in q and "external" in q)
        or ("company policy" in q and any(t in q for t in ["fasb", "pcaob", "sec", "irs", "asu"]))
    )

    if rag_evidence and wag_evidence and explicit_conflict_context:
        best_rag_auth = max(authority_score(ev) for ev in rag_evidence)
        best_wag_auth = max(authority_score(ev) for ev in wag_evidence)
        if best_wag_auth > best_rag_auth:
            conflicts.append("external_source_more_authoritative")

    if any(t in q for t in ["invoice issuance", "invoice-only", "performance obligation"]) and "performance obligation" in wag_text:
        conflicts.append("invoice_only_policy_conflict")

    if "effective date" in q or "applies after" in q or "after the effective" in q:
        conflicts.append("effective_date_sensitive")

    conflicts = list(dict.fromkeys(conflicts))
    requires_review = bool(conflicts) and conflicts != ["none"]

    return {
        "conflicts": conflicts,
        "conflict_detected": requires_review,
        "requires_human_review": requires_review,
        "reason": "Detected from QA labels plus retrieved evidence authority/date signals." if conflicts else "No conflict signal detected."
    }


def check_effective_date(qa_row: Dict[str, Any], evidence: List[Dict[str, Any]], enable_date_check: bool = True) -> Dict[str, Any]:
    q = str(qa_row.get("question", "")).lower()
    needs_date = "effective date" in q or "applies after" in q or "after the effective" in q or "fiscal years" in q
    if not needs_date:
        return {"needed": False, "handled": True, "requires_manual_review": False, "reason": "No effective-date issue expected."}
    if not enable_date_check:
        return {"needed": True, "handled": False, "requires_manual_review": True, "reason": "Date check disabled."}
    text = _doc_text(evidence)
    has_date = bool(re.search(r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}|20\d{2}", text))
    if has_date:
        return {"needed": True, "handled": True, "requires_manual_review": False, "reason": "Retrieved evidence contains date signal."}
    return {"needed": True, "handled": True, "requires_manual_review": True, "reason": "Effective-date issue exists, but retrieved evidence lacks a precise date; manual review required."}
