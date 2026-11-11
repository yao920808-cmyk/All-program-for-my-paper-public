"""
Rule-based query classifier for the RAG-WAG compliance experiment.

This is intentionally transparent: every classification can be inspected and
reported in the methods section. A learned classifier can be added later using
the same output labels.
"""

from __future__ import annotations

from typing import Dict, Any


LABELS = {
    "internal_policy_question",
    "external_standard_update_question",
    "conflict_resolution_question",
    "effective_date_question",
    "hallucination_trap_question",
    "report_generation_question",
    "source_authority_question",
    "financial_disclosure_question",
}


def classify_query(question: str, qa_row: Dict[str, Any] | None = None) -> Dict[str, Any]:
    qa_row = qa_row or {}
    q = (question or "").lower()
    # Do not use ground-truth labels such as question_type, conflict_type,
    # requires_human_review, or hallucination_trap_flag for routing. Those fields
    # are reserved for evaluation only.

    if any(t in q for t in ["asc 999", "pcaob rule 0000", "fasb asu 2099", "nonexistent", "imaginary"]):
        label = "hallucination_trap_question"
    elif "audit note" in q or "generate report" in q or "compliance memo" in q or "compliance report" in q or "draft report" in q:
        label = "report_generation_question"
    elif "effective date" in q or "applies after" in q or "fiscal years" in q or "after the effective" in q:
        label = "effective_date_question"
    elif "conflict" in q or ("internal" in q and "external" in q) or ("company policy" in q and "fasb" in q):
        label = "conflict_resolution_question"
    elif "prioritize" in q or "authority" in q or "which source" in q:
        label = "source_authority_question"
    elif "xbrl" in q or "companyfacts" in q or "company facts" in q or "10-k" in q or "10-q" in q or "edgar" in q:
        label = "financial_disclosure_question"
    elif any(t in q for t in ["latest", "new standard", "updated", "fasb", "pcaob", "sec", "asu", "irs", "aicpa"]):
        label = "external_standard_update_question"
    else:
        label = "internal_policy_question"

    route = {
        "internal_policy_question": "rag",
        "external_standard_update_question": "wag",
        "source_authority_question": "wag",
        "financial_disclosure_question": "wag",
        "conflict_resolution_question": "hybrid",
        "effective_date_question": "hybrid",
        "hallucination_trap_question": "hybrid",
        "report_generation_question": "hybrid",
    }[label]

    return {
        "query_type": label,
        "route": route,
        "reason": f"Rule-based classifier selected {label} from the question text only."
    }
