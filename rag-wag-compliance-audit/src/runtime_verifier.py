"""Runtime verifier for post-generation compliance governance.

This verifier is intentionally label-free: it only uses retrieved evidence,
system conflict/date checks, and the generated answer. It must not use ground
truth fields such as expected_answer or expected_human_review_flag.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


HALLUCINATION_PATTERN = re.compile(
    r"\b(ASC\s+999|PCAOB\s+Rule\s+0000|FASB\s+ASU\s+2099|nonexistent standard)\b",
    re.IGNORECASE,
)


def _answer_text(answer: Dict[str, Any]) -> str:
    parts = []
    for key in [
        "compliance_conclusion",
        "conflict_analysis",
        "effective_date_analysis",
        "limitations",
    ]:
        parts.append(str(answer.get(key, "")))
    for citation in answer.get("supporting_evidence", []) or []:
        if isinstance(citation, dict):
            parts.append(str(citation.get("reason", "")))
    return "\n".join(parts)


def _append_limitation(answer: Dict[str, Any], note: str) -> None:
    existing = str(answer.get("limitations", "")).strip()
    if note not in existing:
        answer["limitations"] = f"{existing} {note}".strip()


def apply_runtime_verifier(
    answer: Dict[str, Any],
    retrieved_evidence: List[Dict[str, Any]],
    conflict_result: Dict[str, Any],
    effective_date_result: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply post-generation checks without using evaluation labels."""
    if not isinstance(answer, dict):
        return answer, {"applied": False, "reason": "answer_not_dict"}

    evidence_ids = {
        str(ev.get("evidence_id", "")).strip()
        for ev in retrieved_evidence
        if isinstance(ev, dict) and ev.get("evidence_id")
    }
    citations = answer.get("supporting_evidence", [])
    cited_ids = {
        str(item.get("evidence_id", "")).strip()
        for item in citations
        if isinstance(item, dict) and item.get("evidence_id")
    } if isinstance(citations, list) else set()

    actions: list[str] = []
    forced_review = False

    if evidence_ids and not cited_ids:
        forced_review = True
        actions.append("missing_citation")
        _append_limitation(answer, "Runtime verifier: retrieved evidence was available but the answer did not cite it.")

    unsupported = sorted(cited_ids - evidence_ids)
    if unsupported:
        forced_review = True
        actions.append("unsupported_citation")
        _append_limitation(answer, "Runtime verifier: unsupported citation IDs remain after citation normalization.")

    if HALLUCINATION_PATTERN.search(_answer_text(answer)):
        forced_review = True
        actions.append("hallucination_marker")
        _append_limitation(answer, "Runtime verifier: potential hallucinated standard or rule marker detected.")

    if bool(conflict_result.get("requires_human_review")):
        conflict_text = str(answer.get("conflict_analysis", "")).lower()
        if not any(token in conflict_text for token in ["conflict", "manual review", "outdated", "external", "internal", "authority"]):
            actions.append("conflict_not_explained")
            _append_limitation(answer, "Runtime verifier: conflict signal was detected but not clearly explained.")
        forced_review = True
        actions.append("conflict_requires_review")

    if bool(effective_date_result.get("requires_manual_review")) or not bool(effective_date_result.get("handled", True)):
        forced_review = True
        actions.append("effective_date_requires_review")
        _append_limitation(answer, "Runtime verifier: effective-date evidence requires manual review.")

    if forced_review and answer.get("human_review_required") is not True:
        answer["human_review_required"] = True
        actions.append("human_review_forced_true")

    return answer, {
        "applied": True,
        "forced_human_review": forced_review,
        "actions": list(dict.fromkeys(actions)),
        "citation_count": len(cited_ids),
        "retrieved_evidence_count": len(evidence_ids),
    }
