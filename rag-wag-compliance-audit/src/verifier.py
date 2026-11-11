"""
Verifier for Hybrid RAG-WAG outputs.
"""

from __future__ import annotations

import re
from typing import Dict, Any
from .common import normalize_review_flag, split_ids


def verify_output(result: Dict[str, Any]) -> Dict[str, Any]:
    answer = result.get("answer", {})
    evidence = []
    evidence.extend(result.get("rag_evidence", []) or [])
    evidence.extend(result.get("wag_evidence", []) or [])
    evidence_ids = {str(e.get("evidence_id")) for e in evidence}

    cited = answer.get("supporting_evidence", []) if isinstance(answer, dict) else []
    cited_ids = {str(c.get("evidence_id")) for c in cited if isinstance(c, dict)}

    has_citation = len(cited_ids) > 0
    # This is an ID-level validity check: every cited evidence_id must come
    # from the retrieved evidence set. It does not prove that the cited text
    # semantically supports every conclusion in the answer.
    citation_correct = bool(cited_ids) and cited_ids.issubset(evidence_ids)

    conclusion = str(answer.get("compliance_conclusion", "") if isinstance(answer, dict) else answer)
    lower_answer = conclusion.lower()
    hallucination_markers = ["asc 999", "pcaob rule 0000", "fasb asu 2099", "nonexistent standard"]
    hallucination_detected = any(m in lower_answer for m in hallucination_markers)

    expected_conflict = str(result.get("conflict_type", "")).strip().lower()
    conflict_expected = expected_conflict not in {"", "none", "nan", "no_conflict"}
    conflict_text = str(answer.get("conflict_analysis", "") if isinstance(answer, dict) else "").lower()
    conflict_handled = True
    if conflict_expected:
        negated = any(t in conflict_text for t in ["no conflict", "no inconsistency", "not detected", "no post-generation review"])
        conflict_handled = (not negated) and any(t in conflict_text for t in ["conflict", "outdated", "manual review", "external", "internal", "authority", "effective_date_sensitive"])

    expected_hr = normalize_review_flag(result.get("requires_human_review") or result.get("expected_human_review_flag"))
    actual_hr_bool = bool(answer.get("human_review_required")) if isinstance(answer, dict) else False
    if expected_hr == "yes":
        human_review_correct = actual_hr_bool is True
    elif expected_hr == "no":
        human_review_correct = actual_hr_bool is False
    elif expected_hr == "conditional":
        human_review_correct = actual_hr_bool is True or "conditional" in conclusion.lower()
    else:
        human_review_correct = True

    expected_chunks = set(split_ids(result.get("mapped_evidence_chunk_ids"))) or {str(result.get("expected_evidence_id", "")).strip()}
    retrieved_ids = {str(e.get("evidence_id")) for e in evidence}
    relevant_retrieved = retrieved_ids & expected_chunks
    evidence_precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else None
    evidence_recall = len(relevant_retrieved) / len(expected_chunks) if expected_chunks else None

    effective_expected = str(result.get("effective_date_case", "")).strip().lower() == "yes"
    date_analysis = str(answer.get("effective_date_analysis", "") if isinstance(answer, dict) else "").lower()
    date_result = result.get("effective_date_result", {}) or {}
    effective_date_correct = True
    if effective_expected:
        effective_date_correct = bool(date_result.get("handled")) and any(t in date_analysis for t in ["effective", "date", "manual", "retrieved"])

    supported = has_citation and citation_correct
    if result.get("model_variant") == "llm_only":
        supported = False

    unsupported_standard_pattern = re.compile(r"\b(ASC\s+999|PCAOB\s+Rule\s+0000|FASB\s+ASU\s+2099)\b", re.I)
    hallucination_detected = hallucination_detected or bool(unsupported_standard_pattern.search(conclusion))

    verifier = {
        "qa_id": result.get("qa_id"),
        "model_variant": result.get("model_variant"),
        "supported": supported,
        "citation_correct": citation_correct if result.get("model_variant") != "llm_only" else None,
        "citation_metric_scope": "retrieved_evidence_id_match_not_semantic_support",
        "hallucination_detected": hallucination_detected,
        "conflict_handled": conflict_handled,
        "effective_date_correct": effective_date_correct,
        "human_review_correct": human_review_correct,
        "evidence_precision": evidence_precision,
        "evidence_recall": evidence_recall,
        "has_citation": has_citation,
        "verifier_reason": "Rule-based verifier result. citation_correct is an ID-level retrieved-citation match; semantic support requires manual or stronger verifier review."
    }
    return verifier
