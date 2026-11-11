"""
Evaluation metrics for Step 6 experiments.

This script computes initial automatic metrics.
For final conference paper results, combine these with manual review for answer accuracy
and report-generation quality.
"""

from __future__ import annotations

from typing import Dict, Any, List
from .common import split_ids


def contains_expected_evidence(result: Dict[str, Any]) -> bool:
    expected_ids = set(split_ids(result.get("mapped_evidence_chunk_ids"))) or {str(result.get("expected_evidence_id", "")).strip()}
    expected_ids = {e for e in expected_ids if e and e.lower() not in {"nan", "none"}}
    if not expected_ids:
        return True
    answer = result.get("answer", {})
    cited = answer.get("supporting_evidence", []) if isinstance(answer, dict) else []
    cited_ids = {str(c.get("evidence_id")) for c in cited if isinstance(c, dict)}
    return bool(expected_ids & cited_ids)


def compute_metrics(results: List[Dict[str, Any]], verifier_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_key = {(v.get("qa_id"), v.get("model_variant")): v for v in verifier_rows}
    n = len(results)
    if n == 0:
        return {}

    citation_scores = []
    hallucinations = []
    conflict_scores = []
    hr_scores = []
    evidence_scores = []
    answer_scores = []
    effective_scores = []
    evidence_precision = []
    evidence_recall = []

    for r in results:
        v = by_key.get((r.get("qa_id"), r.get("model_variant")), {})
        if v.get("citation_correct") is not None:
            citation_scores.append(bool(v.get("citation_correct")))
        hallucinations.append(bool(v.get("hallucination_detected")))
        conflict_scores.append(bool(v.get("conflict_handled")))
        hr_scores.append(bool(v.get("human_review_correct")))
        effective_scores.append(bool(v.get("effective_date_correct")))
        evidence_scores.append(contains_expected_evidence(r))
        if v.get("evidence_precision") is not None:
            evidence_precision.append(float(v.get("evidence_precision")))
        if v.get("evidence_recall") is not None:
            evidence_recall.append(float(v.get("evidence_recall")))
        # Automatic answer accuracy proxy: supported answer, correct review flag,
        # conflict handling, effective date handling, and no hallucination.
        answer_scores.append(
            bool(v.get("supported"))
            and bool(v.get("human_review_correct"))
            and bool(v.get("conflict_handled"))
            and bool(v.get("effective_date_correct"))
            and not bool(v.get("hallucination_detected"))
        )

    return {
        "n": n,
        "answer_accuracy": sum(answer_scores) / len(answer_scores),
        # ID-level citation metric. It checks whether cited evidence IDs are
        # present in the retrieved evidence set. It is intentionally not named
        # semantic citation support because it does not judge whether the cited
        # passage fully supports every compliance conclusion.
        "retrieved_citation_match": sum(citation_scores) / len(citation_scores) if citation_scores else None,
        # Backward-compatible alias for older scripts and archived tables.
        "citation_accuracy": sum(citation_scores) / len(citation_scores) if citation_scores else None,
        "evidence_precision": sum(evidence_precision) / len(evidence_precision) if evidence_precision else None,
        "evidence_recall": sum(evidence_recall) / len(evidence_recall) if evidence_recall else None,
        "hallucination_rate": sum(hallucinations) / len(hallucinations),
        "conflict_detection_accuracy": sum(conflict_scores) / len(conflict_scores),
        "effective_date_accuracy": sum(effective_scores) / len(effective_scores),
        "human_review_flag_accuracy": sum(hr_scores) / len(hr_scores),
        "expected_evidence_match_rate": sum(evidence_scores) / len(evidence_scores)
    }
