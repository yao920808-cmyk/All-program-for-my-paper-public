"""Deterministic answer generator for reproducible pipeline experiments.

This generator is not a substitute for a real LLM in final paper claims. It is
designed to exercise the complete five-model architecture, verifier, and metrics
without hiding any uncertainty.
"""

from __future__ import annotations

from typing import Dict, Any, List

def _cite(evidence: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
    cited = []
    for ev in evidence[:n]:
        cited.append(
            {
                "evidence_id": ev.get("evidence_id"),
                "source_type": ev.get("source_type"),
                "title": ev.get("title"),
                "ranking": ev.get("ranking", {}),
                "reason": "Selected by retrieval and evidence ranking.",
            }
        )
    return cited


def generate_answer(
    qa_row: Dict[str, Any],
    model_variant: str,
    query_classification: Dict[str, Any],
    rag_evidence: List[Dict[str, Any]],
    wag_evidence: List[Dict[str, Any]],
    ranked_evidence: List[Dict[str, Any]],
    conflict_result: Dict[str, Any],
    date_result: Dict[str, Any],
    verifier_enabled: bool = False,
) -> Dict[str, Any]:
    question = qa_row.get("question", "")
    lower = question.lower()
    trap = any(t in lower for t in ["asc 999", "pcaob rule 0000", "fasb asu 2099", "nonexistent", "imaginary"])
    external_question = query_classification.get("route") == "wag"
    internal_question = query_classification.get("route") == "rag"

    if model_variant == "llm_only":
        # This is intentionally evidence-poor to represent the baseline.
        if trap:
            conclusion = "ASC 999 appears to require additional review, but this answer is not evidence-grounded."
            hallucination_risk = "high"
        else:
            conclusion = "A compliance conclusion is provided from general model knowledge, but no retrieved evidence is available."
            hallucination_risk = "medium"
        return {
            "compliance_conclusion": conclusion,
            "supporting_evidence": [],
            "conflict_analysis": "No retrieved internal or external evidence was available to validate conflicts.",
            "effective_date_analysis": "No retrieved evidence was available to validate effective dates.",
            "human_review_required": trap or conflict_result["conflict_detected"] or date_result["requires_manual_review"],
            "query_type": query_classification["query_type"],
            "hallucination_risk": hallucination_risk,
            "limitations": "LLM-only baseline: no RAG, no WAG, no evidence verification.",
        }

    if model_variant == "static_rag":
        cited = _cite(rag_evidence, 2)
        if not cited:
            conclusion = "Insufficient internal policy evidence was retrieved. Manual review is required."
            hr = True
        elif external_question or "latest" in lower:
            conclusion = "Static RAG retrieved internal policy evidence only. It may be outdated for external standard-change questions."
            hr = True
        elif conflict_result["conflict_detected"]:
            conclusion = "Internal policy evidence was retrieved, but external conflict cannot be resolved without WAG evidence."
            hr = True
        else:
            conclusion = "Use the cited internal policy evidence as the basis for the compliance response."
            hr = False
        return {
            "compliance_conclusion": conclusion,
            "supporting_evidence": cited,
            "conflict_analysis": "Static RAG cannot compare against external updates." if conflict_result["conflict_detected"] else "No external conflict check performed.",
            "effective_date_analysis": "Static RAG does not validate newer external effective dates.",
            "human_review_required": hr,
            "query_type": query_classification["query_type"],
            "hallucination_risk": "low" if cited else "medium",
            "limitations": "Static RAG uses internal knowledge only and may lag current external standards.",
        }

    if model_variant == "wag_only":
        cited = _cite(wag_evidence, 2)
        if not cited:
            conclusion = "Insufficient external source evidence was retrieved. Manual review is required."
            hr = True
        elif internal_question:
            conclusion = "WAG retrieved external evidence but lacks internal policy context needed for a complete company-specific conclusion."
            hr = True
        else:
            conclusion = "Use the cited external authoritative evidence as the primary basis for the compliance response."
            hr = conflict_result["conflict_detected"] or date_result["requires_manual_review"]
        return {
            "compliance_conclusion": conclusion,
            "supporting_evidence": cited,
            "conflict_analysis": "WAG can identify external authority but cannot fully compare against internal policy if RAG is absent.",
            "effective_date_analysis": "External evidence was checked for date signals." if date_result["needed"] else "No effective-date issue identified.",
            "human_review_required": hr,
            "query_type": query_classification["query_type"],
            "hallucination_risk": "low" if cited else "medium",
            "limitations": "WAG-only may be unstable without source snapshots and may miss internal policy context.",
        }

    # Hybrid variants.
    cited = _cite(ranked_evidence, 4)
    if trap:
        if verifier_enabled:
            conclusion = "The requested rule or standard cannot be verified from the retrieved evidence. Do not invent unsupported standards; manual review is required."
            hr = True
        else:
            conclusion = "The retrieved evidence should be checked before relying on the standard referenced in the question."
            hr = False
    elif conflict_result["conflict_detected"]:
        if verifier_enabled:
            conclusion = "A potential conflict exists between internal policy and external evidence. Prefer more authoritative external sources where applicable, and require manual compliance review before a final operational decision."
            hr = True
        else:
            conclusion = "Use the merged internal and external evidence to support the compliance response."
            hr = False
    elif date_result["requires_manual_review"]:
        if verifier_enabled:
            conclusion = "The answer depends on an effective-date judgment that is not fully resolved by the retrieved evidence. Manual review is required."
            hr = True
        else:
            conclusion = "Use the retrieved evidence to assess the effective-date question."
            hr = False
    elif cited:
        conclusion = "Use the ranked internal and external evidence to support the compliance response."
        hr = False
    else:
        conclusion = "No sufficient evidence was retrieved. Manual review is required."
        hr = True

    if verifier_enabled:
        # The verifier-enabled variant applies conservative post-generation policy.
        if not cited or trap or conflict_result["conflict_detected"] or date_result["requires_manual_review"]:
            hr = True

    return {
        "compliance_conclusion": conclusion,
        "supporting_evidence": cited,
        "conflict_analysis": (
            "; ".join(conflict_result["conflicts"])
            if conflict_result["conflicts"] and verifier_enabled
            else ("Evidence was merged; no post-generation review was applied." if conflict_result["conflicts"] else "No inconsistency detected from retrieved evidence.")
        ),
        "effective_date_analysis": date_result["reason"],
        "human_review_required": hr,
        "query_type": query_classification["query_type"],
        "hallucination_risk": "low" if cited and not trap else "controlled",
        "limitations": "Hybrid deterministic generator for reproducible architecture testing. Replace with real LLM call for final reported model outputs.",
    }
