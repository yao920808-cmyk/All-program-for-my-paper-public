"""
Evidence ranking utilities.

Score = 0.4 relevance + 0.3 source authority + 0.2 recency + 0.1 effective-date match
as specified in the experimental design.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Dict, Any, List


def authority_score(evidence: Dict[str, Any]) -> float:
    md = evidence.get("metadata", {}) or {}
    level = str(md.get("source_level") or evidence.get("source_level") or "").strip().lower()
    publisher = str(md.get("publisher") or md.get("source_organization") or evidence.get("title") or "").lower()
    if level in {"1", "level 1"}:
        return 1.0
    if level in {"2", "level 2"}:
        return 0.8
    if level in {"3", "level 3"}:
        return 0.6
    if level == "internal":
        return 0.7
    if any(p in publisher for p in ["fasb", "sec", "pcaob", "irs", "aicpa"]):
        return 1.0
    if "big four" in publisher or any(p in publisher for p in ["deloitte", "pwc", "kpmg", "ey"]):
        return 0.8
    return 0.4


def recency_score(evidence: Dict[str, Any]) -> float:
    md = evidence.get("metadata", {}) or {}
    candidates = [md.get("effective_date"), md.get("publication_date"), md.get("access_date")]
    years = []
    for value in candidates:
        m = re.search(r"(20\d{2}|19\d{2})", str(value or ""))
        if m:
            years.append(int(m.group(1)))
    if not years:
        return 0.5
    year = max(years)
    return max(0.1, min(1.0, (year - 2010) / 16))


def effective_date_match(evidence: Dict[str, Any], qa_row: Dict[str, Any]) -> float:
    q = str(qa_row.get("question", "")).lower()
    date_terms = ["effective date", "applies after", "after the effective", "fiscal years"]
    if not any(t in q for t in date_terms):
        return 0.6
    md = evidence.get("metadata", {}) or {}
    text = " ".join([str(md.get("effective_date", "")), str(md.get("effective_date_note", "")), str(evidence.get("text", ""))]).lower()
    if re.search(r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}|20\d{2}", text) and "effective" in text:
        return 1.0
    if "manual" in text and "effective" in text:
        return 0.7
    return 0.2


def normalize_relevance(evidence: Dict[str, Any]) -> float:
    score = float(evidence.get("score") or 0.0)
    return max(0.0, min(1.0, score))


def rank_evidence(evidence: List[Dict[str, Any]], qa_row: Dict[str, Any], enable_source_ranking: bool = True, enable_date_check: bool = True) -> List[Dict[str, Any]]:
    ranked = []
    for item in evidence:
        rel = normalize_relevance(item)
        auth = authority_score(item) if enable_source_ranking else 0.5
        rec = recency_score(item)
        date_match = effective_date_match(item, qa_row) if enable_date_check else 0.5
        final = 0.4 * rel + 0.3 * auth + 0.2 * rec + 0.1 * date_match
        out = dict(item)
        out["ranking"] = {
            "relevance_score": round(rel, 6),
            "authority_score": round(auth, 6),
            "recency_score": round(rec, 6),
            "effective_date_match": round(date_match, 6),
            "final_score": round(final, 6),
        }
        ranked.append(out)
    ranked.sort(key=lambda x: x["ranking"]["final_score"], reverse=True)
    return ranked
