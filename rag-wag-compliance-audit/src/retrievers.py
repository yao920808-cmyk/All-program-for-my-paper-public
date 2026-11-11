"""
Retrieval layer for Step 6 experiments.

This module supports:
1. TF-IDF retrieval if the Step 5 index files are available.
2. Fallback lexical retrieval if the TF-IDF index is not available.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from .common import read_csv, infer_source_type, get_text_field, get_id_field


class LexicalRetriever:
    def __init__(self, chunks: List[Dict[str, str]]):
        self.chunks = chunks
        self.index = []
        for row in chunks:
            text = self._search_text(row)
            toks = self._tokens(text)
            self.index.append((row, toks))

    @staticmethod
    def _tokens(text: str) -> set:
        return set(re.findall(r"[A-Za-z0-9_\\-]+", (text or "").lower()))

    @staticmethod
    def _search_text(row: Dict[str, str]) -> str:
        fields = [
            get_text_field(row),
            row.get("chunk_id", ""),
            row.get("document_id", ""),
            row.get("source_document_id", ""),
            row.get("corpus_id", ""),
            row.get("title", ""),
            row.get("source_name", ""),
        ]
        return " ".join(str(x) for x in fields if x)

    @staticmethod
    def _explicit_ids(query: str) -> set[str]:
        patterns = [
            r"\bRAG-POLICY-\d{3}(?:-C\d{3})?\b",
            r"\bPOLICY-\d{3}\b",
            r"\bWAG-SYN-EXT-\d{3}(?:-C\d{3})?\b",
            r"\bEXT-\d{3}\b",
            r"\bWAG-CHUNK-\d{5}\b",
        ]
        found = set()
        for pattern in patterns:
            found.update(m.upper() for m in re.findall(pattern, query or "", flags=re.I))
        return found

    @staticmethod
    def _matches_explicit_id(row: Dict[str, str], explicit_id: str) -> bool:
        joined = " ".join(
            str(row.get(k, ""))
            for k in ["chunk_id", "document_id", "source_document_id", "corpus_id", "title", "chunk_text"]
        ).upper()
        if explicit_id in joined:
            return True
        if explicit_id.startswith("POLICY-") and f"RAG-{explicit_id}" in joined:
            return True
        if explicit_id.startswith("EXT-") and f"WAG-SYN-{explicit_id}" in joined:
            return True
        return False

    def retrieve(self, query: str, top_k: int = 5, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        q = self._tokens(query)
        explicit_ids = self._explicit_ids(query)
        scored = []
        for row, toks in self.index:
            st = infer_source_type(row)
            if source_filter == "internal" and st != "internal_policy":
                continue
            if source_filter == "external" and st == "internal_policy":
                continue
            overlap = len(q & toks)
            score = overlap / max(1, math.sqrt(len(toks)))
            explicit_hits = sum(1 for explicit_id in explicit_ids if self._matches_explicit_id(row, explicit_id))
            if explicit_hits:
                score += 10.0 * explicit_hits
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, row in scored[:top_k]:
            results.append({
                "evidence_id": get_id_field(row),
                "source_type": infer_source_type(row),
                "score": round(float(score), 6),
                "title": row.get("title") or row.get("document_title") or row.get("source_name") or "",
                "text": get_text_field(row)[:1800],
                "metadata": row
            })
        return results


def load_retriever(chunks_path: str | Path) -> LexicalRetriever:
    chunks = read_csv(chunks_path)
    return LexicalRetriever(chunks)


def evidence_to_text(evidence: List[Dict[str, Any]]) -> str:
    parts = []
    for item in evidence:
        parts.append(
            f"[{item.get('evidence_id')}] "
            f"source_type={item.get('source_type')} "
            f"title={item.get('title')}\n"
            f"{item.get('text')}"
        )
    return "\n\n".join(parts)
