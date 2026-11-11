"""
Common utilities for Step 6 RAG-WAG model experiments.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Iterable


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Attempts to parse model output as JSON.
    If parsing fails, returns a wrapped object so evaluation can continue.
    """
    if isinstance(text, dict):
        return text
    text = str(text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {
        "compliance_conclusion": text,
        "supporting_evidence": [],
        "conflict_analysis": "",
        "effective_date_analysis": "",
        "human_review_required": None,
        "limitations": "Output was not valid JSON."
    }


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"true", "yes", "y", "1", "required"}:
        return True
    if s in {"false", "no", "n", "0", "not required"}:
        return False
    return None


def normalize_review_flag(value: Any) -> str:
    """Normalize human-review labels without collapsing Conditional to None."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "unknown"
    s = str(value).strip().lower()
    if s in {"true", "yes", "y", "1", "required"}:
        return "yes"
    if s in {"false", "no", "n", "0", "not required"}:
        return "no"
    if s in {"conditional", "maybe", "depends"}:
        return "conditional"
    return "unknown"


def split_ids(value: Any) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in re.split(r"[;|,]", str(value)) if x.strip()]


def infer_source_type(row: Dict[str, str]) -> str:
    candidates = [
        row.get("source_type"),
        row.get("document_type"),
        row.get("record_type"),
        row.get("corpus_type"),
        row.get("type")
    ]
    joined = " ".join([str(c).lower() for c in candidates if c])
    if "internal" in joined or "policy" in joined and "external" not in joined:
        return "internal_policy"
    if "wag" in joined or "external" in joined or "public" in joined or "sec" in joined or "fasb" in joined or "pcaob" in joined:
        return "external_source"
    return "unknown"


def get_text_field(row: Dict[str, str]) -> str:
    for key in ["chunk_text", "text", "content", "summary", "dataset_text", "source_summary"]:
        if key in row and row[key]:
            return row[key]
    return " ".join(str(v) for v in row.values() if v)


def get_id_field(row: Dict[str, str]) -> str:
    for key in ["chunk_id", "record_id", "document_id", "external_id", "id"]:
        if key in row and row[key]:
            return row[key]
    return "unknown_id"
