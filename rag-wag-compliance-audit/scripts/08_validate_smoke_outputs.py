#!/usr/bin/env python3
"""Validate real-LLM smoke-test outputs for schema and citation stability."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import read_jsonl


REQUIRED_ANSWER_FIELDS = [
    "compliance_conclusion",
    "supporting_evidence",
    "conflict_analysis",
    "effective_date_analysis",
    "human_review_required",
    "limitations",
]


def validate_row(row: dict, require_real: bool) -> list[str]:
    errors: list[str] = []
    qa_id = row.get("qa_id", "unknown")
    variant = row.get("model_variant", "unknown")
    meta = row.get("answer_generation") or {}
    answer = row.get("answer")

    if require_real and meta.get("mode") != "real_llm":
        errors.append(f"{qa_id}/{variant}: answer_generation.mode is not real_llm")
    if require_real and not meta.get("response_id"):
        errors.append(f"{qa_id}/{variant}: missing real LLM response_id")

    if not isinstance(answer, dict):
        errors.append(f"{qa_id}/{variant}: answer is not a JSON object")
        return errors

    for field in REQUIRED_ANSWER_FIELDS:
        if field not in answer:
            errors.append(f"{qa_id}/{variant}: missing answer field {field}")

    if "supporting_evidence" in answer and not isinstance(answer["supporting_evidence"], list):
        errors.append(f"{qa_id}/{variant}: supporting_evidence is not a list")
    if "human_review_required" in answer and not isinstance(answer["human_review_required"], bool):
        errors.append(f"{qa_id}/{variant}: human_review_required is not boolean")

    retrieved_ids = {
        str(e.get("evidence_id"))
        for e in (row.get("rag_evidence") or []) + (row.get("wag_evidence") or [])
        if isinstance(e, dict)
    }
    cited = answer.get("supporting_evidence") if isinstance(answer, dict) else []
    if retrieved_ids and isinstance(cited, list) and len(cited) == 0:
        errors.append(f"{qa_id}/{variant}: no supporting_evidence citations despite retrieved evidence")
    if isinstance(cited, list):
        for item in cited:
            if not isinstance(item, dict):
                errors.append(f"{qa_id}/{variant}: citation item is not an object")
                continue
            evidence_id = str(item.get("evidence_id", "")).strip()
            if not evidence_id:
                errors.append(f"{qa_id}/{variant}: citation missing evidence_id")
            elif evidence_id not in retrieved_ids:
                errors.append(f"{qa_id}/{variant}: citation {evidence_id} not found in retrieved evidence")

    fallback_text = str(answer.get("limitations", "")).lower()
    if require_real and "could not be parsed" in fallback_text:
        errors.append(f"{qa_id}/{variant}: response fell back after JSON parse failure")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-outputs", default="outputs/model_outputs/real_llm_smoke_hybrid5.jsonl")
    parser.add_argument("--output", default="outputs/evaluation_results/real_llm_smoke_validation.csv")
    parser.add_argument("--expected-rows", type=int, default=5)
    parser.add_argument("--require-real", action="store_true")
    args = parser.parse_args()

    input_path = ROOT / args.model_outputs
    if not input_path.exists():
        print(f"Validation failed: model output file does not exist: {input_path}")
        raise SystemExit(1)

    rows = read_jsonl(input_path)
    validation_rows = []
    all_errors: list[str] = []

    if len(rows) != args.expected_rows:
        all_errors.append(f"Expected {args.expected_rows} rows, found {len(rows)} rows")

    for row in rows:
        errors = validate_row(row, args.require_real)
        all_errors.extend(errors)
        validation_rows.append({
            "qa_id": row.get("qa_id"),
            "model_variant": row.get("model_variant"),
            "answer_generation_mode": (row.get("answer_generation") or {}).get("mode"),
            "response_id": (row.get("answer_generation") or {}).get("response_id"),
            "valid": len(errors) == 0,
            "errors": " | ".join(errors),
        })

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qa_id", "model_variant", "answer_generation_mode", "response_id", "valid", "errors"])
        writer.writeheader()
        writer.writerows(validation_rows)

    if all_errors:
        print(f"Validation failed with {len(all_errors)} issue(s). Details written to {out_path}")
        for error in all_errors[:20]:
            print(error)
        raise SystemExit(1)

    print(f"Validation passed for {len(rows)} row(s). Details written to {out_path}")


if __name__ == "__main__":
    main()
