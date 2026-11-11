#!/usr/bin/env python3
"""Create an error-analysis table from model and verifier outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import read_jsonl


def classify_error(result, verifier):
    if verifier.get("hallucination_detected"):
        return "hallucination"
    if verifier.get("citation_correct") is False:
        return "citation_error"
    if verifier.get("evidence_recall") is not None and float(verifier.get("evidence_recall")) == 0:
        return "retrieval_miss"
    if verifier.get("conflict_handled") is False:
        return "conflict_not_handled"
    if verifier.get("effective_date_correct") is False:
        return "effective_date_error"
    if verifier.get("human_review_correct") is False:
        return "human_review_flag_error"
    if verifier.get("supported") is False:
        return "unsupported_answer"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-outputs", default="outputs/model_outputs/step6_model_outputs.jsonl")
    parser.add_argument("--verifier-outputs", default="outputs/verifier_outputs/step6_verifier_outputs.jsonl")
    parser.add_argument("--output", default="outputs/evaluation_results/error_analysis_cases.csv")
    parser.add_argument("--max-per-model", type=int, default=25)
    args = parser.parse_args()

    results = read_jsonl(ROOT / args.model_outputs)
    verifier_rows = read_jsonl(ROOT / args.verifier_outputs)
    by_key = {(v.get("qa_id"), v.get("model_variant")): v for v in verifier_rows}
    counts = {}
    rows = []
    for r in results:
        key = (r.get("qa_id"), r.get("model_variant"))
        v = by_key.get(key, {})
        error = classify_error(r, v)
        if not error:
            continue
        model = r.get("model_variant")
        counts[model] = counts.get(model, 0) + 1
        if counts[model] > args.max_per_model:
            continue
        answer = r.get("answer", {}) if isinstance(r.get("answer"), dict) else {}
        rows.append(
            {
                "qa_id": r.get("qa_id"),
                "model_variant": model,
                "question_type": r.get("question_type"),
                "error_type": error,
                "question": r.get("question"),
                "expected_evidence_id": r.get("expected_evidence_id"),
                "mapped_evidence_chunk_ids": r.get("mapped_evidence_chunk_ids"),
                "cited_evidence_ids": ";".join(str(c.get("evidence_id")) for c in answer.get("supporting_evidence", []) if isinstance(c, dict)),
                "conflict_type": r.get("conflict_type"),
                "requires_human_review": r.get("requires_human_review"),
                "human_review_required_output": answer.get("human_review_required"),
                "verifier_reason": v.get("verifier_reason"),
                "correct_handling": "Retrieve/cite mapped evidence, handle conflicts/effective dates, and flag manual review when required.",
            }
        )

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["qa_id", "model_variant", "question_type", "error_type", "question", "expected_evidence_id", "mapped_evidence_chunk_ids", "cited_evidence_ids", "conflict_type", "requires_human_review", "human_review_required_output", "verifier_reason", "correct_handling"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} error-analysis rows to {out}")


if __name__ == "__main__":
    main()
