#!/usr/bin/env python3
"""
Evaluate model outputs and verifier outputs.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import read_jsonl
from src.evaluator import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-outputs", default="outputs/model_outputs/step6_model_outputs.jsonl")
    parser.add_argument("--verifier-outputs", default="outputs/verifier_outputs/step6_verifier_outputs.jsonl")
    parser.add_argument("--output", default="outputs/evaluation_results/step6_model_comparison_metrics.csv")
    args = parser.parse_args()

    results = read_jsonl(ROOT / args.model_outputs)
    verifier = read_jsonl(ROOT / args.verifier_outputs)

    by_model = defaultdict(list)
    for r in results:
        by_model[r.get("model_variant")].append(r)

    rows = []
    for model, model_rows in by_model.items():
        metrics = compute_metrics(model_rows, verifier)
        metrics["model_variant"] = model
        rows.append(metrics)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_variant", "n", "answer_accuracy", "retrieved_citation_match",
        "evidence_precision", "evidence_recall", "hallucination_rate",
        "conflict_detection_accuracy", "effective_date_accuracy",
        "human_review_flag_accuracy", "expected_evidence_match_rate"
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        # compute_metrics retains a legacy citation_accuracy alias in memory.
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
