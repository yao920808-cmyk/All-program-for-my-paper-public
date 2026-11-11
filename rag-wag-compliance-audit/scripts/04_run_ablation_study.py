#!/usr/bin/env python3
"""Run ablation variants for the prepared RAG-WAG experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("Running:", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-cases", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model-variants", default="hybrid_rag_wag_verifier")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-name", default="gpt-5.4-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    args = parser.parse_args()
    qa_cases = str(Path(args.qa_cases).resolve())
    chunks = str(Path(args.chunks).resolve())

    variants = [
        ("full_model", []),
        ("without_wag", ["--disable-wag"]),
        ("without_rag", ["--disable-rag"]),
        ("without_verifier", ["--disable-verifier"]),
        ("without_date_check", ["--disable-date-check"]),
        ("without_source_ranking", ["--disable-source-ranking"]),
    ]

    summary_rows = []
    for name, flags in variants:
        model_out = f"outputs/ablation/{name}_model_outputs.jsonl"
        verifier_out = f"outputs/ablation/{name}_verifier_outputs.jsonl"
        metrics_out = f"outputs/ablation/{name}_metrics.csv"
        base = [
            sys.executable, "scripts/01_run_five_models.py",
            "--qa-cases", qa_cases,
            "--chunks", chunks,
            "--output", model_out,
            "--top-k", str(args.top_k),
            "--model-name", args.model_name,
            "--temperature", str(args.temperature),
            "--max-tokens", str(args.max_tokens),
            "--request-delay-seconds", str(args.request_delay_seconds),
            "--model-variants", args.model_variants,
        ]
        if args.mock:
            base.append("--mock")
        if args.resume:
            base.append("--resume")
        if args.limit:
            base.extend(["--limit", str(args.limit)])
        run(base + flags)
        run([sys.executable, "scripts/02_run_verifier.py", "--model-outputs", model_out, "--output", verifier_out])
        run([sys.executable, "scripts/03_evaluate_results.py", "--model-outputs", model_out, "--verifier-outputs", verifier_out, "--output", metrics_out])
        with (ROOT / metrics_out).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("model_variant") == "hybrid_rag_wag_verifier":
                    row["ablation_variant"] = name
                    summary_rows.append(row)

    out = ROOT / "outputs" / "ablation" / "ablation_summary_hybrid_verifier.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ablation_variant", "model_variant", "n", "answer_accuracy", "retrieved_citation_match", "evidence_precision", "evidence_recall", "hallucination_rate", "conflict_detection_accuracy", "effective_date_accuracy", "human_review_flag_accuracy", "expected_evidence_match_rate"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote ablation summary to {out}")


if __name__ == "__main__":
    main()
