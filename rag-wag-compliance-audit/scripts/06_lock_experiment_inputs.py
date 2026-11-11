#!/usr/bin/env python3
"""Lock experiment inputs with deterministic split and provenance hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.common import read_csv


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assign_splits(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_type[row.get("question_type") or "unknown"].append(row)

    output = []
    for qtype, items in sorted(by_type.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n >= 3:
            n_validation = max(1, int(round(n * 0.15)))
            n_test = max(1, int(round(n * 0.15)))
            n_train = max(1, n - n_validation - n_test)
        else:
            n_train = n
            n_validation = 0
            n_test = 0
        for idx, row in enumerate(shuffled):
            split = "train" if idx < n_train else "validation" if idx < n_train + n_validation else "test"
            out = dict(row)
            out["experiment_split"] = split
            out["split_seed"] = str(seed)
            out["split_strategy"] = "deterministic_stratified_by_question_type_70_15_15_min_one_validation_and_test_when_possible"
            output.append(out)
    output.sort(key=lambda r: r.get("qa_id") or r.get("question_id") or "")
    return output


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-cases", default=str(WORKSPACE / "prepared_data" / "experiments" / "final_experiment_input_470.csv"))
    parser.add_argument("--chunks", default=str(WORKSPACE / "prepared_data" / "experiments" / "combined_chunks_internal_external.csv"))
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--output-dir", default=str(WORKSPACE / "prepared_data" / "experiments"))
    args = parser.parse_args()

    qa_path = Path(args.qa_cases).resolve()
    chunks_path = Path(args.chunks).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = read_csv(qa_path)
    chunks = read_csv(chunks_path)
    split_rows = assign_splits(rows, args.seed)

    split_path = output_dir / "final_experiment_input_470_with_split.csv"
    write_csv(split_path, split_rows)

    split_counts = Counter(r["experiment_split"] for r in split_rows)
    type_counts = Counter(r.get("question_type") or "unknown" for r in split_rows)
    type_split_counts = Counter((r.get("question_type") or "unknown", r["experiment_split"]) for r in split_rows)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Locked input manifest for reproducible Step 6 model experiments.",
        "answer_generation_note": "Ground-truth labels are retained for evaluation only. Model routing/generation must not use expected answer labels.",
        "api_key_present": bool(os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")),
        "inputs": {
            "qa_cases": {
                "path": str(qa_path),
                "rows": len(rows),
                "sha256": sha256_file(qa_path),
            },
            "combined_chunks": {
                "path": str(chunks_path),
                "rows": len(chunks),
                "sha256": sha256_file(chunks_path),
            },
            "split_qa_cases": {
                "path": str(split_path),
                "rows": len(split_rows),
                "sha256": sha256_file(split_path),
                "seed": args.seed,
                "strategy": "deterministic_stratified_by_question_type_70_15_15_min_one_validation_and_test_when_possible",
            },
        },
        "split_counts": dict(split_counts),
        "question_type_counts": dict(type_counts),
        "question_type_split_counts": {
            f"{qtype}|{split}": count for (qtype, split), count in sorted(type_split_counts.items())
        },
        "recommended_real_llm_smoke_test": {
            "model_variants": "hybrid_rag_wag_verifier",
            "limit": 5,
            "temperature": 0,
            "top_k": 5,
        },
    }

    manifest_path = ROOT / "outputs" / "experiment_manifest" / "locked_experiment_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote split file to {split_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
