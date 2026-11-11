#!/usr/bin/env python3
"""Apply the runtime verifier to existing real-LLM outputs without new API calls."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import read_jsonl, write_jsonl
from src.runtime_verifier import apply_runtime_verifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-outputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-variant", default="hybrid_rag_wag_verifier")
    parser.add_argument("--skip", action="store_true", help="Copy rows without applying verifier; useful for without_verifier ablation.")
    args = parser.parse_args()

    rows = read_jsonl(ROOT / args.model_outputs)
    applied = 0
    for row in rows:
        if args.skip or row.get("model_variant") != args.target_variant:
            continue
        answer = row.get("answer")
        if not isinstance(answer, dict):
            continue
        answer, runtime_verifier = apply_runtime_verifier(
            answer=answer,
            retrieved_evidence=(row.get("rag_evidence") or []) + (row.get("wag_evidence") or []),
            conflict_result=row.get("conflict_result") or {},
            effective_date_result=row.get("effective_date_result") or {},
        )
        row["answer"] = answer
        meta = row.setdefault("answer_generation", {})
        meta["runtime_verifier"] = runtime_verifier
        applied += 1

    write_jsonl(ROOT / args.output, rows)
    print(f"Wrote {len(rows)} row(s) to {ROOT / args.output}; runtime verifier applied to {applied} row(s).")


if __name__ == "__main__":
    main()
