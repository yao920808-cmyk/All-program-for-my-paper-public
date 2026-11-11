#!/usr/bin/env python3
"""Create a small balanced smoke-test QA file across question types."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.common import read_csv


DEFAULT_TYPES = [
    "internal_policy_qa",
    "external_standard_qa",
    "conflict_resolution_qa",
    "effective_date_qa",
    "hallucination_trap_qa",
    "report_generation_qa",
    "source_authority_citation_qa",
    "public_fasb_update_qa",
    "public_sec_edgar_qa",
    "public_pcaob_evidence_qa",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-cases", default=str(WORKSPACE / "prepared_data" / "experiments" / "final_experiment_input_470_with_split.csv"))
    parser.add_argument("--output", default=str(ROOT / "inputs" / "balanced_smoke_input_10.csv"))
    parser.add_argument("--question-types", default=",".join(DEFAULT_TYPES))
    parser.add_argument("--prefer-split", default="test")
    args = parser.parse_args()

    rows = read_csv(args.qa_cases)
    requested = [x.strip() for x in args.question_types.split(",") if x.strip()]
    selected = []
    used_ids = set()

    for qtype in requested:
        candidates = [r for r in rows if r.get("question_type") == qtype and r.get("experiment_split") == args.prefer_split]
        if not candidates:
            candidates = [r for r in rows if r.get("question_type") == qtype]
        if not candidates:
            print(f"Warning: no candidate for question_type={qtype}")
            continue
        row = sorted(candidates, key=lambda r: r.get("qa_id") or r.get("question_id") or "")[0]
        qa_id = row.get("qa_id") or row.get("question_id")
        if qa_id not in used_ids:
            selected.append(row)
            used_ids.add(qa_id)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected)
    print(f"Wrote {len(selected)} balanced smoke rows to {out_path}")


if __name__ == "__main__":
    main()
