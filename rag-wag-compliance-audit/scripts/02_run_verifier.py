#!/usr/bin/env python3
"""
Run rule-based verifier on model outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import read_jsonl, write_jsonl
from src.verifier import verify_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-outputs", default="outputs/model_outputs/step6_model_outputs.jsonl")
    parser.add_argument("--output", default="outputs/verifier_outputs/step6_verifier_outputs.jsonl")
    args = parser.parse_args()

    rows = read_jsonl(ROOT / args.model_outputs)
    verifier_rows = [verify_output(r) for r in rows]
    write_jsonl(ROOT / args.output, verifier_rows)
    print(f"Wrote {len(verifier_rows)} verifier rows to {ROOT / args.output}")


if __name__ == "__main__":
    main()
