#!/usr/bin/env python3
"""
Run the five comparison models.

Usage:
python scripts/01_run_five_models.py \
  --qa-cases path/to/compliance_qa_cases_470_with_public_validation.csv \
  --chunks path/to/rag_wag_chunks.csv \
  --output outputs/model_outputs/step6_model_outputs.jsonl \
  --mock
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import read_csv, read_jsonl
from src.retrievers import load_retriever
from src.model_runner import ModelRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-cases", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output", default="outputs/model_outputs/step6_model_outputs.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model-variants", default="llm_only,static_rag,wag_only,hybrid_rag_wag,hybrid_rag_wag_verifier")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-name", default="gpt-5.4-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--disable-rag", action="store_true")
    parser.add_argument("--disable-wag", action="store_true")
    parser.add_argument("--disable-verifier", action="store_true")
    parser.add_argument("--disable-date-check", action="store_true")
    parser.add_argument("--disable-source-ranking", action="store_true")
    args = parser.parse_args()

    qa_rows = read_csv(args.qa_cases)
    if args.limit and args.limit > 0:
        qa_rows = qa_rows[:args.limit]

    if not args.mock and not (os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise SystemExit("No OPENAI_API_KEY, LLM_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY found. Use --mock or set an API key before running a real LLM experiment.")

    retriever = load_retriever(args.chunks)
    runner = ModelRunner(
        retriever=retriever,
        prompts_dir=ROOT / "prompts",
        top_k=args.top_k,
        use_mock=args.mock,
        enable_rag=not args.disable_rag,
        enable_wag=not args.disable_wag,
        enable_verifier=not args.disable_verifier,
        enable_date_check=not args.disable_date_check,
        enable_source_ranking=not args.disable_source_ranking,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    allowed_variants = {
        "llm_only",
        "static_rag",
        "wag_only",
        "hybrid_rag_wag",
        "hybrid_rag_wag_verifier",
    }
    model_variants = [v.strip() for v in args.model_variants.split(",") if v.strip()]
    unknown = [v for v in model_variants if v not in allowed_variants]
    if unknown:
        raise SystemExit(f"Unknown model variant(s): {', '.join(unknown)}")

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and out_path.exists():
        for row in read_jsonl(out_path):
            completed.add((str(row.get("qa_id")), str(row.get("model_variant"))))

    mode = "a" if args.resume and out_path.exists() else "w"
    written = 0
    for row in qa_rows:
        for variant in model_variants:
            qa_id = row.get("qa_id") or row.get("question_id")
            if (str(qa_id), variant) in completed:
                continue
            result = runner.run_one(row, variant)
            with out_path.open(mode, encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            mode = "a"
            written += 1
            if not args.mock and args.request_delay_seconds > 0:
                time.sleep(args.request_delay_seconds)

    total_existing = len(completed) if args.resume else 0
    print(f"Wrote {written} outputs to {out_path}; skipped {total_existing} existing rows.")


if __name__ == "__main__":
    main()
