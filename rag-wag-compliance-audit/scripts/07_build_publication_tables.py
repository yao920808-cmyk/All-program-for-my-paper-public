#!/usr/bin/env python3
"""Build thesis/paper-ready CSV tables from Step 6 outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.common import read_csv, read_jsonl
from src.evaluator import compute_metrics


METRIC_FIELDS = [
    "n",
    "answer_accuracy",
    "retrieved_citation_match",
    "evidence_precision",
    "evidence_recall",
    "hallucination_rate",
    "conflict_detection_accuracy",
    "effective_date_accuracy",
    "human_review_flag_accuracy",
    "expected_evidence_match_rate",
]


def publication_metric(row: dict, field: str):
    if field == "retrieved_citation_match":
        return row.get("retrieved_citation_match", row.get("citation_accuracy", ""))
    return row.get(field, "")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_rows_by(results: list[dict], verifier: list[dict], group_keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in results:
        key = tuple(row.get(k, "") for k in group_keys)
        grouped[key].append(row)
    rows = []
    for key, items in sorted(grouped.items()):
        metrics = compute_metrics(items, verifier)
        metrics.pop("citation_accuracy", None)
        out = {k: v for k, v in zip(group_keys, key)}
        out.update(metrics)
        rows.append(out)
    return rows


def infer_result_mode(results: list[dict]) -> tuple[str, dict]:
    modes = Counter()
    models = Counter()
    providers = Counter()
    for row in results:
        meta = row.get("answer_generation") or {}
        modes[str(meta.get("mode", "unknown"))] += 1
        model_name = str(meta.get("model_name", "")).strip()
        provider = str(meta.get("provider", "")).strip()
        if model_name:
            models[model_name] += 1
        if provider:
            providers[provider] += 1

    if set(modes) == {"real_llm"}:
        result_mode = "real_llm_api_result"
    elif set(modes) == {"mock"}:
        result_mode = "deterministic_mock_pipeline_result_not_final_llm_claim"
    else:
        result_mode = "mixed_result_modes_review_before_reporting"

    return result_mode, {
        "answer_generation_modes": dict(modes),
        "model_names": dict(models),
        "providers": dict(providers),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-outputs", default="outputs/model_outputs/prepared_step6_model_outputs.jsonl")
    parser.add_argument("--verifier-outputs", default="outputs/verifier_outputs/prepared_step6_verifier_outputs.jsonl")
    parser.add_argument("--overall-metrics", default="outputs/evaluation_results/prepared_step6_metrics.csv")
    parser.add_argument("--ablation-summary", default="outputs/ablation/ablation_summary_hybrid_verifier.csv")
    parser.add_argument("--error-analysis", default="outputs/evaluation_results/prepared_error_analysis_cases.csv")
    parser.add_argument("--split-file", default=str(WORKSPACE / "prepared_data" / "experiments" / "final_experiment_input_470_with_split.csv"))
    parser.add_argument("--output-dir", default="outputs/publication_tables")
    args = parser.parse_args()

    results = read_jsonl(ROOT / args.model_outputs)
    verifier = read_jsonl(ROOT / args.verifier_outputs)
    overall = read_csv(ROOT / args.overall_metrics)
    ablation = []
    if args.ablation_summary.strip():
        ablation_path = ROOT / args.ablation_summary
        if ablation_path.exists():
            ablation = read_csv(ablation_path)
    errors = read_csv(ROOT / args.error_analysis)
    result_mode, generation_summary = infer_result_mode(results)
    split_rows = read_csv(Path(args.split_file))
    split_by_id = {
        (r.get("qa_id") or r.get("question_id")): r.get("experiment_split", "")
        for r in split_rows
    }
    for row in results:
        row["experiment_split"] = split_by_id.get(row.get("qa_id"), "")

    out_dir = ROOT / args.output_dir

    table1 = []
    for row in overall:
        table1.append({
            "model_variant": row.get("model_variant"),
            **{field: publication_metric(row, field) for field in METRIC_FIELDS},
            "result_mode": result_mode,
        })
    write_csv(out_dir / "table1_model_overall_metrics.csv", table1)

    table2 = metric_rows_by(results, verifier, ["question_type", "model_variant"])
    for row in table2:
        row["result_mode"] = result_mode
    write_csv(out_dir / "table2_question_type_metrics.csv", table2)

    test_results = [row for row in results if row.get("experiment_split") == "test"]
    table2_test = metric_rows_by(test_results, verifier, ["model_variant"])
    for row in table2_test:
        row["experiment_split"] = "test"
        row["result_mode"] = result_mode
    write_csv(out_dir / "table2b_test_split_model_metrics.csv", table2_test)

    table2c = metric_rows_by(results, verifier, ["experiment_split", "question_type", "model_variant"])
    for row in table2c:
        row["result_mode"] = result_mode
    write_csv(out_dir / "table2c_split_question_type_metrics.csv", table2c)

    table3 = []
    for row in ablation:
        table3.append({
            "ablation_variant": row.get("ablation_variant"),
            "model_variant": row.get("model_variant"),
            **{field: publication_metric(row, field) for field in METRIC_FIELDS},
            "result_mode": result_mode if ablation else "ablation_not_run_for_this_table_build",
        })
    write_csv(out_dir / "table3_ablation_summary.csv", table3)

    error_counts = Counter((r.get("model_variant", ""), r.get("error_type", "")) for r in errors)
    table4 = [
        {"model_variant": model, "error_type": error_type, "count": count}
        for (model, error_type), count in sorted(error_counts.items())
    ]
    write_csv(out_dir / "table4_error_type_counts.csv", table4)

    summary = {
        "result_mode": result_mode,
        "rows": {
            "model_outputs": len(results),
            "verifier_outputs": len(verifier),
            "overall_metric_rows": len(overall),
            "ablation_rows": len(ablation),
            "error_analysis_rows": len(errors),
        },
        **generation_summary,
        "publication_tables": [
            "table1_model_overall_metrics.csv",
            "table2_question_type_metrics.csv",
            "table2b_test_split_model_metrics.csv",
            "table2c_split_question_type_metrics.csv",
            "table3_ablation_summary.csv",
            "table4_error_type_counts.csv",
        ],
    }
    (out_dir / "publication_table_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote publication tables to {out_dir}")


if __name__ == "__main__":
    main()
