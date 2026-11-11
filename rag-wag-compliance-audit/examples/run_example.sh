# Example smoke test independent of prior packages

python scripts/01_run_five_models.py \
  --qa-cases examples/example_qa_cases.csv \
  --chunks examples/example_chunks.csv \
  --output outputs/model_outputs/example_outputs.jsonl \
  --mock

python scripts/02_run_verifier.py \
  --model-outputs outputs/model_outputs/example_outputs.jsonl \
  --output outputs/verifier_outputs/example_verifier_outputs.jsonl

python scripts/03_evaluate_results.py \
  --model-outputs outputs/model_outputs/example_outputs.jsonl \
  --verifier-outputs outputs/verifier_outputs/example_verifier_outputs.jsonl \
  --output outputs/evaluation_results/example_metrics.csv
