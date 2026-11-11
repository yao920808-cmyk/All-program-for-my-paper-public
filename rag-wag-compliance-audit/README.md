# Hybrid RAG-WAG Financial Compliance Experiment

Research program comparing five variants: LLM Only, Static RAG, WAG Only,
Hybrid RAG-WAG, and Hybrid RAG-WAG with a runtime verifier.

Here WAG means retrieval from fixed external/public-source snapshots. The
program does not perform uncontrolled live browsing. The bundled CSV examples
are fictional teaching fixtures, including the external-looking excerpts;
they are not authenticated accounting standards or financial advice.

## Requirements

Python 3.10 or newer; the core program uses only the Python standard library.
The original experiments used Python 3.12. PowerShell is needed for the `.ps1`
wrapper. No API key is required for the mock smoke test.

## Offline Quick Start

Clone the repository, then run these commands in PowerShell:

```powershell
cd rag-wag-compliance-audit
.\scripts\run_prepared_step6.ps1 -Mock -QaCases examples\example_qa_cases.csv -Chunks examples\example_chunks.csv -OutputStem example
```

This runs 2 illustrative questions through all 5 variants, producing 10 output
records, verifier records, metrics, and error-analysis files under `outputs/`.
Mock scores only check the pipeline; they are not empirical LLM results.

On Bash, from the same program directory:

```bash
bash examples/run_example.sh
```

## Real LLM Run

Enter your own key into the local PowerShell prompt. This keeps the key out of
the command text and repository. Real API requests incur provider charges.

```powershell
$secureKey = Read-Host "OpenAI API key" -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
}
$modelId = Read-Host "Exact API model ID available to your account"
.\scripts\run_prepared_step6.ps1 -QaCases examples\example_qa_cases.csv -Chunks examples\example_chunks.csv -OutputStem real_example -ModelVariants hybrid_rag_wag_verifier -ModelName $modelId -Temperature 0
python scripts/08_validate_smoke_outputs.py --model-outputs outputs/model_outputs/real_example_model_outputs.jsonl --expected-rows 2 --require-real
```

The client also includes OpenRouter and Gemini adapters. The code chooses
provider and base URL from the model and environment variables; inspect
`src/llm_client.py` when configuring those adapters. Do not commit credentials.
Use a fresh `OutputStem` for a changed model or prompt. `-Resume` skips completed
QA/variant pairs and does not check configuration equivalence.

## Study Data

The original 470-QA dataset, combined corpus, public-source snapshots, transaction
records, human reviews, and experiment outputs are not included in this
source-only upload. The original default paths point to a sibling
`prepared_data/experiments/` directory. Pass explicit input paths when using
your own data; the examples above work independently of those original paths.

QA CSV columns include `qa_id`, `question_type`, `question`, `expected_answer`,
`expected_evidence_id`, `conflict_type`, and `requires_human_review`.
Chunk CSV columns include `chunk_id`, `source_type`, `title`, and `chunk_text`.
See `examples/` for exact formats. Gold fields support offline evaluation and
must not be supplied as answer hints in a confirmatory experiment.

## Program Layout

- `src/`: classifier, lexical retriever, ranking, conflict/date checks, generator,
  provider clients, runtime verifier, gold-aware offline scorer, evaluator.
- `scripts/01_run_five_models.py`: common experiment entry point.
- `scripts/02_run_verifier.py`: offline scoring against labels.
- `scripts/03_evaluate_results.py`: aggregate automatic metrics.
- `scripts/04_run_ablation_study.py`: component ablations.
- `scripts/05_error_analysis.py`: error-case export.
- `scripts/06_lock_experiment_inputs.py`: split and provenance preparation.
- `scripts/07_build_publication_tables.py`: table exports; original-study paths
  require the separately held datasets.
- `scripts/08_validate_smoke_outputs.py`: structural and real-response checks.
- `scripts/09_make_balanced_smoke_subset.py`: question-type sampling.
- `scripts/10_apply_runtime_verifier.py`: verifier replay without API calls.
- `prompts/`: exact prompt files from the local framework.
- `source_manifest.json`: SHA256 inventory of the staged source files.

## Interpretation and Known Limitations

Retrieval is lexical token overlap with explicit-document-ID boosts, not vector
embedding retrieval. Source-ranking scores are recorded, but the current
generation prompt uses the channel retrieval lists rather than the ranked
list; a ranking ablation therefore does not establish a causal ranking benefit.

All real-LLM variants share a citation guard. Retrieved Citation Match measures
ID membership after that guard, not whether the cited passage entails the
answer. Automatic answer accuracy is an evidence-based proxy that structurally
disadvantages LLM Only. The offline scorer uses expected labels and is distinct
from the runtime verifier. Hallucination markers and date/conflict rules are
heuristics and require semantic human validation. A schema validation pass
does not establish correct accounting reasoning.

Temperature zero does not guarantee identical provider responses. Keep model
IDs, prompts, input hashes, snapshots, and run configuration fixed and logged.
The source upload does not itself reproduce the archived paper results.

No redistribution license has been assigned in this upload.
