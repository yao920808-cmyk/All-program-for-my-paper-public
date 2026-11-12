# All-program-for-my-paper

A collection of research code for GRF gait signal analysis, pathology microscopy image tile reconstruction, and RAG/WAG financial compliance question-answering experiments.

The three subprojects implement independent research workflows, each with its own data formats, dependencies, and entry points. This page provides an overview and usage instructions. For algorithm details and mappings between the original and reorganized programs, consult the subproject documentation and source code.

> This repository primarily contains source code, configurations, prompts, and a small amount of demonstration data. Complete research datasets, manuscripts, pretrained weights, and full experimental outputs are not included. Successfully running an example or passing a pipeline check does not establish that the performance reported in a paper has been reproduced.

## Project Overview

| Subproject | Research Task | Main Methods and Entry Points | Documentation |
| --- | --- | --- | --- |
| [grf-analysis](grf-analysis/) | Ground Reaction Force (GRF) feature extraction, gender classification, and participant identification | 108-dimensional features and raw six-channel inputs; RF, NN, CNN, SVM, MLP, a linear regression baseline, and conditional diffusion augmentation | [GRF README](grf-analysis/README.md) |
| [pathology-reconstruction](pathology-reconstruction/) | Adjacency prediction and recovery of tile positions and orientations in pathology microscopy images | Handcrafted features, traditional machine learning, Feature MLP, PairNet, RF/PairNet fusion, and GraphSAGE/GAT reranking | [Configuration](pathology-reconstruction/config.py) · [Integrated Engine](pathology-reconstruction/pathology_reconstruction_experiment.py) |
| [rag-wag-compliance-audit](rag-wag-compliance-audit/) | Financial compliance question answering and verification using internal and external evidence | LLM Only, Static RAG, WAG Only, Hybrid RAG-WAG, and Hybrid RAG-WAG + Runtime Verifier | [RAG/WAG README](rag-wag-compliance-audit/README.md) |

## Repository Structure

```text
All-program-for-my-paper/
├── README.md
├── grf-analysis/
│   ├── main.py                              # Feature extraction and unified training
│   ├── to_juefei.py                         # 18 features per channel
│   ├── diffusion.py                        # Conditional diffusion augmentation and MLP classification
│   ├── verify.py                           # Self-contained synthetic-data pipeline checks
│   ├── requirements.txt
│   ├── sources.json                        # Original source file hashes
│   └── validation.json                     # Recorded validation scope
├── pathology-reconstruction/
│   ├── config.py                           # Default paths and experiment configuration
│   ├── data_pipeline.py                    # Data preparation, evaluation, and output
│   ├── pre_adapter.py                      # Integrated experiment engine loader
│   ├── traditional_ml_suite.py
│   ├── feature_mlp_suite.py
│   ├── pairnet_suite.py
│   ├── fusion_graphsage_suite.py
│   ├── run_full_comparison.py
│   ├── pathology_reconstruction_experiment.py
│   └── pathology_reconstruction_*.py        # Separate entry points for five model variants
└── rag-wag-compliance-audit/
    ├── src/                                # Retrieval, generation, verification, and evaluation
    ├── scripts/                            # Experiments, ablations, checks, and table exports
    ├── prompts/                            # Prompt templates
    ├── examples/                           # Fictional QA cases and evidence examples
    ├── requirements.txt
    └── source_manifest.json                # Uploaded source file inventory and SHA256 hashes
```

## Getting the Code and Setting Up the Environment

```bash
git clone --branch All-program-for-my-paper --single-branch https://github.com/yao920808-cmyk/All-program-for-my-paper.git
cd All-program-for-my-paper
```

The current branch is also named `All-program-for-my-paper`. For each subproject below, run the first command block from the repository root, then run subsequent commands from that subproject's directory.

| Subproject | Python and Dependencies | Hardware Requirements |
| --- | --- | --- |
| GRF Analysis | Existing validation records use Python 3.12; dependency versions are specified in its `requirements.txt` | Supports CPU execution |
| Pathology Image Reconstruction | Python 3.12 recommended; NumPy, pandas, scikit-learn, scikit-image, OpenCV, and PyTorch; XGBoost is optional | Traditional ML, Feature MLP, and standalone PairNet comparisons support CPU execution; integrated reconstruction entry points require a CUDA GPU |
| RAG/WAG Compliance Experiments | Python 3.10+; the core program uses only the standard library; the `.ps1` wrapper requires PowerShell | Mock demonstrations require neither a GPU nor an API key; real generation requires a configured model service |

Separate virtual environments are recommended for the subprojects. The Windows PowerShell examples for GRF and pathology analysis below invoke the virtual environment's Python directly, so activation is unnecessary. On Linux/macOS, the corresponding executable is `.venv/bin/python`. Integrated pathology reconstruction remains subject to its CUDA requirement.

## 1. GRF Gait Signal Analysis

### Features and Inputs

`main.py` provides two processing paths: extracting 18 descriptors per channel from gait CSV files to produce **108-dimensional features** across six channels, or using `--raw` to expand the raw six-channel samples into a table for training.

Raw gait CSV files must contain the following columns:

```text
leftVForce, leftAPForce, leftMLForce,
rightVForce, rightAPForce, rightMLForce
```

Participant metadata must include at least the `index` and `gender` columns. Organize the raw data in participant-specific directories, such as `data/trials/001/trial_01.csv`; the extraction program matches metadata using the participant ID in the directory structure. The output feature table contains 108 feature columns plus `gender`, `candidate`, and `source`. The raw-sample table contains the six force channels plus `gender`, `index`, and `source`.

### Installation and Pipeline Checks

```powershell
cd grf-analysis
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify.py
```

`verify.py` generates temporary synthetic data, checks both input representations, all six models, participant identification, and conditional diffusion, then removes the temporary data. This check does not require the original research data, and its outputs do not measure the performance reported in a paper.

### Using Your Own Data

The following `data/` paths are examples. Prepare the corresponding files or replace these paths with your own.

```powershell
# Extract 108-dimensional features
.\.venv\Scripts\python.exe main.py extract --root "data/trials" --metadata "data/participantData.csv" --output "results/features.csv"

# Random Forest classification
.\.venv\Scripts\python.exe main.py train --csv "results/features.csv" --model rf --output "results/rf"

# Neural network classification
.\.venv\Scripts\python.exe main.py train --csv "results/features.csv" --model nn --epochs 200 --output "results/nn"

# Expand raw six-channel samples and train a CNN
.\.venv\Scripts\python.exe main.py extract --root "data/trials" --metadata "data/participantData.csv" --raw --output "results/raw.csv"
.\.venv\Scripts\python.exe main.py train --csv "results/raw.csv" --raw --model cnn --epochs 200 --output "results/raw-cnn"

# Participant identification using the raw-sample table
.\.venv\Scripts\python.exe main.py train --csv "results/raw.csv" --raw --target index --model nn --output "results/identity"

# Conditional diffusion augmentation followed by MLP classification
.\.venv\Scripts\python.exe diffusion.py --csv "results/features.csv" --epochs 200 --steps 100 --output "results/diffusion"
```

`--model` accepts `rf`, `nn`, `cnn`, `svm`, `mlp`, and `linear`. The default classification target is `gender`; `linear` is a regression baseline that reports MSE. For an initial extraction check, add `--limit 20`. Invalid files trigger an `.errors.json` report and terminate extraction to prevent silent omissions.

The unified training entry point saves preprocessing objects, model files, `metrics.json`, and `predictions.csv`; neural network runs also produce `history.csv`. The diffusion entry point saves the diffusion model, classifier, and preprocessing parameters, and prints the final classification evaluation to the terminal. Use a separate output directory for each experiment.

For mappings between the original programs and their reorganized implementations, differences introduced by the PyTorch port, and feature formulas, see the [GRF project documentation](grf-analysis/README.md).

## 2. Pathology Microscopy Image Tile Reconstruction

### Features and Program Entry Points

This project learns adjacency relationships between image tile edges, then combines candidate-edge scoring, beam search, and simulated annealing to recover tile arrangements. Inputs can be tiles generated from complete images or existing tile directories. Evaluation covers positions, orientations, and image similarity.

| Program | Purpose |
| --- | --- |
| `traditional_ml_suite.py` | Compares four feature families—`color`, `texture`, `stain`, and `fusion`—with LR, SVM, RF, and XGB |
| `feature_mlp_suite.py` | Compares MLP adjacency classifiers using the four handcrafted feature families |
| `pairnet_suite.py` | Compares PairNet with `rgb` and `rgb_hed` inputs |
| `fusion_graphsage_suite.py` | Reads traditional ML and Feature MLP summaries, retrains the selected models, and compares their reconstruction results with PairNet, fusion, and GraphSAGE variants |
| `run_full_comparison.py` | Runs traditional ML, Feature MLP, PairNet, and fusion/GraphSAGE comparisons sequentially using the default configuration |
| `pathology_reconstruction_experiment.py` | Integrated engine for training, graph reranking, reconstruction search, batch evaluation, and result export |
| `pathology_reconstruction_random_forest.py` | Random Forest reconstruction entry point |
| `pathology_reconstruction_pairnet.py` | PairNet reconstruction entry point |
| `pathology_reconstruction_fusion.py` | RF + PairNet fusion reconstruction entry point |
| `pathology_reconstruction_graphsage.py` | Fusion scoring + GraphSAGE reranking entry point |
| `pathology_reconstruction_gat.py` | Fusion scoring + GAT reranking entry point |

### Environment and Data Preparation

This subproject does not yet include a version-pinned `requirements.txt`. The dependency list below was compiled from source imports and can be used to set up an environment; it does not constitute a fixed environment for reproducing paper results.

```powershell
cd pathology-reconstruction
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas scikit-learn scikit-image opencv-python
```

Next, install a **CUDA-enabled PyTorch build** compatible with your machine into this virtual environment. Refer to the [official PyTorch installation page](https://pytorch.org/get-started/locally/). Replace `pip` in the official command with `.\.venv\Scripts\python.exe -m pip` to install into this project's environment, then check:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Integrated reconstruction requires the final line to print `True`. To use the actual XGBoost implementation, install `xgboost` separately. If it is unavailable, the current code falls back to `HistGradientBoostingClassifier` while retaining the output label `xgb`; record the actual implementation when analyzing results.

Place complete pathology images in a directory you provide, such as `data/images/`. The program reads PNG, JPG/JPEG, BMP, and TIF/TIFF files directly from that directory. Use multiple independent images and explicitly specify the test image. If only one image is available, the code uses it for both training and testing.

If you already have tiles, specify their root directory using `--slice_root`. See functions such as `find_slice_folder_for_image` and `read_slice_metadata` in the integrated engine for directory naming and metadata parsing rules. Without existing tiles, use `--no_existing_slices` to generate tiles from complete images.

### Usage Examples

```powershell
# Traditional machine learning comparison for adjacency classification
.\.venv\Scripts\python.exe traditional_ml_suite.py --train_dir "data/images" --test_image "data/images/test.png" --no_existing_slices --rows 4 --cols 5 --output_dir "outputs/traditional_ml"

# Integrated GraphSAGE reconstruction with an explicitly fixed epoch count
.\.venv\Scripts\python.exe pathology_reconstruction_graphsage.py --train_dir "data/images" --test_image "data/images/test.png" --no_existing_slices --rows 4 --cols 5 --epochs 20 --max_epochs 20 --graph_epochs 8 --output_dir "outputs/graphsage"
```

Replace the data paths and test file with your own. The separate model entry points accept `--rows` and `--cols`, and also support `--grid_choice`: `1=3×6`, `2=4×5`, `3=5×4`, `4=5×7`, `5=10×8`, and `6=5×5`. Specifying the grid explicitly avoids interactive selection.

Default paths for the modular comparisons come from [config.py](pathology-reconstruction/config.py) and still contain local `F:` drive paths. Update the default configuration before running `run_full_comparison.py`; this script currently parses default arguments and does not forward custom command-line paths. When running `fusion_graphsage_suite.py` separately, also use `--traditional_summary` and `--feature_mlp_summary` to point to the summary CSV files generated by the preceding comparisons.

**Hardware distinction:** The standalone traditional ML, Feature MLP, and PairNet comparison scripts support CPU execution. The five model entry points that call `run_experiment`, the integrated main program, and the integrated stages of the fusion/GraphSAGE comparison require a CUDA GPU. The integrated entry point's `--cpu` option has been disabled.

### Outputs and Evaluation

The adjacency classification stage saves models, a `metrics.json` file for each combination, and CSV/JSON summaries. Metrics include Accuracy, Precision, Recall, F1, ROC-AUC, and AP.

The reconstruction stage saves `original.png`, `scrambled.png`, `reconstructed.png`, `reference_reassembly.png`, and `metrics.json`, along with model-specific weights and summary tables. Reconstruction metrics include position accuracy, adjacency accuracy, orientation accuracy, joint position-and-orientation accuracy, SSIM, pixel match ratio, and exact-match status. `reference_reassembly.png` is the reference assembly; `reconstructed.png` contains the model's predicted reconstruction.

## 3. RAG/WAG Financial Compliance Question-Answering Experiments

### Comparison Variants and Processing Pipeline

| Command-Line Variant | Method |
| --- | --- |
| `llm_only` | LLM baseline without retrieved evidence |
| `static_rag` | Retrieval-augmented generation using internal materials |
| `wag_only` | Retrieval-augmented generation using external/public-source snapshots |
| `hybrid_rag_wag` | Generation combining internal and external evidence |
| `hybrid_rag_wag_verifier` | Hybrid evidence generation followed by runtime verification |

The pipeline includes query classification, dual-channel lexical retrieval, source and date checks, conflict analysis, structured answers, runtime verification, offline scoring, and error analysis. Here, **WAG uses prepared external-source snapshots and does not perform live web searches**.

### Offline Quick Start

The following PowerShell commands use the fictional examples included in the repository and require no API key:

```powershell
cd rag-wag-compliance-audit
.\scripts\run_prepared_step6.ps1 -Mock -QaCases examples\example_qa_cases.csv -Chunks examples\example_chunks.csv -OutputStem example
python scripts/08_validate_smoke_outputs.py --model-outputs outputs/model_outputs/example_model_outputs.jsonl --output outputs/evaluation_results/example_validation.csv --expected-rows 10
```

The example contains 2 questions, each processed by 5 variants, producing 10 model-output records. Results are saved under `outputs/model_outputs/`, `outputs/verifier_outputs/`, and `outputs/evaluation_results/`. Mock mode checks the pipeline and file structure; it does not call a real model or measure research performance.

Bash users can run `bash examples/run_example.sh` from the project directory to execute generation, offline scoring, and metric aggregation.

### Using Your Own Research Data and Real Models

The QA CSV includes fields such as `qa_id`, `question_type`, `question`, `expected_answer`, `expected_evidence_id`, `conflict_type`, and `requires_human_review`. The evidence CSV includes `chunk_id`, `source_type`, `title`, and `chunk_text`. See [examples](rag-wag-compliance-audit/examples/) for the exact formats.

The complete 470-question research dataset, internal and external corpora, source snapshots, transaction data, and human review records are not included in this upload. Some scripts default to a sibling `prepared_data/experiments/` directory. Provide explicit input paths when using your own data.

Run real generation through `scripts/01_run_five_models.py` or the PowerShell wrapper. Configure the service credentials first, explicitly specify a model ID available to your account, and remove the Mock option. The source includes OpenAI, OpenRouter, and Gemini adapters. Configuration details and examples for entering keys securely are provided in the [subproject README](rag-wag-compliance-audit/README.md) and [llm_client.py](rag-wag-compliance-audit/src/llm_client.py). Real requests may incur service charges. Use a new output name when changing the model, prompts, or data; `--resume` only skips existing question/variant combinations and does not verify configuration consistency.

### Experiment Scripts

| Script | Purpose |
| --- | --- |
| `01_run_five_models.py` | Runs the five comparison variants, with support for selecting a subset |
| `02_run_verifier.py` | Performs offline scoring against reference answers and labels |
| `03_evaluate_results.py` | Aggregates automatic evaluation metrics |
| `04_run_ablation_study.py` | Runs component ablations for RAG, WAG, verification, date checks, and source ranking |
| `05_error_analysis.py` | Exports error cases |
| `06_lock_experiment_inputs.py` | Prepares data splits, input hashes, and provenance records |
| `07_build_publication_tables.py` | Generates CSV files for publication tables |
| `08_validate_smoke_outputs.py` | Checks output structure, citation IDs, and real-response metadata |
| `09_make_balanced_smoke_subset.py` | Samples a pipeline-check subset by question type |
| `10_apply_runtime_verifier.py` | Replays runtime verification on existing model outputs without making new model requests |

Answers include a compliance conclusion, supporting evidence, conflict analysis, effective-date analysis, a human review flag, and limitations. Automatic metrics include the `answer_accuracy` proxy, `retrieved_citation_match`, evidence Precision/Recall, a heuristic hallucination rate, conflict and effective-date handling, and human review flag accuracy.

## Data, Validation, and Result Interpretation

| Project | Included in the Repository | Required for the Complete Research Workflow |
| --- | --- | --- |
| GRF Analysis | Source code, pinned dependencies, original source hashes, historical validation records, and synthetic-data checks | Raw gait CSV files, participant metadata or an existing feature table, and complete training results |
| Pathology Reconstruction | Reconstruction engine, model entry points, modular comparisons, and path configuration | Original images, existing tiles and their metadata if used, a compatible environment, and experimental results |
| RAG/WAG | Source code, prompts, scripts, a source inventory, 2 fictional QA cases, and 3 fictional evidence passages | Complete QA data and corpora, traceable external snapshots, human reviews, and real-model outputs |

Interpret results within the evaluation limits of the current implementation:

- **GRF:** The unified training entry point uses a stratified row-level split with a fixed seed of 42 and training/validation/test proportions of 50%/25%/25%; the diffusion workflow uses 60%/20%/20%. The same participant may appear in different splits, so these results do not establish generalization to unseen participants. Historical features retain compatibility implementations such as the nonstandard `Sample_entropy`; see the subproject documentation for formula details.
- **Pathology reconstruction:** Training and validation are preferably split at the image level, with a tile-pair-level fallback when data are insufficient. The integrated workflow selects the best result among search candidates and progressive training rounds using reference reconstruction metrics. These metrics cannot be treated directly as strict blind-test results. Formal comparisons should fix the data splits, training epochs, search budget, and model-selection rules, and use independent test data.
- **RAG/WAG:** Retrieval currently uses lexical overlap. Although source rankings are recorded, real-generation prompts still use the per-channel retrieval lists, so ranking ablations do not directly establish generation benefits. Citation matching checks whether an ID belongs to the retrieved evidence set; it does not establish that the evidence semantically supports the conclusion. Automatic answer accuracy and hallucination checks use proxy rules and require human semantic evaluation.
- **Verification distinction:** The RAG/WAG Runtime Verifier does not use reference-answer labels, whereas the offline scorer uses expected labels. Formal experiments must define which metadata are available at runtime and prevent reference answers or evaluation labels from becoming generation hints.

[GRF validation.json](grf-analysis/validation.json) records the existing validation scope, including single-epoch neural network training and small-scale diffusion checks. It does not establish completion of the full training described in a paper. During preparation of this README, the RAG/WAG offline Mock pipeline was executed and all 10 output records passed structural checks. Full GRF and pathology training and real-LLM experiments were not run.

For reproducibility, retain the code commit ID, Python and dependency versions, random seeds, data splits, input hashes, complete parameters, and output directories. For real-LLM experiments, also retain the model ID, prompt version, and external-source snapshots. A temperature of zero does not guarantee identical responses across runs.

## Sources, Citation, and Licensing

[grf-analysis/sources.json](grf-analysis/sources.json) records original source file hashes. [rag-wag-compliance-audit/source_manifest.json](rag-wag-compliance-audit/source_manifest.json) records the uploaded source file inventory and SHA256 hashes for that subproject. When citing this work, identify the corresponding paper and the code commit actually used. Specific authorship should be determined from the original project records.

This repository does not currently include a repository-wide open-source license. Before using or redistributing code, data, or third-party materials, check the applicable authorization and licensing terms.
