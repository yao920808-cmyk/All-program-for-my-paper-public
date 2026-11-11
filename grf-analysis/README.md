# GRF Feature Extraction and Classification Experiments

This project was assembled from the local `GRF.docx` manuscript, the `Journal` directory, and the OneDrive source archives. The unified entry point is `main.py`. The original data files remain unchanged. This is a runnable, reorganized port of the supplied programs rather than a verbatim source backup; the accuracy figures in the manuscript have not been independently reproduced. Authorship should be determined from the original project records, since the manuscript and source comments alone do not establish the author of every code segment.

## Setup and Quick Verification

Use Python 3.12 and run the following commands from this directory:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python verify.py
```

`verify.py` generates temporary synthetic data and checks both input representations, all six models, subject identification, and conditional diffusion. It removes the temporary data after verification. Metrics from these synthetic data do not measure research performance. The programs support CPU execution without a GPU. Dependency versions match the environment used for validation.

## Running on the Existing Dataset

The examples below use the original local data paths. Adjust them to match your dataset location. The Chinese characters in the paths are part of the existing directory name.

```powershell
python main.py train --csv "F:/文章数据/Juefei_Yuan_journal/Journal/merged_features_6x18_with_gender_and_name.csv" --model nn --epochs 200 --output results/nn
python main.py train --csv "F:/文章数据/Juefei_Yuan_journal/Journal/merged_features_6x18_with_gender_and_name.csv" --model cnn --epochs 200 --output results/cnn
python main.py train --csv "F:/文章数据/Juefei_Yuan_journal/Journal/merged_features_6x18_with_gender_and_name.csv" --model rf --output results/rf
```

The `--model` option accepts `rf`, `nn`, `cnn`, `svm`, `mlp`, and `linear`. The `linear` option provides the regression baseline corresponding to the original `LinearCode.py` and reports mean squared error (MSE); its continuous predictions are not treated as class labels. Each output directory contains preprocessing objects, model parameters, metrics, and predictions. Neural network runs also save training logs. Use a separate output directory for each experiment. The `--limit` option reads the first N rows for quick checks; this subset may contain only one class.

```powershell
python main.py extract --root "F:/文章数据/Juefei_Yuan_journal/Journal/gaitPatternsInCSVFiles/gaitPatternsInCSVFiles" --metadata "F:/文章数据/Juefei_Yuan_journal/Journal/gaitPatternsInCSVFiles/participantData.csv" --output results/features.csv
python main.py extract --root "F:/文章数据/Juefei_Yuan_journal/Journal/gaitPatternsInCSVFiles/gaitPatternsInCSVFiles" --metadata "F:/文章数据/Juefei_Yuan_journal/Journal/gaitPatternsInCSVFiles/participantData.csv" --raw --output results/raw.csv
python main.py train --csv results/raw.csv --raw --model cnn --output results/raw-cnn
python main.py train --csv results/raw.csv --raw --target index --model nn --output results/identity
python diffusion.py --csv "F:/文章数据/Juefei_Yuan_journal/Journal/merged_features_6x18_with_gender_and_name.csv" --epochs 200 --steps 100 --output results/diffusion
```

Expanding the raw waveforms can produce a large dataset. First use `extract --limit 20` to check paths and file structure. Metadata must contain `index` and `gender`; records are matched by subject ID rather than directory enumeration order. Invalid files are recorded in an `.errors.json` report and cause extraction to fail, preventing silent data loss. The existing feature table contains 13,702 rows, 108 features, and three metadata columns: `gender`, `candidate`, and `source`. Datasets, manuscripts, personal documents, and large archives are not included in the repository. Users must supply their own data.

## Mapping the Original Programs

| Original program | Reorganized implementation |
|---|---|
| `Journal/Training.py` | `main.py train` with `nn`, `cnn`, or `rf`; the main training program for the additional feature experiments |
| `Journal/read_and_save_data.py` | `main.py extract`; creates the unified table of 108 features |
| `Journal/to_juefei.py` | `to_juefei.py`; retains only the functions required for the 108 features and removes unrelated imports |
| `Journal/SVM.py`, `SVM2.py` | `main.py train --model svm`; a shared RBF baseline with fixed `C=10`, without the original automatic parameter search |
| `Journal/MLP.py`, `MLP2.py` | `main.py train --model mlp`; a shared model with two hidden layers of 128 units, which differs from the two original architectures |
| `Journal/transformer.py` | `diffusion.py`; implements conditional diffusion augmentation followed by an MLP, rather than a Transformer |
| Archived `ReadData_And_SaveToOneFile.py` | `main.py extract --raw` |
| Archived `RandomForestCode_gender.py` | `main.py train --raw --model rf` |
| Archived `NeuralNetworkCode_gender.py` | `main.py train --raw --model nn` |
| Archived `ConvolutionalNeuralNetworkCode_gender.py` | `main.py train --raw --model cnn` |
| Archived `NeuralNetworkCode_index.py` | `main.py train --raw --target index --model nn` |
| Archived `LinearCode.py` | `main.py train --raw --model linear` |

Source file hashes are recorded in `sources.json`. Duplicate files from `source code_old`, IDE caches, historical results, and unrelated materials were excluded. All original files remain in their original local locations.

## Methodology and Compatibility

The additional `nn` model retains hidden layer widths of 256, 128, 64, and 32, with a dropout rate of 0.2. The `cnn` model retains convolutional filter counts of 64, 128, and 64, batch normalization, and dropout rates of 0.3, 0.3, and 0.2. Both were ported to PyTorch and output logits for cross-entropy loss. Changes in framework, random initialization, and the consolidation of baseline architectures can affect results. Reproducing the original TensorFlow results is not guaranteed.

The unified training pipeline uses a stratified trial-level split of 50% training, 25% validation, and 25% testing, with random seed 42. Missing-value imputation and standardization are fitted on the training set only. Neural network checkpoints are selected using the validation set, and subject identification labels are encoded dynamically. The conditional diffusion program preserves the original implementation, starting with an 80/20 split and reserving 25% of the training portion for validation, giving a final 60/20/20 split. Synthetic data are generated using only the training set. Standardization parameters are saved, checkpoints are selected using validation performance, and the test set is used only for final evaluation.

For compatibility with the existing feature table, the 18 features per channel retain their original names and formulas: 13 basic statistics, the numerical mean of the frequency-domain feature dictionary, the original `Sample_entropy`, the original Higuchi implementation, impulse integrated over sample indices, and an initial loading rate fitted over sample indices. Frequency-domain calculations assume 100 Hz; impulse and loading rate are not converted to seconds. The original `Sample_entropy` compares only adjacent windows and is not the standard estimator based on all eligible window pairs. It is retained for historical compatibility and should not be interpreted as a standard sample entropy implementation. Combining several frequency-domain quantities into one column also warrants further methodological review. NumPy's `trapz` was replaced with `trapezoid` for compatibility with newer versions.

The same participant may appear in both training and test sets. Trial-level accuracy therefore does not establish generalization to unseen participants. Figures such as 99.12% in the manuscript are historical results; the 200-epoch experiments have not been reproduced in this validation. `validation.json` records the validation scope: the core models used the full existing feature table, neural networks ran for one epoch, and the diffusion check used 256 real records with four sampling steps.
