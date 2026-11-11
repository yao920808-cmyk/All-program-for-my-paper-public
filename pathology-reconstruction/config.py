from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_TRAIN_DIR = Path(r"F:\yicaiyao\image_file")
DEFAULT_SLICE_ROOT = Path(r"F:\yicaiyao\cutting image")
DEFAULT_OUTPUT_ROOT = Path(r"F:\yicaiyao\comparison_outputs_modular")
DEFAULT_BASE_MODULE_CANDIDATES = [
    Path("pathology_reconstruction_experiment.py"),
    Path("pathology_reconstruction_experiment(6).py"),
    Path("pathology_reconstruction_experiment(5).py"),
    Path("pathology_reconstruction_experiment(4).py"),
]

TRADITIONAL_MODELS = ("lr", "svm", "rf", "xgb")
FEATURE_MODES = ("color", "texture", "stain", "fusion")
PAIRNET_INPUT_MODES = ("rgb", "rgb_hed")
GRID_PRESET = (4, 5)

@dataclass(frozen=True)
class ExperimentPaths:
    train_dir: Path = DEFAULT_TRAIN_DIR
    slice_root: Path = DEFAULT_SLICE_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    base_module_candidates: tuple[Path, ...] = tuple(DEFAULT_BASE_MODULE_CANDIDATES)

DEFAULT_PATHS = ExperimentPaths()
