from __future__ import annotations

import inspect
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, precision_recall_fscore_support, roc_auc_score

from config import DEFAULT_PATHS, FEATURE_MODES, GRID_PRESET
from pre_adapter import pre

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class PreparedDataset:
    train_images: list[Path]
    val_images: list[Path]
    test_image: Path
    train_pairs: list[Any]
    val_pairs: list[Any]
    stain_normalizer: Any | None
    split_metadata: dict[str, Any]


class GenericEdgeScorer:
    def __init__(self, model: Any, feature_mode: str, stain_normalizer: Any | None = None):
        self.model = model
        self.feature_mode = feature_mode
        self.stain_normalizer = stain_normalizer

    def _predict_scores(self, strip_pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        if not strip_pairs:
            return np.empty((0,), dtype=np.float32)
        x = np.asarray([
            pre.pair_feature(a, b, feature_mode=self.feature_mode, stain_normalizer=self.stain_normalizer)
            for a, b in strip_pairs
        ], dtype=np.float32)

        # scikit-learn style models
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1].astype(np.float32)
        if hasattr(self.model, "decision_function"):
            raw = self.model.decision_function(x)
            raw = np.asarray(raw, dtype=np.float32)
            return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)

        # torch FeatureMLP style models
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            device = next(self.model.parameters()).device
            with torch.no_grad():
                logits = self.model(torch.from_numpy(x).float().to(device))
                probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            return probs.astype(np.float32)

        # fallback for simple estimators exposing predict()
        if hasattr(self.model, "predict"):
            probs = self.model.predict(x)
            return np.asarray(probs, dtype=np.float32).ravel()

        raise TypeError(
            "GenericEdgeScorer requires a scikit-learn estimator, a torch.nn.Module, "
            "or an object exposing predict_proba/decision_function/predict."
        )

    def score_components_batch(self, strip_pairs: Sequence[tuple[np.ndarray, np.ndarray]], relation: str, nn_batch_size: int = 512) -> Dict[str, np.ndarray]:
        scores = self._predict_scores(strip_pairs)
        return {"rf": scores, "nn": np.full_like(scores, 0.5), "fused": scores}

    def score_batch(self, strip_pairs: Sequence[tuple[np.ndarray, np.ndarray]], relation: str, nn_batch_size: int = 512) -> np.ndarray:
        return self._predict_scores(strip_pairs)


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_train_test(train_dir: str | Path | None = None, test_image: str | Path | None = None) -> tuple[list[Path], Path]:
    return pre.resolve_runtime_inputs(str(train_dir) if train_dir is not None else None, str(test_image) if test_image is not None else None)


def build_stain_normalizer_from_train(train_images: Sequence[str | Path], method: str = "none"):
    if hasattr(pre, "build_stain_normalizer"):
        return pre.build_stain_normalizer(train_images, method=method)
    return None


def prepare_dataset(
    train_dir: str | Path = DEFAULT_PATHS.train_dir,
    test_image: str | Path | None = None,
    rows: int = GRID_PRESET[0],
    cols: int = GRID_PRESET[1],
    negatives_per_positive: int = 4,
    val_ratio: float = 0.25,
    use_existing_slices: bool = True,
    slice_root: str | Path = DEFAULT_PATHS.slice_root,
    stain_norm: str = "none",
) -> PreparedDataset:
    train_images, resolved_test = resolve_train_test(train_dir, test_image)
    train_pairs, val_pairs, split_metadata = pre.build_pair_sets_for_experiment(
        train_images,
        rows=rows,
        cols=cols,
        negatives_per_positive=negatives_per_positive,
        use_existing_slices=use_existing_slices,
        slice_root=str(slice_root),
        val_ratio=val_ratio,
    )
    stain_normalizer = build_stain_normalizer_from_train(train_images if train_images else [resolved_test], method=stain_norm)
    return PreparedDataset(
        train_images=[Path(p) for p in split_metadata.get("train_split_images", [])],
        val_images=[Path(p) for p in split_metadata.get("val_split_images", [])],
        test_image=Path(resolved_test),
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        stain_normalizer=stain_normalizer,
        split_metadata=split_metadata,
    )


def prepare_feature_arrays(pair_samples: Sequence[Any], strip_size: int, feature_mode: str, stain_normalizer: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    return pre.prepare_ml_arrays(
        pair_samples,
        strip_size=strip_size,
        feature_mode=feature_mode,
        stain_normalizer=stain_normalizer,
    )


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float32)
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    has_mixed = len(set(y_true.tolist())) > 1
    auc = roc_auc_score(y_true, y_score) if has_mixed else 0.5
    ap = average_precision_score(y_true, y_score) if has_mixed else float(np.mean(y_true))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "ap": float(ap),
    }


def evaluate_pair_classifier(model: Any, x_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_val)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(x_val)
        y_score = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float32)))
    else:
        y_score = np.asarray(model.predict(x_val), dtype=np.float32)
    return compute_binary_metrics(y_val, y_score)


def save_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_pickle(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)
    return path


def save_table(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _call_reconstruct(pieces, rows: int, cols: int, scorer: Any, strip_size: int, initial_pair_candidates: int, beam_width: int, beam_branch_factor: int, border_bonus_weight: float, anneal_steps: int, fixed_orientation: bool):
    sig = inspect.signature(pre.reconstruct_greedy)
    kwargs = {
        "pieces": pieces,
        "rows": rows,
        "cols": cols,
        "scorer": scorer,
        "strip_size": strip_size,
        "initial_pair_candidates": initial_pair_candidates,
        "anneal_steps": anneal_steps,
    }
    if "beam_width" in sig.parameters:
        kwargs["beam_width"] = beam_width
    if "beam_branch_factor" in sig.parameters:
        kwargs["beam_branch_factor"] = beam_branch_factor
    if "border_bonus_weight" in sig.parameters:
        kwargs["border_bonus_weight"] = border_bonus_weight
    if "fixed_orientation" in sig.parameters:
        kwargs["fixed_orientation"] = fixed_orientation
    result = pre.reconstruct_greedy(**kwargs)
    if len(result) == 3:
        placement_ids, placement_tiles, search_metadata = result
    else:
        placement_ids, placement_tiles = result
        search_metadata = {}
    return placement_ids, placement_tiles, search_metadata


def run_reconstruction_with_scorer(
    scorer: Any,
    test_image: str | Path,
    output_dir: str | Path,
    rows: int = GRID_PRESET[0],
    cols: int = GRID_PRESET[1],
    slice_root: str | Path = DEFAULT_PATHS.slice_root,
    use_existing_slices: bool = True,
    strip_size: int = 24,
    initial_pair_candidates: int = 96,
    beam_width: int = 96,
    beam_branch_factor: int = 20,
    border_bonus_weight: float = 0.25,
    anneal_steps: int = 6000,
    no_flips: bool = True,
) -> dict[str, Any]:
    out_dir = ensure_output_dir(output_dir)
    reference_tiles_data = pre.build_reference_tiles(
        test_image,
        rows,
        cols,
        prefer_existing_slices=use_existing_slices,
        slice_root=str(slice_root),
    )
    if len(reference_tiles_data) == 3:
        reference_tiles, reference_pieces, reference_source = reference_tiles_data
    else:
        reference_tiles, reference_source = reference_tiles_data
        reference_pieces = None

    reference_tile_grid = pre.gridify_tiles(reference_tiles, rows, cols)
    reference_img = pre.merge_tiles(reference_tile_grid)
    reference_visual_img = pre.build_visual_grid(reference_tile_grid)

    if reference_pieces is not None:
        pieces = reference_pieces
        fixed_orientation = False
        scrambled_img = pre.build_visual_grid(pre.gridify_tiles([p.image for p in pieces], rows, cols))
    else:
        set_all_seeds(SEED)
        pieces, _ = pre.scramble_tile_list(reference_tiles, rows, cols, allow_flips=not no_flips)
        fixed_orientation = bool(no_flips)
        scrambled_img = pre.build_visual_grid(pre.gridify_tiles([p.image for p in pieces], rows, cols))

    placement_ids, placement_tiles, search_metadata = _call_reconstruct(
        pieces,
        rows,
        cols,
        scorer,
        strip_size,
        initial_pair_candidates,
        beam_width,
        beam_branch_factor,
        border_bonus_weight,
        anneal_steps,
        fixed_orientation,
    )

    reconstructed_img, metrics = pre.evaluate_reconstruction_result(
        placement_ids,
        placement_tiles,
        pieces,
        rows,
        cols,
        reference_img,
        reference_visual_img,
    )
    pre.save_rgb_image(out_dir / "original.png", reference_img)
    pre.save_rgb_image(out_dir / "scrambled.png", scrambled_img)
    pre.save_rgb_image(out_dir / "reconstructed.png", reconstructed_img)
    pre.save_rgb_image(out_dir / "reference_reassembly.png", reference_img)
    result = {
        "test_image": str(Path(test_image).resolve()),
        "reference_source": reference_source,
        "rows": rows,
        "cols": cols,
        "strip_size": strip_size,
        "fixed_orientation": fixed_orientation,
        **metrics,
        "search_metadata": search_metadata,
        "output_dir": str(out_dir.resolve()),
    }
    save_json(out_dir / "metrics.json", result)
    return result
