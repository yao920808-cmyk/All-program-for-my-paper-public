from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
from sklearn.ensemble import HistGradientBoostingClassifier

from config import DEFAULT_PATHS, FEATURE_MODES, GRID_PRESET, TRADITIONAL_MODELS
from data_pipeline import (
    ensure_output_dir,
    evaluate_pair_classifier,
    prepare_dataset,
    prepare_feature_arrays,
    save_json,
    save_pickle,
    save_table,
)


def build_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "lr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=None, class_weight="balanced")),
        ])
    if model_name == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight="balanced")),
        ])
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            random_state=42,
            n_jobs=1,
            class_weight="balanced",
        )
    if model_name == "xgb":
        if XGBClassifier is not None:
            return XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=1,
            )
        # Fallback when xgboost is not installed. The model name is kept as "xgb"
        # so downstream summaries remain stable.
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        )

    raise ValueError(f"Unsupported traditional model: {model_name}")


def run_traditional_suite(args: argparse.Namespace):
    out_root = ensure_output_dir(args.output_dir)
    dataset = prepare_dataset(
        train_dir=args.train_dir,
        test_image=args.test_image,
        rows=args.rows,
        cols=args.cols,
        negatives_per_positive=args.negatives_per_positive,
        val_ratio=args.val_ratio,
        use_existing_slices=not args.no_existing_slices,
        slice_root=args.slice_root,
        stain_norm=args.stain_norm,
    )
    summary_rows = []
    for feature_mode in FEATURE_MODES:
        x_train, y_train = prepare_feature_arrays(dataset.train_pairs, args.strip_size, feature_mode, dataset.stain_normalizer)
        x_val, y_val = prepare_feature_arrays(dataset.val_pairs, args.strip_size, feature_mode, dataset.stain_normalizer)
        for model_name in TRADITIONAL_MODELS:
            run_dir = ensure_output_dir(out_root / feature_mode / model_name)
            model = build_model(model_name)
            model.fit(x_train, y_train)
            metrics = evaluate_pair_classifier(model, x_val, y_val)
            row = {
                "stage": "A_traditional_ml",
                "family": "traditional_ml",
                "feature_mode": feature_mode,
                "model_name": model_name,
                **metrics,
                "train_pair_count": int(len(dataset.train_pairs)),
                "val_pair_count": int(len(dataset.val_pairs)),
                "rows": int(args.rows),
                "cols": int(args.cols),
                "output_dir": str(run_dir.resolve()),
            }
            save_pickle(run_dir / "model.pkl", model)
            save_json(run_dir / "metrics.json", row)
            summary_rows.append(row)
            print(f"[Traditional ML] {feature_mode} + {model_name} | auc={metrics['auc']:.4f} ap={metrics['ap']:.4f} acc={metrics['accuracy']:.4f}")
    save_table(out_root / "traditional_ml_summary.csv", summary_rows)
    save_json(out_root / "traditional_ml_summary.json", summary_rows)
    return summary_rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traditional ML comparison for pathology tile adjacency")
    parser.add_argument("--train_dir", type=str, default=str(DEFAULT_PATHS.train_dir))
    parser.add_argument("--slice_root", type=str, default=str(DEFAULT_PATHS.slice_root))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_PATHS.output_root / "traditional_ml"))
    parser.add_argument("--test_image", type=str, default=None)
    parser.add_argument("--rows", type=int, default=GRID_PRESET[0])
    parser.add_argument("--cols", type=int, default=GRID_PRESET[1])
    parser.add_argument("--strip_size", type=int, default=24)
    parser.add_argument("--negatives_per_positive", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--stain_norm", type=str, default="none", choices=("none", "reinhard"))
    parser.add_argument("--no_existing_slices", action="store_true")
    return parser


if __name__ == "__main__":
    run_traditional_suite(build_argparser().parse_args())
