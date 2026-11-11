from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import DEFAULT_PATHS, GRID_PRESET
from data_pipeline import (
    GenericEdgeScorer,
    ensure_output_dir,
    prepare_dataset,
    prepare_feature_arrays,
    run_reconstruction_with_scorer,
    save_json,
    save_pickle,
    save_table,
    set_all_seeds,
)
from feature_mlp_suite import FeatureMLP, train_feature_mlp
from pairnet_suite import evaluate_pairnet
from pre_adapter import pre
from traditional_ml_suite import build_model


def load_best_row(summary_csv: str | Path, priority_metric: str = "auc") -> dict:
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError(f"No rows found in {summary_csv}")
    sort_cols = [priority_metric, "ap", "accuracy"]
    available = [c for c in sort_cols if c in df.columns]
    best = df.sort_values(available, ascending=[False] * len(available)).iloc[0]
    return best.to_dict()


def rebuild_best_traditional(best_row: dict, dataset, strip_size: int):
    feature_mode = best_row["feature_mode"]
    model_name = best_row["model_name"]
    x_train, y_train = prepare_feature_arrays(dataset.train_pairs, strip_size, feature_mode, dataset.stain_normalizer)
    model = build_model(model_name)
    model.fit(x_train, y_train)
    return model, feature_mode, model_name


def rebuild_best_feature_mlp(best_row: dict, dataset, strip_size: int, device: str):
    feature_mode = best_row["feature_mode"]
    x_train, y_train = prepare_feature_arrays(dataset.train_pairs, strip_size, feature_mode, dataset.stain_normalizer)
    x_val, y_val = prepare_feature_arrays(dataset.val_pairs, strip_size, feature_mode, dataset.stain_normalizer)
    model, metrics = train_feature_mlp(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=30,
        batch_size=256,
        lr=1e-3,
        device=device,
    )
    return model, feature_mode, metrics


def run_fusion_graphsage_suite(args: argparse.Namespace):
    set_all_seeds()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
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

    # C1. best traditional ML -> reconstruction
    traditional_best = load_best_row(args.traditional_summary)
    traditional_model, traditional_feature_mode, traditional_model_name = rebuild_best_traditional(traditional_best, dataset, args.strip_size)
    traditional_run_dir = ensure_output_dir(out_root / "best_traditional_ml")
    save_pickle(traditional_run_dir / "model.pkl", traditional_model)
    traditional_metrics = run_reconstruction_with_scorer(
        GenericEdgeScorer(traditional_model, traditional_feature_mode, dataset.stain_normalizer),
        dataset.test_image,
        traditional_run_dir,
        rows=args.rows,
        cols=args.cols,
        slice_root=args.slice_root,
        use_existing_slices=not args.no_existing_slices,
        strip_size=args.strip_size,
        initial_pair_candidates=args.initial_pair_candidates,
        beam_width=args.beam_width,
        beam_branch_factor=args.beam_branch_factor,
        border_bonus_weight=args.border_bonus_weight,
        anneal_steps=args.anneal_steps,
        no_flips=args.no_flips,
    )
    traditional_metrics.update({
        "stage": "C_fusion_compare",
        "variant": "best_traditional_ml",
        "model_name": traditional_model_name,
        "feature_mode": traditional_feature_mode,
        "selection_source": str(Path(args.traditional_summary).resolve()),
    })
    save_json(traditional_run_dir / "metrics.json", traditional_metrics)
    summary_rows.append(traditional_metrics)

    # C2. best feature-MLP -> reconstruction
    feature_mlp_best = load_best_row(args.feature_mlp_summary)
    mlp_model, mlp_feature_mode, mlp_pair_metrics = rebuild_best_feature_mlp(feature_mlp_best, dataset, args.strip_size, device)
    mlp_run_dir = ensure_output_dir(out_root / "best_feature_mlp")
    torch.save({"state_dict": mlp_model.state_dict(), "feature_mode": mlp_feature_mode}, mlp_run_dir / "feature_mlp.pt")
    mlp_metrics = run_reconstruction_with_scorer(
        GenericEdgeScorer(mlp_model, mlp_feature_mode, dataset.stain_normalizer),
        dataset.test_image,
        mlp_run_dir,
        rows=args.rows,
        cols=args.cols,
        slice_root=args.slice_root,
        use_existing_slices=not args.no_existing_slices,
        strip_size=args.strip_size,
        initial_pair_candidates=args.initial_pair_candidates,
        beam_width=args.beam_width,
        beam_branch_factor=args.beam_branch_factor,
        border_bonus_weight=args.border_bonus_weight,
        anneal_steps=args.anneal_steps,
        no_flips=args.no_flips,
    )
    mlp_metrics.update({
        "stage": "C_fusion_compare",
        "variant": "best_feature_mlp",
        "model_name": "feature_mlp",
        "feature_mode": mlp_feature_mode,
        "selection_source": str(Path(args.feature_mlp_summary).resolve()),
        "pairwise_auc": mlp_pair_metrics["auc"],
        "pairwise_ap": mlp_pair_metrics["ap"],
        "pairwise_accuracy": mlp_pair_metrics["accuracy"],
    })
    save_json(mlp_run_dir / "metrics.json", mlp_metrics)
    summary_rows.append(mlp_metrics)

    # C3-5 use existing integrated engine for PairNet / RF+PairNet / RF+PairNet+GraphSAGE
    base_args = pre.build_run_args(
        train_dir=str(args.train_dir),
        test_image=str(dataset.test_image),
        output_dir=str(out_root),
        rows=args.rows,
        cols=args.cols,
        strip_size=args.strip_size,
        negatives_per_positive=args.negatives_per_positive,
        epochs=args.epochs,
        epoch_step=args.epoch_step,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rf_jobs=1,
        rf_weight=args.rf_weight,
        graph_weight=args.graph_weight,
        graph_model_type="graphsage",
        graph_candidate_k=args.graph_candidate_k,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_layers=args.graph_layers,
        graph_epochs=args.graph_epochs,
        graph_lr=args.graph_lr,
        graph_hard_negative_ratio=args.graph_hard_negative_ratio,
        initial_pair_candidates=args.initial_pair_candidates,
        beam_width=args.beam_width,
        beam_branch_factor=args.beam_branch_factor,
        border_bonus_weight=args.border_bonus_weight,
        anneal_steps=args.anneal_steps,
        feature_mode="fusion",
        cnn_input_mode=args.cnn_input_mode,
        val_ratio=args.val_ratio,
        use_existing_slices=not args.no_existing_slices,
        slice_root=str(args.slice_root),
        no_flips=args.no_flips,
        cpu=args.cpu,
    )

    for variant in ("pairnet", "fusion", "graphsage"):
        metrics = pre.run_model_variant(base_args, variant, output_dir=out_root / variant)
        row = {"stage": "C_fusion_compare", "variant": variant, **metrics}
        summary_rows.append(row)

    save_table(out_root / "fusion_graphsage_comparison.csv", summary_rows)
    save_json(out_root / "fusion_graphsage_comparison.json", summary_rows)
    return summary_rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fusion and GraphSAGE reconstruction comparison")
    parser.add_argument("--train_dir", type=str, default=str(DEFAULT_PATHS.train_dir))
    parser.add_argument("--slice_root", type=str, default=str(DEFAULT_PATHS.slice_root))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_PATHS.output_root / "fusion_graphsage"))
    parser.add_argument("--traditional_summary", type=str, default=str(DEFAULT_PATHS.output_root / "traditional_ml" / "traditional_ml_summary.csv"))
    parser.add_argument("--feature_mlp_summary", type=str, default=str(DEFAULT_PATHS.output_root / "feature_mlp" / "feature_mlp_summary.csv"))
    parser.add_argument("--test_image", type=str, default=None)
    parser.add_argument("--rows", type=int, default=GRID_PRESET[0])
    parser.add_argument("--cols", type=int, default=GRID_PRESET[1])
    parser.add_argument("--strip_size", type=int, default=24)
    parser.add_argument("--negatives_per_positive", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--stain_norm", type=str, default="none", choices=("none", "reinhard"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epoch_step", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--rf_weight", type=float, default=0.5)
    parser.add_argument("--graph_weight", type=float, default=0.5)
    parser.add_argument("--graph_candidate_k", type=int, default=12)
    parser.add_argument("--graph_hidden_dim", type=int, default=128)
    parser.add_argument("--graph_layers", type=int, default=3)
    parser.add_argument("--graph_epochs", type=int, default=20)
    parser.add_argument("--graph_lr", type=float, default=1e-3)
    parser.add_argument("--graph_hard_negative_ratio", type=float, default=8.0)
    parser.add_argument("--cnn_input_mode", type=str, default="rgb_hed", choices=("rgb", "rgb_hed"))
    parser.add_argument("--initial_pair_candidates", type=int, default=96)
    parser.add_argument("--beam_width", type=int, default=96)
    parser.add_argument("--beam_branch_factor", type=int, default=20)
    parser.add_argument("--border_bonus_weight", type=float, default=0.25)
    parser.add_argument("--anneal_steps", type=int, default=6000)
    parser.add_argument("--no_flips", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_existing_slices", action="store_true")
    return parser


if __name__ == "__main__":
    run_fusion_graphsage_suite(build_argparser().parse_args())
