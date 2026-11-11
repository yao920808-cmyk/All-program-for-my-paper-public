from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DEFAULT_PATHS, GRID_PRESET, PAIRNET_INPUT_MODES
from data_pipeline import ensure_output_dir, prepare_dataset, save_json, save_table, set_all_seeds
from pre_adapter import pre


def evaluate_pairnet(model, val_pairs, strip_size: int, device: str, input_mode: str, stain_normalizer=None):
    ds = pre.EdgePairDataset(val_pairs, strip_size=strip_size, input_mode=input_mode, stain_normalizer=stain_normalizer)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for xa, xb, relation_ids, y in loader:
            xa = xa.to(device)
            xb = xb.to(device)
            relation_ids = relation_ids.to(device)
            logits = model(xa, xb, relation_ids)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            y_score.extend(probs.tolist())
            y_true.extend(y.numpy().ravel().astype(int).tolist())
    from data_pipeline import compute_binary_metrics
    return compute_binary_metrics(np.asarray(y_true), np.asarray(y_score))


def run_pairnet_suite(args: argparse.Namespace):
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
    for input_mode in PAIRNET_INPUT_MODES:
        run_dir = ensure_output_dir(out_root / input_mode)
        model = pre.train_pairnet(
            dataset.train_pairs,
            dataset.val_pairs,
            strip_size=args.strip_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            input_mode=input_mode,
            stain_normalizer=dataset.stain_normalizer,
        )
        metrics = evaluate_pairnet(model, dataset.val_pairs, args.strip_size, device, input_mode, dataset.stain_normalizer)
        torch.save({"state_dict": model.state_dict(), "input_mode": input_mode}, run_dir / "pairnet.pt")
        row = {
            "stage": "B_pairnet",
            "family": "pairnet",
            "feature_mode": None,
            "model_name": f"pairnet_{input_mode}",
            "cnn_input_mode": input_mode,
            **metrics,
            "train_pair_count": int(len(dataset.train_pairs)),
            "val_pair_count": int(len(dataset.val_pairs)),
            "rows": int(args.rows),
            "cols": int(args.cols),
            "device": device,
            "output_dir": str(run_dir.resolve()),
        }
        save_json(run_dir / "metrics.json", row)
        summary_rows.append(row)
        print(f"[PairNet] {input_mode} | auc={metrics['auc']:.4f} ap={metrics['ap']:.4f} acc={metrics['accuracy']:.4f}")
    save_table(out_root / "pairnet_summary.csv", summary_rows)
    save_json(out_root / "pairnet_summary.json", summary_rows)
    return summary_rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PairNet comparison for pathology tile adjacency")
    parser.add_argument("--train_dir", type=str, default=str(DEFAULT_PATHS.train_dir))
    parser.add_argument("--slice_root", type=str, default=str(DEFAULT_PATHS.slice_root))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_PATHS.output_root / "pairnet"))
    parser.add_argument("--test_image", type=str, default=None)
    parser.add_argument("--rows", type=int, default=GRID_PRESET[0])
    parser.add_argument("--cols", type=int, default=GRID_PRESET[1])
    parser.add_argument("--strip_size", type=int, default=24)
    parser.add_argument("--negatives_per_positive", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--stain_norm", type=str, default="none", choices=("none", "reinhard"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_existing_slices", action="store_true")
    return parser


if __name__ == "__main__":
    run_pairnet_suite(build_argparser().parse_args())
