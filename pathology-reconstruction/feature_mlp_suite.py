from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import DEFAULT_PATHS, FEATURE_MODES, GRID_PRESET
from data_pipeline import (
    compute_binary_metrics,
    ensure_output_dir,
    prepare_dataset,
    prepare_feature_arrays,
    save_json,
    save_table,
    set_all_seeds,
)


class FeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (256, 128), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_feature_mlp(x_train, y_train, x_val, y_val, epochs: int, batch_size: int, lr: float, device: str):
    model = FeatureMLP(x_train.shape[1]).to(device)
    pos = max(int(np.sum(y_train == 1)), 1)
    neg = max(int(np.sum(y_train == 0)), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1),
    )
    val_tensor = torch.from_numpy(x_val).float().to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_auc = -1.0
    best_metrics = None
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            y_score = torch.sigmoid(model(val_tensor)).cpu().numpy().ravel()
        metrics = compute_binary_metrics(y_val, y_score)
        print(f"[FeatureMLP] epoch={epoch:02d} loss={np.mean(losses):.4f} auc={metrics['auc']:.4f} ap={metrics['ap']:.4f} acc={metrics['accuracy']:.4f}")
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_metrics = metrics
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics or {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.5, "ap": 0.0}


def run_feature_mlp_suite(args: argparse.Namespace):
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
    for feature_mode in FEATURE_MODES:
        x_train, y_train = prepare_feature_arrays(dataset.train_pairs, args.strip_size, feature_mode, dataset.stain_normalizer)
        x_val, y_val = prepare_feature_arrays(dataset.val_pairs, args.strip_size, feature_mode, dataset.stain_normalizer)
        run_dir = ensure_output_dir(out_root / feature_mode)
        model, metrics = train_feature_mlp(x_train, y_train, x_val, y_val, args.epochs, args.batch_size, args.lr, device)
        torch.save({"state_dict": model.state_dict(), "input_dim": int(x_train.shape[1]), "feature_mode": feature_mode}, run_dir / "feature_mlp.pt")
        row = {
            "stage": "B_feature_mlp",
            "family": "feature_mlp",
            "feature_mode": feature_mode,
            "model_name": "mlp",
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
    save_table(out_root / "feature_mlp_summary.csv", summary_rows)
    save_json(out_root / "feature_mlp_summary.json", summary_rows)
    return summary_rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feature-vector MLP comparison for pathology tile adjacency")
    parser.add_argument("--train_dir", type=str, default=str(DEFAULT_PATHS.train_dir))
    parser.add_argument("--slice_root", type=str, default=str(DEFAULT_PATHS.slice_root))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_PATHS.output_root / "feature_mlp"))
    parser.add_argument("--test_image", type=str, default=None)
    parser.add_argument("--rows", type=int, default=GRID_PRESET[0])
    parser.add_argument("--cols", type=int, default=GRID_PRESET[1])
    parser.add_argument("--strip_size", type=int, default=24)
    parser.add_argument("--negatives_per_positive", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--stain_norm", type=str, default="none", choices=("none", "reinhard"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_existing_slices", action="store_true")
    return parser


if __name__ == "__main__":
    run_feature_mlp_suite(build_argparser().parse_args())
