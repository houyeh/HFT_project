from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


FEATURE_COLUMNS = [
    "spread_ticks",
    "queue_imbalance_calc",
    "deep_imbalance_calc",
    "top_depth",
    "depth_ratio",
    "volatility_past_50",
    "abs_return_past_50",
    "event_rate_past_50",
    "queue_imbalance_change_past",
    "normalized_spread_calc",
]


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        return self.net(x)


def matrix(df: pd.DataFrame, scaler: StandardScaler | None = None, fit: bool = False):
    x = df[FEATURE_COLUMNS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    if scaler is None:
        scaler = StandardScaler()
    x = scaler.fit_transform(x).astype(np.float32) if fit else scaler.transform(x).astype(np.float32)
    y = df["negative_ev_moderate"].astype(int).to_numpy(dtype=np.float32).reshape(-1, 1)
    return x, y, scaler


def safe_auc(y, p):
    y = y.reshape(-1)
    p = p.reshape(-1)
    if len(np.unique(y)) < 2:
        return {"pr_auc": np.nan, "roc_auc": np.nan}
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    root = Path(args.data_dir)
    (root / "models").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()} | device: {device}")

    train = pd.read_parquet(root / "train.parquet")
    val = pd.read_parquet(root / "val.parquet")
    x_train, y_train, scaler = matrix(train, fit=True)
    x_val, y_val, _ = matrix(val, scaler=scaler)

    model = TinyMLP(x_train.shape[1]).to(device)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(torch.from_numpy(x_val).to(device))).cpu().numpy()
        aucs = safe_auc(y_val, val_prob)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **aucs}
        history.append(row)
        print(row)

    pd.DataFrame(history).to_csv(root / "results" / "smoke_test_metrics.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "model_type": "tiny_mlp_smoke",
        },
        root / "models" / "smoke_mlp.pt",
    )
    (root / "results" / "smoke_test_device.json").write_text(json.dumps({"device": str(device), "cuda": torch.cuda.is_available()}, indent=2))


if __name__ == "__main__":
    main()

