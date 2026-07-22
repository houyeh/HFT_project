from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
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


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_xy(path: Path):
    df = pd.read_parquet(path)
    x = df[FEATURE_COLUMNS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y = df["negative_ev_moderate"].astype(int).to_numpy(dtype=np.float32)
    ev = df["quote_ev_moderate"].astype(float).to_numpy(dtype=np.float32)
    return df, x, y, ev


def metrics(y, score):
    out = {"pr_auc": np.nan, "roc_auc": np.nan, "spearman": np.nan}
    if len(np.unique(y)) > 1:
        out["pr_auc"] = float(average_precision_score(y, score))
        out["roc_auc"] = float(roc_auc_score(y, score))
    out["spearman"] = float(pd.Series(score).corr(pd.Series(y), method="spearman")) if len(y) else np.nan
    return out


def train_mlp(x_train, y_train, x_val, y_val, args, root: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32)
    x_val_s = scaler.transform(x_val).astype(np.float32)
    model = MLP(x_train_s.shape[1]).to(device)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_s), torch.from_numpy(y_train.reshape(-1, 1))),
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
            val_score = torch.sigmoid(model(torch.from_numpy(x_val_s).to(device))).cpu().numpy().reshape(-1)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics(y_val, val_score)}
        history.append(row)
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            print("mlp", row)
    pd.DataFrame(history).to_csv(root / "results" / "mlp_training_curve.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "mlp",
            "feature_columns": FEATURE_COLUMNS,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "input_dim": int(x_train_s.shape[1]),
        },
        root / "models" / "mlp_model.pt",
    )
    return model, scaler


def predict_mlp(model, scaler, x):
    device = next(model.parameters()).device
    xs = scaler.transform(x).astype(np.float32)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(xs).to(device))).cpu().numpy().reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--model", default="all", choices=["all", "logistic", "hgb", "mlp"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    root = Path(args.data_dir)
    (root / "models").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    train_df, x_train, y_train, _ = load_xy(root / "train.parquet")
    val_df, x_val, y_val, _ = load_xy(root / "val.parquet")
    split_data = {
        "train": (train_df, x_train, y_train),
        "val": (val_df, x_val, y_val),
        "test": load_xy(root / "test.parquet")[:3],
        "new4": load_xy(root / "new4.parquet")[:3],
    }

    selected = ["logistic", "hgb", "mlp"] if args.model == "all" else [args.model]
    rows = []

    if "logistic" in selected:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=20260701)),
            ]
        )
        model.fit(x_train, y_train)
        joblib.dump({"model": model, "model_type": "logistic", "feature_columns": FEATURE_COLUMNS}, root / "models" / "logistic_model.joblib")
        for split, (_, x, y) in split_data.items():
            score = model.predict_proba(x)[:, 1]
            rows.append({"model": "logistic", "split": split, **metrics(y, score)})

    if "hgb" in selected:
        model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0, random_state=20260701)
        model.fit(x_train, y_train)
        joblib.dump({"model": model, "model_type": "hgb", "feature_columns": FEATURE_COLUMNS}, root / "models" / "hgb_model.joblib")
        for split, (_, x, y) in split_data.items():
            score = model.predict_proba(x)[:, 1]
            rows.append({"model": "hgb", "split": split, **metrics(y, score)})

    if "mlp" in selected:
        model, scaler = train_mlp(x_train, y_train, x_val, y_val, args, root)
        for split, (_, x, y) in split_data.items():
            score = predict_mlp(model, scaler, x)
            rows.append({"model": "mlp", "split": split, **metrics(y, score)})

    out = pd.DataFrame(rows)
    out.to_csv(root / "results" / "model_family_metrics.csv", index=False)
    (root / "results" / "model_family_metadata.json").write_text(
        json.dumps({"features": FEATURE_COLUMNS, "cuda_available": torch.cuda.is_available()}, indent=2)
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

