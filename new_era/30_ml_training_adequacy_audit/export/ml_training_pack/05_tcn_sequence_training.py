from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


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


class SequenceStore:
    def __init__(self, df: pd.DataFrame, scaler: StandardScaler | None = None, fit_scaler: bool = False):
        self.df = df.copy()
        x = self.df[FEATURE_COLUMNS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        self.scaler = scaler or StandardScaler()
        x = self.scaler.fit_transform(x).astype(np.float32) if fit_scaler else self.scaler.transform(x).astype(np.float32)
        self.df["_pos"] = np.arange(len(self.df), dtype=np.int64)
        self.x = x
        self.by_session = {}
        for session, group in self.df.sort_values(["session_id", "local_row_id"]).groupby("session_id", sort=False):
            self.by_session[str(session)] = {
                "row_ids": group["local_row_id"].astype(int).to_numpy(),
                "positions": group["_pos"].astype(int).to_numpy(),
            }

    def slice_window(self, session_id: str, target_row_id: int, window: int) -> np.ndarray:
        info = self.by_session.get(str(session_id))
        out = np.zeros((window, len(FEATURE_COLUMNS)), dtype=np.float32)
        if info is None:
            return out
        row_ids = info["row_ids"]
        positions = info["positions"]
        start_row = int(target_row_id) - int(window) + 1
        mask = (row_ids >= start_row) & (row_ids <= int(target_row_id))
        pos = positions[mask]
        if len(pos) == 0:
            return out
        seq = self.x[pos][-window:]
        out[-len(seq) :] = seq
        return out


class TCNDataset(Dataset):
    def __init__(self, parquet_path: Path, index_path: Path, split: str, store: SequenceStore, window: int):
        df = pd.read_parquet(parquet_path)
        idx = pd.read_parquet(index_path)
        idx = idx[idx["split"].astype(str) == split].reset_index(drop=True)
        self.meta = idx.merge(df[["sample_id", "negative_ev_moderate"]], on="sample_id", how="inner")
        self.store = store
        self.window = int(window)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        x = self.store.slice_window(str(row["session_id"]), int(row["target_row_id"]), self.window)
        y = np.float32(row["negative_ev_moderate"])
        return torch.from_numpy(x.T), torch.tensor([y], dtype=torch.float32)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class CausalTCN(nn.Module):
    def __init__(self, input_channels: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(input_channels, channels, 3, 1),
            nn.ReLU(),
            CausalConv1d(channels, channels, 3, 2),
            nn.ReLU(),
            CausalConv1d(channels, channels, 3, 4),
            nn.ReLU(),
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        h = self.net(x)
        return self.head(h[:, :, -1])


def safe_auc(y, p):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    if len(np.unique(y)) < 2:
        return {"pr_auc": np.nan, "roc_auc": np.nan}
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p))}


def evaluate(model, loader, device):
    model.eval()
    losses, ys, ps = [], [], []
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            losses.append(float(loss_fn(logits, yb).item()))
            ys.append(yb.cpu().numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    y = np.concatenate(ys).reshape(-1) if ys else np.asarray([])
    p = np.concatenate(ps).reshape(-1) if ps else np.asarray([])
    return {"loss": float(np.mean(losses)) if losses else np.nan, **safe_auc(y, p)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    root = Path(args.data_dir)
    (root / "models").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()} | device: {device}")

    train_df = pd.read_parquet(root / "train.parquet")
    val_df = pd.read_parquet(root / "val.parquet")
    train_store = SequenceStore(train_df, fit_scaler=True)
    val_store = SequenceStore(val_df, scaler=train_store.scaler)
    train_ds = TCNDataset(root / "train.parquet", root / "sequence_index.parquet", "train", train_store, args.window)
    val_ds = TCNDataset(root / "val.parquet", root / "sequence_index.parquet", "val", val_store, args.window)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    model = CausalTCN(len(FEATURE_COLUMNS), args.channels).to(device)
    y_train = train_df["negative_ev_moderate"].astype(float).to_numpy()
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        torch.cuda.empty_cache()
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            print(row)

    pd.DataFrame(history).to_csv(root / "results" / "tcn_loss_history.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "tcn",
            "feature_columns": FEATURE_COLUMNS,
            "window": int(args.window),
            "channels": int(args.channels),
            "scaler_mean": train_store.scaler.mean_.tolist(),
            "scaler_scale": train_store.scaler.scale_.tolist(),
        },
        root / "models" / "tcn_model.pt",
    )
    (root / "results" / "tcn_training_metadata.json").write_text(
        json.dumps({"device": str(device), "cuda_available": torch.cuda.is_available(), "window": args.window, "channels": args.channels}, indent=2)
    )


if __name__ == "__main__":
    main()

