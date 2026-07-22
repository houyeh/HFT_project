from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


DEFAULT_FEATURE_COLUMNS = [
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


class SequenceStore:
    def __init__(self, df: pd.DataFrame, feature_columns: list[str], scaler: StandardScaler):
        self.df = df.copy()
        self.feature_columns = feature_columns
        x = self.df[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        self.x = scaler.transform(x).astype(np.float32)
        self.df["_pos"] = np.arange(len(self.df), dtype=np.int64)
        self.by_session = {}
        for session, group in self.df.sort_values(["session_id", "local_row_id"]).groupby("session_id", sort=False):
            self.by_session[str(session)] = {
                "row_ids": group["local_row_id"].astype(int).to_numpy(),
                "positions": group["_pos"].astype(int).to_numpy(),
            }

    def slice_window(self, session_id: str, target_row_id: int, window: int) -> np.ndarray:
        out = np.zeros((window, len(self.feature_columns)), dtype=np.float32)
        info = self.by_session.get(str(session_id))
        if info is None:
            return out
        row_ids = info["row_ids"]
        positions = info["positions"]
        mask = (row_ids >= int(target_row_id) - int(window) + 1) & (row_ids <= int(target_row_id))
        pos = positions[mask]
        if len(pos) == 0:
            return out
        seq = self.x[pos][-window:]
        out[-len(seq) :] = seq
        return out


class TCNEvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, index_df: pd.DataFrame, split: str, store: SequenceStore, window: int):
        idx = index_df[index_df["split"].astype(str) == split].reset_index(drop=True)
        self.meta = idx.merge(df[["sample_id"]], on="sample_id", how="inner")
        self.store = store
        self.window = int(window)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        x = self.store.slice_window(str(row["session_id"]), int(row["target_row_id"]), self.window)
        return torch.from_numpy(x.T)


def load_scaler(checkpoint: dict) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=float)
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=float)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


def predict_tabular_torch(df: pd.DataFrame, checkpoint: dict, device: torch.device, model_type: str) -> np.ndarray:
    feature_columns = checkpoint.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    scaler = load_scaler(checkpoint)
    x = df[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    x = scaler.transform(x).astype(np.float32)
    if model_type == "mlp":
        model = MLP(len(feature_columns)).to(device)
    else:
        raise ValueError(f"Unsupported tabular torch model: {model_type}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            xb = torch.from_numpy(x[start : start + 4096]).to(device)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy().reshape(-1))
    torch.cuda.empty_cache()
    return np.concatenate(preds) if preds else np.asarray([])


def predict_tcn(df: pd.DataFrame, index_df: pd.DataFrame, split: str, checkpoint: dict, device: torch.device, batch_size: int) -> np.ndarray:
    feature_columns = checkpoint.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    window = int(checkpoint.get("window", 50))
    channels = int(checkpoint.get("channels", 16))
    scaler = load_scaler(checkpoint)
    store = SequenceStore(df, feature_columns, scaler)
    dataset = TCNEvalDataset(df, index_df, split, store, window)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    model = CausalTCN(len(feature_columns), channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preds = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy().reshape(-1))
    torch.cuda.empty_cache()
    return np.concatenate(preds) if preds else np.asarray([])


def predict_model(root: Path, model_path: Path, model_type: str, split: str, batch_size: int) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(root / f"{split}.parquet")
    if model_type in {"logistic", "hgb"}:
        payload = joblib.load(model_path)
        model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        feature_columns = payload.get("feature_columns", DEFAULT_FEATURE_COLUMNS) if isinstance(payload, dict) else DEFAULT_FEATURE_COLUMNS
        x = df[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        score = model.predict_proba(x)[:, 1]
        return df, score
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    inferred = checkpoint.get("model_type", model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if inferred == "tcn" or model_type == "tcn":
        index_df = pd.read_parquet(root / "sequence_index.parquet")
        return df, predict_tcn(df, index_df, split, checkpoint, device, batch_size)
    if inferred in {"mlp", "tiny_mlp_smoke"} or model_type == "mlp":
        return df, predict_tabular_torch(df, checkpoint, device, "mlp")
    raise ValueError(f"Cannot infer model type from {model_path}")


def select_threshold_on_val(df: pd.DataFrame, score: np.ndarray, max_veto_rate: float) -> dict:
    values = df["quote_ev_moderate"].astype(float).to_numpy()
    base = float(values.sum())
    finite = score[np.isfinite(score)]
    no_veto = float(np.max(finite) + 1e-6) if len(finite) else 1.000001
    thresholds = np.unique(np.concatenate([[no_veto], np.quantile(finite, np.linspace(0.50, 0.995, 100))])) if len(finite) else np.asarray([no_veto])
    rows = []
    for threshold in thresholds:
        veto = score >= float(threshold)
        veto_rate = float(veto.mean()) if len(veto) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "veto_count": int(veto.sum()),
                "veto_rate": veto_rate,
                "baseline_pnl": base,
                "vetoed_quote_ev": float(values[veto].sum()),
                "after_veto_pnl": float(base - values[veto].sum()),
                "incremental_vs_baseline": float(-values[veto].sum()),
                "constraints_pass": bool(veto_rate <= max_veto_rate),
            }
        )
    curve = pd.DataFrame(rows)
    eligible = curve[curve["constraints_pass"].astype(bool)].copy()
    if eligible.empty:
        eligible = curve.copy()
    return eligible.sort_values(["after_veto_pnl", "veto_count"], ascending=[False, True]).iloc[0].to_dict(), curve


def matched_random_after_pnls(df: pd.DataFrame, veto: np.ndarray, trials: int, seed: int) -> np.ndarray:
    work = df.reset_index(drop=True).copy()
    values = work["quote_ev_moderate"].astype(float).to_numpy()
    base = float(values.sum())
    groups = work.groupby(["side", "session_id", "spread_bucket"], observed=False, sort=True).groups
    rng = np.random.default_rng(seed)
    out = np.zeros(int(trials), dtype=float)
    for t in range(int(trials)):
        random_veto = np.zeros(len(work), dtype=bool)
        for _, idx in groups.items():
            idx_arr = np.asarray(list(idx), dtype=np.int64)
            k = int(veto[idx_arr].sum())
            if k:
                random_veto[rng.choice(idx_arr, size=min(k, len(idx_arr)), replace=False)] = True
        out[t] = float(base - values[random_veto].sum())
    return out


def evaluate_split(df: pd.DataFrame, score: np.ndarray, threshold: float, trials: int, seed: int) -> dict:
    values = df["quote_ev_moderate"].astype(float).to_numpy()
    veto = score >= float(threshold)
    base = float(values.sum())
    after = float(base - values[veto].sum())
    random = matched_random_after_pnls(df, veto, trials=trials, seed=seed)
    return {
        "baseline_pnl": base,
        "ml_veto_count": int(veto.sum()),
        "ml_veto_rate": float(veto.mean()) if len(veto) else 0.0,
        "ml_vetoed_quote_ev": float(values[veto].sum()),
        "baseline_plus_ml_veto_pnl": after,
        "ml_incremental_vs_baseline": after - base,
        "matched_random_veto_mean_pnl": float(np.mean(random)),
        "matched_random_veto_95p_pnl": float(np.quantile(random, 0.95)),
        "matched_random_veto_99p_pnl": float(np.quantile(random, 0.99)),
        "beats_random_95p": bool(after > float(np.quantile(random, 0.95))),
        "improves_baseline": bool(after > base),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", default="auto", choices=["auto", "logistic", "hgb", "mlp", "tcn"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-veto-rate", type=float, default=0.05)
    parser.add_argument("--random-trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args()

    root = Path(args.data_dir)
    (root / "results").mkdir(exist_ok=True)
    model_path = Path(args.model_path)
    model_type = args.model_type
    if model_type == "auto":
        suffix = model_path.suffix.lower()
        if suffix == ".joblib":
            name = model_path.name.lower()
            model_type = "hgb" if "hgb" in name else "logistic"
        else:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            model_type = checkpoint.get("model_type", "mlp")

    val_df, val_score = predict_model(root, model_path, model_type, "val", args.batch_size)
    selected, curve = select_threshold_on_val(val_df, val_score, args.max_veto_rate)
    curve.to_csv(root / "results" / "evaluation_validation_threshold_curve.csv", index=False)

    rows = []
    for split in ["val", "test", "new4"]:
        df, score = predict_model(root, model_path, model_type, split, args.batch_size)
        row = evaluate_split(df, score, float(selected["threshold"]), args.random_trials, args.seed)
        row.update(
            {
                "split": split,
                "model_path": str(model_path),
                "model_type": model_type,
                "threshold_selected_on": "val",
                "threshold": float(selected["threshold"]),
                "test_used_for_selection": False,
                "new4_used_for_selection": False,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(root / "results" / "ml_incremental_value_final.csv", index=False)
    print(out.to_string(index=False))
    print(f"Wrote {root / 'results' / 'ml_incremental_value_final.csv'}")


if __name__ == "__main__":
    main()

