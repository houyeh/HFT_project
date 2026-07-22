from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
BASE_FEATURES = [
    "obi_z",
    "queue_imbalance_z",
    "weighted_mid_deviation_z",
    "trade_intensity_short_z",
    "deep_obi_z",
    "queue_imbalance_change_z",
    "normalized_spread_z",
    "cancel_pressure_short_z",
]


def _import_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _past_z(series: pd.Series, window: int = 2000) -> pd.Series:
    past = series.shift(1)
    mean = past.rolling(window=window, min_periods=100).mean()
    std = past.rolling(window=window, min_periods=100).std().replace(0.0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10, 10)


def _build_augmented_labels(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values(["session_id", "local_row_id"]).reset_index(drop=True)
    parts = []
    for _, g in df.groupby("session_id", sort=True):
        g = g.copy()
        dt = g["timestamp"].diff().replace(0, np.nan)
        inter_ms = (dt / 1_000_000.0).replace([np.inf, -np.inf], np.nan).fillna(method="bfill").fillna(0.0)
        rate = (1_000_000_000.0 / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mid_move_ticks = (g["mid_price"].diff().abs() / g["tick_size"].replace(0, np.nan)).fillna(0.0)
        excite = mid_move_ticks.shift(1).fillna(0.0).ewm(alpha=0.05, adjust=False).mean()
        g["event_interarrival_z"] = _past_z(np.log1p(inter_ms)).astype(np.float32)
        g["event_rate_z"] = _past_z(np.log1p(rate)).astype(np.float32)
        g["causal_excite_z"] = _past_z(excite).astype(np.float32)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _eval_side(train_mod, labels: pd.DataFrame, model_dir: Path, side: str, features: list[str]) -> dict:
    train_mod.Z_FEATURES = features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((model_dir / f"{side}_model" / "model_config.json").read_text(encoding="utf-8"))
    model = train_mod.HLSFriendlyTCNClassifier(
        num_inputs=len(features),
        num_classes=1,
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
        seq_length=int(cfg["window_size"]),
    ).to(device)
    model.load_state_dict(torch.load(model_dir / f"{side}_model" / "best_model.pt", map_location=device))
    data = train_mod._build_split_data(labels, split="test", side=side, stride=1, window_size=int(cfg["window_size"]))
    pred = train_mod._predict(
        model,
        data,
        criterion=nn.BCEWithLogitsLoss(),
        device=device,
        window_size=int(cfg["window_size"]),
        batch_size=4096,
    )
    y = pred["y_true"]
    p = pred["probs"]
    return {"test_pr_auc": float(average_precision_score(y, p)), "test_roc_auc": float(roc_auc_score(y, p))}


def run(results_dir: Path, epochs: int) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    ablation_root = results_dir / "feature_ablation_runs"
    ablation_root.mkdir(parents=True, exist_ok=True)

    train_mod = _import_script(ROOT / "04_train_binary_model.py", "binary_v2_train_mod")
    select_mod = _import_script(ROOT / "05_select_thresholds.py", "binary_v2_select_mod")
    backtest_mod = _import_script(ROOT / "06_backtest_binary_cancel.py", "binary_v2_backtest_mod")

    labels = _build_augmented_labels(OUTPUTS / "binary_labels_v2.parquet")
    labels["event_id"] = np.arange(len(labels), dtype=np.int64)
    augmented_path = ablation_root / "binary_labels_v2_ablation_features.parquet"
    labels.to_parquet(augmented_path, index=False)

    feature_sets = {
        "F8_baseline_existing": BASE_FEATURES,
        "F8_plus_event_rate": BASE_FEATURES + ["event_interarrival_z", "event_rate_z"],
        "F8_plus_causal_excite": BASE_FEATURES + ["causal_excite_z"],
    }

    rows = []
    base_test = pd.read_csv(OUTPUTS / "test_backtest_summary.csv").iloc[0]
    base_metrics = {
        "bid": pd.read_csv(OUTPUTS / "metrics_bid.csv").query("split == 'val'").iloc[0],
        "ask": pd.read_csv(OUTPUTS / "metrics_ask.csv").query("split == 'val'").iloc[0],
    }
    base_bid_test = _eval_side(train_mod, labels, OUTPUTS / "model_weighted_bce", "bid", BASE_FEATURES)
    base_ask_test = _eval_side(train_mod, labels, OUTPUTS / "model_weighted_bce", "ask", BASE_FEATURES)
    rows.append(
        {
            "feature_set": "F8_baseline_existing",
            "features": ",".join(BASE_FEATURES),
            "epochs": 20,
            "bid_val_pr_auc": float(base_metrics["bid"]["pr_auc"]),
            "ask_val_pr_auc": float(base_metrics["ask"]["pr_auc"]),
            "bid_test_pr_auc": base_bid_test["test_pr_auc"],
            "ask_test_pr_auc": base_ask_test["test_pr_auc"],
            "test_net_pnl": float(base_test["net_pnl"]),
            "notes": "Existing approved weighted_bce run reused; test PR-AUC recomputed for ablation table only.",
        }
    )

    for name, features in feature_sets.items():
        if name == "F8_baseline_existing":
            continue
        out_dir = ablation_root / name
        train_mod.Z_FEATURES = features
        run_dir = train_mod.train_binary_models(
            input_path=augmented_path,
            out_dir=out_dir,
            sampling_mode="weighted_bce",
            window_size=100,
            train_stride=5,
            eval_stride=1,
            batch_size=1024,
            epochs=int(epochs),
            lr=1e-3,
            seed=42,
            device_arg="auto",
            downsample_neg_pos_ratio=3.0,
        )
        select_mod.select_thresholds(
            out_dir=out_dir,
            model_dir=run_dir,
            selection_mode="target_cancel_rate",
            target_cancel_rate=0.10,
            target_rates=[0.05, 0.10],
            grid_size=99,
        )
        backtest_mod.Z_FEATURES = features
        backtest_mod.run_backtest(
            labels_path=augmented_path,
            thresholds_path=out_dir / "selected_thresholds.json",
            out_dir=out_dir,
            random_seeds=20,
            batch_size=4096,
            device_arg="auto",
        )
        metrics_bid = pd.read_csv(out_dir / "metrics_bid.csv").query("split == 'val'").iloc[0]
        metrics_ask = pd.read_csv(out_dir / "metrics_ask.csv").query("split == 'val'").iloc[0]
        test_bid = _eval_side(train_mod, labels, run_dir, "bid", features)
        test_ask = _eval_side(train_mod, labels, run_dir, "ask", features)
        test_summary = pd.read_csv(out_dir / "test_backtest_summary.csv").iloc[0]
        rows.append(
            {
                "feature_set": name,
                "features": ",".join(features),
                "epochs": int(epochs),
                "bid_val_pr_auc": float(metrics_bid["pr_auc"]),
                "ask_val_pr_auc": float(metrics_ask["pr_auc"]),
                "bid_test_pr_auc": test_bid["test_pr_auc"],
                "ask_test_pr_auc": test_ask["test_pr_auc"],
                "test_net_pnl": float(test_summary["net_pnl"]),
                "notes": "Retrained via 04_train_binary_model.train_binary_models.",
            }
        )

    pd.DataFrame(rows).to_csv(results_dir / "feature_ablation_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature ablation retraining.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS)
    parser.add_argument("--epochs", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.results_dir, args.epochs)
    print(f"Feature ablation summary written to {args.results_dir}")
