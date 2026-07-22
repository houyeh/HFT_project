from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_search_utils import ROBUSTNESS_DIR, V16_DIR, ensure_dirs, load_training_module


def _make_shuffled_labels(seed: int) -> Path:
    out_path = ROBUSTNESS_DIR / f"binary_labels_train_shuffled_seed{seed}.parquet"
    if out_path.exists():
        return out_path
    labels = pd.read_parquet(V16_DIR / "outputs" / "binary_labels_v2.parquet").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    train_mask = labels["split"].astype(str) == "train"
    for col in ["bid_toxic", "ask_toxic"]:
        vals = labels.loc[train_mask, col].to_numpy(copy=True)
        rng.shuffle(vals)
        labels.loc[train_mask, col] = vals
    labels.to_parquet(out_path, index=False)
    return out_path


def run(seed: int, epochs: int, train_stride: int, batch_size: int) -> None:
    ensure_dirs()
    train_mod = load_training_module()
    shuffled_path = _make_shuffled_labels(seed)
    out_dir = ROBUSTNESS_DIR / f"shuffled_label_control_seed{seed}"
    run_dir = train_mod.train_binary_models(
        input_path=shuffled_path,
        out_dir=out_dir,
        sampling_mode="weighted_bce",
        window_size=100,
        train_stride=int(train_stride),
        eval_stride=1,
        batch_size=int(batch_size),
        epochs=int(epochs),
        lr=1e-3,
        seed=int(seed),
        device_arg="auto",
        downsample_neg_pos_ratio=3.0,
    )
    metrics = pd.read_csv(run_dir / "metrics_all.csv")
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            {
                "control": "shuffled_train_labels_only",
                "seed": int(seed),
                "epochs": int(epochs),
                "train_stride": int(train_stride),
                "side": row["side"],
                "split": row["split"],
                "positive_rate": row["positive_rate"],
                "pr_auc": row["pr_auc"],
                "roc_auc": row["roc_auc"],
                "loss": row["loss"],
                "expected_behavior": "Validation PR-AUC should collapse toward the natural positive rate/random baseline.",
                "val_test_labels_unchanged": True,
                "test_evaluated": False,
                "model_dir": str(run_dir),
            }
        )
    pd.DataFrame(rows).to_csv(ROBUSTNESS_DIR / "shuffled_label_control.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 016_v2 with shuffled train labels only.")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_stride", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.seed, args.epochs, args.train_stride, args.batch_size)
    print(f"Wrote {ROBUSTNESS_DIR / 'shuffled_label_control.csv'}")
