from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_search_utils import ROBUSTNESS_DIR, V16_DIR, ensure_dirs, load_training_module


def run(seeds: list[int], epochs: int, train_stride: int, batch_size: int) -> None:
    ensure_dirs()
    train_mod = load_training_module()
    rows = []
    for seed in seeds:
        out_dir = ROBUSTNESS_DIR / f"multi_seed_seed{seed}"
        run_dir = out_dir / "model_weighted_bce"
        if not (run_dir / "metrics_all.csv").exists():
            run_dir = train_mod.train_binary_models(
                input_path=V16_DIR / "outputs" / "binary_labels_v2.parquet",
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
        for _, row in metrics.iterrows():
            rows.append(
                {
                    "seed": int(seed),
                    "epochs": int(epochs),
                    "train_stride": int(train_stride),
                    "side": row["side"],
                    "split": row["split"],
                    "positive_rate": row["positive_rate"],
                    "pr_auc": row["pr_auc"],
                    "roc_auc": row["roc_auc"],
                    "loss": row["loss"],
                    "test_evaluated": False,
                    "model_dir": str(run_dir),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(ROBUSTNESS_DIR / "multi_seed_stability.csv", index=False)
    val = out[out["split"] == "val"].copy()
    lines = [
        "# Multi-Seed Stability",
        "",
        f"Seeds evaluated: `{','.join(map(str, seeds))}`.",
        f"Epochs per seed: `{epochs}`. Train stride: `{train_stride}`.",
        "",
        "This is a compute-constrained stability pass. Test was not evaluated in this script.",
        "",
        "## Validation PR-AUC Summary",
        "",
    ]
    for side, group in val.groupby("side", sort=True):
        lines.append(
            f"- {side}: mean PR-AUC `{group['pr_auc'].mean():.6f}`, std `{group['pr_auc'].std(ddof=0):.6f}`, "
            f"min `{group['pr_auc'].min():.6f}`, max `{group['pr_auc'].max():.6f}`."
        )
    lines.append("")
    lines.append("The signal is considered seed-stable only if validation PR-AUC remains consistently above the split positive rate across seeds.")
    (ROBUSTNESS_DIR / "multi_seed_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compute-constrained multi-seed 016_v2 stability.")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_stride", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    run(seeds, args.epochs, args.train_stride, args.batch_size)
    print(f"Wrote {ROBUSTNESS_DIR / 'multi_seed_stability.csv'}")
