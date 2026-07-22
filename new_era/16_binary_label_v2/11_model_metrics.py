from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"


def _precision_at_top(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    k = max(1, int(np.ceil(len(p) * frac)))
    idx = np.argsort(p)[::-1][:k]
    return float(np.mean(y[idx]))


def _plot_calibration(side: str, frames: dict[str, pd.DataFrame], results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for split, df in frames.items():
        y = df["y_true"].to_numpy(dtype=int)
        p = df["prob_toxic"].to_numpy(dtype=float)
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=split)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_title(f"{side.capitalize()} Calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical toxic rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"calibration_curve_{side}.png", dpi=150)
    plt.close(fig)


def _plot_score_distribution(side: str, frames: dict[str, pd.DataFrame], results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for split, df in frames.items():
        ax.hist(df["prob_toxic"], bins=80, alpha=0.45, density=True, label=split)
    ax.set_title(f"{side.capitalize()} Score Distribution")
    ax.set_xlabel("P(toxic)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / f"score_distribution_{side}.png", dpi=150)
    plt.close(fig)


def run(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(OUTPUTS / "binary_labels_v2.parquet").reset_index(drop=True)
    labels["event_id"] = np.arange(len(labels), dtype=np.int64)
    test_pred = pd.read_parquet(OUTPUTS / "test_predictions_binary_cancel.parquet")
    rows = []

    for side in ["bid", "ask"]:
        val = pd.read_parquet(OUTPUTS / "model_weighted_bce" / f"val_predictions_{side}.parquet")
        val = val[["event_id", "prob_toxic", "y_true"]].copy()
        test = test_pred[["event_id", f"{side}_prob_toxic"]].rename(columns={f"{side}_prob_toxic": "prob_toxic"})
        test = test.merge(labels[["event_id", f"{side}_toxic"]], on="event_id", how="left")
        test.rename(columns={f"{side}_toxic": "y_true"}, inplace=True)
        frames = {"val": val, "test": test}

        for split, df in frames.items():
            y = df["y_true"].to_numpy(dtype=int)
            p = df["prob_toxic"].to_numpy(dtype=float)
            rows.append(
                {
                    "side": side,
                    "split": split,
                    "rows": int(len(df)),
                    "positive_rate": float(np.mean(y)),
                    "pr_auc": float(average_precision_score(y, p)),
                    "roc_auc": float(roc_auc_score(y, p)),
                    "brier_score": float(brier_score_loss(y, p)),
                    "precision_top_1pct": _precision_at_top(y, p, 0.01),
                    "precision_top_5pct": _precision_at_top(y, p, 0.05),
                    "precision_top_10pct": _precision_at_top(y, p, 0.10),
                }
            )
        _plot_calibration(side, frames, results_dir)
        _plot_score_distribution(side, frames, results_dir)

    pd.DataFrame(rows).to_csv(results_dir / "metrics_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model ML metric audit artifacts.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.results_dir)
    print(f"Model metrics written to {args.results_dir}")
