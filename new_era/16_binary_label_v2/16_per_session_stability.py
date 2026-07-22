from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from round2_audit_utils import RESULTS


def run(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    model_sessions = pd.read_csv(Path(__file__).resolve().parent / "results" / "pnl_by_session.csv")
    random_sessions = pd.read_csv(results_dir / "strict_random_baseline_session_trials.csv")

    model_total = float(model_sessions["net_pnl"].sum())
    rows = []
    for method, method_df in random_sessions.groupby("method", sort=True):
        for session_id, group in method_df.groupby("session_id", sort=True):
            model_row = model_sessions[model_sessions["session_id"].astype(str) == str(session_id)].iloc[0]
            pnl = group["net_pnl"].to_numpy(dtype=float)
            model_pnl = float(model_row["net_pnl"])
            rows.append(
                {
                    "session_id": str(session_id),
                    "random_method": method,
                    "model_net_pnl": model_pnl,
                    "model_share_of_total_pnl": float(model_pnl / model_total) if model_total else float("nan"),
                    "random_mean": float(np.mean(pnl)),
                    "random_p95": float(np.percentile(pnl, 95)),
                    "random_p99": float(np.percentile(pnl, 99)),
                    "model_minus_random_mean": float(model_pnl - np.mean(pnl)),
                    "model_minus_random_p95": float(model_pnl - np.percentile(pnl, 95)),
                    "model_minus_random_p99": float(model_pnl - np.percentile(pnl, 99)),
                    "model_beats_random_mean": bool(model_pnl > np.mean(pnl)),
                    "model_beats_random_p95": bool(model_pnl > np.percentile(pnl, 95)),
                    "model_beats_random_p99": bool(model_pnl > np.percentile(pnl, 99)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "pnl_by_session_model_vs_random.csv", index=False)

    plot_df = out[out["random_method"] == "random_v3"].copy()
    x = np.arange(len(plot_df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - 1.5 * width, plot_df["model_net_pnl"], width, label="model")
    ax.bar(x - 0.5 * width, plot_df["random_mean"], width, label="random_v3 mean")
    ax.bar(x + 0.5 * width, plot_df["random_p95"], width, label="random_v3 95p")
    ax.bar(x + 1.5 * width, plot_df["random_p99"], width, label="random_v3 99p")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["session_id"], rotation=15, ha="right")
    ax.set_ylabel("Net PnL")
    ax.set_title("Per-Session Model PnL vs Strict Matched Random")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "session_pnl_bar_model_vs_random.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model and random PnL per session.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.results_dir)
    print(f"Per-session stability report written to {args.results_dir}")
