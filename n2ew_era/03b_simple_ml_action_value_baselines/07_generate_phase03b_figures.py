from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common_phase03b import FIGURES, RESULTS, ensure_dirs


def save_val_threshold_plot() -> None:
    df = pd.read_csv(RESULTS / "validation_threshold_selection.csv")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model_id, group in df.groupby("model_id"):
        g = group.sort_values("selection_rate")
        ax.plot(g["selection_rate"], g["total_excess_pnl"], marker="o", linewidth=1.2, markersize=3, label=model_id)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Validation Threshold Selection: total excess PnL H20")
    ax.set_xlabel("Selection rate")
    ax.set_ylabel("Validation total excess PnL")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "validation_threshold_selection_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_oos_bar() -> None:
    ml = pd.read_csv(RESULTS / "frozen_oos_evaluation.csv")
    base = pd.read_csv(RESULTS / "frozen_oos_baseline_controls.csv")
    base = base[base["policy_id"].isin(["no_quote", "old_bid_only_spread_ge_1", "side_by_queue_imbalance_abs_0p1"])]
    combined = pd.concat(
        [
            ml.assign(policy_id="phase03b_ml"),
            base[["split", "policy_id", "total_excess_pnl"]],
        ],
        ignore_index=True,
        sort=False,
    )
    piv = combined.pivot_table(index="split", columns="policy_id", values="total_excess_pnl", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    piv.plot(kind="bar", ax=ax)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Frozen OOS Excess PnL: ML vs Controls")
    ax.set_ylabel("Total excess PnL H20")
    ax.set_xlabel("Frozen split")
    fig.tight_layout()
    fig.savefig(FIGURES / "frozen_oos_ml_vs_controls.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_random_hurdle_plot() -> None:
    df = pd.read_csv(RESULTS / "ml_hurdle_summary.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df["model_total_excess_pnl"], width=0.4, label="ML")
    ax.bar([i + 0.2 for i in x], df["matched_random_95p"], width=0.4, label="Matched random 95p")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["split"])
    ax.set_title("Matched Random Hurdle: count/session/side")
    ax.set_ylabel("Total excess PnL H20")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "matched_random_hurdle.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_sequential_curve() -> None:
    curve = pd.read_csv(RESULTS / "causal_fill_sequential_equity_curve.csv")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for split, group in curve.groupby("split"):
        ax.plot(group["point"], group["equity"], marker="o", linewidth=1.5, label=split)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Causal-Fill Sequential Proxy: ML Policy")
    ax.set_xlabel("Session part index")
    ax.set_ylabel("Cumulative sequential excess PnL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "causal_fill_sequential_equity_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    save_val_threshold_plot()
    save_oos_bar()
    save_random_hurdle_plot()
    save_sequential_curve()
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()

