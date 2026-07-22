from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_phase02 import FIGURES_DIR, RESULTS_DIR, ensure_dirs, equity_curve, selected_policy_entries


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#d9dee7",
            "axes.edgecolor": "#30343b",
            "font.size": 10,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def plot_validation_heatmap(grid: pd.DataFrame) -> None:
    val = grid[grid["split"].astype(str) == "val"].copy()
    top_ids = (
        val.groupby("policy_id")["total_pnl"]
        .max()
        .sort_values(ascending=False)
        .head(24)
        .index.tolist()
    )
    piv = val[val["policy_id"].isin(top_ids)].pivot_table(index="policy_id", columns="horizon", values="total_pnl", aggfunc="first")
    piv = piv.loc[top_ids]
    fig, ax = plt.subplots(figsize=(10.5, max(5.5, 0.28 * len(piv))))
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([f"H{h}" for h in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_title("Validation PnL Heatmap: Top Candidate Policies")
    fig.colorbar(im, ax=ax, label="val total_pnl")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "validation_policy_pnl_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_family_heatmap(grid: pd.DataFrame) -> None:
    val = grid[grid["split"].astype(str) == "val"].copy()
    best = val.groupby(["family", "horizon"], as_index=False)["total_pnl"].max()
    piv = best.pivot(index="family", columns="horizon", values="total_pnl")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([f"H{h}" for h in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_title("Validation Best PnL by Policy Family")
    fig.colorbar(im, ax=ax, label="best val total_pnl")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "validation_family_pnl_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_equity_and_drawdown(policy: dict) -> None:
    horizon = int(policy["selected_horizon"])
    fig_eq, ax_eq = plt.subplots(figsize=(11, 5.2))
    fig_dd, ax_dd = plt.subplots(figsize=(11, 5.2))
    for split in ["val", "test", "new4"]:
        curve = equity_curve(policy, split, horizon)
        if curve.empty:
            continue
        x = np.arange(len(curve))
        ax_eq.plot(x, curve["equity"], label=split)
        ax_dd.plot(x, curve["drawdown"], label=split)
    ax_eq.axhline(0, color="#20242b", linewidth=0.9)
    ax_eq.set_title(f"Selected Policy Equity Curves ({policy['policy_id']}, H{horizon})")
    ax_eq.set_ylabel("Cumulative short-horizon proxy PnL")
    ax_eq.set_xlabel("Downsampled chronological event index")
    ax_eq.legend()
    fig_eq.tight_layout()
    fig_eq.savefig(FIGURES_DIR / "selected_policy_equity_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig_eq)

    ax_dd.set_title(f"Selected Policy Drawdown Curves ({policy['policy_id']}, H{horizon})")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Downsampled chronological event index")
    ax_dd.legend()
    fig_dd.tight_layout()
    fig_dd.savefig(FIGURES_DIR / "selected_policy_drawdown_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig_dd)


def plot_random_distributions() -> None:
    trials_path = RESULTS_DIR / "random_baseline_trials.csv"
    summary_path = RESULTS_DIR / "random_baseline_summary.csv"
    if not trials_path.exists() or not summary_path.exists():
        return
    trials = pd.read_csv(trials_path)
    summary = pd.read_csv(summary_path)
    splits = [s for s in ["test", "new4"] if s in set(trials["split"].astype(str))]
    baselines = trials["random_baseline"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(splits), len(baselines), figsize=(6.2 * len(baselines), 4.2 * len(splits)), squeeze=False)
    for i, split in enumerate(splits):
        for j, baseline in enumerate(baselines):
            ax = axes[i, j]
            vals = trials[(trials["split"].astype(str) == split) & (trials["random_baseline"].astype(str) == baseline)]["random_pnl"].to_numpy(dtype=float)
            if len(vals):
                ax.hist(vals, bins=45, color="#6c8ebf", alpha=0.85)
                row = summary[(summary["split"].astype(str) == split) & (summary["random_baseline"].astype(str) == baseline)].iloc[0]
                ax.axvline(float(row["model_pnl"]), color="#c43c35", linewidth=2.0, label="model")
                ax.axvline(float(row["random_95p"]), color="#20242b", linestyle="--", linewidth=1.5, label="random 95p")
            ax.set_title(f"{split}: {baseline}")
            ax.legend()
    fig.suptitle("Random Baseline PnL Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES_DIR / "random_baseline_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    set_style()
    grid = pd.read_csv(RESULTS_DIR / "baseline_grid_all_splits.csv")
    plot_validation_heatmap(grid)
    plot_family_heatmap(grid)
    plot_equity_and_drawdown(selected_policy_entries()[0])
    plot_random_distributions()
    print(f"Wrote Phase 02 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
