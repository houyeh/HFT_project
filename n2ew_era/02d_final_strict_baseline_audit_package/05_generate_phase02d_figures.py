from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_phase02d import (
    FIGURES,
    MODERATE_ADVERSE_MULTIPLIER,
    MODERATE_FILL_KEEP,
    PHASE02C_DIR,
    RESULTS,
    SELECTED_HORIZON,
    SELECTED_POLICY_ID,
    ensure_dirs,
    exact_random_rows,
    read_csv_if_exists,
)


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


def plot_selected_vs_controls() -> None:
    df = read_csv_if_exists(RESULTS / "selected_policy_vs_controls.csv")
    if df.empty:
        return
    splits = df["split"].astype(str).tolist()
    x = np.arange(len(splits))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(x - width / 2, df["total_pnl"], width=width, label="selected")
    ax.bar(x + width / 2, df["old_bid_only_pnl"], width=width, label="old_bid_only")
    ax.axhline(0, color="#20242b", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_title(f"Selected vs Old Bid-Only: {SELECTED_POLICY_ID}, H{SELECTED_HORIZON}")
    ax.set_ylabel("short-horizon proxy PnL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "selected_vs_controls_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_random_hurdle() -> None:
    df = read_csv_if_exists(RESULTS / "selected_policy_vs_controls.csv")
    exact = exact_random_rows()
    if df.empty:
        return
    use = df[["split", "total_pnl", "matched_random_95p"]].copy()
    if not exact.empty:
        exact_small = exact[["split", "exact_random_95p"]].drop_duplicates("split")
        use = use.merge(exact_small, on="split", how="left")
        use["hurdle_95p"] = use["exact_random_95p"].fillna(use["matched_random_95p"])
    else:
        use["hurdle_95p"] = use["matched_random_95p"]
    x = np.arange(len(use))
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(x - 0.18, use["total_pnl"], width=0.36, label="selected")
    ax.bar(x + 0.18, use["hurdle_95p"], width=0.36, label="matched random 95p")
    ax.set_xticks(x)
    ax.set_xticklabels(use["split"])
    ax.set_title(f"Matched Random Hurdle: {SELECTED_POLICY_ID}, H{SELECTED_HORIZON}")
    ax.set_ylabel("short-horizon proxy PnL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "random_hurdle_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fee_sensitivity() -> None:
    df = read_csv_if_exists(RESULTS / "fee_sensitivity_summary.csv")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for split, group in df.groupby("split", sort=False):
        group = group.sort_values("fee_multiplier")
        ax.plot(group["fee_multiplier"], group["adjusted_total_pnl"], marker="o", label=str(split))
    ax.axhline(0, color="#20242b", linewidth=0.9)
    ax.set_title(f"Fee Sensitivity: {SELECTED_POLICY_ID}, H{SELECTED_HORIZON}")
    ax.set_xlabel("fee multiplier")
    ax.set_ylabel("adjusted total PnL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fee_sensitivity_selected_policy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_queue_heatmap() -> None:
    stress = read_csv_if_exists(PHASE02C_DIR / "results" / "queue_stress_grid.csv")
    if stress.empty:
        return
    work = stress[
        (stress["policy_id"].astype(str) == SELECTED_POLICY_ID)
        & (stress["horizon"].astype(int) == SELECTED_HORIZON)
        & (stress["action_delay_events"].astype(int) == 0)
        & (stress["split"].astype(str).isin(["test", "new4"]))
    ].copy()
    splits = work["split"].drop_duplicates().astype(str).tolist()
    fig, axes = plt.subplots(1, max(1, len(splits)), figsize=(6.3 * max(1, len(splits)), 4.9), squeeze=False)
    for ax, split in zip(axes.ravel(), splits):
        g = work[work["split"].astype(str) == split]
        piv = g.pivot_table(index="adverse_multiplier", columns="fill_keep_rate", values="stress_total_pnl", aggfunc="first")
        piv = piv.sort_index().reindex(columns=sorted(piv.columns, reverse=True))
        im = ax.imshow(piv.to_numpy(dtype=float), cmap="RdYlGn", aspect="auto")
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns])
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels([f"{r:g}" for r in piv.index])
        ax.set_xlabel("fill_keep_rate")
        ax.set_ylabel("adverse_multiplier")
        ax.set_title(f"{split}: stress PnL")
        if MODERATE_FILL_KEEP in piv.columns and MODERATE_ADVERSE_MULTIPLIER in piv.index:
            ax.scatter([list(piv.columns).index(MODERATE_FILL_KEEP)], [list(piv.index).index(MODERATE_ADVERSE_MULTIPLIER)], color="black", marker="x", s=80)
        fig.colorbar(im, ax=ax, label="stress_total_pnl")
    fig.suptitle(f"Queue Stress Heatmap: {SELECTED_POLICY_ID}, H{SELECTED_HORIZON}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGURES / "queue_stress_heatmap_selected_policy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delay_sensitivity() -> None:
    delay = read_csv_if_exists(PHASE02C_DIR / "results" / "delay_sensitivity.csv")
    if delay.empty:
        return
    work = delay[
        (delay["policy_id"].astype(str) == SELECTED_POLICY_ID)
        & (delay["horizon"].astype(int) == SELECTED_HORIZON)
        & (delay["fill_keep_rate"].astype(float) == MODERATE_FILL_KEEP)
        & (delay["adverse_multiplier"].astype(float) == MODERATE_ADVERSE_MULTIPLIER)
    ].copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for split, group in work.groupby("split", sort=False):
        group = group.sort_values("action_delay_events")
        ax.plot(group["action_delay_events"], group["stress_total_pnl"], marker="o", label=str(split))
    ax.axhline(0, color="#20242b", linewidth=0.9)
    ax.set_title(f"Delay Sensitivity: {SELECTED_POLICY_ID}, H{SELECTED_HORIZON}, moderate stress")
    ax.set_xlabel("action delay events")
    ax.set_ylabel("stress_total_pnl")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "delay_sensitivity_selected_policy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    set_style()
    plot_selected_vs_controls()
    plot_random_hurdle()
    plot_fee_sensitivity()
    plot_queue_heatmap()
    plot_delay_sensitivity()
    print(f"Wrote Phase 02D figures to {FIGURES}")


if __name__ == "__main__":
    main()
