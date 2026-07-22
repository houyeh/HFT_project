from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from common_phase01 import FEATURE_DRIFT_COLUMNS, FIGURES_DIR, HORIZONS, OLD_RESIDUAL_UNIVERSE, RESULTS_DIR, ensure_dirs, split_dataset_dir


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#30343b",
            "axes.grid": True,
            "grid.color": "#d9dee7",
            "font.size": 10,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def sample_split(split: str, columns: list[str], max_rows: int = 60_000) -> pd.DataFrame:
    dataset = ds.dataset(split_dataset_dir(split), format="parquet")
    parts = []
    remaining = int(max_rows)
    for batch in dataset.scanner(columns=columns, batch_size=50_000).to_batches():
        if remaining <= 0:
            break
        pdf = batch.to_pandas()
        if len(pdf) > remaining:
            pdf = pdf.sample(n=remaining, random_state=20260701)
        parts.append(pdf)
        remaining -= len(pdf)
    if not parts:
        return pd.DataFrame(columns=columns)
    out = pd.concat(parts, ignore_index=True)
    out["split"] = split
    return out


def plot_waterfall(counts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(counts))
    width = 0.22
    for i, col in enumerate(["raw_rows", "quote_eligible_rows", "candidate_action_rows"]):
        ax.bar(x + (i - 1) * width, counts[col], width=width, label=col)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(counts["split"])
    ax.set_title("Action Count Waterfall")
    ax.set_xlabel("Split")
    ax.set_ylabel("Rows/actions (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "action_count_waterfall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_side_distribution(counts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(counts))
    bottom = np.zeros(len(counts))
    colors = {"bid_actions": "#276ef1", "ask_actions": "#c43c35", "no_quote_actions": "#7a4cc2"}
    for col in ["bid_actions", "ask_actions", "no_quote_actions"]:
        vals = counts[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=col.replace("_actions", ""), color=colors[col])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(counts["split"])
    ax.set_title("Action Side Distribution")
    ax.set_xlabel("Split")
    ax.set_ylabel("Candidate actions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "action_side_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_distribution(sample: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    data = []
    labels = []
    work = sample[sample["action_side"].isin(["bid", "ask"])].copy()
    for horizon in HORIZONS:
        pnl = work[f"pnl_H{horizon}"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pnl):
            lo, hi = pnl.quantile([0.01, 0.99])
            data.append(pnl.clip(lo, hi).to_numpy())
            labels.append(f"H{horizon}")
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(0, color="#20242b", linewidth=1.0)
    ax.set_title("Short-Horizon Sequential Proxy Label Distribution")
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("pnl_H clipped to sample 1/99 pct")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "horizon_label_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_snapshot(sample: pd.DataFrame) -> None:
    features = FEATURE_DRIFT_COLUMNS
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    axes = axes.ravel()
    for ax, col in zip(axes, features):
        vals = sample[col].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        else:
            lo, hi = vals.quantile([0.01, 0.99])
            ax.hist(vals.clip(lo, hi), bins=50, color="#4d8bbf", alpha=0.85)
        ax.set_title(col)
        ax.set_xlabel("Clipped feature value")
        ax.set_ylabel("Frequency")
    fig.suptitle("Causal Feature Distribution Snapshot", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "feature_distribution_snapshot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_by_split(sample: pd.DataFrame) -> None:
    features = FEATURE_DRIFT_COLUMNS
    split_order = [s for s in ["train", "val", "test", "new4"] if s in set(sample["split"].astype(str))]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8))
    axes = axes.ravel()
    for ax, col in zip(axes, features):
        data = []
        labels = []
        for split in split_order:
            vals = pd.to_numeric(sample.loc[sample["split"].astype(str) == split, col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if vals.empty:
                continue
            lo, hi = vals.quantile([0.01, 0.99])
            data.append(vals.clip(lo, hi).to_numpy())
            labels.append(split)
        if data:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
            ax.axhline(0, color="#20242b", linewidth=0.8, alpha=0.5)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.set_title(col)
        ax.set_xlabel("Split")
        ax.set_ylabel("Clipped feature value")
    fig.suptitle("Feature Distribution by Split", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "feature_distribution_by_split.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_old_vs_new(counts: pd.DataFrame) -> None:
    old_train_rows = 820
    if OLD_RESIDUAL_UNIVERSE.exists():
        old = pd.read_parquet(OLD_RESIDUAL_UNIVERSE, columns=["split"])
        old_train_rows = int((old["split"].astype(str) == "train").sum())
    train_actions = int(counts[counts["split"].astype(str) == "train"].iloc[0]["candidate_action_rows"])
    total_actions = int(counts["candidate_action_rows"].sum())
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    labels = ["Old residual train", "New train full-action", "New total full-action"]
    vals = [old_train_rows, train_actions, total_actions]
    ax.bar(labels, vals, color=["#c43c35", "#276ef1", "#2f9e44"])
    ax.set_yscale("log")
    ax.set_xlabel("Universe")
    ax.set_ylabel("Action/sample rows (log scale)")
    ax.set_title("Old Residual Universe vs New Full L2 Action Universe")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.08, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "old_residual_vs_new_universe.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    set_style()
    counts = pd.read_csv(RESULTS_DIR / "action_universe_by_split.csv")
    plot_waterfall(counts)
    plot_side_distribution(counts)
    columns = ["action_side", "spread_ticks", "queue_imbalance", "deep_imbalance", "event_rate_past_50", "volatility_past_50", "top_depth"] + [f"pnl_H{h}" for h in HORIZONS]
    split_order = counts["split"].astype(str).tolist()
    sample = pd.concat([sample_split(split, columns) for split in split_order], ignore_index=True)
    plot_horizon_distribution(sample)
    plot_feature_snapshot(sample)
    plot_feature_by_split(sample)
    plot_old_vs_new(counts)
    print(f"Wrote Phase 01 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
