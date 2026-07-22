from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common_03a import (
    FEATURES_REQUESTED,
    FIGURES,
    RESULTS,
    SPLITS,
    TARGET,
    available_columns,
    collect_sample,
    ensure_dirs,
)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(data) < 10 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan
    return float(spearmanr(data["x"], data["y"]).correlation)


def train_edges(train: pd.DataFrame, feature: str, side: str) -> np.ndarray:
    values = pd.to_numeric(train.loc[train["action_side"].astype(str) == side, feature], errors="coerce").dropna()
    if values.empty:
        return np.array([])
    edges = np.unique(values.quantile(np.linspace(0.0, 1.0, 11)).to_numpy(dtype=float))
    if len(edges) < 3:
        return np.array([])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def bucket_stats(sample: pd.DataFrame, feature: str, edges: np.ndarray) -> pd.DataFrame:
    if len(edges) < 3:
        return pd.DataFrame()
    work = sample[[feature, TARGET]].copy()
    work["bucket"] = pd.cut(pd.to_numeric(work[feature], errors="coerce"), bins=edges, labels=False, include_lowest=True)
    work = work.dropna(subset=["bucket", TARGET])
    work["bucket"] = work["bucket"].astype(int)
    return (
        work.groupby("bucket", as_index=False)
        .agg(
            sample_count=(TARGET, "size"),
            mean_feature=(feature, "mean"),
            mean_excess_pnl=(TARGET, "mean"),
            positive_rate=(TARGET, lambda s: float((s > 0.0).mean())),
        )
        .sort_values("bucket")
    )


def plot_feature(feature: str, bucket_all: pd.DataFrame) -> None:
    if bucket_all.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, side in zip(axes, ["bid", "ask"]):
        side_df = bucket_all[bucket_all["action_side"].astype(str) == side]
        for split, group in side_df.groupby("split"):
            group = group.sort_values("bucket")
            ax.plot(group["bucket"], group["mean_excess_pnl"], marker="o", linewidth=1.4, label=str(split))
        ax.axhline(0.0, color="#20242b", linewidth=1.0)
        ax.set_title(f"{feature}: {side}")
        ax.set_xlabel("Train-defined feature decile")
        ax.set_ylabel("Mean excess_pnl_H20")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle(f"Single Feature Bucket Lift: {feature}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    safe = feature.replace("/", "_")
    fig.savefig(FIGURES / f"bucket_lift_{safe}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    available = available_columns(FEATURES_REQUESTED)
    cols = ["split", "session_id", "action_side", TARGET, *available]
    samples = {}
    for split in SPLITS:
        samples[split] = collect_sample(split, cols, per_part=100_000, action_sides=["bid", "ask"])
        print(f"signal diagnostic sample split={split} rows={len(samples[split]):,}")

    rows = []
    bucket_rows = []
    for feature in FEATURES_REQUESTED:
        if feature not in available:
            rows.append(
                {
                    "feature": feature,
                    "split": "__ALL__",
                    "action_side": "__ALL__",
                    "status": "missing_from_phase02f_dataset",
                }
            )
            continue
        for side in ["bid", "ask"]:
            edges = train_edges(samples["train"], feature, side)
            for split, sample in samples.items():
                work = sample[sample["action_side"].astype(str) == side]
                rho = safe_spearman(work[feature], work[TARGET])
                buckets = bucket_stats(work, feature, edges)
                if not buckets.empty:
                    buckets["feature"] = feature
                    buckets["split"] = split
                    buckets["action_side"] = side
                    bucket_rows.append(buckets)
                    base_mean = float(pd.to_numeric(work[TARGET], errors="coerce").mean())
                    top = buckets[buckets["bucket"] == buckets["bucket"].max()]
                    bottom = buckets[buckets["bucket"] == buckets["bucket"].min()]
                    top_mean = float(top["mean_excess_pnl"].iloc[0]) if not top.empty else np.nan
                    bottom_mean = float(bottom["mean_excess_pnl"].iloc[0]) if not bottom.empty else np.nan
                else:
                    base_mean = float(pd.to_numeric(work[TARGET], errors="coerce").mean()) if len(work) else np.nan
                    top_mean = np.nan
                    bottom_mean = np.nan
                rows.append(
                    {
                        "feature": feature,
                        "split": split,
                        "action_side": side,
                        "status": "ok",
                        "sample_count": int(len(work)),
                        "spearman_corr": rho,
                        "base_mean_excess_pnl": base_mean,
                        "bottom_decile_mean_excess_pnl": bottom_mean,
                        "top_decile_mean_excess_pnl": top_mean,
                        "top_decile_lift_vs_base": top_mean - base_mean if np.isfinite(top_mean) and np.isfinite(base_mean) else np.nan,
                        "bottom_decile_lift_vs_base": bottom_mean - base_mean if np.isfinite(bottom_mean) and np.isfinite(base_mean) else np.nan,
                        "decile_edges_source": "train_split_same_action_side",
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "single_feature_signal_diagnostics.csv", index=False)
    bucket_all = pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame()
    bucket_all.to_csv(RESULTS / "single_feature_bucket_lift.csv", index=False)
    for feature in sorted(bucket_all["feature"].unique()) if not bucket_all.empty else []:
        plot_feature(feature, bucket_all[bucket_all["feature"] == feature])
    print("Single-feature diagnostics complete.")
    print(out[out["status"].astype(str) == "ok"][["feature", "split", "action_side", "spearman_corr", "top_decile_lift_vs_base"]].to_string(index=False))


if __name__ == "__main__":
    main()

