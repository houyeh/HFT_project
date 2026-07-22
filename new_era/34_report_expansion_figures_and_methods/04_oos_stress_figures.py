from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_034 import FIGURES, RESULTS, V18, V21, V32, ensure_dirs, fmt_money, read_csv, save_placeholder_figure, set_style, write_markdown


def plot_queue_stress_heatmap() -> None:
    path = V18 / "queue_stress_heatmap_values.csv"
    if not path.exists():
        save_placeholder_figure(FIGURES / "queue_stress_heatmap_summary.png", "Queue stress data missing", f"Missing {path}")
        return
    df = read_csv(path)
    df.to_csv(RESULTS / "queue_stress_heatmap_values_snapshot.csv", index=False)
    policies = df["policy"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(policies), figsize=(6 * len(policies), 5), squeeze=False)
    for ax, policy in zip(axes[0], policies):
        g = df[df["policy"] == policy]
        pivot = g.pivot_table(index="neutral_fill_haircut", columns="toxic_multiplier", values="net_pnl", aggfunc="mean").sort_index(ascending=False)
        max_abs = max(abs(float(pivot.min().min())), abs(float(pivot.max().max())), 1.0)
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-max_abs, vmax=max_abs, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel("Toxic multiplier")
        ax.set_ylabel("Neutral fill haircut")
        ax.set_title(policy)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, fmt_money(pivot.iloc[i, j]), ha="center", va="center", fontsize=8)
        if 0.5 in pivot.index and 1.5 in pivot.columns:
            i = list(pivot.index).index(0.5)
            j = list(pivot.columns).index(1.5)
            ax.scatter([j], [i], s=180, facecolors="none", edgecolors="#111827", linewidths=2)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="Net PnL")
    fig.suptitle("Earlier 015/016 queue-aware stress heatmap: moderate point marked; negative regions remain negative", y=1.05)
    fig.text(
        0.5,
        0.01,
        "Note: this figure is for earlier 015/016 queue-aware stress, not the final non-ML baseline.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#30343b",
    )
    fig.savefig(FIGURES / "queue_stress_heatmap_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loso() -> None:
    path = V32 / "results" / "loso_robustness_metrics.csv"
    if not path.exists():
        save_placeholder_figure(FIGURES / "loso_robustness_bar.png", "LOSO data missing", f"Missing {path}")
        return
    df = read_csv(path)
    df.to_csv(RESULTS / "loso_robustness_metrics_snapshot.csv", index=False)
    plot = df[df["dropped_session"] != "NONE_FULL_SAMPLE"].copy()
    plot["label"] = plot["phase031_split"].astype(str) + "\n-" + plot["dropped_session"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#276ef1" if v >= 0 else "#c43c35" for v in plot["pnl_without_session"]]
    bars = ax.bar(plot["label"], plot["pnl_without_session"], color=colors)
    ax.axhline(0, color="#111827", lw=1)
    ax.set_ylabel("PnL after dropping one session")
    ax.set_title("Leave-one-session-out robustness")
    ax.tick_params(axis="x", rotation=35)
    for bar, val in zip(bars, plot["pnl_without_session"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, fmt_money(val), ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "loso_robustness_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_adverse_grid() -> None:
    path = V32 / "results" / "adverse_stress_grid.csv"
    if not path.exists():
        save_placeholder_figure(FIGURES / "adverse_stress_grid_summary.png", "Adverse stress grid missing", f"Missing {path}")
        return
    df = read_csv(path)
    df.to_csv(RESULTS / "adverse_stress_grid_snapshot.csv", index=False)
    subset = df[(df["fee_multiplier"] == 1.0) & (df["winning_fill_keep_rate"].isin([1.0, 0.5, 0.25]))].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for (split, keep), group in subset.groupby(["phase031_split", "winning_fill_keep_rate"], sort=True):
        group = group.sort_values("adverse_multiplier")
        ax.plot(group["adverse_multiplier"], group["stress_pnl"], marker="o", label=f"{split}, keep={keep:g}")
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xlabel("Adverse multiplier")
    ax.set_ylabel("Stress PnL")
    ax.set_title("Fill/adverse stress diagnostic at fee_multiplier=1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "adverse_stress_grid_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_oracle_gap() -> None:
    oracle_path = V21 / "oracle" / "oracle_perfect_upper_bounds.csv"
    eval_path = V21 / "evaluation" / "frozen_stress_robust_policy_test.csv"
    if not oracle_path.exists() or not eval_path.exists():
        save_placeholder_figure(
            FIGURES / "oracle_vs_causal_gap.png",
            "Oracle comparison missing",
            f"Missing {oracle_path if not oracle_path.exists() else eval_path}",
        )
        return
    oracle = read_csv(oracle_path)
    frozen = read_csv(eval_path)
    oracle_mod = oracle[(oracle["split"] == "test") & (oracle["stress_name"] == "moderate") & (oracle["policy"] == "perfect_quote_pnl_filter")]
    frozen_mod = frozen[(frozen["stress_name"] == "moderate") & (frozen["policy"].isin(["no_cancel", "015_common", "016_v2", "020_selected_policy", "021_selected"]))]
    rows = frozen_mod[["policy", "net_pnl"]].copy()
    if not oracle_mod.empty:
        rows = pd.concat([rows, oracle_mod[["policy", "net_pnl"]]], ignore_index=True)
    rows.to_csv(RESULTS / "oracle_vs_causal_gap_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#276ef1" if v >= 0 else "#c43c35" for v in rows["net_pnl"]]
    bars = ax.bar(rows["policy"].astype(str), rows["net_pnl"], color=colors)
    ax.axhline(0, color="#111827", lw=1)
    ax.set_ylabel("Moderate-stress net PnL")
    ax.set_title("Oracle upper bound vs causal policies under moderate stress")
    ax.tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, rows["net_pnl"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, fmt_money(val), ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "oracle_vs_causal_gap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary() -> None:
    lines = ["# OOS and Stress Figure Notes", ""]
    stress_path = V18 / "queue_stress_heatmap_values.csv"
    if stress_path.exists():
        stress = read_csv(stress_path)
        moderate = stress[(stress["neutral_fill_haircut"] == 0.5) & (stress["toxic_multiplier"] == 1.5)]
        for row in moderate.itertuples(index=False):
            lines.append(f"- `{row.policy}` moderate stress PnL: `{fmt_money(row.net_pnl)}`.")
    oracle_data = RESULTS / "oracle_vs_causal_gap_data.csv"
    if oracle_data.exists():
        gap = read_csv(oracle_data)
        for row in gap.itertuples(index=False):
            lines.append(f"- Moderate stress `{row.policy}`: `{fmt_money(row.net_pnl)}`.")
    lines.extend(
        [
            "",
            "The queue heatmap is for earlier 015/016 queue-aware stress, not the final non-ML baseline.",
            "",
            "The adverse stress summary filters `fee_multiplier == 1`; see the underlying snapshot CSV for the full fee grid.",
            "",
            "Stress and oracle charts remain proxy diagnostics. They do not establish real-world profitability or true queue replay.",
        ]
    )
    write_markdown(RESULTS / "oos_stress_figure_notes.md", lines)


def main() -> None:
    ensure_dirs()
    set_style()
    plot_queue_stress_heatmap()
    plot_loso()
    plot_adverse_grid()
    plot_oracle_gap()
    write_summary()


if __name__ == "__main__":
    main()
    print(f"Wrote OOS/stress figures to {FIGURES}")
