from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common_031 import RESULTS, ensure_dirs, load_full_actions


def main() -> None:
    ensure_dirs()
    actions = load_full_actions()
    ev_map = (
        actions.groupby(["phase031_split", "side", "spread_bucket", "event_rate_bucket", "volatility_bucket"], observed=False)
        .agg(
            action_count=("sample_id", "count"),
            avg_quote_ev=("quote_ev_moderate", "mean"),
            total_quote_ev=("quote_ev_moderate", "sum"),
            positive_ev_rate=("positive_ev_moderate", "mean"),
            baseline_selection_rate=("selected_by_baseline", "mean"),
        )
        .reset_index()
    )
    ev_map.to_csv(RESULTS / "action_space_ev_map.csv", index=False)

    plot_df = (
        actions.groupby(["side", "spread_bucket"], observed=False)
        .agg(avg_quote_ev=("quote_ev_moderate", "mean"), action_count=("sample_id", "count"))
        .reset_index()
    )
    pivot = plot_df.pivot(index="side", columns="spread_bucket", values="avg_quote_ev").fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=160)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("Spread Bucket")
    ax.set_ylabel("Side")
    ax.set_title("Full Action Average Moderate Quote EV")
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            ax.text(x, y, f"{pivot.iloc[y, x]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Avg Quote EV")
    fig.tight_layout()
    fig.savefig(RESULTS / "action_space_ev_heatmap.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
    print(f"Wrote action space EV map to {RESULTS}")

