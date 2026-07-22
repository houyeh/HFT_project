from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common_034 import FIGURES, RESULTS, ensure_dirs, load_presentation_action_universe, set_style, write_markdown


def main() -> None:
    ensure_dirs()
    set_style()
    actions = load_presentation_action_universe()
    actions["split_side"] = actions["eval_group"].astype(str) + " | " + actions["side"].astype(str)
    summary = (
        actions.groupby(["eval_group", "side"], observed=False)
        .agg(
            action_count=("spread_ticks", "size"),
            min_spread=("spread_ticks", "min"),
            median_spread=("spread_ticks", "median"),
            mean_spread=("spread_ticks", "mean"),
            max_spread=("spread_ticks", "max"),
            spread_ge_1_count=("spread_ticks", lambda s: int((s >= 1.0).sum())),
            spread_lt_1_count=("spread_ticks", lambda s: int((s < 1.0).sum())),
        )
        .reset_index()
    )
    summary["spread_ge_1_rate"] = summary["spread_ge_1_count"] / summary["action_count"].replace(0, pd.NA)
    summary["spread_filter_saturated"] = summary["spread_ge_1_rate"].eq(1.0)
    summary.to_csv(RESULTS / "spread_distribution_summary.csv", index=False)

    order = (
        summary.sort_values(["eval_group", "side"])["eval_group"].drop_duplicates().tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, side in zip(axes, ["bid", "ask"]):
        data = actions[actions["side"] == side]
        groups = [g for g in order if g in set(data["eval_group"])]
        values = [data.loc[data["eval_group"] == g, "spread_ticks"].to_numpy() for g in groups]
        ax.boxplot(values, labels=[g.replace("unused_", "").replace("phase031_", "") for g in groups], showfliers=False)
        ax.axhline(1.0, color="#c43c35", lw=1.6, linestyle="--", label="spread_ticks = 1")
        ax.set_title(f"{side.capitalize()} spread distribution")
        ax.set_xlabel("Evaluation group")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Spread ticks")
    fig.suptitle("Spread Distribution by Split/Side: spread>=1 is saturated for quote-eligible actions", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "spread_distribution_by_split_side.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    saturated_rows = int(summary["spread_filter_saturated"].sum())
    total_rows = int(len(summary))
    write_markdown(
        ROOT_PATH(),
        [
            "# Spread Distribution Audit",
            "",
            f"Audited `{int(len(actions))}` quote-candidate actions across `{actions['eval_group'].nunique()}` evaluation groups.",
            "",
            f"`spread_ticks >= 1` is saturated in `{saturated_rows}` of `{total_rows}` group/side buckets.",
            "",
            "Interpretation: the final rule should be explained as a bid-side regime filter. The spread threshold documents quote eligibility but does not provide fine-grained filtering power in this action universe.",
            "",
            "Figure: `figures/spread_distribution_by_split_side.png`.",
        ],
    )


def ROOT_PATH():
    from common_034 import ROOT

    return ROOT / "spread_distribution_audit.md"


if __name__ == "__main__":
    main()
    print(f"Wrote spread distribution audit to {FIGURES}")
