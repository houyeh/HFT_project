from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from root_cause_utils import RESULTS_DIR, ensure_dirs, load_context, write_markdown


def run() -> None:
    ensure_dirs()
    _, actions = load_context("test")
    values = actions["quote_value_moderate"].to_numpy(dtype=float)
    pos = np.maximum(values, 0.0)
    total_pos = float(pos.sum())
    order = np.argsort(pos)[::-1]
    rows = []
    for frac in [0.01, 0.05, 0.10, 0.25, 1.00]:
        k = max(1, int(np.ceil(frac * len(actions))))
        pnl = float(pos[order[:k]].sum())
        rows.append(
            {
                "split": "test_diagnostic_burned",
                "moderate_stress": "neutral_fill_haircut=0.5,toxic_multiplier=1.5,delay=0",
                "top_fraction": frac,
                "top_quote_count": k,
                "captured_oracle_positive_pnl": pnl,
                "share_of_oracle_positive_pnl": pnl / total_pos if total_pos else np.nan,
                "total_oracle_positive_pnl": total_pos,
                "oracle_positive_quote_count": int((values > 0.0).sum()),
                "all_quote_action_count": int(len(actions)),
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "oracle_opportunity_map.csv", index=False)

    side_session = (
        actions.assign(positive_pnl=pos, oracle_positive=values > 0.0)
        .groupby(["session_id", "side"], observed=False)
        .agg(
            num_quote_actions=("action_id", "count"),
            oracle_positive_quote_count=("oracle_positive", "sum"),
            oracle_positive_pnl=("positive_pnl", "sum"),
            avg_quote_value=("quote_value_moderate", "mean"),
        )
        .reset_index()
    )
    side_session["oracle_pnl_share"] = side_session["oracle_positive_pnl"] / total_pos if total_pos else np.nan
    side_session.to_csv(RESULTS_DIR / "oracle_by_session_side.csv", index=False)

    session_side_spread = (
        actions.assign(positive_pnl=pos, oracle_positive=values > 0.0)
        .groupby(["session_id", "side", "spread_bucket"], observed=False)
        .agg(
            num_quote_actions=("action_id", "count"),
            oracle_positive_quote_count=("oracle_positive", "sum"),
            oracle_positive_pnl=("positive_pnl", "sum"),
            avg_quote_value=("quote_value_moderate", "mean"),
        )
        .reset_index()
    )
    session_side_spread["oracle_pnl_share"] = session_side_spread["oracle_positive_pnl"] / total_pos if total_pos else np.nan
    session_side_spread.to_csv(RESULTS_DIR / "oracle_by_session_side_spread.csv", index=False)

    spread = (
        actions.assign(positive_pnl=pos, oracle_positive=values > 0.0)
        .groupby(["spread_bucket"], observed=False)
        .agg(
            num_quote_actions=("action_id", "count"),
            oracle_positive_quote_count=("oracle_positive", "sum"),
            oracle_positive_pnl=("positive_pnl", "sum"),
            avg_quote_value=("quote_value_moderate", "mean"),
        )
        .reset_index()
    )
    spread["oracle_pnl_share"] = spread["oracle_positive_pnl"] / total_pos if total_pos else np.nan
    spread.to_csv(RESULTS_DIR / "oracle_by_spread_bucket.csv", index=False)

    sorted_pos = np.sort(pos)[::-1]
    cumulative = np.cumsum(sorted_pos) / total_pos if total_pos else np.zeros_like(sorted_pos)
    x = np.arange(1, len(cumulative) + 1) / max(len(cumulative), 1)
    plt.figure(figsize=(8, 5))
    plt.plot(x, cumulative, linewidth=2)
    for frac in [0.01, 0.05]:
        plt.axvline(frac, linestyle="--", color="tab:red", alpha=0.7)
    plt.title("Oracle Positive PnL Concentration")
    plt.xlabel("Top fraction of quote actions by oracle value")
    plt.ylabel("Cumulative share of positive oracle PnL")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "oracle_pnl_concentration.png", dpi=160)
    plt.close()

    write_markdown(
        RESULTS_DIR / "oracle_opportunity_map.md",
        [
            "# Oracle Opportunity Map",
            "",
            "Oracle definitions are locked to moderate stress: `neutral_fill_haircut=0.5`, `toxic_multiplier=1.5`, `delay=0`.",
            "`oracle_positive_quote` means action-level realized stress-adjusted quote PnL is greater than zero.",
            "The oracle is non-tradable, uses future information, and is only an upper bound.",
        ],
    )


if __name__ == "__main__":
    run()
    print(f"Wrote oracle opportunity outputs to {RESULTS_DIR}")
