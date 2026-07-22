from __future__ import annotations

import pandas as pd

from common_028 import (
    DATA_DIR,
    RESULTS,
    baseline_universe,
    build_historical_actions_with_features,
    build_new4_actions_with_features,
    ensure_dirs,
    write_markdown,
)


def main() -> None:
    ensure_dirs()
    historical_parts = []
    summaries = []
    for split in ["train", "val", "test"]:
        actions = build_historical_actions_with_features(split)
        universe = baseline_universe(actions)
        historical_parts.append(universe)
        summaries.append(
            {
                "split": split,
                "baseline_quote_count": int(len(universe)),
                "quote_ev_sum_moderate": float(universe["quote_ev_moderate"].sum()),
                "negative_ev_rate_moderate": float(universe["negative_ev_moderate"].mean()) if len(universe) else 0.0,
                "side_set": ",".join(sorted(universe["side"].astype(str).unique())),
            }
        )

    hist = pd.concat(historical_parts, ignore_index=True)
    hist.to_parquet(DATA_DIR / "baseline_universe_quote_ev.parquet", index=False)

    new4_actions = build_new4_actions_with_features()
    new4 = baseline_universe(new4_actions)
    new4.to_parquet(DATA_DIR / "new4_baseline_universe_quote_ev.parquet", index=False)
    summaries.append(
        {
            "split": "external_oos_new4",
            "baseline_quote_count": int(len(new4)),
            "quote_ev_sum_moderate": float(new4["quote_ev_moderate"].sum()),
            "negative_ev_rate_moderate": float(new4["negative_ev_moderate"].mean()) if len(new4) else 0.0,
            "side_set": ",".join(sorted(new4["side"].astype(str).unique())),
        }
    )
    pd.DataFrame(summaries).to_csv(RESULTS / "baseline_universe_label_summary.csv", index=False)

    lines = [
        "# Residual Quote-EV Label Definition",
        "",
        "Universe: quotes selected by the frozen non-ML controller `bid_only_spread_ge_1`, i.e. bid-side quote actions with `spread_ticks >= 1`.",
        "",
        "Labels:",
        "",
        "- `quote_ev_moderate`: simulator-derived quote PnL under moderate stress for a quote that the baseline would keep.",
        "- `negative_ev_moderate = 1` when `quote_ev_moderate < 0`.",
        "",
        "Important safeguards:",
        "",
        "- The toxic label is not used for this residual model.",
        "- Labels use future outcomes and are training/evaluation targets only; they are not causal features.",
        "- NEW4 is external OOS only and is not used for model fitting or threshold selection.",
        "- Quote-EV remains a simulator-derived proxy, not true exchange queue-aware realized PnL.",
    ]
    write_markdown(DATA_DIR / "baseline_universe_quote_ev_label_definition.md", lines)


if __name__ == "__main__":
    main()
    print(f"Wrote baseline universe Quote-EV labels to {DATA_DIR}")

