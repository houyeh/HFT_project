from __future__ import annotations

import pandas as pd

from common_031 import DATA, FULL_ACTION_FILE, RESULTS, build_full_action_universe, ensure_dirs


def main() -> None:
    ensure_dirs()
    actions = build_full_action_universe()
    actions.to_parquet(FULL_ACTION_FILE, index=False)
    summary = (
        actions.groupby(["phase031_split", "side"], observed=False)
        .agg(
            action_count=("sample_id", "count"),
            selected_by_baseline_count=("selected_by_baseline", "sum"),
            positive_ev_count=("positive_ev_moderate", "sum"),
            negative_ev_count=("negative_ev_moderate", "sum"),
            total_quote_ev=("quote_ev_moderate", "sum"),
            positive_ev_pnl=("quote_ev_moderate", lambda s: float(s[s > 0].sum())),
            negative_ev_loss=("quote_ev_moderate", lambda s: float(-s[s < 0].sum())),
            avg_quote_ev=("quote_ev_moderate", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS / "full_action_universe_summary.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote full action universe to {DATA}")

