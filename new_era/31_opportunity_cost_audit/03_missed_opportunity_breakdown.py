from __future__ import annotations

import pandas as pd

from common_031 import RESULTS, ensure_dirs, load_full_actions


def summarize(df: pd.DataFrame, group_cols: list[str], group_type: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby(group_cols, observed=False)
        .agg(
            missed_positive_count=("sample_id", "count"),
            missed_positive_pnl=("quote_ev_moderate", "sum"),
            avg_missed_positive_ev=("quote_ev_moderate", "mean"),
            median_missed_positive_ev=("quote_ev_moderate", "median"),
        )
        .reset_index()
    )
    out.insert(0, "group_type", group_type)
    for col in ["phase031_split", "side", "session_id", "spread_bucket", "event_rate_bucket", "volatility_bucket"]:
        if col not in out.columns:
            out[col] = ""
    return out[
        [
            "group_type",
            "phase031_split",
            "side",
            "session_id",
            "spread_bucket",
            "event_rate_bucket",
            "volatility_bucket",
            "missed_positive_count",
            "missed_positive_pnl",
            "avg_missed_positive_ev",
            "median_missed_positive_ev",
        ]
    ]


def main() -> None:
    ensure_dirs()
    actions = load_full_actions()
    missed = actions[(~actions["selected_by_baseline"].astype(bool)) & (actions["quote_ev_moderate"].astype(float) > 0.0)].copy()
    parts = [
        summarize(missed, ["phase031_split", "side"], "split_side"),
        summarize(missed, ["phase031_split", "spread_bucket"], "split_spread"),
        summarize(missed, ["phase031_split", "session_id"], "split_session"),
        summarize(missed, ["phase031_split", "side", "spread_bucket"], "split_side_spread"),
        summarize(missed, ["phase031_split", "side", "spread_bucket", "event_rate_bucket", "volatility_bucket"], "split_side_spread_event_vol"),
    ]
    pd.concat([p for p in parts if not p.empty], ignore_index=True).to_csv(RESULTS / "missed_opportunity_breakdown.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote missed opportunity breakdown to {RESULTS}")

