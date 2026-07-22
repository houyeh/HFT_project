from __future__ import annotations

import pandas as pd

from common_031 import RESULTS, ensure_dirs, eval_selection, load_full_actions, split_df, write_markdown


def main() -> None:
    ensure_dirs()
    actions = load_full_actions()
    rows: list[dict] = []
    for split in ["train", "val", "test", "new4"]:
        df = split_df(actions, split)
        baseline = df["selected_by_baseline"].astype(bool).to_numpy()
        row = eval_selection(df, baseline, "bid_only_spread_ge_1")
        row["phase031_split"] = split
        row["selection_role"] = "frozen_baseline"
        row["test_used_for_selection"] = False
        rows.append(row)
        for side, group in df.groupby("side", observed=False):
            side_row = eval_selection(group, group["selected_by_baseline"].astype(bool).to_numpy(), "bid_only_spread_ge_1")
            side_row["phase031_split"] = split
            side_row["side"] = side
            side_row["selection_role"] = "frozen_baseline_by_side"
            side_row["test_used_for_selection"] = False
            rows.append(side_row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "baseline_oracle_coverage.csv", index=False)

    lines = [
        "# Baseline Coverage vs Oracle",
        "",
        "Oracle-positive quote means `quote_ev_moderate > 0`. The oracle is non-tradable and uses future information; this table measures opportunity cost only.",
        "",
    ]
    overall = out[out["selection_role"] == "frozen_baseline"].copy()
    for _, row in overall.iterrows():
        lines.append(
            f"- `{row['phase031_split']}`: baseline PnL `{row['selected_quote_pnl']:.0f}`, captured oracle-positive PnL `{row['captured_oracle_positive_pnl']:.0f}`, "
            f"missed oracle-positive PnL `{row['missed_oracle_positive_pnl']:.0f}`, avoided negative EV `{row['avoided_negative_ev_pnl']:.0f}`."
        )
    lines.extend(
        [
            "",
            "Interpretation: high missed oracle-positive PnL means the bid-only spread baseline is conservative. Whether that opportunity is tradable depends on causal selector performance, not the oracle itself.",
        ]
    )
    write_markdown(RESULTS / "baseline_oracle_coverage.md", lines)


if __name__ == "__main__":
    main()
    print(f"Wrote baseline/oracle coverage to {RESULTS}")

