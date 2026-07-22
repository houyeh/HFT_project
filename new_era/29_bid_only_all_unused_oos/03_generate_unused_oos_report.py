from __future__ import annotations

import pandas as pd

from common_032 import RESULTS, ROOT, ensure_dirs, write_markdown


def fmt(v: float) -> str:
    return f"{float(v):,.0f}"


def main() -> None:
    ensure_dirs()
    provenance = pd.read_csv(RESULTS / "training_selection_sessions.csv")
    inventory = pd.read_csv(RESULTS / "data_new_to_new4_session_inventory.csv")
    summary = pd.read_csv(RESULTS / "bid_only_unused_oos_summary_by_dir_symbol.csv")
    by_session = pd.read_csv(RESULTS / "bid_only_unused_oos_by_session_side.csv")

    eval_rows = summary[summary["status"].astype(str) == "evaluated_unused_oos"].copy()
    total_pnl = float(eval_rows["selected_quote_pnl"].sum()) if not eval_rows.empty else 0.0
    total_selected = int(eval_rows["selected_quote_count"].sum()) if not eval_rows.empty else 0
    total_actions = int(eval_rows["action_count"].sum()) if not eval_rows.empty else 0

    lines = [
        "# bid_only_spread_ge_1 Unused DATA OOS Report",
        "",
        "## 1. What Was Used For Training / Selection",
        "",
        "`bid_only_spread_ge_1` is a non-ML rule, but it was selected in Phase 026 using validation metrics. Therefore this audit excludes both train sessions and the validation-selection session.",
        "",
    ]
    for _, row in provenance.iterrows():
        lines.append(f"- `{row['session_id']}`: split `{row['split']}`, dataset `{row['dataset']}`, role `{row['used_for_bid_only_policy']}`.")
    lines.extend(["", "## 2. DATA/NEW Through DATA/NEW4 Inclusion Audit", ""])
    inv_summary = inventory.groupby(["data_dir", "included_in_unused_oos", "exclusion_reason"], observed=False).agg(files=("file", "count"), bytes=("bytes", "sum")).reset_index()
    for _, row in inv_summary.iterrows():
        lines.append(f"- `{row['data_dir']}` reason `{row['exclusion_reason']}` included `{row['included_in_unused_oos']}` files `{int(row['files'])}` bytes `{int(row['bytes'])}`.")
    lines.extend(["", "## 3. OOS Results", ""])
    lines.append(f"- Total evaluated action count: `{total_actions}`.")
    lines.append(f"- Total selected quote count: `{total_selected}`.")
    lines.append(f"- Total selected quote PnL: `{fmt(total_pnl)}`.")
    lines.append("")
    for _, row in eval_rows.sort_values(["data_dir", "symbol"]).iterrows():
        lines.append(
            f"- `{row['data_dir']}` `{row['symbol']}` sessions `{row.get('included_sessions', '')}`: "
            f"actions `{int(row['action_count'])}`, selected `{int(row['selected_quote_count'])}`, PnL `{fmt(row['selected_quote_pnl'])}`, "
            f"precision `{row['oracle_precision']:.3f}`."
        )
    lines.extend(["", "## 4. Session/Side Highlights", ""])
    if not by_session.empty:
        top = by_session.sort_values("selected_quote_pnl", ascending=False).head(10)
        bottom = by_session.sort_values("selected_quote_pnl", ascending=True).head(10)
        lines.append("Top selected PnL groups:")
        for _, row in top.iterrows():
            lines.append(f"- `{row['data_dir']}` `{row['session_id']}` `{row['side']}`: selected PnL `{fmt(row['selected_quote_pnl'])}`, selected `{int(row['selected_quote_count'])}`.")
        lines.append("")
        lines.append("Worst selected PnL groups:")
        for _, row in bottom.iterrows():
            lines.append(f"- `{row['data_dir']}` `{row['session_id']}` `{row['side']}`: selected PnL `{fmt(row['selected_quote_pnl'])}`, selected `{int(row['selected_quote_count'])}`.")
    lines.extend(
        [
            "",
            "## 5. Caveat",
            "",
            "PnL remains the same simulator-derived moderate-stress quote proxy used in prior phases. This is not true exchange-level queue replay or real-world profitability proof.",
        ]
    )
    write_markdown(ROOT / "BID_ONLY_ALL_UNUSED_DATA_OOS_REPORT.md", lines)


if __name__ == "__main__":
    main()
    print(f"Wrote report to {ROOT / 'BID_ONLY_ALL_UNUSED_DATA_OOS_REPORT.md'}")

