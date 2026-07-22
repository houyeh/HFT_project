from __future__ import annotations

import pandas as pd

from common_032 import DATA_DIR_NAMES, DATA_ROOT, RESULTS, ensure_dirs, evaluate_data_dir, excluded_session_map


def main() -> None:
    ensure_dirs()
    excluded = excluded_session_map()
    rows = []
    symbol_tables = []
    details = []
    for name in DATA_DIR_NAMES:
        data_dir = DATA_ROOT / name
        dir_rows, dir_details, symbol_table = evaluate_data_dir(data_dir, excluded)
        rows.extend(dir_rows)
        if not symbol_table.empty:
            symbol_tables.append(symbol_table)
        details.extend(dir_details)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "bid_only_unused_oos_summary_by_dir_symbol.csv", index=False)
    if symbol_tables:
        pd.concat(symbol_tables, ignore_index=True).to_csv(RESULTS / "raw_symbol_detection.csv", index=False)
    if details:
        all_detail = pd.concat(details, ignore_index=True)
        by_session = (
            all_detail.groupby(["data_dir", "symbol", "session_id", "side"], observed=False)
            .agg(
                action_count=("action_id", "count"),
                selected_quote_count=("selected_by_bid_only_spread_ge_1", "sum"),
                selected_quote_pnl=("quote_value_moderate", lambda s: float(s[all_detail.loc[s.index, "selected_by_bid_only_spread_ge_1"].astype(bool)].sum())),
                all_action_pnl=("quote_value_moderate", "sum"),
                positive_ev_count=("oracle_positive_quote", "sum"),
                avg_quote_ev=("quote_value_moderate", "mean"),
            )
            .reset_index()
        )
        by_session.to_csv(RESULTS / "bid_only_unused_oos_by_session_side.csv", index=False)
        all_detail.to_parquet(RESULTS / "bid_only_unused_oos_action_details.parquet", index=False)
    else:
        pd.DataFrame().to_csv(RESULTS / "bid_only_unused_oos_by_session_side.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Wrote unused OOS evaluation to {RESULTS}")

