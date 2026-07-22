from __future__ import annotations

import numpy as np
import pandas as pd

from common_033 import (
    RESULTS,
    add_lagged_spreads,
    decision_source_audit,
    delayed_entry_values,
    ensure_dirs,
    load_all_eval_actions,
    write_markdown,
)


def main() -> None:
    ensure_dirs()
    actions = add_lagged_spreads(load_all_eval_actions(), [0, 1])
    rows = []
    boundary_rows = []
    for eval_group, group in actions.groupby("eval_group", sort=True):
        group = group.sort_values(["symbol", "session_id", "local_row_id", "side"]).reset_index(drop=True)
        side_bid = group["side"].astype(str) == "bid"
        values = group["quote_ev_moderate"].astype(float).to_numpy()
        current_selected = (side_bid & (group["spread_lag_0"].astype(float) >= 1.0)).to_numpy(dtype=bool)
        previous_selected = (side_bid & (group["spread_lag_1"].astype(float) >= 1.0)).to_numpy(dtype=bool)
        delayed_values, missing = delayed_entry_values(group, previous_selected, 1)
        setups = [
            ("A_current_row_decision_current_entry", current_selected, values, 0, "Uses row t spread. Kept only as legacy proxy timing reference."),
            ("B_previous_row_decision_current_entry", previous_selected, values, 0, "Uses row t-1 spread for row t entry; decision feature is available before entry."),
            ("C_previous_row_decision_delayed_entry", previous_selected, delayed_values, missing, "Uses row t-1 spread and row t+1 proxy entry value; strictest timing stress here."),
        ]
        for setup, selected, value_array, missing_count, interpretation in setups:
            rows.append(
                {
                    "eval_group": eval_group,
                    "setup": setup,
                    "selected_quote_count": int(selected.sum()),
                    "selected_quote_pnl": float(value_array[selected].sum()),
                    "missing_delayed_entry_count": int(missing_count),
                    "uses_current_row_book_for_decision": setup.startswith("A_"),
                    "uses_previous_row_book_for_decision": setup.startswith(("B_", "C_")),
                    "uses_future_or_outcome_in_decision": False,
                    "interpretation": interpretation,
                }
            )
        first_rows = (
            group[["symbol", "session_id", "local_row_id", "spread_lag_1"]]
            .drop_duplicates(["symbol", "session_id", "local_row_id"])
            .sort_values(["symbol", "session_id", "local_row_id"])
            .groupby(["symbol", "session_id"], sort=True)
            .head(1)
        )
        boundary_rows.append(
            {
                "eval_group": eval_group,
                "session_count": int(first_rows.shape[0]),
                "first_row_lag1_non_null_count": int(first_rows["spread_lag_1"].notna().sum()),
                "session_boundary_lag_reset_pass": bool(first_rows["spread_lag_1"].isna().all()),
            }
        )
    timing = pd.DataFrame(rows)
    boundary = pd.DataFrame(boundary_rows)
    source = pd.DataFrame([decision_source_audit()])
    timing.to_csv(RESULTS / "strict_timing_rebuild_audit.csv", index=False)
    boundary.to_csv(RESULTS / "strict_timing_boundary_audit.csv", index=False)
    source.to_csv(RESULTS / "strict_timing_source_audit.csv", index=False)
    write_markdown(
        RESULTS / "strict_timing_rebuild_audit.md",
        [
            "# Strict Timing Rebuild Audit",
            "",
            "This audit separates three questions that were conflated in Phase 032:",
            "",
            "- A: current-row book decision with current proxy EV. This is retained only as a legacy timing reference.",
            "- B: previous-row book decision with current proxy EV. Decision features are available before the entry row.",
            "- C: previous-row book decision with one-tick delayed-entry proxy EV. This is the strictest timing rebuild in this audit.",
            "",
            "No setup uses `quote_ev`, `future_*`, toxicity labels, oracle labels, or PnL columns inside the decision rule. Those fields are used only after selection for evaluation.",
            "",
            f"Decision source no-future-token pass: `{bool(source.loc[0, 'passes_no_future_token_check'])}`.",
            f"Session-boundary lag reset pass across groups: `{bool(boundary['session_boundary_lag_reset_pass'].all())}`.",
        ],
    )


if __name__ == "__main__":
    main()
    print(f"Wrote strict timing rebuild audit to {RESULTS}")
