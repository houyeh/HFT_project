from __future__ import annotations

import numpy as np
import pandas as pd

from common_phase02d import (
    MODERATE_ADVERSE_MULTIPLIER,
    MODERATE_DELAY,
    MODERATE_FILL_KEEP,
    PHASE02B_DIR,
    PHASE02C_DIR,
    RESULTS,
    SELECTED_HORIZON,
    SELECTED_POLICY_ID,
    ensure_dirs,
    read_csv_if_exists,
)


ELIGIBILITY_NOTE = (
    "`eligible_for_final_selection_at_1x` is meaningful only on validation. "
    "Test and NEW4 are frozen evaluation splits and are not used for selection."
)


def main() -> None:
    ensure_dirs()
    fee = read_csv_if_exists(PHASE02B_DIR / "results" / "fee_sensitivity.csv")
    stress = read_csv_if_exists(PHASE02C_DIR / "results" / "queue_stress_grid.csv")
    break_even = read_csv_if_exists(PHASE02C_DIR / "results" / "fill_keep_break_even.csv")

    selected_fee = fee[
        (fee["policy_id"].astype(str) == SELECTED_POLICY_ID)
        & (fee["horizon"].astype(int) == SELECTED_HORIZON)
        & (fee["split"].astype(str).isin(["val", "test", "new4"]))
    ].copy() if not fee.empty else pd.DataFrame()
    if not selected_fee.empty:
        selected_fee["eligibility_column_note"] = ELIGIBILITY_NOTE
    selected_fee.to_csv(RESULTS / "fee_sensitivity_summary.csv", index=False)

    selected_stress = stress[
        (stress["policy_id"].astype(str) == SELECTED_POLICY_ID)
        & (stress["horizon"].astype(int) == SELECTED_HORIZON)
        & (stress["split"].astype(str).isin(["val", "test", "new4"]))
    ].copy() if not stress.empty else pd.DataFrame()
    moderate = selected_stress[
        (selected_stress["fill_keep_rate"].astype(float) == MODERATE_FILL_KEEP)
        & (selected_stress["adverse_multiplier"].astype(float) == MODERATE_ADVERSE_MULTIPLIER)
        & (selected_stress["action_delay_events"].astype(int) == MODERATE_DELAY)
    ].copy() if not selected_stress.empty else pd.DataFrame()
    if not moderate.empty:
        moderate["eligibility_column_note"] = ELIGIBILITY_NOTE
    moderate.to_csv(RESULTS / "queue_stress_selected_policy_summary.csv", index=False)

    if not selected_fee.empty:
        fee_break_even = (
            selected_fee.groupby(["split", "policy_id", "horizon"], as_index=False)
            .agg(
                base_total_pnl_fee_1x=("base_total_pnl_fee_1x", "first"),
                selected_action_count=("selected_action_count", "first"),
                break_even_fee_multiplier=("break_even_fee_multiplier", "first"),
                max_tested_fee_multiplier_positive=("fee_multiplier", lambda s: float(selected_fee.loc[s.index][selected_fee.loc[s.index, "positive_after_fee"].astype(bool)]["fee_multiplier"].max()) if selected_fee.loc[s.index, "positive_after_fee"].any() else np.nan),
            )
        )
    else:
        fee_break_even = pd.DataFrame()
    if not fee_break_even.empty:
        fee_break_even["eligibility_column_note"] = ELIGIBILITY_NOTE
    fee_break_even.to_csv(RESULTS / "fee_break_even_points.csv", index=False)

    if not break_even.empty:
        queue_break_even = break_even[
            break_even["split"].astype(str).isin(["test", "new4"])
        ].copy()
    else:
        queue_break_even = pd.DataFrame()
    queue_break_even.to_csv(RESULTS / "queue_fill_keep_break_even_points.csv", index=False)
    print("Fee and queue stress summaries complete.")
    print(moderate[["split", "stress_total_pnl", "selected_action_count", "win_rate", "max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
