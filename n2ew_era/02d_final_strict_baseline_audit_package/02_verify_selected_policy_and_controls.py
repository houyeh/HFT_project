from __future__ import annotations

import pandas as pd

from common_phase02d import (
    RESULTS,
    SELECTED_HORIZON,
    SELECTED_POLICY_ID,
    ensure_dirs,
    exact_random_rows,
    selected_control_table,
)


def main() -> None:
    ensure_dirs()
    table = selected_control_table()
    if table.empty:
        out = pd.DataFrame()
    else:
        exact = exact_random_rows()
        out = table[
            [
                "split",
                "policy_id",
                "horizon",
                "total_pnl",
                "selected_action_count",
                "selection_rate",
                "win_rate",
                "max_drawdown",
                "old_bid_only_pnl",
                "matched_random_mean",
                "matched_random_95p",
                "matched_random_99p",
                "empirical_p_value_random_ge_model",
            ]
        ].copy()
        if not exact.empty:
            exact_small = exact[
                [
                    "split",
                    "policy_id",
                    "horizon",
                    "exact_random_mean",
                    "exact_random_95p",
                    "exact_random_99p",
                    "empirical_p_value_exact_random_ge_model",
                ]
            ].drop_duplicates(["split", "policy_id", "horizon"])
            out = out.merge(exact_small, on=["split", "policy_id", "horizon"], how="left")
            out["control_random_mean"] = out["exact_random_mean"].fillna(out["matched_random_mean"])
            out["control_random_95p"] = out["exact_random_95p"].fillna(out["matched_random_95p"])
            out["control_random_99p"] = out["exact_random_99p"].fillna(out["matched_random_99p"])
            out["control_random_p_value"] = out["empirical_p_value_exact_random_ge_model"].fillna(out["empirical_p_value_random_ge_model"])
            out["random_source"] = out["exact_random_mean"].notna().map({True: "exact_without_replacement_matched_random", False: "phase02b_finite_population_matched_random"})
        else:
            out["control_random_mean"] = out["matched_random_mean"]
            out["control_random_95p"] = out["matched_random_95p"]
            out["control_random_99p"] = out["matched_random_99p"]
            out["control_random_p_value"] = out["empirical_p_value_random_ge_model"]
            out["random_source"] = "phase02b_finite_population_matched_random"
        out["selected_minus_old_bid_only"] = out["total_pnl"].astype(float) - out["old_bid_only_pnl"].astype(float)
        out["selected_minus_random_mean"] = out["total_pnl"].astype(float) - out["control_random_mean"].astype(float)
        out["selected_minus_random_95p"] = out["total_pnl"].astype(float) - out["control_random_95p"].astype(float)
        out["beats_old_bid_only"] = out["selected_minus_old_bid_only"] > 0.0
        out["beats_random_95p"] = out["selected_minus_random_95p"] > 0.0
        out["selection_split"] = "val"
        out["test_used_for_selection"] = False
        out["new4_used_for_selection"] = False
    out.to_csv(RESULTS / "selected_policy_vs_controls.csv", index=False)
    print(f"Verified selected policy `{SELECTED_POLICY_ID}` H{SELECTED_HORIZON} against controls.")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
