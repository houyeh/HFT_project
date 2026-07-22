from __future__ import annotations

import numpy as np
import pandas as pd

from common_phase02e import (
    FROZEN_SPLITS,
    OLD_BID_POLICY_ID,
    PHASE02D_DIR,
    RESULTS,
    ROOT,
    SELECTED_POLICY_ID,
    ensure_dirs,
    write_markdown,
)


def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def policy_pass_all_splits(df: pd.DataFrame, policy_id: str, column: str) -> bool:
    if df.empty or column not in df:
        return False
    rows = df[(df["policy_id"].astype(str) == policy_id) & (df["split"].astype(str).isin(FROZEN_SPLITS))]
    return bool(rows[column].astype(bool).all()) if not rows.empty else False


def display_pvalues(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "empirical_p_value_random_ge_model" not in df:
        return df
    out = df.copy()
    out["empirical_p_value_random_ge_model"] = out["empirical_p_value_random_ge_model"].map(
        lambda value: "<0.001" if pd.notna(value) and float(value) == 0.0 else value
    )
    return out


def main() -> None:
    ensure_dirs()
    new3 = read_csv(RESULTS / "new3_frozen_oos_results.csv")
    seq = read_csv(RESULTS / "sequential_inventory_backtest.csv")
    stress = read_csv(RESULTS / "strict_execution_stress_comparison.csv")
    decomp = read_csv(RESULTS / "old_bid_only_pnl_decomposition.csv")
    selected_controls = read_csv(PHASE02D_DIR / "results" / "selected_policy_vs_controls.csv")
    new3_audit = read_csv(RESULTS / "new3_dedup_audit.csv")

    rows = []
    for policy_id in [OLD_BID_POLICY_ID, SELECTED_POLICY_ID]:
        if policy_id == SELECTED_POLICY_ID and not selected_controls.empty:
            selected_oos = selected_controls[selected_controls["split"].astype(str).isin(["test", "new4"])]
            beats_random_test_new4 = bool(selected_oos["beats_random_95p"].astype(bool).all()) if not selected_oos.empty else False
        else:
            beats_random_test_new4 = False
        new3_row = new3[new3["policy_id"].astype(str) == policy_id] if not new3.empty else pd.DataFrame()
        beats_random_new3 = bool(new3_row["beats_matched_random_95p"].astype(bool).all()) if not new3_row.empty and "beats_matched_random_95p" in new3_row else False
        fee5 = stress[
            (stress["policy_id"].astype(str) == policy_id)
            & (stress["split"].astype(str).isin(FROZEN_SPLITS))
            & (stress["fee_multiplier"].astype(float) == 5.0)
            & (stress["positive_fill_keep_rate"].astype(float) == 1.0)
            & (stress["adverse_multiplier"].astype(float) == 1.0)
            & (stress["delay_events"].astype(int) == 0)
        ] if not stress.empty else pd.DataFrame()
        survives_fee5 = bool(fee5["positive_under_stress"].astype(bool).all()) if not fee5.empty else False
        moderate = stress[
            (stress["policy_id"].astype(str) == policy_id)
            & (stress["split"].astype(str).isin(FROZEN_SPLITS))
            & (stress["fee_multiplier"].astype(float) == 1.0)
            & (stress["positive_fill_keep_rate"].astype(float) == 0.5)
            & (stress["adverse_multiplier"].astype(float) == 1.5)
            & (stress["delay_events"].astype(int) == 0)
        ] if not stress.empty else pd.DataFrame()
        survives_adverse_fill = bool(moderate["positive_under_stress"].astype(bool).all()) if not moderate.empty else False
        strict_seq = seq[
            (seq["policy_id"].astype(str) == policy_id)
            & (seq["split"].astype(str).isin(FROZEN_SPLITS))
            & (seq["fill_mode"].astype(str) == "conservative_adverse_fill")
            & (seq["inventory_cap"].astype(str) == "1")
            & (seq["exit_rule"].astype(str) == "fixed_horizon_H20")
        ] if not seq.empty else pd.DataFrame()
        random_seq = seq[
            (seq["policy_id"].astype(str) == "matched_random_95p")
            & (seq["split"].astype(str).isin(FROZEN_SPLITS))
            & (seq["fill_mode"].astype(str) == "conservative_adverse_fill")
            & (seq["inventory_cap"].astype(str) == "1")
            & (seq["exit_rule"].astype(str) == "fixed_horizon_H20")
        ] if not seq.empty else pd.DataFrame()
        inventory_cap_ok = bool((strict_seq["sequential_pnl"].astype(float) > 0.0).all()) if not strict_seq.empty else False
        strict_seq_min_pnl = float(strict_seq["sequential_pnl"].min()) if not strict_seq.empty else np.nan
        if not strict_seq.empty and not random_seq.empty:
            joined_seq = strict_seq[["split", "sequential_pnl"]].merge(
                random_seq[["split", "sequential_pnl"]].rename(columns={"sequential_pnl": "matched_random_95p_sequential_pnl"}),
                on="split",
                how="left",
            )
            beats_seq_random = bool((joined_seq["sequential_pnl"].astype(float) > joined_seq["matched_random_95p_sequential_pnl"].astype(float)).all())
        else:
            joined_seq = pd.DataFrame()
            beats_seq_random = False
        rows.append(
            {
                "policy_id": policy_id,
                "beats_matched_random_test_new4": beats_random_test_new4 if policy_id == SELECTED_POLICY_ID else np.nan,
                "beats_matched_random_new3": beats_random_new3,
                "beats_sequential_matched_random_95p_cap1_H20": beats_seq_random,
                "survives_fee_5x_all_frozen": survives_fee5,
                "survives_moderate_adverse_fill_all_frozen": survives_adverse_fill,
                "inventory_cap_1_fixed_H20_positive_all_frozen": inventory_cap_ok,
                "strict_sequential_min_pnl_cap1_H20": strict_seq_min_pnl,
                "min_gap_vs_sequential_random_cap1_H20": float(
                    (joined_seq["sequential_pnl"].astype(float) - joined_seq["matched_random_95p_sequential_pnl"].astype(float)).min()
                ) if not joined_seq.empty else np.nan,
                "strategy_candidate_pass": bool(
                    (beats_random_test_new4 or policy_id == OLD_BID_POLICY_ID)
                    and beats_random_new3
                    and beats_seq_random
                    and survives_fee5
                    and survives_adverse_fill
                    and inventory_cap_ok
                ),
            }
        )
    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(RESULTS / "non_ml_strategy_feasibility_scorecard.csv", index=False)

    valid_new3 = bool((new3_audit["classification"].astype(str) == "valid_frozen_oos").all()) if not new3_audit.empty else False
    any_strategy = bool(scorecard["strategy_candidate_pass"].astype(bool).any()) if not scorecard.empty else False
    if any_strategy:
        status = "final_strategy_candidate_found"
        decision_text = "At least one non-ML rule survives the full 02E scorecard. It can be treated as a proxy strategy candidate, still not real-world profitability."
    else:
        status = "signal_found_but_strategy_not_proven"
        decision_text = "The current non-ML results should be classified as signal discovery, not a final strategy. The proxy environment was too weak."

    h20_decomp = decomp[
        (decomp["session_id"].astype(str) == "__ALL__")
        & (decomp["horizon"].astype(int) == 20)
    ] if not decomp.empty else pd.DataFrame()
    lines = [
        "# Final Non-ML Feasibility Decision Report",
        "",
        f"Final classification: `{status}`.",
        "",
        decision_text,
        "",
        "No ML was trained, and no parameters were tuned on `test`, `NEW3`, or `NEW4`. The selected L2 queue rule remains the fixed Phase 02D rule: `side_by_queue_imbalance_abs_0p1` at `H20`.",
        "",
        "## 1. NEW3 Frozen OOS Validity",
        "",
        f"NEW3 validity audit pass: `{valid_new3}`.",
        "",
        new3_audit.to_markdown(index=False) if not new3_audit.empty else "NEW3 audit missing.",
        "",
        "## 2. NEW3 Frozen Results",
        "",
        display_pvalues(new3).to_markdown(index=False) if not new3.empty else "NEW3 frozen results missing.",
        "",
        "## 3. Old Bid-Only PnL Decomposition",
        "",
        h20_decomp.to_markdown(index=False) if not h20_decomp.empty else "Old bid-only decomposition missing.",
        "",
        "Interpretation: if half-spread advantage dominates, old bid-only is likely benefiting from the full-action markout simulator assuming quote opportunities without true fill probability or queue priority.",
        "",
        "## 4. Sequential Inventory Proxy",
        "",
        "Conservative view: `fill_mode=conservative_adverse_fill`, `inventory_cap=1`, `exit_rule=fixed_horizon_H20`.",
        "",
        "The strategy-candidate hurdle requires the policy to stay positive **and** beat the `matched_random_95p` sequential proxy comparator under this conservative view.",
        "",
        seq[
            (seq["fill_mode"].astype(str) == "conservative_adverse_fill")
            & (seq["inventory_cap"].astype(str) == "1")
            & (seq["exit_rule"].astype(str) == "fixed_horizon_H20")
        ].to_markdown(index=False) if not seq.empty else "Sequential inventory output missing.",
        "",
        "## 5. Strict Execution Stress",
        "",
        "Moderate stress for this phase is `fee_multiplier=1`, `positive_fill_keep_rate=0.5`, `adverse_multiplier=1.5`, `delay_events=0`. Fee stress also checks `fee_multiplier=5`.",
        "",
        stress[stress["moderate_stress"].astype(bool)].to_markdown(index=False) if not stress.empty else "Strict stress output missing.",
        "",
        "## 6. Feasibility Scorecard",
        "",
        scorecard.to_markdown(index=False),
        "",
        "## 7. Final Decision",
        "",
        decision_text,
        "",
        "Allowed wording: `non-trivial L2 signal discovery`, `proxy quote-control primitive`, and `candidate for stricter execution modeling`.",
        "",
        "Forbidden wording: do not claim true queue replay, real-world profitability, or FPGA-driven PnL improvement.",
    ]
    write_markdown(ROOT / "FINAL_NON_ML_FEASIBILITY_DECISION_REPORT.md", lines)
    print("Feasibility scorecard and final decision report complete.")
    print(scorecard.to_string(index=False))
    print(f"Final classification: {status}")


if __name__ == "__main__":
    main()
