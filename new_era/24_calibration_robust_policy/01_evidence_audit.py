from __future__ import annotations

import pandas as pd

from calibration_utils import PROJECT_ROOT, RESULTS_DIR, ensure_dirs


V23_RESULTS = PROJECT_ROOT / "new_era" / "23_root_cause_stress_profitability" / "results"
V21_ORACLE = PROJECT_ROOT / "new_era" / "21_stress_robust_policy" / "oracle" / "oracle_perfect_upper_bounds.csv"
V21_EVAL = PROJECT_ROOT / "new_era" / "21_stress_robust_policy" / "evaluation" / "frozen_stress_robust_policy_test.csv"


def _metric() -> dict[str, str]:
    out: dict[str, str] = {}
    opp = pd.read_csv(V23_RESULTS / "oracle_opportunity_map.csv")
    top5 = opp.loc[opp["top_fraction"].round(2) == 0.05].iloc[0]
    out["top5"] = f"top 5% quote actions captured {top5['share_of_oracle_positive_pnl']:.3f} of positive oracle PnL"
    lift = pd.read_csv(V23_RESULTS / "within_spread_ml_lift.csv")
    inc = lift.groupby("score_name")["incremental_pnl_vs_same_bucket_random"].sum().sort_values(ascending=False)
    out["lift"] = f"best within-spread incremental PnL vs matched random: {inc.index[0]} = {inc.iloc[0]:.3f}"
    rolling = pd.read_csv(V23_RESULTS / "rolling_quantile_test_diagnostic.csv")
    best_diag = rolling.sort_values("selected_quote_pnl", ascending=False).iloc[0]
    selected = rolling[rolling["is_selected_validation_policy"].astype(bool)].iloc[0]
    out["diag_sweep"] = f"best test diagnostic sweep {best_diag['policy']} = {best_diag['selected_quote_pnl']:.3f}; validation-selected row = {selected['selected_quote_pnl']:.3f}"
    oracle = pd.read_csv(V21_ORACLE)
    oracle_row = oracle[(oracle["policy"] == "perfect_quote_pnl_filter") & (oracle["split"] == "test") & (oracle["stress_name"] == "moderate")].iloc[0]
    out["oracle"] = f"non-tradable oracle moderate PnL = {oracle_row['net_pnl']:.3f}"
    eval_df = pd.read_csv(V21_EVAL)
    causal = eval_df[(eval_df["policy"] == "021_selected") & (eval_df["split"] == "test") & (eval_df["stress_name"] == "moderate")].iloc[0]
    out["causal_fail"] = f"021 selected causal policy moderate PnL = {causal['net_pnl']:.3f}"
    return out


def run() -> None:
    ensure_dirs()
    m = _metric()
    rows = [
        {
            "claim": "Oracle opportunity exists under moderate stress",
            "supporting_metric": m["oracle"],
            "split_used": "burned diagnostic test",
            "test_used_for_selection": False,
            "same_universe": True,
            "same_bucket_random_control": False,
            "session_stability_checked": True,
            "side_stability_checked": True,
            "status": "evidence_of_upper_bound",
            "safe_wording": "Evidence that positive moderate-stress PnL is feasible for a non-tradable future-information oracle, not proof of a causal policy.",
        },
        {
            "claim": "Oracle opportunity is learnable",
            "supporting_metric": m["top5"],
            "split_used": "burned diagnostic test",
            "test_used_for_selection": False,
            "same_universe": True,
            "same_bucket_random_control": False,
            "session_stability_checked": True,
            "side_stability_checked": True,
            "status": "evidence_of_structure_not_proof",
            "safe_wording": "Evidence of concentrated structure in oracle-positive quotes; not absolute proof that causal features can learn it.",
        },
        {
            "claim": "ML ranking has incremental value inside spread buckets",
            "supporting_metric": m["lift"],
            "split_used": "burned diagnostic test",
            "test_used_for_selection": False,
            "same_universe": True,
            "same_bucket_random_control": True,
            "session_stability_checked": True,
            "side_stability_checked": True,
            "status": "evidence_of_incremental_value",
            "safe_wording": "Evidence that ML ranking can beat matched same-bucket random controls; not a profitability guarantee.",
        },
        {
            "claim": "Static absolute thresholds fail across regimes",
            "supporting_metric": m["causal_fail"],
            "split_used": "validation-selected policy, burned diagnostic test evaluation",
            "test_used_for_selection": False,
            "same_universe": True,
            "same_bucket_random_control": True,
            "session_stability_checked": True,
            "side_stability_checked": True,
            "status": "supported_diagnostic",
            "safe_wording": "Evidence of calibration shift: validation-selected static/EV thresholds did not transfer to the burned diagnostic test.",
        },
        {
            "claim": "+275k rolling threshold test sweep proves profitability",
            "supporting_metric": m["diag_sweep"],
            "split_used": "burned diagnostic test sweep",
            "test_used_for_selection": True,
            "same_universe": True,
            "same_bucket_random_control": False,
            "session_stability_checked": True,
            "side_stability_checked": True,
            "status": "diagnostic_only_not_selectable",
            "safe_wording": "The +275k row is diagnostic-only. It may motivate rolling calibration but cannot be used as a selected-policy claim.",
        },
    ]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "023_evidence_audit_table.csv", index=False)


if __name__ == "__main__":
    run()
    print(f"Wrote {RESULTS_DIR / '023_evidence_audit_table.csv'}")

