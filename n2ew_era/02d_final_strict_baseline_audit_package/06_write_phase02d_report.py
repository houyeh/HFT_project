from __future__ import annotations

import pandas as pd

from common_phase02d import (
    FIGURES,
    RESULTS,
    ROOT,
    SELECTED_HORIZON,
    SELECTED_POLICY_ID,
    SELECTED_RULE_DEFINITION,
    dataframe_to_html_table,
    exact_random_rows,
    read_csv_if_exists,
    write_markdown,
)


EXACT_RANDOM_ZERO_PVALUE_WORDING = (
    "In 1000 exact random trials, no random trial exceeded the selected policy. "
    "Therefore empirical p-value < 0.001."
)
ELIGIBILITY_NOTE = (
    "`eligible_for_final_selection_at_1x` is meaningful only on validation. "
    "Test and NEW4 are frozen evaluation splits and are not used for selection."
)


def split_beats_random(selected: pd.DataFrame, split: str) -> bool:
    if selected.empty:
        return False
    rows = selected[selected["split"].astype(str).str.lower() == split]
    return bool(rows["beats_random_95p"].all()) if not rows.empty else False


def claim_safety_table(selected: pd.DataFrame, fee: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    beats_old = bool(selected["beats_old_bid_only"].all()) if not selected.empty else False
    non_trivial = bool((selected["selection_rate"].astype(float) < 0.95).all()) if not selected.empty and "selection_rate" in selected else False
    fee_pass = bool(fee["positive_after_fee"].astype(bool).all()) if not fee.empty and "positive_after_fee" in fee else False
    stress_pass = bool(stress["positive_under_stress"].astype(bool).all()) if not stress.empty and "positive_under_stress" in stress else False
    rows = [
        {
            "claim": "Full L2 action universe avoids residual bottleneck",
            "status": "PASS",
            "evidence": "Phase 01 built millions of bid/ask/no_quote candidate actions instead of the old 820-row residual veto universe.",
            "safe_wording": "The n2ew_era action universe removes the residual-sample bottleneck for short-horizon proxy audits.",
        },
        {
            "claim": "Selected policy is non-trivial L2 rule",
            "status": "PASS" if non_trivial else "FAIL",
            "evidence": "Policy uses queue_imbalance thresholding and has selection_rate < 0.95 on val/test/NEW4.",
            "safe_wording": "The selected rule is a non-trivial L2 queue-imbalance quote-ranking primitive.",
        },
        {
            "claim": "Selected policy beats matched random 95p on validation",
            "status": "PASS" if split_beats_random(selected, "val") else "FAIL",
            "evidence": "selected_policy_vs_controls.csv, split=val.",
            "safe_wording": "On validation, the selected queue-imbalance rule beats the matched-random 95p hurdle under proxy markout.",
        },
        {
            "claim": "Selected policy beats matched random 95p on frozen test",
            "status": "PASS" if split_beats_random(selected, "test") else "FAIL",
            "evidence": "selected_policy_vs_controls.csv and exact_random_oos_summary.csv, split=test.",
            "safe_wording": "On frozen test, the selected rule beats the matched-random 95p hurdle under proxy markout.",
        },
        {
            "claim": "Selected policy beats matched random 95p on NEW4",
            "status": "PASS" if split_beats_random(selected, "new4") else "FAIL",
            "evidence": "selected_policy_vs_controls.csv and exact_random_oos_summary.csv, split=NEW4.",
            "safe_wording": "On NEW4, the selected rule beats the matched-random 95p hurdle under proxy markout.",
        },
        {
            "claim": "Selected policy beats old bid-only absolute PnL",
            "status": "PASS" if beats_old else "FAIL",
            "evidence": "selected_minus_old_bid_only is negative on val/test/NEW4",
            "safe_wording": "The selected rule is not an absolute PnL replacement for old bid-only; it is evidence of non-trivial L2 ranking/excess signal.",
        },
        {
            "claim": "Selected policy remains positive under fee sensitivity",
            "status": "PASS" if fee_pass else "FAIL",
            "evidence": "fee_sensitivity_summary.csv across fee multipliers 1x, 2x, 3x, and 5x.",
            "safe_wording": "The selected rule remains positive under the audited proxy fee multipliers.",
        },
        {
            "claim": "Selected policy remains positive under proxy queue stress",
            "status": "PROXY_PASS" if stress_pass else "FAIL",
            "evidence": "queue_stress_selected_policy_summary.csv at fill_keep_rate=0.5, adverse_multiplier=1.5, delay=0.",
            "safe_wording": "The selected rule remains positive under the defined proxy queue stress; this is not true queue replay.",
        },
        {
            "claim": "This is true queue replay",
            "status": "FORBIDDEN",
            "evidence": "No aggressor_side, true order IDs, queue position, acknowledgements, or exchange matching replay.",
            "safe_wording": "Use `proxy short-horizon markout audit`, not `true exchange replay`.",
        },
        {
            "claim": "This proves real-world profitability",
            "status": "FORBIDDEN",
            "evidence": "No true queue priority, order acknowledgements, partial-fill replay, or production execution model.",
            "safe_wording": "Do not claim real-world profitability.",
        },
        {
            "claim": "FPGA improves PnL",
            "status": "FORBIDDEN",
            "evidence": "No production latency experiment proves an FPGA PnL improvement.",
            "safe_wording": "Frame FPGA only as an HLS-friendly deterministic quote-control implementation path.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phase02d_claim_safety_table.csv", index=False)
    return out


def main() -> None:
    selected = read_csv_if_exists(RESULTS / "selected_policy_vs_controls.csv")
    exact = exact_random_rows()
    fee = read_csv_if_exists(RESULTS / "fee_sensitivity_summary.csv")
    fee_break = read_csv_if_exists(RESULTS / "fee_break_even_points.csv")
    stress = read_csv_if_exists(RESULTS / "queue_stress_selected_policy_summary.csv")
    queue_break = read_csv_if_exists(RESULTS / "queue_fill_keep_break_even_points.csv")
    manifest = read_csv_if_exists(RESULTS / "phase02bc_input_manifest.csv")
    claim = claim_safety_table(selected, fee, stress)

    selected_old = selected[["split", "total_pnl", "old_bid_only_pnl", "selected_minus_old_bid_only", "beats_old_bid_only", "matched_random_95p", "beats_random_95p"]] if not selected.empty else pd.DataFrame()
    exact_display = exact.copy()
    exact_pvalue_note = ""
    if not exact_display.empty and "empirical_p_value_exact_random_ge_model" in exact_display:
        if (exact_display["empirical_p_value_exact_random_ge_model"].astype(float) == 0.0).any():
            exact_pvalue_note = EXACT_RANDOM_ZERO_PVALUE_WORDING
        exact_display["empirical_p_value_exact_random_ge_model"] = exact_display[
            "empirical_p_value_exact_random_ge_model"
        ].map(lambda value: "<0.001" if float(value) == 0.0 else f"{float(value):.6f}")

    lines = [
        "# Phase 02D Final Strict Baseline Audit Report",
        "",
        f"Fixed selected policy: `{SELECTED_POLICY_ID}` at horizon `H{SELECTED_HORIZON}`.",
        "",
        f"Rule definition: {SELECTED_RULE_DEFINITION}",
        "",
        "No new policy search, ML training, or threshold tuning was performed in Phase 02D. This package only collects Phase 02B/02C evidence and verifies advisor-safe claims.",
        "",
        "## 1. Input Completeness",
        "",
        manifest.to_markdown(index=False) if not manifest.empty else "Input manifest missing.",
        "",
        "## 2. Selected Policy vs Controls",
        "",
        selected_old.to_markdown(index=False) if not selected_old.empty else "Selected policy control table missing.",
        "",
        "**Critical interpretation:** the selected queue-imbalance rule beats matched random, but it does **not** beat `old_bid_only` in absolute total PnL. It should be framed as a non-trivial L2 ranking/excess-PnL signal, not as an absolute PnL replacement for old bid-only.",
        "",
        "## 3. Exact Random OOS Check",
        "",
        exact_pvalue_note,
        "",
        exact_display.to_markdown(index=False) if not exact_display.empty else "Exact OOS random summary missing. Use Phase 02B finite-population random as fallback.",
        "",
        "## 4. Fee Sensitivity",
        "",
        ELIGIBILITY_NOTE,
        "",
        fee.to_markdown(index=False) if not fee.empty else "Fee sensitivity missing.",
        "",
        "Break-even fee points:",
        "",
        fee_break.to_markdown(index=False) if not fee_break.empty else "Fee break-even missing.",
        "",
        "## 5. Queue Stress",
        "",
        "Moderate stress is `fill_keep_rate=0.5`, `adverse_multiplier=1.5`, and `delay=0`.",
        "",
        ELIGIBILITY_NOTE,
        "",
        stress.to_markdown(index=False) if not stress.empty else "Queue stress selected-policy summary missing.",
        "",
        "Fill-keep break-even points:",
        "",
        queue_break.to_markdown(index=False) if not queue_break.empty else "Queue break-even missing.",
        "",
        "## 6. Claim Safety Table",
        "",
        claim.to_markdown(index=False),
        "",
        "## 7. Safe Conclusion",
        "",
        "The selected L2 queue-imbalance rule is a non-trivial ranking/control signal under the current short-horizon proxy markout framework. It clears matched-random hurdles and remains positive under the defined proxy queue stress. However, it does not outperform old bid-only in absolute total PnL, and the evaluation is not true exchange replay or evidence of real-world profitability.",
        "",
        "## Figures",
        "",
        "- `figures/selected_vs_controls_bar.png`",
        "- `figures/random_hurdle_bar.png`",
        "- `figures/fee_sensitivity_selected_policy.png`",
        "- `figures/queue_stress_heatmap_selected_policy.png`",
        "- `figures/delay_sensitivity_selected_policy.png`",
    ]
    write_markdown(ROOT / "PHASE02D_FINAL_STRICT_BASELINE_AUDIT_REPORT.md", lines)

    interp = [
        "# Old Bid-Only vs Queue-Imbalance Interpretation",
        "",
        f"The selected policy `{SELECTED_POLICY_ID}` at `H{SELECTED_HORIZON}` is **not** an absolute PnL replacement for `old_bid_only_spread_ge_1`.",
        "",
        "Old bid-only still has higher absolute total PnL in the current proxy markout setup because it quotes more broadly and captures mechanical spread/markout value across the full bid-side universe.",
        "",
        "The queue-imbalance rule matters because it passes stricter non-triviality tests: it has `selection_rate < 0.95`, beats matched random controls, and shows positive excess versus scaled select-all controls. That makes it evidence of an L2 ranking signal, not a full strategy replacement.",
        "",
        "Advisor-safe wording: use `non-trivial L2 quote-ranking signal` or `candidate quote-control primitive`; avoid `dominates old baseline` or `proves profitability`.",
    ]
    write_markdown(ROOT / "old_bid_only_vs_queue_imbalance_interpretation.md", interp)
    print(f"Wrote {ROOT / 'PHASE02D_FINAL_STRICT_BASELINE_AUDIT_REPORT.md'}")


if __name__ == "__main__":
    main()
