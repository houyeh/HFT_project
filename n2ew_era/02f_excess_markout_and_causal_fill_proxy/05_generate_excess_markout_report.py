from __future__ import annotations

import pandas as pd

from common_phase02f import (
    FROZEN_SPLITS,
    RESULTS,
    ROOT,
    SELECTED_POLICY_ID,
    write_markdown,
)


def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    labels = read_csv(RESULTS / "excess_label_summary.csv")
    eval_df = read_csv(RESULTS / "excess_policy_evaluation.csv")
    fill = read_csv(RESULTS / "causal_fill_proxy_summary.csv")
    seq = read_csv(RESULTS / "sequential_excess_inventory_backtest.csv")
    random = read_csv(RESULTS / "excess_matched_random_hurdles.csv")

    h20_eval = eval_df[
        (eval_df["horizon"].astype(int) == 20)
        & (eval_df["split"].astype(str).isin(FROZEN_SPLITS))
        & (eval_df["policy_id"].astype(str).isin(["old_bid_only_spread_ge_1", SELECTED_POLICY_ID]))
    ].copy() if not eval_df.empty else pd.DataFrame()
    selected_h20 = h20_eval[h20_eval["policy_id"].astype(str) == SELECTED_POLICY_ID] if not h20_eval.empty else pd.DataFrame()
    selected_beats_random_all = bool(selected_h20["beats_matched_random_95p"].astype(bool).all()) if not selected_h20.empty else False
    any_beats_random = bool(h20_eval["beats_matched_random_95p"].astype(bool).any()) if not h20_eval.empty else False
    any_positive_excess = bool((h20_eval["total_excess_pnl"].astype(float) > 0.0).any()) if not h20_eval.empty else False

    seq_focus = seq[
        (seq["split"].astype(str).isin(FROZEN_SPLITS))
        & (seq["policy_id"].astype(str).isin(["old_bid_only_spread_ge_1", SELECTED_POLICY_ID]))
        & (seq["fill_model"].astype(str) == "conservative_l2_stress")
        & (seq["inventory_cap"].astype(int) == 1)
        & (seq["fee_multiplier"].astype(float) == 1.0)
    ].copy() if not seq.empty else pd.DataFrame()
    selected_seq_focus = seq_focus[seq_focus["policy_id"].astype(str) == SELECTED_POLICY_ID] if not seq_focus.empty else pd.DataFrame()
    selected_seq_positive_fee1 = bool((selected_seq_focus["sequential_excess_pnl"].astype(float) > 0.0).all()) if not selected_seq_focus.empty else False
    selected_seq_fee5 = seq[
        (seq["split"].astype(str).isin(FROZEN_SPLITS))
        & (seq["policy_id"].astype(str) == SELECTED_POLICY_ID)
        & (seq["fill_model"].astype(str) == "conservative_l2_stress")
        & (seq["inventory_cap"].astype(int) == 1)
        & (seq["fee_multiplier"].astype(float) == 5.0)
    ].copy() if not seq.empty else pd.DataFrame()
    selected_seq_positive_fee5 = bool((selected_seq_fee5["sequential_excess_pnl"].astype(float) > 0.0).all()) if not selected_seq_fee5.empty else False

    if selected_beats_random_all and selected_seq_positive_fee1 and selected_seq_positive_fee5:
        status = "final_strategy_candidate_found"
        decision = "A non-ML policy survived excess markout, matched-random, and causal-fill sequential hurdles."
    elif any_beats_random or any_positive_excess:
        status = "signal_only_after_spread_adjustment"
        decision = "After removing half-spread capture, some residual signal remains, but it does not satisfy the full strategy-candidate hurdle."
    elif not any_beats_random:
        status = "proxy_artifact_confirmed"
        decision = "The high raw markout PnL was primarily a proxy artifact from mechanical half-spread capture. The remaining value should be treated as signal discovery / HLS quote-gate evidence only."
    else:
        status = "fail"
        decision = "The excess markout and causal-fill audit found no deployable non-ML signal under the proxy framework."

    lines = [
        "# Excess Markout and Causal Fill Proxy Report",
        "",
        f"Final classification: `{status}`.",
        "",
        decision,
        "",
        "No ML was trained. No policy thresholds or parameters were tuned on `test`, `NEW3`, or `NEW4`. The only evaluated non-ML policies are frozen controls from earlier phases.",
        "",
        "## 1. Excess Markout Definition",
        "",
        "The raw Phase 01 markout included mechanical half-spread capture. Phase 02F removes that component:",
        "",
        "```text",
        "excess_bid_pnl_H = (future_mid_H - mid_t) * point_value - fee",
        "excess_ask_pnl_H = (mid_t - future_mid_H) * point_value - fee",
        "no_quote excess_pnl_H = 0",
        "```",
        "",
        "Future mid is used only as a label. It is not used for policy decisions or fill-proxy decisions.",
        "",
        "## 2. Excess Label Summary",
        "",
        labels[labels["horizon"].astype(int) == 20].to_markdown(index=False) if not labels.empty else "Excess label summary missing.",
        "",
        "## 3. Excess Policy Evaluation",
        "",
        h20_eval.to_markdown(index=False) if not h20_eval.empty else "Excess policy evaluation missing.",
        "",
        "Matched random is a finite-population normal approximation matching side, session, spread bucket, and selected count. It is used as a strict non-ML hurdle, not for tuning.",
        "",
        "## 4. Causal Fill Proxy",
        "",
        "Fill decisions use current L2 state only. They do not inspect `pnl_H`, `excess_pnl_H`, future mid, or whether the quote was profitable.",
        "",
        fill.to_markdown(index=False) if not fill.empty else "Causal fill proxy summary missing.",
        "",
        "## 5. Sequential Excess Inventory",
        "",
        "Focus view: `conservative_l2_stress`, `inventory_cap=1`, `fee_multiplier=1`, `H20`.",
        "",
        seq_focus.to_markdown(index=False) if not seq_focus.empty else "Sequential excess inventory focus rows missing.",
        "",
        "Fee-5 strict view for the selected queue rule:",
        "",
        selected_seq_fee5.to_markdown(index=False) if not selected_seq_fee5.empty else "Selected fee-5 rows missing.",
        "",
        "Equity figure: `figures/sequential_excess_equity_curves.png`.",
        "",
        "## 6. Final Environment Adequacy Decision",
        "",
        decision,
        "",
        "If policies fail to beat matched random under `excess_pnl`, the previous high raw PnL must be described as a proxy artifact. Any remaining value is limited to signal discovery and an HLS-friendly quote gate primitive, not a final strategy.",
        "",
        "Allowed wording: `spread-adjusted signal discovery`, `causal L2 quote-gate proxy`, `environment adequacy failure for final strategy claims`.",
        "",
        "Forbidden wording: real-world profitability, true queue replay, or FPGA-proven PnL improvement.",
    ]
    write_markdown(ROOT / "EXCESS_MARKOUT_CAUSAL_FILL_REPORT.md", lines)
    print(f"Final classification: {status}")
    print(decision)


if __name__ == "__main__":
    main()
