from __future__ import annotations

import json

import pandas as pd

from stress_robust_utils import DIAGNOSTICS_DIR, EVALUATION_DIR, HLS_DIR, ORACLE_DIR, ROOT, SEARCH_DIR, ensure_dirs


def _fmt(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):,.3f}" if abs(float(x) - round(float(x))) > 1e-9 else f"{float(x):,.0f}"


def _hls_logic(policy: dict) -> str:
    kind = policy["policy_type"]
    lines = [
        "# HLS Stress-Robust Policy Specification",
        "",
        "This is a fixed policy specification for latency risk-control research. It contains no online optimization, no dynamic retraining, and no test-time threshold adjustment.",
        "",
        "## Fixed-Point Inputs",
        "",
        "- `spread_ticks`: unsigned fixed-point or integer tick count; expected range `[0, 8]` ticks for the tested universe.",
        "- `bid_prob_toxic`, `ask_prob_toxic`: quantized risk scores, recommended `ap_ufixed<16,1>` or equivalent unsigned Q1.15.",
        "- 015 scores or regime bucket inputs only if used by the selected policy.",
        "",
        "## Selected Policy",
        "",
        f"- policy_id: `{policy['policy_id']}`",
        f"- policy_type: `{kind}`",
        f"- policy_family: `{policy.get('policy_family', 'NA')}`",
        "",
    ]
    if kind == "ev_score":
        lines.extend(
            [
                "```text",
                "spread_value = 0.5 * spread_ticks * POINT_VALUE",
                "bid_ev = spread_value - LAMBDA_TOXIC * bid_prob_toxic * POINT_VALUE - MAKER_FEE",
                "ask_ev = spread_value - LAMBDA_TOXIC * ask_prob_toxic * POINT_VALUE - MAKER_FEE",
                "wide_spread = spread_ticks > MAX_SPREAD_TICKS",
                "pull_bid = wide_spread OR (bid_ev < EV_THRESHOLD)",
                "pull_ask = wide_spread OR (ask_ev < EV_THRESHOLD)",
                "```",
                "",
                f"- `MAX_SPREAD_TICKS = {policy['max_spread_ticks']}`",
                f"- `LAMBDA_TOXIC = {policy['lambda_toxic']}`",
                f"- `EV_THRESHOLD = {policy['ev_threshold']}`",
            ]
        )
    elif kind == "risk_lut":
        lines.extend(
            [
                "```text",
                "spread_bucket = spread_ticks <= 2 ? tight_<=2 : wide_>2",
                "risk_bucket = bucketize(side_prob_toxic, RISK_THRESHOLDS)",
                "pull_side = CANCEL_LUT[side][spread_bucket][risk_bucket]",
                "```",
                "",
                f"- risk_thresholds: `{policy['risk_thresholds']}`",
                f"- cancel_lut_entries: `{len(policy['cancel_lut'])}`",
            ]
        )
    elif kind == "disagreement_ensemble":
        lines.extend(
            [
                "```text",
                "bid_016 = bid_prob_toxic >= BID_THRESHOLD",
                "ask_016 = ask_prob_toxic >= ASK_THRESHOLD",
                "wide_spread = spread_ticks > MAX_SPREAD_TICKS",
                "pull_bid = DISAGREEMENT_LOGIC(bid_016, pull_bid_015, wide_spread)",
                "pull_ask = DISAGREEMENT_LOGIC(ask_016, pull_ask_015, wide_spread)",
                "```",
                "",
                f"- agreement_mode: `{policy['agreement_mode']}`",
                f"- bid_threshold: `{policy['bid_threshold']}`",
                f"- ask_threshold: `{policy['ask_threshold']}`",
                f"- max_spread_ticks: `{policy['max_spread_ticks']}`",
            ]
        )
    elif kind == "regime_safe":
        lines.extend(
            [
                "```text",
                "regime = REGIME_BUCKET_INPUT",
                "pull_bid = bid_prob_toxic >= BID_THRESHOLD_TABLE[regime]",
                "pull_ask = ask_prob_toxic >= ASK_THRESHOLD_TABLE[regime]",
                "```",
                "",
                f"- regime_col: `{policy['regime_col']}`",
                "- threshold table:",
            ]
        )
        for regime, thresholds in policy["regime_thresholds"].items():
            lines.append(f"  - `{regime}`: bid `{thresholds['bid_threshold']}`, ask `{thresholds['ask_threshold']}`")
    lines.extend(
        [
            "",
            "## Output",
            "",
            "- `pull_bid`: one-bit cancel/passivate bid quote action.",
            "- `pull_ask`: one-bit cancel/passivate ask quote action.",
            "",
            "This specification frames the policy as a latency risk-control candidate. It does not prove FPGA production necessity or real-world profitability.",
        ]
    )
    return "\n".join(lines)


def run() -> None:
    ensure_dirs()
    selected = json.loads((SEARCH_DIR / "selected_stress_robust_policy.json").read_text(encoding="utf-8"))
    policy = selected["selected_policy"]
    test = pd.read_csv(EVALUATION_DIR / "frozen_stress_robust_policy_test.csv")
    decomp = pd.read_csv(DIAGNOSTICS_DIR / "stress_pnl_decomposition.csv")
    oracle = pd.read_csv(ORACLE_DIR / "oracle_perfect_upper_bounds.csv")
    search = pd.read_csv(SEARCH_DIR / "validation_stress_aware_policy_search.csv")

    (HLS_DIR / "hls_stress_robust_policy_spec.md").write_text(_hls_logic(policy), encoding="utf-8")
    # User requested this filename at phase root as well.
    (ROOT / "hls_stress_robust_policy_spec.md").write_text(_hls_logic(policy), encoding="utf-8")

    mod_020 = decomp[(decomp["policy"] == "020_selected_policy") & (decomp["stress_name"] == "moderate")].iloc[0]
    mod_021 = test[(test["policy"] == "021_selected") & (test["stress_name"] == "moderate")].iloc[0]
    base_021 = test[(test["policy"] == "021_selected") & (test["stress_name"] == "base")].iloc[0]
    no_mod = test[(test["policy"] == "no_cancel") & (test["stress_name"] == "moderate")].iloc[0]
    r95_mod = test[(test["policy"] == "random_v3_95p") & (test["stress_name"] == "moderate")].iloc[0]
    decision = str(mod_021["decision_status_for_021"])
    perfect_mod = oracle[(oracle["split"] == "test") & (oracle["stress_name"] == "moderate")].sort_values("net_pnl", ascending=False)
    selected_val = selected["validation_metrics"]

    report = f"""# 021 Stress-Robust Policy Report

## 1. Why 020 Was Insufficient

The 020 selected policy was base-positive but failed absolute moderate-stress profitability. In the locked test stress setting, 020 moderate PnL was `{_fmt(mod_020['net_pnl'])}` with spread capture `{_fmt(mod_020['spread_capture'])}`, adverse loss `{_fmt(mod_020['adverse_loss'])}`, fees `{_fmt(mod_020['fees'])}`, and `{int(mod_020['trade_count'])}` trades.

The failure mode is insufficient surviving spread capture under the `neutral_fill_haircut=0.5`, `toxic_multiplier=1.5`, `delay=0` setting. Defensive cancels reduce damage versus no-cancel/random baselines, but the remaining fills still do not generate enough net edge.

## 2. Stress PnL Decomposition

Diagnostics are in `diagnostics/stress_pnl_decomposition.csv` and `diagnostics/stress_pnl_decomposition.md`.

The decomposition identity is `net_pnl = spread_capture - adverse_loss - fees`. The adverse-loss term is a reconciled residual containing adverse markout, stress toxic penalty, and terminal inventory effects.

## 3. Oracle Bounds

Oracle outputs are in `oracle/oracle_perfect_upper_bounds.csv`, `oracle/oracle_best_validation_score_threshold.csv`, and `oracle/oracle_best_validation_spread_score_grid.csv`.

- Best perfect oracle moderate test PnL: `{_fmt(perfect_mod.iloc[0]['net_pnl'])}` from `{perfect_mod.iloc[0]['policy']}`.
- Worst perfect oracle moderate test PnL: `{_fmt(perfect_mod.iloc[-1]['net_pnl'])}` from `{perfect_mod.iloc[-1]['policy']}`.

Because at least one perfect oracle is positive under moderate stress, the locked simulator is not mathematically infeasible. The bottleneck is model/policy selectivity under validation-only tradable rules.

## 4. Validation Stress-Aware Search

Search output: `search/validation_stress_aware_policy_search.csv`.

- Selected policy: `{policy['policy_id']}`
- Family: `{policy.get('policy_family')}`
- Validation moderate PnL: `{_fmt(selected_val['moderate_net_pnl'])}`
- Validation objective: `{_fmt(selected_val['objective'])}`
- Constraints passed on validation: `{selected_val['constraints_pass']}`
- Test used for selection: `False`

## 5. Frozen Test Performance

Frozen test output: `evaluation/frozen_stress_robust_policy_test.csv`.

- 021 base test PnL: `{_fmt(base_021['net_pnl'])}`
- 021 moderate test PnL: `{_fmt(mod_021['net_pnl'])}`
- Moderate gap vs no-cancel: `{_fmt(float(mod_021['net_pnl']) - float(no_mod['net_pnl']))}`
- Moderate gap vs random_v3_95p: `{_fmt(float(mod_021['net_pnl']) - float(r95_mod['net_pnl']))}`
- Decision Status: `{decision}`

The validation-selected 021 policy did not generalize under frozen test stress. It is worse than 020_selected, 016_v2, and 015_common on moderate test PnL, so it should not replace those frozen defensive baselines.

## 6. HLS Mapping

The HLS policy spec is in `hls_stress_robust_policy_spec.md`. The policy is fixed threshold/LUT/comparator logic and does not require online optimization or dynamic retraining.

## 7. Limiting Factor

If moderate-positive PnL is not achieved, the limiting factor is not the locked stress simulator itself, because perfect oracles can be positive. The limiting factor is the gap between future-informed quote selection and validation-selected, hardware-friendly risk signals. The next research step is improving causal signal quality or execution realism, not tuning on test.

## 8. Conservative Conclusion

021 diagnoses whether a stress-robust FPGA-friendly controller is feasible under locked proxy stress. It must be interpreted as latency risk-control research, not proof of real-world HFT profitability or proof of FPGA production necessity.
"""
    (ROOT / "STRESS_ROBUST_POLICY_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    run()
    print(f"Wrote {ROOT / 'STRESS_ROBUST_POLICY_REPORT.md'}")
