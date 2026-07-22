from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_search_utils import EVALUATION_DIR, ROBUSTNESS_DIR, SEARCH_DIR, json_sanitize


def _fmt(x: float) -> str:
    return f"{float(x):,.3f}" if abs(float(x) - round(float(x))) > 1e-9 else f"{float(x):,.0f}"


def _policy_hls_text(policy: dict) -> str:
    kind = policy["policy_type"]
    lines = [
        "# HLS Policy Specification",
        "",
        "This specification maps the validation-selected policy into fixed hardware-friendly logic. It does not require dynamic retraining, online optimization, or floating-point heavy inference in the policy layer.",
        "",
        "## Inputs Per Event",
        "",
        "- `spread_ticks`",
        "- `bid_prob_toxic` and `ask_prob_toxic` from the frozen 016_v2 risk scorer",
        "- `score_bid_015` and `score_ask_015` from the frozen 015 scorer when the selected policy uses an ensemble",
        "",
        "## Output Actions",
        "",
        "- `pull_bid`: cancel/passivate the bid quote for the event",
        "- `pull_ask`: cancel/passivate the ask quote for the event",
        "",
        "## Selected Logic",
        "",
        f"- policy_id: `{policy['policy_id']}`",
        f"- policy_type: `{kind}`",
        f"- max_spread_ticks: `{policy.get('max_spread_ticks', 'NA')}`",
        "",
    ]
    if kind == "spread_toxic":
        lines.extend(
            [
                "```text",
                "wide_spread = spread_ticks > MAX_SPREAD_TICKS",
                "pull_bid = wide_spread OR (bid_prob_toxic >= BID_THRESHOLD)",
                "pull_ask = wide_spread OR (ask_prob_toxic >= ASK_THRESHOLD)",
                "```",
                "",
                f"- `BID_THRESHOLD = {policy['bid_threshold']}`",
                f"- `ASK_THRESHOLD = {policy['ask_threshold']}`",
            ]
        )
    elif kind == "regime_threshold":
        lines.extend(
            [
                "```text",
                "wide_spread = spread_ticks > MAX_SPREAD_TICKS",
                "regime = coarse_spread_bucket",
                "pull_bid = wide_spread OR (bid_prob_toxic >= BID_THRESHOLD_TABLE[regime])",
                "pull_ask = wide_spread OR (ask_prob_toxic >= ASK_THRESHOLD_TABLE[regime])",
                "```",
                "",
                "Threshold table:",
            ]
        )
        for regime, thresholds in policy["regime_thresholds"].items():
            lines.append(f"- `{regime}`: bid `{thresholds['bid_threshold']}`, ask `{thresholds['ask_threshold']}`")
    elif kind == "ensemble":
        mode = policy["ensemble_mode"]
        if mode == "weighted":
            lines.extend(
                [
                    "```text",
                    "wide_spread = spread_ticks > MAX_SPREAD_TICKS",
                    "bid_weighted_score = W_016 * bid_prob_toxic + (1 - W_016) * score_bid_015",
                    "ask_weighted_score = W_016 * ask_prob_toxic + (1 - W_016) * score_ask_015",
                    "pull_bid = wide_spread OR (bid_weighted_score >= BID_WEIGHTED_THRESHOLD)",
                    "pull_ask = wide_spread OR (ask_weighted_score >= ASK_WEIGHTED_THRESHOLD)",
                    "```",
                    "",
                    f"- `W_016 = {policy['weight_016']}`",
                    f"- `BID_WEIGHTED_THRESHOLD = {policy['bid_weighted_threshold']}`",
                    f"- `ASK_WEIGHTED_THRESHOLD = {policy['ask_weighted_threshold']}`",
                    f"- audit reference 016 bid threshold: `{policy['bid_threshold']}`",
                    f"- audit reference 016 ask threshold: `{policy['ask_threshold']}`",
                    "",
                    "The `0.75` weight can be represented as a fixed-point multiply or as a simple `3/4` shift/add approximation after score quantization.",
                ]
            )
        else:
            lines.extend(
                [
                    "```text",
                    "wide_spread = spread_ticks > MAX_SPREAD_TICKS",
                    "bid_016 = bid_prob_toxic >= BID_THRESHOLD",
                    "ask_016 = ask_prob_toxic >= ASK_THRESHOLD",
                    "pull_bid = wide_spread OR ENSEMBLE_BOOL(bid_016, pull_bid_015)",
                    "pull_ask = wide_spread OR ENSEMBLE_BOOL(ask_016, pull_ask_015)",
                    "```",
                    "",
                    f"- `ENSEMBLE_BOOL = {mode}`",
                    f"- `BID_THRESHOLD = {policy['bid_threshold']}`",
                    f"- `ASK_THRESHOLD = {policy['ask_threshold']}`",
                ]
            )
    lines.extend(
        [
            "",
            "## Hardware Notes",
            "",
            "- Threshold comparisons can be implemented as fixed-point comparators.",
            "- Spread filtering is an integer tick comparison.",
            "- The policy layer is branch/table logic, not online learning.",
            "- The report demonstrates defensive value under proxy stress, not real-world production profitability.",
        ]
    )
    return "\n".join(lines)


def run() -> None:
    selected = json.loads((SEARCH_DIR / "selected_policy.json").read_text(encoding="utf-8"))
    test = pd.read_csv(EVALUATION_DIR / "frozen_policy_test_results.csv")
    main_policy = selected["selected_main_policy"]["policy"]
    main_val = selected["selected_main_policy"]["validation_metrics"]
    main_test = test[test["policy_role"] == "selected_main"].copy()
    base = main_test[main_test["stress_name"] == "base"].iloc[0]
    mild = main_test[main_test["stress_name"] == "mild"].iloc[0]
    moderate = main_test[main_test["stress_name"] == "moderate"].iloc[0]

    (EVALUATION_DIR / "hls_policy_spec.md").write_text(_policy_hls_text(main_policy), encoding="utf-8")

    shuffled_path = ROBUSTNESS_DIR / "shuffled_label_control.csv"
    multiseed_path = ROBUSTNESS_DIR / "multi_seed_stability.csv"
    simple_path = ROBUSTNESS_DIR / "simple_rule_baselines.csv"
    econ_path = ROBUSTNESS_DIR / "score_bucket_economic_value.csv"
    robustness_notes = []
    economic_note = ""
    if shuffled_path.exists():
        shuffled = pd.read_csv(shuffled_path)
        val = shuffled[shuffled["split"] == "val"]
        robustness_notes.append(f"Shuffled-label val PR-AUC mean: `{val['pr_auc'].mean():.6f}`.")
    if multiseed_path.exists():
        ms = pd.read_csv(multiseed_path)
        val = ms[ms["split"] == "val"]
        robustness_notes.append(f"Multi-seed val PR-AUC range: `{val['pr_auc'].min():.6f}` to `{val['pr_auc'].max():.6f}`.")
    if simple_path.exists():
        simple = pd.read_csv(simple_path)
        best = simple.sort_values("net_pnl", ascending=False).iloc[0]
        robustness_notes.append(f"Best validation simple-rule baseline: `{best['policy_id']}` with PnL `{_fmt(best['net_pnl'])}`.")
    if econ_path.exists():
        econ = pd.read_csv(econ_path)
        val_top = econ[(econ["split"] == "val") & (econ["bucket"] == "top_1pct")]
        if not val_top.empty:
            bid = val_top[val_top["side"] == "bid"].iloc[0]
            ask = val_top[val_top["side"] == "ask"].iloc[0]
            economic_note = (
                f"Validation score buckets show economic meaning: top 1% bid-risk quotes have expected not-cancelled quote PnL "
                f"`{float(bid['expected_quote_pnl_if_not_cancelled']):.3f}` versus base `{float(bid['base_expected_quote_pnl']):.3f}`; "
                f"top 1% ask-risk quotes have expected not-cancelled quote PnL "
                f"`{float(ask['expected_quote_pnl_if_not_cancelled']):.3f}` versus base `{float(ask['base_expected_quote_pnl']):.3f}`."
            )
        robustness_notes.append("Score bucket economic value audit is available in `robustness/score_bucket_economic_value.csv`.")

    text = f"""# FPGA-Friendly Policy Search Report

## 1. Objective

The objective is to convert existing risk signals (`015_common` and `016_v2_binary`) into a fixed, hardware-synthesizable market-making cancel policy. This phase does not add model features, does not tune on test, and does not make a real-world profitability claim.

## 2. Robustness Checks

{chr(10).join(f"- {note}" for note in robustness_notes) if robustness_notes else "- Robustness scripts are present; run outputs were not found when this report was generated."}

{economic_note}

## 3. Validation-Only Policy Search

Candidate policies were searched on validation only. Policy B depth proxy rows are diagnostic only and were excluded from the main claim.

- selection_split: `{selected['selection_split']}`
- test_used_for_selection: `{selected['test_used_for_selection']}`
- selected policy: `{main_policy['policy_id']}`
- policy family: `{main_policy['policy_family']}`
- validation PnL: `{_fmt(main_val['net_pnl'])}`
- validation trades: `{main_val['trade_count']}`
- validation max drawdown: `{_fmt(main_val['max_drawdown'])}`
- validation any-cancel rate: `{main_val['any_cancel_rate']:.4f}`

## 4. Frozen Test Evaluation

The selected policy was applied to test using frozen validation-selected parameters.

| Stress | Net PnL | Gap vs No-Cancel | Gap vs random_v3_mean | Trades | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| Base | {_fmt(base['net_pnl'])} | {_fmt(base['gap_vs_no_cancel'])} | {_fmt(base['gap_vs_random_v3_mean'])} | {int(base['trade_count'])} | {_fmt(base['max_drawdown'])} |
| Mild | {_fmt(mild['net_pnl'])} | {_fmt(mild['gap_vs_no_cancel'])} | {_fmt(mild['gap_vs_random_v3_mean'])} | {int(mild['trade_count'])} | {_fmt(mild['max_drawdown'])} |
| Moderate | {_fmt(moderate['net_pnl'])} | {_fmt(moderate['gap_vs_no_cancel'])} | {_fmt(moderate['gap_vs_random_v3_mean'])} | {int(moderate['trade_count'])} | {_fmt(moderate['max_drawdown'])} |

The selected policy is base-positive and remains defensive under mild/moderate stress, but it does not remain absolutely profitable under stress. The stronger test outcome of a backup policy is retained only as a diagnostic appendix result and is not used to reselect the main policy.

## 5. HLS Mapping

The policy maps to threshold comparators, spread tick comparisons, and optional small threshold tables. See `hls_policy_spec.md` for the exact mapping.

## 6. Diagnostic Appendix

Backup policies and diagnostic Policy B rows are retained in `search/policy_search_grid.csv`, `search/policy_selection_ranked.csv`, and `evaluation/frozen_policy_test_results.csv`. They are not the main positive-PnL claim.

## 7. Conservative Interpretation

This phase demonstrates an FPGA-friendly policy that retains defensive value under the current proxy execution framework. It is not proof of real-world HFT profitability, not proof of exchange fill toxicity, and not a substitute for queue-position-aware validation.
"""
    (EVALUATION_DIR / "POLICY_SEARCH_REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    run()
    print(f"Wrote {EVALUATION_DIR / 'POLICY_SEARCH_REPORT.md'}")
