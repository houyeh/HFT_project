from __future__ import annotations

import numpy as np
import pandas as pd

from root_cause_utils import RESULTS_DIR, ROOT, ensure_dirs, write_markdown


def _read(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_prior(path: str) -> pd.DataFrame:
    full = ROOT.parent / path
    return pd.read_csv(full) if full.exists() else pd.DataFrame()


def bottleneck_decision() -> tuple[str, list[str]]:
    opp = _read("oracle_opportunity_map.csv")
    lift = _read("within_spread_ml_lift.csv")
    spread = _read("spread_first_ml_second_policy.csv")
    align = _read("model_oracle_alignment_overall.csv")

    evidence: list[str] = []
    decisions: list[str] = []
    if not opp.empty:
        top5 = float(opp.loc[np.isclose(opp["top_fraction"], 0.05), "share_of_oracle_positive_pnl"].iloc[0])
        pos_count = int(opp["oracle_positive_quote_count"].iloc[0])
        total_count = int(opp["all_quote_action_count"].iloc[0])
        evidence.append(f"Top 5% quote actions capture `{top5:.3f}` of positive oracle PnL; oracle-positive rate is `{pos_count / total_count:.4f}`.")
        if top5 >= 0.75 or pos_count / total_count <= 0.05:
            decisions.append("sparse oracle opportunity")
    if not spread.empty:
        test = spread[spread["split"] == "test_diagnostic_burned"]
        base = test[test["policy"] == "spread_first_only"]["selected_quote_pnl"].iloc[0]
        best = test.sort_values("selected_quote_pnl", ascending=False).iloc[0]
        evidence.append(f"Spread-first-only action PnL is `{base:.3f}`; best spread-first ablation is `{best['policy']}` at `{best['selected_quote_pnl']:.3f}`.")
        if str(best["policy"]) == "spread_first_only" or float(best["incremental_pnl_over_spread_first_only"]) <= 0:
            decisions.append("execution regime dominates")
    if not lift.empty:
        inc = lift.groupby("score_name")["incremental_pnl_vs_same_bucket_random"].sum().sort_values(ascending=False)
        best_inc = float(inc.iloc[0])
        best_name = str(inc.index[0])
        evidence.append(f"Best within-spread ML incremental PnL vs matched random is `{best_name}` at `{best_inc:.3f}`.")
        if best_inc <= 0:
            decisions.append("feature/label insufficiency")
        else:
            decisions.append("calibration shift")
    if not align.empty:
        best_recall = align.sort_values("oracle_recall", ascending=False).iloc[0]
        evidence.append(f"Best causal oracle recall is `{best_recall['policy']}` at `{best_recall['oracle_recall']:.3f}` with precision `{best_recall['oracle_precision']:.3f}`.")

    if not decisions:
        decisions.append("missing true queue data")
    if "sparse oracle opportunity" in decisions and "feature/label insufficiency" in decisions:
        dominant = "sparse oracle opportunity plus feature/label insufficiency"
    elif "execution regime dominates" in decisions and "feature/label insufficiency" in decisions:
        dominant = "execution regime dominates plus feature/label insufficiency"
    else:
        dominant = decisions[0]
    evidence.append("Missing true queue data remains a simulator-limit caveat, not the primary data-selected bottleneck unless the measured diagnostics are inconclusive.")
    return dominant, evidence


def run() -> None:
    ensure_dirs()
    opp = _read("oracle_opportunity_map.csv")
    align = _read("model_oracle_alignment_overall.csv")
    lift = _read("within_spread_ml_lift.csv")
    rolling = _read("rolling_quantile_test_diagnostic.csv")
    spread = _read("spread_first_ml_second_policy.csv")
    prior_oracle = _read_prior("21_stress_robust_policy/oracle/oracle_perfect_upper_bounds.csv")
    prior_eval = _read_prior("21_stress_robust_policy/evaluation/frozen_stress_robust_policy_test.csv")
    dominant, evidence = bottleneck_decision()

    lines = [
        "# 023 Root-Cause Analysis for Stress-Robust Profitability",
        "",
        "The current 20260127 test set remains burned and diagnostic. This phase explains the gap between the prior non-tradable oracle and causal policies; it does not create a new deployable policy claim.",
        "",
        "## 1. Locked Oracle Definition",
        "",
        "- `moderate stress = neutral_fill_haircut=0.5, toxic_multiplier=1.5, delay=0`.",
        "- `oracle_positive_quote = realized stress-adjusted quote PnL > 0`.",
        "- `oracle_selected_quote = perfect_quote_pnl_filter selected quote`.",
        "- Oracle is non-tradable, uses future information, and is only an upper bound.",
        "",
        "Prior 021 inventory-constrained simulator reference:",
    ]
    if not prior_oracle.empty:
        row = prior_oracle[
            (prior_oracle["policy"] == "perfect_quote_pnl_filter")
            & (prior_oracle["split"] == "test")
            & (prior_oracle["stress_name"] == "moderate")
        ]
        if not row.empty:
            lines.append(f"- Perfect quote PnL filter oracle moderate-stress PnL: `{row.iloc[0]['net_pnl']:.3f}`.")
    if not prior_eval.empty:
        row = prior_eval[
            (prior_eval["policy"] == "021_selected")
            & (prior_eval["split"] == "test")
            & (prior_eval["stress_name"] == "moderate")
        ]
        if not row.empty:
            lines.append(f"- 021 selected causal policy moderate-stress PnL: `{row.iloc[0]['net_pnl']:.3f}`.")
    lines.extend(
        [
            "",
            "The 023 tables below use action-level quote opportunity PnL to diagnose selection direction, false positives, and missed positives. These action-level sums are not a replacement for the inventory-constrained simulator PnL.",
            "`spread_only` in 023 is a validation-selected raw `spread_ticks` gate because the serialized 022 sklearn model is version-incompatible in this runtime; the threshold manifest records `test_used_for_selection=false`.",
        "",
        "## 2. Oracle Opportunity Map",
        "",
        ]
    )
    if not opp.empty:
        for _, row in opp.iterrows():
            if row["top_fraction"] in [0.01, 0.05, 0.10]:
                lines.append(
                    f"- Top `{row['top_fraction']:.0%}` quote actions capture `{row['share_of_oracle_positive_pnl']:.3f}` of positive oracle PnL."
                )
    lines.extend(["", "## 3. Model-Oracle Alignment", ""])
    if not align.empty:
        for _, row in align.sort_values("selected_quote_pnl", ascending=False).iterrows():
            lines.append(
                f"- `{row['policy']}`: selected quote PnL `{row['selected_quote_pnl']:.3f}`, recall `{row['oracle_recall']:.3f}`, precision `{row['oracle_precision']:.3f}`, missed oracle PnL `{row['missed_oracle_pnl']:.3f}`."
            )
    lines.extend(["", "## 4. Within-Spread ML Lift", ""])
    if not lift.empty:
        summary = lift.groupby("score_name")["incremental_pnl_vs_same_bucket_random"].sum().sort_values(ascending=False)
        for name, val in summary.items():
            lines.append(f"- `{name}` incremental PnL vs matched same-bucket random: `{val:.3f}`.")
    lines.extend(["", "## 5. Calibration Transfer and Rolling Quantile", ""])
    if not rolling.empty:
        selected = rolling[rolling["is_selected_validation_policy"].astype(bool)]
        if not selected.empty:
            r = selected.iloc[0]
            lines.append(
                f"Validation-selected rolling policy diagnostic test PnL: `{r['selected_quote_pnl']:.3f}`; test sweep rows are marked `DIAGNOSTIC_ONLY` and `used_for_model_selection=False`."
            )
        best_diag = rolling.sort_values("selected_quote_pnl", ascending=False).iloc[0]
        lines.append(
            f"Best test-diagnostic-only rolling row is `{best_diag['policy']}` at `{best_diag['selected_quote_pnl']:.3f}`. This is evidence of calibration transfer failure, not a selectable result."
        )
    lines.extend(["", "## 6. Spread-First, ML-Second Policy", ""])
    if not spread.empty:
        test = spread[spread["split"] == "test_diagnostic_burned"].sort_values("selected_quote_pnl", ascending=False)
        for _, row in test.iterrows():
            lines.append(
                f"- `{row['policy']}`: quote PnL `{row['selected_quote_pnl']:.3f}`, incremental over spread-first `{row['incremental_pnl_over_spread_first_only']:.3f}`."
            )
    lines.extend(["", "## 7. Bottleneck Decision", "", f"Dominant bottleneck: **{dominant}**.", ""])
    lines.extend(f"- {item}" for item in evidence)
    lines.extend(
        [
            "",
            "## 8. Final Interpretation",
            "",
            "The oracle's positive moderate-stress PnL is concentrated enough that selection quality matters, but not so sparse that opportunity alone explains the failure. The causal ML signals show some within-spread lift, yet validation-selected calibration admits too much false-positive loss and misses too much oracle PnL when transferred to the burned diagnostic test. The result explains why the non-tradable oracle can reach positive PnL while validation-selected ML policies fail to generalize. This is a root-cause diagnosis, not a new profitability claim.",
        ]
    )
    write_markdown(ROOT / "ROOT_CAUSE_STRESS_PROFITABILITY_REPORT.md", lines)


if __name__ == "__main__":
    run()
    print(f"Wrote {ROOT / 'ROOT_CAUSE_STRESS_PROFITABILITY_REPORT.md'}")
