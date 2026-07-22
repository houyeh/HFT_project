from __future__ import annotations

import json

import pandas as pd

from common_031 import RESULTS, ROOT, ensure_dirs, write_markdown


def fmt(v: float) -> str:
    return f"{float(v):,.0f}"


def main() -> None:
    ensure_dirs()
    universe = pd.read_csv(RESULTS / "full_action_universe_summary.csv")
    coverage = pd.read_csv(RESULTS / "baseline_oracle_coverage.csv")
    missed = pd.read_csv(RESULTS / "missed_opportunity_breakdown.csv")
    recovery = pd.read_csv(RESULTS / "opportunity_recovery_baselines.csv")
    ml_metrics = pd.read_csv(RESULTS / "full_action_ml_metrics.csv")
    ml_results = pd.read_csv(RESULTS / "full_action_ml_policy_results.csv")
    selected_recovery = json.loads((RESULTS / "selected_recovery_rule.json").read_text(encoding="utf-8"))
    selected_ml = json.loads((RESULTS / "selected_full_action_ml_policy.json").read_text(encoding="utf-8"))

    overall = coverage[coverage["selection_role"] == "frozen_baseline"].copy()
    test_cov = overall[overall["phase031_split"] == "test"].iloc[0]
    new4_cov = overall[overall["phase031_split"] == "new4"].iloc[0]
    test_ml = ml_results[ml_results["phase031_split"] == "test"].iloc[0]
    new4_ml = ml_results[ml_results["phase031_split"] == "new4"].iloc[0]
    viable = bool(test_ml["viable_full_action_ml"] and new4_ml["viable_full_action_ml"])

    if viable:
        decision = "ML is viable as a full-action selector under this audit because it beats both the strong baseline and matched random on frozen test and NEW4."
    else:
        decision = "ML should remain disabled for deployment. The baseline is conservative, but the validation-selected full-action ML selector does not clear both the strong baseline and matched-random hurdles on frozen test and NEW4."

    lines = [
        "# Opportunity Cost Audit Report",
        "",
        "## 1. Purpose",
        "",
        "Phase 031 tests whether the core `bid_only_spread_ge_1` baseline is too conservative and whether ML is more appropriate as a full-action quote selector rather than a residual veto.",
        "",
        "## 2. Full Action Universe",
        "",
    ]
    for _, row in universe.iterrows():
        lines.append(
            f"- `{row['phase031_split']}` `{row['side']}`: actions `{int(row['action_count'])}`, baseline-selected `{int(row['selected_by_baseline_count'])}`, "
            f"positive-EV actions `{int(row['positive_ev_count'])}`, total EV `{fmt(row['total_quote_ev'])}`."
        )
    lines.extend(["", "## 3. Baseline Coverage vs Oracle", ""])
    for _, row in overall.iterrows():
        lines.append(
            f"- `{row['phase031_split']}`: baseline PnL `{fmt(row['selected_quote_pnl'])}`, missed oracle-positive PnL `{fmt(row['missed_oracle_positive_pnl'])}`, "
            f"avoided negative EV `{fmt(row['avoided_negative_ev_pnl'])}`, oracle recall `{row['oracle_recall']:.3f}`."
        )
    lines.extend(
        [
            "",
            "## 4. Missed Opportunities",
            "",
        ]
    )
    missed_top = missed.sort_values("missed_positive_pnl", ascending=False).head(8)
    for _, row in missed_top.iterrows():
        lines.append(
            f"- `{row['group_type']}` split `{row['phase031_split']}` side `{row['side']}` spread `{row['spread_bucket']}` session `{row['session_id']}`: "
            f"missed PnL `{fmt(row['missed_positive_pnl'])}`, count `{int(row['missed_positive_count'])}`."
        )
    lines.extend(["", "## 5. Opportunity Recovery Baselines", ""])
    lines.append(f"Validation-selected simple recovery rule: `{selected_recovery['selected_rule']['rule_id']}`.")
    for _, row in recovery[recovery["phase031_split"].isin(["val", "test", "new4"])].tail(3).iterrows():
        lines.append(
            f"- `{row['phase031_split']}`: after-recovery PnL `{fmt(row['after_recovery_pnl'])}`, incremental `{fmt(row['incremental_vs_baseline'])}`, "
            f"matched random 95p `{fmt(row.get('matched_random_recovery_95p', float('nan')))}`."
        )
    lines.extend(["", "## 6. Full-Action ML Feasibility", ""])
    lines.append(f"Validation-selected ML policy: `{selected_ml['selected_policy']['model_name']}` threshold `{float(selected_ml['selected_policy']['threshold']):.6f}`.")
    selected_name = selected_ml["selected_policy"]["model_name"]
    for _, row in ml_metrics[(ml_metrics["model_name"] == selected_name) & (ml_metrics["split"].isin(["val", "test", "new4"]))].iterrows():
        lines.append(
            f"- `{row['split']}` metrics: PR-AUC `{row['pr_auc_positive_ev']:.4f}`, ROC-AUC `{row['roc_auc_positive_ev']:.4f}`, Spearman `{row['spearman_score_vs_positive_ev']:.4f}`."
        )
    for _, row in ml_results.iterrows():
        lines.append(
            f"- `{row['phase031_split']}` ML PnL `{fmt(row['ml_selected_pnl'])}`, baseline `{fmt(row['baseline_pnl'])}`, "
            f"gap `{fmt(row['gap_vs_baseline'])}`, random 95p `{fmt(row['matched_random_95p'])}`, viable `{bool(row['viable_full_action_ml'])}`."
        )
    lines.extend(
        [
            "",
            "## 7. Decision",
            "",
            f"Baseline conservative? `{'yes' if (test_cov['missed_oracle_positive_pnl'] > 0 or new4_cov['missed_oracle_positive_pnl'] > 0) else 'no'}`.",
            f"Full-action ML viable? `{viable}`.",
            "",
            decision,
            "",
            "The oracle opportunity is not a tradable claim. Test and NEW4 are not used for model or threshold tuning.",
        ]
    )
    write_markdown(ROOT / "OPPORTUNITY_COST_AUDIT_REPORT.md", lines)


if __name__ == "__main__":
    main()
    print(f"Wrote opportunity cost report to {ROOT / 'OPPORTUNITY_COST_AUDIT_REPORT.md'}")

