from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "25_audit_and_walk_forward"
RESULTS = ROOT / "results"


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)


def read(name: str) -> pd.DataFrame:
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    ensure_dirs()
    leakage = (RESULTS / "rolling_threshold_leakage_audit.md").read_text(encoding="utf-8") if (RESULTS / "rolling_threshold_leakage_audit.md").exists() else ""
    dist = read("selected_quote_pnl_distribution.csv")
    side = read("side_specific_breakdown.csv")
    group = read("group_level_performance.csv")
    spread = read("spread_only_vs_ml_rolling_frozen_test.csv")
    wf = read("fresh_walk_forward_results.csv")
    wf_summary = read("fresh_walk_forward_summary.csv")

    leakage_detected = "No lookahead leakage detected" not in leakage
    top5 = float(dist.loc[dist["top_fraction"].round(2) == 0.05, "top_positive_contribution_pct_of_total_pnl"].iloc[0]) if not dist.empty else float("nan")
    concentrated = bool(top5 > 0.75) if not pd.isna(top5) else True
    ml_beats_spread = False
    ml_pnl = spread_pnl = float("nan")
    if not spread.empty:
        ml_pnl = float(spread.loc[spread["policy_role"] == "024_ml_selected", "selected_quote_pnl"].iloc[0])
        spread_pnl = float(spread.loc[spread["policy_role"] == "spread_only_validation_best", "selected_quote_pnl"].iloc[0])
        ml_beats_spread = ml_pnl > spread_pnl
    wf_survives = False
    if not wf_summary.empty:
        fixed = wf_summary[wf_summary["policy_role"] == "fixed_024_architecture"]
        if not fixed.empty:
            wf_survives = int(fixed["positive_fold_count"].iloc[0]) >= max(2, int(fixed["fold_count"].iloc[0]))
    safety_rows = [
        {
            "claim": "024 rolling threshold is leakage-free",
            "status": "pass" if not leakage_detected else "fail",
            "metric": "shift(1) recomputation and session-boundary audit",
            "safe_wording": "No lookahead detected" if not leakage_detected else "Leakage/mismatch requires remediation before claims.",
        },
        {
            "claim": "024 PnL is not dominated by a tiny number of quotes",
            "status": "pass" if not concentrated else "caution",
            "metric": f"top 5% contribution pct of total PnL = {top5:.3f}",
            "safe_wording": "PnL concentration is acceptable" if not concentrated else "PnL is concentrated; claim must be caveated.",
        },
        {
            "claim": "ML adds value beyond spread-only rolling calibration",
            "status": "pass" if ml_beats_spread else "fail",
            "metric": f"ML frozen PnL = {ml_pnl:.3f}; spread-only frozen PnL = {spread_pnl:.3f}",
            "safe_wording": "ML beats validation-selected spread-only baseline" if ml_beats_spread else "Spread-only explains equal or greater frozen-test PnL.",
        },
        {
            "claim": "024 survives fresh walk-forward splits",
            "status": "pass" if wf_survives else "pending_or_fail",
            "metric": "fixed 024 architecture positive folds in pre-burned walk-forward",
            "safe_wording": "Fresh folds support the policy" if wf_survives else "Fresh folds do not fully confirm the policy; require new holdout.",
        },
    ]
    pd.DataFrame(safety_rows).to_csv(RESULTS / "024_claim_safety_table.csv", index=False)

    lines = [
        "# 025 Final Confirmation Report",
        "",
        "## 1. Leakage Audit",
        "",
        "Leakage detected: `" + str(leakage_detected) + "`.",
        "",
        "## 2. PnL Concentration",
        "",
    ]
    if not dist.empty:
        first = dist.iloc[0]
        lines.append(f"Selected quote count `{int(first['selected_quote_count'])}`, total PnL `{first['total_selected_quote_pnl']:.3f}`, mean `{first['mean_quote_pnl']:.3f}`, median `{first['median_quote_pnl']:.3f}`, max win `{first['max_win']:.3f}`, max loss `{first['max_loss']:.3f}`.")
        for _, row in dist.iterrows():
            lines.append(f"- Top `{row['top_fraction']:.0%}` positive quotes contribute `{row['top_positive_contribution_pct_of_total_pnl']:.3f}` of total selected PnL.")
    lines.extend(["", "## 3. Side and Group Breakdown", ""])
    if not side.empty:
        for _, row in side.iterrows():
            lines.append(f"- `{row['side']}`: selected PnL `{row['selected_quote_pnl']:.3f}`, selected count `{int(row['selected_quote_count'])}`, selection rate `{row['selection_rate_within_side']:.3f}`.")
    if not group.empty:
        lines.append(f"Positive session/side/spread groups: `{int((group['selected_quote_pnl'] > 0).sum())}` / `{len(group)}`.")
    lines.extend(["", "## 4. Spread-Only vs ML Frozen Test", ""])
    lines.append(f"ML beats spread-only frozen baseline: `{ml_beats_spread}`. ML `{ml_pnl:.3f}` vs spread-only `{spread_pnl:.3f}`.")
    lines.extend(["", "## 5. Fresh Walk-Forward Confirmation", ""])
    lines.append("Fresh folds exclude 20260127. Because original out-of-sample 016 scores are unavailable for pre-burned training sessions, the walk-forward check trains a fold-local 016-style causal linear risk ranker on earlier sessions; this confirms the rolling-calibration architecture, not the exact original 016 TCN weights.")
    if not wf_summary.empty:
        for _, row in wf_summary.iterrows():
            lines.append(f"- `{row['policy_role']}`: folds `{int(row['fold_count'])}`, mean test PnL `{row['mean_test_pnl']:.3f}`, positive folds `{int(row['positive_fold_count'])}`.")
    lines.extend(["", "## 6. Final Decision", ""])
    if (not leakage_detected) and (not concentrated) and ml_beats_spread and wf_survives:
        decision = "Pass: rolling calibration appears to solve the observed calibration shift under the current proxy framework."
    else:
        decision = "Pending/Partial: 024 remains a breakthrough candidate, but at least one audit gate requires caution or more fresh holdout evidence."
    lines.append(decision)
    lines.append("")
    lines.append("No real-world profitability, true queue priority, or FPGA production necessity claim is made.")
    write(ROOT / "025_FINAL_CONFIRMATION_REPORT.md", lines)


if __name__ == "__main__":
    run()
    print(f"Wrote {ROOT / '025_FINAL_CONFIRMATION_REPORT.md'}")
