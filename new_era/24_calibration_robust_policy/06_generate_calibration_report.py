from __future__ import annotations

import json

import pandas as pd

from calibration_utils import RESULTS_DIR, ROOT, ensure_dirs, write_markdown


def _read(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run() -> None:
    ensure_dirs()
    audit = _read("023_evidence_audit_table.csv")
    grid = _read("combined_validation_selection_grid.csv")
    test = _read("frozen_calibration_robust_test.csv")
    detail = _read("frozen_calibration_robust_test_by_session_side_spread.csv")
    policy_path = ROOT / "selected_calibration_robust_policy.json"
    selected = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {}
    selected_policy = selected.get("selected_policy", {})

    lines = [
        "# 024 Calibration-Robust Policy Report",
        "",
        "## 1. Purpose",
        "",
        "023 found evidence that ranking signals have value, but static absolute thresholds suffer calibration shift. 024 tests whether causal past-only rolling thresholds can convert ranking into a more stable action policy.",
        "",
        "## 2. Evidence Audit",
        "",
    ]
    if not audit.empty:
        for _, row in audit.iterrows():
            lines.append(f"- `{row['claim']}`: {row['status']}. Safe wording: {row['safe_wording']}")
    lines.extend(["", "## 3. Past-Only Calibration Design", ""])
    lines.append("For every candidate, `threshold_t = quantile(score[t-N:t-1], q)`. The current quote action and all future actions are excluded from the threshold calculation.")
    lines.append("Candidate families: global rolling quantile, spread-bucket conditional rolling quantile, and side-specific rolling quantile.")
    lines.extend(["", "## 4. Validation Selection", ""])
    if selected_policy:
        lines.append(f"Selected policy: `{selected_policy.get('policy_id')}`.")
        lines.append(f"Score: `{selected_policy.get('score_name')}` / `{selected_policy.get('score_column')}`; window `{selected_policy.get('window')}`; groups `{selected_policy.get('group_cols')}`.")
        lines.append("Selection split: `validation`; `test_used_for_selection=false`.")
    if not grid.empty:
        top = grid.sort_values("selected_quote_pnl", ascending=False).head(5)
        lines.append("")
        lines.append("Top validation rows by selected quote PnL, including non-main spread-only diagnostic baselines:")
        for _, row in top.iterrows():
            lines.append(f"- `{row['policy_id']}`: validation quote PnL `{row['selected_quote_pnl']:.3f}`, precision `{row['oracle_precision']:.3f}`.")
        eligible = grid[grid["eligible_for_main_selection"].astype(bool)].copy()
        if not eligible.empty:
            lines.append("")
            lines.append("Top main-eligible ML rows:")
            for _, row in eligible.sort_values("selected_quote_pnl", ascending=False).head(5).iterrows():
                lines.append(f"- `{row['policy_id']}`: validation quote PnL `{row['selected_quote_pnl']:.3f}`, precision `{row['oracle_precision']:.3f}`.")
    lines.extend(["", "## 5. Frozen Diagnostic Test", ""])
    if not test.empty:
        r = test.iloc[0]
        lines.append(f"Frozen test selected quote PnL: `{r['selected_quote_pnl']:.3f}`.")
        lines.append(f"Oracle recall `{r['oracle_recall']:.3f}`, precision `{r['oracle_precision']:.3f}`, false-positive loss `{r['false_positive_loss']:.3f}`, missed oracle PnL `{r['missed_oracle_pnl']:.3f}`.")
        lines.append(f"Selected quote count `{int(r['selected_quote_count'])}`, selection rate `{r['selection_rate']:.3f}`.")
    if not detail.empty:
        pos_groups = int((detail["selected_quote_pnl"] > 0).sum())
        lines.append(f"Session/side/spread groups with positive selected quote PnL: `{pos_groups}` / `{len(detail)}`.")
    lines.extend(["", "## 6. Did Rolling Thresholds Bridge Calibration Shift?", ""])
    if not test.empty:
        pnl = float(test.iloc[0]["selected_quote_pnl"])
        if pnl > 0:
            conclusion = "Yes under the action-level moderate-stress proxy: the validation-selected rolling calibration produced positive frozen diagnostic test quote PnL."
        else:
            conclusion = "No: the validation-selected rolling calibration did not produce positive frozen diagnostic test quote PnL, although it may still reduce false-positive exposure."
        lines.append(conclusion)
    lines.extend(
        [
            "",
            "## 7. Conservative Interpretation",
            "",
            "This phase tests calibration robustness under the current simulator-derived action-level proxy. It does not prove real-world profitability, true exchange fill toxicity, or FPGA production necessity. Any positive result is evidence that rolling/rank-based calibration may reduce regime sensitivity; it remains subject to the missing true queue and aggressor-side limitations documented in prior phases.",
        ]
    )
    write_markdown(ROOT / "CALIBRATION_ROBUST_POLICY_REPORT.md", lines)


if __name__ == "__main__":
    run()
    print(f"Wrote {ROOT / 'CALIBRATION_ROBUST_POLICY_REPORT.md'}")
