from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "26_robust_baseline_side_aware"
RESULTS = ROOT / "results"


def read(name: str) -> pd.DataFrame:
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run() -> None:
    search = read("non_ml_policy_search_val.csv")
    frozen = read("non_ml_frozen_test.csv")
    wf = read("non_ml_walk_forward.csv")
    ml = read("ml_incremental_value.csv")
    side = read("side_specific_ml_value.csv")
    overlap = read("quote_decision_overlap.csv")
    align = read("toxic_vs_pnl_alignment.csv")
    new4 = read("new4_oos_non_ml.csv")
    new4_session = read("new4_oos_by_session_side.csv")
    selected = search[search["is_selected_policy"].astype(bool)].iloc[0] if not search.empty else None
    lines = [
        "# 026 Robust Baseline & ML Incremental Diagnosis",
        "",
        "## 1. Robust Non-ML Baseline",
        "",
    ]
    if selected is not None:
        lines.append(f"Validation-selected non-ML policy: `{selected['policy_id']}` with validation quote PnL `{selected['selected_quote_pnl']:.3f}`.")
    if not frozen.empty:
        r = frozen.iloc[0]
        lines.append(f"Frozen diagnostic test non-ML PnL: `{r['selected_quote_pnl']:.3f}`, selected quotes `{int(r['selected_quote_count'])}`, precision `{r['oracle_precision']:.3f}`.")
    if not wf.empty:
        summary = wf.groupby("policy_role")["selected_quote_pnl"].agg(["count", "mean", lambda s: int((s > 0).sum())]).reset_index()
        summary.columns = ["policy_role", "fold_count", "mean_pnl", "positive_folds"]
        lines.append("")
        lines.append("Walk-forward non-ML summary:")
        for _, row in summary.iterrows():
            lines.append(f"- `{row['policy_role']}`: mean PnL `{row['mean_pnl']:.3f}`, positive folds `{int(row['positive_folds'])}` / `{int(row['fold_count'])}`.")
    lines.extend(["", "## 2. ML Incremental Veto", ""])
    if not ml.empty:
        r = ml.iloc[0]
        lines.append(f"Baseline PnL `{r['baseline_pnl']:.3f}`; baseline+ML veto PnL `{r['ml_veto_pnl']:.3f}`; incremental `{r['ml_incremental_vs_baseline']:.3f}`.")
        lines.append(f"Matched random veto 95p `{r['random_veto_95p_pnl']:.3f}`; ML beats random 95p: `{bool(r['ml_beats_random_95p'])}`.")
        lines.append("Interpretation: ML chooses less damaging vetoes than matched random, but the vetoed quotes are still net positive, so ML does not add economic value beyond the robust non-ML baseline.")
    if not side.empty:
        for _, row in side.iterrows():
            lines.append(f"- `{row['side']}` ML incremental vs baseline: `{row['ml_incremental_vs_baseline']:.3f}`, vetoed quote PnL `{row['vetoed_quote_pnl']:.3f}`.")
    lines.extend(["", "## 3. Decision Overlap", ""])
    if not overlap.empty:
        blocked = overlap[overlap["decision_overlap"] == "blocked_by_ML_only"]
        if not blocked.empty:
            for _, row in blocked.iterrows():
                lines.append(f"- ML-only blocked `{row['side']}` quotes: count `{int(row['quote_count'])}`, total PnL `{row['total_pnl']:.3f}`, toxic rate `{row['toxic_rate']:.3f}`.")
    lines.extend(["", "## 4. Calibration and Label Alignment", ""])
    if not align.empty:
        test = align[align["split"] == "test_diagnostic_burned"]
        toxic_bad = test[(test["side_toxic"].astype(bool)) & (~test["pnl_positive"].astype(bool))]["quote_count"].sum()
        toxic_good = test[(test["side_toxic"].astype(bool)) & (test["pnl_positive"].astype(bool))]["quote_count"].sum()
        non_toxic_bad = test[(~test["side_toxic"].astype(bool)) & (~test["pnl_positive"].astype(bool))]["quote_count"].sum()
        non_toxic_good = test[(~test["side_toxic"].astype(bool)) & (test["pnl_positive"].astype(bool))]["quote_count"].sum()
        lines.append(f"Test label/PnL matrix counts: toxic_bad `{int(toxic_bad)}`, toxic_good `{int(toxic_good)}`, non_toxic_bad `{int(non_toxic_bad)}`, non_toxic_good `{int(non_toxic_good)}`.")
        lines.append("The toxic label only partially aligns with the quote-PnL objective: many bid-side toxic quotes are still economically positive under the current proxy, while ask-side toxic quotes are mostly bad.")
    lines.extend(["", "## 5. NEW4 External OOS Check", ""])
    if not new4.empty:
        r = new4.iloc[0]
        lines.append(f"NEW4 external OOS applies the frozen non-ML policy `{r['policy_id']}` with no NEW4 tuning.")
        lines.append(f"NEW4 selected quote PnL `{r['selected_quote_pnl']:.3f}`, selected quotes `{int(r['selected_quote_count'])}`, precision `{r['oracle_precision']:.3f}`, symbol `{r['symbol']}`.")
        if not new4_session.empty:
            pos_groups = int((new4_session["selected_quote_pnl"].astype(float) > 0).sum())
            lines.append(f"NEW4 positive session/side groups: `{pos_groups}` / `{len(new4_session)}`.")
    else:
        lines.append("NEW4 external OOS has not been generated yet.")
    lines.extend(["", "## 6. Final Diagnosis", ""])
    spread_dominant = not frozen.empty and (frozen.iloc[0]["selected_quote_pnl"] > 0)
    ml_value = not ml.empty and (ml.iloc[0]["ml_veto_pnl"] > ml.iloc[0]["baseline_pnl"]) and bool(ml.iloc[0]["ml_beats_random_95p"])
    lines.append(f"Is spread/side the dominant signal? `{spread_dominant}`.")
    lines.append(f"Does ML have incremental value beyond the robust baseline? `{ml_value}`.")
    if not side.empty:
        for _, row in side.iterrows():
            val = row["ml_incremental_vs_baseline"] > 0
            lines.append(f"ML incremental value on `{row['side']}`: `{bool(val)}`.")
    lines.append("No further ML training is recommended unless ML veto beats matched random controls and shows stable side-specific value.")
    (ROOT / "026_ROBUST_BASELINE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
    print(f"Wrote {ROOT / '026_ROBUST_BASELINE_REPORT.md'}")
