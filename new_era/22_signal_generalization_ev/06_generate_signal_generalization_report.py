from __future__ import annotations

import pandas as pd

from ev_utils import RESULTS_DIR, ROOT, ensure_dirs


def _safe_read(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run() -> None:
    ensure_dirs()
    regime = _safe_read(RESULTS_DIR / "regime_shift_summary.csv")
    failed = _safe_read(RESULTS_DIR / "failed_021_region_val_test.csv")
    transfer = _safe_read(RESULTS_DIR / "score_bucket_transfer.csv")
    metrics = _safe_read(RESULTS_DIR / "quote_ev_model_metrics.csv")
    wf = _safe_read(RESULTS_DIR / "walk_forward_policy_results.csv")

    top_shift = regime.sort_values("psi", ascending=False).head(5) if not regime.empty else pd.DataFrame()
    best_val_model = metrics[(metrics["split"] == "val") & (metrics["metric_type"] == "validation_selected_policy")].sort_values("validation_selected_policy_pnl", ascending=False).head(1)
    wf_test = wf[wf["split"] == "test"] if not wf.empty else pd.DataFrame()

    lines = [
        "# 022 Signal Generalization & Quote-EV Modeling Report",
        "",
        "The current 20260127 test set is treated as burned for diagnosis. It is not used to tune thresholds, model families, feature sets, or final claims.",
        "",
        "## 1. Regime Shift",
        "",
    ]
    if not top_shift.empty:
        for _, row in top_shift.iterrows():
            lines.append(f"- {row['variable']}: PSI `{row['psi']:.4f}`, val mean `{row['val_mean']:.6g}`, diagnostic test mean `{row['test_mean']:.6g}`.")
    if not failed.empty:
        r = failed.iloc[0]
        lines.extend(
            [
                "",
                f"The 021 selected region had validation avg stress quote PnL `{r['val_mean']:.6f}` and diagnostic test avg `{r['test_mean']:.6f}`.",
                "Answer to the key diagnostic question: no, this specific 021 region was not profitable in validation and then unprofitable only on the burned diagnostic test. It was already negative under the moderate stress proxy on validation, then remained negative on diagnostic test with a different magnitude.",
            ]
        )
    lines.extend(["", "## 2. Score Bucket Transfer", ""])
    if not transfer.empty:
        for score in transfer["score_name"].unique():
            sub = transfer[(transfer["score_name"] == score) & (transfer["bucket"] == 10)]
            val = sub[sub["split"] == "val"].iloc[0]
            test = sub[sub["split"] == "test_diagnostic"].iloc[0]
            lines.append(f"- {score} bucket 10 stress avg PnL: val `{val['stress_adjusted_avg_quote_pnl']:.6f}`, diagnostic test `{test['stress_adjusted_avg_quote_pnl']:.6f}`.")
    lines.append("")
    lines.append("High-risk buckets preserve their broad economic meaning as loss-heavy quote regions, but the toxic rates and stress-adjusted PnL magnitudes do not transfer exactly from validation to the burned diagnostic test. This is partial transfer, not a clean generalization proof.")
    lines.extend(["", "## 3. Quote-EV Labels", ""])
    lines.append("Quote-EV labels are simulator-derived proxy labels, not true exchange-level queue-aware PnL. They use deterministic moderate-stress proxy fills and future mid-price information.")
    lines.extend(["", "## 4. Quote-EV Models", ""])
    if not best_val_model.empty:
        b = best_val_model.iloc[0]
        lines.append(f"Best validation-selected Quote-EV model policy: `{b['model_name']}` / `{b['feature_set']}` with validation proxy PnL `{b['validation_selected_policy_pnl']:.3f}`.")
    lines.append("Simple HLS-friendly baselines are included: spread_only, spread+queue_imbalance, spread+event_rate, and spread+cancel_pressure.")
    lines.append("The best validation policy coming from `spread_only` is an important warning sign: the richer signal set did not yet demonstrate a robust EV advantage over a very simple HLS-friendly baseline.")
    lines.extend(["", "## 5. Walk-Forward", ""])
    if not wf_test.empty:
        lines.append(f"Walk-forward frozen test folds: `{len(wf_test)}`; positive folds `{int((wf_test['policy_pnl'] > 0).sum())}`; mean PnL `{wf_test['policy_pnl'].mean():.3f}`.")
        lines.append("This is encouraging but mixed: one pre-burned fold is negative, so the evidence supports continued diagnosis rather than a production-style profitability claim.")
    else:
        lines.append("Walk-forward was not feasible or produced no frozen test folds.")
    lines.extend(
        [
            "",
            "## 6. Limiting Factor Diagnosis",
            "",
            "- Regime shift: present but moderate. The largest shifts are volatility, 016 score, 021 EV score, event rate, and 015 score.",
            "- Score bucket non-transfer: partial. High-risk buckets remain economically bad, but toxic rates and loss magnitudes change across validation and burned diagnostic test.",
            "- Label instability: unresolved. Quote-EV labels are deterministic simulator proxies with sparse positive-EV events, not exchange-confirmed queue outcomes.",
            "- Feature insufficiency: likely. The validation winner is `spread_only`, while richer feature sets do not yet show a stable incremental EV edge.",
            "- Policy overfit: still a risk. The current test is burned, and walk-forward consistency is positive in 2 of 3 pre-burned folds, not unanimous.",
            "- Missing true queue data: the central unresolved limitation. The proxy EV label improves objective alignment, but it still lacks exchange-confirmed queue position, true aggressor side, and true fill priority.",
            "",
            "## 7. Conservative Conclusion",
            "",
            "022 diagnoses signal generalization rather than declaring a production-ready policy. Final claims should rely on walk-forward consistency across fresh pre-test splits, not current burned test performance.",
        ]
    )
    (ROOT / "SIGNAL_GENERALIZATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
    print(f"Wrote {ROOT / 'SIGNAL_GENERALIZATION_REPORT.md'}")
