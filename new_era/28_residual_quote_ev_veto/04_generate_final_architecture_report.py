from __future__ import annotations

import json

import pandas as pd

from common_028 import RESULTS, ensure_dirs, write_markdown


def _fmt(value: float) -> str:
    return f"{float(value):,.0f}"


def _write_success_hls_spec(policy: dict) -> None:
    lines = [
        "# HLS Residual EV Veto Model Specification",
        "",
        f"Selected residual model: `{policy['model_name']}`.",
        f"Frozen threshold: `{policy['threshold']}`.",
        "",
        "This optional extension is evaluated only after the non-ML bid-only spread controller passes.",
        "",
        "## Data Path",
        "",
        "- Inputs: spread ticks, queue imbalance, depth imbalance, top depth, depth ratio, past-only volatility/event-rate features.",
        "- Model: lightweight fixed logistic/MLP parameters trained offline.",
        "- Output: veto baseline bid quote when the negative-EV score exceeds the frozen validation-selected threshold.",
        "",
        "No online optimization, retraining, or test-set threshold updates are allowed.",
    ]
    write_markdown(RESULTS / "hls_ev_veto_model_spec.md", lines)


def _write_rejection_report(eval_df: pd.DataFrame) -> None:
    lines = [
        "# ML Veto Rejection Report",
        "",
        "The residual Quote-EV veto is not accepted as part of the final architecture.",
        "",
        "Rejection rule: the optional ML veto must improve the frozen non-ML baseline and beat the matched random veto 95th-percentile hurdle on both frozen test and NEW4 external OOS.",
        "",
    ]
    for _, row in eval_df.iterrows():
        lines.append(
            f"- `{row['split']}`: baseline `{_fmt(row['baseline_pnl'])}`, ML after-veto `{_fmt(row['after_veto_pnl'])}`, "
            f"incremental `{_fmt(row['incremental_vs_baseline'])}`, random 95p `{_fmt(row['random_veto_95p_pnl'])}`, "
            f"beats random 95p `{bool(row['beats_random_veto_95p'])}`, improves baseline `{bool(row['improves_baseline'])}`."
        )
    lines.extend(
        [
            "",
            "Interpretation: the non-ML spread/side controller remains the core architecture. ML can be revisited only after it demonstrates incremental value over that robust baseline, not merely over naive random controls.",
        ]
    )
    write_markdown(RESULTS / "ml_veto_rejection_report.md", lines)


def main() -> None:
    ensure_dirs()
    policy = json.loads((RESULTS / "selected_ev_veto_policy.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(RESULTS / "residual_ev_model_metrics.csv")
    lift = pd.read_csv(RESULTS / "residual_ev_economic_lift.csv")
    eval_df = pd.read_csv(RESULTS / "frozen_veto_test_and_new4.csv")
    success = bool(
        (eval_df["beats_random_veto_95p"].astype(bool).all())
        and (eval_df["improves_baseline"].astype(bool).all())
        and (eval_df.loc[eval_df["split"] == "new4", "after_veto_pnl"].astype(float).iloc[0] > 0)
    )

    if success:
        _write_success_hls_spec(policy)
    else:
        _write_rejection_report(eval_df)

    lines = [
        "# Final Architecture Report",
        "",
        "## Core System",
        "",
        "The final core strategy remains the frozen non-ML side-aware spread controller `bid_only_spread_ge_1`. It is HLS-friendly and does not rely on ML calibration.",
        "",
        "## Residual Quote-EV Veto",
        "",
        f"Selected residual policy on validation: `{policy['policy_id']}`.",
        f"Selection split: `{policy['selection_split']}`; test used for selection: `{policy['test_used_for_selection']}`.",
        "",
        "## Model Diagnostics",
        "",
    ]
    selected_metrics = metrics[metrics["model_name"] == policy["model_name"]]
    for _, row in selected_metrics.iterrows():
        lines.append(
            f"- `{row['split']}` PR-AUC `{row['pr_auc_negative_ev']:.4f}`, ROC-AUC `{row['roc_auc_negative_ev']:.4f}`, "
            f"Spearman `{row['spearman_score_vs_negative_ev']:.4f}`."
        )
    lines.extend(["", "## Frozen Evaluation", ""])
    for _, row in eval_df.iterrows():
        lines.append(
            f"- `{row['split']}`: baseline `{_fmt(row['baseline_pnl'])}`, ML after-veto `{_fmt(row['after_veto_pnl'])}`, "
            f"incremental `{_fmt(row['incremental_vs_baseline'])}`, random 95p `{_fmt(row['random_veto_95p_pnl'])}`."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Residual ML veto accepted: `{success}`.",
            "",
            "If rejected, the architecture remains evidence-driven and conservative: deploy the non-ML spread/side controller as the final baseline, keep ML as diagnostic research, and do not claim real-world profitability or FPGA production necessity.",
            "",
            "All PnL values remain simulator-derived quote-EV proxies under the current queue/fill stress framework.",
        ]
    )
    write_markdown(RESULTS / "FINAL_ARCHITECTURE_REPORT.md", lines)


if __name__ == "__main__":
    main()
    print(f"Wrote final architecture report to {RESULTS}")

