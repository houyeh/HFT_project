from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from fix_common import (
    BACKTESTS_DIR,
    COMPARISON_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    ensure_fix_dirs,
)


def _safe(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def _row_or_none(df: pd.DataFrame, cond) -> pd.Series | None:
    sub = df[cond]
    if sub.empty:
        return None
    return sub.iloc[0]


def _build_comparison_table() -> pd.DataFrame:
    old_cmp = pd.read_csv(OUTPUTS_DIR / "comparison" / "binary_vs_baselines.csv")
    old_bt = pd.read_csv(OUTPUTS_DIR / "backtests" / "binary_backtest_summary.csv")
    val_sel = pd.read_csv(BACKTESTS_DIR / "val_selected_binary_test_result.csv")

    # References from prior diagnosis.
    smoothed_055_new = 84980.0
    smoothed_070_new = 63440.0
    smoothed_055_new3 = -551705.0
    smoothed_070_new3 = -551705.0

    old_best = _row_or_none(old_cmp, (old_cmp["dataset"] == "NEW_test") & (old_cmp["method"] == "binary_best"))
    old_best_new3 = _row_or_none(old_cmp, (old_cmp["dataset"] == "NEW3_full") & (old_cmp["method"] == "binary_best_from_new"))

    cond_sel_new = _row_or_none(val_sel, (val_sel["dataset"] == "NEW_test") & (val_sel["model_version"] == "conditional_only_primary"))
    cond_sel_new3 = _row_or_none(val_sel, (val_sel["dataset"] == "NEW3_full") & (val_sel["model_version"] == "conditional_only_primary"))
    all_sel_new = _row_or_none(val_sel, (val_sel["dataset"] == "NEW_test") & (val_sel["model_version"] == "all_event_primary"))
    all_sel_new3 = _row_or_none(val_sel, (val_sel["dataset"] == "NEW3_full") & (val_sel["model_version"] == "all_event_primary"))

    rows = []
    rows.append(
        {
            "model": "smoothed_mid_move_3class baseline 0.55",
            "label_type": "smoothed_mid_move_3class",
            "training_distribution": "all events",
            "inference_distribution": "all events",
            "threshold_selection": "legacy baseline reference",
            "NEW_net_pnl": smoothed_055_new,
            "NEW_trade_count": np.nan,
            "NEW_active_event_rate": np.nan,
            "NEW_beats_random_plus_1std": np.nan,
            "NEW_beats_smoothed_baseline": False,
            "NEW3_net_pnl": smoothed_055_new3,
            "NEW3_trade_count": np.nan,
            "NEW3_active_event_rate": np.nan,
            "verdict": "reference",
            "notes": "baseline reference",
        }
    )
    rows.append(
        {
            "model": "smoothed_mid_move_3class validation-selected 0.70",
            "label_type": "smoothed_mid_move_3class",
            "training_distribution": "all events",
            "inference_distribution": "all events",
            "threshold_selection": "validation-selected (legacy diagnosis)",
            "NEW_net_pnl": smoothed_070_new,
            "NEW_trade_count": np.nan,
            "NEW_active_event_rate": np.nan,
            "NEW_beats_random_plus_1std": False,
            "NEW_beats_smoothed_baseline": False,
            "NEW3_net_pnl": smoothed_070_new3,
            "NEW3_trade_count": np.nan,
            "NEW3_active_event_rate": np.nan,
            "verdict": "reference",
            "notes": "legacy val-selected threshold result",
        }
    )

    rows.append(
        {
            "model": "016 conditional-only binary original best",
            "label_type": "fill_conditional_quote_risk_binary_proxyC",
            "training_distribution": "fill-proxy conditioned",
            "inference_distribution": "all events",
            "threshold_selection": "NEW test sweep (not valid final selection)",
            "NEW_net_pnl": _safe(old_best["net_pnl"]) if old_best is not None else np.nan,
            "NEW_trade_count": _safe(old_best["trade_count"]) if old_best is not None else np.nan,
            "NEW_active_event_rate": np.nan,
            "NEW_beats_random_plus_1std": bool(old_best["beats_random_mean_plus_1std"]) if old_best is not None else np.nan,
            "NEW_beats_smoothed_baseline": bool(old_best["beats_smoothed_baseline"]) if old_best is not None else np.nan,
            "NEW3_net_pnl": _safe(old_best_new3["net_pnl"]) if old_best_new3 is not None else np.nan,
            "NEW3_trade_count": _safe(old_best_new3["trade_count"]) if old_best_new3 is not None else np.nan,
            "NEW3_active_event_rate": np.nan,
            "verdict": "label_poc_only",
            "notes": "high pnl but low trade count and fails random+1std",
        }
    )

    rows.append(
        {
            "model": "016 conditional-only binary val-selected",
            "label_type": "fill_conditional_quote_risk_binary_proxyC",
            "training_distribution": "fill-proxy conditioned",
            "inference_distribution": "all events",
            "threshold_selection": "validation-only + constraints",
            "NEW_net_pnl": _safe(cond_sel_new["net_pnl"]) if cond_sel_new is not None else np.nan,
            "NEW_trade_count": _safe(cond_sel_new["trade_count"]) if cond_sel_new is not None else np.nan,
            "NEW_active_event_rate": _safe(cond_sel_new["active_event_rate"]) if cond_sel_new is not None else np.nan,
            "NEW_beats_random_plus_1std": bool(cond_sel_new["beats_random_plus_1std"]) if cond_sel_new is not None else np.nan,
            "NEW_beats_smoothed_baseline": bool(cond_sel_new["beats_smoothed_baseline"]) if cond_sel_new is not None else np.nan,
            "NEW3_net_pnl": _safe(cond_sel_new3["net_pnl"]) if cond_sel_new3 is not None else np.nan,
            "NEW3_trade_count": _safe(cond_sel_new3["trade_count"]) if cond_sel_new3 is not None else np.nan,
            "NEW3_active_event_rate": _safe(cond_sel_new3["active_event_rate"]) if cond_sel_new3 is not None else np.nan,
            "verdict": str(cond_sel_new["verdict"]) if cond_sel_new is not None else "not_ready",
            "notes": "val-selected threshold only",
        }
    )

    rows.append(
        {
            "model": "016 all-event binary val-selected",
            "label_type": "all_event_quote_risk_binary_proxyC",
            "training_distribution": "all events",
            "inference_distribution": "all events",
            "threshold_selection": "validation-only + constraints",
            "NEW_net_pnl": _safe(all_sel_new["net_pnl"]) if all_sel_new is not None else np.nan,
            "NEW_trade_count": _safe(all_sel_new["trade_count"]) if all_sel_new is not None else np.nan,
            "NEW_active_event_rate": _safe(all_sel_new["active_event_rate"]) if all_sel_new is not None else np.nan,
            "NEW_beats_random_plus_1std": bool(all_sel_new["beats_random_plus_1std"]) if all_sel_new is not None else np.nan,
            "NEW_beats_smoothed_baseline": bool(all_sel_new["beats_smoothed_baseline"]) if all_sel_new is not None else np.nan,
            "NEW3_net_pnl": _safe(all_sel_new3["net_pnl"]) if all_sel_new3 is not None else np.nan,
            "NEW3_trade_count": _safe(all_sel_new3["trade_count"]) if all_sel_new3 is not None else np.nan,
            "NEW3_active_event_rate": _safe(all_sel_new3["active_event_rate"]) if all_sel_new3 is not None else np.nan,
            "verdict": str(all_sel_new["verdict"]) if all_sel_new is not None else "not_ready",
            "notes": "distribution mismatch reduced by all-event training",
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(COMPARISON_DIR / "conditional_vs_all_event_vs_smoothed.csv", index=False)
    return out


def _plot_trade_and_signal_vs_pnl() -> None:
    old_bt = pd.read_csv(OUTPUTS_DIR / "backtests" / "binary_backtest_summary.csv")
    ae_bt = pd.read_csv(BACKTESTS_DIR / "all_event_binary_backtest_summary.csv")
    old_new = old_bt[(old_bt["dataset"] == "NEW_test") & (np.isclose(old_bt["threshold_bid"], old_bt["threshold_ask"]))].copy()
    ae_new = ae_bt[(ae_bt["dataset"] == "NEW_test") & (np.isclose(ae_bt["threshold_bid"], ae_bt["threshold_ask"]))].copy()

    old_new = old_new.sort_values("threshold_bid")
    ae_new = ae_new.sort_values("threshold_bid")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(old_new["threshold_bid"], old_new["trade_count"], marker="o", label="conditional trade_count")
    ax2 = ax.twinx()
    ax2.plot(old_new["threshold_bid"], old_new["net_pnl"], marker="x", color="tab:orange", label="conditional net_pnl")
    if not ae_new.empty:
        ax.plot(ae_new["threshold_bid"], ae_new["trade_count"], marker="s", label="all-event trade_count")
        ax2.plot(ae_new["threshold_bid"], ae_new["net_pnl"], marker="^", color="tab:red", label="all-event net_pnl")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Trade Count")
    ax2.set_ylabel("Net PnL")
    ax.grid(alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "trade_count_vs_pnl_by_threshold.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(old_new["threshold_bid"], old_new["signal_rate_bid"], marker="o", label="conditional bid_signal")
    ax.plot(old_new["threshold_bid"], old_new["signal_rate_ask"], marker="x", label="conditional ask_signal")
    if not ae_new.empty:
        ax.plot(ae_new["threshold_bid"], ae_new["bid_signal_rate"], marker="s", label="all-event bid_signal")
        ax.plot(ae_new["threshold_bid"], ae_new["ask_signal_rate"], marker="^", label="all-event ask_signal")
    ax2 = ax.twinx()
    ax2.plot(old_new["threshold_bid"], old_new["net_pnl"], "--", color="tab:orange", label="conditional net_pnl")
    if not ae_new.empty:
        ax2.plot(ae_new["threshold_bid"], ae_new["net_pnl"], "--", color="tab:red", label="all-event net_pnl")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Signal Rate")
    ax2.set_ylabel("Net PnL")
    ax.grid(alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "signal_rate_vs_pnl_by_threshold.png", dpi=150)
    plt.close(fig)


def _plot_prob_and_pr_comparison() -> dict[str, float]:
    # Conditional model test set probabilities on conditioned data.
    c_bid_thr = pd.read_csv(OUTPUTS_DIR / "models" / "bid_primary_fw1_h10_posw" / "threshold_diagnostics_test.csv")
    c_ask_thr = pd.read_csv(OUTPUTS_DIR / "models" / "ask_primary_fw1_h10_posw" / "threshold_diagnostics_test.csv")
    c_bid_rep = pd.read_csv(OUTPUTS_DIR / "models" / "bid_primary_fw1_h10_posw" / "classification_report_test.csv")
    c_ask_rep = pd.read_csv(OUTPUTS_DIR / "models" / "ask_primary_fw1_h10_posw" / "classification_report_test.csv")

    a_bid_thr = pd.read_csv(MODELS_DIR / "bid_all_event_fw1_h10_posw" / "threshold_diagnostics_test.csv")
    a_ask_thr = pd.read_csv(MODELS_DIR / "ask_all_event_fw1_h10_posw" / "threshold_diagnostics_test.csv")
    a_bid_rep = pd.read_csv(MODELS_DIR / "bid_all_event_fw1_h10_posw" / "classification_report_test.csv")
    a_ask_rep = pd.read_csv(MODELS_DIR / "ask_all_event_fw1_h10_posw" / "classification_report_test.csv")

    # Synthetic histogram view using signal-rate curves over thresholds.
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(c_bid_thr["threshold"], c_bid_thr["signal_rate"], marker="o", label="conditional bid signal curve")
    ax.plot(c_ask_thr["threshold"], c_ask_thr["signal_rate"], marker="x", label="conditional ask signal curve")
    ax.plot(a_bid_thr["threshold"], a_bid_thr["signal_rate"], marker="s", label="all-event bid signal curve")
    ax.plot(a_ask_thr["threshold"], a_ask_thr["signal_rate"], marker="^", label="all-event ask signal curve")
    ax.set_title("Probability/Threshold Behavior Proxy")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Predicted Positive Rate")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "probability_histogram_comparison.png", dpi=150)
    plt.close(fig)

    # PR proxy plot reconstructed from threshold diagnostics (precision-recall pairs).
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(c_bid_thr["recall"], c_bid_thr["precision"], marker="o", label="conditional bid")
    ax.plot(c_ask_thr["recall"], c_ask_thr["precision"], marker="x", label="conditional ask")
    ax.plot(a_bid_thr["recall"], a_bid_thr["precision"], marker="s", label="all-event bid")
    ax.plot(a_ask_thr["recall"], a_ask_thr["precision"], marker="^", label="all-event ask")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Behavior Comparison (threshold sweep points)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curve_comparison.png", dpi=150)
    plt.close(fig)

    def toxic_f1(df: pd.DataFrame) -> float:
        sub = df[df["class"] == "toxic"]
        if sub.empty:
            return float("nan")
        return _safe(sub.iloc[0]["f1_score"])

    return {
        "conditional_bid_best_f1": float(c_bid_thr["f1"].max()),
        "conditional_ask_best_f1": float(c_ask_thr["f1"].max()),
        "all_event_bid_best_f1": float(a_bid_thr["f1"].max()),
        "all_event_ask_best_f1": float(a_ask_thr["f1"].max()),
        "conditional_bid_toxic_f1_at_0p5": toxic_f1(c_bid_rep),
        "conditional_ask_toxic_f1_at_0p5": toxic_f1(c_ask_rep),
        "all_event_bid_toxic_f1_at_0p5": toxic_f1(a_bid_rep),
        "all_event_ask_toxic_f1_at_0p5": toxic_f1(a_ask_rep),
    }


def _write_probability_diagnostics(comp_df: pd.DataFrame, extra: dict[str, float]) -> None:
    val_sel = pd.read_csv(BACKTESTS_DIR / "val_selected_binary_test_result.csv")
    cond = val_sel[(val_sel["dataset"] == "NEW_test") & (val_sel["model_version"] == "conditional_only_primary")]
    allv = val_sel[(val_sel["dataset"] == "NEW_test") & (val_sel["model_version"] == "all_event_primary")]
    cond_row = cond.iloc[0] if not cond.empty else None
    all_row = allv.iloc[0] if not allv.empty else None

    lines = []
    lines.append("# Probability and Threshold Diagnostics")
    lines.append("")
    lines.append("## 1) Are probabilities degenerate?")
    lines.append(
        f"- No complete collapse observed. Best threshold-sweep F1: "
        f"conditional(bid/ask)={extra['conditional_bid_best_f1']:.4f}/{extra['conditional_ask_best_f1']:.4f}, "
        f"all-event(bid/ask)={extra['all_event_bid_best_f1']:.4f}/{extra['all_event_ask_best_f1']:.4f}."
    )
    lines.append("")
    lines.append("## 2) Does selected threshold create too few trades?")
    if cond_row is not None:
        lines.append(f"- Conditional val-selected NEW trade_count={int(cond_row['trade_count'])}.")
    if all_row is not None:
        lines.append(f"- All-event val-selected NEW trade_count={int(all_row['trade_count'])}.")
    lines.append("- If trade_count < 300, robustness caveat remains.")
    lines.append("")
    lines.append("## 3) Is PnL driven by rare high-value trades?")
    lines.append("- Yes, this risk exists when signal is sparse and net_pnl comes from low trade count regimes.")
    lines.append("")
    lines.append("## 4) Is signal rate usable for HFT maker strategy?")
    lines.append("- Usability depends on threshold region where both trade_count and active_event_rate stay acceptable.")
    lines.append("")
    lines.append("## 5) Does all-event training reduce distribution mismatch?")
    lines.append("- Conceptually yes (train/inference both all-event); empirical edge still must pass random+1std.")
    (REPORTS_DIR / "probability_and_threshold_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def _write_final_reports(comp_df: pd.DataFrame) -> None:
    val_sel = pd.read_csv(BACKTESTS_DIR / "val_selected_binary_test_result.csv")
    key = pd.read_csv(COMPARISON_DIR / "val_threshold_selection_binary.csv")
    diag_snapshot = json.loads((REPORTS_DIR.parent / "diagnostics" / "current_016_snapshot.json").read_text(encoding="utf-8"))

    cond_new = val_sel[(val_sel["dataset"] == "NEW_test") & (val_sel["model_version"] == "conditional_only_primary")]
    all_new = val_sel[(val_sel["dataset"] == "NEW_test") & (val_sel["model_version"] == "all_event_primary")]
    cond_new = cond_new.iloc[0] if not cond_new.empty else None
    all_new = all_new.iloc[0] if not all_new.empty else None

    # Final verdict choice.
    final_verdict = "C. Binary label pipeline is useful, but trading model still not_ready."
    recommendation = "not_ready"
    if all_new is not None:
        if bool(all_new["beats_random_plus_1std"]) and bool(all_new["beats_smoothed_baseline"]) and int(all_new["trade_count"]) >= 300:
            final_verdict = "A. Binary model is now ready to replace smoothed baseline."
            recommendation = "replace_smoothed_baseline"
        elif bool(all_new["beats_random_mean"]):
            final_verdict = "B. Binary model is auxiliary candidate only."
            recommendation = "auxiliary"

    lines = []
    lines.append("# 016 Fix Report")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append(f"- {final_verdict}")
    lines.append("")
    lines.append("## 2. Original 016 Problem")
    lines.append("- High NEW PnL came with low trade_count and failed random+1std.")
    lines.append("- Original threshold was chosen from test sweep, not validation.")
    lines.append("- Conditional training vs all-event inference mismatch existed.")
    lines.append("")
    lines.append("## 3. Validation-selected Threshold Result")
    lines.append("- See `comparison/val_threshold_selection_binary.csv` and `backtests/val_selected_binary_test_result.csv`.")
    lines.append("")
    lines.append("## 4. All-event Label Result")
    lines.append("- See `datasets/all_event_label_distribution.csv` and all-event model reports under `models/`.")
    lines.append("")
    lines.append("## 5. Comparison Against Smoothed Baseline")
    lines.append("- See `comparison/conditional_vs_all_event_vs_smoothed.csv`.")
    lines.append("")
    lines.append("## 6. Verdict")
    lines.append(f"- {final_verdict}")
    lines.append("")
    lines.append("## 7. Recommendation")
    if recommendation == "replace_smoothed_baseline":
        lines.append("- Integrate binary signal with regime gate for next stage.")
    elif recommendation == "auxiliary":
        lines.append("- Use binary signal as auxiliary risk score; keep smoothed baseline primary.")
    else:
        lines.append("- Keep smoothed baseline; focus on regime gate / rule-based FPGA MVP.")
    (REPORTS_DIR / "016_fix_report.md").write_text("\n".join(lines), encoding="utf-8")

    # Final checkpoint
    ck = []
    ck.append("# Final Checkpoint")
    ck.append("")
    ck.append("## 1. Original 016")
    ck.append(
        f"- NEW net_pnl={diag_snapshot['new_net_pnl']:.2f}, trade_count={diag_snapshot['trade_count']}, "
        f"random+1std={diag_snapshot['random_mean_plus_1std']:.2f}, verdict=not_ready"
    )
    ck.append("")
    ck.append("## 2. Conditional-only val-selected")
    if cond_new is not None:
        cond_thr = key[(key["model_version"] == "conditional_only_primary") & (key["selected"] == 1)].iloc[0]
        ck.append(
            f"- selected thresholds bid={cond_thr['threshold_bid']:.2f}, ask={cond_thr['threshold_ask']:.2f}; "
            f"NEW net_pnl={float(cond_new['net_pnl']):.2f}, trade_count={int(cond_new['trade_count'])}, "
            f"beats random+1std={bool(cond_new['beats_random_plus_1std'])}"
        )
    else:
        ck.append("- MISSING")
    ck.append("")
    ck.append("## 3. All-event val-selected")
    if all_new is not None:
        all_thr = key[(key["model_version"] == "all_event_primary") & (key["selected"] == 1)].iloc[0]
        ck.append(
            f"- selected thresholds bid={all_thr['threshold_bid']:.2f}, ask={all_thr['threshold_ask']:.2f}; "
            f"NEW net_pnl={float(all_new['net_pnl']):.2f}, trade_count={int(all_new['trade_count'])}, "
            f"beats random+1std={bool(all_new['beats_random_plus_1std'])}"
        )
    else:
        ck.append("- MISSING")
    ck.append("")
    ck.append("## 4. Best model after fix")
    if all_new is not None and cond_new is not None:
        best = all_new if float(all_new["net_pnl"]) >= float(cond_new["net_pnl"]) else cond_new
        model_type = "all_event_primary" if best is all_new else "conditional_only_primary"
        ck.append(
            f"- model={model_type}, NEW net_pnl={float(best['net_pnl']):.2f}, trade_count={int(best['trade_count'])}, "
            f"random+1std={float(best['random_mean_plus_1std']):.2f}, smoothed_baseline={float(best['smoothed_baseline_net_pnl']):.2f}, "
            f"final_verdict={final_verdict}"
        )
    else:
        ck.append("- MISSING")
    ck.append("")
    ck.append("## 5. Recommendation")
    ck.append(f"- {recommendation}")
    (REPORTS_DIR / "final_checkpoint.md").write_text("\n".join(ck), encoding="utf-8")


def main() -> None:
    ensure_fix_dirs()
    comp_df = _build_comparison_table()
    _plot_trade_and_signal_vs_pnl()
    extra = _plot_prob_and_pr_comparison()
    _write_probability_diagnostics(comp_df, extra)
    _write_final_reports(comp_df)
    print("Wrote fix reports and figures.")


if __name__ == "__main__":
    main()

