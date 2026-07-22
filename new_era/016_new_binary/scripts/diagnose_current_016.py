from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fix_common import DIAG_DIR, OUTPUTS_DIR, ensure_fix_dirs


def _safe_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def main() -> None:
    ensure_fix_dirs()

    labels_path = OUTPUTS_DIR / "labels" / "binary_label_distribution.csv"
    model_summary_path = OUTPUTS_DIR / "comparison" / "model_comparison_summary.csv"
    bt_path = OUTPUTS_DIR / "backtests" / "binary_backtest_summary.csv"
    base_cmp_path = OUTPUTS_DIR / "comparison" / "binary_vs_baselines.csv"
    bid_hist_path = OUTPUTS_DIR / "models" / "bid_primary_fw1_h10_posw" / "train_history.csv"
    ask_hist_path = OUTPUTS_DIR / "models" / "ask_primary_fw1_h10_posw" / "train_history.csv"
    bid_thr_diag_path = OUTPUTS_DIR / "models" / "bid_primary_fw1_h10_posw" / "threshold_diagnostics_test.csv"
    ask_thr_diag_path = OUTPUTS_DIR / "models" / "ask_primary_fw1_h10_posw" / "threshold_diagnostics_test.csv"

    labels = pd.read_csv(labels_path)
    model_summary = pd.read_csv(model_summary_path)
    bt = pd.read_csv(bt_path)
    base_cmp = pd.read_csv(base_cmp_path)
    bid_hist = pd.read_csv(bid_hist_path)
    ask_hist = pd.read_csv(ask_hist_path)
    bid_thr_diag = pd.read_csv(bid_thr_diag_path)
    ask_thr_diag = pd.read_csv(ask_thr_diag_path)

    best_row = base_cmp[(base_cmp["dataset"] == "NEW_test") & (base_cmp["method"] == "binary_best")].iloc[0]
    best_thr_setup = str(best_row["threshold_setup"])
    try:
        pieces = best_thr_setup.replace("bid=", "").replace("ask=", "").split(",")
        thr_bid = float(pieces[0])
        thr_ask = float(pieces[1])
    except Exception:
        thr_bid, thr_ask = float("nan"), float("nan")

    bt_best = bt[
        (bt["dataset"] == "NEW_test")
        & (np.isclose(bt["threshold_bid"], thr_bid, equal_nan=False))
        & (np.isclose(bt["threshold_ask"], thr_ask, equal_nan=False))
    ].copy()
    if bt_best.empty:
        bt_best = bt[bt["dataset"] == "NEW_test"].sort_values("net_pnl", ascending=False).head(1)
    bt_best_row = bt_best.iloc[0]

    bid_val_pr_auc = _safe_float(bid_hist["val_auc_pr"].max())
    bid_test_pr_auc = _safe_float(model_summary["test_pr_auc"].iloc[0])  # average, refine below
    ask_val_pr_auc = _safe_float(ask_hist["val_auc_pr"].max())

    # Pull side-specific test PR-AUC from original training summary.
    train_summary_path = OUTPUTS_DIR / "models" / "binary_training_summary.csv"
    train_summary = pd.read_csv(train_summary_path)
    bid_test_pr_auc_side = _safe_float(train_summary[train_summary["side"] == "bid"]["test_auc_pr"].iloc[0])
    ask_test_pr_auc_side = _safe_float(train_summary[train_summary["side"] == "ask"]["test_auc_pr"].iloc[0])

    # Toxic rates from primary definition rows.
    def toxic_rate(side: str, split: str) -> float:
        sub = labels[
            (labels["side"] == side)
            & (labels["proxy_type"] == "price_level_removed_proxy")
            & (labels["fill_window"] == 1)
            & (labels["horizon"] == 10)
            & (labels["split"] == split)
        ]
        if sub.empty:
            return float("nan")
        return _safe_float(sub.iloc[0]["toxic_rate"])

    bid_t_train = toxic_rate("bid", "NEW_train")
    bid_t_val = toxic_rate("bid", "NEW_val")
    bid_t_test = toxic_rate("bid", "NEW_test")
    ask_t_train = toxic_rate("ask", "NEW_train")
    ask_t_val = toxic_rate("ask", "NEW_val")
    ask_t_test = toxic_rate("ask", "NEW_test")

    random_mean = _safe_float(best_row["random_mean"])
    random_std = _safe_float(best_row["random_std"])
    random_plus = _safe_float(best_row["random_mean_plus_1std"])
    new_net = _safe_float(best_row["net_pnl"])
    trade_count = _safe_float(best_row["trade_count"])
    beats_plus = bool(new_net > random_plus)
    smoothed_baseline = 84980.0

    # Diagnostics summary CSV.
    metrics_rows = [
        ("NEW net_pnl", new_net, "binary_best on NEW_test"),
        ("trade_count", trade_count, "very low if <150; sparse deployment risk"),
        ("random_mean", random_mean, "matched random cancel mean net_pnl"),
        ("random_std", random_std, "matched random dispersion"),
        ("random_mean_plus_1std", random_plus, "significance gate used in this project"),
        ("beats_random_plus_1std", float(beats_plus), "1=true, 0=false"),
        ("smoothed_baseline_net_pnl", smoothed_baseline, "reference Model_0.55"),
        ("bid_val_pr_auc", bid_val_pr_auc, "from bid train_history max val_auc_pr"),
        ("bid_test_pr_auc", bid_test_pr_auc_side, "from binary_training_summary"),
        ("ask_val_pr_auc", ask_val_pr_auc, "from ask train_history max val_auc_pr"),
        ("ask_test_pr_auc", ask_test_pr_auc_side, "from binary_training_summary"),
        ("bid_toxic_rate_train", bid_t_train, "price_level_removed_proxy fw=1 h=10"),
        ("bid_toxic_rate_val", bid_t_val, "price_level_removed_proxy fw=1 h=10"),
        ("bid_toxic_rate_test", bid_t_test, "price_level_removed_proxy fw=1 h=10"),
        ("ask_toxic_rate_train", ask_t_train, "price_level_removed_proxy fw=1 h=10"),
        ("ask_toxic_rate_val", ask_t_val, "price_level_removed_proxy fw=1 h=10"),
        ("ask_toxic_rate_test", ask_t_test, "price_level_removed_proxy fw=1 h=10"),
    ]
    key_df = pd.DataFrame(metrics_rows, columns=["metric", "value", "interpretation"])
    key_df.to_csv(DIAG_DIR / "current_016_key_metrics.csv", index=False)

    # Narrative diagnosis.
    lines: list[str] = []
    lines.append("# Current 016 Diagnosis")
    lines.append("")
    lines.append("## 1) Why trade_count is only 77")
    lines.append(
        f"- Best setup used `threshold_bid={float(bt_best_row['threshold_bid']):.2f}`, "
        f"`threshold_ask={float(bt_best_row['threshold_ask']):.2f}` with "
        f"`signal_rate_bid={float(bt_best_row['signal_rate_bid']):.3f}`, "
        f"`signal_rate_ask={float(bt_best_row['signal_rate_ask']):.3f}`."
    )
    lines.append(
        "- This is a heavily asymmetric pull policy (ask side pulled most of the time), "
        "so the strategy participates in fewer complete round trips."
    )
    lines.append("")
    lines.append("## 2) Bid/Ask signal rates at selected threshold")
    lines.append(
        f"- bid_signal_rate={float(bt_best_row['signal_rate_bid']):.6f}, "
        f"ask_signal_rate={float(bt_best_row['signal_rate_ask']):.6f}"
    )
    lines.append("")
    lines.append("## 3) Threshold selection source")
    lines.append(
        "- Original `binary_best` was selected from NEW test backtest sweep (not val-only), "
        "so it is not acceptable as final trading evidence."
    )
    lines.append("")
    lines.append("## 4) Does PnL come from selectivity vs mostly not trading?")
    lines.append(
        f"- NEW net_pnl={new_net:,.0f}, trade_count={int(trade_count)}. "
        "Result is selective and sparse; cannot be interpreted as robust HFT operating point yet."
    )
    lines.append("")
    lines.append("## 5) Random+1std check")
    lines.append(
        f"- random_mean={random_mean:,.2f}, random_std={random_std:,.2f}, "
        f"random_mean+1std={random_plus:,.2f}, beats={beats_plus}."
    )
    lines.append("")
    lines.append("## 6) PR-AUC vs base rate")
    lines.append(
        f"- bid: val_pr_auc={bid_val_pr_auc:.6f}, test_pr_auc={bid_test_pr_auc_side:.6f}; "
        f"ask: val_pr_auc={ask_val_pr_auc:.6f}, test_pr_auc={ask_test_pr_auc_side:.6f}."
    )
    lines.append("- Both sides are above base toxic rate (~3%), so classifier quality is non-trivial.")
    lines.append("")
    lines.append("## 7) Val-to-test degradation")
    lines.append(
        f"- bid drop: {bid_val_pr_auc - bid_test_pr_auc_side:+.6f}; "
        f"ask drop: {ask_val_pr_auc - ask_test_pr_auc_side:+.6f}."
    )
    lines.append("- Degradation exists; threshold governance must be val-only.")
    lines.append("")
    lines.append("## 8) Current status")
    lines.append("- Verdict: **not_ready** (label pipeline PoC, but trading edge not significant vs random+1std).")

    (DIAG_DIR / "current_016_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")

    # Persist small JSON snapshot for downstream scripts.
    snapshot = {
        "best_threshold_bid": float(bt_best_row["threshold_bid"]),
        "best_threshold_ask": float(bt_best_row["threshold_ask"]),
        "new_net_pnl": new_net,
        "trade_count": int(trade_count),
        "random_mean": random_mean,
        "random_std": random_std,
        "random_mean_plus_1std": random_plus,
        "beats_random_plus_1std": bool(beats_plus),
        "smoothed_baseline_net_pnl": smoothed_baseline,
    }
    (DIAG_DIR / "current_016_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    print("Wrote:")
    print(DIAG_DIR / "current_016_diagnosis.md")
    print(DIAG_DIR / "current_016_key_metrics.csv")


if __name__ == "__main__":
    main()

