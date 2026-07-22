from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import COMPARISON_DIR, MODELS_DIR, REPORTS_DIR


def _side_reliability(side: str) -> dict[str, object]:
    model_dir = MODELS_DIR / f"{side}_primary_fw1_h10_posw"
    cfg = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    hist = pd.read_csv(model_dir / "train_history.csv")
    tdiag = pd.read_csv(model_dir / "threshold_diagnostics_test.csv")

    first_train = float(hist.iloc[0]["train_loss"])
    last_train = float(hist.iloc[-1]["train_loss"])
    first_val = float(hist.iloc[0]["val_loss"])
    min_val = float(hist["val_loss"].min())
    last_val = float(hist.iloc[-1]["val_loss"])
    best_epoch = int(hist.loc[hist["val_loss"].idxmin(), "epoch"])

    converged = (last_train < first_train) and (min_val <= first_val)
    pr_auc_above_base = float(cfg["test_pr_auc"]) > float(cfg["positive_rate_test"])
    overfit = (last_val > min_val * 1.10) and (last_train < first_train * 0.8)
    signal_rate_05 = float(tdiag.loc[(tdiag["threshold"] - 0.5).abs().idxmin(), "signal_rate"])
    degenerate = bool((signal_rate_05 < 1e-4) or (signal_rate_05 > 0.9999))
    usable = bool(converged and pr_auc_above_base and (not degenerate))

    best_row = tdiag.sort_values("f1", ascending=False).iloc[0]
    return {
        "side": side,
        "best_epoch": best_epoch,
        "first_train_loss": first_train,
        "last_train_loss": last_train,
        "first_val_loss": first_val,
        "min_val_loss": min_val,
        "last_val_loss": last_val,
        "val_pr_auc": float(cfg["best_val_pr_auc"]),
        "test_pr_auc": float(cfg["test_pr_auc"]),
        "test_roc_auc": float(cfg["test_roc_auc"]),
        "positive_rate_test": float(cfg["positive_rate_test"]),
        "converged": converged,
        "pr_auc_above_base_rate": pr_auc_above_base,
        "overfit_risk": overfit,
        "probability_degenerate": degenerate,
        "usable_for_backtest": usable,
        "best_threshold": float(best_row["threshold"]),
        "best_threshold_f1": float(best_row["f1"]),
        "signal_rate_at_0p5": signal_rate_05,
    }


def rewrite_training_reliability_report() -> None:
    bid = _side_reliability("bid")
    ask = _side_reliability("ask")
    rows = [bid, ask]
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "training_reliability_metrics.csv", index=False)

    lines = []
    lines.append("# Training Reliability Report")
    lines.append("")
    for item in rows:
        lines.append(f"## {item['side']} model")
        lines.append(f"- Did {item['side']} model converge? {'YES' if item['converged'] else 'NO'}")
        lines.append(
            f"- Does PR-AUC exceed positive base rate? {'YES' if item['pr_auc_above_base_rate'] else 'NO'} "
            f"(test_pr_auc={item['test_pr_auc']:.6f}, base_rate={item['positive_rate_test']:.6f})"
        )
        lines.append(f"- Is there overfitting risk? {'YES' if item['overfit_risk'] else 'NO'}")
        lines.append(f"- Are probabilities degenerate? {'YES' if item['probability_degenerate'] else 'NO'}")
        lines.append(f"- Is model usable for backtest? {'YES' if item['usable_for_backtest'] else 'NO'}")
        lines.append(
            f"- Best epoch={item['best_epoch']}, min_val_loss={item['min_val_loss']:.6f}, "
            f"best_threshold={item['best_threshold']:.2f}, best_threshold_f1={item['best_threshold_f1']:.6f}"
        )
        lines.append("")
    lines.append("Notes:")
    lines.append("- This report evaluates pos_weight configuration only (no unweighted run in this pass).")
    (REPORTS_DIR / "training_reliability_report.md").write_text("\n".join(lines), encoding="utf-8")


def rewrite_model_comparison_summary() -> None:
    comp = pd.read_csv(COMPARISON_DIR / "binary_vs_baselines.csv")
    train_summary = pd.read_csv(MODELS_DIR / "binary_training_summary.csv")
    bid_cfg = json.loads((MODELS_DIR / "bid_primary_fw1_h10_posw" / "model_config.json").read_text(encoding="utf-8"))
    ask_cfg = json.loads((MODELS_DIR / "ask_primary_fw1_h10_posw" / "model_config.json").read_text(encoding="utf-8"))
    bid_td = pd.read_csv(MODELS_DIR / "bid_primary_fw1_h10_posw" / "threshold_diagnostics_test.csv")
    ask_td = pd.read_csv(MODELS_DIR / "ask_primary_fw1_h10_posw" / "threshold_diagnostics_test.csv")

    bid_best = bid_td.sort_values("f1", ascending=False).iloc[0]
    ask_best = ask_td.sort_values("f1", ascending=False).iloc[0]
    test_f1_best_threshold = float((float(bid_best["f1"]) + float(ask_best["f1"])) / 2.0)

    best = comp[(comp["dataset"] == "NEW_test") & (comp["method"] == "binary_best")].iloc[0]
    new3 = comp[(comp["dataset"] == "NEW3_full") & (comp["method"] == "binary_best_from_new")]
    new3_net = float(new3.iloc[0]["net_pnl"]) if not new3.empty else np.nan

    rec = "not_ready"
    if bool(best["beats_random_mean_plus_1std"]) and bool(best["beats_smoothed_baseline"]):
        rec = "replace_smoothed_baseline"
    elif bool(best["beats_random_mean_plus_1std"]):
        rec = "keep_as_auxiliary_signal"

    row = {
        "model_name": "bid_quote_toxic_binary+ask_quote_toxic_binary",
        "label_type": "fill_conditional_quote_risk_binary_proxyC",
        "side_specific": True,
        "fill_conditional": True,
        "train_sample_count": int(bid_cfg["train_samples"]),
        "val_sample_count": int(bid_cfg["val_samples"]),
        "test_sample_count": int(bid_cfg["test_samples"]),
        "positive_rate_train": float(bid_cfg["positive_rate_train"]),
        "positive_rate_val": float(bid_cfg["positive_rate_val"]),
        "positive_rate_test": float(bid_cfg["positive_rate_test"]),
        "test_pr_auc": float(train_summary["test_auc_pr"].mean()),
        "test_roc_auc": float(train_summary["test_auc_roc"].mean()),
        "test_f1_best_threshold": test_f1_best_threshold,
        "best_threshold": str(best["threshold_setup"]),
        "test_net_pnl": float(best["net_pnl"]),
        "test_vs_random_plus_1std": float(best["net_pnl"] - best["random_mean_plus_1std"]),
        "test_vs_smoothed_baseline": float(best["net_pnl"] - 84980.0),
        "NEW3_net_pnl": new3_net,
        "recommendation": rec,
    }
    pd.DataFrame([row]).to_csv(COMPARISON_DIR / "model_comparison_summary.csv", index=False)


def main() -> None:
    rewrite_training_reliability_report()
    rewrite_model_comparison_summary()
    print("Postprocess reports updated.")


if __name__ == "__main__":
    main()
