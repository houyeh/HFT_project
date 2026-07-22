from __future__ import annotations

import numpy as np
import pandas as pd

from common_03a import RESULTS, ROOT, write_markdown


def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    size = read_csv(RESULTS / "ml_dataset_size_by_split.csv")
    indep = read_csv(RESULTS / "session_independence_audit.csv")
    feature = read_csv(RESULTS / "feature_drift_for_ml.csv")
    label = read_csv(RESULTS / "label_drift_for_ml.csv")
    signal = read_csv(RESULTS / "single_feature_signal_diagnostics.csv")

    bidask_size = size[size["action_side"].astype(str).isin(["bid", "ask"])] if not size.empty else pd.DataFrame()
    train_rows = int(bidask_size[bidask_size["split"].astype(str) == "train"]["row_count"].sum()) if not bidask_size.empty else 0
    train_positive_rate = float(
        bidask_size[bidask_size["split"].astype(str) == "train"]["positive_count"].sum()
        / max(bidask_size[bidask_size["split"].astype(str) == "train"]["finite_label_count"].sum(), 1)
    ) if not bidask_size.empty else np.nan

    high_overlap_rate = float(indep["high_overlap_warning"].astype(bool).mean()) if not indep.empty else 1.0
    max_label_psi = float(label[label["split"].astype(str).isin(["test", "new4", "new3"])]["psi_vs_train"].max()) if not label.empty and "psi_vs_train" in label else np.nan
    max_feature_psi = float(feature[feature["split"].astype(str).isin(["test", "new4", "new3"])]["psi_vs_train"].max()) if not feature.empty and "psi_vs_train" in feature else np.nan
    missing_event_rate = bool((feature["feature_or_label"].astype(str) == "event_rate_past_50").any() and (feature["status"].astype(str) == "missing_from_phase02f_dataset").any()) if not feature.empty else True

    ok_signal = False
    if not signal.empty and "spearman_corr" in signal:
        valid_signal = signal[signal["status"].astype(str) == "ok"].copy()
        ok_signal = bool((valid_signal["spearman_corr"].abs() > 0.02).any()) if not valid_signal.empty else False

    if train_rows >= 1_000_000 and train_positive_rate > 0.05 and high_overlap_rate < 0.75 and max_label_psi < 0.25 and ok_signal and not missing_event_rate:
        status = "ready_for_tabular_ml"
        reason = "The dataset is large, target balance is usable, drift is moderate, independence is acceptable, and requested causal features are present."
    elif train_rows >= 1_000_000 and train_positive_rate > 0.05 and ok_signal:
        status = "ready_only_for_simple_linear_models"
        reason = "The dataset is large and contains measurable L2 signal, but readiness is constrained by overlapping labels, drift, or missing requested features. Start with simple linear/action-value baselines before larger models."
    else:
        status = "not_ready_due_to_label_noise"
        reason = "The audit does not show enough reliable signal or dataset quality for ML beyond diagnostics."

    lines = [
        "# ML Readiness Report",
        "",
        f"Readiness classification: `{status}`.",
        "",
        reason,
        "",
        "No ML models were trained in this phase. Test, NEW3, and NEW4 were used only for frozen/OOS diagnostics, not tuning.",
        "",
        "## 1. Dataset Size",
        "",
        size.to_markdown(index=False) if not size.empty else "Dataset size audit missing.",
        "",
        f"Bid/ask train action rows: `{train_rows:,}`. Train positive excess rate across bid/ask: `{train_positive_rate:.4f}`.",
        "",
        "## 2. Session Independence",
        "",
        "Autocorrelation is measured by event order within each session/action side. High lag-20 autocorrelation indicates overlapping H20 labels and reduced effective sample independence.",
        "",
        indep.groupby(["split", "action_side"], as_index=False).agg(
            sessions=("session_id", "nunique"),
            mean_lag1_autocorr=("lag1_autocorr", "mean"),
            mean_lag20_autocorr=("lag20_autocorr", "mean"),
            median_effective_sample_ratio_lag1=("effective_sample_ratio_lag1", "median"),
            high_overlap_rate=("high_overlap_warning", "mean"),
        ).to_markdown(index=False) if not indep.empty else "Independence audit missing.",
        "",
        "## 3. Feature / Label Drift",
        "",
        f"Max feature PSI vs train across frozen/OOS splits: `{max_feature_psi:.4f}`.",
        "",
        f"Max label PSI vs train across frozen/OOS splits: `{max_label_psi:.4f}`.",
        "",
        "Feature drift output: `results/feature_drift_for_ml.csv`.",
        "",
        "Label drift output: `results/label_drift_for_ml.csv`.",
        "",
        "## 4. Single-Feature Signal Diagnostics",
        "",
        signal[signal["status"].astype(str) == "ok"].sort_values("spearman_corr", key=lambda s: s.abs(), ascending=False).head(20).to_markdown(index=False) if not signal.empty else "Signal diagnostics missing.",
        "",
        "Bucket-lift figures are in `figures/bucket_lift_*.png`.",
        "",
        "## 5. Readiness Decision",
        "",
        f"`{status}`: {reason}",
        "",
        "Important caveat: `event_rate_past_50` is requested for ML readiness but is absent from the Phase 02F parquet schema. It should be restored before any full ML pipeline if the source supports it.",
        "",
        "Recommended next step: run a small validation-only baseline suite using linear/logistic action-value models before considering tree or neural models.",
    ]
    write_markdown(ROOT / "ML_READINESS_REPORT.md", lines)
    print(f"ML readiness classification: {status}")
    print(reason)


if __name__ == "__main__":
    main()

