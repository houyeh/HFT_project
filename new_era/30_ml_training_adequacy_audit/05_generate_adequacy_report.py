from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common_030 import EXPORT, RESULTS, ROOT, ensure_dirs, write_markdown


def _fmt(value: float) -> str:
    return f"{float(value):,.0f}"


def main() -> None:
    ensure_dirs()
    split_summary = pd.read_csv(EXPORT / "split_summary.csv")
    pipeline = pd.read_csv(RESULTS / "training_pipeline_audit.csv")
    difficulty = pd.read_csv(RESULTS / "residual_universe_difficulty.csv")
    existing = pd.read_csv(RESULTS / "existing_model_threshold_curves.csv")
    metrics = pd.read_csv(RESULTS / "model_family_metrics.csv")
    selection = json.loads((RESULTS / "selected_model_family.json").read_text(encoding="utf-8"))
    final = pd.read_csv(RESULTS / "ml_incremental_value_final.csv")

    selected = selection["selected_model"]
    test_row = final[final["split"] == "test"].iloc[0]
    new4_row = final[final["split"] == "new4"].iloc[0]
    useful = bool(test_row["accepted_useful_ml"] and new4_row["accepted_useful_ml"])
    best_val_models = pd.read_csv(RESULTS / "model_family_validation_selection.csv").sort_values(
        ["after_veto_pnl", "veto_count"], ascending=[False, True]
    ).head(8)

    if useful:
        conclusion = "Expanded tabular models unlocked incremental value under the current validation-selected low-veto rule."
    else:
        conclusion = (
            "Expanded tabular models did not unlock incremental value. This does not prove the residual signal is impossible, "
            "but it does show that the current small tabular residual universe is not adequate; the remaining test is external sequence TCN training using the exported GPU pack."
        )

    lines = [
        "# ML Training Adequacy Report",
        "",
        "## 1. Purpose",
        "",
        "Phase 030 audits whether the residual Quote-EV ML veto failed in Phase 028 because the model was too shallow or because the residual signal is not learnable from the available tabular causal features.",
        "",
        "## 2. GPU Training Pack",
        "",
        f"Pack location: `{EXPORT}`.",
        "",
    ]
    for _, row in split_summary.iterrows():
        lines.append(
            f"- `{row['split']}`: rows `{int(row['rows'])}`, negative EV rate `{row['negative_ev_rate']:.3f}`, "
            f"severe <= -5000 rate `{row['severe_negative_ev_rate_le_minus_5000']:.3f}`, quote EV sum `{_fmt(row['quote_ev_sum'])}`."
        )
    lines.extend(
        [
            "",
            "The pack includes `sequence_index.parquet`, `README_GPU_TRAINING.md`, and `environment.yml`. Sequence rows are pointers only; full temporal windows are not expanded into RAM.",
            "",
            "## 3. Pipeline Audit",
            "",
        ]
    )
    for _, row in pipeline.iterrows():
        lines.append(f"- `{row['audit_item']}`: `{row['value']}` ({row['status']}).")

    lines.extend(["", "## 4. Residual Universe Difficulty", ""])
    for _, row in difficulty.iterrows():
        lines.append(
            f"- `{row['split']}`: samples `{int(row['sample_count'])}`, negative count `{int(row['negative_ev_count'])}`, "
            f"<= -5000 count `{int(row['severe_negative_le_5000_count'])}`, top-10 loss share `{row['top_10_negative_loss_share']:.3f}`."
        )

    lines.extend(["", "## 5. Existing Phase 028 Threshold Diagnostic", ""])
    existing_best = existing[existing["constraints_pass"].astype(bool)].sort_values(["after_veto_pnl", "veto_count"], ascending=[False, True]).head(5)
    for _, row in existing_best.iterrows():
        lines.append(
            f"- `{row['model_name']}` threshold `{row['threshold']:.6f}`: val after-veto PnL `{_fmt(row['after_veto_pnl'])}`, veto count `{int(row['veto_count'])}`."
        )

    lines.extend(["", "## 6. Expanded Model Families", ""])
    lines.append("The expanded audit includes calibrated logistic, histogram gradient boosting, sklearn 64-32 MLP, and PyTorch 64-32 MLP classifier/regressor variants. PyTorch was run locally on CPU; the exported pack is intended for external RTX sequence training.")
    lines.append("")
    for _, row in best_val_models.iterrows():
        lines.append(
            f"- `{row['model_name']}` ({row['objective']}): val after-veto PnL `{_fmt(row['after_veto_pnl'])}`, "
            f"veto count `{int(row['veto_count'])}`, threshold `{row['threshold']:.6f}`."
        )
    selected_metrics = metrics[(metrics["model_name"] == selected["model_name"]) & (metrics["split"].isin(["val", "test", "new4"]))]
    lines.extend(["", f"Selected validation model: `{selected['model_name']}` with threshold `{float(selected['threshold']):.6f}`.", ""])
    for _, row in selected_metrics.iterrows():
        lines.append(
            f"- `{row['split']}` PR-AUC `{row['pr_auc_negative_ev']:.4f}`, ROC-AUC `{row['roc_auc_negative_ev']:.4f}`, Spearman `{row['spearman_score_vs_negative_ev']:.4f}`."
        )

    lines.extend(["", "## 7. Final Incremental Value Test", ""])
    for _, row in final.iterrows():
        lines.append(
            f"- `{row['split']}`: baseline `{_fmt(row['baseline_pnl'])}`, ML after-veto `{_fmt(row['after_veto_pnl'])}`, "
            f"incremental `{_fmt(row['incremental_vs_baseline'])}`, veto rate `{row['veto_rate']:.3f}`, "
            f"random 95p `{_fmt(row['random_veto_95p_pnl'])}`, accepted `{bool(row['accepted_useful_ml'])}`."
        )

    lines.extend(
        [
            "",
            "## 8. Conclusion",
            "",
            conclusion,
            "",
            "The useful-ML rule is strict: validation-selected low-veto ML must improve the frozen baseline and beat matched random veto 95p. Test and NEW4 are not used for model or threshold tuning.",
        ]
    )
    write_markdown(ROOT / "ML_TRAINING_ADEQUACY_REPORT.md", lines)


if __name__ == "__main__":
    main()
    print(f"Wrote adequacy report to {ROOT / 'ML_TRAINING_ADEQUACY_REPORT.md'}")
