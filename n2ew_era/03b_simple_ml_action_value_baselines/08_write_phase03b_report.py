from __future__ import annotations

import pandas as pd

from common_phase03b import RESULTS, ROOT, write_markdown, read_json


def classify(hurdles: pd.DataFrame, sequential: pd.DataFrame) -> str:
    beats_random_all = bool(hurdles["beats_matched_random_95p"].astype(bool).all())
    beats_queue_all = bool(hurdles["beats_queue_imbalance_rule"].astype(bool).all())
    seq_positive_all = bool((sequential["sequential_excess_pnl"] > 0).all())
    if beats_random_all and beats_queue_all and seq_positive_all:
        return "ml_incremental_value_confirmed"
    if beats_random_all or seq_positive_all:
        return "ml_signal_exists_but_no_strategy_gain"
    return "ml_fails_to_beat_queue_rule"


def main() -> None:
    selected = read_json(RESULTS / "selected_phase03b_policy.json")
    metrics = pd.read_csv(RESULTS / "simple_model_training_metrics.csv")
    thresholds = pd.read_csv(RESULTS / "validation_threshold_selection.csv")
    frozen = pd.read_csv(RESULTS / "frozen_oos_evaluation.csv")
    base = pd.read_csv(RESULTS / "frozen_oos_baseline_controls.csv")
    hurdles = pd.read_csv(RESULTS / "ml_hurdle_summary.csv")
    sequential = pd.read_csv(RESULTS / "causal_fill_sequential_eval.csv")
    status = classify(hurdles, sequential)

    best_thresholds = thresholds.sort_values("total_excess_pnl", ascending=False).head(10)
    lines = [
        "# Simple ML Action-Value Baselines Report",
        "",
        f"Final classification: `{status}`.",
        "",
        "No deep MLP, TCN, or massive grid search was used. No threshold, model, or feature decision was selected on `test`, `NEW3`, or `NEW4`.",
        "",
        "## 1. Input Contract",
        "",
        "Phase 03B used only verified causal L2 features plus rebuilt causal dynamic features. Raw absolute price columns were forbidden by Phase 03A.6 and are absent from the ML feature matrix.",
        "",
        "Feature schema: `feature_schema_03b.json`.",
        "",
        "## 2. Simple Model Training",
        "",
        metrics.to_markdown(index=False),
        "",
        "## 3. Validation-Only Threshold Selection",
        "",
        f"Selected model: `{selected['model_id']}`.",
        "",
        f"Selected threshold: `{selected['threshold']}`.",
        "",
        "Selection split: `val`; selection metric: `validation_total_excess_pnl_H20`.",
        "",
        best_thresholds.to_markdown(index=False),
        "",
        "## 4. Frozen OOS Evaluation",
        "",
        frozen.to_markdown(index=False),
        "",
        "## 5. Baseline Controls",
        "",
        base.to_markdown(index=False),
        "",
        "## 6. Matched Random and Queue-Rule Hurdles",
        "",
        hurdles.to_markdown(index=False),
        "",
        "Matched random uses 1000 finite-population random trials matched by selected count, session, and side. This is a hurdle for signal value, not a strategy tuning mechanism.",
        "",
        "If `empirical_p_value_fraction_random_ge_model` is displayed as `0`, interpret it as: in 1000 matched random trials, no random trial exceeded the selected ML policy; therefore empirical p-value `< 0.001`, not exactly zero.",
        "",
        "## 7. Causal-Fill Sequential Proxy",
        "",
        sequential.to_markdown(index=False),
        "",
        "The sequential fill proxy uses current L2 state only, conservative L2 stress fill probabilities, inventory cap = 1, fee multiplier = 1, and H20 scheduled exit.",
        "",
        "## 8. Interpretation",
        "",
    ]
    if status == "ml_incremental_value_confirmed":
        lines.extend(
            [
                "The simple ML action-value model passed the strict Phase 03B hurdle: it beat matched random 95p, beat the hand-written queue imbalance rule, and remained positive in causal-fill sequential proxy on frozen OOS splits.",
                "",
                "This remains a proxy research result, not real exchange replay or proof of real-world profitability.",
            ]
        )
    elif status == "ml_signal_exists_but_no_strategy_gain":
        lines.extend(
            [
                "The ML model shows some evidence of signal value, but it does not clear every deployment-style hurdle against the queue imbalance rule and causal-fill sequential proxy.",
                "",
                "The correct interpretation is signal discovery, not final strategy replacement.",
            ]
        )
    else:
        lines.extend(
            [
                "The simple ML action-value route failed to beat the hand-written queue imbalance rule under the frozen Phase 03B hurdles.",
                "",
                "The correct next step is not ad-hoc threshold tweaking; any new issue should be handled through a root-cause audit before expanding model complexity.",
            ]
        )
    lines.extend(
        [
            "",
            "Allowed wording: `simple causal L2 ML baseline`, `validation-selected action-value proxy`, `frozen OOS hurdle result`.",
            "",
            "Forbidden wording: real-world profitability, true queue replay, FPGA-proven PnL improvement, or test-tuned threshold success.",
        ]
    )
    write_markdown(ROOT / "SIMPLE_ML_ACTION_VALUE_REPORT.md", lines)
    print(f"wrote {ROOT / 'SIMPLE_ML_ACTION_VALUE_REPORT.md'}")
    print(f"Final classification: {status}")


if __name__ == "__main__":
    main()
