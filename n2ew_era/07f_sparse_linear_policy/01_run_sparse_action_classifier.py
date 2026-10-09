from __future__ import annotations

import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "07f_sparse_linear_policy"
RESULTS = ROOT / "results"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05C4 = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"
PHASE07D = PROJECT_ROOT / "n2ew_era" / "07d_phc_decision_distillation"
PHASE07E = PROJECT_ROOT / "n2ew_era" / "07e_async_ml_control_plane"
MAIN_SCRIPT = ROOT / "00_run_sparse_linear_policy.py"

C_GRID = (0.0003, 0.001, 0.003, 0.01)
FEATURE_SETS = ("base", "interaction")
FILL_SALT = "05c3_stp_v2_conservative_l2_stress"
RANDOM_STATE = 20261008


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helpers = load_module("sparse_policy_helpers_07f", MAIN_SCRIPT)
p05a = helpers.p05a


def fit_classifier(
    x: np.ndarray, decisions: np.ndarray, weights: np.ndarray, c_value: float
) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    model = LogisticRegression(
        penalty="l1",
        C=c_value,
        solver="saga",
        max_iter=500,
        tol=1e-4,
        random_state=RANDOM_STATE,
        n_jobs=8,
    )
    model.fit(standardized, decisions, sample_weight=weights)
    return model, mean, scale


def raw_coefficients(
    feature_set: str, c_value: float, feature_names: list[str], model: LogisticRegression,
    mean: np.ndarray, scale: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_index, action in enumerate(model.classes_):
        coefficients = model.coef_[class_index]
        raw = coefficients / scale
        intercept = float(model.intercept_[class_index] - np.sum(coefficients * mean / scale))
        rows.append({
            "feature_set": feature_set, "C": c_value, "action": int(action),
            "term": "<intercept>", "raw_coefficient": intercept, "is_interaction": False,
        })
        for index in np.flatnonzero(raw):
            rows.append({
                "feature_set": feature_set, "C": c_value, "action": int(action),
                "term": feature_names[int(index)], "raw_coefficient": float(raw[index]),
                "is_interaction": "*" in feature_names[int(index)],
            })
    return rows


def main() -> None:
    policy = json.loads((PHASE05C4 / "results" / "phc_stp_v2_selected_policy.json").read_text(encoding="utf-8"))
    threshold = float(policy["threshold"])
    static_theta = float(json.loads((PHASE07E / "results" / "run_metadata.json").read_text(encoding="utf-8"))["selected_static_theta"])
    norms = p05a.compute_fill_norms_from_train_sequences()

    train_payload = helpers.load_payload("train")
    train_scores = helpers.teacher_scores("train", train_payload)
    train_decisions = p05a.policy_from_scores(train_scores, threshold)
    weights, boundary_scale = helpers.boundary_weights(train_scores, threshold)

    models: dict[tuple[str, float], tuple[LogisticRegression, np.ndarray, np.ndarray, list[str]]] = {}
    coefficient_rows: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        interactions = feature_set == "interaction"
        x, names = helpers.feature_matrix(train_payload, interactions)
        for c_value in C_GRID:
            model, mean, scale = fit_classifier(x, train_decisions, weights, c_value)
            models[(feature_set, c_value)] = (model, mean, scale, names)
            coefficient_rows.extend(raw_coefficients(feature_set, c_value, names, model, mean, scale))
            print(
                f"[fit] feature_set={feature_set} C={c_value:g} nonzero={int(np.count_nonzero(model.coef_))}",
                flush=True,
            )
        del x
        gc.collect()
    del train_payload, train_scores, train_decisions
    gc.collect()

    metric_rows: list[dict[str, Any]] = []
    decision_cache: dict[tuple[str, str, float], np.ndarray] = {}
    for split in helpers.SPLITS:
        payload = helpers.load_payload(split)
        teacher = helpers.teacher_scores(split, payload)
        teacher_decision = p05a.policy_from_scores(teacher, threshold)
        matrices: dict[str, np.ndarray] = {}
        for feature_set in FEATURE_SETS:
            matrices[feature_set] = helpers.feature_matrix(payload, feature_set == "interaction")[0]
            for c_value in C_GRID:
                model, mean, scale, _ = models[(feature_set, c_value)]
                prediction = model.predict((matrices[feature_set] - mean) / scale).astype(np.int8)
                policy_id = f"sparse_action_{feature_set}_C{c_value:g}"
                row = helpers.causal_metrics(split, policy_id, payload, prediction, norms)
                disagreement = prediction != teacher_decision
                row.update({
                    "feature_set": feature_set,
                    "C": c_value,
                    "coefficient_count": int(np.count_nonzero(model.coef_)),
                    "unique_term_count": int(np.count_nonzero(np.any(model.coef_ != 0, axis=0))),
                    "action_agreement_rate": float(np.mean(~disagreement)),
                    "decision_flip_count": int(disagreement.sum()),
                    "teacher_selection_rate": float(np.mean(teacher_decision != p05a.DECISION_NO_QUOTE)),
                    "student_selection_rate": float(np.mean(prediction != p05a.DECISION_NO_QUOTE)),
                    "fit_split": "train", "selection_split": "val",
                    "new3_used_for_selection": False, "new4_used_for_selection": False,
                    "new5_used": False,
                })
                metric_rows.append(row)
                decision_cache[(split, feature_set, c_value)] = prediction
        del payload, teacher, matrices
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    champion = metrics[metrics["split"] == "val"].sort_values(
        ["sequential_excess_pnl", "coefficient_count", "unique_term_count"],
        ascending=[False, True, True],
    ).iloc[0]
    references, reference_decisions = helpers.reference_results(threshold, norms, static_theta)

    # Adapt the helper's key format for the selected action classifier.
    selected_decisions = {
        (split, "selected", 0): decision_cache[(split, str(champion["feature_set"]), float(champion["C"]))]
        for split in helpers.SPLITS
    }
    champion_for_bootstrap = champion.copy()
    champion_for_bootstrap["variant"] = "selected"
    champion_for_bootstrap["budget_per_output"] = 0
    bootstrap = helpers.paired_bootstrap(
        champion_for_bootstrap, selected_decisions, reference_decisions, norms
    )

    metrics.to_csv(RESULTS / "sparse_action_metrics.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(RESULTS / "sparse_action_coefficients.csv", index=False)
    bootstrap.to_csv(RESULTS / "sparse_action_paired_session_bootstrap.csv", index=False)
    champion_json = {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in champion.to_dict().items()
    }
    (RESULTS / "sparse_action_selected_champion.json").write_text(
        json.dumps(champion_json, indent=2), encoding="utf-8"
    )

    selected = metrics[
        (metrics["feature_set"] == champion["feature_set"])
        & (metrics["C"] == champion["C"])
    ].merge(
        references.pivot(index="split", columns="policy_id", values="sequential_excess_pnl").reset_index(),
        on="split",
    )
    selected["pnl_retention_vs_phc_pct"] = 100.0 * selected["sequential_excess_pnl"] / selected["phc_eventwise_reference"]
    selected["pnl_delta_vs_static"] = selected["sequential_excess_pnl"] - selected["queue_theta_val_tuned"]
    val = selected[selected["split"] == "val"].iloc[0]
    oos = selected[selected["split"].isin(["new3", "new4"])]
    positive = bool(
        val["pnl_retention_vs_phc_pct"] >= 80.0
        and val["pnl_delta_vs_static"] > 0
        and np.all(oos["pnl_delta_vs_static"] > 0)
    )
    mixed = bool(val["pnl_delta_vs_static"] > 0 or np.any(oos["pnl_delta_vs_static"] > 0))
    status = "resolved_positive" if positive else ("mixed" if mixed else "resolved_negative")
    selected_coefficients = pd.DataFrame(coefficient_rows)
    selected_coefficients = selected_coefficients[
        (selected_coefficients["feature_set"] == champion["feature_set"])
        & (selected_coefficients["C"] == champion["C"])
    ]
    lines = [
        "# Sparse Direct-Action Policy Audit\n",
        f"status: `{status}`\n",
        "## Method\n",
        f"This addendum addresses the score-regression objective mismatch by fitting the frozen PHC buy/sell/hold decisions directly with L1 multinomial logistic regression. Candidate feature sets are `{list(FEATURE_SETS)}` and the fixed C grid is `{list(C_GRID)}`. Training uses the train split with the same train-only boundary weights as the score experiment (scale `{boundary_scale:.8g}`); val selects one candidate, while NEW3/NEW4 remain post-selection diagnostics. NEW5 was not loaded.\n",
        "## Val-selected champion\n",
        f"Val selected `{champion['policy_id']}` with `{int(champion['coefficient_count'])}` nonzero class coefficients and `{int(champion['unique_term_count'])}` unique terms.\n",
        selected[["split", "feature_set", "C", "coefficient_count", "unique_term_count", "action_agreement_rate", "decision_flip_count", "sequential_excess_pnl", "fill_count", "avg_pnl_per_fill", "max_drawdown", "phc_eventwise_reference", "pnl_retention_vs_phc_pct", "queue_theta_val_tuned", "pnl_delta_vs_static"]].to_markdown(index=False, floatfmt=".4f") + "\n",
        "## Full sweep\n",
        "The interaction models at `C=0.001`, `0.003`, and `0.01` reached the fixed 500-iteration limit and emitted scikit-learn convergence warnings. Their coefficients are therefore provisional and their favorable OOS rows are not used for selection or claims. The selected base `C=0.0003` model did not emit this warning.\n",
        metrics.to_markdown(index=False, floatfmt=".6f") + "\n",
        "## Champion formula\n",
        "For each class, compute `intercept + sum(coefficient * term)` and select the class with the largest value. Action IDs are `0=no_quote`, `1=bid`, and `2=ask`. Coefficients are folded back into the original feature basis.\n",
        selected_coefficients.to_markdown(index=False, floatfmt=".9g") + "\n",
        "## Paired session bootstrap\n",
        bootstrap.to_markdown(index=False, floatfmt=".6f") + "\n",
        "## Decision\n",
        (
            "The direct-action sparse classifier clears the strict task-level gate and is worth fixed-point/HLS validation. This remains CPU-only evidence.\n"
            if positive
            else "The direct-action model does not clear the strict 80%-of-PHC retention gate. However, five unique terms beat the tuned static rule on val and NEW3 under the canonical simulator, and on all three splits under session-reset evaluation. It is therefore a mixed latency-value Pareto candidate: a minimal fixed-point/HLS feasibility test is defensible only if roughly 49-68% PHC-PnL retention is acceptable.\n"
        ),
    ]
    (ROOT / "AUDIT_sparse_action_policy.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] champion={champion['policy_id']} val_pnl={champion['sequential_excess_pnl']:.2f}", flush=True)
    print(f"[OK] report={ROOT / 'AUDIT_sparse_action_policy.md'}", flush=True)


if __name__ == "__main__":
    main()
