from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "07f_sparse_linear_policy"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05C4 = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"
PHASE07D = PROJECT_ROOT / "n2ew_era" / "07d_phc_decision_distillation"
PHASE07E = PROJECT_ROOT / "n2ew_era" / "07e_async_ml_control_plane"

SPLITS = ("val", "new3", "new4")
BUDGETS = (2, 4, 6, 8, 12, 16)
VARIANTS = ("base_uniform", "base_boundary_weighted", "interaction_boundary_weighted")
FILL_SALT = "05c3_stp_v2_conservative_l2_stress"
RIDGE_ALPHA = 1.0
BOUNDARY_WEIGHT_GAIN = 4.0
BOOTSTRAP_SEED = 20261007


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(PHASE05A))
p05a = load_module("phase05a_common_for_07f", PHASE05A / "common_05a.py")


@dataclass
class SparseOutput:
    indices: np.ndarray
    raw_coefficients: np.ndarray
    raw_intercept: float


@dataclass
class SparseModel:
    variant: str
    budget_per_output: int
    feature_names: list[str]
    outputs: list[SparseOutput]

    def predict(self, raw_features: np.ndarray) -> np.ndarray:
        prediction = np.empty((len(raw_features), 2), dtype=np.float64)
        for output_index, output in enumerate(self.outputs):
            prediction[:, output_index] = (
                output.raw_intercept
                + raw_features[:, output.indices] @ output.raw_coefficients
            )
        return prediction


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def load_payload(split: str) -> dict[str, Any]:
    if split.lower() == "new5":
        raise RuntimeError("NEW5 is forbidden in phase 07F")
    payload = p05a.load_sequences(split)
    if bool(payload.get("new5_used", False)):
        raise RuntimeError(f"Payload {split} is marked new5_used=True")
    return payload


def teacher_scores(split: str, payload: dict[str, Any]) -> np.ndarray:
    path = PHASE07D / "results" / f"teacher_scores_{split}.npy"
    values = np.load(path).astype(np.float64, copy=False)
    if values.shape != (len(payload["x"]), 2):
        raise RuntimeError(f"Teacher cache mismatch for {split}: {values.shape}")
    return values


def base_features(payload: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    x = payload["x"][:, -1, :].float().numpy().astype(np.float64, copy=False)
    return x, list(payload["feature_names"])


def feature_matrix(payload: dict[str, Any], interactions: bool) -> tuple[np.ndarray, list[str]]:
    base, names = base_features(payload)
    if not interactions:
        return base, names
    columns = [base]
    output_names = list(names)
    pair_columns = []
    for left in range(base.shape[1]):
        for right in range(left + 1, base.shape[1]):
            pair_columns.append(base[:, left] * base[:, right])
            output_names.append(f"{names[left]}*{names[right]}")
    columns.append(np.column_stack(pair_columns))
    return np.column_stack(columns), output_names


def boundary_weights(scores: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    threshold_distance = np.abs(np.max(scores, axis=1) - threshold)
    side_distance = 0.5 * np.abs(scores[:, 0] - scores[:, 1])
    distance = np.minimum(threshold_distance, side_distance)
    positive = distance[np.isfinite(distance) & (distance > 0)]
    scale = float(np.quantile(positive, 0.25)) if len(positive) else 1.0
    scale = max(scale, 1e-12)
    weights = 1.0 + BOUNDARY_WEIGHT_GAIN * np.exp(-distance / scale)
    return weights.astype(np.float64), scale


def fit_sparse_output(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, budget: int,
    mean: np.ndarray, scale: np.ndarray
) -> SparseOutput:
    standardized = (x - mean) / scale
    ranking_model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ranking_model.fit(standardized, y, sample_weight=weights)
    order = np.argsort(np.abs(ranking_model.coef_))[::-1]
    indices = np.sort(order[: min(budget, x.shape[1])])
    selected_model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    selected_model.fit(standardized[:, indices], y, sample_weight=weights)
    raw_coefficients = selected_model.coef_ / scale[indices]
    raw_intercept = float(selected_model.intercept_ - np.sum(selected_model.coef_ * mean[indices] / scale[indices]))
    return SparseOutput(indices=indices, raw_coefficients=raw_coefficients, raw_intercept=raw_intercept)


def fit_models(train_payload: dict[str, Any], train_scores: np.ndarray, threshold: float) -> tuple[dict[tuple[str, int], SparseModel], dict[str, Any]]:
    models: dict[tuple[str, int], SparseModel] = {}
    weighted, boundary_scale = boundary_weights(train_scores, threshold)
    fit_metadata: dict[str, Any] = {"boundary_scale": boundary_scale}
    for variant in VARIANTS:
        interactions = variant.startswith("interaction")
        x, names = feature_matrix(train_payload, interactions)
        mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        scale[scale < 1e-12] = 1.0
        weights = weighted if "boundary_weighted" in variant else np.ones(len(x), dtype=np.float64)
        for budget in BUDGETS:
            outputs = [
                fit_sparse_output(x, train_scores[:, output], weights, budget, mean, scale)
                for output in range(2)
            ]
            models[(variant, budget)] = SparseModel(variant, budget, names, outputs)
        del x
        gc.collect()
    return models, fit_metadata


def causal_metrics(split: str, policy_id: str, payload: dict[str, Any], decision: np.ndarray, norms: Any) -> dict[str, Any]:
    probabilities = p05a.fill_probabilities(payload, decision, norms)
    fills = (decision != p05a.DECISION_NO_QUOTE) & (
        p05a.stable_uniform(payload, decision, FILL_SALT) < probabilities
    )
    pnl = p05a.decision_pnl_from_payload(payload, decision)
    total, fills_count, forced, max_inventory, max_drawdown, wins, losses, _ = p05a.simulate_scheduled_by_event_index(
        payload, decision, fills, pnl, inventory_cap=1
    )
    return {
        "split": split,
        "policy_id": policy_id,
        "sequential_excess_pnl": float(total),
        "fill_count": int(fills_count),
        "avg_pnl_per_fill": float(total / fills_count) if fills_count else 0.0,
        "max_drawdown": float(max_drawdown),
        "forced_no_quote_count": int(forced),
        "max_abs_inventory": int(max_inventory),
        "fill_win_rate": float(wins / fills_count) if fills_count else 0.0,
        "fill_loss_rate": float(losses / fills_count) if fills_count else 0.0,
        "selected_action_count": int((decision != p05a.DECISION_NO_QUOTE).sum()),
        "bid_count": int((decision == p05a.DECISION_BID).sum()),
        "ask_count": int((decision == p05a.DECISION_ASK).sum()),
        "no_quote_count": int((decision == p05a.DECISION_NO_QUOTE).sum()),
    }


def evaluate_models(
    models: dict[tuple[str, int], SparseModel], threshold: float, norms: Any
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    decisions: dict[tuple[str, str, int], np.ndarray] = {}
    for split in SPLITS:
        payload = load_payload(split)
        teacher = teacher_scores(split, payload)
        teacher_decision = p05a.policy_from_scores(teacher, threshold)
        matrices: dict[bool, np.ndarray] = {}
        for variant in VARIANTS:
            interactions = variant.startswith("interaction")
            if interactions not in matrices:
                matrices[interactions] = feature_matrix(payload, interactions)[0]
            x = matrices[interactions]
            for budget in BUDGETS:
                model = models[(variant, budget)]
                prediction = model.predict(x)
                decision = p05a.policy_from_scores(prediction, threshold)
                policy_id = f"sparse_{variant}_k{budget}"
                row = causal_metrics(split, policy_id, payload, decision, norms)
                disagreement = decision != teacher_decision
                teacher_selected = teacher_decision != p05a.DECISION_NO_QUOTE
                student_selected = decision != p05a.DECISION_NO_QUOTE
                selected_union = teacher_selected | student_selected
                row.update({
                    "variant": variant,
                    "budget_per_output": budget,
                    "coefficient_count": int(sum(len(output.indices) for output in model.outputs)),
                    "unique_term_count": int(len(set(np.concatenate([output.indices for output in model.outputs]).tolist()))),
                    "score_r2_uniform": float(r2_score(teacher, prediction, multioutput="uniform_average")),
                    "score_r2_bid": float(r2_score(teacher[:, 0], prediction[:, 0])),
                    "score_r2_ask": float(r2_score(teacher[:, 1], prediction[:, 1])),
                    "action_agreement_rate": float(np.mean(~disagreement)),
                    "decision_flip_count": int(disagreement.sum()),
                    "selected_union_agreement_rate": float(np.mean(~disagreement[selected_union])) if selected_union.any() else 1.0,
                    "teacher_selection_rate": float(teacher_selected.mean()),
                    "student_selection_rate": float(student_selected.mean()),
                    "selection_split": "val",
                    "new3_used_for_selection": False,
                    "new4_used_for_selection": False,
                    "new5_used": False,
                })
                rows.append(row)
                decisions[(split, variant, budget)] = decision
        del payload, teacher, matrices
        gc.collect()
    return pd.DataFrame(rows), decisions


def reference_results(threshold: float, norms: Any, static_theta: float) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rows = []
    decisions: dict[tuple[str, str], np.ndarray] = {}
    for split in SPLITS:
        payload = load_payload(split)
        teacher = teacher_scores(split, payload)
        queue = payload["target_raw"].numpy()[:, p05a.TARGET_RAW_COLUMNS.index("queue_imbalance")]
        static = np.zeros(len(queue), dtype=np.int8)
        static[np.isfinite(queue) & (queue > static_theta)] = p05a.DECISION_BID
        static[np.isfinite(queue) & (queue < -static_theta)] = p05a.DECISION_ASK
        policies = {
            "queue_theta_val_tuned": static,
            "phc_eventwise_reference": p05a.policy_from_scores(teacher, threshold),
        }
        for policy_id, decision in policies.items():
            rows.append(causal_metrics(split, policy_id, payload, decision, norms))
            decisions[(split, policy_id)] = decision
        del payload, teacher
        gc.collect()
    return pd.DataFrame(rows), decisions


def select_champion(metrics: pd.DataFrame) -> pd.Series:
    val = metrics[metrics["split"] == "val"].copy()
    return val.sort_values(
        ["sequential_excess_pnl", "coefficient_count", "unique_term_count"],
        ascending=[False, True, True],
    ).iloc[0]


def coefficient_table(models: dict[tuple[str, int], SparseModel]) -> pd.DataFrame:
    rows = []
    for (variant, budget), model in models.items():
        for output_index, output in enumerate(model.outputs):
            rows.append({
                "variant": variant,
                "budget_per_output": budget,
                "output": "bid" if output_index == 0 else "ask",
                "term": "<intercept>",
                "raw_coefficient": output.raw_intercept,
                "is_interaction": False,
            })
            for index, coefficient in zip(output.indices, output.raw_coefficients):
                term = model.feature_names[int(index)]
                rows.append({
                    "variant": variant,
                    "budget_per_output": budget,
                    "output": "bid" if output_index == 0 else "ask",
                    "term": term,
                    "raw_coefficient": float(coefficient),
                    "is_interaction": "*" in term,
                })
    return pd.DataFrame(rows)


def session_values(payload: dict[str, Any], decision: np.ndarray, norms: Any) -> dict[str, float]:
    session_codes = payload["session_code"].numpy()
    n = len(decision)
    values: dict[str, float] = {}
    for code, label in enumerate(payload["session_labels"]):
        mask = session_codes == code
        sub: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == n:
                sub[key] = value[mask]
            else:
                sub[key] = value
        sub["session_code"] = torch.full((int(mask.sum()),), int(code), dtype=torch.int32)
        sub["session_labels"] = list(payload["session_labels"])
        values[str(label)] = causal_metrics("session", "session", sub, decision[mask], norms)["sequential_excess_pnl"]
    return values


def paired_bootstrap(
    champion: pd.Series, model_decisions: dict[tuple[str, str, int], np.ndarray],
    reference_decisions: dict[tuple[str, str], np.ndarray], norms: Any, n_trials: int = 10000
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    variant = str(champion["variant"])
    budget = int(champion["budget_per_output"])
    for split in SPLITS:
        payload = load_payload(split)
        student_decision = model_decisions[(split, variant, budget)]
        student_sessions = session_values(payload, student_decision, norms)
        student_full = causal_metrics(split, "student", payload, student_decision, norms)["sequential_excess_pnl"]
        for comparator in ("queue_theta_val_tuned", "phc_eventwise_reference"):
            comparator_decision = reference_decisions[(split, comparator)]
            comparator_sessions = session_values(payload, comparator_decision, norms)
            comparator_full = causal_metrics(split, comparator, payload, comparator_decision, norms)["sequential_excess_pnl"]
            labels = sorted(student_sessions)
            differences = np.array(
                [student_sessions[label] - comparator_sessions[label] for label in labels], dtype=np.float64
            )
            samples = rng.choice(differences, size=(n_trials, len(differences)), replace=True).mean(axis=1)
            rows.append({
                "split": split,
                "student_policy_id": champion["policy_id"],
                "comparator": comparator,
                "session_count": len(labels),
                "full_split_total_difference": float(student_full - comparator_full),
                "session_reset_total_difference": float(differences.sum()),
                "mean_session_difference": float(differences.mean()),
                "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
                "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_probability_difference_le_zero": float((samples <= 0).mean()),
                "n_bootstrap": n_trials,
            })
        del payload
        gc.collect()
    return pd.DataFrame(rows)


def make_figure(metrics: pd.DataFrame, references: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for axis, split in zip(axes, SPLITS):
        subset = metrics[metrics["split"] == split]
        for variant in VARIANTS:
            rows = subset[subset["variant"] == variant].sort_values("budget_per_output")
            axis.plot(rows["coefficient_count"], rows["sequential_excess_pnl"], marker="o", label=variant)
        phc = references[(references["split"] == split) & (references["policy_id"] == "phc_eventwise_reference")].iloc[0]
        static = references[(references["split"] == split) & (references["policy_id"] == "queue_theta_val_tuned")].iloc[0]
        axis.axhline(float(phc["sequential_excess_pnl"]), color="gray", linestyle=":", label="PHC")
        axis.axhline(float(static["sequential_excess_pnl"]), color="black", linestyle="--", label="static queue")
        axis.set_title(split)
        axis.set_xlabel("Nonzero coefficients (bid + ask)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Causal-fill sequential excess PnL (proxy units)")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Sparse eventwise policy capacity sweep")
    fig.tight_layout()
    fig.savefig(FIGURES / "sparse_policy_pnl_frontier.png", dpi=180)
    plt.close(fig)


def write_report(
    metrics: pd.DataFrame, references: pd.DataFrame, champion: pd.Series,
    coefficients: pd.DataFrame, bootstrap: pd.DataFrame, threshold: float,
    static_theta: float, fit_metadata: dict[str, Any]
) -> None:
    champion_rows = metrics[
        (metrics["variant"] == champion["variant"])
        & (metrics["budget_per_output"] == champion["budget_per_output"])
    ].copy()
    merged = champion_rows.merge(
        references.pivot(index="split", columns="policy_id", values="sequential_excess_pnl").reset_index(),
        on="split",
    )
    merged["pnl_retention_vs_phc_pct"] = 100.0 * merged["sequential_excess_pnl"] / merged["phc_eventwise_reference"]
    merged["pnl_delta_vs_static"] = merged["sequential_excess_pnl"] - merged["queue_theta_val_tuned"]
    val = merged[merged["split"] == "val"].iloc[0]
    oos = merged[merged["split"].isin(["new3", "new4"])]
    positive = bool(
        val["pnl_retention_vs_phc_pct"] >= 80.0
        and val["pnl_delta_vs_static"] > 0
        and np.all(oos["pnl_delta_vs_static"] > 0)
    )
    status = "resolved_positive" if positive else "resolved_negative"
    champion_coefficients = coefficients[
        (coefficients["variant"] == champion["variant"])
        & (coefficients["budget_per_output"] == champion["budget_per_output"])
    ]
    summary_cols = [
        "split", "variant", "budget_per_output", "coefficient_count", "unique_term_count",
        "score_r2_uniform", "action_agreement_rate", "decision_flip_count",
        "sequential_excess_pnl", "fill_count", "avg_pnl_per_fill", "max_drawdown",
        "phc_eventwise_reference", "pnl_retention_vs_phc_pct", "queue_theta_val_tuned",
        "pnl_delta_vs_static",
    ]
    lines = [
        "# Sparse Linear Policy Audit\n",
        f"status: `{status}`\n",
        "## Claim boundary\n",
        "This is a CPU-only task-level feasibility experiment. It does not provide HLS, routed timing, power, or on-board evidence. The frozen 05C4 seed-2 PHC scores are the distillation target; PHC itself is not retrained. Train fits coefficients, val selects the model, and NEW3/NEW4 are post-selection diagnostics. NEW5 was not loaded.\n",
        "## Pre-registered candidates\n",
        f"Three model families were evaluated with `{list(BUDGETS)}` nonzero terms per output: base features with uniform score regression, base features with decision-boundary weighting, and base plus all pairwise interactions with boundary weighting. Ridge alpha is `{RIDGE_ALPHA}`. Boundary samples receive up to `{1 + BOUNDARY_WEIGHT_GAIN:.1f}x` weight; the train-only distance scale is `{fit_metadata['boundary_scale']:.8g}`. The frozen PHC action threshold is `{threshold:g}` and the val-tuned queue comparator threshold is `{static_theta:g}`.\n",
        "## Val-selected champion\n",
        f"Val selected `{champion['policy_id']}`: variant `{champion['variant']}`, `{int(champion['budget_per_output'])}` terms per output, `{int(champion['coefficient_count'])}` total coefficients, and `{int(champion['unique_term_count'])}` unique terms. Selection used causal-fill sequential PnL, not R-squared.\n",
        merged[summary_cols].to_markdown(index=False, floatfmt=".4f") + "\n",
        "## Full capacity sweep\n",
        metrics[["split", "variant", "budget_per_output", "coefficient_count", "unique_term_count", "score_r2_uniform", "action_agreement_rate", "sequential_excess_pnl", "fill_count", "avg_pnl_per_fill", "max_drawdown"]].to_markdown(index=False, floatfmt=".4f") + "\n",
        "## Champion formula\n",
        "The following coefficients are in the original feature basis. The score is `intercept + sum(coefficient * term)`; a term containing `*` is one pairwise product. These are directly inspectable formulas, although fixed-point implementation has not yet been tested.\n",
        champion_coefficients.to_markdown(index=False, floatfmt=".9g") + "\n",
        "## Paired session bootstrap\n",
        "The canonical simulator carries its inventory variable across session boundaries. Session bootstrap necessarily resets inventory for each sampled session, so both the canonical full-split difference and the session-reset estimand are shown. Four to six sessions per split provide low statistical power.\n",
        bootstrap.to_markdown(index=False, floatfmt=".6f") + "\n",
        "## Decision\n",
        (
            "The sparse policy clears the pre-registered 80% PHC-PnL retention threshold on val and remains above the tuned static queue rule on both OOS diagnostics. It is worth a separate fixed-point/HLS validation.\n"
            if positive
            else "The sparse policy does not simultaneously retain at least 80% of PHC val PnL and beat the tuned static queue rule on both NEW3 and NEW4. This experiment therefore does not yet justify an HLS implementation.\n"
        ),
        "## Outputs\n",
        "- `results/sparse_policy_metrics.csv`\n- `results/sparse_policy_coefficients.csv`\n- `results/reference_results.csv`\n- `results/selected_champion.json`\n- `results/paired_session_bootstrap.csv`\n- `figures/sparse_policy_pnl_frontier.png`\n",
    ]
    (ROOT / "AUDIT_sparse_linear_policy.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    policy = json.loads((PHASE05C4 / "results" / "phc_stp_v2_selected_policy.json").read_text(encoding="utf-8"))
    threshold = float(policy["threshold"])
    metadata_07e = json.loads((PHASE07E / "results" / "run_metadata.json").read_text(encoding="utf-8"))
    static_theta = float(metadata_07e["selected_static_theta"])
    norms = p05a.compute_fill_norms_from_train_sequences()

    train_payload = load_payload("train")
    train_teacher = teacher_scores("train", train_payload)
    models, fit_metadata = fit_models(train_payload, train_teacher, threshold)
    del train_payload, train_teacher
    gc.collect()

    metrics, model_decisions = evaluate_models(models, threshold, norms)
    references, reference_decisions = reference_results(threshold, norms, static_theta)
    champion = select_champion(metrics)
    coefficients = coefficient_table(models)
    bootstrap = paired_bootstrap(champion, model_decisions, reference_decisions, norms)

    metrics.to_csv(RESULTS / "sparse_policy_metrics.csv", index=False)
    references.to_csv(RESULTS / "reference_results.csv", index=False)
    coefficients.to_csv(RESULTS / "sparse_policy_coefficients.csv", index=False)
    bootstrap.to_csv(RESULTS / "paired_session_bootstrap.csv", index=False)
    champion_payload = {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in champion.to_dict().items()
    }
    (RESULTS / "selected_champion.json").write_text(json.dumps(champion_payload, indent=2), encoding="utf-8")
    run_metadata = {
        "teacher": "05c4_phc_stp_cosine_lr seed2 frozen score cache from 07D",
        "fit_split": "train", "selection_split": "val",
        "oos_diagnostic_splits": ["new3", "new4"], "new5_used": False,
        "budgets_per_output": list(BUDGETS), "variants": list(VARIANTS),
        "ridge_alpha": RIDGE_ALPHA, "boundary_weight_gain": BOUNDARY_WEIGHT_GAIN,
        "phc_threshold": threshold, "static_queue_theta": static_theta,
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    make_figure(metrics, references)
    write_report(metrics, references, champion, coefficients, bootstrap, threshold, static_theta, fit_metadata)
    print(f"[OK] champion={champion['policy_id']} val_pnl={champion['sequential_excess_pnl']:.2f}", flush=True)
    print(f"[OK] report={ROOT / 'AUDIT_sparse_linear_policy.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
