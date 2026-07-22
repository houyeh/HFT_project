from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import common_05d as c

PHASE05A = c.PHASE05A
if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))

import common_05a as p05a  # noqa: E402

SEEDS = [0, 1, 2]
LAMBDA_GRID = [0.0, 0.25, 0.5, 1.0]
POLICY_ID = "phc_uncertainty_gating"


def load_seed_scores(split: str) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    payload = p05a.load_sequences(split)
    scores: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        model, ckpt = p05a.load_model(PHASE05A / "models" / f"phc_lite_seed{seed}_best_val.pt")
        scores[seed] = p05a.predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
    return payload, scores


def adjusted_scores(seed_scores: dict[int, np.ndarray], lambda_uncertainty: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack([seed_scores[seed] for seed in SEEDS], axis=0)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    adjusted = mean - float(lambda_uncertainty) * std
    return adjusted, mean, std


def threshold_grid(scores: np.ndarray) -> list[float]:
    max_scores = np.max(scores, axis=1)
    finite = max_scores[np.isfinite(max_scores)]
    if len(finite) == 0:
        return [0.0]
    qs = np.quantile(finite, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return sorted(set([0.0, *qs.tolist()]))


def select_on_validation() -> dict[str, Any]:
    payload, seed_scores = load_seed_scores("val")
    rows = []
    for lam in LAMBDA_GRID:
        scores, _, _ = adjusted_scores(seed_scores, lam)
        for threshold in threshold_grid(scores):
            decision = p05a.policy_from_scores(scores, threshold)
            metric = p05a.metric_from_decision("val", POLICY_ID, payload, decision)
            oracle = c.oracle_capture_for_decision(payload, decision)
            causal = c.causal_fill_for_decision("val", POLICY_ID, payload, decision, f"05d_uncertainty_{lam}_{threshold}")
            _, oracle_action, _ = _oracle_tuple(payload)
            err = c.error_type(decision, oracle_action)
            metric.update(
                {
                    "lambda_uncertainty": float(lam),
                    "threshold": float(threshold),
                    "selection_split": "val",
                    "oracle_capture_ratio": oracle["capture_ratio"],
                    "causal_fill_pnl": causal["sequential_excess_pnl"],
                    "false_positive_rate": float((err == "false_positive").mean()),
                    "missed_opportunity_rate": float((err == "missed_opportunity").mean()),
                    "side_error_rate": float((err == "side_error").mean()),
                    "new3_used_for_selection": False,
                    "new4_used_for_selection": False,
                    "new5_used_for_selection": False,
                }
            )
            rows.append(metric)
    grid = pd.DataFrame(rows)
    grid.to_csv(c.RESULTS / "phc_uncertainty_threshold_search.csv", index=False)
    best = grid.sort_values(
        ["total_excess_pnl_H20", "causal_fill_pnl", "oracle_capture_ratio", "selected_action_count"],
        ascending=[False, False, False, True],
    ).iloc[0].to_dict()
    selected = {
        "model": POLICY_ID,
        "selection_split": "val",
        "selection_metric": "validation_total_excess_pnl_H20_then_causal_fill_then_oracle_capture",
        "lambda_uncertainty": float(best["lambda_uncertainty"]),
        "threshold": float(best["threshold"]),
        "target": "excess_pnl_H20",
        "seed_ensemble": SEEDS,
        "score_adjustment": "adjusted_score = seed_mean_score - lambda_uncertainty * seed_score_std",
        "validation_metrics": p05a.json_sanitize(best),
        "new3_used_for_selection": False,
        "new4_used_for_selection": False,
        "new5_used_for_selection": False,
    }
    c.write_json(c.RESULTS / "phc_uncertainty_selected_policy.json", selected)
    return selected


def _oracle_tuple(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oracle_action, oracle_pnl, oracle_value = c.oracle_decision_and_pnl(payload)
    return oracle_pnl, oracle_action, oracle_value


def evaluate_policy(selected: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    action_rows = []
    causal_rows = []
    error_rows = []
    for split in c.SPLITS:
        payload, seed_scores = load_seed_scores(split)
        scores, mean_scores, std_scores = adjusted_scores(seed_scores, float(selected["lambda_uncertainty"]))
        decision = p05a.policy_from_scores(scores, float(selected["threshold"]))
        metric = p05a.metric_from_decision(split, POLICY_ID, payload, decision)
        oracle = c.oracle_capture_for_decision(payload, decision)
        metric.update(
            {
                "lambda_uncertainty": float(selected["lambda_uncertainty"]),
                "threshold": float(selected["threshold"]),
                "oracle_capture_ratio": oracle["capture_ratio"],
                "matched_count_oracle_pnl": oracle["matched_count_oracle_pnl"],
                "opportunity_gap": oracle["opportunity_gap"],
                "mean_score_bid": float(np.nanmean(mean_scores[:, 0])),
                "mean_score_ask": float(np.nanmean(mean_scores[:, 1])),
                "mean_std_bid": float(np.nanmean(std_scores[:, 0])),
                "mean_std_ask": float(np.nanmean(std_scores[:, 1])),
            }
        )
        action_rows.append(metric)
        causal_rows.append(c.causal_fill_for_decision(split, POLICY_ID, payload, decision, "05d_uncertainty_selected"))
        oracle_action, _, _ = c.oracle_decision_and_pnl(payload)
        err = c.error_type(decision, oracle_action)
        for error_name, count in pd.Series(err).value_counts().items():
            error_rows.append(
                {
                    "split": split,
                    "policy_id": POLICY_ID,
                    "error_type": error_name,
                    "state_count": int(count),
                    "error_rate": float(count / max(len(err), 1)),
                }
            )
    action_df = pd.DataFrame(action_rows)
    causal_df = pd.DataFrame(causal_rows)
    error_df = pd.DataFrame(error_rows)
    action_df.to_csv(c.RESULTS / "phc_uncertainty_oos_eval.csv", index=False)
    causal_df.to_csv(c.RESULTS / "phc_uncertainty_causal_fill.csv", index=False)
    error_df.to_csv(c.RESULTS / "phc_uncertainty_oracle_error.csv", index=False)
    return action_df, causal_df, error_df


def write_report(selected: dict[str, Any], action: pd.DataFrame, causal: pd.DataFrame, errors: pd.DataFrame) -> None:
    v1_action = pd.read_csv(PHASE05A / "results" / "phc_lite_oos_policy_eval.csv")
    v1_action = v1_action[v1_action["policy_id"].eq("phc_lite")][["split", "total_excess_pnl_H20"]].rename(columns={"total_excess_pnl_H20": "v1_action_pnl"})
    merged = action.merge(v1_action, on="split", how="left")
    v1_causal = pd.read_csv(PHASE05A / "results" / "phc_lite_causal_fill_sequential_eval.csv")
    v1_causal = v1_causal[v1_causal["policy_id"].eq("phc_lite")][["split", "sequential_excess_pnl"]].rename(columns={"sequential_excess_pnl": "v1_causal_pnl"})
    merged = merged.merge(v1_causal, on="split", how="left")
    merged = merged.merge(causal[["split", "sequential_excess_pnl"]].rename(columns={"sequential_excess_pnl": "uncertainty_causal_pnl"}), on="split", how="left")
    merged["action_delta_vs_v1"] = merged["total_excess_pnl_H20"] - merged["v1_action_pnl"]
    merged["causal_delta_vs_v1"] = merged["uncertainty_causal_pnl"] - merged["v1_causal_pnl"]
    merged.to_csv(c.RESULTS / "phc_uncertainty_vs_v1_summary.csv", index=False)
    fp = errors[errors["error_type"].eq("false_positive")].set_index("split")["error_rate"].to_dict()
    missed = errors[errors["error_type"].eq("missed_opportunity")].set_index("split")["error_rate"].to_dict()
    accepted = bool((merged["action_delta_vs_v1"] >= 0).all() and (merged["causal_delta_vs_v1"] >= 0).all())
    lines = [
        "# PHC Uncertainty Gating Report",
        "",
        "This is a validation-selected ensemble disagreement diagnostic. It uses the frozen PHC-lite v1 seed checkpoints and does not retrain the model.",
        "",
        "## Contract",
        "- NEW5 usage: none.",
        "- Selection split: validation only.",
        "- Adjustment: `adjusted_score = seed_mean_score - lambda_uncertainty * seed_score_std`.",
        f"- Selected lambda: `{selected['lambda_uncertainty']}`.",
        f"- Selected threshold: `{selected['threshold']}`.",
        "",
        "## OOS Summary",
        merged[["split", "total_excess_pnl_H20", "v1_action_pnl", "action_delta_vs_v1", "uncertainty_causal_pnl", "v1_causal_pnl", "causal_delta_vs_v1", "oracle_capture_ratio"]].to_markdown(index=False),
        "",
        "## Oracle Error Direction",
        f"- False-positive rates by split: `{p05a.json_sanitize(fp)}`.",
        f"- Missed-opportunity rates by split: `{p05a.json_sanitize(missed)}`.",
        "",
        "## Decision",
        (
            "`uncertainty_gate_candidate`: uncertainty gating improves or matches PHC-lite v1 on both action-level and causal-fill metrics across val/NEW3/NEW4."
            if accepted
            else "`uncertainty_gate_not_adopted`: validation-selected uncertainty gating does not improve PHC-lite v1 across all required gates."
        ),
        "",
        "Oracle labels are diagnostic only and were not used for threshold selection. This does not imply real-world profitability or true exchange replay.",
    ]
    c.write_markdown(c.ROOT / "PHC_UNCERTAINTY_GATING_REPORT.md", lines)


def main() -> None:
    c.ensure_dirs()
    selected = select_on_validation()
    action, causal, errors = evaluate_policy(selected)
    write_report(selected, action, causal, errors)
    print("wrote PHC uncertainty gating outputs")


if __name__ == "__main__":
    main()
