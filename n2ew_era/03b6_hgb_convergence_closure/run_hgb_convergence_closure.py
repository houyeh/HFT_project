from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "03b6_hgb_convergence_closure"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
PREDICTIONS = ROOT / "predictions"
PHASE03B = PROJECT_ROOT / "n2ew_era" / "03b_simple_ml_action_value_baselines"

if str(PHASE03B) not in sys.path:
    sys.path.insert(0, str(PHASE03B))

from common_phase03b import (  # noqa: E402
    ACTION_ASK,
    ACTION_BID,
    DECISION_ASK,
    DECISION_BID,
    DECISION_NO_QUOTE,
    FROZEN_SPLITS,
    HORIZON,
    POSITIVE_TARGET,
    RANDOM_TRIALS,
    TARGET,
    build_feature_frame,
    compute_train_fill_norms,
    conservative_l2_fill_mask,
    evaluate_decision,
    feature_matrix_columns,
    load_train_matrix,
    load_val_matrix,
    max_drawdown,
    part_paths,
    policy_pnl_from_decision,
    read_json,
    safe_spearman,
    select_best_quote,
    simulate_scheduled_excess,
    write_json,
    write_markdown,
)


BUDGETS = [40, 80, 120, 160, 240, 320]
SEED = 20260601


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, MODELS, FIGURES, PREDICTIONS]:
        path.mkdir(parents=True, exist_ok=True)


def hgb_regressor(max_iter: int, early_stopping: bool = False) -> HistGradientBoostingRegressor:
    params: dict[str, Any] = {
        "max_iter": int(max_iter),
        "max_leaf_nodes": 15,
        "learning_rate": 0.05,
        "l2_regularization": 0.1,
        "random_state": SEED,
        "early_stopping": early_stopping,
    }
    if early_stopping:
        params.update({"validation_fraction": 0.1, "n_iter_no_change": 10, "tol": 1e-7})
    return HistGradientBoostingRegressor(**params)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return np.array([np.inf])
    quantiles = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995, 1.0]
    return np.sort(np.unique(np.concatenate([np.asarray([0.0]), np.nanquantile(finite, quantiles)])))


def validation_threshold_select(model: Any, model_id: str, val: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    scores = model.predict(val[feature_matrix_columns()].to_numpy(dtype=np.float32)).astype(np.float32)
    quote_mask = val["action_slot"].astype(int).isin([ACTION_BID, ACTION_ASK]).to_numpy()
    best: dict[str, Any] | None = None
    best_threshold = 0.0
    for threshold in threshold_grid(scores[quote_mask]):
        decision = select_best_quote(val, scores, float(threshold))
        metrics = evaluate_decision(val, decision, "val", "hgb_iter_budget_threshold", model_id, float(threshold))
        if best is None or metrics["total_excess_pnl"] > best["total_excess_pnl"]:
            best = metrics
            best_threshold = float(threshold)
    assert best is not None
    return best_threshold, best


def train_eval_budget_sweep() -> pd.DataFrame:
    train = load_train_matrix()
    val = load_val_matrix()
    x_train = train[feature_matrix_columns()].to_numpy(dtype=np.float32)
    y_train = train[TARGET].to_numpy(dtype=np.float32)
    y_train_cls = train[POSITIVE_TARGET].astype(int).to_numpy()
    x_val = val[feature_matrix_columns()].to_numpy(dtype=np.float32)
    y_val = val[TARGET].to_numpy(dtype=np.float32)
    y_val_cls = val[POSITIVE_TARGET].astype(int).to_numpy()
    rows = []
    for budget in BUDGETS:
        model_id = f"hgb_regressor_iter_{budget}"
        model = hgb_regressor(budget, early_stopping=False)
        model.fit(x_train, y_train)
        joblib.dump(model, MODELS / f"{model_id}.joblib")
        train_score = model.predict(x_train).astype(np.float32)
        val_score = model.predict(x_val).astype(np.float32)
        threshold, val_policy = validation_threshold_select(model, model_id, val)
        rows.append(
            {
                "model_id": model_id,
                "max_iter": budget,
                "n_iter": int(getattr(model, "n_iter_", budget)),
                "hit_iteration_cap": bool(int(getattr(model, "n_iter_", budget)) >= budget),
                "train_rmse": rmse(y_train, train_score),
                "val_rmse": rmse(y_val, val_score),
                "train_mae": float(mean_absolute_error(y_train, train_score)),
                "val_mae": float(mean_absolute_error(y_val, val_score)),
                "train_spearman": safe_spearman(train_score, y_train),
                "val_spearman": safe_spearman(val_score, y_val),
                "train_pr_auc_positive_excess": float(average_precision_score(y_train_cls, train_score)),
                "val_pr_auc_positive_excess": float(average_precision_score(y_val_cls, val_score)),
                "train_roc_auc_positive_excess": float(roc_auc_score(y_train_cls, train_score)),
                "val_roc_auc_positive_excess": float(roc_auc_score(y_val_cls, val_score)),
                "selected_threshold": threshold,
                "val_selected_action_count": val_policy["selected_action_count"],
                "val_selection_rate": val_policy["selection_rate"],
                "val_policy_total_excess_pnl": val_policy["total_excess_pnl"],
                "val_policy_max_drawdown": val_policy["max_drawdown"],
                "selection_split": "val",
                "test_new3_new4_used_for_selection": False,
            }
        )
        print(f"budget sweep complete max_iter={budget}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "hgb_iteration_budget_sweep.csv", index=False)
    plot_budget_sweep(out)
    return out


def plot_budget_sweep(sweep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(sweep["max_iter"], sweep["val_policy_total_excess_pnl"], marker="o", linewidth=1.8)
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_title("HGB Validation Policy PnL by Iteration Budget")
    ax.set_xlabel("max_iter")
    ax.set_ylabel("Validation total excess PnL H20")
    fig.tight_layout()
    fig.savefig(FIGURES / "hgb_val_policy_pnl_by_iter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def early_stopping_audit() -> pd.DataFrame:
    train = load_train_matrix()
    val = load_val_matrix()
    x_train = train[feature_matrix_columns()].to_numpy(dtype=np.float32)
    y_train = train[TARGET].to_numpy(dtype=np.float32)
    y_train_cls = train[POSITIVE_TARGET].astype(int).to_numpy()
    x_val = val[feature_matrix_columns()].to_numpy(dtype=np.float32)
    y_val = val[TARGET].to_numpy(dtype=np.float32)
    y_val_cls = val[POSITIVE_TARGET].astype(int).to_numpy()
    rows = []
    staged_rows = []
    for max_iter in [320, 500, 800]:
        model_id = f"hgb_regressor_early_stop_{max_iter}"
        model = hgb_regressor(max_iter=max_iter, early_stopping=True)
        model.fit(x_train, y_train)
        joblib.dump(model, MODELS / f"{model_id}.joblib")
        val_score = model.predict(x_val).astype(np.float32)
        train_score = model.predict(x_train).astype(np.float32)
        threshold, val_policy = validation_threshold_select(model, model_id, val)
        train_scores = list(getattr(model, "train_score_", []))
        val_scores = list(getattr(model, "validation_score_", []))
        for i in range(max(len(train_scores), len(val_scores))):
            staged_rows.append(
                {
                    "model_id": model_id,
                    "iteration": i,
                    "train_score": train_scores[i] if i < len(train_scores) else np.nan,
                    "validation_score": val_scores[i] if i < len(val_scores) else np.nan,
                }
            )
        rows.append(
            {
                "model_id": model_id,
                "max_iter": max_iter,
                "n_iter": int(getattr(model, "n_iter_", max_iter)),
                "terminated_before_cap": bool(int(getattr(model, "n_iter_", max_iter)) < max_iter),
                "tol": 1e-7,
                "n_iter_no_change": 10,
                "validation_fraction": 0.1,
                "train_rmse": rmse(y_train, train_score),
                "val_rmse": rmse(y_val, val_score),
                "train_pr_auc_positive_excess": float(average_precision_score(y_train_cls, train_score)),
                "val_pr_auc_positive_excess": float(average_precision_score(y_val_cls, val_score)),
                "val_spearman": safe_spearman(val_score, y_val),
                "selected_threshold": threshold,
                "val_policy_total_excess_pnl": val_policy["total_excess_pnl"],
                "selection_split": "val",
                "test_new3_new4_used_for_selection": False,
            }
        )
        print(f"early stopping audit complete max_iter={max_iter}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "hgb_early_stopping_audit.csv", index=False)
    pd.DataFrame(staged_rows).to_csv(RESULTS / "hgb_early_stopping_staged_scores.csv", index=False)
    write_markdown(
        ROOT / "HGB_EARLY_STOPPING_AUDIT.md",
        [
            "# HGB Early Stopping Audit",
            "",
            "Explicit early stopping was configured with `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=10`, and `tol=1e-7`. The internal validation fraction is drawn from the training split only; frozen OOS splits are not used.",
            "",
            out.to_markdown(index=False),
            "",
            "Staged internal scores are written to `results/hgb_early_stopping_staged_scores.csv`.",
        ],
    )
    return out


def state_triplets(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bid = matrix[matrix["action_slot"].astype(int) == ACTION_BID].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    ask = matrix[matrix["action_slot"].astype(int) == ACTION_ASK].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    return bid, ask


def selected_actions_for_matrix(matrix: pd.DataFrame, model: Any, model_id: str, threshold: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    scores = model.predict(matrix[feature_matrix_columns()].to_numpy(dtype=np.float32)).astype(np.float32)
    result = select_best_quote(matrix, scores, threshold)
    metrics = evaluate_decision(matrix, result, str(matrix["split"].iloc[0]), "hgb_convergence_selected", model_id, threshold)
    states = matrix[matrix["action_slot"].astype(int) == ACTION_BID].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    selected = pd.DataFrame(
        {
            "split": states["split"].astype(str).to_numpy(),
            "session_id": states["session_id"].astype(str).to_numpy(),
            "symbol": states["symbol"].astype(str).to_numpy(),
            "event_index": states["event_index"].to_numpy(),
            "timestamp": states["timestamp"].to_numpy(),
            "decision": result.decision.astype(np.int8),
            "selected_side": result.selected_side,
            "selected_excess_pnl_H20": result.selected_pnl.astype(np.float32),
            "selected_score": result.selected_score.astype(np.float32),
        }
    )
    return selected, metrics


def queue_rule_summary(matrix: pd.DataFrame, split: str) -> pd.DataFrame:
    base = pd.read_csv(PHASE03B / "results" / "frozen_oos_baseline_controls.csv")
    return base[(base["split"].eq(split)) & (base["policy_id"].eq("side_by_queue_imbalance_abs_0p1"))].copy()


def exact_hypergeom_trials(values: np.ndarray, k: int, trials: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    k = int(k)
    if k <= 0 or n <= 0:
        return np.zeros(trials, dtype=np.float64)
    if k >= n:
        return np.full(trials, float(values.sum()), dtype=np.float64)
    unique, counts = np.unique(values, return_counts=True)
    out = np.zeros(trials, dtype=np.float64)
    for i in range(trials):
        sampled_counts = rng.multivariate_hypergeometric(counts.astype(np.int64), k)
        out[i] = float(np.dot(sampled_counts, unique))
    return out


def exact_random_summary(split: str, selected: pd.DataFrame, trials: int = RANDOM_TRIALS) -> dict[str, Any]:
    rng = np.random.default_rng(SEED + len(split))
    selected_nonzero = selected[selected["decision"].astype(int) != DECISION_NO_QUOTE].copy()
    required = selected_nonzero.groupby(["session_id", "selected_side"]).size().to_dict()
    random_trials = np.zeros(trials, dtype=np.float64)
    for path in part_paths(split):
        matrix = build_feature_frame(path, include_targets=True)
        for side, slot in [("bid", ACTION_BID), ("ask", ACTION_ASK)]:
            part = matrix[matrix["action_slot"].astype(int) == slot]
            session_id = str(part["session_id"].iloc[0])
            k = int(required.get((session_id, side), 0))
            random_trials += exact_hypergeom_trials(part[TARGET].to_numpy(dtype=np.float64), k, trials, rng)
    model_total = float(selected_nonzero["selected_excess_pnl_H20"].sum())
    return {
        "matched_random_mean": float(np.mean(random_trials)),
        "matched_random_95p": float(np.quantile(random_trials, 0.95)),
        "matched_random_99p": float(np.quantile(random_trials, 0.99)),
        "empirical_p_value_fraction_random_ge_model": float(np.mean(random_trials >= model_total)),
        "random_trials": int(trials),
        "random_method": "exact_multivariate_hypergeometric_matched_count_session_side",
    }


def frozen_oos_eval(selected_model_id: str, selected_threshold: float) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    model = joblib.load(MODELS / f"{selected_model_id}.joblib")
    old = pd.read_csv(PHASE03B / "results" / "frozen_oos_evaluation.csv").assign(comparison_policy="old_selected_hgb_iter_80")
    queue = pd.read_csv(PHASE03B / "results" / "frozen_oos_baseline_controls.csv")
    queue = queue[queue["policy_id"].eq("side_by_queue_imbalance_abs_0p1")].assign(comparison_policy="queue_imbalance_rule")
    rows = []
    selected_by_split: dict[str, pd.DataFrame] = {}
    for split in FROZEN_SPLITS:
        metrics_rows = []
        selected_frames = []
        for path in part_paths(split):
            matrix = build_feature_frame(path, include_targets=True)
            selected, metrics = selected_actions_for_matrix(matrix, model, selected_model_id, selected_threshold)
            selected_frames.append(selected)
            metrics_rows.append(metrics)
            print(f"frozen oos selected model split={split} part={path.name}")
        selected_all = pd.concat(selected_frames, ignore_index=True)
        selected_by_split[split] = selected_all
        metrics_df = pd.DataFrame(metrics_rows)
        summary = metrics_df.groupby(["split", "policy_id", "model_id", "horizon", "threshold"], as_index=False).agg(
            state_count=("state_count", "sum"),
            selected_action_count=("selected_action_count", "sum"),
            bid_count=("bid_count", "sum"),
            ask_count=("ask_count", "sum"),
            total_excess_pnl=("total_excess_pnl", "sum"),
            max_drawdown=("max_drawdown", "max"),
        )
        summary["selection_rate"] = summary["selected_action_count"] / summary["state_count"].clip(lower=1)
        rand = exact_random_summary(split, selected_all)
        q = queue[queue["split"].eq(split)]
        o = old[old["split"].eq(split)]
        for row in summary.to_dict(orient="records"):
            row.update(rand)
            row["comparison_policy"] = "validation_selected_hgb_convergence"
            row["queue_rule_total_excess_pnl"] = float(q["total_excess_pnl"].iloc[0]) if not q.empty else np.nan
            row["old_selected_hgb_total_excess_pnl"] = float(o["total_excess_pnl"].iloc[0]) if not o.empty else np.nan
            row["beats_matched_random_95p"] = bool(row["total_excess_pnl"] > row["matched_random_95p"])
            row["beats_queue_imbalance_rule"] = bool(row["total_excess_pnl"] > row["queue_rule_total_excess_pnl"])
            row["beats_old_selected_hgb"] = bool(row["total_excess_pnl"] > row["old_selected_hgb_total_excess_pnl"])
            row["selection_split"] = "val"
            row["test_new3_new4_used_for_selection"] = False
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "hgb_convergence_frozen_oos_eval.csv", index=False)
    return out, selected_by_split


def selected_pnl_from_matrix(matrix: pd.DataFrame, decision: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    states = matrix[matrix["action_slot"].astype(int) == ACTION_BID].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    bid = matrix[matrix["action_slot"].astype(int) == ACTION_BID].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    ask = matrix[matrix["action_slot"].astype(int) == ACTION_ASK].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    pnl = np.zeros(len(states), dtype=np.float32)
    bid_mask = decision == DECISION_BID
    ask_mask = decision == DECISION_ASK
    pnl[bid_mask] = bid[TARGET].to_numpy(dtype=np.float32)[bid_mask]
    pnl[ask_mask] = ask[TARGET].to_numpy(dtype=np.float32)[ask_mask]
    return states, np.nan_to_num(pnl, nan=0.0)


def causal_fill_eval(selected_by_split: dict[str, pd.DataFrame]) -> pd.DataFrame:
    norms = compute_train_fill_norms()
    rows = []
    for split, selected in selected_by_split.items():
        selected_by_session = {str(k): g.sort_values("event_index").reset_index(drop=True) for k, g in selected.groupby("session_id")}
        totals = {"pnl": 0.0, "fills": 0, "forced": 0, "max_inv": 0, "max_dd": 0.0, "wins": 0.0, "losses": 0.0}
        for path in part_paths(split):
            matrix = build_feature_frame(path, include_targets=True)
            session_id = str(matrix["session_id"].iloc[0])
            decision = selected_by_session[session_id]["decision"].astype(np.int8).to_numpy()
            states, pnl = selected_pnl_from_matrix(matrix, decision)
            fills, _ = conservative_l2_fill_mask(states, decision, norms)
            result = simulate_scheduled_excess(decision, fills, pnl, horizon=HORIZON, inventory_cap=1)
            totals["pnl"] += result["sequential_excess_pnl"]
            totals["fills"] += result["fill_count"]
            totals["forced"] += result["forced_no_quote_count"]
            totals["max_inv"] = max(totals["max_inv"], result["max_abs_inventory"])
            totals["max_dd"] = max(totals["max_dd"], result["max_drawdown"])
            totals["wins"] += result["win_rate"] * result["fill_count"]
            totals["losses"] += result["loss_rate"] * result["fill_count"]
        fills_n = int(totals["fills"])
        rows.append(
            {
                "split": split,
                "policy_id": "hgb_convergence_selected",
                "horizon": HORIZON,
                "fill_model": "conservative_l2_stress",
                "inventory_cap": 1,
                "fee_multiplier": 1.0,
                "sequential_excess_pnl": float(totals["pnl"]),
                "fill_count": fills_n,
                "forced_no_quote_count": int(totals["forced"]),
                "max_abs_inventory": int(totals["max_inv"]),
                "max_drawdown": float(totals["max_dd"]),
                "win_rate": float(totals["wins"] / max(fills_n, 1)) if fills_n else 0.0,
                "loss_rate": float(totals["losses"] / max(fills_n, 1)) if fills_n else 0.0,
                "avg_trade_excess_pnl": float(totals["pnl"] / max(fills_n, 1)) if fills_n else 0.0,
                "fill_uses_future_pnl": False,
                "test_new3_new4_used_for_selection": False,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "hgb_convergence_causal_fill_eval.csv", index=False)
    return out


def final_status(sweep: pd.DataFrame, early: pd.DataFrame, frozen: pd.DataFrame) -> str:
    best = sweep.sort_values("val_policy_total_excess_pnl", ascending=False).iloc[0]
    best_iter = int(best["max_iter"])
    old_val = float(sweep.loc[sweep["max_iter"].eq(80), "val_policy_total_excess_pnl"].iloc[0])
    best_val = float(best["val_policy_total_excess_pnl"])
    val_improved = best_val > old_val * 1.01
    selected_hit_cap = bool(best["hit_iteration_cap"])
    early_terminated = bool(early["terminated_before_cap"].astype(bool).any())
    beats_queue_all = bool(frozen["beats_queue_imbalance_rule"].astype(bool).all())
    beats_old_all = bool(frozen["beats_old_selected_hgb"].astype(bool).all())
    if selected_hit_cap and best_iter == max(BUDGETS) and val_improved and not early_terminated:
        return "hgb_underfit_at_80_rerun_required"
    if val_improved and not beats_old_all:
        return "hgb_overfits_with_more_iterations"
    if beats_queue_all:
        return "hgb_convergence_closed_ml_gain_confirmed"
    return "hgb_convergence_closed_but_no_strategy_gain"


def write_report(sweep: pd.DataFrame, early: pd.DataFrame, frozen: pd.DataFrame, seq: pd.DataFrame) -> None:
    best = sweep.sort_values("val_policy_total_excess_pnl", ascending=False).iloc[0]
    old80 = sweep[sweep["max_iter"].eq(80)].iloc[0]
    status = final_status(sweep, early, frozen)
    lines = [
        "# HGB Convergence Closure Report",
        "",
        f"Final status: `{status}`.",
        "",
        "No TCN, large MLP, or frozen-OOS tuning was performed. Iteration budget and threshold selection were based strictly on validation.",
        "",
        "## 1. Iteration Budget Sweep",
        "",
        sweep.to_markdown(index=False),
        "",
        f"Best validation budget: `{int(best['max_iter'])}` with validation policy PnL `{float(best['val_policy_total_excess_pnl']):,.0f}`.",
        "",
        f"Old max_iter=80 validation policy PnL: `{float(old80['val_policy_total_excess_pnl']):,.0f}`.",
        "",
        "Figure: `figures/hgb_val_policy_pnl_by_iter.png`.",
        "",
        "## 2. Early Stopping Audit",
        "",
        early.to_markdown(index=False),
        "",
        "## 3. Frozen OOS Evaluation",
        "",
        frozen.to_markdown(index=False),
        "",
        "## 4. Causal-Fill Sequential Recheck",
        "",
        seq.to_markdown(index=False),
        "",
        "## 5. Answers",
        "",
        f"Did validation policy PnL improve with more iterations? `{'yes' if float(best['val_policy_total_excess_pnl']) > float(old80['val_policy_total_excess_pnl']) else 'no'}`.",
        "",
        "Did frozen OOS improve? See `beats_old_selected_hgb` and total excess PnL columns in `results/hgb_convergence_frozen_oos_eval.csv`.",
        "",
        f"Did ML finally beat the queue rule on NEW3/NEW4? `{'yes' if bool(frozen[frozen['split'].isin(['new3','new4'])]['beats_queue_imbalance_rule'].all()) else 'no'}`.",
        "",
        "If empirical random p-value is displayed as 0, interpret it as no matched random trial exceeded the model in 1000 trials, i.e. empirical p-value `< 0.001`, not exactly zero probability.",
    ]
    write_markdown(ROOT / "HGB_CONVERGENCE_CLOSURE_REPORT.md", lines)


def main() -> None:
    ensure_dirs()
    sweep = train_eval_budget_sweep()
    early = early_stopping_audit()
    best = sweep.sort_values("val_policy_total_excess_pnl", ascending=False).iloc[0]
    selected_model_id = str(best["model_id"])
    selected_threshold = float(best["selected_threshold"])
    write_json(
        RESULTS / "selected_hgb_convergence_policy.json",
        {
            "policy_id": "hgb_convergence_selected",
            "model_id": selected_model_id,
            "max_iter": int(best["max_iter"]),
            "threshold": selected_threshold,
            "selection_split": "val",
            "selection_metric": "validation_total_excess_pnl_H20",
            "validation_metrics": best.to_dict(),
            "test_new3_new4_used_for_selection": False,
        },
    )
    frozen, selected_by_split = frozen_oos_eval(selected_model_id, selected_threshold)
    seq = causal_fill_eval(selected_by_split)
    write_report(sweep, early, frozen, seq)
    print(f"wrote {ROOT / 'HGB_CONVERGENCE_CLOSURE_REPORT.md'}")
    print(frozen.to_string(index=False))


if __name__ == "__main__":
    main()

