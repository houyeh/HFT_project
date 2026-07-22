from __future__ import annotations

import hashlib
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "04b1_ml_completion_patch"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

PHASE04B = PROJECT_ROOT / "n2ew_era" / "04b_txf_only_final_development_rerun"
if str(PHASE04B) not in sys.path:
    sys.path.insert(0, str(PHASE04B))

import run_phase04b_pipeline as p04b  # noqa: E402


RANDOM_TRIALS = 1000
RANDOM_SEED = 20260603
ITER_BUDGETS = [40, 80, 120, 160, 240, 320]
TRAIN_SAMPLE_ROWS = 400_000
VAL_SAMPLE_ROWS = 200_000
SPLITS = ["val", "new3", "new4"]
OOS_SPLITS = ["new3", "new4"]
ML_POLICY_ID = "ml_hgb_regressor_shallow"
QUEUE_POLICY_ID = "side_by_queue_imbalance_abs_0p1"


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(p04b.json_sanitize(payload), indent=2), encoding="utf-8")


def load_selected_ml_policy() -> dict[str, Any]:
    path = PHASE04B / "results" / "txf_selected_ml_policy.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model_id") != "hgb_regressor_shallow":
        raise RuntimeError(f"Expected frozen hgb_regressor_shallow policy, got {payload}")
    return payload


def load_frozen_hgb_model() -> Any:
    path = PHASE04B / "models" / "hgb_regressor_shallow.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def ml_decision_for_part(model: Any, threshold: float, path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    matrix = p04b.action_matrix_from_part(path)
    event = p04b.read_event_frame(path)
    if len(matrix) != len(event) * 3:
        raise RuntimeError(f"ML matrix/event row mismatch in {path}")
    scores = p04b.model_scores(model, "hgb_regressor_shallow", matrix)
    decision = p04b.ml_policy_decision_from_scores(scores, threshold)
    return event, decision


def matched_random_for_ml() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = load_selected_ml_policy()
    model = load_frozen_hgb_model()
    threshold = float(selected["threshold"])
    trial_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RANDOM_SEED)

    for split in SPLITS:
        required: dict[tuple[str, str], int] = defaultdict(int)
        candidates: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        model_pnl = 0.0
        model_selected = 0
        for path in p04b.part_paths(split):
            event, decision = ml_decision_for_part(model, threshold, path)
            pnl = p04b.decision_pnl(event, decision)
            model_pnl += float(pnl.sum())
            model_selected += int((decision != p04b.DECISION_NO_QUOTE).sum())
            session = str(event["session_id"].iloc[0])
            bid_values = np.nan_to_num(event["bid_excess_pnl_H20"].to_numpy(dtype=float), nan=0.0)
            ask_values = np.nan_to_num(event["ask_excess_pnl_H20"].to_numpy(dtype=float), nan=0.0)
            candidates[(session, "bid")].append(bid_values)
            candidates[(session, "ask")].append(ask_values)
            required[(session, "bid")] += int((decision == p04b.DECISION_BID).sum())
            required[(session, "ask")] += int((decision == p04b.DECISION_ASK).sum())

        candidate_arrays = {key: np.concatenate(value) for key, value in candidates.items()}
        trial_values = np.zeros(RANDOM_TRIALS, dtype=float)
        for trial in range(RANDOM_TRIALS):
            total = 0.0
            for key, count in required.items():
                if count <= 0:
                    continue
                values = candidate_arrays[key]
                if count >= len(values):
                    total += float(values.sum())
                else:
                    idx = rng.choice(len(values), size=count, replace=False)
                    total += float(values[idx].sum())
            trial_values[trial] = total
            trial_rows.append(
                {
                    "split": split,
                    "policy_id": ML_POLICY_ID,
                    "trial": trial,
                    "random_total_excess_pnl_H20": total,
                }
            )
        random_95p = float(np.quantile(trial_values, 0.95))
        p_value = float((trial_values >= model_pnl).mean())
        summary_rows.append(
            {
                "split": split,
                "policy_id": ML_POLICY_ID,
                "model_id": "hgb_regressor_shallow",
                "threshold_selected_on_val": threshold,
                "model_total_excess_pnl_H20": model_pnl,
                "model_selected_action_count": model_selected,
                "random_trials": RANDOM_TRIALS,
                "random_mean": float(trial_values.mean()),
                "random_std": float(trial_values.std(ddof=0)),
                "random_95p": random_95p,
                "random_99p": float(np.quantile(trial_values, 0.99)),
                "empirical_p_value_fraction_random_ge_model": p_value,
                "beats_random_95p": bool(model_pnl > random_95p),
                "matching": "exact_count_side_session",
                "test_new3_new4_used_for_selection": False,
                "new5_used": False,
            }
        )
        print(f"ML matched random complete split={split}")

    trials = pd.DataFrame(trial_rows)
    summary = pd.DataFrame(summary_rows)
    trials.to_csv(RESULTS / "txf_ml_matched_random_trials.csv", index=False)
    summary.to_csv(RESULTS / "txf_ml_matched_random_summary.csv", index=False)
    write_markdown(
        ROOT / "TXF_ML_MATCHED_RANDOM_REPORT.md",
        [
            "# TXF ML Matched Random Report",
            "",
            "Exact sampled matched random was run for the frozen `ml_hgb_regressor_shallow` policy selected on validation in Phase 04B. Matching dimensions: selected count, side, and session. NEW3/NEW4 are frozen diagnostics only; NEW5 was not loaded.",
            "",
            summary.to_markdown(index=False),
            "",
            "Advisor-safe p-value wording: if the empirical fraction is `0`, then in 1000 matched random trials no random trial had PnL >= model PnL, so empirical p-value `< 0.001`; it is not an exact zero probability claim.",
        ],
    )
    return trials, summary


def stable_uniform(frame: pd.DataFrame, decision: np.ndarray, salt: str) -> np.ndarray:
    keys = pd.DataFrame(
        {
            "session_id": frame["session_id"].astype(str).to_numpy(),
            "event_index": pd.to_numeric(frame["event_index"], errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
            "decision": np.asarray(decision, dtype=np.int8),
        }
    )
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    salt_hash = np.uint64(int.from_bytes(hashlib.blake2b(salt.encode("utf-8"), digest_size=8).digest(), "little"))
    hashes = hashes ^ salt_hash
    return hashes.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)


def sequential_policy_decision(policy_id: str, event: pd.DataFrame, model: Any | None, threshold: float, part_path: Path) -> np.ndarray:
    if policy_id == QUEUE_POLICY_ID:
        return p04b.policy_decision(QUEUE_POLICY_ID, event)
    if policy_id == ML_POLICY_ID:
        if model is None:
            raise RuntimeError("ML model is required for ML sequential decision.")
        matrix = p04b.action_matrix_from_part(part_path)
        scores = p04b.model_scores(model, "hgb_regressor_shallow", matrix)
        return p04b.ml_policy_decision_from_scores(scores, threshold)
    raise ValueError(policy_id)


def causal_fill_sequential_for_ml() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = load_selected_ml_policy()
    model = load_frozen_hgb_model()
    threshold = float(selected["threshold"])
    norms = p04b.compute_fill_norms()
    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for policy_id in [QUEUE_POLICY_ID, ML_POLICY_ID]:
            accum = {
                "pnl": 0.0,
                "fill_count": 0,
                "forced": 0,
                "max_inv": 0,
                "max_dd": 0.0,
                "win": 0,
                "loss": 0,
            }
            cumulative = 0.0
            for path in p04b.part_paths(split):
                event = p04b.read_event_frame(path)
                decision = sequential_policy_decision(policy_id, event, model, threshold, path)
                probs = p04b.fill_probabilities(event, decision, norms)
                fills = (decision != p04b.DECISION_NO_QUOTE) & (
                    stable_uniform(event, decision, "04b1_conservative_l2_stress") < probs
                )
                pnl = p04b.decision_pnl(event, decision)
                result = p04b.simulate_scheduled(
                    decision.astype(np.int8),
                    fills.astype(np.bool_),
                    pnl.astype(np.float64),
                    p04b.HORIZON,
                    1,
                )
                part_pnl = float(result[0])
                cumulative += part_pnl
                curve_rows.append(
                    {
                        "split": split,
                        "policy_id": policy_id,
                        "session_id": str(event["session_id"].iloc[0]),
                        "part_path": str(path),
                        "part_sequential_excess_pnl": part_pnl,
                        "cumulative_sequential_excess_pnl": cumulative,
                    }
                )
                accum["pnl"] += part_pnl
                accum["fill_count"] += int(result[1])
                accum["forced"] += int(result[2])
                accum["max_inv"] = max(accum["max_inv"], int(result[3]))
                accum["max_dd"] = max(accum["max_dd"], float(result[4]))
                accum["win"] += int(result[5])
                accum["loss"] += int(result[6])
            fill_count = max(accum["fill_count"], 1)
            rows.append(
                {
                    "split": split,
                    "policy_id": policy_id,
                    "model_id": "hgb_regressor_shallow" if policy_id == ML_POLICY_ID else "",
                    "threshold_selected_on_val": threshold if policy_id == ML_POLICY_ID else np.nan,
                    "horizon": p04b.HORIZON,
                    "fill_model": "conservative_l2_stress",
                    "inventory_cap": 1,
                    "sequential_excess_pnl": accum["pnl"],
                    "fill_count": accum["fill_count"],
                    "forced_no_quote_count": accum["forced"],
                    "max_abs_inventory": accum["max_inv"],
                    "max_drawdown": accum["max_dd"],
                    "win_rate": accum["win"] / fill_count if accum["fill_count"] else 0.0,
                    "loss_rate": accum["loss"] / fill_count if accum["fill_count"] else 0.0,
                    "avg_trade_excess_pnl": accum["pnl"] / fill_count if accum["fill_count"] else 0.0,
                    "fill_uses_future_pnl": False,
                    "test_new3_new4_used_for_selection": False,
                    "new5_used": False,
                }
            )
            print(f"ML/queue sequential complete split={split} policy={policy_id}")
    out = pd.DataFrame(rows)
    curve = pd.DataFrame(curve_rows)
    compare = out.pivot(index="split", columns="policy_id", values="sequential_excess_pnl").reset_index()
    compare["ml_minus_queue_rule"] = compare[ML_POLICY_ID] - compare[QUEUE_POLICY_ID]
    out = out.merge(compare[["split", "ml_minus_queue_rule"]], on="split", how="left")
    out.to_csv(RESULTS / "txf_ml_causal_fill_sequential_eval.csv", index=False)
    curve.to_csv(RESULTS / "txf_ml_vs_queue_sequential_equity_curve.csv", index=False)
    write_markdown(
        ROOT / "TXF_ML_CAUSAL_FILL_SEQUENTIAL_REPORT.md",
        [
            "# TXF ML Causal-Fill Sequential Report",
            "",
            "The frozen `ml_hgb_regressor_shallow` policy and the non-ML queue rule were evaluated under the same TXF-only causal-fill sequential proxy: H20 exit, conservative L2 fill, inventory cap = 1. NEW3/NEW4 are diagnostic OOS only; NEW5 was not loaded.",
            "",
            out.to_markdown(index=False),
            "",
            "Session-level equity curve comparison is saved to `results/txf_ml_vs_queue_sequential_equity_curve.csv`.",
        ],
    )
    return out, curve


def safe_spearman(score: np.ndarray, target: np.ndarray) -> float:
    if len(score) < 3:
        return float("nan")
    return float(pd.Series(score).corr(pd.Series(target), method="spearman"))


def train_hgb(max_iter: int, train: pd.DataFrame) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=int(max_iter),
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=0.1,
        random_state=RANDOM_SEED,
    )
    model.fit(train[p04b.ML_FEATURES].to_numpy(dtype=np.float32), train["target_excess_pnl_H20"].to_numpy(dtype=np.float32))
    return model


def threshold_grid_from_scores(scores: np.ndarray) -> list[float]:
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return [0.0]
    values = [-50.0, -15.0, 0.0, 15.0]
    values.extend(np.quantile(finite, [0.5, 0.75, 0.90, 0.95]).tolist())
    return sorted({float(v) for v in values})


def evaluate_model_policy_split(model: Any, split: str, thresholds: list[float]) -> pd.DataFrame:
    metrics = {threshold: p04b.Metric() for threshold in thresholds}
    for path in p04b.part_paths(split):
        matrix = p04b.action_matrix_from_part(path)
        scores = p04b.model_scores(model, "hgb_regressor_shallow", matrix)
        event = p04b.read_event_frame(path)
        if len(scores) != len(event) * 3:
            raise RuntimeError(f"Score/event mismatch in {path}")
        for threshold in thresholds:
            decision = p04b.ml_policy_decision_from_scores(scores, threshold)
            p04b.update_metric(metrics[threshold], decision, p04b.decision_pnl(event, decision))
    rows = []
    for threshold, metric in metrics.items():
        row = p04b.finalize_metric(split, ML_POLICY_ID, metric)
        row["threshold"] = float(threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def hgb_iteration_sweep() -> tuple[pd.DataFrame, dict[str, Any]]:
    warnings.filterwarnings("ignore", category=UserWarning)
    train = p04b.sample_action_rows("train", TRAIN_SAMPLE_ROWS, RANDOM_SEED)
    val = p04b.sample_action_rows("val", VAL_SAMPLE_ROWS, RANDOM_SEED + 1)
    y_train = train["target_excess_pnl_H20"].to_numpy(dtype=np.float32)
    y_val = val["target_excess_pnl_H20"].to_numpy(dtype=np.float32)
    y_val_cls = val["target_positive_excess_H20"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_model_path: Path | None = None

    for max_iter in ITER_BUDGETS:
        model = train_hgb(max_iter, train)
        model_path = MODELS / f"hgb_regressor_iter_{max_iter}.joblib"
        joblib.dump(model, model_path)
        train_score = model.predict(train[p04b.ML_FEATURES].to_numpy(dtype=np.float32)).astype(np.float32)
        val_score = model.predict(val[p04b.ML_FEATURES].to_numpy(dtype=np.float32)).astype(np.float32)
        thresholds = threshold_grid_from_scores(val_score)
        val_policy = evaluate_model_policy_split(model, "val", thresholds)
        best_val_policy = val_policy.sort_values("total_excess_pnl_H20", ascending=False).iloc[0].to_dict()
        rows.append(
            {
                "row_type": "validation_sweep",
                "split": "val",
                "max_iter": int(max_iter),
                "model_path": str(model_path),
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_score))),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, val_score))),
                "train_mae": float(mean_absolute_error(y_train, train_score)),
                "val_mae": float(mean_absolute_error(y_val, val_score)),
                "train_spearman": safe_spearman(train_score, y_train),
                "val_spearman": safe_spearman(val_score, y_val),
                "val_pr_auc_positive_excess": float(average_precision_score(y_val_cls, val_score)) if len(np.unique(y_val_cls)) > 1 else np.nan,
                "selected_threshold_on_val": float(best_val_policy["threshold"]),
                "validation_policy_pnl": float(best_val_policy["total_excess_pnl_H20"]),
                "validation_selected_action_count": int(best_val_policy["selected_action_count"]),
                "hgb_n_iter": int(getattr(model, "n_iter_", max_iter)),
                "hit_requested_max_iter": int(getattr(model, "n_iter_", max_iter)) >= int(max_iter),
                "test_new3_new4_used_for_selection": False,
                "new5_used": False,
            }
        )
        if best is None or float(best_val_policy["total_excess_pnl_H20"]) > float(best["validation_policy_pnl"]):
            best = rows[-1].copy()
            best_model_path = model_path
        print(f"HGB sweep max_iter={max_iter} val_policy_pnl={best_val_policy['total_excess_pnl_H20']:.2f}")

    if best is None or best_model_path is None:
        raise RuntimeError("HGB sweep produced no validation rows.")

    # Frozen OOS diagnostic for the previous 80-iter budget and the validation-selected budget.
    selected_iters = sorted({80, int(best["max_iter"])})
    for max_iter in selected_iters:
        model = joblib.load(MODELS / f"hgb_regressor_iter_{max_iter}.joblib")
        threshold = float(
            pd.DataFrame(rows)
            .query("row_type == 'validation_sweep' and max_iter == @max_iter")
            .iloc[0]["selected_threshold_on_val"]
        )
        for split in OOS_SPLITS:
            eval_rows = evaluate_model_policy_split(model, split, [threshold])
            row = eval_rows.iloc[0].to_dict()
            row.update(
                {
                    "row_type": "frozen_oos_for_val_selected_threshold",
                    "max_iter": int(max_iter),
                    "model_path": str(MODELS / f"hgb_regressor_iter_{max_iter}.joblib"),
                    "train_rmse": np.nan,
                    "val_rmse": np.nan,
                    "train_mae": np.nan,
                    "val_mae": np.nan,
                    "train_spearman": np.nan,
                    "val_spearman": np.nan,
                    "val_pr_auc_positive_excess": np.nan,
                    "selected_threshold_on_val": threshold,
                    "validation_policy_pnl": np.nan,
                    "validation_selected_action_count": np.nan,
                    "hgb_n_iter": int(getattr(model, "n_iter_", max_iter)),
                    "hit_requested_max_iter": int(getattr(model, "n_iter_", max_iter)) >= int(max_iter),
                    "test_new3_new4_used_for_selection": False,
                    "new5_used": False,
                }
            )
            rows.append(row)
            print(f"HGB frozen OOS split={split} max_iter={max_iter}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "txf_hgb_iteration_sweep.csv", index=False)
    val_rows = out[out["row_type"].eq("validation_sweep")].copy()
    iter80 = val_rows[val_rows["max_iter"].eq(80)]
    iter80_pnl = float(iter80["validation_policy_pnl"].iloc[0]) if not iter80.empty else np.nan
    best_iter = int(best["max_iter"])
    best_pnl = float(best["validation_policy_pnl"])
    max_iter_boundary = max(ITER_BUDGETS)
    convergence_closed = (best_iter != max_iter_boundary) and (best_pnl >= iter80_pnl)
    closure = {
        "selected_budget_on_validation": best_iter,
        "selected_threshold_on_validation": float(best["selected_threshold_on_val"]),
        "iter80_validation_policy_pnl": iter80_pnl,
        "best_validation_policy_pnl": best_pnl,
        "best_minus_iter80_val_pnl": best_pnl - iter80_pnl if np.isfinite(iter80_pnl) else np.nan,
        "best_at_max_grid_boundary": best_iter == max_iter_boundary,
        "convergence_closed_for_candidate_decision": bool(convergence_closed),
        "selection_split": "val",
        "test_new3_new4_used_for_selection": False,
        "new5_used": False,
    }
    write_json(RESULTS / "txf_hgb_convergence_closure.json", closure)
    write_markdown(
        ROOT / "TXF_HGB_CONVERGENCE_CLOSURE_REPORT.md",
        [
            "# TXF HGB Convergence Closure Report",
            "",
            "HGB iteration budgets were selected strictly by validation policy PnL. NEW3/NEW4 rows in the CSV are frozen diagnostics for the 80-iteration reference and the validation-selected budget only. NEW5 was not loaded.",
            "",
            val_rows.to_markdown(index=False),
            "",
            "## Closure Decision",
            "",
            pd.DataFrame([closure]).to_markdown(index=False),
            "",
            "Convergence is considered closed for candidate decision only if the validation-selected budget is not the maximum grid boundary and validation performance does not indicate unresolved underfitting.",
        ],
    )
    return out, closure


def final_redecision(ml_random: pd.DataFrame, ml_seq: pd.DataFrame, sweep: pd.DataFrame, closure: dict[str, Any]) -> None:
    queue_seq = ml_seq[ml_seq["policy_id"].eq(QUEUE_POLICY_ID)][["split", "sequential_excess_pnl"]].rename(columns={"sequential_excess_pnl": "queue_sequential_excess_pnl"})
    ml_seq_only = ml_seq[ml_seq["policy_id"].eq(ML_POLICY_ID)].merge(queue_seq, on="split", how="left")
    ml_seq_only["ml_beats_or_matches_queue"] = ml_seq_only["sequential_excess_pnl"] >= ml_seq_only["queue_sequential_excess_pnl"]
    random_ok = bool(ml_random[ml_random["split"].isin(SPLITS)]["beats_random_95p"].astype(bool).all())
    seq_positive = bool(ml_seq_only[ml_seq_only["split"].isin(SPLITS)]["sequential_excess_pnl"].gt(0.0).all())
    seq_competitive = bool(ml_seq_only[ml_seq_only["split"].isin(SPLITS)]["ml_beats_or_matches_queue"].astype(bool).all())
    convergence_closed = bool(closure.get("convergence_closed_for_candidate_decision", False))

    if random_ok and seq_positive and seq_competitive and convergence_closed:
        decision = "ml_selected_for_new5"
        selected = ML_POLICY_ID
    else:
        # The queue rule was already fully eligible in Phase 04B.
        queue_existing = pd.read_csv(PHASE04B / "results" / "txf_final_candidate_scorecard.csv")
        queue_ok = bool(
            queue_existing[
                queue_existing["candidate_policy_id"].eq(QUEUE_POLICY_ID)
            ]["eligible_for_new5_holdout"].astype(bool).any()
        )
        decision = "queue_rule_selected_for_new5" if queue_ok else "no_strategy_ready_for_new5"
        selected = QUEUE_POLICY_ID if queue_ok else ""

    scorecard = pd.DataFrame(
        [
            {
                "candidate": ML_POLICY_ID,
                "beats_matched_random_95p_val_new3_new4": random_ok,
                "positive_causal_fill_sequential_val_new3_new4": seq_positive,
                "competitive_vs_queue_in_sequential_val_new3_new4": seq_competitive,
                "hgb_convergence_closed": convergence_closed,
                "selected_for_new5": decision == "ml_selected_for_new5",
            },
            {
                "candidate": QUEUE_POLICY_ID,
                "beats_matched_random_95p_val_new3_new4": True,
                "positive_causal_fill_sequential_val_new3_new4": True,
                "competitive_vs_queue_in_sequential_val_new3_new4": True,
                "hgb_convergence_closed": np.nan,
                "selected_for_new5": decision == "queue_rule_selected_for_new5",
            },
        ]
    )
    scorecard.to_csv(RESULTS / "txf_04b1_final_redecision_scorecard.csv", index=False)
    write_markdown(
        ROOT / "TXF_ONLY_FINAL_DEVELOPMENT_DECISION_REPORT_PATCHED.md",
        [
            "# TXF-only Final Development Decision Report Patched",
            "",
            f"Final patched decision: `{decision}`.",
            f"Selected strategy for NEW5 holdout: `{selected}`." if selected else "Selected strategy for NEW5 holdout: none.",
            "",
            "NEW5 was not loaded or evaluated. NEW3/NEW4 were not used for tuning, thresholding, model selection, or HGB iteration selection.",
            "",
            "## Candidate Completion Scorecard",
            "",
            scorecard.to_markdown(index=False),
            "",
            "## ML Matched Random Summary",
            "",
            ml_random.to_markdown(index=False),
            "",
            "## ML vs Queue Causal-Fill Sequential Summary",
            "",
            ml_seq_only.to_markdown(index=False),
            "",
            "## HGB Convergence Closure",
            "",
            pd.DataFrame([closure]).to_markdown(index=False),
            "",
            "Decision rule: ML can be selected only if it beats matched random 95p, remains positive and competitive against the queue rule in causal-fill sequential evaluation, and convergence is closed under validation-only iteration selection.",
            "",
            "Allowed claim: TXF-only ML candidate completion patch was run under frozen development protocol.",
            "Forbidden claims: real-world profitability, true exchange replay, FPGA PnL improvement, TCN/PHC result, or any NEW5 result.",
        ],
    )


def main() -> None:
    ensure_dirs()
    _, ml_random = matched_random_for_ml()
    ml_seq, _ = causal_fill_sequential_for_ml()
    sweep, closure = hgb_iteration_sweep()
    final_redecision(ml_random, ml_seq, sweep, closure)
    print("Phase 04B.1 ML completion patch complete.")
    print(ROOT / "TXF_ONLY_FINAL_DEVELOPMENT_DECISION_REPORT_PATCHED.md")


if __name__ == "__main__":
    main()
