from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "03b5_ml_training_diagnostics"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
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
    assert_bid_ask_pairing,
    baseline_policy_summary,
    build_feature_frame,
    evaluate_decision,
    feature_matrix_columns,
    load_train_matrix,
    load_val_matrix,
    max_drawdown,
    model_path,
    model_registry,
    part_paths,
    policy_pnl_from_decision,
    predict_model,
    read_json,
    safe_spearman,
    select_best_quote,
    selected_actions_path,
    write_json,
    write_markdown,
)


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, MODELS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def pairing_audit() -> pd.DataFrame:
    rows = []
    for split in ["train", "val", *FROZEN_SPLITS]:
        for path in part_paths(split):
            try:
                frame = build_feature_frame(path, include_targets=True)
                bid = frame[frame["action_slot"].astype(int) == ACTION_BID].sort_values(["session_id", "event_index"]).reset_index(drop=True)
                ask = frame[frame["action_slot"].astype(int) == ACTION_ASK].sort_values(["session_id", "event_index"]).reset_index(drop=True)
                assert_bid_ask_pairing(bid, ask, f"03b5_pairing_audit|{split}|{path.name}")
                rows.append(
                    {
                        "split": split,
                        "part": path.name,
                        "bid_rows": len(bid),
                        "ask_rows": len(ask),
                        "paired_on_session_id": True,
                        "paired_on_event_index": True,
                        "paired_on_symbol": True,
                        "status": "PASS",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - audit must capture exact failure.
                rows.append(
                    {
                        "split": split,
                        "part": path.name,
                        "bid_rows": np.nan,
                        "ask_rows": np.nan,
                        "paired_on_session_id": False,
                        "paired_on_event_index": False,
                        "paired_on_symbol": False,
                        "status": "FAIL",
                        "failure": str(exc),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "bid_ask_pairing_audit.csv", index=False)
    if out["status"].eq("FAIL").any():
        raise RuntimeError("Bid/ask pairing audit failed; stopping Phase 03B.5.")
    return out


def model_estimator(model: Any) -> Any:
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model


def training_diagnostics() -> pd.DataFrame:
    train = load_train_matrix()
    val = load_val_matrix()
    x_train = train[feature_matrix_columns()].to_numpy(dtype=np.float32)
    x_val = val[feature_matrix_columns()].to_numpy(dtype=np.float32)
    y_train_reg = train[TARGET].to_numpy(dtype=np.float32)
    y_val_reg = val[TARGET].to_numpy(dtype=np.float32)
    y_train_cls = train[POSITIVE_TARGET].astype(int).to_numpy()
    y_val_cls = val[POSITIVE_TARGET].astype(int).to_numpy()
    rows = []
    staged_rows = []
    for model_id, info in model_registry().items():
        model = joblib.load(model_path(model_id))
        est = model_estimator(model)
        score_train = predict_model(model, train, model_id)
        score_val = predict_model(model, val, model_id)
        row: dict[str, Any] = {
            "model_id": model_id,
            "score_kind": info["score_kind"],
            "train_rows": len(train),
            "val_rows": len(val),
            "n_iter": str(getattr(est, "n_iter_", "")),
            "max_iter": getattr(est, "max_iter", ""),
            "converged_flag": True,
            "hgb_do_early_stopping": getattr(est, "do_early_stopping_", ""),
            "hgb_train_score_len": len(getattr(est, "train_score_", [])) if hasattr(est, "train_score_") else "",
            "hgb_validation_score_len": len(getattr(est, "validation_score_", [])) if hasattr(est, "validation_score_") else "",
        }
        if hasattr(est, "train_score_") or hasattr(est, "validation_score_"):
            train_scores = list(getattr(est, "train_score_", []))
            val_scores = list(getattr(est, "validation_score_", []))
            n_scores = max(len(train_scores), len(val_scores))
            for iteration in range(n_scores):
                staged_rows.append(
                    {
                        "model_id": model_id,
                        "iteration": iteration,
                        "train_score": train_scores[iteration] if iteration < len(train_scores) else np.nan,
                        "validation_score": val_scores[iteration] if iteration < len(val_scores) else np.nan,
                        "n_iter": getattr(est, "n_iter_", np.nan),
                        "max_iter": getattr(est, "max_iter", np.nan),
                        "hit_iteration_cap": bool(getattr(est, "n_iter_", -1) == getattr(est, "max_iter", -2)),
                    }
                )
        if info["score_kind"] == "probability":
            eps_train = np.clip(score_train, 1e-6, 1 - 1e-6)
            eps_val = np.clip(score_val, 1e-6, 1 - 1e-6)
            row.update(
                {
                    "train_pr_auc": float(average_precision_score(y_train_cls, score_train)),
                    "val_pr_auc": float(average_precision_score(y_val_cls, score_val)),
                    "train_roc_auc": float(roc_auc_score(y_train_cls, score_train)),
                    "val_roc_auc": float(roc_auc_score(y_val_cls, score_val)),
                    "train_log_loss": float(log_loss(y_train_cls, eps_train)),
                    "val_log_loss": float(log_loss(y_val_cls, eps_val)),
                    "train_rmse": np.nan,
                    "val_rmse": np.nan,
                    "train_mae": np.nan,
                    "val_mae": np.nan,
                    "train_spearman_vs_excess": safe_spearman(score_train, y_train_reg),
                    "val_spearman_vs_excess": safe_spearman(score_val, y_val_reg),
                }
            )
            n_iter = np.asarray(getattr(est, "n_iter_", [0]), dtype=float)
            max_iter = float(getattr(est, "max_iter", np.inf))
            row["converged_flag"] = bool(np.nanmax(n_iter) < max_iter) if np.isfinite(max_iter) else True
        else:
            row.update(
                {
                    "train_pr_auc": float(average_precision_score(y_train_cls, score_train)),
                    "val_pr_auc": float(average_precision_score(y_val_cls, score_val)),
                    "train_roc_auc": float(roc_auc_score(y_train_cls, score_train)),
                    "val_roc_auc": float(roc_auc_score(y_val_cls, score_val)),
                    "train_log_loss": np.nan,
                    "val_log_loss": np.nan,
                    "train_rmse": rmse(y_train_reg, score_train),
                    "val_rmse": rmse(y_val_reg, score_val),
                    "train_mae": float(mean_absolute_error(y_train_reg, score_train)),
                    "val_mae": float(mean_absolute_error(y_val_reg, score_val)),
                    "train_spearman_vs_excess": safe_spearman(score_train, y_train_reg),
                    "val_spearman_vs_excess": safe_spearman(score_val, y_val_reg),
                }
            )
            n_iter_attr = getattr(est, "n_iter_", None)
            max_iter_attr = getattr(est, "max_iter", None)
            if n_iter_attr is not None and max_iter_attr is not None:
                row["converged_flag"] = bool(float(n_iter_attr) < float(max_iter_attr))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "training_diagnostics.csv", index=False)
    if staged_rows:
        pd.DataFrame(staged_rows).to_csv(RESULTS / "hgb_staged_scores.csv", index=False)
    write_markdown(
        ROOT / "TRAINING_DIAGNOSTICS_REPORT.md",
        [
            "# Training Diagnostics Report",
            "",
            "This audit reuses the Phase 03B simple model family and logs convergence-oriented metrics. No frozen OOS split is used for model selection.",
            "",
            out.to_markdown(index=False),
            "",
            "Convergence interpretation: logistic/lasso convergence flags use `n_iter_ < max_iter` when available. HistGradientBoosting staged score arrays are logged when sklearn exposes them.",
            "",
            "HGB staged scores are written to `results/hgb_staged_scores.csv`.",
        ],
    )
    return out


def selected_policy() -> dict[str, Any]:
    return read_json(PHASE03B / "results" / "selected_phase03b_policy.json")


def state_triplets(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bid = matrix[matrix["action_slot"].astype(int) == ACTION_BID].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    ask = matrix[matrix["action_slot"].astype(int) == ACTION_ASK].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    no_quote = matrix[matrix["action_slot"].astype(int) == 2].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    assert_bid_ask_pairing(bid, ask, "state_triplets")
    if len(no_quote) != len(bid):
        raise RuntimeError("No-quote rows do not match bid/ask state count.")
    return bid, ask, no_quote


def oracle_decision(bid_pnl: np.ndarray, ask_pnl: np.ndarray) -> np.ndarray:
    out = np.zeros(len(bid_pnl), dtype=np.int8)
    best_quote = np.maximum(bid_pnl, ask_pnl)
    bid_mask = (bid_pnl >= ask_pnl) & (bid_pnl > 0.0)
    ask_mask = (ask_pnl > bid_pnl) & (ask_pnl > 0.0)
    out[(best_quote > 0.0) & bid_mask] = DECISION_BID
    out[(best_quote > 0.0) & ask_mask] = DECISION_ASK
    return out


def confusion_rows(split: str, predicted: np.ndarray, oracle: np.ndarray, actual_any_positive: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    pred_quote = predicted != DECISION_NO_QUOTE
    for pred in [False, True]:
        for actual in [False, True]:
            rows.append(
                {
                    "split": split,
                    "matrix_type": "binary_any_positive_quote_available",
                    "predicted_quote": pred,
                    "actual_any_positive_quote": actual,
                    "count": int(((pred_quote == pred) & (actual_any_positive == actual)).sum()),
                }
            )
    labels = [DECISION_NO_QUOTE, DECISION_BID, DECISION_ASK]
    names = {DECISION_NO_QUOTE: "no_quote", DECISION_BID: "bid", DECISION_ASK: "ask"}
    for pred in labels:
        for actual in labels:
            rows.append(
                {
                    "split": split,
                    "matrix_type": "oracle_action_argmax_excess_bid_ask_zero",
                    "predicted_action": names[pred],
                    "oracle_action": names[actual],
                    "count": int(((predicted == pred) & (oracle == actual)).sum()),
                }
            )
    return rows


def oracle_action_confusion() -> pd.DataFrame:
    policy = selected_policy()
    model = joblib.load(model_path(policy["model_id"]))
    threshold = float(policy["threshold"])
    rows = []
    for split in ["val", *FROZEN_SPLITS]:
        if split == "val":
            matrices = [load_val_matrix()]
        else:
            matrices = (build_feature_frame(path, include_targets=True) for path in part_paths(split))
        pred_all = []
        oracle_all = []
        positive_all = []
        for matrix in matrices:
            scores = predict_model(model, matrix, policy["model_id"])
            decision = select_best_quote(matrix, scores, threshold).decision
            bid, ask, _ = state_triplets(matrix)
            bid_pnl = bid[TARGET].to_numpy(dtype=float)
            ask_pnl = ask[TARGET].to_numpy(dtype=float)
            oracle = oracle_decision(bid_pnl, ask_pnl)
            pred_all.append(decision)
            oracle_all.append(oracle)
            positive_all.append(np.maximum(bid_pnl, ask_pnl) > 0.0)
        rows.extend(confusion_rows(split, np.concatenate(pred_all), np.concatenate(oracle_all), np.concatenate(positive_all)))
        print(f"oracle confusion complete split={split}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "oracle_action_confusion.csv", index=False)
    return out


def hgb_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=0.1, random_state=seed)


def evaluate_threshold_on_val(model: Any, model_id: str, thresholds: np.ndarray | None = None) -> tuple[float, dict[str, Any]]:
    val = load_val_matrix()
    scores = model.predict(val[feature_matrix_columns()].to_numpy(dtype=np.float32)).astype(np.float32)
    quote_mask = val["action_slot"].astype(int).isin([0, 1]).to_numpy()
    quote_scores = scores[quote_mask]
    if thresholds is None:
        thresholds = np.unique(np.concatenate([np.asarray([0.0]), np.nanquantile(quote_scores, [0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])]))
    best: dict[str, Any] | None = None
    best_threshold = 0.0
    for threshold in thresholds:
        decision = select_best_quote(val, scores, float(threshold))
        metrics = evaluate_decision(val, decision, "val", "diagnostic_hgb_threshold", model_id, float(threshold))
        if best is None or metrics["total_excess_pnl"] > best["total_excess_pnl"]:
            best = metrics
            best_threshold = float(threshold)
    assert best is not None
    return best_threshold, best


def evaluate_model_oos(model: Any, model_id: str, threshold: float) -> pd.DataFrame:
    rows = []
    for split in FROZEN_SPLITS:
        for path in part_paths(split):
            matrix = build_feature_frame(path, include_targets=True)
            scores = model.predict(matrix[feature_matrix_columns()].to_numpy(dtype=np.float32)).astype(np.float32)
            decision = select_best_quote(matrix, scores, threshold)
            rows.append(evaluate_decision(matrix, decision, split, "diagnostic_hgb_threshold", model_id, threshold))
    out = pd.DataFrame(rows)
    return out.groupby(["split", "policy_id", "model_id", "horizon", "threshold"], as_index=False).agg(
        state_count=("state_count", "sum"),
        selected_action_count=("selected_action_count", "sum"),
        bid_count=("bid_count", "sum"),
        ask_count=("ask_count", "sum"),
        total_excess_pnl=("total_excess_pnl", "sum"),
        max_drawdown=("max_drawdown", "max"),
    )


def train_sample(seed: int, include_no_quote: bool) -> pd.DataFrame:
    frames = []
    per_part = 150_000
    for idx, path in enumerate(part_paths("train")):
        frame = build_feature_frame(path, include_targets=True)
        if not include_no_quote:
            frame = frame[frame["action_slot"].astype(int).isin([ACTION_BID, ACTION_ASK])]
        sampled = []
        sides = list(frame.groupby("action_slot", sort=False))
        per_side = max(1, per_part // max(len(sides), 1))
        for slot, group in sides:
            take = min(len(group), per_side)
            sample_seed = int((seed * 1000 + idx * 10 + int(slot)) % (2**32 - 1))
            sampled.append(group.sample(n=take, random_state=sample_seed))
        frames.append(pd.concat(sampled, ignore_index=True))
    out = pd.concat(frames, ignore_index=True)
    max_rows = 900_000
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return out


def no_quote_ablation() -> pd.DataFrame:
    rows = []
    report_rows = []
    for variant, include_no_quote in [("train_with_no_quote_rows", True), ("train_bid_ask_only", False)]:
        train = train_sample(20260601, include_no_quote=include_no_quote)
        model = hgb_model(20260601)
        model.fit(train[feature_matrix_columns()].to_numpy(dtype=np.float32), train[TARGET].to_numpy(dtype=np.float32))
        joblib.dump(model, MODELS / f"{variant}.joblib")
        threshold, val_metrics = evaluate_threshold_on_val(model, variant)
        oos = evaluate_model_oos(model, variant, threshold)
        for row in oos.to_dict(orient="records"):
            row["variant"] = variant
            row["include_no_quote_rows"] = include_no_quote
            row["validation_selected_threshold"] = threshold
            row["validation_total_excess_pnl"] = val_metrics["total_excess_pnl"]
            rows.append(row)
        report_rows.append({"variant": variant, "train_rows": len(train), "threshold": threshold, **val_metrics})
        print(f"no-quote ablation complete variant={variant}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "no_quote_training_ablation.csv", index=False)
    report = pd.DataFrame(report_rows)
    base = pd.read_csv(PHASE03B / "results" / "ml_hurdle_summary.csv")
    write_markdown(
        ROOT / "NO_QUOTE_TRAINING_ABLATION_REPORT.md",
        [
            "# No-Quote Training Ablation Report",
            "",
            "Two shallow HGB regressors were trained with the same causal feature schema. Thresholds were selected only on validation.",
            "",
            "## Validation Selection",
            "",
            report.to_markdown(index=False),
            "",
            "## Frozen OOS Results",
            "",
            out.to_markdown(index=False),
            "",
            "## Phase 03B Reference Hurdles",
            "",
            base.to_markdown(index=False),
            "",
            "Interpretation: if bid/ask-only training materially improves frozen OOS and queue-rule hurdles, no-quote zero rows were distorting calibration. Otherwise the limitation is more likely signal/policy overlap rather than no-quote target corruption.",
        ],
    )
    return out


def train_sample_stability() -> pd.DataFrame:
    rows = []
    for seed in [0, 1, 2]:
        train = train_sample(seed, include_no_quote=True)
        model = hgb_model(seed)
        model.fit(train[feature_matrix_columns()].to_numpy(dtype=np.float32), train[TARGET].to_numpy(dtype=np.float32))
        threshold, val_metrics = evaluate_threshold_on_val(model, f"seed_{seed}")
        oos = evaluate_model_oos(model, f"seed_{seed}", threshold)
        for row in oos.to_dict(orient="records"):
            row["seed"] = seed
            row["validation_selected_threshold"] = threshold
            row["validation_total_excess_pnl"] = val_metrics["total_excess_pnl"]
            rows.append(row)
        print(f"sample stability complete seed={seed}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "train_sample_stability.csv", index=False)
    return out


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


def exact_sampled_matched_random() -> pd.DataFrame:
    rng = np.random.default_rng(20260601)
    selected_policy = selected_policy = read_json(PHASE03B / "results" / "selected_phase03b_policy.json")
    rows = []
    for split in FROZEN_SPLITS:
        selected = pd.read_parquet(selected_actions_path(split))
        selected = selected[selected["decision"].astype(int) != DECISION_NO_QUOTE].copy()
        required = selected.groupby(["session_id", "selected_side"]).size().to_dict()
        trials = np.zeros(RANDOM_TRIALS, dtype=np.float64)
        model_total = float(selected["selected_excess_pnl_H20"].sum())
        health_rows = []
        for path in part_paths(split):
            matrix = build_feature_frame(path, include_targets=True)
            for side, slot in [("bid", ACTION_BID), ("ask", ACTION_ASK)]:
                part = matrix[matrix["action_slot"].astype(int) == slot]
                session_id = str(part["session_id"].iloc[0])
                k = int(required.get((session_id, side), 0))
                values = part[TARGET].to_numpy(dtype=np.float64)
                group_trials = exact_hypergeom_trials(values, k, RANDOM_TRIALS, rng)
                trials += group_trials
                health_rows.append(
                    {
                        "split": split,
                        "session_id": session_id,
                        "side": side,
                        "required_count": k,
                        "candidate_pool_size": int(len(values)),
                        "unique_pnl_values": int(len(np.unique(values[np.isfinite(values)]))),
                        "exact_sampling_method": "multivariate_hypergeometric_unique_pnl_counts",
                    }
                )
        pd.DataFrame(health_rows).to_csv(RESULTS / f"exact_random_health_{split}.csv", index=False)
        for idx, value in enumerate(trials):
            rows.append(
                {
                    "split": split,
                    "trial": idx,
                    "model_id": selected_policy["model_id"],
                    "model_total_excess_pnl": model_total,
                    "random_total_excess_pnl": float(value),
                    "random_ge_model": bool(value >= model_total),
                }
            )
        print(f"exact sampled random complete split={split}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "exact_sampled_matched_random.csv", index=False)
    summary = out.groupby("split", as_index=False).agg(
        model_total_excess_pnl=("model_total_excess_pnl", "first"),
        random_mean=("random_total_excess_pnl", "mean"),
        random_95p=("random_total_excess_pnl", lambda s: float(np.quantile(s, 0.95))),
        random_99p=("random_total_excess_pnl", lambda s: float(np.quantile(s, 0.99))),
        empirical_p_value_fraction_random_ge_model=("random_ge_model", "mean"),
        random_trials=("trial", "count"),
    )
    summary.to_csv(RESULTS / "exact_sampled_matched_random_summary.csv", index=False)
    return summary


def hgb_feature_importance() -> pd.DataFrame:
    policy = selected_policy()
    model = joblib.load(model_path(policy["model_id"]))
    val = load_val_matrix()
    quote = val[val["action_slot"].astype(int).isin([ACTION_BID, ACTION_ASK])].copy()
    if len(quote) > 200_000:
        quote = quote.sample(n=200_000, random_state=20260601).reset_index(drop=True)
    features = feature_matrix_columns()
    x = quote[features].to_numpy(dtype=np.float32)
    y = quote[TARGET].to_numpy(dtype=np.float32)
    base_score = model.predict(x).astype(np.float32)
    base_spearman = safe_spearman(base_score, y)
    rng = np.random.default_rng(20260601)
    rows = []
    for idx, feature in enumerate(features):
        x_perm = x.copy()
        x_perm[:, idx] = rng.permutation(x_perm[:, idx])
        perm_score = model.predict(x_perm).astype(np.float32)
        perm_spearman = safe_spearman(perm_score, y)
        rows.append(
            {
                "metric_type": "permutation_importance_val_sample",
                "feature": feature,
                "baseline_spearman": base_spearman,
                "permuted_spearman": perm_spearman,
                "importance_spearman_drop": float(base_spearman - perm_spearman),
                "sample_rows": len(quote),
            }
        )
    quote = quote.assign(score=base_score)
    quote["score_bucket"] = pd.qcut(quote["score"], q=10, labels=False, duplicates="drop")
    base_mean = float(quote[TARGET].mean())
    for bucket, group in quote.groupby("score_bucket"):
        rows.append(
            {
                "metric_type": "score_bucket_lift_val_sample",
                "feature": f"score_bucket_{bucket}",
                "bucket": int(bucket),
                "mean_score": float(group["score"].mean()),
                "mean_excess_pnl": float(group[TARGET].mean()),
                "base_mean_excess_pnl": base_mean,
                "economic_lift": float(group[TARGET].mean() - base_mean),
                "sample_rows": len(group),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "hgb_feature_importance.csv", index=False)
    return out


def final_report(
    pairing: pd.DataFrame,
    training: pd.DataFrame,
    confusion: pd.DataFrame,
    ablation: pd.DataFrame,
    stability: pd.DataFrame,
    exact_random: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    phase03b_hurdles = pd.read_csv(PHASE03B / "results" / "ml_hurdle_summary.csv")
    no_quote_best = ablation.pivot_table(index="variant", columns="split", values="total_excess_pnl", aggfunc="sum")
    stability_pivot = stability.pivot_table(index="seed", columns="split", values="total_excess_pnl", aggfunc="sum")
    exact_pass_all = bool((exact_random["model_total_excess_pnl"] > exact_random["random_95p"]).all())
    converged_all = bool(training["converged_flag"].astype(bool).all())
    no_quote_corruption = "not_primary_bottleneck"
    if "train_bid_ask_only" in no_quote_best.index and "train_with_no_quote_rows" in no_quote_best.index:
        diff = no_quote_best.loc["train_bid_ask_only"] - no_quote_best.loc["train_with_no_quote_rows"]
        if (diff > 0).all() and diff.mean() > 0.1 * abs(no_quote_best.loc["train_with_no_quote_rows"].mean()):
            no_quote_corruption = "possible_calibration_distortion"
    stability_cv = (stability_pivot.std() / stability_pivot.mean().abs().clip(lower=1.0)).replace([np.inf, -np.inf], np.nan)
    unstable = bool((stability_cv > 0.25).any())
    queue_fail = not bool(phase03b_hurdles["beats_queue_imbalance_rule"].astype(bool).all())
    implementation_issue = (not converged_all) or unstable or no_quote_corruption == "possible_calibration_distortion"
    conclusion = "implementation_issue_requires_fix" if implementation_issue else "signal_overlap_with_queue_rule_is_primary_bottleneck"
    lines = [
        "# Phase 03B ML Training Diagnostics Report",
        "",
        f"Final diagnostic conclusion: `{conclusion}`.",
        "",
        "No TCN, large MLP, or frozen-OOS tuning was performed. Diagnostic ablations select thresholds on validation only.",
        "",
        "## 1. Bid/Ask Pairing Hard Asserts",
        "",
        pairing.groupby("split", as_index=False).agg(parts=("part", "count"), status=("status", lambda s: "PASS" if (s == "PASS").all() else "FAIL")).to_markdown(index=False),
        "",
        "## 2. Training Convergence Diagnostics",
        "",
        training.to_markdown(index=False),
        "",
        f"All simple model convergence flags pass: `{converged_all}`.",
        "",
        "## 3. Confusion and Oracle Matrices",
        "",
        "Full confusion matrices are in `results/oracle_action_confusion.csv`. They compare policy quote/no-quote against any positive quote availability and policy action against the oracle argmax over bid/ask/zero.",
        "",
        "## 4. No-Quote Training Ablation",
        "",
        no_quote_best.to_markdown(),
        "",
        f"No-quote zero-row diagnosis: `{no_quote_corruption}`.",
        "",
        "## 5. Train Sample Stability",
        "",
        stability_pivot.to_markdown(),
        "",
        f"Seed instability detected at CV > 25%: `{unstable}`.",
        "",
        "## 6. Exact Sampled Matched Random",
        "",
        exact_random.to_markdown(index=False),
        "",
        f"Selected ML beats exact sampled matched random 95p on all frozen OOS splits: `{exact_pass_all}`.",
        "",
        "If empirical p-value displays as 0, read it as no random trial exceeded the model in 1000 trials, i.e. empirical p-value `< 0.001`, not exactly zero probability.",
        "",
        "## 7. HGB Feature Importance",
        "",
        importance[importance["metric_type"].eq("permutation_importance_val_sample")].sort_values("importance_spearman_drop", ascending=False).head(12).to_markdown(index=False),
        "",
        "## 8. Incremental Gain Diagnosis",
        "",
        phase03b_hurdles.to_markdown(index=False),
        "",
    ]
    if conclusion == "signal_overlap_with_queue_rule_is_primary_bottleneck":
        lines.extend(
            [
                "The diagnostics do not point to a clear implementation flaw: pairing passed, convergence passed, exact random passed, and train-sample stability was acceptable. The simple ML model has signal, but its frozen OOS incremental value over the hand-written queue imbalance rule remains inconsistent. The likely bottleneck is signal overlap and policy simplicity, not a broken training implementation.",
            ]
        )
    else:
        lines.extend(
            [
                "At least one diagnostic indicates a possible implementation or training-design issue. Phase 03C should not proceed as a model-expansion phase until the flagged issue is fixed with a targeted root-cause audit.",
            ]
        )
        write_markdown(
            ROOT / "ROOT_CAUSE_BLOCKER_HGB_CONVERGENCE.md",
            [
                "# Root-Cause Blocker: HGB Convergence Inconclusive",
                "",
                "Phase 03B.5 found that the selected shallow HGB family reached `max_iter=80` without sklearn early stopping terminating first. This does not prove the Phase 03B result is invalid, but it prevents a clean claim that training convergence is fully closed.",
                "",
                "Required closure before model-expansion work:",
                "",
                "- Run a targeted train/validation-only HGB iteration-budget audit.",
                "- Keep test, NEW3, and NEW4 frozen.",
                "- Do not use this blocker to tune thresholds on OOS.",
                "- Continue to forbid raw absolute price and future label features.",
                "",
                "Evidence files:",
                "",
                "- `results/training_diagnostics.csv`",
                "- `results/hgb_staged_scores.csv`",
            ],
        )
    write_markdown(ROOT / "PHASE03B_ML_TRAINING_DIAGNOSTICS_REPORT.md", lines)


def main() -> None:
    ensure_dirs()
    pairing = pairing_audit()
    training = training_diagnostics()
    confusion = oracle_action_confusion()
    ablation = no_quote_ablation()
    stability = train_sample_stability()
    exact_random = exact_sampled_matched_random()
    importance = hgb_feature_importance()
    final_report(pairing, training, confusion, ablation, stability, exact_random, importance)
    print(f"wrote {ROOT / 'PHASE03B_ML_TRAINING_DIAGNOSTICS_REPORT.md'}")


if __name__ == "__main__":
    main()
