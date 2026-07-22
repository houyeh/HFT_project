from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "25_audit_and_walk_forward"
RESULTS = ROOT / "results"
V16 = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
V18 = PROJECT_ROOT / "new_era" / "18_queue_aware"
V24 = PROJECT_ROOT / "new_era" / "24_calibration_robust_policy"
for path in [V16, V18, V24]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from queue_stress_core import MAKER_FEE, POINT_VALUE, build_action_table  # noqa: E402
from round2_audit_utils import add_market_buckets  # noqa: E402
from calibration_utils import (  # noqa: E402
    apply_policy,
    evaluate_candidate,
    rolling_quantile_select,
    write_markdown,
)


FEATURES = [
    "obi_z",
    "queue_imbalance_z",
    "weighted_mid_deviation_z",
    "trade_intensity_short_z",
    "deep_obi_z",
    "queue_imbalance_change_z",
    "normalized_spread_z",
    "cancel_pressure_short_z",
]
SIDE_QS = [0.10, 0.20, 0.30, 0.50]
WINDOWS = [25, 50, 100, 200]


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)


def stable_u(event_id: int, side: str, salt: str) -> float:
    digest = hashlib.blake2b(f"025|{salt}|{int(event_id)}|{side}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64 - 1)


def load_labels() -> pd.DataFrame:
    labels = pd.read_parquet(V16 / "outputs" / "binary_labels_v2.parquet").reset_index(drop=True)
    labels["event_id"] = np.arange(len(labels), dtype=np.int64)
    return add_market_buckets(labels.sort_values(["session_id", "local_row_id"]).reset_index(drop=True))


def train_side_models(train: pd.DataFrame) -> dict[str, object]:
    models: dict[str, object] = {}
    for side in ["bid", "ask"]:
        y = train[f"{side}_toxic"].astype(int).to_numpy()
        x = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        valid = np.isfinite(x).all(axis=1)
        x = x[valid]
        y = y[valid]
        if len(x) > 300_000:
            rng = np.random.default_rng(20260525 + (0 if side == "bid" else 1))
            idx = rng.choice(np.arange(len(x)), size=300_000, replace=False)
            x = x[idx]
            y = y[idx]
        if len(np.unique(y)) < 2:
            models[side] = float(np.mean(y)) if len(y) else 0.0
        else:
            model = make_pipeline(
                StandardScaler(),
                SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, tol=1e-3, class_weight="balanced", random_state=20260525),
            )
            model.fit(x, y)
            models[side] = model
    return models


def add_fold_scores(frame: pd.DataFrame, models: dict[str, object]) -> pd.DataFrame:
    out = frame.copy()
    x = out[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    for side in ["bid", "ask"]:
        model = models[side]
        if isinstance(model, float):
            score = np.full(len(out), float(model), dtype=float)
        else:
            score = model.predict_proba(x)[:, 1].astype(float)
        out[f"{side}_prob_toxic"] = score
    return out


def action_frame(frame: pd.DataFrame, fold_id: int) -> pd.DataFrame:
    frame = frame.sort_values(["session_id", "local_row_id"]).reset_index(drop=True).copy()
    actions = build_action_table(frame)
    if actions.empty:
        return actions
    idx = actions["row_index"].to_numpy(dtype=np.int64)
    side = actions["side"].astype(str).to_numpy()
    actions["event_id"] = frame.loc[idx, "event_id"].to_numpy(dtype=np.int64)
    actions["spread_ticks"] = frame.loc[idx, "spread_ticks"].to_numpy(dtype=float)
    actions["spread_bucket"] = frame.loc[idx, "spread_bucket"].astype(str).to_numpy()
    actions["side_score_016"] = np.where(side == "bid", frame.loc[idx, "bid_prob_toxic"].to_numpy(dtype=float), frame.loc[idx, "ask_prob_toxic"].to_numpy(dtype=float))
    actions["spread_only_score"] = actions["spread_ticks"].astype(float)
    final_mid = frame.groupby("session_id", sort=True)["mid_price"].last().astype(float).to_dict()
    actions["final_mid"] = actions["session_id"].map(final_mid).astype(float)
    actions["neutral_u"] = [stable_u(eid, s, f"fold{fold_id}_neutral") for eid, s in zip(actions["event_id"], side)]
    actions["good_u"] = [stable_u(eid, s, f"fold{fold_id}_good") for eid, s in zip(actions["event_id"], side)]
    fut = actions["future_mid_return_H_ticks"].astype(float).to_numpy()
    side_markout = np.where(side == "bid", fut, -fut)
    terminal_ticks = np.where(side == "bid", actions["final_mid"].astype(float).to_numpy() - actions["price"].astype(float).to_numpy(), actions["price"].astype(float).to_numpy() - actions["final_mid"].astype(float).to_numpy())
    eligible = actions["neutral_u"].astype(float).to_numpy() <= 0.5
    favorable = side_markout > 0.0
    adverse = np.maximum(0.0, -side_markout)
    value = terminal_ticks * POINT_VALUE - MAKER_FEE - 0.5 * adverse * POINT_VALUE
    actions["quote_value_moderate"] = np.where(eligible & favorable | eligible & ~favorable, value, 0.0)
    actions["oracle_positive_quote"] = actions["quote_value_moderate"].astype(float) > 0.0
    return actions


def side_specific_policy(score_name: str, score_col: str, window: int, bid_q: float, ask_q: float, policy_id: str) -> dict:
    return {
        "policy_id": policy_id,
        "family": "fresh_walk_forward_side_specific",
        "score_name": score_name,
        "score_column": score_col,
        "window": int(window),
        "quote_fraction": None,
        "higher_is_better": False if score_name != "spread_only" else True,
        "group_cols": ["session_id", "side"],
        "selection_split": "fold_validation",
        "test_used_for_selection": False,
        "bid_quote_fraction": float(bid_q),
        "ask_quote_fraction": float(ask_q),
        "bid_cancel_quantile": 1.0 - float(bid_q),
        "ask_cancel_quantile": 1.0 - float(ask_q),
    }


def eval_policy(actions: pd.DataFrame, policy: dict, split_name: str) -> tuple[dict, np.ndarray]:
    selected, _ = rolling_quantile_select(
        actions,
        score_col=policy["score_column"],
        window=int(policy["window"]),
        quote_fraction=0.0,
        higher_is_better=bool(policy["higher_is_better"]),
        group_cols=["session_id", "side"],
        side_quote_fractions={"bid": float(policy["bid_quote_fraction"]), "ask": float(policy["ask_quote_fraction"])},
    )
    row = evaluate_candidate(
        actions,
        selected,
        policy_id=policy["policy_id"],
        family=policy["family"],
        score_name=policy["score_name"],
        score_col=policy["score_column"],
        window=int(policy["window"]),
        quote_fraction=None,
        group_cols=["session_id", "side"],
        higher_is_better=bool(policy["higher_is_better"]),
        split=split_name,
        extra={
            "bid_quote_fraction": float(policy["bid_quote_fraction"]),
            "ask_quote_fraction": float(policy["ask_quote_fraction"]),
            "test_used_for_selection": False,
        },
    )
    return row, selected


def select_on_validation(val_actions: pd.DataFrame, score_name: str, score_col: str) -> tuple[dict, dict]:
    rows = []
    policies = []
    for window in WINDOWS:
        for bid_q in SIDE_QS:
            for ask_q in SIDE_QS:
                policy = side_specific_policy(score_name, score_col, window, bid_q, ask_q, f"{score_name}__w{window}__bid{bid_q:.2f}__ask{ask_q:.2f}")
                row, _ = eval_policy(val_actions, policy, "fold_validation")
                row["selection_objective"] = row["selected_quote_pnl"] - 0.02 * row["false_positive_loss"]
                rows.append(row)
                policies.append(policy)
    grid = pd.DataFrame(rows)
    idx = int(grid.sort_values(["selection_objective", "selected_quote_pnl"], ascending=[False, False]).index[0])
    return policies[idx], rows[idx]


def run() -> None:
    ensure_dirs()
    labels = load_labels()
    sessions = [s for s in labels["session_id"].drop_duplicates().tolist() if not str(s).startswith("20260127")]
    folds = [
        {"fold": 0, "train": sessions[:3], "val": sessions[3], "test": sessions[4]},
        {"fold": 1, "train": sessions[:4], "val": sessions[4], "test": sessions[5]},
        {"fold": 2, "train": sessions[:5], "val": sessions[5], "test": sessions[6]},
    ]
    rows = []
    for fold in folds:
        train = labels[labels["session_id"].isin(fold["train"])].copy()
        val = labels[labels["session_id"].astype(str) == str(fold["val"])].copy()
        test = labels[labels["session_id"].astype(str) == str(fold["test"])].copy()
        models = train_side_models(train)
        val_scored = add_fold_scores(val, models)
        test_scored = add_fold_scores(test, models)
        val_actions = action_frame(val_scored, int(fold["fold"]))
        test_actions = action_frame(test_scored, int(fold["fold"]))
        if val_actions.empty or test_actions.empty:
            continue

        fixed = side_specific_policy("016_v2_fold_local", "side_score_016", 50, 0.50, 0.10, "fixed_024_architecture__016_v2_fold_local__w50__bid0.50__ask0.10")
        val_fixed, _ = eval_policy(val_actions, fixed, "fold_validation")
        test_fixed, _ = eval_policy(test_actions, fixed, "fresh_test_fold")
        rows.append({**fold, "policy_role": "fixed_024_architecture", "policy_json": json.dumps(fixed), **{f"val_{k}": v for k, v in val_fixed.items()}, **{f"test_{k}": v for k, v in test_fixed.items()}})

        ml_policy, val_ml = select_on_validation(val_actions, "016_v2_fold_local", "side_score_016")
        test_ml, _ = eval_policy(test_actions, ml_policy, "fresh_test_fold")
        rows.append({**fold, "policy_role": "val_selected_ml_architecture", "policy_json": json.dumps(ml_policy), **{f"val_{k}": v for k, v in val_ml.items()}, **{f"test_{k}": v for k, v in test_ml.items()}})

        spread_policy, val_spread = select_on_validation(val_actions, "spread_only", "spread_only_score")
        test_spread, _ = eval_policy(test_actions, spread_policy, "fresh_test_fold")
        rows.append({**fold, "policy_role": "val_selected_spread_only", "policy_json": json.dumps(spread_policy), **{f"val_{k}": v for k, v in val_spread.items()}, **{f"test_{k}": v for k, v in test_spread.items()}})

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "fresh_walk_forward_results.csv", index=False)
    summary = out.groupby("policy_role")["test_selected_quote_pnl"].agg(["count", "mean", "median", lambda s: int((s > 0).sum())]).reset_index()
    summary.columns = ["policy_role", "fold_count", "mean_test_pnl", "median_test_pnl", "positive_fold_count"]
    summary.to_csv(RESULTS / "fresh_walk_forward_summary.csv", index=False)
    write_markdown(
        RESULTS / "fresh_holdout_requirement.md",
        [
            "# Fresh Holdout Requirement",
            "",
            "The 20260127 test set is burned and is not used in these walk-forward folds.",
            "Each fold trains a fold-local causal risk ranker on earlier sessions, selects rolling calibration on the next validation session, and evaluates on the next unseen test session.",
            "Because only three pre-burned fresh folds are available, any positive result should be treated as confirmation evidence, not final proof. A future truly fresh holdout remains required before advisor-facing profitability claims.",
        ],
    )


if __name__ == "__main__":
    run()
    print(f"Wrote fresh walk-forward outputs to {RESULTS}")

