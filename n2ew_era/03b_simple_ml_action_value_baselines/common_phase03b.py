from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "03b_simple_ml_action_value_baselines"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
PREDICTIONS = ROOT / "predictions"
FIGURES = ROOT / "figures"

PHASE02F = PROJECT_ROOT / "n2ew_era" / "02f_excess_markout_and_causal_fill_proxy"
PHASE02F_DATA = PHASE02F / "data" / "full_action_universe_with_excess_markout.parquet"
PHASE02F_RESULTS = PHASE02F / "results"
PHASE03A5 = PROJECT_ROOT / "n2ew_era" / "03a5_pre_ml_freeze_audit"
PHASE03A6 = PROJECT_ROOT / "n2ew_era" / "03a6_root_cause_feature_and_contract_closure"
REBUILT_DYNAMICS = PHASE03A6 / "data" / "rebuilt_dynamic_features.parquet"

SPLITS = ["train", "val", "test", "new4", "new3"]
FROZEN_SPLITS = ["test", "new4", "new3"]
TARGET = "excess_pnl_H20"
POSITIVE_TARGET = "positive_excess_pnl_H20"
HORIZON = 20
POINT_VALUE = 200.0
MAKER_FEE = 15.0
ACTION_BID = 0
ACTION_ASK = 1
ACTION_NO_QUOTE = 2
DECISION_NO_QUOTE = 0
DECISION_BID = 1
DECISION_ASK = 2
RANDOM_Z95 = NormalDist().inv_cdf(0.95)
RANDOM_TRIALS = 1000
TRAIN_SAMPLE_ACTIONS = 900_000

IDENTITY_COLUMNS = [
    "split",
    "session_id",
    "symbol",
    "event_index",
    "timestamp",
    "action_side",
    "action_slot",
]

SOURCE_CAUSAL_COLUMNS = [
    "split",
    "session_id",
    "symbol",
    "event_index",
    "timestamp",
    "action_side",
    "action_slot",
    "spread_ticks",
    "bid_sz_1",
    "ask_sz_1",
    "top_depth",
    "queue_imbalance",
    "deep_imbalance",
    "volatility_past_50",
    TARGET,
    POSITIVE_TARGET,
]

REBUILT_DYNAMIC_COLUMNS = [
    "event_rate_past_50",
    "inter_arrival_ms",
    "imbalance_change_past",
    "spread_change_past",
]

NUMERIC_BASE_FEATURES = [
    "spread_ticks",
    "bid_sz_1",
    "ask_sz_1",
    "top_depth",
    "queue_imbalance",
    "deep_imbalance",
    "volatility_past_50",
    "event_rate_past_50",
    "inter_arrival_ms",
    "imbalance_change_past",
    "spread_change_past",
    "action_is_bid",
    "action_is_ask",
    "action_is_no_quote",
    "side_sign",
    "side_queue_imbalance",
    "side_deep_imbalance",
    "side_imbalance_change_past",
]

RAW_ABSOLUTE_PRICE_FEATURES = ["best_bid", "best_ask", "mid_price", "quote_price"]
FORBIDDEN_PATTERNS = [
    "raw_pnl",
    "pnl_H",
    "future_mid_H",
    "excess_bid_pnl_H",
    "excess_ask_pnl_H",
    "excess_pnl_H",
    "positive_excess_pnl_H",
    "negative_excess_pnl_H",
    "target_",
]


def ensure_dirs() -> None:
    for path in [ROOT, DATA, RESULTS, MODELS, PREDICTIONS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def part_paths(split: str) -> list[Path]:
    return sorted((PHASE02F_DATA / f"split={split}").glob("*.parquet"))


def dynamic_paths(split: str) -> list[Path]:
    return sorted((REBUILT_DYNAMICS / f"split={split}").glob("*.parquet"))


def schema_columns(path: Path) -> list[str]:
    return list(pq.read_schema(path).names)


def validate_feature_schema(feature_columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    for col in feature_columns:
        forbidden_reason = ""
        if col in RAW_ABSOLUTE_PRICE_FEATURES:
            forbidden_reason = "raw_absolute_price_forbidden_by_03a6"
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in col:
                forbidden_reason = f"forbidden_pattern:{pattern}"
        rows.append(
            {
                "feature": col,
                "allowed_for_phase03b": forbidden_reason == "",
                "forbidden_reason": forbidden_reason,
            }
        )
    return rows


def assert_no_feature_schema_violation(feature_columns: list[str]) -> None:
    audit = pd.DataFrame(validate_feature_schema(feature_columns))
    bad = audit[~audit["allowed_for_phase03b"].astype(bool)]
    if not bad.empty:
        blocker = ROOT / "ROOT_CAUSE_BLOCKER.md"
        write_markdown(
            blocker,
            [
                "# Phase 03B Root-Cause Blocker",
                "",
                "The requested ML training was stopped because forbidden features appeared in the Phase 03B feature schema.",
                "",
                bad.to_markdown(index=False),
            ],
        )
        raise RuntimeError(f"Forbidden Phase 03B feature columns detected. See {blocker}")


def load_dynamic_frame(path: Path) -> pd.DataFrame:
    cols = ["session_id", "event_index", *REBUILT_DYNAMIC_COLUMNS]
    return pd.read_parquet(path, columns=cols)


def add_action_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    slot = pd.to_numeric(out["action_slot"], errors="coerce").fillna(ACTION_NO_QUOTE).astype(np.int8)
    out["action_is_bid"] = (slot == ACTION_BID).astype(np.float32)
    out["action_is_ask"] = (slot == ACTION_ASK).astype(np.float32)
    out["action_is_no_quote"] = (slot == ACTION_NO_QUOTE).astype(np.float32)
    side_sign = np.zeros(len(out), dtype=np.float32)
    side_sign[slot.to_numpy() == ACTION_BID] = 1.0
    side_sign[slot.to_numpy() == ACTION_ASK] = -1.0
    out["side_sign"] = side_sign
    out["side_queue_imbalance"] = side_sign * pd.to_numeric(out["queue_imbalance"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    out["side_deep_imbalance"] = side_sign * pd.to_numeric(out["deep_imbalance"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    out["side_imbalance_change_past"] = side_sign * pd.to_numeric(out["imbalance_change_past"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return out


def build_feature_frame(path: Path, include_targets: bool = True) -> pd.DataFrame:
    cols = SOURCE_CAUSAL_COLUMNS if include_targets else SOURCE_CAUSAL_COLUMNS[:-2]
    frame = pd.read_parquet(path, columns=cols)
    dyn_path = REBUILT_DYNAMICS / f"split={str(frame['split'].iloc[0])}" / path.name
    if not dyn_path.exists():
        # Dynamic rebuild uses the same part-session naming as Phase 02F.
        matches = list((REBUILT_DYNAMICS / f"split={str(frame['split'].iloc[0])}").glob(path.name))
        dyn_path = matches[0] if matches else dyn_path
    if not dyn_path.exists():
        raise FileNotFoundError(f"Missing rebuilt dynamics for {path}")
    dynamic = load_dynamic_frame(dyn_path)
    frame = frame.merge(dynamic, on=["session_id", "event_index"], how="left", validate="many_to_one")
    for col in REBUILT_DYNAMIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame = add_action_features(frame)
    for col in NUMERIC_BASE_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    if include_targets:
        frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce").fillna(0.0).astype(np.float32)
        frame[POSITIVE_TARGET] = pd.to_numeric(frame[POSITIVE_TARGET], errors="coerce").fillna(False).astype(bool)
    return frame


def feature_matrix_columns() -> list[str]:
    assert_no_feature_schema_violation(NUMERIC_BASE_FEATURES)
    return list(NUMERIC_BASE_FEATURES)


def write_feature_schema() -> None:
    features = feature_matrix_columns()
    write_json(
        ROOT / "feature_schema_03b.json",
        {
            "target": TARGET,
            "positive_target": POSITIVE_TARGET,
            "feature_columns": features,
            "raw_absolute_price_features": RAW_ABSOLUTE_PRICE_FEATURES,
            "absolute_price_decision": read_json(PHASE03A6 / "absolute_price_feature_decision.json"),
            "dynamic_feature_decision": read_json(PHASE03A6 / "dynamic_feature_decision.json"),
            "test_new3_new4_used_for_feature_selection": False,
        },
    )


def train_matrix_path() -> Path:
    return DATA / "ml_feature_matrix_train.parquet"


def val_matrix_path() -> Path:
    return DATA / "ml_feature_matrix_val.parquet"


def save_feature_matrix(frame: pd.DataFrame, path: Path) -> None:
    cols = [*IDENTITY_COLUMNS, *feature_matrix_columns(), TARGET, POSITIVE_TARGET]
    table = pa.Table.from_pandas(frame[cols], preserve_index=False)
    pq.write_table(table, path, compression="zstd", use_dictionary=True, row_group_size=250_000)


def load_train_matrix() -> pd.DataFrame:
    return pd.read_parquet(train_matrix_path())


def load_val_matrix() -> pd.DataFrame:
    return pd.read_parquet(val_matrix_path())


def model_registry() -> dict[str, dict[str, Any]]:
    return {
        "ridge_l2": {"family": "linear_regression", "score_kind": "regression"},
        "lasso_l1": {"family": "linear_regression", "score_kind": "regression"},
        "logistic_l2": {"family": "linear_classification", "score_kind": "probability"},
        "logistic_l1": {"family": "linear_classification", "score_kind": "probability"},
        "hgb_regressor_shallow": {"family": "tree_boosting", "score_kind": "regression"},
        "hgb_classifier_shallow": {"family": "tree_boosting", "score_kind": "probability"},
    }


def model_path(model_id: str) -> Path:
    return MODELS / f"{model_id}.joblib"


def prediction_path(split: str, model_id: str) -> Path:
    return PREDICTIONS / f"{split}_predictions_{model_id}.parquet"


def selected_actions_path(split: str) -> Path:
    return PREDICTIONS / f"selected_actions_{split}.parquet"


def predict_model(model: Any, matrix: pd.DataFrame, model_id: str) -> np.ndarray:
    features = feature_matrix_columns()
    x = matrix[features].to_numpy(dtype=np.float32)
    score_kind = model_registry()[model_id]["score_kind"]
    if score_kind == "probability":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x)[:, 1].astype(np.float32)
        return model.decision_function(x).astype(np.float32)
    return model.predict(x).astype(np.float32)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    ar = pd.Series(a[mask]).rank(method="average").to_numpy(dtype=float)
    br = pd.Series(b[mask]).rank(method="average").to_numpy(dtype=float)
    if np.std(ar) == 0.0 or np.std(br) == 0.0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def max_drawdown(values: np.ndarray) -> float:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    equity = np.cumsum(values)
    peak = np.maximum.accumulate(np.insert(equity, 0, 0.0))[1:]
    return float(np.max(peak - equity)) if len(values) else 0.0


def assert_bid_ask_pairing(bid: pd.DataFrame, ask: pd.DataFrame, context: str) -> None:
    """Hard-stop if state-action bid/ask pairing is not exact."""
    keys = ["session_id", "event_index", "symbol"]
    if len(bid) != len(ask):
        raise RuntimeError(f"Bid/ask pairing length mismatch in {context}: bid={len(bid)} ask={len(ask)}")
    for key in keys:
        if not np.array_equal(bid[key].astype(str).to_numpy(), ask[key].astype(str).to_numpy()):
            mismatch = np.flatnonzero(bid[key].astype(str).to_numpy() != ask[key].astype(str).to_numpy())
            first = int(mismatch[0]) if len(mismatch) else -1
            raise RuntimeError(f"Bid/ask pairing key mismatch in {context}: key={key} first_mismatch={first}")


@dataclass
class DecisionResult:
    decision: np.ndarray
    selected_pnl: np.ndarray
    selected_score: np.ndarray
    selected_side: np.ndarray


def select_best_quote(matrix: pd.DataFrame, scores: np.ndarray, threshold: float) -> DecisionResult:
    quote = matrix[matrix["action_slot"].astype(int).isin([ACTION_BID, ACTION_ASK])].copy()
    quote = quote.assign(score=np.asarray(scores, dtype=np.float32)[matrix["action_slot"].astype(int).isin([ACTION_BID, ACTION_ASK]).to_numpy()])
    bid = quote[quote["action_slot"].astype(int) == ACTION_BID].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    ask = quote[quote["action_slot"].astype(int) == ACTION_ASK].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    assert_bid_ask_pairing(bid, ask, "select_best_quote")
    bid_score = bid["score"].to_numpy(dtype=np.float32)
    ask_score = ask["score"].to_numpy(dtype=np.float32)
    bid_pnl = bid[TARGET].to_numpy(dtype=np.float32)
    ask_pnl = ask[TARGET].to_numpy(dtype=np.float32)
    choose_bid = (bid_score >= ask_score) & (bid_score > threshold)
    choose_ask = (ask_score > bid_score) & (ask_score > threshold)
    decision = np.zeros(len(bid), dtype=np.int8)
    decision[choose_bid] = DECISION_BID
    decision[choose_ask] = DECISION_ASK
    pnl = np.zeros(len(bid), dtype=np.float32)
    pnl[choose_bid] = bid_pnl[choose_bid]
    pnl[choose_ask] = ask_pnl[choose_ask]
    selected_score = np.maximum(bid_score, ask_score)
    side = np.full(len(bid), "no_quote", dtype=object)
    side[choose_bid] = "bid"
    side[choose_ask] = "ask"
    return DecisionResult(decision=decision, selected_pnl=pnl, selected_score=selected_score, selected_side=side)


def evaluate_decision(matrix: pd.DataFrame, result: DecisionResult, split: str, policy_id: str, model_id: str, threshold: float) -> dict[str, Any]:
    selected = result.decision != DECISION_NO_QUOTE
    pnl = result.selected_pnl
    selected_pnl = pnl[selected]
    bid_count = int((result.decision == DECISION_BID).sum())
    ask_count = int((result.decision == DECISION_ASK).sum())
    state_count = int(len(result.decision))
    return {
        "split": split,
        "policy_id": policy_id,
        "model_id": model_id,
        "horizon": HORIZON,
        "threshold": float(threshold),
        "state_count": state_count,
        "selected_action_count": int(selected.sum()),
        "selection_rate": float(selected.sum() / max(state_count, 1)),
        "bid_count": bid_count,
        "ask_count": ask_count,
        "total_excess_pnl": float(np.nansum(selected_pnl)),
        "win_rate": float((selected_pnl > 0).mean()) if len(selected_pnl) else 0.0,
        "loss_rate": float((selected_pnl < 0).mean()) if len(selected_pnl) else 0.0,
        "max_drawdown": max_drawdown(pnl),
        "selection_split": "val",
        "test_new3_new4_used_for_selection": False,
    }


def selected_actions_frame(matrix: pd.DataFrame, result: DecisionResult, model_id: str, threshold: float) -> pd.DataFrame:
    states = matrix[matrix["action_slot"].astype(int) == ACTION_BID].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    return pd.DataFrame(
        {
            "split": states["split"].astype(str).to_numpy(),
            "session_id": states["session_id"].astype(str).to_numpy(),
            "symbol": states["symbol"].astype(str).to_numpy(),
            "event_index": states["event_index"].to_numpy(),
            "timestamp": states["timestamp"].to_numpy(),
            "decision": result.decision.astype(np.int8),
            "selected_side": result.selected_side,
            "selected_score": result.selected_score.astype(np.float32),
            "selected_excess_pnl_H20": result.selected_pnl.astype(np.float32),
            "model_id": model_id,
            "threshold": float(threshold),
        }
    )


def queue_rule_decision_from_state(states: pd.DataFrame) -> np.ndarray:
    q = pd.to_numeric(states["queue_imbalance"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    decision = np.zeros(len(states), dtype=np.int8)
    decision[q > 0.1] = DECISION_BID
    decision[q < -0.1] = DECISION_ASK
    return decision


def old_bid_only_decision_from_state(states: pd.DataFrame) -> np.ndarray:
    return np.full(len(states), DECISION_BID, dtype=np.int8)


def policy_pnl_from_decision(matrix: pd.DataFrame, decision: np.ndarray) -> np.ndarray:
    bid = matrix[matrix["action_slot"].astype(int) == ACTION_BID].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    ask = matrix[matrix["action_slot"].astype(int) == ACTION_ASK].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    assert_bid_ask_pairing(bid, ask, "policy_pnl_from_decision")
    out = np.zeros(len(decision), dtype=np.float32)
    bid_mask = decision == DECISION_BID
    ask_mask = decision == DECISION_ASK
    bid_pnl = bid[TARGET].to_numpy(dtype=np.float32)
    ask_pnl = ask[TARGET].to_numpy(dtype=np.float32)
    out[bid_mask] = bid_pnl[bid_mask]
    out[ask_mask] = ask_pnl[ask_mask]
    return np.nan_to_num(out, nan=0.0)


def baseline_policy_summary(matrix: pd.DataFrame, split: str) -> pd.DataFrame:
    states = matrix[matrix["action_slot"].astype(int) == ACTION_BID].sort_values(["session_id", "event_index"]).reset_index(drop=True)
    rows = []
    for policy_id, decision in [
        ("no_quote", np.zeros(len(states), dtype=np.int8)),
        ("old_bid_only_spread_ge_1", old_bid_only_decision_from_state(states)),
        ("side_by_queue_imbalance_abs_0p1", queue_rule_decision_from_state(states)),
    ]:
        pnl = policy_pnl_from_decision(matrix, decision)
        selected = decision != 0
        selected_pnl = pnl[selected]
        rows.append(
            {
                "split": split,
                "policy_id": policy_id,
                "horizon": HORIZON,
                "state_count": int(len(states)),
                "selected_action_count": int(selected.sum()),
                "selection_rate": float(selected.sum() / max(len(states), 1)),
                "bid_count": int((decision == DECISION_BID).sum()),
                "ask_count": int((decision == DECISION_ASK).sum()),
                "total_excess_pnl": float(selected_pnl.sum()),
                "win_rate": float((selected_pnl > 0).mean()) if len(selected_pnl) else 0.0,
                "loss_rate": float((selected_pnl < 0).mean()) if len(selected_pnl) else 0.0,
                "max_drawdown": max_drawdown(pnl),
            }
        )
    return pd.DataFrame(rows)


def model_score_frame_for_split(split: str, model_id: str) -> Iterable[tuple[Path, pd.DataFrame, np.ndarray]]:
    model = joblib.load(model_path(model_id))
    for path in part_paths(split):
        matrix = build_feature_frame(path, include_targets=True)
        scores = predict_model(model, matrix, model_id)
        yield path, matrix, scores


def random_normal_trials(values: np.ndarray, k: int, trials: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(len(values))
    k = int(k)
    if k <= 0 or n <= 0:
        return np.zeros(trials, dtype=float)
    if k >= n:
        return np.full(trials, float(values.sum()), dtype=float)
    mean = float(values.mean())
    var = float(values.var(ddof=0))
    expected = k * mean
    variance = k * (1.0 - k / n) * var * (n / max(n - 1, 1))
    std = math.sqrt(max(variance, 0.0))
    return rng.normal(expected, std, size=trials)


def stable_uniform(session_id: pd.Series, event_index: pd.Series, decision: np.ndarray, salt: str) -> np.ndarray:
    keys = pd.DataFrame(
        {
            "session_id": session_id.astype(str).to_numpy(),
            "event_index": pd.to_numeric(event_index, errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
            "decision": np.asarray(decision, dtype=np.int8),
        }
    )
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    salt_hash = np.uint64(int.from_bytes(hashlib.blake2b(salt.encode("utf-8"), digest_size=8).digest(), "little"))
    hashes = hashes ^ salt_hash
    return hashes.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)


@dataclass
class FillNorms:
    top_depth_median: float
    bid_sz_1_median: float
    ask_sz_1_median: float


def compute_train_fill_norms() -> FillNorms:
    values = {"top_depth": [], "bid_sz_1": [], "ask_sz_1": []}
    for path in part_paths("train"):
        frame = build_feature_frame(path, include_targets=False)
        state = frame[frame["action_slot"].astype(int) == ACTION_BID]
        for col in values:
            arr = pd.to_numeric(state[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(arr):
                values[col].append(arr[arr > 0])
    med = {}
    for col, chunks in values.items():
        arr = np.concatenate(chunks) if chunks else np.array([1.0])
        med[col] = float(np.nanmedian(arr)) if len(arr) else 1.0
    return FillNorms(top_depth_median=med["top_depth"], bid_sz_1_median=med["bid_sz_1"], ask_sz_1_median=med["ask_sz_1"])


def conservative_l2_fill_mask(states: pd.DataFrame, decision: np.ndarray, norms: FillNorms) -> tuple[np.ndarray, np.ndarray]:
    top_depth = pd.to_numeric(states["top_depth"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    bid_depth = pd.to_numeric(states["bid_sz_1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ask_depth = pd.to_numeric(states["ask_sz_1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    depth = 1.0 / (1.0 + np.maximum(top_depth / max(norms.top_depth_median, 1.0), 0.0))
    side_norm = np.zeros(len(states), dtype=float)
    bid_mask = decision == DECISION_BID
    ask_mask = decision == DECISION_ASK
    side_norm[bid_mask] = bid_depth[bid_mask] / max(norms.bid_sz_1_median, 1.0)
    side_norm[ask_mask] = ask_depth[ask_mask] / max(norms.ask_sz_1_median, 1.0)
    side = 1.0 / (1.0 + np.maximum(side_norm, 0.0))
    probs = np.zeros(len(states), dtype=float)
    selected = decision != DECISION_NO_QUOTE
    probs[selected] = np.clip(0.5 * depth[selected] * side[selected], 0.01, 1.0)
    u = stable_uniform(states["session_id"], states["event_index"], decision, salt="03b|conservative_l2_stress")
    return selected & (u < probs), probs


def simulate_scheduled_excess(decision: np.ndarray, fills: np.ndarray, pnl: np.ndarray, horizon: int = HORIZON, inventory_cap: int = 1) -> dict[str, Any]:
    n = len(decision)
    close_side = np.zeros(n, dtype=np.int32)
    close_pnl = np.zeros(n, dtype=np.float64)
    inventory = 0
    max_abs_inventory = 0
    forced = 0
    fills_n = 0
    wins = 0
    losses = 0
    equity = 0.0
    total = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in range(n):
        if close_side[t] != 0 or close_pnl[t] != 0.0:
            inventory -= int(close_side[t])
            equity += float(close_pnl[t])
            total += float(close_pnl[t])
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        if not fills[t]:
            continue
        side_sign = 1 if decision[t] == DECISION_BID else -1
        next_inventory = inventory + side_sign
        if abs(next_inventory) > inventory_cap:
            forced += 1
            continue
        trade_pnl = float(pnl[t])
        inventory = next_inventory
        max_abs_inventory = max(max_abs_inventory, abs(inventory))
        close_idx = min(t + horizon, n - 1)
        close_side[close_idx] += side_sign
        close_pnl[close_idx] += trade_pnl
        fills_n += 1
        if trade_pnl > 0:
            wins += 1
        elif trade_pnl < 0:
            losses += 1
    return {
        "sequential_excess_pnl": float(total),
        "fill_count": int(fills_n),
        "forced_no_quote_count": int(forced),
        "max_abs_inventory": int(max_abs_inventory),
        "max_drawdown": float(max_dd),
        "win_rate": float(wins / max(fills_n, 1)) if fills_n else 0.0,
        "loss_rate": float(losses / max(fills_n, 1)) if fills_n else 0.0,
        "avg_trade_excess_pnl": float(total / max(fills_n, 1)) if fills_n else 0.0,
        "fill_uses_future_pnl": False,
    }
