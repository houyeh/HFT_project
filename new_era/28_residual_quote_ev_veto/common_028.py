from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "28_residual_quote_ev_veto"
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

V16_DIR = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
V26_DIR = PROJECT_ROOT / "new_era" / "26_robust_baseline_side_aware"
V27_DIR = PROJECT_ROOT / "new_era" / "27_non_ml_final_baseline_audit"

for path in [PROJECT_ROOT, V26_DIR, V27_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_utils import actions_from_frame, apply_non_ml_policy, load_selected_non_ml_policy  # noqa: E402
from common_027 import build_new4_actions, build_new4_frame, load_actions_for_split, write_markdown  # noqa: E402
from round2_audit_utils import add_market_buckets  # noqa: E402


FEATURE_COLUMNS = [
    "spread_ticks",
    "queue_imbalance_calc",
    "deep_imbalance_calc",
    "top_depth",
    "depth_ratio",
    "volatility_past_50",
    "abs_return_past_50",
    "event_rate_past_50",
    "queue_imbalance_change_past",
    "normalized_spread_calc",
]

LABEL_FRAME_COLUMNS = [
    "dataset",
    "symbol",
    "session_id",
    "split",
    "local_row_id",
    "timestamp",
    "bid_px_1",
    "bid_vol_1",
    "ask_px_1",
    "ask_vol_1",
    "bid_px_2",
    "bid_vol_2",
    "ask_px_2",
    "ask_vol_2",
    "bid_px_3",
    "bid_vol_3",
    "ask_px_3",
    "ask_vol_3",
    "bid_px_4",
    "bid_vol_4",
    "ask_px_4",
    "ask_vol_4",
    "bid_px_5",
    "bid_vol_5",
    "ask_px_5",
    "ask_vol_5",
    "tick_size",
    "mid_price",
    "spread_ticks",
    "bid_toxic",
    "ask_toxic",
    "future_mid_return_H_ticks",
]


def ensure_dirs() -> None:
    for path in [ROOT, DATA_DIR, RESULTS, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


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
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def save_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_label_split(split: str) -> pd.DataFrame:
    frame = pd.read_parquet(V16_DIR / "outputs" / "binary_labels_v2.parquet", columns=LABEL_FRAME_COLUMNS)
    frame = frame[frame["split"].astype(str) == split].sort_values(["session_id", "local_row_id"]).reset_index(drop=True)
    frame["event_id"] = np.arange(len(frame), dtype=np.int64)
    return add_market_buckets(frame)


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_causal_frame_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["session_id", "local_row_id"]).reset_index(drop=True).copy()
    bid1 = out["bid_vol_1"].astype(float)
    ask1 = out["ask_vol_1"].astype(float)
    out["queue_imbalance_calc"] = _safe_ratio(bid1 - ask1, bid1 + ask1)
    bid_depth = sum(out[f"bid_vol_{i}"].astype(float) for i in range(1, 6) if f"bid_vol_{i}" in out.columns)
    ask_depth = sum(out[f"ask_vol_{i}"].astype(float) for i in range(1, 6) if f"ask_vol_{i}" in out.columns)
    out["deep_imbalance_calc"] = _safe_ratio(bid_depth - ask_depth, bid_depth + ask_depth)
    out["top_depth"] = (bid1 + ask1).fillna(0.0)
    out["depth_ratio"] = _safe_ratio(bid1, ask1)
    out["normalized_spread_calc"] = out["spread_ticks"].astype(float)

    parts: list[pd.DataFrame] = []
    for _, group in out.groupby("session_id", sort=False):
        g = group.copy()
        mid_ret = g["mid_price"].astype(float).diff()
        past_ret = mid_ret.shift(1)
        g["volatility_past_50"] = past_ret.rolling(50, min_periods=3).std().fillna(0.0)
        g["abs_return_past_50"] = past_ret.abs().rolling(50, min_periods=3).mean().fillna(0.0)
        dt = g["timestamp"].astype(float).diff()
        rate = (1_000_000_000.0 / dt.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        g["event_rate_past_50"] = rate.rolling(50, min_periods=3).mean().fillna(0.0).clip(lower=0.0, upper=1_000_000.0)
        g["queue_imbalance_change_past"] = g["queue_imbalance_calc"].astype(float).diff().shift(1).fillna(0.0)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    for col in FEATURE_COLUMNS:
        out[col] = out[col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_features_to_actions(actions: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    frame_features = add_causal_frame_features(frame)
    out = actions.reset_index(drop=True).copy()
    idx = out["row_index"].to_numpy(dtype=np.int64)
    for col in FEATURE_COLUMNS:
        out[col] = frame_features.loc[idx, col].to_numpy(dtype=float)
    for col in ["dataset", "symbol", "split", "local_row_id", "timestamp"]:
        if col in frame_features.columns:
            out[col] = frame_features.loc[idx, col].to_numpy()
    return out


def build_historical_actions_with_features(split: str) -> pd.DataFrame:
    frame = load_label_split(split)
    if split in {"val", "test"}:
        actions = load_actions_for_split(split)
    else:
        actions = actions_from_frame(frame, salt=f"028_{split}")
        actions["quote_value_moderate"] = actions["quote_value_moderate"].astype(float)
        actions["oracle_positive_quote"] = actions["quote_value_moderate"] > 0.0
    return add_features_to_actions(actions, frame)


def build_new4_actions_with_features() -> pd.DataFrame:
    frame, symbol, tick_size = build_new4_frame()
    actions = build_new4_actions(frame)
    actions = add_features_to_actions(actions, frame)
    actions["dataset"] = "NEW4"
    actions["symbol"] = symbol
    actions["split"] = "external_oos_new4"
    actions["tick_size"] = tick_size
    return actions


def baseline_universe(actions: pd.DataFrame) -> pd.DataFrame:
    policy = load_selected_non_ml_policy()
    selected = apply_non_ml_policy(actions, policy)
    out = actions.loc[selected].reset_index(drop=True).copy()
    out["baseline_policy_id"] = policy["policy_id"]
    out["quote_ev_moderate"] = out["quote_value_moderate"].astype(float)
    out["negative_ev_moderate"] = (out["quote_ev_moderate"] < 0.0).astype(int)
    return out


def load_baseline_dataset(split: str) -> pd.DataFrame:
    if split == "new4":
        return pd.read_parquet(DATA_DIR / "new4_baseline_universe_quote_ev.parquet")
    return pd.read_parquet(DATA_DIR / "baseline_universe_quote_ev.parquet").query("split == @split").reset_index(drop=True)


def economic_lift_rows(df: pd.DataFrame, score_col: str, model_name: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    work = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(work)
    for frac in [0.01, 0.05, 0.10]:
        k = max(1, int(np.ceil(frac * n))) if n else 0
        bucket = work.head(k)
        rows.append(
            {
                "model_name": model_name,
                "split": split,
                "bucket": f"top_{int(frac * 100)}pct_negative_ev_score",
                "quote_count": int(len(bucket)),
                "negative_ev_rate": float(bucket["negative_ev_moderate"].mean()) if len(bucket) else np.nan,
                "avg_quote_ev_moderate": float(bucket["quote_ev_moderate"].mean()) if len(bucket) else np.nan,
                "total_quote_ev_moderate": float(bucket["quote_ev_moderate"].sum()) if len(bucket) else 0.0,
                "economic_value_if_vetoed": float(-bucket["quote_ev_moderate"].sum()) if len(bucket) else 0.0,
            }
        )
    return rows


def matched_random_veto_after_pnls(
    df: pd.DataFrame,
    veto_mask: np.ndarray,
    *,
    trials: int = 5000,
    seed: int = 20260628,
) -> np.ndarray:
    work = df.reset_index(drop=True).copy()
    veto_mask = np.asarray(veto_mask, dtype=bool)
    values = work["quote_ev_moderate"].astype(float).to_numpy()
    groups = work.groupby(["side", "session_id", "spread_bucket"], observed=False, sort=True).groups
    rng = np.random.default_rng(seed)
    base_total = float(values.sum())
    out = np.zeros(int(trials), dtype=float)
    for t in range(int(trials)):
        random_veto = np.zeros(len(work), dtype=bool)
        for _, idx in groups.items():
            idx_arr = np.asarray(list(idx), dtype=np.int64)
            k = int(veto_mask[idx_arr].sum())
            if k:
                random_veto[rng.choice(idx_arr, size=min(k, len(idx_arr)), replace=False)] = True
        out[t] = base_total - float(values[random_veto].sum())
    return out

