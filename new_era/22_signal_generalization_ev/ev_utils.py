from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "22_signal_generalization_ev"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
PLOTS_DIR = RESULTS_DIR / "plots"

V16_DIR = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
V20_DIR = PROJECT_ROOT / "new_era" / "20_policy_search"
V21_DIR = PROJECT_ROOT / "new_era" / "21_stress_robust_policy"
for path in [V16_DIR, V20_DIR, V21_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from policy_search_utils import MAKER_FEE, POINT_VALUE, add_market_buckets, load_policy_frame  # noqa: E402


MODERATE_NEUTRAL_HAIRCUT = 0.5
MODERATE_TOXIC_MULTIPLIER = 1.5
EV_POLICY_LAMBDA = 5.0
EV_POLICY_MAX_SPREAD = 2.0
EV_POLICY_THRESHOLD = 0.0


def ensure_dirs() -> None:
    for path in [ROOT, DATA_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_labels_with_event_id() -> pd.DataFrame:
    labels = pd.read_parquet(V16_DIR / "outputs" / "binary_labels_v2.parquet").reset_index(drop=True)
    labels["event_id"] = np.arange(len(labels), dtype=np.int64)
    return labels


def add_causal_volatility(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["session_id", "local_row_id"]).copy()
    parts = []
    for _, group in out.groupby("session_id", sort=False):
        g = group.copy()
        mid = g["mid_price"].astype(float)
        tick = g["tick_size"].astype(float).replace(0, np.nan).ffill().bfill().fillna(1.0)
        ret_ticks = mid.diff() / tick
        g["volatility"] = ret_ticks.rolling(100, min_periods=10).std().shift(1).fillna(0.0).to_numpy(dtype=float)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def load_frame(split: str) -> pd.DataFrame:
    split = split.lower()
    if split in {"val", "test"}:
        return add_causal_volatility(load_policy_frame(split))
    labels = load_labels_with_event_id()
    frame = labels[labels["split"].astype(str) == split].copy()
    frame = add_market_buckets(frame.sort_values(["session_id", "local_row_id"]).reset_index(drop=True))
    return add_causal_volatility(frame)


def load_all_label_frame() -> pd.DataFrame:
    labels = load_labels_with_event_id()
    labels = add_market_buckets(labels.sort_values(["session_id", "local_row_id"]).reset_index(drop=True))
    return add_causal_volatility(labels)


def _deterministic_u(event_id: np.ndarray, side: str) -> np.ndarray:
    out = np.empty(len(event_id), dtype=np.float64)
    for i, eid in enumerate(event_id):
        key = f"022|{int(eid)}|{side}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        out[i] = int.from_bytes(digest, "little") / float(2**64 - 1)
    return out


def side_quote_rows(frame: pd.DataFrame, *, include_scores: bool) -> pd.DataFrame:
    rows = []
    base_cols = [
        "event_id",
        "dataset",
        "symbol",
        "session_id",
        "split",
        "local_row_id",
        "timestamp",
        "mid_price",
        "tick_size",
        "spread_ticks",
        "event_rate_hz",
        "volatility",
        "queue_imbalance",
        "queue_imbalance_z",
        "cancel_pressure_short",
        "cancel_pressure_short_z",
        "trade_intensity_short",
        "trade_intensity_short_z",
        "normalized_spread_z",
    ]
    for side in ["bid", "ask"]:
        out = frame[[c for c in base_cols if c in frame.columns]].copy()
        out["side"] = side
        out["score_016"] = frame[f"{side}_prob_toxic"].to_numpy(dtype=float) if include_scores and f"{side}_prob_toxic" in frame.columns else np.nan
        out["score_015"] = frame[f"score_{side}_015"].to_numpy(dtype=float) if include_scores and f"score_{side}_015" in frame.columns else np.nan
        out["toxic"] = frame[f"{side}_toxic"].to_numpy(dtype=int)
        out["fill_proxy"] = frame[f"{side}_fill_proxy"].to_numpy(dtype=int)
        side_markout = frame["future_mid_return_H_ticks"].to_numpy(dtype=float)
        if side == "ask":
            side_markout = -side_markout
        out["side_markout_ticks"] = side_markout
        out["neutral_u_moderate"] = _deterministic_u(out["event_id"].to_numpy(dtype=np.int64), side)
        out["fill_kept_moderate"] = (out["fill_proxy"].astype(bool).to_numpy() & (out["neutral_u_moderate"].to_numpy(dtype=float) <= MODERATE_NEUTRAL_HAIRCUT)).astype(int)
        adverse = np.maximum(0.0, -side_markout)
        gross = 0.5 * out["spread_ticks"].astype(float).to_numpy() * POINT_VALUE + side_markout * POINT_VALUE - MAKER_FEE
        stress_penalty = (MODERATE_TOXIC_MULTIPLIER - 1.0) * adverse * POINT_VALUE
        out["realized_quote_pnl_moderate"] = np.where(out["fill_kept_moderate"].astype(bool), gross - stress_penalty, 0.0)
        out["quote_positive_ev_moderate"] = (out["realized_quote_pnl_moderate"].to_numpy(dtype=float) > 0.0).astype(int)
        if include_scores and "score_016" in out:
            out["ev_score_021"] = 0.5 * out["spread_ticks"].astype(float) * POINT_VALUE - EV_POLICY_LAMBDA * out["score_016"].astype(float) * POINT_VALUE - MAKER_FEE
        else:
            out["ev_score_021"] = np.nan
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def make_quote_ev_labels() -> pd.DataFrame:
    path = DATA_DIR / "quote_ev_labels.parquet"
    if path.exists():
        return pd.read_parquet(path)
    ensure_dirs()
    labels = load_all_label_frame()
    quote = side_quote_rows(labels, include_scores=False)
    quote.to_parquet(path, index=False)
    return quote


def quote_rows_for_split(split: str) -> pd.DataFrame:
    split = split.lower()
    if split in {"val", "test"}:
        return side_quote_rows(load_frame(split), include_scores=True)
    labels = make_quote_ev_labels()
    return labels[labels["split"].astype(str) == split].copy()


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(min(expected.min(), actual.min()), max(expected.max(), actual.max()), bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    e = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def bucket_by_validation_quantiles(val_scores: np.ndarray, scores: np.ndarray, bins: int = 10, higher_is_risk: bool = True) -> np.ndarray:
    val_scores = np.asarray(val_scores, dtype=float)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(val_scores)
    qs = np.unique(np.quantile(val_scores[finite], np.linspace(0, 1, bins + 1))) if finite.any() else np.array([0, 1])
    if len(qs) < 3:
        qs = np.linspace(np.nanmin(scores), np.nanmax(scores), bins + 1)
    qs[0] = -np.inf
    qs[-1] = np.inf
    bucket = np.digitize(scores, qs[1:-1], right=False) + 1
    if higher_is_risk:
        return bucket
    return bins - bucket + 1


def select_threshold_for_val_pnl(scores: np.ndarray, pnl: np.ndarray, *, quote_high_scores: bool) -> tuple[float, float, float]:
    scores = np.asarray(scores, dtype=float)
    pnl = np.asarray(pnl, dtype=float)
    finite = np.isfinite(scores) & np.isfinite(pnl)
    if not finite.any():
        return np.nan, np.nan, 0.0
    candidates = np.unique(np.quantile(scores[finite], np.linspace(0.05, 0.95, 37)))
    best = (-np.inf, np.nan, 0.0)
    for thr in candidates:
        quote = scores >= thr if quote_high_scores else scores <= thr
        value = float(np.nansum(pnl[quote & finite]))
        quote_rate = float(np.mean(quote & finite))
        if value > best[0]:
            best = (value, float(thr), quote_rate)
    return best[1], best[0], best[2]


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
