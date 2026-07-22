from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"

POINT_VALUE = 200.0
MAKER_FEE = 15.0
MAX_INV = 3


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def threshold_for_target_rate(scores: np.ndarray, target_rate: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan")
    sorted_scores = np.sort(scores)[::-1]
    kth = max(0, min(len(sorted_scores) - 1, int(np.ceil(float(target_rate) * len(sorted_scores))) - 1))
    return float(sorted_scores[kth])


def add_market_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["spread_bucket"] = pd.cut(
        out["spread_ticks"].astype(float),
        bins=[-np.inf, 1.0, 2.0, 3.0, np.inf],
        labels=["<=1", "(1,2]", "(2,3]", ">3"],
    ).astype(str)
    out["coarse_spread_bucket"] = np.where(out["spread_ticks"].astype(float) <= 2.0, "tight_<=2", "wide_>2")

    dt = out.groupby("session_id")["timestamp"].diff()
    positive_dt = dt[dt > 0]
    fallback_dt = float(positive_dt.median()) if len(positive_dt) else 1.0
    dt_filled = dt.fillna(fallback_dt).clip(lower=1.0)
    out["event_rate_hz"] = 1_000_000_000.0 / dt_filled
    out["event_rate_bucket"] = pd.qcut(
        out["event_rate_hz"].rank(method="first"),
        q=4,
        labels=["q1_slow", "q2", "q3", "q4_fast"],
    ).astype(str)
    out["coarse_event_rate_bucket"] = np.where(
        out["event_rate_bucket"].isin(["q1_slow", "q2"]),
        "slow_half",
        "fast_half",
    )
    return out


def load_labels_with_event_id() -> pd.DataFrame:
    labels = pd.read_parquet(OUTPUTS / "binary_labels_v2.parquet").reset_index(drop=True)
    labels["event_id"] = np.arange(len(labels), dtype=np.int64)
    return labels


def load_test_frame() -> pd.DataFrame:
    labels = load_labels_with_event_id()
    preds = pd.read_parquet(OUTPUTS / "test_predictions_binary_cancel.parquet")
    keep = [
        "event_id",
        "timestamp",
        "tick_size",
        "spread_ticks",
        "future_mid_return_H_ticks",
        "bid_fill_proxy",
        "ask_fill_proxy",
        "bid_toxic",
        "ask_toxic",
    ]
    merged = preds.merge(labels[keep], on="event_id", how="left")
    return add_market_buckets(merged.sort_values(["session_id", "local_row_id"]).reset_index(drop=True))


def load_val_frame() -> pd.DataFrame:
    labels = load_labels_with_event_id()
    bid = pd.read_parquet(OUTPUTS / "model_weighted_bce" / "val_predictions_bid.parquet")
    ask = pd.read_parquet(OUTPUTS / "model_weighted_bce" / "val_predictions_ask.parquet")
    frame = bid[
        [
            "event_id",
            "dataset",
            "symbol",
            "session_id",
            "split",
            "local_row_id",
            "bid_px_1",
            "ask_px_1",
            "mid_price",
            "tick_size",
            "future_mid_return_H_ticks",
            "prob_toxic",
            "y_true",
            "fill_proxy",
        ]
    ].rename(
        columns={
            "prob_toxic": "bid_prob_toxic",
            "y_true": "bid_toxic",
            "fill_proxy": "bid_fill_proxy",
        }
    )
    ask = ask[["event_id", "prob_toxic", "y_true", "fill_proxy"]].rename(
        columns={
            "prob_toxic": "ask_prob_toxic",
            "y_true": "ask_toxic",
            "fill_proxy": "ask_fill_proxy",
        }
    )
    frame = frame.merge(ask, on="event_id", how="inner")
    frame = frame.merge(labels[["event_id", "timestamp", "spread_ticks"]], on="event_id", how="left")
    frame["pull_bid"] = False
    frame["pull_ask"] = False
    return add_market_buckets(frame.sort_values(["session_id", "local_row_id"]).reset_index(drop=True))


def simulate_cancel_policy(
    frame: pd.DataFrame,
    pull_bid: np.ndarray,
    pull_ask: np.ndarray,
    *,
    method: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pull_bid = np.asarray(pull_bid, dtype=bool)
    pull_ask = np.asarray(pull_ask, dtype=bool)
    session_rows: list[dict[str, Any]] = []

    for session_id, group in frame.groupby("session_id", sort=True):
        idx = group.index.to_numpy()
        summary = _simulate_one_session(
            group.sort_values("local_row_id").reset_index(drop=True),
            pull_bid[idx],
            pull_ask[idx],
        )
        summary["session_id"] = str(session_id)
        session_rows.append(summary)

    session_df = pd.DataFrame(session_rows)
    rows = int(session_df["rows"].sum()) if not session_df.empty else 0
    agg = {
        "method": method,
        "sessions": int(len(session_df)),
        "rows": rows,
        "bid_signal_rate": float(np.sum(pull_bid) / max(rows, 1)),
        "ask_signal_rate": float(np.sum(pull_ask) / max(rows, 1)),
        "cancel_count": int(np.sum(pull_bid) + np.sum(pull_ask)),
        "bid_cancel_count": int(np.sum(pull_bid)),
        "ask_cancel_count": int(np.sum(pull_ask)),
        "trade_count": int(session_df["trade_count"].sum()) if not session_df.empty else 0,
        "net_pnl": float(session_df["net_pnl"].sum()) if not session_df.empty else 0.0,
        "max_drawdown": float(session_df["max_drawdown"].max()) if not session_df.empty else 0.0,
    }
    agg["pnl_per_trade"] = float(agg["net_pnl"] / max(agg["trade_count"], 1))
    return agg, session_df


def _simulate_one_session(group: pd.DataFrame, pull_bid: np.ndarray, pull_ask: np.ndarray) -> dict[str, Any]:
    bid = group["bid_px_1"].to_numpy(dtype=float)
    ask = group["ask_px_1"].to_numpy(dtype=float)
    mid = group["mid_price"].to_numpy(dtype=float)

    cash = 0.0
    fees = 0.0
    inventory = 0
    trades = 0
    equity_curve: list[float] = []

    for i in range(max(len(group) - 1, 0)):
        if not pull_bid[i] and inventory < MAX_INV and ask[i + 1] <= bid[i]:
            cash -= bid[i] * POINT_VALUE
            fees += MAKER_FEE
            inventory += 1
            trades += 1
        if not pull_ask[i] and inventory > -MAX_INV and bid[i + 1] >= ask[i]:
            cash += ask[i] * POINT_VALUE
            fees += MAKER_FEE
            inventory -= 1
            trades += 1
        equity_curve.append(cash - fees + inventory * mid[i] * POINT_VALUE)

    final_equity = cash - fees + inventory * mid[-1] * POINT_VALUE if len(group) else 0.0
    if equity_curve:
        equity_curve[-1] = final_equity
    equity = np.asarray(equity_curve, dtype=float)
    return {
        "rows": int(len(group)),
        "trade_count": int(trades),
        "ending_inventory": int(inventory),
        "net_pnl": float(final_equity),
        "max_drawdown": max_drawdown(equity),
    }
