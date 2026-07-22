from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.features.sliding_window import label_direction

LabelType = Literal[
    "mid_move_3class",
    "smoothed_mid_move_3class",
    "toxic_bid_binary",
    "toxic_ask_binary",
    "adverse_bid_fill_binary",
    "adverse_ask_fill_binary",
    "active_buy_profitable_binary",
    "active_sell_profitable_binary",
]


def _future_windows(values: np.ndarray, horizon: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive; got {horizon}")
    if n <= horizon:
        raise ValueError(f"Need more than horizon rows (n={n}, horizon={horizon}).")
    return sliding_window_view(values[1:], window_shape=horizon)


def generate_labels(
    label_type: LabelType,
    *,
    mid_prices: np.ndarray,
    bid_px_1: np.ndarray | None,
    ask_px_1: np.ndarray | None,
    tick_size: float,
    horizon: int,
    threshold_ticks: float,
    adverse_threshold_ticks: float = 1.0,
    taker_fee_twd: float = 50.0,
    point_value_twd_per_point: float = 200.0,
    exit_cost_ticks: float = 0.0,
    margin_ticks: float = 0.0,
) -> np.ndarray:
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive; got {tick_size}")

    mid_prices = np.asarray(mid_prices, dtype=np.float64)
    n = len(mid_prices)
    if n <= horizon:
        raise ValueError(f"Need more than horizon rows (n={n}, horizon={horizon}).")

    if label_type == "mid_move_3class":
        return label_direction(mid_prices=mid_prices / tick_size, horizon=horizon, threshold=float(threshold_ticks))

    future_mid = _future_windows(mid_prices, horizon=horizon)

    if label_type == "smoothed_mid_move_3class":
        future_mid_smooth = future_mid.mean(axis=1)
        current_mid = mid_prices[: n - horizon]
        delta_ticks = (future_mid_smooth - current_mid) / tick_size
        labels = np.full(n - horizon, 1, dtype=np.int64)
        labels[delta_ticks > threshold_ticks] = 2
        labels[delta_ticks < -threshold_ticks] = 0
        return labels

    if bid_px_1 is None or ask_px_1 is None:
        raise ValueError(f"{label_type} requires bid_px_1 and ask_px_1 arrays.")

    bid_px_1 = np.asarray(bid_px_1, dtype=np.float64)
    ask_px_1 = np.asarray(ask_px_1, dtype=np.float64)
    if len(bid_px_1) != n or len(ask_px_1) != n:
        raise ValueError("bid_px_1/ask_px_1 must have the same length as mid_prices.")

    threshold_price = float(adverse_threshold_ticks) * float(tick_size)
    current_bid = bid_px_1[: n - horizon]
    current_ask = ask_px_1[: n - horizon]

    if label_type == "toxic_bid_binary":
        future_mid_min = future_mid.min(axis=1)
        toxic = (future_mid_min < (current_bid - threshold_price)).astype(np.int64)
        toxic[current_bid <= 0] = 0
        return toxic

    if label_type == "toxic_ask_binary":
        future_mid_max = future_mid.max(axis=1)
        toxic = (future_mid_max > (current_ask + threshold_price)).astype(np.int64)
        toxic[current_ask <= 0] = 0
        return toxic

    next_ask = ask_px_1[1 : n - horizon + 1]
    next_bid = bid_px_1[1 : n - horizon + 1]

    if label_type == "adverse_bid_fill_binary":
        bid_fill = next_ask <= current_bid
        future_mid_min = future_mid.min(axis=1)
        adverse = bid_fill & (future_mid_min < (current_bid - threshold_price))
        out = adverse.astype(np.int64)
        out[current_bid <= 0] = 0
        return out

    if label_type == "adverse_ask_fill_binary":
        ask_fill = next_bid >= current_ask
        future_mid_max = future_mid.max(axis=1)
        adverse = ask_fill & (future_mid_max > (current_ask + threshold_price))
        out = adverse.astype(np.int64)
        out[current_ask <= 0] = 0
        return out

    taker_fee_points = float(taker_fee_twd) / float(point_value_twd_per_point)
    taker_fee_ticks = taker_fee_points / float(tick_size)
    total_cost_ticks = float(taker_fee_ticks) + float(exit_cost_ticks) + float(margin_ticks)

    if label_type == "active_buy_profitable_binary":
        future_mid_max = future_mid.max(axis=1)
        profit_ticks = (future_mid_max - current_ask) / tick_size
        out = (profit_ticks > total_cost_ticks).astype(np.int64)
        out[current_ask <= 0] = 0
        return out

    if label_type == "active_sell_profitable_binary":
        future_mid_min = future_mid.min(axis=1)
        profit_ticks = (current_bid - future_mid_min) / tick_size
        out = (profit_ticks > total_cost_ticks).astype(np.int64)
        out[current_bid <= 0] = 0
        return out

    raise ValueError(f"Unknown label_type: {label_type}")

