"""Probability-gating helpers that turn classifier outputs into tradable signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rowwise_symbol_value(
    symbols: pd.Series,
    default_value: float | int | None,
    overrides: dict[str, float | int] | None,
) -> np.ndarray:
    values = np.full(len(symbols), default_value if default_value is not None else np.nan, dtype=float)
    if overrides:
        for symbol, override in overrides.items():
            values[symbols.to_numpy() == symbol] = float(override)
    return values


def apply_probability_gating(
    predictions: pd.DataFrame,
    probability_threshold: float = 0.5,
    margin_threshold: float = 0.0,
    spread_penalty: float = 0.0,
    volatility_penalty: float = 0.0,
    symbol_probability_thresholds: dict[str, float] | None = None,
    symbol_margin_thresholds: dict[str, float] | None = None,
    symbol_end_times: dict[str, int] | None = None,
    regime_volatility_thresholds: dict[str, float] | None = None,
    regime_trade_intensity_thresholds: dict[str, float] | None = None,
    regime_min_visible_queue: dict[str, float] | None = None,
    regime_max_cancel_pressure: dict[str, float] | None = None,
    allowed_symbols: set[str] | None = None,
) -> pd.DataFrame:
    """
    Convert raw class probabilities into trade signals with confidence gating.

    A direction is traded only if:
    1. The directional probability clears a dynamic threshold.
    2. The directional probability is sufficiently above stationary probability.
    Otherwise the signal is forced back to Stationary.
    """

    gated = predictions.copy()
    direction_up = gated["prob_up"].to_numpy(dtype=float)
    direction_down = gated["prob_down"].to_numpy(dtype=float)
    stationary = gated["prob_stationary"].to_numpy(dtype=float)
    spread_ticks = gated.get("spread_ticks", pd.Series(0.0, index=gated.index)).to_numpy(dtype=float)
    rolling_vol = gated.get("rolling_volatility", pd.Series(0.0, index=gated.index)).to_numpy(dtype=float)
    symbols = gated.get("symbol", pd.Series(["UNKNOWN"] * len(gated), index=gated.index))
    timestamps = gated.get("timestamp", pd.Series(np.zeros(len(gated)), index=gated.index)).to_numpy(dtype=float)
    trade_intensity = gated.get("trade_intensity_short", pd.Series(0.0, index=gated.index)).to_numpy(dtype=float)
    visible_queue = (
        gated.get("bid_vol_1", pd.Series(0.0, index=gated.index)).to_numpy(dtype=float)
        + gated.get("ask_vol_1", pd.Series(0.0, index=gated.index)).to_numpy(dtype=float)
    )
    cancel_pressure = np.abs(
        gated.get("cancel_pressure_short", pd.Series(0.0, index=gated.index)).to_numpy(dtype=float)
    )

    directional_prob = np.maximum(direction_up, direction_down)
    directional_class = np.where(direction_up >= direction_down, 2, 0)
    symbol_prob_threshold = _rowwise_symbol_value(symbols, probability_threshold, symbol_probability_thresholds)
    symbol_margin_threshold = _rowwise_symbol_value(symbols, margin_threshold, symbol_margin_thresholds)
    dynamic_threshold = (
        symbol_prob_threshold
        + spread_penalty * spread_ticks
        + volatility_penalty * rolling_vol
    )
    margin = directional_prob - stationary
    end_time_threshold = _rowwise_symbol_value(symbols, np.inf, symbol_end_times)
    volatility_threshold = _rowwise_symbol_value(symbols, np.inf, regime_volatility_thresholds)
    trade_intensity_threshold = _rowwise_symbol_value(symbols, np.inf, regime_trade_intensity_thresholds)
    visible_queue_threshold = _rowwise_symbol_value(symbols, 0.0, regime_min_visible_queue)
    cancel_pressure_threshold = _rowwise_symbol_value(symbols, np.inf, regime_max_cancel_pressure)

    symbol_allowed = np.ones(len(gated), dtype=bool)
    if allowed_symbols is not None:
        symbol_allowed = symbols.isin(sorted(allowed_symbols)).to_numpy(dtype=bool)

    within_symbol_time = timestamps <= end_time_threshold
    volatility_ok = rolling_vol <= volatility_threshold
    trade_intensity_ok = trade_intensity <= trade_intensity_threshold
    visible_queue_ok = visible_queue >= visible_queue_threshold
    cancel_pressure_ok = cancel_pressure <= cancel_pressure_threshold
    regime_ok = volatility_ok & trade_intensity_ok & visible_queue_ok & cancel_pressure_ok

    take_trade = (
        (directional_prob >= dynamic_threshold)
        & (margin >= symbol_margin_threshold)
        & within_symbol_time
        & regime_ok
        & symbol_allowed
    )

    gated["gated_signal"] = np.where(take_trade, directional_class, 1)
    gated["gated_directional_prob"] = directional_prob
    gated["gated_required_prob"] = dynamic_threshold
    gated["gated_margin"] = margin
    gated["trade_taken_by_gate"] = take_trade
    gated["symbol_prob_threshold"] = symbol_prob_threshold
    gated["symbol_margin_threshold"] = symbol_margin_threshold
    gated["symbol_end_time"] = end_time_threshold
    gated["trade_allowed_by_symbol"] = symbol_allowed
    gated["trade_allowed_by_time"] = within_symbol_time
    gated["trade_allowed_by_regime"] = regime_ok
    gated["regime_volatility_ok"] = volatility_ok
    gated["regime_trade_intensity_ok"] = trade_intensity_ok
    gated["regime_visible_queue_ok"] = visible_queue_ok
    gated["regime_cancel_pressure_ok"] = cancel_pressure_ok
    return gated
