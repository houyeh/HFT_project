"""Regression tests for event-driven backtesting and gating behavior."""

import pandas as pd

from src.backtest.event_driven import run_event_driven_backtest
from src.backtest.signals import apply_probability_gating


def test_event_driven_backtest_opens_and_closes_positions():
    predictions = pd.DataFrame(
        [
            {"symbol": "TXFJ5", "timestamp": 1, "y_pred": 2, "current_mid": 100.0, "spread_ticks": 1.0, "tick_size": 1.0},
            {"symbol": "TXFJ5", "timestamp": 2, "y_pred": 2, "current_mid": 102.0, "spread_ticks": 1.0, "tick_size": 1.0},
            {"symbol": "TXFJ5", "timestamp": 3, "y_pred": 1, "current_mid": 103.0, "spread_ticks": 1.0, "tick_size": 1.0},
        ]
    )

    trade_log, summary, by_symbol = run_event_driven_backtest(
        predictions,
        spread_scale=1.0,
        max_holding_steps=10,
        close_on_stationary=True,
    )

    assert len(trade_log) == 1
    assert summary["trades"] == 1
    assert by_symbol.loc[0, "symbol"] == "TXFJ5"


def test_probability_gating_can_flatten_low_confidence_rows():
    predictions = pd.DataFrame(
        [
            {"prob_up": 0.4, "prob_down": 0.3, "prob_stationary": 0.3, "spread_ticks": 1.0, "rolling_volatility": 0.0},
            {"prob_up": 0.34, "prob_down": 0.33, "prob_stationary": 0.33, "spread_ticks": 1.0, "rolling_volatility": 0.0},
        ]
    )
    gated = apply_probability_gating(
        predictions,
        probability_threshold=0.38,
        margin_threshold=0.05,
    )

    assert list(gated["gated_signal"]) == [2, 1]


def test_probability_gating_supports_symbol_rules_and_regime_switches():
    predictions = pd.DataFrame(
        [
            {
                "symbol": "TXFJ5",
                "timestamp": 12,
                "prob_up": 0.36,
                "prob_down": 0.30,
                "prob_stationary": 0.34,
                "spread_ticks": 1.0,
                "rolling_volatility": 0.5,
                "trade_intensity_short": 0.2,
                "cancel_pressure_short": 0.1,
                "bid_vol_1": 5.0,
                "ask_vol_1": 5.0,
            },
            {
                "symbol": "MXFJ5",
                "timestamp": 14,
                "prob_up": 0.35,
                "prob_down": 0.30,
                "prob_stationary": 0.35,
                "spread_ticks": 1.0,
                "rolling_volatility": 0.9,
                "trade_intensity_short": 1.1,
                "cancel_pressure_short": 0.2,
                "bid_vol_1": 1.0,
                "ask_vol_1": 1.0,
            },
        ]
    )

    gated = apply_probability_gating(
        predictions,
        probability_threshold=0.334,
        margin_threshold=0.0,
        symbol_probability_thresholds={"TXFJ5": 0.355},
        symbol_end_times={"TXFJ5": 11, "MXFJ5": 20},
        regime_trade_intensity_thresholds={"MXFJ5": 1.0},
        regime_min_visible_queue={"MXFJ5": 3.0},
    )

    assert list(gated["gated_signal"]) == [1, 1]
    assert list(gated["trade_allowed_by_time"]) == [False, True]
    assert list(gated["trade_allowed_by_regime"]) == [True, False]


def test_probability_gating_can_limit_trade_universe():
    predictions = pd.DataFrame(
        [
            {
                "symbol": "TXFJ5",
                "timestamp": 10,
                "prob_up": 0.36,
                "prob_down": 0.30,
                "prob_stationary": 0.34,
                "spread_ticks": 1.0,
                "rolling_volatility": 0.1,
                "trade_intensity_short": 0.1,
                "cancel_pressure_short": 0.0,
                "bid_vol_1": 5.0,
                "ask_vol_1": 5.0,
            },
            {
                "symbol": "CDFJ5",
                "timestamp": 10,
                "prob_up": 0.36,
                "prob_down": 0.30,
                "prob_stationary": 0.34,
                "spread_ticks": 1.0,
                "rolling_volatility": 0.1,
                "trade_intensity_short": 0.1,
                "cancel_pressure_short": 0.0,
                "bid_vol_1": 5.0,
                "ask_vol_1": 5.0,
            },
        ]
    )

    gated = apply_probability_gating(
        predictions,
        probability_threshold=0.334,
        margin_threshold=0.0,
        allowed_symbols={"TXFJ5"},
    )

    assert list(gated["gated_signal"]) == [2, 1]
    assert list(gated["trade_allowed_by_symbol"]) == [True, False]
