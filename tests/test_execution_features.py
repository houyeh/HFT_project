"""Unit tests for execution-aware feature engineering."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.execution_features import (
    align_trade_features_to_quotes,
    compute_queue_imbalance,
    infer_trade_signs,
)


def make_quotes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [100, 200, 300],
            "bid_px_1": [100.0, 100.0, 100.0],
            "ask_px_1": [101.0, 101.0, 101.0],
            "bid_vol_1": [10.0, 6.0, 8.0],
            "ask_vol_1": [10.0, 8.0, 7.0],
            "bid_px_2": [99.0, 99.0, 99.0],
            "ask_px_2": [102.0, 102.0, 102.0],
            "bid_vol_2": [6.0, 5.0, 5.0],
            "ask_vol_2": [6.0, 5.0, 4.0],
            "bid_px_3": [98.0, 98.0, 98.0],
            "ask_px_3": [103.0, 103.0, 103.0],
            "bid_vol_3": [4.0, 4.0, 4.0],
            "ask_vol_3": [4.0, 4.0, 4.0],
            "bid_px_4": [97.0, 97.0, 97.0],
            "ask_px_4": [104.0, 104.0, 104.0],
            "bid_vol_4": [3.0, 3.0, 3.0],
            "ask_vol_4": [3.0, 3.0, 3.0],
            "bid_px_5": [96.0, 96.0, 96.0],
            "ask_px_5": [105.0, 105.0, 105.0],
            "bid_vol_5": [2.0, 2.0, 2.0],
            "ask_vol_5": [2.0, 2.0, 2.0],
        }
    )


def test_compute_queue_imbalance_uses_near_touch_weights():
    quotes = make_quotes()
    queue_imbalance = compute_queue_imbalance(quotes, depth=5)

    assert queue_imbalance.iloc[0] == 0.0
    assert queue_imbalance.iloc[1] < 0.0
    assert queue_imbalance.iloc[2] > queue_imbalance.iloc[1]


def test_infer_trade_signs_uses_quotes_and_tick_rule():
    quotes = make_quotes()
    trades = pd.DataFrame(
        {
            "timestamp": [150, 175, 250],
            "trade_price": [101.0, 100.0, 100.5],
            "trade_volume": [2.0, 1.0, 3.0],
        }
    )
    signed = infer_trade_signs(trades, quotes, tick_size=1.0)

    assert list(signed["trade_sign"]) == [1, -1, 1]


def test_align_trade_features_to_quotes_builds_trade_and_cancel_proxies():
    quotes = make_quotes()
    trades = pd.DataFrame(
        {
            "timestamp": [150, 175],
            "trade_price": [101.0, 100.0],
            "trade_volume": [2.0, 1.0],
        }
    )
    features = align_trade_features_to_quotes(quotes, trades, tick_size=1.0, rolling_window=2)

    assert features.loc[1, "trade_count_interval"] == 2.0
    assert features.loc[1, "signed_trade_volume_interval"] == 1.0
    assert features.loc[1, "buy_aggressor_volume_interval"] == 2.0
    assert features.loc[1, "sell_aggressor_volume_interval"] == 1.0
    assert features.loc[1, "bid_cancel_proxy"] == 3.0
    assert features.loc[1, "ask_cancel_proxy"] == 0.0
    assert features.loc[1, "net_cancel_pressure"] < 0.0
