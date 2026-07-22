"""Regression tests for the queue-aware hybrid market maker."""

import pandas as pd

from src.backtest.hybrid_market_maker import run_hybrid_market_maker_backtest


def _base_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "TXFJ5",
                "timestamp": 1,
                "prob_up": 0.334,
                "prob_down": 0.334,
                "prob_stationary": 0.332,
                "current_mid": 100.5,
                "spread_ticks": 1.0,
                "tick_size": 1.0,
                "bid_px_1": 100.0,
                "ask_px_1": 101.0,
                "bid_vol_1": 0.0,
                "ask_vol_1": 0.0,
                "rolling_volatility": 0.1,
                "trade_intensity_short": 0.1,
                "cancel_pressure_short": 0.0,
            },
            {
                "symbol": "TXFJ5",
                "timestamp": 2,
                "prob_up": 0.334,
                "prob_down": 0.334,
                "prob_stationary": 0.332,
                "current_mid": 100.5,
                "spread_ticks": 1.0,
                "tick_size": 1.0,
                "bid_px_1": 100.0,
                "ask_px_1": 101.0,
                "bid_vol_1": 0.0,
                "ask_vol_1": 0.0,
                "rolling_volatility": 0.1,
                "trade_intensity_short": 0.1,
                "cancel_pressure_short": 0.0,
            },
            {
                "symbol": "TXFJ5",
                "timestamp": 3,
                "prob_up": 0.334,
                "prob_down": 0.334,
                "prob_stationary": 0.332,
                "current_mid": 100.5,
                "spread_ticks": 1.0,
                "tick_size": 1.0,
                "bid_px_1": 100.0,
                "ask_px_1": 101.0,
                "bid_vol_1": 0.0,
                "ask_vol_1": 0.0,
                "rolling_volatility": 0.1,
                "trade_intensity_short": 0.1,
                "cancel_pressure_short": 0.0,
            },
            {
                "symbol": "TXFJ5",
                "timestamp": 4,
                "prob_up": 0.334,
                "prob_down": 0.334,
                "prob_stationary": 0.332,
                "current_mid": 100.5,
                "spread_ticks": 1.0,
                "tick_size": 1.0,
                "bid_px_1": 100.0,
                "ask_px_1": 101.0,
                "bid_vol_1": 0.0,
                "ask_vol_1": 0.0,
                "rolling_volatility": 0.1,
                "trade_intensity_short": 0.1,
                "cancel_pressure_short": 0.0,
            },
        ]
    )


def test_hybrid_market_maker_can_capture_spread():
    predictions = _base_predictions()
    trade_tapes = {
        "TXFJ5": pd.DataFrame(
            [
                {"timestamp": 2, "trade_price": 100.0, "trade_volume": 1.0},
                {"timestamp": 4, "trade_price": 101.0, "trade_volume": 1.0},
            ]
        )
    }

    trade_log, summary, by_symbol, quote_log = run_hybrid_market_maker_backtest(
        predictions,
        trade_tapes=trade_tapes,
        score_threshold=0.01,
        one_sided_threshold=0.02,
        max_inventory=1,
        max_holding_steps=10,
        adverse_score_threshold=0.05,
    )

    assert summary["fills"] == 2
    assert summary["realized_trades"] == 1
    assert summary["net_pnl_ticks"] == 1.0
    assert by_symbol.loc[0, "symbol"] == "TXFJ5"
    assert not quote_log.empty
    assert trade_log.loc[0, "reason"] == "passive_ask_fill"


def test_hybrid_market_maker_latency_delays_quote_activation():
    predictions = _base_predictions().iloc[:2].copy().reset_index(drop=True)
    trade_tapes = {
        "TXFJ5": pd.DataFrame(
            [
                {"timestamp": 2, "trade_price": 100.0, "trade_volume": 1.0},
            ]
        )
    }

    _, summary_no_latency, _, _ = run_hybrid_market_maker_backtest(
        predictions,
        trade_tapes=trade_tapes,
        score_threshold=0.01,
        one_sided_threshold=0.02,
        max_inventory=1,
        latency_steps=0,
        max_holding_steps=10,
        adverse_score_threshold=0.05,
    )
    _, summary_with_latency, _, _ = run_hybrid_market_maker_backtest(
        predictions,
        trade_tapes=trade_tapes,
        score_threshold=0.01,
        one_sided_threshold=0.02,
        max_inventory=1,
        latency_steps=1,
        max_holding_steps=10,
        adverse_score_threshold=0.05,
    )

    assert summary_no_latency["fills"] == 1
    assert summary_with_latency["fills"] == 0


def test_hybrid_market_maker_quote_off_threshold_can_disable_quoting():
    predictions = _base_predictions().iloc[:2].copy().reset_index(drop=True)
    trade_tapes = {
        "TXFJ5": pd.DataFrame(
            [
                {"timestamp": 2, "trade_price": 100.0, "trade_volume": 1.0},
                {"timestamp": 2, "trade_price": 101.0, "trade_volume": 1.0},
            ]
        )
    }

    _, summary, _, quote_log = run_hybrid_market_maker_backtest(
        predictions,
        trade_tapes=trade_tapes,
        score_threshold=0.01,
        one_sided_threshold=0.02,
        quote_off_score_threshold=0.01,
        max_inventory=1,
        max_holding_steps=10,
        adverse_score_threshold=0.05,
    )

    assert summary["fills"] == 0
    assert quote_log.empty


def test_hybrid_market_maker_can_require_directional_gate():
    predictions = _base_predictions().iloc[:2].copy().reset_index(drop=True)
    predictions["gated_signal"] = [1, 1]
    trade_tapes = {
        "TXFJ5": pd.DataFrame(
            [
                {"timestamp": 2, "trade_price": 100.0, "trade_volume": 1.0},
                {"timestamp": 2, "trade_price": 101.0, "trade_volume": 1.0},
            ]
        )
    }

    _, summary, _, quote_log = run_hybrid_market_maker_backtest(
        predictions,
        trade_tapes=trade_tapes,
        score_threshold=0.01,
        one_sided_threshold=0.02,
        max_inventory=1,
        require_directional_gate=True,
        max_holding_steps=10,
        adverse_score_threshold=0.05,
    )

    assert summary["fills"] == 0
    assert quote_log.empty


def test_hybrid_market_maker_withdraws_toxic_side_quotes():
    predictions = _base_predictions().iloc[:2].copy().reset_index(drop=True)
    predictions["queue_imbalance"] = [-0.9, -0.9]
    predictions["queue_imbalance_change"] = [-0.8, -0.8]
    predictions["aggressor_imbalance_interval"] = [-1.0, -1.0]
    predictions["bid_cancel_proxy"] = [12.0, 12.0]
    predictions["ask_cancel_proxy"] = [0.0, 0.0]
    predictions["net_cancel_pressure"] = [-1.0, -1.0]
    predictions["cancel_pressure_short"] = [-1.0, -1.0]

    _, summary, _, quote_log = run_hybrid_market_maker_backtest(
        predictions,
        trade_tapes={"TXFJ5": pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume"])},
        score_threshold=0.01,
        one_sided_threshold=0.02,
        quote_off_score_threshold=0.0,
        toxicity_threshold=99.0,
        one_sided_toxicity_threshold=1.0,
        alpha_coefficients={
            "toxicity_queue_imbalance": 1.0,
            "toxicity_queue_change": 1.0,
            "toxicity_aggressor_flow": 1.0,
            "toxicity_cancel_pressure": 1.0,
            "toxicity_bid_cancel_proxy": 1.0,
        },
        max_inventory=1,
        max_holding_steps=10,
        adverse_score_threshold=0.05,
    )

    assert summary["quote_updates"] > 0
    assert (quote_log["side"] == "bid").sum() == 0
    assert (quote_log["side"] == "ask").sum() > 0


def test_hybrid_market_maker_can_freeze_on_extreme_toxicity():
    predictions = _base_predictions().copy().reset_index(drop=True)
    predictions.loc[1, "queue_imbalance"] = -1.0
    predictions.loc[1, "queue_imbalance_change"] = -1.0
    predictions.loc[1, "aggressor_imbalance_interval"] = -1.0
    predictions.loc[1, "bid_cancel_proxy"] = 20.0
    predictions.loc[1, "cancel_pressure_short"] = -1.0

    _, summary, _, quote_log = run_hybrid_market_maker_backtest(
        predictions,
        trade_tapes={"TXFJ5": pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume"])},
        score_threshold=0.01,
        one_sided_threshold=0.02,
        quote_off_score_threshold=0.0,
        toxicity_freeze_threshold=1.0,
        toxicity_freeze_steps=2,
        alpha_coefficients={
            "toxicity_queue_imbalance": 1.0,
            "toxicity_queue_change": 1.0,
            "toxicity_aggressor_flow": 1.0,
            "toxicity_cancel_pressure": 1.0,
            "toxicity_bid_cancel_proxy": 1.0,
        },
        max_inventory=1,
        max_holding_steps=10,
        adverse_score_threshold=0.05,
    )

    assert summary["toxicity_freeze_events"] == 1
    assert "freeze" in quote_log["action"].tolist()


def test_hybrid_market_maker_skews_quotes_for_inventory_unwind():
    predictions = _base_predictions().copy().reset_index(drop=True)
    trade_tapes = {
        "TXFJ5": pd.DataFrame(
            [
                {"timestamp": 2, "trade_price": 100.0, "trade_volume": 1.0},
            ]
        )
    }

    _, _, _, quote_log = run_hybrid_market_maker_backtest(
        predictions,
        trade_tapes=trade_tapes,
        score_threshold=0.01,
        one_sided_threshold=0.02,
        max_inventory=1,
        inventory_tighten_ticks=1.0,
        inventory_widen_ticks=1.0,
        max_holding_steps=10,
        adverse_score_threshold=0.05,
    )

    unwind_asks = quote_log.loc[(quote_log["timestamp"] >= 3) & (quote_log["side"] == "ask") & (quote_log["action"] == "place")]
    assert not unwind_asks.empty
    assert float(unwind_asks["order_price"].min()) < 101.0
