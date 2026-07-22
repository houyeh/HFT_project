"""Tests for walk-forward fold generation and empty-symbol evaluation behavior."""

import pandas as pd

from src.models.walk_forward_optimize import build_walk_forward_slices, evaluate_symbol_config


def test_build_walk_forward_slices_preserves_chronological_order():
    rows = []
    for symbol in ["TXFJ5", "MXFJ5"]:
        for ts in range(10):
            rows.append({"symbol": symbol, "timestamp": ts, "prob_up": 0.34, "prob_down": 0.33})
    predictions = pd.DataFrame(rows)

    slices = build_walk_forward_slices(predictions, folds=2, warmup_frac=0.5)

    assert len(slices) == 2
    for fold_slice in slices:
        for symbol in ["TXFJ5", "MXFJ5"]:
            train_ts = fold_slice.train_predictions.loc[fold_slice.train_predictions["symbol"] == symbol, "timestamp"]
            test_ts = fold_slice.test_predictions.loc[fold_slice.test_predictions["symbol"] == symbol, "timestamp"]
            assert not train_ts.empty
            assert not test_ts.empty
            assert train_ts.max() < test_ts.min()


def test_evaluate_symbol_config_preserves_symbol_on_empty_input():
    config = {
        "score_threshold": 0.004,
        "one_sided_threshold": 0.008,
        "quote_off_threshold": 0.004,
        "stationary_cutoff": 0.338,
        "alpha_profile": "flow_queue",
        "inventory_penalty_ticks": 0.5,
        "quote_confirmation_steps": 0,
        "alpha_smoothing_window": 1,
        "stop_loss_ticks": 4.0,
        "take_profit_ticks": 0.0,
        "alpha_decay_threshold": 0.0,
        "require_directional_gate": False,
        "force_directional_one_sided": False,
        "regime_profile": "none",
        "max_holding_steps": 20,
        "adverse_score_threshold": 0.01,
        "end_time_ns": 43_200_000_000_000,
    }

    result = evaluate_symbol_config(
        symbol="TXFJ5",
        train_predictions=pd.DataFrame(columns=["symbol", "timestamp"]),
        trade_tapes={},
        config=config,
        alpha_profiles={"flow_queue": {"prob": 1.0}},
    )

    assert result["symbol"] == "TXFJ5"
    assert result["net_pnl_ticks"] == 0.0
    assert result["realized_trades"] == 0
