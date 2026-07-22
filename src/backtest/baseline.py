"""Vectorized signal backtest for fast directional-PnL sanity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd


POSITION_MAP = {0: -1, 1: 0, 2: 1}


def run_signal_backtest(
    predictions: pd.DataFrame,
    spread_cost_multiplier: float = 1.0,
    signal_column: str = "y_pred",
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    trades = predictions.copy()
    trades["position"] = trades[signal_column].map(POSITION_MAP).astype(int)
    trades["trade_taken"] = trades["position"] != 0
    trades["gross_pnl_ticks"] = trades["position"] * trades["future_return_ticks"]
    trades["transaction_cost_ticks"] = np.where(
        trades["trade_taken"],
        trades["spread_ticks"] * spread_cost_multiplier,
        0.0,
    )
    trades["net_pnl_ticks"] = trades["gross_pnl_ticks"] - trades["transaction_cost_ticks"]
    trades["correct_direction"] = (
        ((trades["position"] > 0) & (trades["future_return_ticks"] > 0))
        | ((trades["position"] < 0) & (trades["future_return_ticks"] < 0))
    )

    active = trades.loc[trades["trade_taken"]].copy()
    trade_count = len(active)
    pnl_values = active["net_pnl_ticks"].to_numpy(dtype=float) if trade_count else np.array([], dtype=float)
    pnl_mean = float(pnl_values.mean()) if trade_count else 0.0
    pnl_std = float(pnl_values.std(ddof=1)) if trade_count > 1 else 0.0
    sharpe_like = float((pnl_mean / pnl_std) * np.sqrt(trade_count)) if pnl_std > 0 else 0.0

    summary = {
        "samples": int(len(trades)),
        "trades": int(trade_count),
        "trade_rate": float(trade_count / len(trades)),
        "gross_pnl_ticks": float(trades["gross_pnl_ticks"].sum()),
        "net_pnl_ticks": float(trades["net_pnl_ticks"].sum()),
        "avg_net_pnl_ticks": pnl_mean,
        "hit_rate": float(active["correct_direction"].mean()) if trade_count else 0.0,
        "long_trades": int((trades["position"] > 0).sum()),
        "short_trades": int((trades["position"] < 0).sum()),
        "sharpe_like": sharpe_like,
    }

    by_symbol = (
        trades.groupby("symbol", as_index=False)
        .agg(
            samples=("symbol", "size"),
            trades=("trade_taken", "sum"),
            gross_pnl_ticks=("gross_pnl_ticks", "sum"),
            net_pnl_ticks=("net_pnl_ticks", "sum"),
        )
        .sort_values("net_pnl_ticks", ascending=False)
        .reset_index(drop=True)
    )

    return trades, summary, by_symbol
