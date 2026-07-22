"""Event-driven single-position backtest with market and passive entry modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


POSITION_MAP = {0: -1, 1: 0, 2: 1}


@dataclass
class OpenPosition:
    symbol: str
    side: int
    entry_timestamp: int
    entry_index: int
    entry_mid: float
    entry_price: float
    tick_size: float
    spread_scale: float
    entry_mode: str
    size: float


@dataclass
class RestingOrder:
    symbol: str
    side: int
    placed_timestamp: int
    placed_index: int
    order_price: float
    queue_ahead: float
    filled_volume: float
    tick_size: float
    size: float


def _best_quotes(row: pd.Series, spread_scale: float) -> tuple[float, float]:
    if "bid_px_1" in row and "ask_px_1" in row and row["bid_px_1"] > 0 and row["ask_px_1"] > 0:
        return float(row["bid_px_1"]), float(row["ask_px_1"])

    half_spread = row["spread_ticks"] * row["tick_size"] * 0.5 * spread_scale
    best_bid = row["current_mid"] - half_spread
    best_ask = row["current_mid"] + half_spread
    return float(best_bid), float(best_ask)


def _close_position(position: OpenPosition, row: pd.Series, exit_reason: str, exit_index: int) -> dict:
    best_bid, best_ask = _best_quotes(row, position.spread_scale)
    if position.side > 0:
        exit_price = best_bid
        pnl_ticks = (exit_price - position.entry_price) / position.tick_size
    else:
        exit_price = best_ask
        pnl_ticks = (position.entry_price - exit_price) / position.tick_size
    pnl_ticks *= position.size

    return {
        "symbol": position.symbol,
        "side": "long" if position.side > 0 else "short",
        "entry_mode": position.entry_mode,
        "size": position.size,
        "entry_timestamp": position.entry_timestamp,
        "exit_timestamp": int(row["timestamp"]),
        "entry_mid": position.entry_mid,
        "exit_mid": float(row["current_mid"]),
        "entry_price": position.entry_price,
        "exit_price": float(exit_price),
        "holding_steps": int(exit_index - position.entry_index),
        "pnl_ticks": float(pnl_ticks),
        "exit_reason": exit_reason,
    }


def _trade_signal(row: pd.Series, signal_column: str) -> int:
    raw_signal = int(row[signal_column])
    return POSITION_MAP[raw_signal]


def _trade_slice(trade_tape: pd.DataFrame | None, start_ts: int, end_ts: int) -> pd.DataFrame:
    if trade_tape is None or trade_tape.empty:
        return pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume"])
    mask = (trade_tape["timestamp"] > start_ts) & (trade_tape["timestamp"] <= end_ts)
    return trade_tape.loc[mask]


def _try_fill_resting_order(
    order: RestingOrder,
    current_row: pd.Series,
    trade_tape: pd.DataFrame | None,
    spread_scale: float,
) -> tuple[bool, float]:
    trades = _trade_slice(trade_tape, order.placed_timestamp, int(current_row["timestamp"]))
    if trades.empty:
        return False, 0.0

    if order.side > 0:
        executable = trades.loc[trades["trade_price"] <= order.order_price, "trade_volume"].sum()
    else:
        executable = trades.loc[trades["trade_price"] >= order.order_price, "trade_volume"].sum()

    if executable >= order.queue_ahead + order.size:
        return True, float(order.order_price)
    return False, 0.0


def _compute_position_size(
    confidence: float,
    mode: str,
    max_position_size: int,
    confidence_floor: float,
    confidence_ceiling: float,
) -> float:
    if mode == "fixed" or max_position_size <= 1:
        return 1.0

    if confidence <= confidence_floor:
        return 1.0
    if confidence >= confidence_ceiling:
        return float(max_position_size)

    scaled = 1.0 + (confidence - confidence_floor) / max(confidence_ceiling - confidence_floor, 1e-9)
    size = min(max_position_size, max(1.0, scaled * max_position_size / 2.0))
    return float(round(size, 2))


def run_event_driven_backtest(
    predictions: pd.DataFrame,
    spread_scale: float = 1.0,
    max_holding_steps: int | None = None,
    close_on_stationary: bool = True,
    signal_column: str = "y_pred",
    trade_tapes: dict[str, pd.DataFrame] | None = None,
    entry_mode: str = "market",
    passive_confidence_threshold: float = 0.55,
    max_passive_wait_steps: int = 3,
    cooldown_steps: int = 0,
    symbol_cooldown_steps: dict[str, int] | None = None,
    size_mode: str = "fixed",
    max_position_size: int = 1,
    confidence_floor: float = 0.334,
    confidence_ceiling: float = 0.360,
    min_timestamp: int | None = None,
    max_timestamp: int | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    trade_rows: list[dict] = []

    for symbol, symbol_df in predictions.sort_values(["symbol", "timestamp"]).groupby("symbol", sort=False):
        position: OpenPosition | None = None
        resting_order: RestingOrder | None = None
        cooldown_remaining = 0
        symbol_cooldown = cooldown_steps if symbol_cooldown_steps is None else int(
            symbol_cooldown_steps.get(symbol, cooldown_steps)
        )
        trade_tape = None if trade_tapes is None else trade_tapes.get(symbol)
        ordered = symbol_df.reset_index(drop=True)

        for idx, row in ordered.iterrows():
            current_timestamp = int(row["timestamp"])
            signal = _trade_signal(row, signal_column)
            directional_prob = float(
                max(
                    row.get("gated_directional_prob", 0.0),
                    row.get("prob_up", 0.0),
                    row.get("prob_down", 0.0),
                )
            )
            best_bid, best_ask = _best_quotes(row, spread_scale)
            if cooldown_remaining > 0:
                cooldown_remaining -= 1

            within_time_window = (
                (min_timestamp is None or current_timestamp >= min_timestamp)
                and (max_timestamp is None or current_timestamp <= max_timestamp)
            )

            if resting_order is not None:
                filled, fill_price = _try_fill_resting_order(resting_order, row, trade_tape, spread_scale)
                if filled:
                    position = OpenPosition(
                        symbol=symbol,
                        side=resting_order.side,
                        entry_timestamp=current_timestamp,
                        entry_index=idx,
                        entry_mid=float(row["current_mid"]),
                        entry_price=fill_price,
                        tick_size=float(row["tick_size"]),
                        spread_scale=spread_scale,
                        entry_mode="passive",
                        size=resting_order.size,
                    )
                    resting_order = None
                else:
                    if signal == 0 or signal != resting_order.side:
                        resting_order = None
                    elif idx - resting_order.placed_index >= max_passive_wait_steps:
                        resting_order = None

            if position is not None and max_holding_steps is not None:
                if idx - position.entry_index >= max_holding_steps:
                    closed = _close_position(position, row, "max_holding_steps", idx)
                    trade_rows.append(closed)
                    if closed["pnl_ticks"] < 0 and symbol_cooldown > 0:
                        cooldown_remaining = symbol_cooldown
                    position = None

            if position is not None:
                if signal == 0 and close_on_stationary:
                    closed = _close_position(position, row, "stationary_signal", idx)
                    trade_rows.append(closed)
                    if closed["pnl_ticks"] < 0 and symbol_cooldown > 0:
                        cooldown_remaining = symbol_cooldown
                    position = None
                    continue
                if signal != 0 and signal != position.side:
                    closed = _close_position(position, row, "reverse_signal", idx)
                    trade_rows.append(closed)
                    if closed["pnl_ticks"] < 0 and symbol_cooldown > 0:
                        cooldown_remaining = symbol_cooldown
                    position = None

            if position is None and resting_order is None and signal != 0 and within_time_window and cooldown_remaining == 0:
                size = _compute_position_size(
                    confidence=directional_prob,
                    mode=size_mode,
                    max_position_size=max_position_size,
                    confidence_floor=confidence_floor,
                    confidence_ceiling=confidence_ceiling,
                )
                use_passive = (
                    entry_mode == "passive"
                    or (entry_mode == "hybrid" and directional_prob < passive_confidence_threshold)
                )
                if use_passive:
                    queue_ahead = float(row["bid_vol_1"] if signal > 0 else row["ask_vol_1"])
                    order_price = best_bid if signal > 0 else best_ask
                    resting_order = RestingOrder(
                        symbol=symbol,
                        side=signal,
                        placed_timestamp=int(row["timestamp"]),
                        placed_index=idx,
                        order_price=float(order_price),
                        queue_ahead=max(queue_ahead, 0.0),
                        filled_volume=0.0,
                        tick_size=float(row["tick_size"]),
                        size=size,
                    )
                else:
                    entry_price = best_ask if signal > 0 else best_bid
                    position = OpenPosition(
                        symbol=symbol,
                        side=signal,
                        entry_timestamp=current_timestamp,
                        entry_index=idx,
                        entry_mid=float(row["current_mid"]),
                        entry_price=float(entry_price),
                        tick_size=float(row["tick_size"]),
                        spread_scale=spread_scale,
                        entry_mode="market",
                        size=size,
                    )

        if resting_order is not None and trade_tape is not None and not ordered.empty:
            last_row = ordered.iloc[-1]
            filled, fill_price = _try_fill_resting_order(resting_order, last_row, trade_tape, spread_scale)
            if filled:
                position = OpenPosition(
                    symbol=symbol,
                    side=resting_order.side,
                    entry_timestamp=int(last_row["timestamp"]),
                    entry_index=len(ordered) - 1,
                    entry_mid=float(last_row["current_mid"]),
                    entry_price=fill_price,
                    tick_size=float(last_row["tick_size"]),
                    spread_scale=spread_scale,
                    entry_mode="passive",
                    size=resting_order.size,
                )

        if position is not None:
            trade_rows.append(_close_position(position, ordered.iloc[-1], "end_of_stream", len(ordered) - 1))

    trade_log = pd.DataFrame(trade_rows)
    if trade_log.empty:
        summary = {
            "trades": 0,
            "win_rate": 0.0,
            "net_pnl_ticks": 0.0,
            "avg_pnl_ticks": 0.0,
            "avg_holding_steps": 0.0,
            "sharpe_like": 0.0,
        }
        by_symbol = pd.DataFrame(columns=["symbol", "trades", "net_pnl_ticks", "avg_pnl_ticks"])
        return trade_log, summary, by_symbol

    pnl_values = trade_log["pnl_ticks"].to_numpy(dtype=float)
    pnl_mean = float(pnl_values.mean())
    pnl_std = float(pnl_values.std(ddof=1)) if len(pnl_values) > 1 else 0.0
    sharpe_like = float((pnl_mean / pnl_std) * np.sqrt(len(pnl_values))) if pnl_std > 0 else 0.0

    summary = {
        "trades": int(len(trade_log)),
        "win_rate": float((trade_log["pnl_ticks"] > 0).mean()),
        "net_pnl_ticks": float(trade_log["pnl_ticks"].sum()),
        "avg_pnl_ticks": pnl_mean,
        "avg_holding_steps": float(trade_log["holding_steps"].mean()),
        "sharpe_like": sharpe_like,
    }

    by_symbol = (
        trade_log.groupby("symbol", as_index=False)
        .agg(
            trades=("symbol", "size"),
            net_pnl_ticks=("pnl_ticks", "sum"),
            avg_pnl_ticks=("pnl_ticks", "mean"),
        )
        .sort_values("net_pnl_ticks", ascending=False)
        .reset_index(drop=True)
    )

    return trade_log, summary, by_symbol
