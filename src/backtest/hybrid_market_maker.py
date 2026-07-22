"""Queue-aware hybrid market-making simulator with alpha, toxicity, and inventory controls."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np
import pandas as pd


@dataclass
class InventoryState:
    """做市商當前的庫存狀態追蹤器"""
    symbol: str               # 交易標的代碼
    quantity: float = 0.0     # 當前持倉量 (正=Long, 負=Short)
    average_price: float = 0.0 # 持倉平均成本價
    entry_timestamp: int = 0  # 建倉的奈秒級時間戳
    entry_index: int = 0      # 建倉在 DataFrame 中的索引值
    max_abs_inventory: float = 0.0 # 紀錄歷史最大絕對持倉，用於懲罰機制


@dataclass
class QuoteOrder:
    """掛出的限價單 (Limit Order) 物理狀態"""
    symbol: str               # 交易標的代碼
    side: int                 # 掛單方向：1 為買(Bid)，-1 為賣(Ask)
    placed_timestamp: int     # 訂單送達到交易所的奈秒時間戳
    placed_index: int         # 訂單送達時的資料流索引
    order_price: float        # 限價單價格
    queue_ahead: float        # 排在我們前面的單量 (關鍵：這模擬了排隊延遲 Queue-Delay)
    filled_volume: float      # 訂單送出後可成交量累積（用 trade tape 增量更新）
    size: float               # 我們掛出的口數
    alpha_score: float        # 掛單當下的 TCN 模型預測分數 (用於後續分析)


@dataclass
class QuoteTarget:
    apply_index: int
    bid_active: bool
    ask_active: bool
    bid_price: float
    ask_price: float
    bid_queue_ahead: float
    ask_queue_ahead: float
    alpha_score: float
    bid_toxicity: float
    ask_toxicity: float


def _same_target(left: QuoteTarget, right: QuoteTarget) -> bool:
    return (
        left.bid_active == right.bid_active
        and left.ask_active == right.ask_active
        and abs(left.bid_price - right.bid_price) < 1e-9
        and abs(left.ask_price - right.ask_price) < 1e-9
    )


def _symbol_value(symbol: str, default_value: float | int, overrides: dict[str, float | int] | None) -> float:
    if overrides and symbol in overrides:
        return float(overrides[symbol])
    return float(default_value)


def _scaled_positive(value: float) -> float:
    return float(np.log1p(max(value, 0.0)))


def _alpha_score_from_row(
    row: pd.Series,
    alpha_coefficients: dict[str, float] | None,
) -> float:
    # Combine model probabilities with queue/trade/cancel features into a single
    # directional score that decides whether quotes should lean bid or ask.
    coeffs = {
        "prob_edge": 1.0,
        "queue_imbalance": 0.0,
        "queue_change": 0.0,
        "aggressor_flow": 0.0,
        "cancel_pressure": 0.0,
        "net_cancel_pressure": 0.0,
        "bid_cancel_proxy": 0.0,
        "ask_cancel_proxy": 0.0,
    }
    if alpha_coefficients:
        coeffs.update(alpha_coefficients)

    bid_cancel_proxy = _scaled_positive(float(row.get("bid_cancel_proxy", 0.0)))
    ask_cancel_proxy = _scaled_positive(float(row.get("ask_cancel_proxy", 0.0)))
    score = (
        coeffs["prob_edge"] * float(row.get("prob_up", 0.0) - row.get("prob_down", 0.0))
        + coeffs["queue_imbalance"] * float(row.get("queue_imbalance", 0.0))
        + coeffs["queue_change"] * float(row.get("queue_imbalance_change", 0.0))
        + coeffs["aggressor_flow"] * float(row.get("aggressor_imbalance_interval", 0.0))
        + coeffs["cancel_pressure"] * float(row.get("cancel_pressure_short", 0.0))
        + coeffs["net_cancel_pressure"] * float(row.get("net_cancel_pressure", 0.0))
        - coeffs["bid_cancel_proxy"] * bid_cancel_proxy
        + coeffs["ask_cancel_proxy"] * ask_cancel_proxy
    )
    return float(score)


def _toxicity_scores_from_row(
    row: pd.Series,
    alpha_coefficients: dict[str, float] | None,
) -> tuple[float, float]:
    coeffs = {
        "toxicity_queue_imbalance": 0.0,
        "toxicity_queue_change": 0.0,
        "toxicity_aggressor_flow": 0.0,
        "toxicity_cancel_pressure": 0.0,
        "toxicity_bid_cancel_proxy": 0.0,
        "toxicity_ask_cancel_proxy": 0.0,
    }
    if alpha_coefficients:
        coeffs.update(alpha_coefficients)

    queue_imbalance = float(row.get("queue_imbalance", 0.0))
    queue_change = float(row.get("queue_imbalance_change", 0.0))
    aggressor_flow = float(row.get("aggressor_imbalance_interval", 0.0))
    cancel_pressure = float(row.get("cancel_pressure_short", 0.0))
    bid_cancel_proxy = _scaled_positive(float(row.get("bid_cancel_proxy", 0.0)))
    ask_cancel_proxy = _scaled_positive(float(row.get("ask_cancel_proxy", 0.0)))

    bid_toxicity = (
        coeffs["toxicity_queue_imbalance"] * max(-queue_imbalance, 0.0)
        + coeffs["toxicity_queue_change"] * max(-queue_change, 0.0)
        + coeffs["toxicity_aggressor_flow"] * max(-aggressor_flow, 0.0)
        + coeffs["toxicity_cancel_pressure"] * max(-cancel_pressure, 0.0)
        + coeffs["toxicity_bid_cancel_proxy"] * bid_cancel_proxy
    )
    ask_toxicity = (
        coeffs["toxicity_queue_imbalance"] * max(queue_imbalance, 0.0)
        + coeffs["toxicity_queue_change"] * max(queue_change, 0.0)
        + coeffs["toxicity_aggressor_flow"] * max(aggressor_flow, 0.0)
        + coeffs["toxicity_cancel_pressure"] * max(cancel_pressure, 0.0)
        + coeffs["toxicity_ask_cancel_proxy"] * ask_cancel_proxy
    )
    return float(bid_toxicity), float(ask_toxicity)


def _symbol_alpha_coefficients(
    symbol: str,
    alpha_coefficients: dict[str, float] | None,
    symbol_alpha_coefficients: dict[str, dict[str, float]] | None,
) -> dict[str, float] | None:
    if symbol_alpha_coefficients and symbol in symbol_alpha_coefficients:
        return symbol_alpha_coefficients[symbol]
    return alpha_coefficients


def _best_quotes(row: pd.Series, spread_scale: float) -> tuple[float, float]:
    if "bid_px_1" in row and "ask_px_1" in row and row["bid_px_1"] > 0 and row["ask_px_1"] > 0:
        return float(row["bid_px_1"]), float(row["ask_px_1"])

    half_spread = row["spread_ticks"] * row["tick_size"] * 0.5 * spread_scale
    best_bid = row["current_mid"] - half_spread
    best_ask = row["current_mid"] + half_spread
    return float(best_bid), float(best_ask)


def _trade_slice(trade_tape: pd.DataFrame | None, start_ts: int, end_ts: int) -> pd.DataFrame:
    if trade_tape is None or trade_tape.empty:
        return pd.DataFrame(columns=["timestamp", "trade_price", "trade_volume"])
    mask = (trade_tape["timestamp"] > start_ts) & (trade_tape["timestamp"] <= end_ts)
    return trade_tape.loc[mask]


def _resting_fill(order: QuoteOrder, current_row: pd.Series, trade_tape: pd.DataFrame | None) -> bool:
    """
    極度關鍵的 HFT 保守填單演算法 (Queue-Aware Fill)
    用來判斷我們掛在交易所的限價單是否已經被真實成交。
    絕對不會因為價格碰到就假設成交，而是必須等前面的隊列 (Queue Ahead) 被消耗完。
    """
    # 擷取從「我們下單那一刻」到「現在這個 Tick」之間的所有真實市場成交紀錄
    trades = _trade_slice(trade_tape, order.placed_timestamp, int(current_row["timestamp"]))
    if trades.empty:
        return False # 如果期間內市場完全沒交易，我們的單絕對不可能成交

    # 計算這段時間內，跟我們同價位或穿透我們價位的 Taker 單總成交量
    if order.side > 0:
        # 如果我們掛買單 (Bid)，只有當實力賣單 (Trade Price <= 我們的買價) 砸下來時才會消耗隊列
        executable = trades.loc[trades["trade_price"] <= order.order_price, "trade_volume"].sum()
    else:
        # 如果我們掛賣單 (Ask)，只有當實力買單 (Trade Price >= 我們的賣價) 敲上去時才會消耗隊列
        executable = trades.loc[trades["trade_price"] >= order.order_price, "trade_volume"].sum()

    # 必須等市場真正的成交量 (executable) 大於「排在我們前面的單量 + 我們的單量」，才算真正成交
    return float(executable) >= float(order.queue_ahead + order.size)


def _mark_to_market_exit(
    inventory: InventoryState,
    row: pd.Series,
    reason: str,
    exit_index: int,
    spread_scale: float,
) -> dict:
    best_bid, best_ask = _best_quotes(row, spread_scale)
    tick_size = float(row["tick_size"])
    size = abs(inventory.quantity)
    if inventory.quantity > 0:
        exit_price = best_bid
        pnl_ticks = (exit_price - inventory.average_price) / tick_size * size
        close_side = "sell"
        position_side = "long"
    else:
        exit_price = best_ask
        pnl_ticks = (inventory.average_price - exit_price) / tick_size * size
        close_side = "buy"
        position_side = "short"

    return {
        "symbol": inventory.symbol,
        "position_side": position_side,
        "close_side": close_side,
        "entry_timestamp": inventory.entry_timestamp,
        "exit_timestamp": int(row["timestamp"]),
        "entry_price": inventory.average_price,
        "exit_price": float(exit_price),
        "size": float(size),
        "holding_steps": int(exit_index - inventory.entry_index),
        "pnl_ticks": float(pnl_ticks),
        "reason": reason,
    }


def _apply_fill(
    inventory: InventoryState,
    side: int,
    price: float,
    size: float,
    timestamp: int,
    index: int,
    tick_size: float,
    reason: str,
) -> list[dict]:
    realized_rows: list[dict] = []
    signed_fill = float(side) * float(size)

    if inventory.quantity == 0 or np.sign(inventory.quantity) == np.sign(signed_fill):
        total_size = abs(inventory.quantity) + abs(signed_fill)
        if total_size > 0:
            inventory.average_price = (
                abs(inventory.quantity) * inventory.average_price + abs(signed_fill) * price
            ) / total_size
        inventory.quantity += signed_fill
        if inventory.entry_timestamp == 0:
            inventory.entry_timestamp = int(timestamp)
            inventory.entry_index = int(index)
        inventory.max_abs_inventory = max(inventory.max_abs_inventory, abs(inventory.quantity))
        return realized_rows

    closing_size = min(abs(inventory.quantity), abs(signed_fill))
    if inventory.quantity > 0:
        pnl_ticks = (price - inventory.average_price) / tick_size * closing_size
        close_side = "sell"
        position_side = "long"
    else:
        pnl_ticks = (inventory.average_price - price) / tick_size * closing_size
        close_side = "buy"
        position_side = "short"

    realized_rows.append(
        {
            "symbol": inventory.symbol,
            "position_side": position_side,
            "close_side": close_side,
            "entry_timestamp": inventory.entry_timestamp,
            "exit_timestamp": int(timestamp),
            "entry_price": float(inventory.average_price),
            "exit_price": float(price),
            "size": float(closing_size),
            "holding_steps": int(index - inventory.entry_index),
            "pnl_ticks": float(pnl_ticks),
            "reason": reason,
        }
    )

    inventory.quantity += signed_fill
    if inventory.quantity == 0:
        inventory.average_price = 0.0
        inventory.entry_timestamp = 0
        inventory.entry_index = 0
        return realized_rows

    if np.sign(inventory.quantity) == np.sign(signed_fill):
        inventory.average_price = float(price)
        inventory.entry_timestamp = int(timestamp)
        inventory.entry_index = int(index)
        inventory.max_abs_inventory = max(inventory.max_abs_inventory, abs(inventory.quantity))

    return realized_rows


def _build_quote_target(
    row: pd.Series,
    symbol: str,
    inventory: InventoryState,
    spread_scale: float,
    score_threshold: float,
    one_sided_threshold: float,
    inventory_penalty_ticks: float,
    quote_step_ticks: float,
    max_inventory: int,
    regime_active: bool,
    end_time_ns: float,
    allowed_symbols: set[str] | None,
    quote_off_threshold: float,
    stationary_cutoff: float,
    alpha_coefficients: dict[str, float] | None,
    require_directional_gate: bool,
    force_directional_one_sided: bool,
    toxicity_threshold: float,
    one_sided_toxicity_threshold: float,
    inventory_widen_ticks: float,
    inventory_tighten_ticks: float,
    macro_trend_threshold: float,
) -> QuoteTarget:
    # Turn the current prediction row plus inventory state into a quote plan.
    # The result decides whether each side is active, where quotes sit, and how
    # much queue is assumed to be ahead at the best price.
    best_bid, best_ask = _best_quotes(row, spread_scale)
    tick_size = float(row["tick_size"])
    directional_score = _alpha_score_from_row(row, alpha_coefficients)
    bid_toxicity, ask_toxicity = _toxicity_scores_from_row(row, alpha_coefficients)
    inventory_penalty = (inventory.quantity / max(float(max_inventory), 1.0)) * inventory_penalty_ticks * 0.001
    alpha_score = directional_score - inventory_penalty
    timestamp = int(row["timestamp"])
    within_time = timestamp <= end_time_ns
    symbol_allowed = allowed_symbols is None or symbol in allowed_symbols
    stationary_prob = float(row.get("prob_stationary", 0.0))
    gated_signal = int(row.get("gated_signal", 1))
    macro_trend = float(row.get("macro_trend", 0.0))
    if not np.isfinite(macro_trend):
        macro_trend = 0.0
    quote_enabled = (
        regime_active
        and within_time
        and symbol_allowed
        and abs(alpha_score) >= quote_off_threshold
        and stationary_prob <= stationary_cutoff
        and (not require_directional_gate or gated_signal != 1)
    )
    if max(bid_toxicity, ask_toxicity) >= toxicity_threshold and abs(alpha_score) < one_sided_threshold:
        quote_enabled = False

    bid_active = quote_enabled and inventory.quantity < float(max_inventory)
    ask_active = quote_enabled and inventory.quantity > -float(max_inventory)
    bid_price = best_bid
    ask_price = best_ask

    inventory_abs = abs(float(inventory.quantity))
    if inventory_abs > 0 and tick_size > 0:
        widen = inventory_widen_ticks * inventory_abs * tick_size
        tighten = inventory_tighten_ticks * inventory_abs * tick_size
        if inventory.quantity > 0:
            bid_price = best_bid - widen
            ask_price = max(best_bid, best_ask - tighten)
        elif inventory.quantity < 0:
            bid_price = min(best_ask, best_bid + tighten)
            ask_price = best_ask + widen

    if quote_enabled and alpha_score > score_threshold:
        ask_price = best_ask + quote_step_ticks * tick_size
    elif quote_enabled and alpha_score < -score_threshold:
        bid_price = best_bid - quote_step_ticks * tick_size

    if quote_enabled and alpha_score >= one_sided_threshold:
        ask_active = inventory.quantity > 0
        ask_price = best_ask
    elif quote_enabled and alpha_score <= -one_sided_threshold:
        bid_active = inventory.quantity < 0
        bid_price = best_bid

    if quote_enabled and macro_trend <= -float(macro_trend_threshold):
        # In downtrend: do not open new longs, but allow passive bid to close shorts.
        bid_active = inventory.quantity < 0
    elif quote_enabled and macro_trend >= float(macro_trend_threshold):
        # In uptrend: do not open new shorts, but allow passive ask to close longs.
        ask_active = inventory.quantity > 0

    if quote_enabled and force_directional_one_sided:
        if gated_signal == 2:
            ask_active = inventory.quantity > 0
        elif gated_signal == 0:
            bid_active = inventory.quantity < 0

    if quote_enabled and bid_toxicity >= one_sided_toxicity_threshold:
        bid_active = False
        ask_active = inventory.quantity > -float(max_inventory)

    if quote_enabled and ask_toxicity >= one_sided_toxicity_threshold:
        ask_active = False
        bid_active = inventory.quantity < float(max_inventory)

    if bid_toxicity >= toxicity_threshold and ask_toxicity >= toxicity_threshold:
        bid_active = False
        ask_active = False

    return QuoteTarget(
        apply_index=0,
        bid_active=bid_active,
        ask_active=ask_active,
        bid_price=float(bid_price),
        ask_price=float(ask_price),
        bid_queue_ahead=float(max(row.get("bid_vol_1", 0.0), 0.0)) if bid_price == best_bid else 0.0,
        ask_queue_ahead=float(max(row.get("ask_vol_1", 0.0), 0.0)) if ask_price == best_ask else 0.0,
        alpha_score=float(alpha_score),
        bid_toxicity=float(bid_toxicity),
        ask_toxicity=float(ask_toxicity),
    )


def _apply_target(
    symbol: str,
    target: QuoteTarget,
    current_timestamp: int,
    idx: int,
    quote_size: float,
    min_quote_lifetime_steps: int,
    bid_order: QuoteOrder | None,
    ask_order: QuoteOrder | None,
) -> tuple[QuoteOrder | None, QuoteOrder | None, list[dict]]:
    quote_rows: list[dict] = []

    if target.bid_active:
        if bid_order is None or bid_order.order_price != target.bid_price or (
            idx - bid_order.placed_index >= max(min_quote_lifetime_steps, 1)
        ):
            bid_order = QuoteOrder(
                symbol=symbol,
                side=1,
                placed_timestamp=current_timestamp,
                placed_index=idx,
                order_price=target.bid_price,
                queue_ahead=target.bid_queue_ahead,
                filled_volume=0.0,
                size=quote_size,
                alpha_score=target.alpha_score,
            )
            quote_rows.append(
                {
                    "symbol": symbol,
                    "timestamp": current_timestamp,
                    "side": "bid",
                    "action": "place",
                    "order_price": target.bid_price,
                    "queue_ahead": target.bid_queue_ahead,
                    "alpha_score": target.alpha_score,
                    "bid_toxicity": target.bid_toxicity,
                    "ask_toxicity": target.ask_toxicity,
                }
            )
    elif bid_order is not None:
        quote_rows.append(
            {
                "symbol": symbol,
                "timestamp": current_timestamp,
                "side": "bid",
                "action": "cancel",
                "order_price": bid_order.order_price,
                "queue_ahead": bid_order.queue_ahead,
                "alpha_score": target.alpha_score,
                "bid_toxicity": target.bid_toxicity,
                "ask_toxicity": target.ask_toxicity,
            }
        )
        bid_order = None

    if target.ask_active:
        if ask_order is None or ask_order.order_price != target.ask_price or (
            idx - ask_order.placed_index >= max(min_quote_lifetime_steps, 1)
        ):
            ask_order = QuoteOrder(
                symbol=symbol,
                side=-1,
                placed_timestamp=current_timestamp,
                placed_index=idx,
                order_price=target.ask_price,
                queue_ahead=target.ask_queue_ahead,
                filled_volume=0.0,
                size=quote_size,
                alpha_score=target.alpha_score,
            )
            quote_rows.append(
                {
                    "symbol": symbol,
                    "timestamp": current_timestamp,
                    "side": "ask",
                    "action": "place",
                    "order_price": target.ask_price,
                    "queue_ahead": target.ask_queue_ahead,
                    "alpha_score": target.alpha_score,
                    "bid_toxicity": target.bid_toxicity,
                    "ask_toxicity": target.ask_toxicity,
                }
            )
    elif ask_order is not None:
        quote_rows.append(
            {
                "symbol": symbol,
                "timestamp": current_timestamp,
                "side": "ask",
                "action": "cancel",
                "order_price": ask_order.order_price,
                "queue_ahead": ask_order.queue_ahead,
                "alpha_score": target.alpha_score,
                "bid_toxicity": target.bid_toxicity,
                "ask_toxicity": target.ask_toxicity,
            }
        )
        ask_order = None

    return bid_order, ask_order, quote_rows


def run_hybrid_market_maker_backtest(
    predictions: pd.DataFrame,
    trade_tapes: dict[str, pd.DataFrame] | None,
    spread_scale: float = 1.0,
    quote_size: float = 1.0,
    max_inventory: int = 2,
    score_threshold: float = 0.002,
    one_sided_threshold: float = 0.006,
    macro_trend_threshold: float = 5.0,
    inventory_penalty_ticks: float = 0.5,
    quote_step_ticks: float = 1.0,
    max_holding_steps: int = 20,
    adverse_score_threshold: float = 0.010,
    latency_steps: int = 0,
    quote_confirmation_steps: int = 0,
    alpha_smoothing_window: int = 1,
    min_quote_lifetime_steps: int = 1,
    symbol_score_thresholds: dict[str, float] | None = None,
    symbol_one_sided_thresholds: dict[str, float] | None = None,
    symbol_inventory_penalties: dict[str, float] | None = None,
    symbol_max_inventory: dict[str, int] | None = None,
    symbol_end_times: dict[str, int] | None = None,
    quote_off_score_threshold: float = 0.0,
    stationary_probability_cutoff: float = 1.0,
    symbol_quote_off_thresholds: dict[str, float] | None = None,
    symbol_stationary_cutoffs: dict[str, float] | None = None,
    symbol_quote_confirmation_steps: dict[str, int] | None = None,
    symbol_alpha_smoothing_windows: dict[str, int] | None = None,
    require_directional_gate: bool = False,
    force_directional_one_sided: bool = False,
    alpha_coefficients: dict[str, float] | None = None,
    symbol_alpha_coefficients: dict[str, dict[str, float]] | None = None,
    toxicity_threshold: float = np.inf,
    one_sided_toxicity_threshold: float = np.inf,
    symbol_toxicity_thresholds: dict[str, float] | None = None,
    symbol_one_sided_toxicity_thresholds: dict[str, float] | None = None,
    toxicity_freeze_threshold: float = np.inf,
    alpha_freeze_threshold: float = np.inf,
    toxicity_freeze_steps: int = 0,
    symbol_toxicity_freeze_thresholds: dict[str, float] | None = None,
    symbol_alpha_freeze_thresholds: dict[str, float] | None = None,
    symbol_toxicity_freeze_steps: dict[str, int] | None = None,
    inventory_widen_ticks: float = 0.0,
    inventory_tighten_ticks: float = 0.0,
    symbol_inventory_widen_ticks: dict[str, float] | None = None,
    symbol_inventory_tighten_ticks: dict[str, float] | None = None,
    inventory_stop_loss_ticks: float | None = None,
    inventory_take_profit_ticks: float | None = None,
    inventory_flatten_score_threshold: float = 0.0,
    symbol_stop_loss_ticks: dict[str, float] | None = None,
    symbol_take_profit_ticks: dict[str, float] | None = None,
    symbol_alpha_decay_thresholds: dict[str, float] | None = None,
    regime_volatility_thresholds: dict[str, float] | None = None,
    regime_trade_intensity_thresholds: dict[str, float] | None = None,
    regime_min_visible_queue: dict[str, float] | None = None,
    regime_max_cancel_pressure: dict[str, float] | None = None,
    allowed_symbols: set[str] | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    trade_rows: list[dict] = []
    quote_rows: list[dict] = []
    fill_count = 0
    freeze_event_count = 0

    for symbol, symbol_df in predictions.sort_values(["symbol", "timestamp"]).groupby("symbol", sort=False):
        # Each symbol is replayed independently so latency, toxicity, and queue
        # effects can be measured without cross-symbol state leakage.
        inventory = InventoryState(symbol=symbol)
        bid_order: QuoteOrder | None = None
        ask_order: QuoteOrder | None = None
        pending_target: QuoteTarget | None = None
        freeze_remaining = 0
        trade_tape = None if trade_tapes is None else trade_tapes.get(symbol)
        ordered = symbol_df.reset_index(drop=True)
        smoothing_window = max(int(_symbol_value(symbol, alpha_smoothing_window, symbol_alpha_smoothing_windows)), 1)
        confirmation_steps = max(int(_symbol_value(symbol, quote_confirmation_steps, symbol_quote_confirmation_steps)), 0)
        symbol_toxicity_freeze_threshold_value = _symbol_value(
            symbol,
            toxicity_freeze_threshold,
            symbol_toxicity_freeze_thresholds,
        )
        symbol_alpha_freeze_threshold_value = _symbol_value(
            symbol,
            alpha_freeze_threshold,
            symbol_alpha_freeze_thresholds,
        )
        symbol_toxicity_freeze_step_count = max(
            int(_symbol_value(symbol, toxicity_freeze_steps, symbol_toxicity_freeze_steps)),
            0,
        )
        stop_loss_ticks = _symbol_value(symbol, inventory_stop_loss_ticks or 0.0, symbol_stop_loss_ticks)
        take_profit_ticks = _symbol_value(symbol, inventory_take_profit_ticks or 0.0, symbol_take_profit_ticks)
        alpha_decay_threshold = _symbol_value(
            symbol,
            inventory_flatten_score_threshold,
            symbol_alpha_decay_thresholds,
        )
        alpha_history: deque[float] = deque(maxlen=smoothing_window)
        symbol_alpha_coeffs = _symbol_alpha_coefficients(symbol, alpha_coefficients, symbol_alpha_coefficients)
        trade_ts: np.ndarray | None = None
        trade_px: np.ndarray | None = None
        trade_vol: np.ndarray | None = None
        trade_idx = 0
        if trade_tape is not None and not trade_tape.empty:
            trade_ordered = trade_tape.sort_values("timestamp").reset_index(drop=True)
            trade_ts = trade_ordered["timestamp"].to_numpy(dtype=np.int64, copy=False)
            trade_px = trade_ordered["trade_price"].to_numpy(dtype=np.float64, copy=False)
            trade_vol = trade_ordered["trade_volume"].to_numpy(dtype=np.float64, copy=False)

        for idx, row in ordered.iterrows():
            current_timestamp = int(row["timestamp"])
            raw_alpha_score = _alpha_score_from_row(row, symbol_alpha_coeffs)
            row_bid_toxicity, row_ask_toxicity = _toxicity_scores_from_row(row, symbol_alpha_coeffs)
            alpha_history.append(raw_alpha_score)
            smoothed_alpha_score = float(np.mean(alpha_history))

            # Apply previously scheduled quote changes only after the requested
            # latency and confirmation delay have passed.
            if pending_target is not None and pending_target.apply_index <= idx:
                bid_order, ask_order, new_quote_rows = _apply_target(
                    symbol=symbol,
                    target=pending_target,
                    current_timestamp=current_timestamp,
                    idx=idx,
                    quote_size=quote_size,
                    min_quote_lifetime_steps=min_quote_lifetime_steps,
                    bid_order=bid_order,
                    ask_order=ask_order,
                )
                quote_rows.extend(new_quote_rows)
                pending_target = None

            # -----------------------------------------------------------------
            # 毒性極速撤單保護機制 (Smart Canceling / Toxicity Freeze)
            # 這是本專案「第一性原理」的核心。當 TCN 嗅到危險時，零延遲撤單！
            # -----------------------------------------------------------------
            if (
                symbol_toxicity_freeze_step_count > 0 # 設定有開啟
                and (
                    # 條件1：傳統微觀毒性指標超標
                    max(row_bid_toxicity, row_ask_toxicity) >= symbol_toxicity_freeze_threshold_value
                    # 條件2：我們訓練的 TCN AI 模型表現出強烈的單邊偏好 (大於 Freeze 門檻)
                    or abs(smoothed_alpha_score) >= symbol_alpha_freeze_threshold_value
                )
                and freeze_remaining == 0 # 當前未處於凍結狀態
            ):
                # 啟動毒性凍結，設定接下來 N 個 ticks 禁止造市
                freeze_remaining = symbol_toxicity_freeze_step_count
                freeze_event_count += 1
                
                # 寫入 Action Log，動作為 "freeze" (代表立刻抽走所有單子)
                quote_rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": current_timestamp,
                        "side": "both",
                        "action": "freeze",
                        "order_price": 0.0,
                        "queue_ahead": 0.0,
                        "alpha_score": smoothed_alpha_score,
                        "bid_toxicity": row_bid_toxicity,
                        "ask_toxicity": row_ask_toxicity,
                    }
                )
                # 將實體記憶體中的單子直接設為 None (模擬零延遲 Withdraw)
                bid_order = None
                ask_order = None
                pending_target = None

            # Incrementally update fillable volumes using trade tape (O(n_trades + n_ticks)).
            if trade_ts is not None and trade_px is not None and trade_vol is not None:
                while trade_idx < len(trade_ts) and int(trade_ts[trade_idx]) <= current_timestamp:
                    ts = int(trade_ts[trade_idx])
                    px = float(trade_px[trade_idx])
                    vol = float(trade_vol[trade_idx])
                    if bid_order is not None and ts > bid_order.placed_timestamp and px <= bid_order.order_price:
                        bid_order.filled_volume += vol
                    if ask_order is not None and ts > ask_order.placed_timestamp and px >= ask_order.order_price:
                        ask_order.filled_volume += vol
                    trade_idx += 1

            if bid_order is not None and float(bid_order.filled_volume) >= float(bid_order.queue_ahead + bid_order.size):
                fill_count += 1
                trade_rows.extend(
                    _apply_fill(
                        inventory=inventory,
                        side=1,
                        price=bid_order.order_price,
                        size=bid_order.size,
                        timestamp=current_timestamp,
                        index=idx,
                        tick_size=float(row["tick_size"]),
                        reason="passive_bid_fill",
                    )
                )
                bid_order = None

            if ask_order is not None and float(ask_order.filled_volume) >= float(ask_order.queue_ahead + ask_order.size):
                fill_count += 1
                trade_rows.extend(
                    _apply_fill(
                        inventory=inventory,
                        side=-1,
                        price=ask_order.order_price,
                        size=ask_order.size,
                        timestamp=current_timestamp,
                        index=idx,
                        tick_size=float(row["tick_size"]),
                        reason="passive_ask_fill",
                    )
                )
                ask_order = None

            # -----------------------------------------------------------------
            # 持倉管理與出場邏輯 (Position Management & Mark-to-Market Exit)
            # 當我們手上有部位 (Inventory != 0) 時，持續監控停損停利與超時時間
            # -----------------------------------------------------------------
            if inventory.quantity != 0:
                # Once inventory exists, the simulator behaves more like a real
                # quoting engine and manages risk with holding, alpha, and pnl
                # exits before placing the next quote target.
                score = smoothed_alpha_score
                best_bid, best_ask = _best_quotes(row, spread_scale)
                
                # 計算未實現損益 (Unrealized PnL)，單位為 Ticks
                if inventory.quantity > 0:
                    # 如果做多 (Long)，用當前最佳買價 (Best Bid) 模擬市價平倉
                    unrealized_ticks = (best_bid - inventory.average_price) / float(row["tick_size"]) * abs(inventory.quantity)
                else:
                    # 如果做空 (Short)，用當前最佳賣價 (Best Ask) 模擬市價平倉
                    unrealized_ticks = (inventory.average_price - best_ask) / float(row["tick_size"]) * abs(inventory.quantity)

                # 條件 1：停利出場 (Take Profit)
                if take_profit_ticks > 0 and unrealized_ticks >= take_profit_ticks:
                    trade_rows.append(_mark_to_market_exit(inventory, row, "take_profit_exit", idx, spread_scale))
                    inventory = InventoryState(symbol=symbol, max_abs_inventory=inventory.max_abs_inventory)
                    bid_order = None; ask_order = None; pending_target = None
                    continue

                # 條件 2：停損出場 (Stop Loss) - 最關鍵的風險控制機制
                if stop_loss_ticks > 0 and unrealized_ticks <= -stop_loss_ticks:
                    trade_rows.append(_mark_to_market_exit(inventory, row, "stop_loss_exit", idx, spread_scale))
                    inventory = InventoryState(symbol=symbol, max_abs_inventory=inventory.max_abs_inventory)
                    bid_order = None; ask_order = None; pending_target = None
                    continue

                # 條件 3：超時期限出場 (Max Holding Steps) - 造市商不喜歡持倉過夜或過久
                if idx - inventory.entry_index >= max_holding_steps:
                    trade_rows.append(_mark_to_market_exit(inventory, row, "max_holding_steps", idx, spread_scale))
                    inventory = InventoryState(symbol=symbol, max_abs_inventory=inventory.max_abs_inventory)
                    bid_order = None; ask_order = None; pending_target = None
                    continue

                if inventory.quantity > 0 and score <= -adverse_score_threshold:
                    trade_rows.append(_mark_to_market_exit(inventory, row, "adverse_alpha_exit", idx, spread_scale))
                    inventory = InventoryState(symbol=symbol, max_abs_inventory=inventory.max_abs_inventory)
                    bid_order = None
                    ask_order = None
                    pending_target = None
                    continue

                if inventory.quantity < 0 and score >= adverse_score_threshold:
                    trade_rows.append(_mark_to_market_exit(inventory, row, "adverse_alpha_exit", idx, spread_scale))
                    inventory = InventoryState(symbol=symbol, max_abs_inventory=inventory.max_abs_inventory)
                    bid_order = None
                    ask_order = None
                    pending_target = None
                    continue

                if alpha_decay_threshold > 0 and abs(score) <= alpha_decay_threshold and idx > inventory.entry_index:
                    trade_rows.append(_mark_to_market_exit(inventory, row, "alpha_decay_exit", idx, spread_scale))
                    inventory = InventoryState(symbol=symbol, max_abs_inventory=inventory.max_abs_inventory)
                    bid_order = None
                    ask_order = None
                    pending_target = None
                    continue

            visible_queue = float(row.get("bid_vol_1", 0.0) + row.get("ask_vol_1", 0.0))
            rolling_vol = float(row.get("rolling_volatility", 0.0))
            trade_intensity = float(row.get("trade_intensity_short", 0.0))
            cancel_pressure = abs(float(row.get("cancel_pressure_short", 0.0)))
            end_time_ns = _symbol_value(symbol, np.inf, symbol_end_times)
            regime_active = (
                rolling_vol <= _symbol_value(symbol, np.inf, regime_volatility_thresholds)
                and trade_intensity <= _symbol_value(symbol, np.inf, regime_trade_intensity_thresholds)
                and visible_queue >= _symbol_value(symbol, 0.0, regime_min_visible_queue)
                and cancel_pressure <= _symbol_value(symbol, np.inf, regime_max_cancel_pressure)
            )
            if freeze_remaining > 0:
                regime_active = False
                freeze_remaining -= 1

            if current_timestamp > end_time_ns:
                bid_order = None
                ask_order = None
                if inventory.quantity != 0:
                    trade_rows.append(_mark_to_market_exit(inventory, row, "end_time_cutoff", idx, spread_scale))
                    inventory = InventoryState(symbol=symbol, max_abs_inventory=inventory.max_abs_inventory)
                continue

            target = _build_quote_target(
                row=row,
                symbol=symbol,
                inventory=inventory,
                spread_scale=spread_scale,
                score_threshold=_symbol_value(symbol, score_threshold, symbol_score_thresholds),
                one_sided_threshold=_symbol_value(symbol, one_sided_threshold, symbol_one_sided_thresholds),
                inventory_penalty_ticks=_symbol_value(symbol, inventory_penalty_ticks, symbol_inventory_penalties),
                quote_step_ticks=quote_step_ticks,
                max_inventory=int(_symbol_value(symbol, max_inventory, symbol_max_inventory)),
                regime_active=regime_active,
                end_time_ns=end_time_ns,
                allowed_symbols=allowed_symbols,
                quote_off_threshold=_symbol_value(symbol, quote_off_score_threshold, symbol_quote_off_thresholds),
                stationary_cutoff=_symbol_value(
                    symbol, stationary_probability_cutoff, symbol_stationary_cutoffs
                ),
                alpha_coefficients=symbol_alpha_coeffs,
                require_directional_gate=require_directional_gate,
                force_directional_one_sided=force_directional_one_sided,
                toxicity_threshold=_symbol_value(symbol, toxicity_threshold, symbol_toxicity_thresholds),
                one_sided_toxicity_threshold=_symbol_value(
                    symbol,
                    one_sided_toxicity_threshold,
                    symbol_one_sided_toxicity_thresholds,
                ),
                inventory_widen_ticks=_symbol_value(symbol, inventory_widen_ticks, symbol_inventory_widen_ticks),
                inventory_tighten_ticks=_symbol_value(
                    symbol,
                    inventory_tighten_ticks,
                    symbol_inventory_tighten_ticks,
                ),
                macro_trend_threshold=float(macro_trend_threshold),
            )
            target.alpha_score = smoothed_alpha_score
            if pending_target is None:
                target.apply_index = idx + max(int(latency_steps), 0) + confirmation_steps
                pending_target = target
            elif not _same_target(pending_target, target):
                target.apply_index = idx + max(int(latency_steps), 0) + confirmation_steps
                pending_target = target
            else:
                target.apply_index = pending_target.apply_index
                pending_target = target

            if pending_target is not None and pending_target.apply_index <= idx:
                bid_order, ask_order, new_quote_rows = _apply_target(
                    symbol=symbol,
                    target=pending_target,
                    current_timestamp=current_timestamp,
                    idx=idx,
                    quote_size=quote_size,
                    min_quote_lifetime_steps=min_quote_lifetime_steps,
                    bid_order=bid_order,
                    ask_order=ask_order,
                )
                quote_rows.extend(new_quote_rows)
                pending_target = None

        if not ordered.empty and inventory.quantity != 0:
            trade_rows.append(
                _mark_to_market_exit(inventory, ordered.iloc[-1], "end_of_stream", len(ordered) - 1, spread_scale)
            )

    trade_log = pd.DataFrame(trade_rows)
    quote_log = pd.DataFrame(quote_rows)
    if trade_log.empty:
        summary = {
            "realized_trades": 0,
            "fills": int(fill_count),
            "quote_updates": int(len(quote_log)),
            "toxicity_freeze_events": int(freeze_event_count),
            "net_pnl_ticks": 0.0,
            "avg_pnl_ticks": 0.0,
            "win_rate": 0.0,
            "avg_holding_steps": 0.0,
            "sharpe_like": 0.0,
        }
        by_symbol = pd.DataFrame(columns=["symbol", "realized_trades", "net_pnl_ticks", "avg_pnl_ticks"])
        return trade_log, summary, by_symbol, quote_log

    pnl_values = trade_log["pnl_ticks"].to_numpy(dtype=float)
    pnl_mean = float(pnl_values.mean())
    pnl_std = float(pnl_values.std(ddof=1)) if len(pnl_values) > 1 else 0.0
    sharpe_like = float((pnl_mean / pnl_std) * np.sqrt(len(pnl_values))) if pnl_std > 0 else 0.0

    summary = {
        "realized_trades": int(len(trade_log)),
        "fills": int(fill_count),
        "quote_updates": int(len(quote_log)),
        "toxicity_freeze_events": int(freeze_event_count),
        "net_pnl_ticks": float(trade_log["pnl_ticks"].sum()),
        "avg_pnl_ticks": pnl_mean,
        "win_rate": float((trade_log["pnl_ticks"] > 0).mean()),
        "avg_holding_steps": float(trade_log["holding_steps"].mean()),
        "sharpe_like": sharpe_like,
    }
    by_symbol = (
        trade_log.groupby("symbol", as_index=False)
        .agg(
            realized_trades=("symbol", "size"),
            net_pnl_ticks=("pnl_ticks", "sum"),
            avg_pnl_ticks=("pnl_ticks", "mean"),
        )
        .sort_values("net_pnl_ticks", ascending=False)
        .reset_index(drop=True)
    )
    return trade_log, summary, by_symbol, quote_log
