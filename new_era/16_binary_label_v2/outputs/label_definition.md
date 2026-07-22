# Binary Labels V2 Definition

## Inputs
- Quote-eligible events: `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\16_binary_label_v2\outputs\quote_eligible_events.parquet`
- Raw LOB/trade source for future proxy construction: `C:\Users\holbo\00VS_projjects\00_HFT_project\data\NEW_ALL`
- Symbol: `TXFB6`

## Fill Proxy
- Fill window: next 1 raw quote ticks within the same session.
- `bid_fill_proxy = bid_trade_hits_quote OR bid_price_cross_hits_quote`.
- `bid_trade_hits_quote`: a future trade in the fill window has `trade_price <= current_bid_px_1`.
- `bid_price_cross_hits_quote`: future best ask in the fill window has `ask_px_1 <= current_bid_px_1`.
- `ask_fill_proxy = ask_trade_hits_quote OR ask_price_cross_hits_quote`.
- `ask_trade_hits_quote`: a future trade in the fill window has `trade_price >= current_ask_px_1`.
- `ask_price_cross_hits_quote`: future best bid in the fill window has `bid_px_1 >= current_ask_px_1`.

This is a proxy because true aggressor-side and queue-position fills are not available.

## Toxicity
- Label horizon: 10 raw quote ticks within the same session.
- Adverse threshold: 0.5 ticks.
- `future_mid_return_H_ticks = (mid_price[t + H] - mid_price[t]) / tick_size`.
- `bid_toxic = 1` iff `bid_fill_proxy == True` and `future_mid_return_H_ticks <= -adverse_threshold_ticks`.
- `ask_toxic = 1` iff `ask_fill_proxy == True` and `future_mid_return_H_ticks >= adverse_threshold_ticks`.

## Leakage Control
- Labels use future information by definition, but only as targets.
- Model features come from `quote_eligible_events.parquet`, whose z-scores were computed with past-only rolling statistics.
- No threshold selection or model tuning is performed in this script.