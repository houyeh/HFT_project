# Fill Evidence Audit

## Available LOB Columns

### NEW_ALL / TXFB6
Columns:
timestamp, session_id, bid_px_1, bid_vol_1, bid_px_2, bid_vol_2, bid_px_3, bid_vol_3, bid_px_4, bid_vol_4, bid_px_5, bid_vol_5, ask_px_1, ask_vol_1, ask_px_2, ask_vol_2, ask_px_3, ask_vol_3, ask_px_4, ask_vol_4, ask_px_5, ask_vol_5

Dtypes:
timestamp       int64
session_id     object
bid_px_1      float64
bid_vol_1       int64
bid_px_2      float64
bid_vol_2       int64
bid_px_3      float64
bid_vol_3       int64
bid_px_4      float64
bid_vol_4       int64
bid_px_5      float64
bid_vol_5       int64
ask_px_1      float64
ask_vol_1       int64
ask_px_2      float64
ask_vol_2       int64
ask_px_3      float64
ask_vol_3       int64
ask_px_4      float64
ask_vol_4       int64
ask_px_5      float64
ask_vol_5       int64

### NEW3 / TXFA6
Columns:
timestamp, session_id, bid_px_1, bid_vol_1, bid_px_2, bid_vol_2, bid_px_3, bid_vol_3, bid_px_4, bid_vol_4, bid_px_5, bid_vol_5, ask_px_1, ask_vol_1, ask_px_2, ask_vol_2, ask_px_3, ask_vol_3, ask_px_4, ask_vol_4, ask_px_5, ask_vol_5

Dtypes:
timestamp       int64
session_id     object
bid_px_1      float64
bid_vol_1       int64
bid_px_2      float64
bid_vol_2       int64
bid_px_3      float64
bid_vol_3       int64
bid_px_4      float64
bid_vol_4       int64
bid_px_5      float64
bid_vol_5       int64
ask_px_1      float64
ask_vol_1       int64
ask_px_2      float64
ask_vol_2       int64
ask_px_3      float64
ask_vol_3       int64
ask_px_4      float64
ask_vol_4       int64
ask_px_5      float64
ask_vol_5       int64

## Available Trade Columns

### NEW_ALL / TXFB6
Columns:
timestamp, session_id, trade_price, trade_volume

Dtypes:
timestamp         int64
session_id       object
trade_price     float64
trade_volume      int64

### NEW3 / TXFA6
Columns:
timestamp, session_id, trade_price, trade_volume

Dtypes:
timestamp         int64
session_id       object
trade_price     float64
trade_volume      int64

## Fill Evidence Availability

- TRADE_TAPE_AVAILABLE: YES
- AGGRESSOR_SIDE_AVAILABLE: NO
- TRADE_DIRECTION_AVAILABLE: NO
- SIZE_DEPLETION_AVAILABLE: YES
- MULTI_LEVEL_DEPTH_AVAILABLE: YES
- ONLY_TOP_OF_BOOK_AVAILABLE: NO

## Conclusion

True exchange-level fill direction is not available because aggressor_side is missing.
This task uses LOB-based fill proxies, not true fills.

## Usable Proxies

Use in this audit:
1. queue_depletion_proxy
2. price_level_removed_proxy
3. price_through_proxy

Not used for fill truth:
- trade_tape_fill: not reliable due to missing aggressor_side
- touch_or_move_proxy: risk proxy only, not fill proxy