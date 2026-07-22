# Queue Data Audit

## Dataset Summary
- NEW source: `C:\Users\holbo\00VS_projjects\00_HFT_project\data\NEW_ALL` (requested symbol `TXFB6`)
- NEW3 source: `C:\Users\holbo\00VS_projjects\00_HFT_project\data\NEW3` (requested symbol `TXFA6`)
- NEW sampled rows scanned: 32,000
- NEW3 sampled rows scanned: 32,000

## Field Availability
- LOB depth available? YES
- Best level volume available? YES
- Trade tape available? YES
- Aggressor side available? NO (likely)

## Queue Reconstruction Feasibility
- Can true exchange-level queue be reconstructed? NO (expected).
- True queue replay is not possible. This task uses conservative queue-aware sensitivity proxies.

## Proxy Assumptions Used in 018
- Fill probability haircut scenarios with fixed random seeds.
- Profitable-fill haircut stronger than toxic-fill haircut (adverse selection stress).
- Best-level queue-ahead approximation using `bid_vol_1` / `ask_vol_1`.
- Trade-tape volume in future event windows is used as a conservative proxy for queue depletion when aggressor side is unavailable.
- Cancel-delay sensitivity combined with conservative fill assumptions.

## Caveat
- Without aggressor_side and full per-order queue events, outputs are sensitivity tests, not exchange-faithful queue replay.

## Raw Header Sample (NEW)
- ['info_time', 'channel_id', 'channel_seq', 'product_id', 'msg_type', 'index', 'last_trade_price', 'last_trade_qty', 'total_match_qty', 'match_time_dif', 'der_buy_qty', 'der_buy_price', '5th_buy_qty', '5th_buy_price', '4th_buy_qty', '4th_buy_price', '3th_buy_qty', '3th_buy_price', '2th_buy_qty', '2th_buy_price']
## Raw Header Sample (NEW3)
- ['info_time', 'channel_id', 'channel_seq', 'product_id', 'msg_type', 'index', 'last_trade_price', 'last_trade_qty', 'total_match_qty', 'match_time_dif', 'der_buy_qty', 'der_buy_price', '5th_buy_qty', '5th_buy_price', '4th_buy_qty', '4th_buy_price', '3th_buy_qty', '3th_buy_price', '2th_buy_qty', '2th_buy_price']