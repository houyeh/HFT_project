# Binary Label Definition Report

## Why old binary labels failed
- toxic_bid/ask_binary was not fill-conditional.
- adverse_bid/ask_fill_binary used immediate-cross fill proxy and became too sparse.
- active_buy/sell labels target taker alpha, not maker cancel risk.

## New definition
- side-specific fill-conditional quote-risk labels.
- primary proxy: price_level_removed (Proxy C).
- primary setting: fill_window=1, horizon=10, toxic if markout_ticks < -0.5.
- proxy limitation: level removed can be fill or cancel.
- distribution-shift limitation: training conditioned on proxy-filled events.

### bid toxic rates (primary)
- train: samples=15,643, fill_proxy_rate=0.021770, toxic_rate=0.035863
- val: samples=12,603, fill_proxy_rate=0.022366, toxic_rate=0.032294
- test: samples=24,329, fill_proxy_rate=0.020860, toxic_rate=0.027704
- new3: samples=119,809, fill_proxy_rate=0.021178, toxic_rate=0.031684

### ask toxic rates (primary)
- train: samples=15,856, fill_proxy_rate=0.022066, toxic_rate=0.030777
- val: samples=13,314, fill_proxy_rate=0.023628, toxic_rate=0.026889
- test: samples=25,037, fill_proxy_rate=0.021467, toxic_rate=0.026641
- new3: samples=119,183, fill_proxy_rate=0.021067, toxic_rate=0.032840
