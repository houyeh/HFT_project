# Raw Data Capability Report

This audit defines what can and cannot be simulated from the available order-book snapshots.

- Historical sample header file: `data\NEW\20260121Day\my_orderbook_20260513_21295_1.csv`.
- NEW4 sample header file: `data\NEW4\20260102\my_orderbook_20260102_1.csv`.
- Canonical loader: `new_era/models_playaround_15/data_loading.py::load_new_tw_data`.

## Capability Table

- `timestamp/session/symbol`: `available`. Chronological replay and session-reset audits are possible.
- `L1-L5 bid/ask price and depth`: `available`. Spread and depth-aware quote eligibility can be simulated as a proxy.
- `last trade price and quantity`: `available`. Trade-hit proxy can be constructed, but not aggressor-side truth.
- `order_id`: `unavailable`. Cannot reconstruct exact queue position or order-level fills.
- `aggressor_side`: `unavailable`. Fill toxicity remains a proxy, not true exchange-side fill toxicity.
- `exchange queue position`: `unavailable`. Queue-aware stress tests are scenario/proxy analyses, not production fill replay.

## Conservative Interpretation

The dataset supports spread/depth/event-rate proxy controls. It does not support exact exchange queue reconstruction, true order-level fill priority, or aggressor-side-labeled toxicity. All queue/fill PnL remains a simulator-derived proxy.
