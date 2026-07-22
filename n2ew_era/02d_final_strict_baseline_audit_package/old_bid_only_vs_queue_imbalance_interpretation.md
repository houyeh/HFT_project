# Old Bid-Only vs Queue-Imbalance Interpretation

The selected policy `side_by_queue_imbalance_abs_0p1` at `H20` is **not** an absolute PnL replacement for `old_bid_only_spread_ge_1`.

Old bid-only still has higher absolute total PnL in the current proxy markout setup because it quotes more broadly and captures mechanical spread/markout value across the full bid-side universe.

The queue-imbalance rule matters because it passes stricter non-triviality tests: it has `selection_rate < 0.95`, beats matched random controls, and shows positive excess versus scaled select-all controls. That makes it evidence of an L2 ranking signal, not a full strategy replacement.

Advisor-safe wording: use `non-trivial L2 quote-ranking signal` or `candidate quote-control primitive`; avoid `dominates old baseline` or `proves profitability`.
