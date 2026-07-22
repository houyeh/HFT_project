# TXF-only Action Universe Report

Phase 04B rebuilds the action universe from raw CSVs using the frozen Phase 04A.2 TXF product filter `^TXF[A-Z0-9]*$` and the frozen malformed `info_time` drop-and-log loader contract.

Dataset role mapping: `NEW2 -> train`, `NEW1 -> val`, `NEW3 -> new3 diagnostic OOS`, `NEW4 -> new4 diagnostic OOS`. `NEW5` is not loaded.

Each eligible L2 state emits exactly three candidate actions: `bid`, `ask`, and `no_quote`. Labels are H20 future-mid markout labels; `excess_pnl_H20` removes the mechanical half-spread component and remains the main target.

## Counts by Split

| split   |   sessions |   source_rows |   action_rows |
|:--------|-----------:|--------------:|--------------:|
| new3    |          6 |       5656822 |      16970466 |
| new4    |          4 |       2196280 |       6588840 |
| train   |          3 |       2470559 |       7411677 |
| val     |          6 |       2655369 |       7966107 |

## Loader Notes

Malformed `info_time` rows dropped and logged: `2`.
Boundary rule: drop first `100` and last `100` events within each session/product stream.

Output dataset: `data/txf_only_action_universe.parquet/`.
