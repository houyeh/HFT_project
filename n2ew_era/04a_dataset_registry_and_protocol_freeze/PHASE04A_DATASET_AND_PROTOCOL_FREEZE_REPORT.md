# Phase 04A Dataset and Protocol Freeze Report

Final Status: `dataset_protocol_frozen_ready_for_final_development_rerun`.

## 1. True Physical Dataset After Dedup

The true TW L2 physical dataset excludes the `NEW` junction alias and the `NEW_ALL` aggregate duplicate. Exact SHA256 deduplication over `NEW1` through `NEW5` leaves 58 unique CSV files.

| source_group   | assigned_role        |   unique_files |   sessions |   row_count |   file_size_bytes |
|:---------------|:---------------------|---------------:|-----------:|------------:|------------------:|
| NEW1           | development          |             17 |          6 |     9956599 |        2209134819 |
| NEW2           | development          |             16 |          3 |    10035946 |        2194671009 |
| NEW3           | used_oos_diagnostic  |             15 |          7 |     8243068 |        1604298370 |
| NEW4           | used_oos_diagnostic  |              7 |          4 |     2944344 |         547457632 |
| NEW5           | final_holdout_locked |              3 |          2 |     1403447 |         253498322 |

Total unique CSV files: `58`.
Total raw rows: `32583404`.

## 2. Frozen Roles

- `NEW1` + `NEW2`: `development`.
- `NEW3` + `NEW4`: `used_oos_diagnostic`.
- `NEW5`: `final_holdout_locked`.

## 3. Frozen Label Hierarchy

Diagnostic-only labels are `raw_pnl_H` and toxic labels. The main signal target is `excess_pnl_H20`. Strategy evaluation uses causal-fill sequential excess PnL.

## 4. Frozen Evaluation Protocol

Selection is allowed only in the development pool. `NEW3`/`NEW4` are diagnostic OOS. `NEW5` is one-shot final holdout only after final development is complete.

## 5. NEW5 Lock

NEW5 was checked only for metadata, hashes, row counts, schema, product samples, and timestamp range. No PnL or policy scoring was run.
