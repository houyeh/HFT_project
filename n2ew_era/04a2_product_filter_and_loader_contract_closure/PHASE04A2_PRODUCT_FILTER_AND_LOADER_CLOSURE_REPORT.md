# Phase 04A.2 Product Filter and Loader Contract Closure Report

Final Status: `clean_protocol_freeze_ready`.

No ML training, strategy tuning, PnL evaluation, model scoring, or threshold evaluation was performed on NEW5.

## Product Universe Frozen

Primary final product universe is TXF-family only: `^TXF[A-Z0-9]*$`. MXF/TMF are not pooled with TXF unless product-specific normalization is implemented and reported separately.

| contract_item                  | decision                                      | regex                    | allowed_in_final_evaluation   | reason                                                                                                                                                    |
|:-------------------------------|:----------------------------------------------|:-------------------------|:------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------|
| primary_final_product_universe | TXF family only                               | ^TXF[A-Z0-9]*$           | True                          | Inherited proxy point_value=200 and maker_fee=15 are TXF-compatible project assumptions; pooling MXF/TMF without normalization would mix contract scales. |
| legacy_broad_target_universe   | TXF/MXF/TMF diagnostic only unless normalized | ^(TXF|MXF|TMF)[A-Z0-9]*$ | False                         | MXF and TMF are not assumed comparable to TXF in point value, tick value, or fee. They require separate metrics or product-normalized PnL.                |
| raw_mixed_product_stream       | forbidden for pooled final evaluation         | all product_id values    | False                         | Raw files contain hundreds of product IDs; unfiltered evaluation would mix unrelated markets.                                                             |

## Target Product Filter Impact

| source_group   | assigned_role        |   files |   sessions |   raw_rows |   target_rows |   target_valid_rows |   malformed_info_time_rows |   target_valid_rate |
|:---------------|:---------------------|--------:|-----------:|-----------:|--------------:|--------------------:|---------------------------:|--------------------:|
| NEW1           | development          |      17 |          6 |    9956601 |       2727797 |             2727797 |                          1 |            0.273969 |
| NEW2           | development          |      16 |          3 |   10035946 |       2544915 |             2544915 |                          0 |            0.25358  |
| NEW3           | used_oos_diagnostic  |      15 |          7 |    8243070 |       5854187 |             5854187 |                          1 |            0.710195 |
| NEW4           | used_oos_diagnostic  |       7 |          4 |    2944345 |       2280575 |             2280575 |                          0 |            0.774561 |
| NEW5           | final_holdout_locked |       3 |          2 |    1403447 |        849477 |              849477 |                          0 |            0.605279 |

## NEW5 Final Usable State

Usable NEW5 sessions for primary TXF final holdout: `1/2`.

| source_group   | assigned_role        |   session_id |   files |   raw_rows |   target_rows |   target_valid_info_time_rows |   malformed_info_time_rows |   target_timestamp_min_us |   target_timestamp_max_us |   target_row_rate | usability_threshold                             | usable_for_final_holdout   | decision                                                           |
|:---------------|:---------------------|-------------:|--------:|-----------:|--------------:|------------------------------:|---------------------------:|--------------------------:|--------------------------:|------------------:|:------------------------------------------------|:---------------------------|:-------------------------------------------------------------------|
| NEW5           | final_holdout_locked |     20260128 |       2 |    1262091 |        848983 |                        848983 |                          0 |               2.89292e+10 |               4.95e+10    |        0.67268    | target_row_rate >= 0.5 and target_rows >= 50000 | True                       | usable_after_txf_filter_metadata_only_locked_until_final_one_shot  |
| NEW5           | final_holdout_locked |     20260518 |       1 |     141356 |           494 |                           494 |                          0 |               2.88152e+10 |               3.48151e+10 |        0.00349472 | target_row_rate >= 0.5 and target_rows >= 50000 | False                      | not_usable_for_primary_txf_final_holdout_due_to_sparse_target_rows |

NEW5 sessions marked unusable are not discarded from disk; they remain locked metadata-only files and are excluded from the primary TXF final holdout because their target TXF row rate is too sparse.

## Raw Loader Contract Frozen

Malformed `info_time` rows must be dropped before feature or label construction and logged. Silent timestamp coercion is forbidden.

| rule_id    | rule                                                                                          | implementation_requirement                                                                | silent_coercion_allowed   | status   |
|:-----------|:----------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|:--------------------------|:---------|
| loader_001 | Malformed info_time rows MUST be dropped before feature or label construction.                | Log physical_path, session_id, row locator, raw info_time, and product_id when available. | False                     | FROZEN   |
| loader_002 | Rows must pass the frozen primary product filter before final target-universe evaluation.     | product_id must match ^TXF[A-Z0-9]*$                                                      | False                     | FROZEN   |
| loader_003 | Valid info_time parser interprets hour:minute:second:millisecond:flag and requires flag == 0. | Non-zero or missing fifth segment is malformed until separately audited.                  | False                     | FROZEN   |

## Closure Decision

The product universe, target filter, NEW5 usability rule, and malformed-row loader behavior are now explicit and machine-readable. Product mismatch ambiguity is closed by narrowing the primary final universe to TXF. Parser ambiguity is closed by the strict drop-and-log loader contract.
