# NEW5 Holdout Usability Report

NEW5 was evaluated only with metadata and product counts. No PnL, score, policy, threshold, or model output was computed.

A NEW5 session is usable for the primary TXF final holdout only if `target_row_rate >= 0.5 and target_rows >= 50000` and malformed `info_time` rows are zero.

| source_group   | assigned_role        |   session_id |   files |   raw_rows |   target_rows |   target_valid_info_time_rows |   malformed_info_time_rows |   target_timestamp_min_us |   target_timestamp_max_us |   target_row_rate | usability_threshold                             | usable_for_final_holdout   | decision                                                           |
|:---------------|:---------------------|-------------:|--------:|-----------:|--------------:|------------------------------:|---------------------------:|--------------------------:|--------------------------:|------------------:|:------------------------------------------------|:---------------------------|:-------------------------------------------------------------------|
| NEW5           | final_holdout_locked |     20260128 |       2 |    1262091 |        848983 |                        848983 |                          0 |               2.89292e+10 |               4.95e+10    |        0.67268    | target_row_rate >= 0.5 and target_rows >= 50000 | True                       | usable_after_txf_filter_metadata_only_locked_until_final_one_shot  |
| NEW5           | final_holdout_locked |     20260518 |       1 |     141356 |           494 |                           494 |                          0 |               2.88152e+10 |               3.48151e+10 |        0.00349472 | target_row_rate >= 0.5 and target_rows >= 50000 | False                      | not_usable_for_primary_txf_final_holdout_due_to_sparse_target_rows |

Decision: use only NEW5 sessions marked `usable_for_final_holdout = True` after applying the frozen TXF product filter. Sessions with sparse TXF rows remain locked but are not part of the primary TXF final holdout.
