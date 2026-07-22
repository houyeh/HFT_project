# Filtered Target Product Registry Report

This registry represents the final metadata-only target-product universe after the frozen TXF product filter and strict `info_time` validity rule. It does not compute PnL, model scores, future labels, or thresholds.

| source_group   | assigned_role        |   files |   sessions |   raw_rows |   target_valid_info_time_rows |   malformed_info_time_rows |   target_valid_rate |
|:---------------|:---------------------|--------:|-----------:|-----------:|------------------------------:|---------------------------:|--------------------:|
| NEW1           | development          |      17 |          6 |    9956601 |                       2727797 |                          1 |            0.273969 |
| NEW2           | development          |      16 |          3 |   10035946 |                       2544915 |                          0 |            0.25358  |
| NEW3           | used_oos_diagnostic  |      15 |          7 |    8243070 |                       5854187 |                          1 |            0.710195 |
| NEW4           | used_oos_diagnostic  |       7 |          4 |    2944345 |                       2280575 |                          0 |            0.774561 |
| NEW5           | final_holdout_locked |       3 |          2 |    1403447 |                        849477 |                          0 |            0.605279 |

Full file-level registry: `results/filtered_target_product_registry_summary.csv`.
