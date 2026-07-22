# Dedup Validation Report

Deduplication was performed by exact SHA256 over physical CSV files in `NEW1`, `NEW2`, `NEW3`, `NEW4`, and `NEW5`. `NEW` is treated as an alias/junction layer and `NEW_ALL` as an aggregate duplicate, not as additional data.

## Canonical Dataset Summary

| source_group   | assigned_role        |   unique_files |   sessions |   row_count |   file_size_bytes |
|:---------------|:---------------------|---------------:|-----------:|------------:|------------------:|
| NEW1           | development          |             17 |          6 |     9956599 |        2209134819 |
| NEW2           | development          |             16 |          3 |    10035946 |        2194671009 |
| NEW3           | used_oos_diagnostic  |             15 |          7 |     8243068 |        1604298370 |
| NEW4           | used_oos_diagnostic  |              7 |          4 |     2944344 |         547457632 |
| NEW5           | final_holdout_locked |              3 |          2 |     1403447 |         253498322 |

## Alias Audit

| alias_group   | path_exists   |   child_dirs_resolving_to_other_paths | alias_or_aggregate_interpretation   |   csv_files_seen |   unique_hashes_seen |   new_unique_hashes_not_in_canonical_registry | alias_contributes_new_data   |
|:--------------|:--------------|--------------------------------------:|:------------------------------------|-----------------:|---------------------:|----------------------------------------------:|:-----------------------------|
| NEW           | True          |                                     9 | junction_alias                      |               34 |                   33 |                                             0 | False                        |
| NEW_ALL       | True          |                                     0 | aggregate_duplicate_or_regular_dir  |               34 |                   33 |                                             0 | False                        |

## Internal Duplicate Resolved

| sha256                                                           | source_group   | session_id    | physical_path                                                                                         | canonical_path                                                                                        | is_canonical   |
|:-----------------------------------------------------------------|:---------------|:--------------|:------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|:---------------|
| b5c7081bef4dac7b6860c70c7e028783646cab846580504ed03abcec34590d48 | NEW1           | 20260127Night | C:\Users\holbo\00VS_projjects\00_HFT_project\data\NEW1\20260127Night\my_orderbook_20260331_9002_2.csv | C:\Users\holbo\00VS_projjects\00_HFT_project\data\NEW1\20260127Night\my_orderbook_20260331_9002_2.csv | True           |
| b5c7081bef4dac7b6860c70c7e028783646cab846580504ed03abcec34590d48 | NEW1           | 20260127Night | C:\Users\holbo\00VS_projjects\00_HFT_project\data\NEW1\20260127Night\my_orderbook_20260331_9004_2.csv | C:\Users\holbo\00VS_projjects\00_HFT_project\data\NEW1\20260127Night\my_orderbook_20260331_9002_2.csv | False          |

Final canonical unique CSV count: `58`.
Final canonical raw row count: `32583404`.
