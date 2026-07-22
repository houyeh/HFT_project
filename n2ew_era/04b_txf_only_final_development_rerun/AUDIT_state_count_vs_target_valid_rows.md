status: resolved
# Audit: `state_count` vs `target_valid_rows`

The apparent mismatch is due to denominator changes, not an unexplained row loss.

| split   |   target_valid_rows |   raw_rows |   raw_txf_rows |   valid_book_rows |   boundary_kept_rows |   action_rows_from_filter |   sessions |   source_rows |   action_rows |   phc_window_state_count |   phc_event_rows |   target_valid_minus_raw_txf | state_count_equals_boundary_kept   | action_rows_equals_state_x3   |   boundary_drop_from_target_valid |   phc_window_retention_vs_state |
|:--------|--------------------:|-----------:|---------------:|------------------:|---------------------:|--------------------------:|-----------:|--------------:|--------------:|-------------------------:|-----------------:|-----------------------------:|:-----------------------------------|:------------------------------|----------------------------------:|--------------------------------:|
| new3    |             5854187 |    8243070 |        5854187 |           5658022 |              5656822 |                  16970466 |          6 |       5656822 |      16970466 |                   176754 |          5656822 |                            0 | True                               | True                          |                            197365 |                       0.0312462 |
| new4    |             2280575 |    2944345 |        2280575 |           2197080 |              2196280 |                   6588840 |          4 |       2196280 |       6588840 |                    68620 |          2196280 |                            0 | True                               | True                          |                             84295 |                       0.0312437 |
| train   |             2544915 |   10035946 |        2544915 |           2471159 |              2470559 |                   7411677 |          3 |       2470559 |       7411677 |                    77194 |          2470559 |                            0 | True                               | True                          |                             74356 |                       0.0312456 |
| val     |             2727797 |    9956601 |        2727797 |           2656569 |              2655369 |                   7966107 |          6 |       2655369 |       7966107 |                    82960 |          2655369 |                            0 | True                               | True                          |                             72428 |                       0.0312424 |

Interpretation:

- `target_valid_rows` from 04A.2 equals 04B `raw_txf_rows` exactly for NEW1/NEW2/NEW3/NEW4.
- 04B `source_rows` is after valid-book and first/last 100-tick boundary filtering; it equals `boundary_kept_rows`.
- Full action `action_rows = source_rows * 3` because Phase 04B builds bid/ask/no_quote actions per state.
- PHC `state_count` is smaller again because 05A uses sequence terminal windows (`seq_len=128`, `stride=32`), not dense event rows.
