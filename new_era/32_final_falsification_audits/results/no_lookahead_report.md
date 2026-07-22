# No-Lookahead Audit

Decision rule audited: quote only when `side == bid` and decision-book `spread_ticks >= 1`.

- Source function: `baseline_utils.apply_non_ml_policy`.
- Future/EV token found in decision source: `False`.
- Tokens found: `none`.

## Timing Results

- `new4` `current_row`: PnL `6348770`, selected `714`, boundary bad decisions `0`.
- `new4` `1_row_lagged`: PnL `6348770`, selected `714`, boundary bad decisions `0`.
- `new4` `2_row_lagged`: PnL `6348770`, selected `714`, boundary bad decisions `0`.
- `test` `current_row`: PnL `2312590`, selected `419`, boundary bad decisions `0`.
- `test` `1_row_lagged`: PnL `2312590`, selected `419`, boundary bad decisions `0`.
- `test` `2_row_lagged`: PnL `2312590`, selected `419`, boundary bad decisions `0`.

Current-row decision should match the stored baseline. Lagged rows are a deterministic delay/timing stress, not a reselected policy.
