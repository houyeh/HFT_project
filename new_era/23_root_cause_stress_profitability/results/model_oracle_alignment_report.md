# Model-Oracle Alignment Report

Comparison uses action-level selected quotes versus `oracle_positive_quote` under moderate stress.
For cancel/block policies, `selected_quote` is the action that survives the cancel mask.
`spread_only` is implemented as a validation-selected raw `spread_ticks` gate because the 022 sklearn pickle is version-incompatible in this runtime; test is not used for that threshold.

- Best oracle recall: `015` at `0.9072`.
- Best selected quote PnL: `spread_only` at `-577930.000`.

Required fields are included in every split table: oracle recall, oracle precision, captured oracle PnL, missed oracle PnL, false positive loss, avoided bad PnL, and missed good PnL.
