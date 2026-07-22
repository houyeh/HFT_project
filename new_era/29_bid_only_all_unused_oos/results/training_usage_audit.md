# Training/Selection Data Usage Audit

`bid_only_spread_ge_1` is a non-ML rule, but the rule was selected in Phase 026 on validation. For strict OOS testing, both train sessions and validation-selection sessions are excluded.

## Sessions Used Upstream

- `20260127Day` split `test` dataset `NEW_ALL` symbol `TXFB6` rows `410586` role `not_training_or_selection`.
- `20260127Night` split `test` dataset `NEW_ALL` symbol `TXFB6` rows `755735` role `not_training_or_selection`.
- `20260121Day` split `train` dataset `NEW_ALL` symbol `TXFB6` rows `705812` role `training_context`.
- `20260122Day` split `train` dataset `NEW_ALL` symbol `TXFB6` rows `944608` role `training_context`.
- `20260122Night` split `train` dataset `NEW_ALL` symbol `TXFB6` rows `820409` role `training_context`.
- `20260123Day` split `train` dataset `NEW_ALL` symbol `TXFB6` rows `182203` role `training_context`.
- `20260123Night` split `train` dataset `NEW_ALL` symbol `TXFB6` rows `565974` role `training_context`.
- `20260126Day` split `train` dataset `NEW_ALL` symbol `TXFB6` rows `373758` role `training_context`.
- `20260126Night` split `val` dataset `NEW_ALL` symbol `TXFB6` rows `563477` role `validation_selection`.

## Exclusion Rule

- Exclude `20260121Day`: `excluded_train_split_used_by_016_026_context`.
- Exclude `20260122Day`: `excluded_train_split_used_by_016_026_context`.
- Exclude `20260122Night`: `excluded_train_split_used_by_016_026_context`.
- Exclude `20260123Day`: `excluded_train_split_used_by_016_026_context`.
- Exclude `20260123Night`: `excluded_train_split_used_by_016_026_context`.
- Exclude `20260126Day`: `excluded_train_split_used_by_016_026_context`.
- Exclude `20260126Night`: `excluded_validation_selection_for_bid_only_spread_ge_1`.

All other sessions under `data/NEW` through `data/NEW4` are treated as unused OOS for this audit.
