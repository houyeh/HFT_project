# bid_only_spread_ge_1 Unused DATA OOS Report

## 1. What Was Used For Training / Selection

`bid_only_spread_ge_1` is a non-ML rule, but it was selected in Phase 026 using validation metrics. Therefore this audit excludes both train sessions and the validation-selection session.

- `20260127Day`: split `test`, dataset `NEW_ALL`, role `not_training_or_selection`.
- `20260127Night`: split `test`, dataset `NEW_ALL`, role `not_training_or_selection`.
- `20260121Day`: split `train`, dataset `NEW_ALL`, role `training_context`.
- `20260122Day`: split `train`, dataset `NEW_ALL`, role `training_context`.
- `20260122Night`: split `train`, dataset `NEW_ALL`, role `training_context`.
- `20260123Day`: split `train`, dataset `NEW_ALL`, role `training_context`.
- `20260123Night`: split `train`, dataset `NEW_ALL`, role `training_context`.
- `20260126Day`: split `train`, dataset `NEW_ALL`, role `training_context`.
- `20260126Night`: split `val`, dataset `NEW_ALL`, role `validation_selection`.

## 2. DATA/NEW Through DATA/NEW4 Inclusion Audit

- `NEW` reason `excluded_train_split_used_by_016_026_context` included `False` files `23` bytes `3143755051`.
- `NEW` reason `excluded_validation_selection_for_bid_only_spread_ge_1` included `False` files `3` bytes `471859660`.
- `NEW` reason `included_unused_oos` included `True` files `8` bytes `945477582`.
- `NEW1` reason `excluded_train_split_used_by_016_026_context` included `False` files `7` bytes `949084042`.
- `NEW1` reason `excluded_validation_selection_for_bid_only_spread_ge_1` included `False` files `3` bytes `471859660`.
- `NEW1` reason `included_unused_oos` included `True` files `8` bytes `945477582`.
- `NEW2` reason `excluded_train_split_used_by_016_026_context` included `False` files `16` bytes `2194671009`.
- `NEW3` reason `included_unused_oos` included `True` files `15` bytes `1604298370`.
- `NEW4` reason `included_unused_oos` included `True` files `7` bytes `547457632`.

## 3. OOS Results

- Total evaluated action count: `12259`.
- Total selected quote count: `6067`.
- Total selected quote PnL: `37,836,440`.

- `NEW` `MXFB6` sessions `20260127Day,20260127Night`: actions `1387`, selected `682`, PnL `3,992,335`, precision `0.475`.
- `NEW` `TMFB6` sessions `20260127Day,20260127Night`: actions `1686`, selected `873`, PnL `4,874,985`, precision `0.438`.
- `NEW` `TXFB6` sessions `20260127Day,20260127Night`: actions `941`, selected `422`, PnL `2,624,355`, precision `0.441`.
- `NEW1` `MXFB6` sessions `20260127Day,20260127Night`: actions `1387`, selected `682`, PnL `4,174,225`, precision `0.482`.
- `NEW1` `TMFB6` sessions `20260127Day,20260127Night`: actions `1686`, selected `873`, PnL `5,665,815`, precision `0.478`.
- `NEW1` `TXFB6` sessions `20260127Day,20260127Night`: actions `941`, selected `422`, PnL `2,788,070`, precision `0.474`.
- `NEW3` `TXFA6` sessions `20260113,20260114,20260115,20260116,20260119Day,20260330`: actions `2925`, selected `1399`, PnL `7,367,885`, precision `0.303`.
- `NEW4` `TXFA6` sessions `20260102,20260105,20260109,20260112`: actions `1306`, selected `714`, PnL `6,348,770`, precision `0.363`.

## 4. Session/Side Highlights

Top selected PnL groups:
- `NEW3` `20260116` `bid`: selected PnL `4,478,390`, selected `371`.
- `NEW4` `20260102` `bid`: selected PnL `3,773,330`, selected `190`.
- `NEW3` `20260115` `bid`: selected PnL `3,356,150`, selected `236`.
- `NEW1` `20260127Night` `bid`: selected PnL `3,020,160`, selected `639`.
- `NEW` `20260127Night` `bid`: selected PnL `2,916,465`, selected `639`.
- `NEW1` `20260127Night` `bid`: selected PnL `2,842,485`, selected `579`.
- `NEW` `20260127Night` `bid`: selected PnL `2,775,575`, selected `579`.
- `NEW1` `20260127Day` `bid`: selected PnL `2,645,655`, selected `234`.
- `NEW4` `20260109` `bid`: selected PnL `2,082,470`, selected `207`.
- `NEW` `20260127Day` `bid`: selected PnL `1,958,520`, selected `234`.

Worst selected PnL groups:
- `NEW3` `20260113` `bid`: selected PnL `-1,520,870`, selected `301`.
- `NEW3` `20260114` `bid`: selected PnL `-728,710`, selected `228`.
- `NEW3` `20260330` `bid`: selected PnL `-154,930`, selected `81`.
- `NEW4` `20260105` `bid`: selected PnL `-33,740`, selected `44`.
- `NEW1` `20260127Day` `ask`: selected PnL `0`, selected `0`.
- `NEW1` `20260127Night` `ask`: selected PnL `0`, selected `0`.
- `NEW` `20260127Day` `ask`: selected PnL `0`, selected `0`.
- `NEW` `20260127Night` `ask`: selected PnL `0`, selected `0`.
- `NEW1` `20260127Day` `ask`: selected PnL `0`, selected `0`.
- `NEW1` `20260127Night` `ask`: selected PnL `0`, selected `0`.

## 5. Caveat

PnL remains the same simulator-derived moderate-stress quote proxy used in prior phases. This is not true exchange-level queue replay or real-world profitability proof.
