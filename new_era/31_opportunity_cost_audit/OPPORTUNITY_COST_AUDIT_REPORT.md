# Opportunity Cost Audit Report

## 1. Purpose

Phase 031 tests whether the core `bid_only_spread_ge_1` baseline is too conservative and whether ML is more appropriate as a full-action quote selector rather than a residual veto.

## 2. Full Action Universe

- `new4` `ask`: actions `592`, baseline-selected `0`, positive-EV actions `77`, total EV `-6,912,720`.
- `new4` `bid`: actions `714`, baseline-selected `714`, positive-EV actions `259`, total EV `6,348,770`.
- `test` `ask`: actions `515`, baseline-selected `0`, positive-EV actions `6`, total EV `-3,437,810`.
- `test` `bid`: actions `419`, baseline-selected `419`, positive-EV actions `188`, total EV `2,312,590`.
- `train` `ask`: actions `906`, baseline-selected `0`, positive-EV actions `240`, total EV `2,053,810`.
- `train` `bid`: actions `820`, baseline-selected `820`, positive-EV actions `179`, total EV `-1,586,875`.
- `val` `ask`: actions `128`, baseline-selected `0`, positive-EV actions `9`, total EV `-1,295,855`.
- `val` `bid`: actions `134`, baseline-selected `134`, positive-EV actions `50`, total EV `1,218,095`.

## 3. Baseline Coverage vs Oracle

- `train`: baseline PnL `-1,586,875`, missed oracle-positive PnL `4,437,350`, avoided negative EV `2,383,540`, oracle recall `0.427`.
- `val`: baseline PnL `1,218,095`, missed oracle-positive PnL `19,215`, avoided negative EV `1,315,070`, oracle recall `0.847`.
- `test`: baseline PnL `2,312,590`, missed oracle-positive PnL `3,160`, avoided negative EV `3,440,970`, oracle recall `0.969`.
- `new4`: baseline PnL `6,348,770`, missed oracle-positive PnL `373,845`, avoided negative EV `7,286,565`, oracle recall `0.771`.

## 4. Missed Opportunities

- `split_side` split `train` side `ask` spread `nan` session `nan`: missed PnL `4,437,350`, count `240`.
- `split_session` split `train` side `nan` spread `nan` session `20260122Day`: missed PnL `2,498,495`, count `127`.
- `split_side_spread` split `train` side `ask` spread `(1,2]` session `nan`: missed PnL `1,940,405`, count `103`.
- `split_spread` split `train` side `nan` spread `(1,2]` session `nan`: missed PnL `1,940,405`, count `103`.
- `split_session` split `train` side `nan` spread `nan` session `20260121Day`: missed PnL `1,710,720`, count `72`.
- `split_side_spread` split `train` side `ask` spread `(2,3]` session `nan`: missed PnL `1,515,310`, count `76`.
- `split_spread` split `train` side `nan` spread `(2,3]` session `nan`: missed PnL `1,515,310`, count `76`.
- `split_spread` split `train` side `nan` spread `<=1` session `nan`: missed PnL `913,875`, count `55`.

## 5. Opportunity Recovery Baselines

Validation-selected simple recovery rule: `no_recovery`.
- `val`: after-recovery PnL `1,218,095`, incremental `0`, matched random 95p `nan`.
- `test`: after-recovery PnL `2,312,590`, incremental `0`, matched random 95p `2,312,590`.
- `new4`: after-recovery PnL `6,348,770`, incremental `0`, matched random 95p `6,348,770`.

## 6. Full-Action ML Feasibility

Validation-selected ML policy: `full_hist_gradient_boosting_classifier` threshold `0.267313`.
- `val` metrics: PR-AUC `0.2223`, ROC-AUC `0.4938`, Spearman `-0.0090`.
- `test` metrics: PR-AUC `0.2447`, ROC-AUC `0.5490`, Spearman `0.0689`.
- `new4` metrics: PR-AUC `0.2633`, ROC-AUC `0.4930`, Spearman `-0.0105`.
- `val` ML PnL `154,460`, baseline `1,218,095`, gap `-1,063,635`, random 95p `234,522`, viable `False`.
- `test` ML PnL `-297,665`, baseline `2,312,590`, gap `-2,610,255`, random 95p `-4,700`, viable `False`.
- `new4` ML PnL `-447,895`, baseline `6,348,770`, gap `-6,796,665`, random 95p `166,679`, viable `False`.

## 7. Decision

Baseline conservative? `yes`.
Full-action ML viable? `False`.

ML should remain disabled for deployment. The baseline is conservative, but the validation-selected full-action ML selector does not clear both the strong baseline and matched-random hurdles on frozen test and NEW4.

The oracle opportunity is not a tradable claim. Test and NEW4 are not used for model or threshold tuning.
