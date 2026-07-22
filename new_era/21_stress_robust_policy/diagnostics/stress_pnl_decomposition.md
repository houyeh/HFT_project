# Stress PnL Decomposition

Decomposition identity: `net_pnl = spread_capture - adverse_loss - fees`.
`adverse_loss` is the reconciled residual containing adverse markout, stress toxic penalty, and terminal inventory mark-to-market effects.

## Moderate Stress Summary

- 016_v2: net `-38490`, spread_capture `44400`, adverse_loss `79050`, fees `3840`, trades `256`, toxic fills `170`, non-toxic fills `86`.
- 015_common: net `-52345`, spread_capture `51600`, adverse_loss `99250`, fees `4695`, trades `313`, toxic fills `229`, non-toxic fills `84`.
- 020_selected_policy: net `-74805`, spread_capture `50100`, adverse_loss `120450`, fees `4455`, trades `297`, toxic fills `213`, non-toxic fills `84`.
- random_v3_mean: net `-88881`, spread_capture `45894`, adverse_loss `130941`, fees `3834`, trades `255`, toxic fills `195`, non-toxic fills `59`.
- no_cancel: net `-97270`, spread_capture `56900`, adverse_loss `148950`, fees `5220`, trades `348`, toxic fills `262`, non-toxic fills `86`.

## Interpretation

The moderate-stress failure occurs when spread capture and fewer surviving fills cannot offset adverse residual loss plus fees. Policies can still be defensive if they reduce adverse loss relative to no-cancel/random, but absolute profitability requires positive net after this decomposition.