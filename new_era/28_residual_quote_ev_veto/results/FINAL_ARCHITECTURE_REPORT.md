# Final Architecture Report

## Core System

The final core strategy remains the frozen non-ML side-aware spread controller `bid_only_spread_ge_1`. It is HLS-friendly and does not rely on ML calibration.

## Residual Quote-EV Veto

Selected residual policy on validation: `residual_ev_veto__logistic_l2_balanced__thr0.83338837`.
Selection split: `validation`; test used for selection: `False`.

## Model Diagnostics

- `train` PR-AUC `0.3309`, ROC-AUC `0.5968`, Spearman `0.1499`.
- `val` PR-AUC `0.1071`, ROC-AUC `0.3705`, Spearman `-0.1493`.
- `test` PR-AUC `0.0286`, ROC-AUC `0.2914`, Spearman `-0.1385`.
- `new4` PR-AUC `0.1473`, ROC-AUC `0.5067`, Spearman `0.0081`.

## Frozen Evaluation

- `test`: baseline `2,312,590`, ML after-veto `2,312,590`, incremental `0`, random 95p `2,312,590`.
- `new4`: baseline `6,348,770`, ML after-veto `6,348,770`, incremental `0`, random 95p `6,348,770`.

## Decision

Residual ML veto accepted: `False`.

If rejected, the architecture remains evidence-driven and conservative: deploy the non-ML spread/side controller as the final baseline, keep ML as diagnostic research, and do not claim real-world profitability or FPGA production necessity.

All PnL values remain simulator-derived quote-EV proxies under the current queue/fill stress framework.
