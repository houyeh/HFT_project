# ML Epoch Curve Recovery Report

Step 1 searched Phase 28 and Phase 30 for existing `*loss*.csv` or `*history*.csv` files.

- Recovered existing logs: `0`.
- No usable epoch logs were found, so a diagnostic rerun was performed.

## Diagnostic Rerun

- Model: `small_mlp_64_32_epoch_rerun`.
- Universe: Phase 030 residual baseline universe, equivalent to the Phase 028 `bid_only_spread_ge_1` universe.
- Features: causal residual EV features from `common_028.FEATURE_COLUMNS`.
- Early stopping: validation log loss only.
- Selected checkpoint epoch: `2`.
- Best validation loss: `0.454067`.
- Learning rate: `0.001`.
- Train samples: `820`.
- Validation samples: `134`.
- Test and NEW4 were not used for tuning, threshold selection, or checkpoint selection.

## Validation Threshold

- Threshold selected on validation: `0.38208633`.
- Validation veto count: `0`.
- Validation incremental PnL: `0`.

## Frozen Incremental Value

- `val`: baseline `1,218,095`, after veto `1,218,095`, incremental `0`, beats baseline `False`.
- `test`: baseline `2,312,590`, after veto `2,312,590`, incremental `0`, beats baseline `False`.
- `new4`: baseline `6,348,770`, after veto `6,329,015`, incremental `-19,755`, beats baseline `False`.

Accepted useful ML on both OOS splits: `False`.

Conclusion: the real epoch curves improve transparency of the rejected ML route. They do not revive ML unless frozen test and NEW4 both beat the non-ML baseline proxy PnL.

Figures:

- `figures/ml_epoch_loss_curves.png`
- `figures/ml_epoch_metric_curves.png`
- `figures/ml_overfit_diagnostic.png`
