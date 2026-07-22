# ML Model Results Explanation

The ML workstreams tested whether residual ML could improve the conservative baseline by vetoing bad quotes inside the `bid_only_spread_ge_1` universe.

Inputs and safeguards:

- Training used causal tabular features such as spread, queue imbalance, depth, volatility, event rate, and imbalance changes.
- Thresholds were selected on validation only.
- Frozen test and NEW4 were not used for model selection.
- ML had to beat matched random veto 95p and improve baseline PnL at low veto rates.

Observed outcome:

- Validation sometimes showed small improvements.
- Frozen test produced zero useful vetoes or no improvement.
- NEW4 did not survive the matched-random veto hurdle.

Conservative interpretation: residual ML did not pass the incremental-value standard. This does not prove all ML is useless; it says this residual veto framing was too narrow or unstable.

Phase 034 recovered this gap by running `small_mlp_64_32_epoch_rerun` on the same residual universe with validation-only early stopping. The real epoch curves are now included as `figures/ml_epoch_loss_curves.png`, `figures/ml_epoch_metric_curves.png`, and `figures/ml_overfit_diagnostic.png`.

The recovered curves improve transparency but do not revive ML: the frozen rerun still failed to improve both test and NEW4 over the non-ML baseline.
