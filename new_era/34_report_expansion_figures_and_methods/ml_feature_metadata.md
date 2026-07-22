# ML Feature Metadata

All residual ML families in Phase 28/30 used the same causal residual EV tabular feature set unless noted otherwise.

## Shared Tabular Feature Set

- `spread_ticks`
- `queue_imbalance_calc`
- `deep_imbalance_calc`
- `top_depth`
- `depth_ratio`
- `volatility_past_50`
- `abs_return_past_50`
- `event_rate_past_50`
- `queue_imbalance_change_past`
- `normalized_spread_calc`

## Feature Categories

- L2 depth features: `queue_imbalance_calc`, `deep_imbalance_calc`, `top_depth`, `depth_ratio`.
- Imbalance ratios: `queue_imbalance_calc`, `deep_imbalance_calc`, `depth_ratio`.
- Rolling/lagged features: `volatility_past_50`, `abs_return_past_50`, `event_rate_past_50`, `queue_imbalance_change_past`.
- Spread features: `spread_ticks`, `normalized_spread_calc`.

## Model Family Feature Table

| Model family | Color | Input format | Uses L2 depth | Uses imbalance ratios | Uses rolling/lagged features | Sequence window | Feature columns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Models (Logistic) | `#276ef1` | Tabular | Yes | Yes | Yes | None | `spread_ticks`, `queue_imbalance_calc`, `deep_imbalance_calc`, `top_depth`, `depth_ratio`, `volatility_past_50`, `abs_return_past_50`, `event_rate_past_50`, `queue_imbalance_change_past`, `normalized_spread_calc` |
| Tree Boosting (HGB) | `#2f9e44` | Tabular | Yes | Yes | Yes | None | `spread_ticks`, `queue_imbalance_calc`, `deep_imbalance_calc`, `top_depth`, `depth_ratio`, `volatility_past_50`, `abs_return_past_50`, `event_rate_past_50`, `queue_imbalance_change_past`, `normalized_spread_calc` |
| Neural Networks (MLP) | `#c43c35` | Tabular | Yes | Yes | Yes | None | `spread_ticks`, `queue_imbalance_calc`, `deep_imbalance_calc`, `top_depth`, `depth_ratio`, `volatility_past_50`, `abs_return_past_50`, `event_rate_past_50`, `queue_imbalance_change_past`, `normalized_spread_calc` |
| Sequence Models (TCN) | `#7a4cc2` | Causal sequence | Yes | Yes | Yes | Default `50` rows | `spread_ticks`, `queue_imbalance_calc`, `deep_imbalance_calc`, `top_depth`, `depth_ratio`, `volatility_past_50`, `abs_return_past_50`, `event_rate_past_50`, `queue_imbalance_change_past`, `normalized_spread_calc` over a past-only window |

## Model Family Notes

- Linear Models (Logistic): calibrated logistic classifier on the shared tabular feature set.
- Tree Boosting (HGB): histogram gradient boosting classifier/regressor on the shared tabular feature set.
- Neural Networks (MLP): sklearn/PyTorch MLPs, including `small_mlp_64_32_epoch_rerun`, on the shared tabular feature set.
- Sequence Models (TCN): causal TCN in the exported RTX pack, using the same feature columns over a default causal window of `50` rows.

## TCN Sequence Details

- Script: `new_era/30_ml_training_adequacy_audit/export/ml_training_pack/05_tcn_sequence_training.py`.
- Default window: `--window 50`.
- Causal padding is used; the target row sees only historical rows in its window.
- The full sequence tensor is not expanded into RAM; windows are sliced dynamically by the PyTorch Dataset.

Conservative note: richer feature use does not imply deployable value. Residual ML still failed the frozen incremental-value hurdle.
