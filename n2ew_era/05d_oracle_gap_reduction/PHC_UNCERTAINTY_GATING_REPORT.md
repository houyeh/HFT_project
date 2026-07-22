# PHC Uncertainty Gating Report

This is a validation-selected ensemble disagreement diagnostic. It uses the frozen PHC-lite v1 seed checkpoints and does not retrain the model.

## Contract
- NEW5 usage: none.
- Selection split: validation only.
- Adjustment: `adjusted_score = seed_mean_score - lambda_uncertainty * seed_score_std`.
- Selected lambda: `0.0`.
- Selected threshold: `0.0`.

## OOS Summary
| split   |   total_excess_pnl_H20 |   v1_action_pnl |   action_delta_vs_v1 |   uncertainty_causal_pnl |   v1_causal_pnl |   causal_delta_vs_v1 |   oracle_capture_ratio |
|:--------|-----------------------:|----------------:|---------------------:|-------------------------:|----------------:|---------------------:|-----------------------:|
| val     |            1.9222e+06  |     1.92717e+06 |                -4975 |                   166695 |          170310 |                -3615 |               0.248583 |
| new3    |            4.98981e+06 |     5.02152e+06 |               -31705 |                   492030 |          488870 |                 3160 |               0.26468  |
| new4    |            1.7409e+06  |     1.7625e+06  |               -21600 |                   117080 |          121765 |                -4685 |               0.235897 |

## Oracle Error Direction
- False-positive rates by split: `{'val': 0.23099083895853423, 'new3': 0.20523439356393633, 'new4': 0.20450306033226465}`.
- Missed-opportunity rates by split: `{'val': 0.1638138862102218, 'new3': 0.16099211333265442, 'new4': 0.1681725444476829}`.

## Decision
`uncertainty_gate_not_adopted`: validation-selected uncertainty gating does not improve PHC-lite v1 across all required gates.

Oracle labels are diagnostic only and were not used for threshold selection. This does not imply real-world profitability or true exchange replay.
