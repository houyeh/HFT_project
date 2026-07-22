# PHC Ranking Head Report

This experiment adds a validation-selected ranking-aware auxiliary loss to PHC-lite v1. Oracle labels are used only inside train/validation labels for the ranking target; NEW3/NEW4/NEW5 are not used for selection.

## Selected Configuration
- Variant: `phc_rank_head`.
- Run id: `phc_rank_head_lam0p3_seed0`.
- Lambda rank: `0.3`.
- Lambda gate: `0.0`.
- Threshold: `0.9369753003120422`.

## Comparison Against PHC-lite v1
| split   |   total_excess_pnl_H20 |   v1_action_pnl |   action_delta_vs_v1 |   sequential_excess_pnl |   v1_causal_pnl |   causal_delta_vs_v1 |   capture_ratio |
|:--------|-----------------------:|----------------:|---------------------:|------------------------:|----------------:|---------------------:|----------------:|
| val     |            1.98794e+06 |     1.92717e+06 |                60775 |                  174780 |          170310 |                 4470 |        0.256633 |
| new3    |            5.1532e+06  |     5.02152e+06 |               131690 |                  494545 |          488870 |                 5675 |        0.273693 |
| new4    |            1.81528e+06 |     1.7625e+06  |                52780 |                  125720 |          121765 |                 3955 |        0.24615  |

## Matched Random
| split   |   model_total_excess_pnl_H20 |        random_95p | beats_random_95p   | empirical_p_value_wording   |
|:--------|-----------------------------:|------------------:|:-------------------|:----------------------------|
| val     |                  1.98794e+06 | -885152           | True               | < 0.001                     |
| new3    |                  5.1532e+06  |      -2.01079e+06 | True               | < 0.001                     |
| new4    |                  1.81528e+06 | -749940           | True               | < 0.001                     |

## Decision
`ranking_head_promising_but_not_adopted_seed_stability_caveat`: the selected seed0 ranking head passes action-level, causal-fill, and matched-random gates versus PHC-lite v1, but the follow-up seed stability audit shows weaker seed1/seed2 performance. Under the Phase 05D hard rule, this is not adopted as PHC-v2 without stronger seed stability.

This remains proxy excess markout research, not true exchange replay or real-world profitability proof.
