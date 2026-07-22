# NEW5 One-Shot Final Holdout Report

Final outcome: `ml_passes_new5_holdout`.

Scope: `NEW5 20260128` TXF-only data. This was a one-shot frozen evaluation. No parameters, features, labels, thresholds, fill proxy settings, or model weights were changed using NEW5.

## Frozen Policy Evaluation

| split              | policy_id                                     |   horizon |   state_count |   selected_action_count |   selection_rate |   bid_count |   ask_count |   total_excess_pnl_H20 |   avg_excess_pnl_per_selected |   win_rate |   loss_rate |    max_drawdown | holdout_scope   | new5_used_for_tuning   |
|:-------------------|:----------------------------------------------|----------:|--------------:|------------------------:|-----------------:|------------:|------------:|-----------------------:|------------------------------:|-----------:|------------:|----------------:|:----------------|:-----------------------|
| new5_final_holdout | no_quote                                      |        20 |        815912 |                       0 |         0        |           0 |           0 |            0           |                        0      |   0        |    0        |     0           | NEW5_20260128   | False                  |
| new5_final_holdout | old_bid_only_spread_ge_1                      |        20 |        815912 |                  815912 |         1        |      815912 |           0 |           -1.14592e+07 |                      -14.0446 |   0.357817 |    0.642159 |     1.16053e+07 | NEW5_20260128   | False                  |
| new5_final_holdout | side_by_queue_imbalance_abs_0p1               |        20 |        815912 |                  672049 |         0.823678 |      325153 |      346896 |            1.14398e+07 |                       17.0223 |   0.427244 |    0.572732 | 80890           | NEW5_20260128   | False                  |
| new5_final_holdout | ml_hgb_regressor_shallow_convergence_selected |        20 |        815912 |                  617738 |         0.757114 |      318294 |      299444 |            1.56481e+07 |                       25.3312 |   0.456074 |    0.5439   | 51785           | NEW5_20260128   | False                  |

## Matched Random Summary

| split              | session_id          | policy_id                                     |   model_total_excess_pnl_H20 |   model_selected_action_count |   bid_count |   ask_count |   random_trials |   random_mean |   random_std |   random_95p |   random_99p |   empirical_p_value_fraction_random_ge_model | empirical_p_value_wording   | beats_random_95p   | matching                 | new5_used_for_tuning   |
|:-------------------|:--------------------|:----------------------------------------------|-----------------------------:|------------------------------:|------------:|------------:|----------------:|--------------:|-------------:|-------------:|-------------:|---------------------------------------------:|:----------------------------|:-------------------|:-------------------------|:-----------------------|
| new5_final_holdout | NEW5_20260128_TXFB6 | ml_hgb_regressor_shallow_convergence_selected |                  1.56481e+07 |                        617738 |      318294 |      299444 |            1000 |  -9.24726e+06 |      98761.9 | -9.08599e+06 | -9.02187e+06 |                                            0 | < 0.001                     | True               | exact_count_side_session | False                  |
| new5_final_holdout | NEW5_20260128_TXFB6 | side_by_queue_imbalance_abs_0p1               |                  1.14398e+07 |                        672049 |      325153 |      346896 |            1000 |  -1.00975e+07 |     102906   | -9.92494e+06 | -9.84803e+06 |                                            0 | < 0.001                     | True               | exact_count_side_session | False                  |

## Causal-Fill Sequential Summary

| split              | holdout_scope   | policy_id                                     |   horizon | fill_model             |   inventory_cap |   fee_multiplier |   sequential_excess_pnl |   fill_count |   forced_no_quote_count |   max_abs_inventory |   max_drawdown |   win_rate |   loss_rate |   avg_trade_excess_pnl | fill_uses_future_pnl   | new5_used_for_tuning   |   ml_minus_queue_rule |
|:-------------------|:----------------|:----------------------------------------------|----------:|:-----------------------|----------------:|-----------------:|------------------------:|-------------:|------------------------:|--------------------:|---------------:|-----------:|------------:|-----------------------:|:-----------------------|:-----------------------|----------------------:|
| new5_final_holdout | NEW5_20260128   | no_quote                                      |        20 | conservative_l2_stress |               1 |                1 |                       0 |            0 |                       0 |                   0 |              0 |   0        |    0        |                 0      | False                  | False                  |                170810 |
| new5_final_holdout | NEW5_20260128   | old_bid_only_spread_ge_1                      |        20 | conservative_l2_stress |               1 |                1 |                 -553365 |        30212 |                   86478 |                   1 |         556140 |   0.350225 |    0.649742 |               -18.3161 | False                  | False                  |                170810 |
| new5_final_holdout | NEW5_20260128   | side_by_queue_imbalance_abs_0p1               |        20 | conservative_l2_stress |               1 |                1 |                  722745 |        39600 |                   31932 |                   1 |           7760 |   0.436919 |    0.563005 |                18.2511 | False                  | False                  |                170810 |
| new5_final_holdout | NEW5_20260128   | ml_hgb_regressor_shallow_convergence_selected |        20 | conservative_l2_stress |               1 |                1 |                  893555 |        40526 |                   30780 |                   1 |           4610 |   0.453733 |    0.546193 |                22.0489 | False                  | False                  |                170810 |

## Key Comparison

ML action-level excess PnL: `15,648,070.00`.
Queue action-level excess PnL: `11,439,805.00`.
ML matched-random p-value wording: `< 0.001`.
ML sequential excess PnL: `893,555.00`.
Queue sequential excess PnL: `722,745.00`.

## Claim Safety Table

| claim                                                    | status                 | safe_wording                                                                                             |
|:---------------------------------------------------------|:-----------------------|:---------------------------------------------------------------------------------------------------------|
| ML beats matched random 95p on NEW5                      | PASS                   | The frozen ML candidate beat the exact matched random 95p hurdle on NEW5 20260128.                       |
| ML causal-fill sequential PnL is positive on NEW5        | PASS                   | The frozen ML candidate had positive causal-fill sequential excess PnL under the proxy simulator.        |
| ML beats queue rule in NEW5 causal-fill sequential proxy | PASS                   | The frozen ML candidate exceeded the frozen queue rule under the same NEW5 causal-fill sequential proxy. |
| NEW5 was used for tuning                                 | FORBIDDEN_AND_NOT_DONE | NEW5 was used only for one-shot final holdout evaluation.                                                |
| This proves real-world profitability                     | FORBIDDEN              | The result is proxy TXF-only holdout evidence, not proof of real-world profitability.                    |
| This is true exchange replay / true queue priority       | FORBIDDEN              | The causal-fill sequential simulator remains a proxy and does not replay exchange queue priority.        |
| FPGA improves PnL                                        | FORBIDDEN              | No FPGA PnL improvement claim is made.                                                                   |

Conservative interpretation: this is TXF-only proxy holdout evidence under frozen contracts. It does not prove real-world profitability, true queue replay, or FPGA-driven PnL improvement.
