# TXF-only Final Development Decision Report

Final decision: `one_candidate_ready_for_new5_holdout_metadata_locked`.
Selected candidate for upcoming NEW5 holdout: `side_by_queue_imbalance_abs_0p1`.

This decision is based only on TXF-only development validation plus NEW3/NEW4 diagnostic OOS. NEW5 was not loaded or evaluated.

A strategy must remain positive under TXF-only excess markout, beat matched random 95p where applicable, and survive the causal-fill sequential proxy before being proposed for NEW5.

## Candidate Scorecard

| candidate_policy_id             | strategy_family   | positive_excess_pnl_val_new3_new4   | beats_matched_random_95p_val_new3_new4   | positive_causal_fill_sequential_val_new3_new4   | eligible_for_new5_holdout   |   beats_queue_rule_val_new3_new4 | eligibility_note                                                     |
|:--------------------------------|:------------------|:------------------------------------|:-----------------------------------------|:------------------------------------------------|:----------------------------|---------------------------------:|:---------------------------------------------------------------------|
| old_bid_only_spread_ge_1        | non_ml            | False                               | False                                    | False                                           | False                       |                              nan | nan                                                                  |
| side_by_queue_imbalance_abs_0p1 | non_ml            | True                                | True                                     | True                                            | True                        |                              nan | nan                                                                  |
| ml_hgb_regressor_shallow        | simple_ml         | True                                | False                                    | False                                           | False                       |                                1 | diagnostic_only_until_matched_random_and_causal_fill_hurdles_are_run |

## Non-ML Summary

| split   | policy_id                       |   horizon |   state_count |   selected_action_count |   selection_rate |   bid_count |   ask_count |   total_excess_pnl_H20 |   avg_excess_pnl_per_selected |   win_rate |   loss_rate |     max_drawdown |
|:--------|:--------------------------------|----------:|--------------:|------------------------:|-----------------:|------------:|------------:|-----------------------:|------------------------------:|-----------:|------------:|-----------------:|
| train   | no_quote                        |        20 |       2470559 |                       0 |         0        |           0 |           0 |            0           |                        0      |   0        |    0        |      0           |
| train   | old_bid_only_spread_ge_1        |        20 |       2470559 |                 2470559 |         1        |     2470559 |           0 |           -3.62846e+07 |                      -14.6868 |   0.35792  |    0.642056 |      3.66361e+07 |
| train   | side_by_queue_imbalance_abs_0p1 |        20 |       2470559 |                 2032673 |         0.822758 |      988422 |     1044251 |            4.20832e+07 |                       20.7034 |   0.438631 |    0.561344 | 318275           |
| val     | no_quote                        |        20 |       2655369 |                       0 |         0        |           0 |           0 |            0           |                        0      |   0        |    0        |      0           |
| val     | old_bid_only_spread_ge_1        |        20 |       2655369 |                 2655369 |         1        |     2655369 |           0 |           -3.71563e+07 |                      -13.9929 |   0.349603 |    0.650352 |      3.72467e+07 |
| val     | side_by_queue_imbalance_abs_0p1 |        20 |       2655369 |                 2208424 |         0.831683 |     1096770 |     1111654 |            4.04909e+07 |                       18.3348 |   0.427169 |    0.572786 | 293385           |
| new3    | no_quote                        |        20 |       5656822 |                       0 |         0        |           0 |           0 |            0           |                        0      |   0        |    0        |      0           |
| new3    | old_bid_only_spread_ge_1        |        20 |       5656822 |                 5656822 |         1        |     5656822 |           0 |           -8.28904e+07 |                      -14.6532 |   0.366588 |    0.633391 |      8.30932e+07 |
| new3    | side_by_queue_imbalance_abs_0p1 |        20 |       5656822 |                 4630854 |         0.818632 |     2291447 |     2339407 |            9.38726e+07 |                       20.2711 |   0.445766 |    0.554215 | 112205           |
| new4    | no_quote                        |        20 |       2196280 |                       0 |         0        |           0 |           0 |            0           |                        0      |   0        |    0        |      0           |
| new4    | old_bid_only_spread_ge_1        |        20 |       2196280 |                 2196280 |         1        |     2196280 |           0 |           -3.18796e+07 |                      -14.5153 |   0.365855 |    0.634109 |      3.20455e+07 |
| new4    | side_by_queue_imbalance_abs_0p1 |        20 |       2196280 |                 1798986 |         0.819106 |      887377 |      911609 |            3.23474e+07 |                       17.9809 |   0.436614 |    0.563348 |  79265           |

## Matched Random Summary

| split   | policy_id                       |   model_total_excess_pnl_H20 |   random_trials |   random_mean |   random_std |   random_95p |   random_99p |   empirical_p_value_fraction_random_ge_model | beats_random_95p   | matching                 | test_new3_new4_used_for_selection   |
|:--------|:--------------------------------|-----------------------------:|----------------:|--------------:|-------------:|-------------:|-------------:|---------------------------------------------:|:-------------------|:-------------------------|:------------------------------------|
| val     | old_bid_only_spread_ge_1        |                 -3.71563e+07 |            1000 |  -3.71563e+07 |            0 | -3.71563e+07 | -3.71563e+07 |                                            1 | False              | exact_count_side_session | False                               |
| val     | side_by_queue_imbalance_abs_0p1 |                  4.04909e+07 |            1000 |  -3.31442e+07 |       175098 | -3.28666e+07 | -3.27528e+07 |                                            0 | True               | exact_count_side_session | False                               |
| new3    | old_bid_only_spread_ge_1        |                 -8.28904e+07 |            1000 |  -8.28904e+07 |            0 | -8.28904e+07 | -8.28904e+07 |                                            1 | False              | exact_count_side_session | False                               |
| new3    | side_by_queue_imbalance_abs_0p1 |                  9.38726e+07 |            1000 |  -6.94426e+07 |       258507 | -6.9037e+07  | -6.88e+07    |                                            0 | True               | exact_count_side_session | False                               |
| new4    | old_bid_only_spread_ge_1        |                 -3.18796e+07 |            1000 |  -3.18796e+07 |            0 | -3.18796e+07 | -3.18796e+07 |                                            1 | False              | exact_count_side_session | False                               |
| new4    | side_by_queue_imbalance_abs_0p1 |                  3.23474e+07 |            1000 |  -2.69947e+07 |       171980 | -2.6709e+07  | -2.65855e+07 |                                            0 | True               | exact_count_side_session | False                               |

When `empirical_p_value_fraction_random_ge_model` is `0` in the CSV, it means that in 1000 exact matched random trials no random trial achieved PnL greater than or equal to the model/policy PnL; the advisor-safe wording is empirical p-value `< 0.001`, not exactly zero.

## Causal-Fill Sequential Summary

| split   | policy_id                       |   horizon | fill_model             |   inventory_cap |   sequential_excess_pnl |   fill_count |   forced_no_quote_count |   max_abs_inventory |     max_drawdown |   win_rate |   loss_rate |   avg_trade_excess_pnl | fill_uses_future_pnl   |
|:--------|:--------------------------------|----------:|:-----------------------|----------------:|------------------------:|-------------:|------------------------:|--------------------:|-----------------:|-----------:|------------:|-----------------------:|:-----------------------|
| val     | no_quote                        |        20 | conservative_l2_stress |               1 |             0           |            0 |                       0 |                   0 |      0           |   0        |    0        |                 0      | False                  |
| val     | old_bid_only_spread_ge_1        |        20 | conservative_l2_stress |               1 |            -1.76664e+06 |        94442 |                  240238 |                   1 | 385600           |   0.337858 |    0.662078 |               -18.7061 | False                  |
| val     | side_by_queue_imbalance_abs_0p1 |        20 | conservative_l2_stress |               1 |             2.04026e+06 |       123934 |                   85773 |                   1 |  16930           |   0.425872 |    0.574088 |                16.4625 | False                  |
| new3    | no_quote                        |        20 | conservative_l2_stress |               1 |             0           |            0 |                       0 |                   0 |      0           |   0        |    0        |                 0      | False                  |
| new3    | old_bid_only_spread_ge_1        |        20 | conservative_l2_stress |               1 |            -4.07424e+06 |       211122 |                  611280 |                   1 |      1.01148e+06 |   0.355875 |    0.644097 |               -19.298  | False                  |
| new3    | side_by_queue_imbalance_abs_0p1 |        20 | conservative_l2_stress |               1 |             5.73154e+06 |       282746 |                  223118 |                   1 |  17630           |   0.450036 |    0.549932 |                20.271  | False                  |
| new4    | no_quote                        |        20 | conservative_l2_stress |               1 |             0           |            0 |                       0 |                   0 |      0           |   0        |    0        |                 0      | False                  |
| new4    | old_bid_only_spread_ge_1        |        20 | conservative_l2_stress |               1 |            -1.5599e+06  |        81171 |                  231372 |                   1 | 727350           |   0.354449 |    0.645501 |               -19.2175 | False                  |
| new4    | side_by_queue_imbalance_abs_0p1 |        20 | conservative_l2_stress |               1 |             2.07526e+06 |       107832 |                   82924 |                   1 |  15790           |   0.444729 |    0.555243 |                19.2454 | False                  |

## Simple ML Selected Policy Summary

| split   | policy_id                |   horizon |   state_count |   selected_action_count |   selection_rate |   bid_count |   ask_count |   total_excess_pnl_H20 |   avg_excess_pnl_per_selected |   win_rate |   loss_rate |   max_drawdown | model_id              |   threshold | selected_on_val_only   |
|:--------|:-------------------------|----------:|--------------:|------------------------:|-----------------:|------------:|------------:|-----------------------:|------------------------------:|-----------:|------------:|---------------:|:----------------------|------------:|:-----------------------|
| val     | ml_hgb_regressor_shallow |        20 |       2655369 |                 1968250 |         0.741234 |     1001948 |      966302 |            5.43816e+07 |                       27.6294 |   0.46071  |    0.539253 |         204185 | hgb_regressor_shallow |           0 | True                   |
| new3    | ml_hgb_regressor_shallow |        20 |       5656822 |                 4212753 |         0.744721 |     2143429 |     2069324 |            1.29338e+08 |                       30.7015 |   0.480671 |    0.519311 |          98830 | hgb_regressor_shallow |           0 | True                   |
| new4    | ml_hgb_regressor_shallow |        20 |       2196280 |                 1622095 |         0.738565 |      823048 |      799047 |            4.59844e+07 |                       28.3488 |   0.471429 |    0.528534 |          48850 | hgb_regressor_shallow |           0 | True                   |

Allowed claim: TXF-only development rerun completed under frozen product and loader contracts.
Forbidden claims: real-world profitability, true exchange replay, FPGA PnL improvement, or any NEW5 result.
