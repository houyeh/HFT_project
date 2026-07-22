# TXF-only Non-ML Baseline Report

Policies evaluated on TXF-only `excess_pnl_H20`. `NEW5` was not loaded or evaluated.

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

Interpretation: this is proxy short-horizon excess markout, not true exchange replay or real-world profitability.
