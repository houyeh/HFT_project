# PHC Streaming Dense Inference Report

This audit runs the frozen PHC-lite v1 as a session-reset streaming state machine. It does not retrain the model and does not use NEW5.

## Terminal Equivalence
| split   |   matched_terminal_count |   window_terminal_count |   dense_state_count |   score_bid_mae |   score_ask_mae |   score_bid_max_abs_error |   score_ask_max_abs_error |   action_agreement |   selected_action_agreement | equivalence_exact_expected   | reason                                                                                                 |
|:--------|-------------------------:|------------------------:|--------------------:|----------------:|----------------:|--------------------------:|--------------------------:|-------------------:|----------------------------:|:-----------------------------|:-------------------------------------------------------------------------------------------------------|
| val     |                    82960 |                   82960 |             2655369 |       0.0131406 |       0.0137669 |                   10.6872 |                   14.5224 |           0.999747 |                    0.999812 | False                        | window inference resets state at each 128-row window; streaming dense carries state from session start |
| new3    |                   176754 |                  176754 |             5656822 |       0.0113112 |       0.0123398 |                   57.6646 |                   60.7075 |           0.999842 |                    0.999899 | False                        | window inference resets state at each 128-row window; streaming dense carries state from session start |
| new4    |                    68620 |                   68620 |             2196280 |       0.0121025 |       0.0130397 |                   15.0822 |                   21.9814 |           0.999825 |                    0.999793 | False                        | window inference resets state at each 128-row window; streaming dense carries state from session start |

Exact equality is not expected because the original PHC-window inference resets hidden state inside each 128-row window, while dense streaming carries state from the session start. The key diagnostic is action agreement at matching terminal events.

## Dense Streaming Action-Level Evaluation
| split   |   state_count |   selected_action_count |   selection_rate |   total_excess_pnl_H20 |   oracle_capture_ratio |   max_drawdown |
|:--------|--------------:|------------------------:|-----------------:|-----------------------:|-----------------------:|---------------:|
| val     |       2655369 |                 2045109 |         0.770179 |            6.33814e+07 |               0.256251 |         217360 |
| new3    |       5656822 |                 4420337 |         0.781417 |            1.61325e+08 |               0.267151 |          94820 |
| new4    |       2196280 |                 1702278 |         0.775073 |            5.7938e+07  |               0.244843 |          41880 |

## Dense Streaming Causal-Fill Evaluation
| split   |   sequential_excess_pnl |   fill_count |   max_abs_inventory |   max_drawdown |
|:--------|------------------------:|-------------:|--------------------:|---------------:|
| val     |             2.9147e+06  |       121302 |                   1 |          17120 |
| new3    |             8.69735e+06 |       283977 |                   1 |           7115 |
| new4    |             3.12098e+06 |       108955 |                   1 |           6645 |

## Interpretation
Dense streaming is HLS-relevant because each event updates one recurrent state and emits bid/ask scores. This remains proxy excess markout and causal-fill simulation, not true exchange replay.
