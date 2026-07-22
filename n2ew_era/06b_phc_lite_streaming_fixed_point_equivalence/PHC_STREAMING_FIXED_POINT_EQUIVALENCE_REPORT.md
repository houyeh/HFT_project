# PHC-lite Streaming and Fixed-point Equivalence Report

Final status: `phc_streaming_fixed_point_ready`.

This audit uses frozen PHC-lite v1 only. It does not retrain or retune the model. NEW5 is not used.

## Streaming Equivalence

| split   |   row_count |   max_abs_score_diff |   mean_abs_score_diff |   action_agreement_rate |   threshold | new5_used   |
|:--------|------------:|---------------------:|----------------------:|------------------------:|------------:|:------------|
| val     |       82960 |          7.62939e-05 |           2.96535e-07 |                       1 |           0 | False       |
| new3    |      176754 |          8.39233e-05 |           2.92929e-07 |                       1 |           0 | False       |
| new4    |       68620 |          7.62939e-05 |           2.89983e-07 |                       1 |           0 | False       |

## Fixed-point Emulation

| split   |   row_count |   q_total_bits |   q_frac_bits |   max_abs_score_diff |   mean_abs_score_diff |   action_agreement_rate |   selected_action_agreement_rate |   fp32_total_excess_pnl_H20 |   fixed_total_excess_pnl_H20 |   pnl_delta_fixed_minus_fp32 | new5_used   |
|:--------|------------:|---------------:|--------------:|---------------------:|----------------------:|------------------------:|---------------------------------:|----------------------------:|-----------------------------:|-----------------------------:|:------------|
| val     |       82960 |             24 |            12 |             0.778292 |              0.141548 |                0.997565 |                         0.997601 |                 1.92717e+06 |                  1.92813e+06 |                          960 | False       |
| new3    |      176754 |             24 |            12 |             0.955395 |              0.139582 |                0.997997 |                         0.998156 |                 5.02152e+06 |                  5.01946e+06 |                        -2060 | False       |
| new4    |       68620 |             24 |            12 |             0.665    |              0.142023 |                0.997799 |                         0.997931 |                 1.7625e+06  |                  1.76053e+06 |                        -1965 | False       |

## Causal-fill Sequential Check

| split   | policy_id                     |   sequential_excess_pnl |   fill_count |   forced_no_quote_count |   max_abs_inventory |   max_drawdown |   win_rate |   avg_trade_excess_pnl | fill_uses_future_pnl   | new5_used   |
|:--------|:------------------------------|------------------------:|-------------:|------------------------:|--------------------:|---------------:|-----------:|-----------------------:|:-----------------------|:------------|
| val     | phc_lite_fp32                 |                  170310 |         6746 |                       0 |                   1 |           2365 |   0.464572 |                25.2461 | False                  | False       |
| val     | phc_lite_fixed_point_emulated |                  169535 |         6731 |                       0 |                   1 |           2365 |   0.464567 |                25.1872 | False                  | False       |
| new3    | phc_lite_fp32                 |                  488870 |        15123 |                    2344 |                   1 |           3335 |   0.498578 |                32.3263 | False                  | False       |
| new3    | phc_lite_fixed_point_emulated |                  491325 |        15106 |                    2338 |                   1 |           3335 |   0.498941 |                32.5252 | False                  | False       |
| new4    | phc_lite_fp32                 |                  121765 |         4110 |                    2574 |                   1 |           1925 |   0.485401 |                29.6265 | False                  | False       |
| new4    | phc_lite_fixed_point_emulated |                  121285 |         4102 |                    2575 |                   1 |           1925 |   0.485617 |                29.5673 | False                  | False       |

Max absolute causal-fill PnL delta: `2,455.00`.
Material degradation detected using 95% FP32-PnL tolerance: `False`.

## Resource Estimate

| model       |   q_total_bits |   q_frac_bits |   parameter_count |   state_elements |   state_memory_bits |   estimated_mac_per_event | streaming_state_update   | window_buffer_required_for_exact_window_mode   | streaming_inference_plausible   | resource_estimate_type                       |
|:------------|---------------:|--------------:|------------------:|-----------------:|--------------------:|--------------------------:|:-------------------------|:-----------------------------------------------|:--------------------------------|:---------------------------------------------|
| PHC-lite v1 |             24 |            12 |              1203 |               16 |                 384 |                      1056 | True                     | True                                           | True                            | static_pre_hls_estimate_not_vendor_synthesis |

This is a pre-HLS fixed-point emulation and static resource estimate. It does not claim FPGA improves PnL, true exchange replay, or real-world profitability.
