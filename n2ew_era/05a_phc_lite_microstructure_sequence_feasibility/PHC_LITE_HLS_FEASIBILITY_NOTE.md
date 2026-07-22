# PHC-lite HLS / Streaming Feasibility Note

This is a feasibility note only, not an HLS implementation and not a claim that FPGA improves PnL.

| model    |   feature_dim |   hidden_dim |   mtl_steps |   parameter_count |   state_values |   estimated_mac_per_event | target_under_20k_params   | streaming_inference_plausible   |
|:---------|--------------:|-------------:|------------:|------------------:|---------------:|--------------------------:|:--------------------------|:--------------------------------|
| phc_lite |            16 |           16 |           1 |              1203 |             16 |                      1280 | True                      | True                            |

## Fixed-Point Proxy Diagnostic

| split   | quantization              |   mean_abs_score_error |   max_abs_score_error |   action_agreement_rate | diagnostic_only   |
|:--------|:--------------------------|-----------------------:|----------------------:|------------------------:|:------------------|
| val     | score_round_to_1_16_proxy |              0.0156304 |               0.03125 |                0.999446 | True              |

The model keeps a small recurrent state and one within-timestep interaction loop, making streaming inference plausible if later validated with fixed-point and HLS tooling.
