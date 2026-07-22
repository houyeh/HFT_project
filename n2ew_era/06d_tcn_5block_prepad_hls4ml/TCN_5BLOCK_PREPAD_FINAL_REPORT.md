# TCN 5-Block PrePad Final Report

Final status: `tcn_5block_prepad_not_adopted`.

## 1. Architecture Summary

- Blocks: `5`.
- Dilations: `[1, 2, 4, 8, 16]`.
- Kernel size: `3`.
- Receptive field: `125` ticks.
- Output sequence length: `128`.
- Pre-padded input length: `252`.
- Total left prepad: `124`.
- Channels: `[16, 16, 16, 16, 16]`.
- Variant: `18feat` only.

## 2. Receptive Field Comparison

| phase   | model                         | dilations        | internal_padding   |   seq_len_in |   seq_len_out |   receptive_field_ticks | hls4ml_issue                           |
|:--------|:------------------------------|:-----------------|:-------------------|-------------:|--------------:|------------------------:|:---------------------------------------|
| 06c     | 3-block dilated TCN           | [1, 2, 4]        | ConstantPad1d      |          128 |           128 |                      29 | ConstantPad1d parser failed            |
| 06d     | 5-block prepad valid-conv TCN | [1, 2, 4, 8, 16] | none               |          252 |           128 |                     125 | prepad outside model; no ConstantPad1d |

## 3. Model Comparison

| split   | policy_id                                     |   action_pnl |   causal_fill_pnl |   oracle_capture |   action_delta_vs_phc_lite_v1 |   causal_fill_delta_vs_phc_lite_v1 |   oracle_capture_delta_vs_phc_lite_v1 |
|:--------|:----------------------------------------------|-------------:|------------------:|-----------------:|------------------------------:|-----------------------------------:|--------------------------------------:|
| val     | side_by_queue_imbalance_abs_0p1               |  1.23714e+06 |            103260 |         0.161827 |             -690025           |                             -67050 |                            -0.0877068 |
| val     | ml_hgb_regressor_shallow_convergence_selected |  1.67168e+06 |            142435 |         0.215578 |             -255495           |                             -27875 |                            -0.0339559 |
| val     | phc_lite                                      |  1.92717e+06 |            170310 |         0.249534 |                   0           |                                  0 |                             0         |
| val     | phc_stp_v2_cosine                             |  2.03792e+06 |            171835 |         0.264287 |              110755           |                               1525 |                             0.0147525 |
| val     | tcn_action_value_18feat                       |  1.69129e+06 |            147400 |         0.217644 |             -235880           |                             -22910 |                            -0.0318903 |
| val     | tcn_5block_prepad_18feat                      |  1.71809e+06 |            151305 |         0.221504 |             -209080           |                             -19005 |                            -0.0280306 |
| new3    | side_by_queue_imbalance_abs_0p1               |  2.95477e+06 |            303700 |         0.157638 |                  -2.06674e+06 |                            -185170 |                            -0.108879  |
| new3    | ml_hgb_regressor_shallow_convergence_selected |  4.01058e+06 |            411090 |         0.212178 |                  -1.01094e+06 |                             -77780 |                            -0.0543398 |
| new3    | phc_lite                                      |  5.02152e+06 |            488870 |         0.266517 |                   0           |                                  0 |                             0         |
| new3    | phc_stp_v2_cosine                             |  5.23741e+06 |            611095 |         0.27863  |              215895           |                             122225 |                             0.0121124 |
| new3    | tcn_action_value_18feat                       |  4.4655e+06  |            500505 |         0.235982 |             -556020           |                              11635 |                            -0.0305353 |
| new3    | tcn_5block_prepad_18feat                      |  4.50013e+06 |            441990 |         0.238185 |             -521385           |                             -46880 |                            -0.0283326 |
| new4    | side_by_queue_imbalance_abs_0p1               |  1.00391e+06 |             72415 |         0.136996 |             -758585           |                             -49350 |                            -0.102061  |
| new4    | ml_hgb_regressor_shallow_convergence_selected |  1.39542e+06 |             97680 |         0.188725 |             -367075           |                             -24085 |                            -0.0503321 |
| new4    | phc_lite                                      |  1.7625e+06  |            121765 |         0.239057 |                   0           |                                  0 |                             0         |
| new4    | phc_stp_v2_cosine                             |  1.87168e+06 |            129160 |         0.254392 |              109190           |                               7395 |                             0.0153349 |
| new4    | tcn_action_value_18feat                       |  1.58397e+06 |            111490 |         0.213822 |             -178525           |                             -10275 |                            -0.0252349 |
| new4    | tcn_5block_prepad_18feat                      |  1.6034e+06  |            121000 |         0.216751 |             -159095           |                               -765 |                            -0.0223057 |

## 4. Architecture Comparison Notes

`tcn_5block_prepad_18feat` is compared against the 06C `tcn_action_value_18feat`, PHC-lite v1, and 05C4 STP v2. The 06D change is the deeper valid-conv TCN with external pre-padding; it still uses the 18-feature sequence tensor.

## 5. Adoption Gates

| model_variant   | policy_id                | gate                                        | pass   | rule                                         |
|:----------------|:-------------------------|:--------------------------------------------|:-------|:---------------------------------------------|
| 18feat          | tcn_5block_prepad_18feat | Convergence closed                          | True   | all seeds early_stop=True and best_epoch<185 |
| 18feat          | tcn_5block_prepad_18feat | Seed stability                              | True   | all seeds x splits action PnL > 0            |
| 18feat          | tcn_5block_prepad_18feat | Matched random                              | True   | val/NEW3/NEW4 all beat random 95p            |
| 18feat          | tcn_5block_prepad_18feat | Action PnL improvement over PHC-lite v1     | False  | val/NEW3/NEW4 action PnL >= PHC-lite v1      |
| 18feat          | tcn_5block_prepad_18feat | Causal-fill improvement over PHC-lite v1    | False  | val/NEW3/NEW4 sequential PnL >= PHC-lite v1  |
| 18feat          | tcn_5block_prepad_18feat | Oracle capture improvement over PHC-lite v1 | False  | val/NEW3/NEW4 capture ratio >= PHC-lite v1   |

## 6. hls4ml Status

| model_variant   | policy_id                | hls4ml_attempted   | cpp_project_generated   | hls4ml_convertible   | hls4ml_compile_success   | vivado_synthesis_run   | input_shape   | precision      |   reuse_factor | strategy   | constantpad1d_layers_present   |   parsed_layer_count | hls4ml_version   | error                                                                                                                                                   |
|:----------------|:-------------------------|:-------------------|:------------------------|:---------------------|:-------------------------|:-----------------------|:--------------|:---------------|---------------:|:-----------|:-------------------------------|---------------------:|:-----------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
| 18feat          | tcn_5block_prepad_18feat | True               | False                   | False                | False                    | False                  | 1,18,252      | ap_fixed<16,6> |              1 | Latency    | False                          |                    0 | 1.3.0            | TraceError: symbolically traced variables cannot be used as inputs to control flow                                                                      |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  | Traceback (most recent call last):                                                                                                                      |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  |   File "C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06d_tcn_5block_prepad_hls4ml\common_06d.py", line 1113, in hls4ml_conversion              |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  |     cfg = hls4ml.utils.config_from_pytorch_model(                                                                                                       |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  |         model,                                                                                                                                          |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  |     ...<2 lines>...                                                                                                                                     |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  |         granularity="name",                                                                                                                             |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  |     )                                                                                                                                                   |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  |   File "C:\Users\holbo\AppData\Local\Programs\Python\Python313\Lib\site-packages\hls4ml\utils\config.py", line 380, in config_from_pytorch_model        |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  |     ) = parse_pytorch_model(config, verbose=False)                                                                                                      |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  |         ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^                                                                                                      |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  |   File "C:\Users\holbo\AppData\Local\Programs\Python\Python313\Lib\site-packages\hls4ml\converters\pytorch_to_hls.py", line 145, in parse_pytorch_model |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  |     traced_model = tracer.trace(model)                                                                                                                  |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  |   File "C:\Users\holbo\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\fx\_symbolic_trace.py", line 868, in trace                       |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  |     (self.create_arg(fn(*args)),),                                                                                                                      |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  |                      ~~^^^^^^^                                                                                                                          |
|                 |                          |                    |                         |                      |                          |                        |               |                |                |            |                                |                      |                  | torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow                                                       |

## 7. Conclusion

Do not claim this TCN beats PHC-lite unless all six gates pass. If all gates fail to pass, 06D is a diagnostic A3D3-track contribution: it tests whether the 06C receptive-field and ConstantPad1d issues were the limiting factors.

## 8. Forbidden Claims

- No NEW5 usage.
- No true exchange replay claim.
- No real-world profitability claim.
- No FPGA PnL improvement claim.
- hls4ml conversion, if successful, is C++ project generation only; Vivado synthesis is not run.
- Do not claim TCN beats PHC-lite unless all six adoption gates pass.
