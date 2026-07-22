# TCN Action-Value Final Report

Final status: `tcn_action_value_not_adopted`.

## 8a. Model Comparison

| split   | policy_id                                     |   action_pnl |   causal_fill_pnl |   oracle_capture |   action_delta_vs_phc_lite_v1 |   causal_fill_delta_vs_phc_lite_v1 |   oracle_capture_delta_vs_phc_lite_v1 |
|:--------|:----------------------------------------------|-------------:|------------------:|-----------------:|------------------------------:|-----------------------------------:|--------------------------------------:|
| val     | side_by_queue_imbalance_abs_0p1               |  1.23714e+06 |            103260 |         0.161827 |             -690025           |                             -67050 |                            -0.0877068 |
| val     | ml_hgb_regressor_shallow_convergence_selected |  1.67168e+06 |            142435 |         0.215578 |             -255495           |                             -27875 |                            -0.0339559 |
| val     | phc_lite                                      |  1.92717e+06 |            170310 |         0.249534 |                   0           |                                  0 |                             0         |
| val     | phc_stp_v2_cosine                             |  2.03792e+06 |            171835 |         0.264287 |              110755           |                               1525 |                             0.0147525 |
| val     | tcn_action_value_16feat                       |  1.6516e+06  |            138735 |         0.218583 |             -275565           |                             -31575 |                            -0.0309511 |
| val     | tcn_action_value_18feat                       |  1.69129e+06 |            147400 |         0.217644 |             -235880           |                             -22910 |                            -0.0318903 |
| new3    | side_by_queue_imbalance_abs_0p1               |  2.95477e+06 |            303700 |         0.157638 |                  -2.06674e+06 |                            -185170 |                            -0.108879  |
| new3    | ml_hgb_regressor_shallow_convergence_selected |  4.01058e+06 |            411090 |         0.212178 |                  -1.01094e+06 |                             -77780 |                            -0.0543398 |
| new3    | phc_lite                                      |  5.02152e+06 |            488870 |         0.266517 |                   0           |                                  0 |                             0         |
| new3    | phc_stp_v2_cosine                             |  5.23741e+06 |            611095 |         0.27863  |              215895           |                             122225 |                             0.0121124 |
| new3    | tcn_action_value_16feat                       |  4.24756e+06 |            466395 |         0.231792 |             -773960           |                             -22475 |                            -0.0347252 |
| new3    | tcn_action_value_18feat                       |  4.4655e+06  |            500505 |         0.235982 |             -556020           |                              11635 |                            -0.0305353 |
| new4    | side_by_queue_imbalance_abs_0p1               |  1.00391e+06 |             72415 |         0.136996 |             -758585           |                             -49350 |                            -0.102061  |
| new4    | ml_hgb_regressor_shallow_convergence_selected |  1.39542e+06 |             97680 |         0.188725 |             -367075           |                             -24085 |                            -0.0503321 |
| new4    | phc_lite                                      |  1.7625e+06  |            121765 |         0.239057 |                   0           |                                  0 |                             0         |
| new4    | phc_stp_v2_cosine                             |  1.87168e+06 |            129160 |         0.254392 |              109190           |                               7395 |                             0.0153349 |
| new4    | tcn_action_value_16feat                       |  1.50386e+06 |            108440 |         0.211846 |             -258630           |                             -13325 |                            -0.0272108 |
| new4    | tcn_action_value_18feat                       |  1.58397e+06 |            111490 |         0.213822 |             -178525           |                             -10275 |                            -0.0252349 |

## 8b. Feature Contribution

| split   |   action_pnl_16feat |   action_pnl_18feat |   action_delta_18_minus_16 |   causal_fill_pnl_16feat |   causal_fill_pnl_18feat |   causal_fill_delta_18_minus_16 | feature_enrichment_decision   |
|:--------|--------------------:|--------------------:|---------------------------:|-------------------------:|-------------------------:|--------------------------------:|:------------------------------|
| val     |         1.6516e+06  |         1.69129e+06 |                      39685 |                   138735 |                   147400 |                            8665 | feature_enrichment_helps      |
| new3    |         4.24756e+06 |         4.4655e+06  |                     217940 |                   466395 |                   500505 |                           34110 | feature_enrichment_helps      |
| new4    |         1.50386e+06 |         1.58397e+06 |                      80105 |                   108440 |                   111490 |                            3050 | feature_enrichment_helps      |

## 8c. Architecture Comparison Notes

`tcn_av_16feat` is the fair architecture comparison versus PHC-lite because it uses the same 16 causal PHC features. `tcn_av_18feat` adds causal `hawkes_excite` and `time_in_session_norm`, so any improvement includes feature enrichment rather than architecture alone.

## 8d. Adoption Gates

| model_variant   | policy_id               | gate                                        | pass   | rule                                         |
|:----------------|:------------------------|:--------------------------------------------|:-------|:---------------------------------------------|
| 16feat          | tcn_action_value_16feat | Convergence closed                          | False  | all seeds early_stop=True and best_epoch<185 |
| 16feat          | tcn_action_value_16feat | Seed stability                              | True   | all seeds x splits action PnL > 0            |
| 16feat          | tcn_action_value_16feat | Matched random                              | True   | val/NEW3/NEW4 all beat random 95p            |
| 16feat          | tcn_action_value_16feat | Action PnL improvement over PHC-lite v1     | False  | val/NEW3/NEW4 action PnL >= PHC-lite v1      |
| 16feat          | tcn_action_value_16feat | Causal-fill improvement over PHC-lite v1    | False  | val/NEW3/NEW4 sequential PnL >= PHC-lite v1  |
| 16feat          | tcn_action_value_16feat | Oracle capture improvement over PHC-lite v1 | False  | val/NEW3/NEW4 capture ratio >= PHC-lite v1   |
| 18feat          | tcn_action_value_18feat | Convergence closed                          | False  | all seeds early_stop=True and best_epoch<185 |
| 18feat          | tcn_action_value_18feat | Seed stability                              | True   | all seeds x splits action PnL > 0            |
| 18feat          | tcn_action_value_18feat | Matched random                              | True   | val/NEW3/NEW4 all beat random 95p            |
| 18feat          | tcn_action_value_18feat | Action PnL improvement over PHC-lite v1     | False  | val/NEW3/NEW4 action PnL >= PHC-lite v1      |
| 18feat          | tcn_action_value_18feat | Causal-fill improvement over PHC-lite v1    | False  | val/NEW3/NEW4 sequential PnL >= PHC-lite v1  |
| 18feat          | tcn_action_value_18feat | Oracle capture improvement over PHC-lite v1 | False  | val/NEW3/NEW4 capture ratio >= PHC-lite v1   |

## 8e. hls4ml Status

| model_variant   | policy_id               | hls4ml_attempted   | cpp_project_generated   | hls4ml_convertible   | vivado_synthesis_run   | input_shape   | precision      |   reuse_factor | strategy   | error                                                                                                                                                          |
|:----------------|:------------------------|:-------------------|:------------------------|:---------------------|:-----------------------|:--------------|:---------------|---------------:|:-----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 16feat          | tcn_action_value_16feat | True               | False                   | False                | False                  | 1,16,128      | ap_fixed<16,6> |              1 | Latency    | ValueError: too many values to unpack (expected 3)                                                                                                             |
|                 |                         |                    |                         |                      |                        |               |                |                |            | Traceback (most recent call last):                                                                                                                             |
|                 |                         |                    |                         |                      |                        |               |                |                |            |   File "C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06c_tcn_action_value_hls4ml\common_06c.py", line 995, in hls4ml_conversion                       |
|                 |                         |                    |                         |                      |                        |               |                |                |            |     cfg = hls4ml.utils.config_from_pytorch_model(                                                                                                              |
|                 |                         |                    |                         |                      |                        |               |                |                |            |         model,                                                                                                                                                 |
|                 |                         |                    |                         |                      |                        |               |                |                |            |     ...<2 lines>...                                                                                                                                            |
|                 |                         |                    |                         |                      |                        |               |                |                |            |         granularity="name",                                                                                                                                    |
|                 |                         |                    |                         |                      |                        |               |                |                |            |     )                                                                                                                                                          |
|                 |                         |                    |                         |                      |                        |               |                |                |            |   File "C:\Users\holbo\AppData\Local\Programs\Python\Python313\Lib\site-packages\hls4ml\utils\config.py", line 380, in config_from_pytorch_model               |
|                 |                         |                    |                         |                      |                        |               |                |                |            |     ) = parse_pytorch_model(config, verbose=False)                                                                                                             |
|                 |                         |                    |                         |                      |                        |               |                |                |            |         ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^                                                                                                             |
|                 |                         |                    |                         |                      |                        |               |                |                |            |   File "C:\Users\holbo\AppData\Local\Programs\Python\Python313\Lib\site-packages\hls4ml\converters\pytorch_to_hls.py", line 273, in parse_pytorch_model        |
|                 |                         |                    |                         |                      |                        |               |                |                |            |     layer, output_shape = layer_handlers[pytorch_class](                                                                                                       |
|                 |                         |                    |                         |                      |                        |               |                |                |            |                           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^                                                                                                       |
|                 |                         |                    |                         |                      |                        |               |                |                |            |         pytorch_class, layer_name, input_names, input_shapes, node, class_object, reader, config                                                               |
|                 |                         |                    |                         |                      |                        |               |                |                |            |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                               |
|                 |                         |                    |                         |                      |                        |               |                |                |            |     )                                                                                                                                                          |
|                 |                         |                    |                         |                      |                        |               |                |                |            |     ^                                                                                                                                                          |
|                 |                         |                    |                         |                      |                        |               |                |                |            |   File "C:\Users\holbo\AppData\Local\Programs\Python\Python313\Lib\site-packages\hls4ml\converters\pytorch\reshape.py", line 241, in parse_constantpad1d_layer |
|                 |                         |                    |                         |                      |                        |               |                |                |            |     batch, channels, width = input_shapes[0]                                                                                                                   |
|                 |                         |                    |                         |                      |                        |               |                |                |            |     ^^^^^^^^^^^^^^^^^^^^^^                                                                                                                                     |
|                 |                         |                    |                         |                      |                        |               |                |                |            | ValueError: too many values to unpack (expected 3)                                                                                                             |

## 8f. Forbidden Claims

- No NEW5 usage.
- No true exchange replay claim.
- No real-world profitability claim.
- No FPGA PnL improvement claim.
- hls4ml conversion, if successful, is C++ project generation only; Vivado synthesis is not run.
- Do not claim TCN beats PHC-lite unless all six adoption gates pass.
