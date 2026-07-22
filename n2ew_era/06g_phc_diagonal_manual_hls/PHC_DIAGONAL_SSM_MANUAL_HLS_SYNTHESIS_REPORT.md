# PHC Diagonal SSM Manual HLS Synthesis Report

Status: `hls_package_ready_vitis_cli_not_available_in_current_shell`.

This phase builds a manual HLS package for the Phase 05F PHCDiagonalSSM selected model. In the current Codex shell, `vitis-run` / `vitis-run.bat` was not found on PATH, so C-sim, C-synthesis, and C/RTL cosim were not executed here. The generated package is ready for the user to run in a Vitis-enabled shell.

## Source And Protocol

- Source 05F selected checkpoint: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05f_phc_diagonal_ssm\models\phc_diagonal_ssm_seed2_best_val.pt`.
- Selected seed: `2`.
- Selected best epoch: `157`.
- Product/model data is inherited from Phase 05F; this phase does not retrain, reselect thresholds, or touch NEW5.
- Diagonal recurrence is exactly `state_t = DIAG_DECAY * state_prev + DIAG_GAIN * encoded_t`.
- `DIAG_DECAY` and `DIAG_GAIN` are baked sigmoid constants; no runtime sigmoid is used in recurrence.

## Weight Export Audit

- Q24.12 saturation status: `PASS`.
- Total saturated values: `0`.
- Weight audit CSV: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06g_phc_diagonal_manual_hls\results\weight_export_audit.csv`.

| phase   | name                     | cpp_name       | shape   |   num_values |   float_min |   float_max |   quantized_min |   quantized_max |   num_saturated |   total_bits |   frac_bits | exported_to_header   | note                                                    |
|:--------|:-------------------------|:---------------|:--------|-------------:|------------:|------------:|----------------:|----------------:|----------------:|-------------:|------------:|:---------------------|:--------------------------------------------------------|
| 06g     | encoder.0.weight         | ENC_LINEAR_W   | 16x16   |          256 |  -1.02133   |   0.807634  |           -4183 |            3308 |               0 |           24 |          12 | True                 | nan                                                     |
| 06g     | encoder.0.bias           | ENC_LINEAR_B   | 16      |           16 |  -0.606813  |   0.303344  |           -2486 |            1242 |               0 |           24 |          12 | True                 | nan                                                     |
| 06g     | encoder.1.weight         | ENC_NORM_GAMMA | 16      |           16 |   0.785521  |   0.988402  |            3217 |            4048 |               0 |           24 |          12 | True                 | nan                                                     |
| 06g     | encoder.1.bias           | ENC_NORM_BETA  | 16      |           16 |  -0.0621345 |   0.191721  |            -255 |             785 |               0 |           24 |          12 | True                 | nan                                                     |
| 06g     | diag_decay_baked_sigmoid | DIAG_DECAY     | 16      |           16 |   0.822606  |   0.918752  |            3369 |            3763 |               0 |           24 |          12 | True                 | sigmoid baked into exported constant                    |
| 06g     | diag_gain_baked_sigmoid  | DIAG_GAIN      | 16      |           16 |   0.474065  |   0.518719  |            1942 |            2125 |               0 |           24 |          12 | True                 | sigmoid baked into exported constant                    |
| 06g     | readout.weight           | READOUT_W      | 2x16    |           32 |  -0.256461  |   0.263358  |           -1050 |            1079 |               0 |           24 |          12 | True                 | nan                                                     |
| 06g     | readout.bias             | READOUT_B      | 2       |            2 |   0.011036  |   0.215284  |              45 |             882 |               0 |           24 |          12 | True                 | nan                                                     |
| 06g     | diag_decay_param         | nan            | 16      |           16 |   1.5341    |   2.42551   |            6284 |            9935 |               0 |           24 |          12 | False                | raw parameter; sigmoid is baked into DIAG_DECAY for HLS |
| 06g     | diag_gain_param          | nan            | 16      |           16 |  -0.103834  |   0.0749105 |            -425 |             307 |               0 |           24 |          12 | False                | raw parameter; sigmoid is baked into DIAG_GAIN for HLS  |

## Test Vector Audit

- Test-vector rows: `25600`.
- Full-model vs step-replay allclose: `True`.
- Test vector CSV: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06g_phc_diagonal_manual_hls\results\phc_diag_step_test_vectors.csv`.

## C-Sim / C-Synthesis / Cosim Status

| stage | status | total | mismatches | max_abs_diff | mean_abs_diff | path |
|---|---|---:|---:|---:|---:|---|
| csim | not_run_output_missing | None | None | None | None | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06g_phc_diagonal_manual_hls\results\phc_diag_step_csim_out.csv` |
| cosim | not_run_output_missing | None | None | None | None | `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06g_phc_diagonal_manual_hls\results\phc_diag_step_cosim_out.csv` |
| csynth | not_run_report_missing |  |  |  |  | `` |

## HLS Resource And Latency Comparison

| model             |   latency_cycles |   latency_us_100mhz |   interval_cycles |   DSP |   DSP_pct |   LUT |   LUT_pct |    FF |   FF_pct |   BRAM_18K |   BRAM_18K_pct |
|:------------------|-----------------:|--------------------:|------------------:|------:|----------:|------:|----------:|------:|---------:|-----------:|---------------:|
| 06f_dense16_final |             8316 |               83.16 |              8317 |    46 |   20.9091 | 24836 |   46.6842 | 14518 |  13.6447 |          6 |        2.14286 |
| 06g_diagonal16    |              nan |              nan    |               nan |   nan |  nan      |   nan |  nan      |   nan | nan      |        nan |      nan       |

06F dense16 baseline values are pulled from `n2ew_era/PHC_HLS_Technical_Report.md`: 8316 cycles / 83.16 us, DSP 46, LUT 24836, FF 14518, BRAM_18K 6.

## Stage Pipeline Intent

- Encoder stage: 16x16 matmul + LayerNorm + SiLU is kept sequential with explicit `PIPELINE off` on inner loops to avoid the known 06F auto-pipeline/auto-unroll DSP blowup.
- Recurrence stage: diagonal elementwise recurrence has explicit `#pragma HLS PIPELINE II=1` on the 16-channel loop.
- Readout stage: 2x16 matmul is kept sequential with `PIPELINE off`.

Because Vitis was not available in this shell, the achieved encoder/recurrence II and final resource usage are not measured yet. Run the commands below from a Vitis-enabled shell to fill this report with actual numbers:

```powershell
cd C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06g_phc_diagonal_manual_hls\hls_component
vitis-run.bat --mode hls --csim --config hls_config.cfg --work_dir hls_component
vitis-run.bat --mode hls --syn --config hls_config.cfg --work_dir hls_component
vitis-run.bat --mode hls --cosim --config hls_config.cfg --work_dir hls_component
```

## PnL Framing Caveat From Phase 05F

Diagonal16 is not broken and is not negative-PnL. It still beats HGB / queue / random baselines in Phase 05F, but regresses versus stronger PHC-family variants. The clean, artifact-safe cost to cite is the action PnL / oracle capture regression of roughly 8% to 13.5% versus dense16/dense8. The raw NEW3 causal-fill regression in 05F is not re-quoted as a clean capability-loss estimate here because it has not received the same simulator-state-carryover diagnostic that Phase 05E received.

## Forbidden Claims

- This phase does not retrain.
- This phase does not touch NEW5.
- This phase does not change the 06F deployed weights or claim to replace production.
- This phase does not prove FPGA PnL improvement.
- This phase does not prove true exchange replay or real-world profitability.
- Until Vitis HLS is run, this phase does not prove diagonal16 latency, resource fit, csim, csynth, or cosim success.
