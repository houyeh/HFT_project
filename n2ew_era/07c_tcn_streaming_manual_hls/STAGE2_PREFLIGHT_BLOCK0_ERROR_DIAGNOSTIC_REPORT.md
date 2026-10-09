status: stage2_preflight_pass_bounded_error
finding: block0_fixed_point_error_is_bounded_post_warmup
# Stage 2 Preflight Block0 Error Diagnostic

## Scope

This diagnostic reads the existing Stage 1 CSim output and recomputes timestep-wise RMSE against the Stage 0 Python streaming reference. It does not modify 07a, 07b, Stage 0 outputs, or Stage 1 outputs. It does not use NEW5.

## RMSE Curve Artifact

- Curve CSV: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage2_preflight_block0_timestep_rmse.csv`
- Curve PNG: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\figures\stage2_preflight_block0_timestep_rmse.png`

## Summary

| variant | rows | timesteps | first_quarter_mean_rmse | last_quarter_mean_rmse | last_over_first_quarter_rmse | post10_rmse_min | post10_rmse_median | post10_rmse_max | post10_rmse_slope_per_timestep | late_median_over_steady_median | max_rmse_timestep | classification | stage2_gate_pass | gate_basis | source_csim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | 49600 | 248 | 9.1903956e-05 | 0.00063793982 | 6.941375 | 9.1903956e-05 | 0.00061289662 | 0.00066688632 | 3.425883e-06 | 6.9469765 | 244 | bounded_steady_offset | True | post10_max<0.01 and abs(post10_linear_slope)<1e-5; diagnostic only, no model tuning | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage1_block0_scratch_csim_out.csv |
| distilled | 49600 | 248 | 0.00013302885 | 0.00083159331 | 6.2512251 | 0.00013302885 | 0.00078994662 | 0.00089148852 | 4.3910907e-06 | 6.2535431 | 244 | bounded_steady_offset | True | post10_max<0.01 and abs(post10_linear_slope)<1e-5; diagnostic only, no model tuning | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07c_tcn_streaming_manual_hls\results\stage1_block0_distilled_csim_out.csv |

## Selected Timesteps

| variant | out_t | rmse | mean_abs_diff | max_abs_diff | sample_count | compared_values |
| --- | --- | --- | --- | --- | --- | --- |
| scratch | 0 | 9.1903956e-05 | 5.104583e-05 | 0.00021117937 | 200 | 3200 |
| scratch | 1 | 9.1903956e-05 | 5.104583e-05 | 0.00021117937 | 200 | 3200 |
| scratch | 2 | 9.1903956e-05 | 5.104583e-05 | 0.00021117937 | 200 | 3200 |
| scratch | 3 | 9.1903956e-05 | 5.104583e-05 | 0.00021117937 | 200 | 3200 |
| scratch | 4 | 9.1903956e-05 | 5.104583e-05 | 0.00021117937 | 200 | 3200 |
| scratch | 5 | 9.1903956e-05 | 5.104583e-05 | 0.00021117937 | 200 | 3200 |
| scratch | 10 | 9.1903956e-05 | 5.104583e-05 | 0.00021117937 | 200 | 3200 |
| scratch | 20 | 9.1903956e-05 | 5.104583e-05 | 0.00021117937 | 200 | 3200 |
| scratch | 40 | 9.1903956e-05 | 5.104583e-05 | 0.00021117937 | 200 | 3200 |
| scratch | 80 | 9.1903956e-05 | 5.104583e-05 | 0.00021117937 | 200 | 3200 |
| scratch | 120 | 0.00038834461 | 0.00021028196 | 0.0021539925 | 200 | 3200 |
| scratch | 160 | 0.00063382296 | 0.00035007847 | 0.0055522925 | 200 | 3200 |
| scratch | 200 | 0.00064120454 | 0.00034907711 | 0.0036506625 | 200 | 3200 |
| scratch | 247 | 0.00063943432 | 0.00034846182 | 0.0039525062 | 200 | 3200 |
| distilled | 0 | 0.00013302885 | 7.5153964e-05 | 0.00037175387 | 200 | 3200 |
| distilled | 1 | 0.00013302885 | 7.5153964e-05 | 0.00037175387 | 200 | 3200 |
| distilled | 2 | 0.00013302885 | 7.5153964e-05 | 0.00037175387 | 200 | 3200 |
| distilled | 3 | 0.00013302885 | 7.5153964e-05 | 0.00037175387 | 200 | 3200 |
| distilled | 4 | 0.00013302885 | 7.5153964e-05 | 0.00037175387 | 200 | 3200 |
| distilled | 5 | 0.00013302885 | 7.5153964e-05 | 0.00037175387 | 200 | 3200 |
| distilled | 10 | 0.00013302885 | 7.5153964e-05 | 0.00037175387 | 200 | 3200 |
| distilled | 20 | 0.00013302885 | 7.5153964e-05 | 0.00037175387 | 200 | 3200 |
| distilled | 40 | 0.00013302885 | 7.5153964e-05 | 0.00037175387 | 200 | 3200 |
| distilled | 80 | 0.00013302885 | 7.5153964e-05 | 0.00037175387 | 200 | 3200 |
| distilled | 120 | 0.00045175527 | 0.00023958005 | 0.0038392581 | 200 | 3200 |
| distilled | 160 | 0.00082187434 | 0.00049224001 | 0.0044164681 | 200 | 3200 |
| distilled | 200 | 0.00082134491 | 0.00049044613 | 0.0048052075 | 200 | 3200 |
| distilled | 247 | 0.00085285886 | 0.00048992174 | 0.00591803 | 200 | 3200 |

## Interpretation

The high last/first quarter ratio comes mainly from an extremely small first-quarter denominator. After excluding the first 10 valid timesteps, the absolute RMSE remains below 0.01 and does not show a large linear slope in absolute units. This supports bounded fixed-point error rather than an accumulating FIFO-alignment bug.

## Stage Boundary

Stage 2 block1-4 implementation is allowed only if all `stage2_gate_pass` values are True.
