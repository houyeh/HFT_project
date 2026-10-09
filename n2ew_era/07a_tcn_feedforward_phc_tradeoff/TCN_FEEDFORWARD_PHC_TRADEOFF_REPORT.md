# TCN Feedforward vs PHC-lite Trade-off Report

Final status: `tcn_distillation_closes_gap`.

## Scope

- This phase evaluates a feedforward `TCNPrePadRegressor` on the Phase 05A TXF-only sequence-window universe.
- Variants: `tcn_prepad_scratch_16feat` and `tcn_prepad_distilled_16feat`.
- Both variants use the same 16 PHC-lite causal features, `seq_len=128`, horizon `H20`, and validation-only threshold/seed selection.
- Distillation uses frozen PHC-lite v1 bid/ask action-value scores as soft labels. NEW3, NEW4, and NEW5 are not used for training or selection.

## Model / Latency Estimate

| model | estimate_type | feature_dim | seq_len_in | seq_len_out | channels | blocks | kernel_size | receptive_field | parameter_count | macs_per_sample | assumed_rate_name | assumed_macs_per_second | estimated_latency_us | vendor_synthesis_run | csim_run | csynth_run | cosim_run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tcn_prepad_16feat | pre_synthesis_macs_only_not_hls_verified | 16 | 252 | 128 | 16 | 5 | 3 | 125 | 7874 | 1632800 | fpga_effective_50gmac_per_s | 5e+10 | 32.656 | False | False | False | False |
| tcn_prepad_16feat | pre_synthesis_macs_only_not_hls_verified | 16 | 252 | 128 | 16 | 5 | 3 | 125 | 7874 | 1632800 | cpp_int8_reference_8gmac_per_s | 8e+09 | 204.1 | False | False | False | False |

The latency values above are pre-synthesis MAC-rate estimates. They are not csynth/cosim/P&R validated and must not be cited as FPGA-measured latency.

## PHC-lite vs TCN Trade-off

| model | split | action_pnl | causal_fill_pnl | selected_action_count | win_rate | matched_random_pass | empirical_p_value_fraction_random_ge_model | empirical_p_value_wording | matched_random_source | estimated_latency_us | latency_source | action_delta_vs_phc | causal_delta_vs_phc | latency_speedup_vs_phc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PHC-lite v1 teacher | val | 1.92717e+06 | 170310 | 63766 | 0.469451 | True | 0 | < 0.001 | selected_policy_matched_random_summary | 83.16 | 06F routed PHC step baseline at 100MHz | 0 | 0 | 1 |
| tcn_prepad_scratch_16feat | val | 1.92089e+06 | 172195 | 61837 | 0.470333 | True | 0 | < 0.001 | selected_policy_matched_random_summary | 32.656 | pre-synthesis MAC estimate only, not HLS-verified | -6280 | 1885 | 2.54655 |
| tcn_prepad_distilled_16feat | val | 1.97155e+06 | 173650 | 61114 | 0.474752 | True | 0 | < 0.001 | selected_policy_matched_random_summary | 32.656 | pre-synthesis MAC estimate only, not HLS-verified | 44380 | 3340 | 2.54655 |
| PHC-lite v1 teacher | new3 | 5.02152e+06 | 488870 | 138261 | 0.501045 | True | 0 | < 0.001 | selected_policy_matched_random_summary | 83.16 | 06F routed PHC step baseline at 100MHz | 0 | 0 | 1 |
| tcn_prepad_scratch_16feat | new3 | 5.00592e+06 | 565260 | 137866 | 0.502814 | True | 0 | < 0.001 | selected_policy_matched_random_summary | 32.656 | pre-synthesis MAC estimate only, not HLS-verified | -15590 | 76390 | 2.54655 |
| tcn_prepad_distilled_16feat | new3 | 5.09123e+06 | 587795 | 134699 | 0.506967 | True | 0 | < 0.001 | selected_policy_matched_random_summary | 32.656 | pre-synthesis MAC estimate only, not HLS-verified | 69715 | 98925 | 2.54655 |
| PHC-lite v1 teacher | new4 | 1.7625e+06 | 121765 | 53169 | 0.490605 | True | 0 | < 0.001 | selected_policy_matched_random_summary | 83.16 | 06F routed PHC step baseline at 100MHz | 0 | 0 | 1 |
| tcn_prepad_scratch_16feat | new4 | 1.79528e+06 | 126915 | 52789 | 0.494345 | True | 0 | < 0.001 | selected_policy_matched_random_summary | 32.656 | pre-synthesis MAC estimate only, not HLS-verified | 32785 | 5150 | 2.54655 |
| tcn_prepad_distilled_16feat | new4 | 1.81446e+06 | 133720 | 51604 | 0.497306 | True | 0 | < 0.001 | selected_policy_matched_random_summary | 32.656 | pre-synthesis MAC estimate only, not HLS-verified | 51960 | 11955 | 2.54655 |

## Cross-Split Pooled Delta Test

Exact sign tests and exact nonparametric bootstrap CIs use the three frozen diagnostic splits as the resampling units. With only three split-level observations, the sign test has low power; this table formalizes rather than visually infers the TCN-vs-teacher deltas.

| model | metric | split_count | positive_delta_count | delta_sum | delta_mean | exact_sign_test_two_sided_p | exact_split_bootstrap_mean_ci95_low | exact_split_bootstrap_mean_ci95_high | formal_readout | new5_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tcn_prepad_scratch_16feat | action_pnl_delta_vs_phc | 3 | 1 | 10915 | 3638.33 | 1 | -13572.8 | 24320.9 | sign_test_does_not_reject_equal_median_at_5pct | False |
| tcn_prepad_scratch_16feat | causal_fill_delta_vs_phc | 3 | 3 | 83425 | 27808.3 | 0.25 | 2592.42 | 60954.7 | sign_test_does_not_reject_equal_median_at_5pct | False |
| tcn_prepad_distilled_16feat | action_pnl_delta_vs_phc | 3 | 3 | 166055 | 55351.7 | 0.25 | 46022.3 | 65868.1 | sign_test_does_not_reject_equal_median_at_5pct | False |
| tcn_prepad_distilled_16feat | causal_fill_delta_vs_phc | 3 | 3 | 114220 | 38073.3 | 0.25 | 5206.58 | 80081.5 | sign_test_does_not_reject_equal_median_at_5pct | False |

## Seed Stability

| variant | split | action_pnl_mean | action_pnl_std | causal_fill_pnl_mean | causal_fill_pnl_std | matched_random_pass_all | empirical_p_value_max | all_positive | seed_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| distilled | new3 | 5.05537e+06 | 44211.8 | 578973 | 8408.57 | True | 0 | True | 3 |
| distilled | new4 | 1.80159e+06 | 11186.6 | 130382 | 4146.58 | True | 0 | True | 3 |
| distilled | val | 1.9444e+06 | 24332 | 175273 | 1854.86 | True | 0 | True | 3 |
| scratch | new3 | 4.939e+06 | 58052.2 | 563015 | 8926.8 | True | 0 | True | 3 |
| scratch | new4 | 1.76378e+06 | 31853 | 129323 | 3755 | True | 0 | True | 3 |
| scratch | val | 1.903e+06 | 17945.3 | 166205 | 5208.29 | True | 0 | True | 3 |

## Existing Context

- PHC latency path report: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06o_phc_lite_latency_path_to_1us\PHC_LATENCY_PATH_TO_1US_REPORT.md`.
- Existing 06C/06D TCN reports are useful context, but this phase is the 16-feature scratch/distillation comparison requested here.

## Claim Boundaries

- No NEW5 usage.
- No real-world profitability claim.
- No true exchange replay claim.
- No FPGA PnL improvement claim.
- TCN latency here is pre-synthesis MAC estimate only unless a later HLS phase runs csim/csynth/cosim. It is not HLS-verified in this report.
