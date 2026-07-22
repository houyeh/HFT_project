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

| model | split | action_pnl | causal_fill_pnl | selected_action_count | win_rate | matched_random_pass | estimated_latency_us | latency_source | action_delta_vs_phc | causal_delta_vs_phc | latency_speedup_vs_phc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PHC-lite v1 teacher | val | 1.92717e+06 | 170310 | 63766 | 0.469451 | True | 83.16 | 06F routed PHC step baseline at 100MHz | 0 | 0 | 1 |
| tcn_prepad_scratch_16feat | val | 1.92089e+06 | 172195 | 61837 | 0.470333 |  | 32.656 | pre-synthesis MAC estimate only, not csynth/cosim | -6280 | 1885 | 2.54655 |
| tcn_prepad_distilled_16feat | val | 1.97155e+06 | 173650 | 61114 | 0.474752 |  | 32.656 | pre-synthesis MAC estimate only, not csynth/cosim | 44380 | 3340 | 2.54655 |
| PHC-lite v1 teacher | new3 | 5.02152e+06 | 488870 | 138261 | 0.501045 | True | 83.16 | 06F routed PHC step baseline at 100MHz | 0 | 0 | 1 |
| tcn_prepad_scratch_16feat | new3 | 5.00592e+06 | 565260 | 137866 | 0.502814 |  | 32.656 | pre-synthesis MAC estimate only, not csynth/cosim | -15590 | 76390 | 2.54655 |
| tcn_prepad_distilled_16feat | new3 | 5.09123e+06 | 587795 | 134699 | 0.506967 |  | 32.656 | pre-synthesis MAC estimate only, not csynth/cosim | 69715 | 98925 | 2.54655 |
| PHC-lite v1 teacher | new4 | 1.7625e+06 | 121765 | 53169 | 0.490605 | True | 83.16 | 06F routed PHC step baseline at 100MHz | 0 | 0 | 1 |
| tcn_prepad_scratch_16feat | new4 | 1.79528e+06 | 126915 | 52789 | 0.494345 |  | 32.656 | pre-synthesis MAC estimate only, not csynth/cosim | 32785 | 5150 | 2.54655 |
| tcn_prepad_distilled_16feat | new4 | 1.81446e+06 | 133720 | 51604 | 0.497306 |  | 32.656 | pre-synthesis MAC estimate only, not csynth/cosim | 51960 | 11955 | 2.54655 |

## Seed Stability

Seed stability summary missing because seed evaluation has not been generated.

## Existing Context

- PHC latency path report: `/home/holborn/projects/01_HFT_projject/n2ew_era/06o_phc_lite_latency_path_to_1us/PHC_LATENCY_PATH_TO_1US_REPORT.md`.
- Existing 06C/06D TCN reports are useful context, but this phase is the 16-feature scratch/distillation comparison requested here.

## Claim Boundaries

- No NEW5 usage.
- No real-world profitability claim.
- No true exchange replay claim.
- No FPGA PnL improvement claim.
- TCN latency here is pre-synthesis estimate only unless a later HLS phase runs csim/csynth/cosim/P&R.
