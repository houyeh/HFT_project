# PHC STP v2 Q24.12 Fixed-point Equivalence Report

Final status: `phc_stp_v2_fixed_point_equivalence_ready_to_replace_gap`.

This result can directly replace the AMD-track proposal Section 4.5 / Section 5 item 4 pending note for 05C4 v2 seed=2 fixed-point equivalence.

## Scope

- Target model: `05C4 phc_stp_v2_cosine_lr`, selected seed `2`.
- Checkpoint: `n2ew_era/05c4_phc_stp_cosine_lr/models/phc_stp_v2_seed2_best_val.pt`.
- Target: `excess_pnl_H20`.
- Splits evaluated: `val`, `new3`, `new4`.
- NEW5 is not read or used.
- This is strategy-level `ap_fixed<24,12>` emulation against the float32 05C4 model. It is not HLS csim/cosim.
- The Q24.12 method follows the Phase 06B fixed-point emulation pattern, but the results below are recomputed for the deployed 05C4 v2 seed=2 model. They are not copied from the PHC-lite v1 Phase 06B audit.

## Action Equivalence

| split   | model                |   seed | checkpoint                                                                                                       |   threshold |   q_total_bits |   q_frac_bits |   state_count |   max_abs_score_diff |   mean_abs_score_diff |   action_agreement_rate |   selected_action_agreement_rate |   decision_flip_count |   fp32_selected_action_count |   fixed_selected_action_count |   fp32_total_excess_pnl_H20 |   fixed_total_excess_pnl_H20 |   pnl_delta_fixed_minus_fp32 |   action_agreement_pass_threshold | action_agreement_pass   | new5_used   |
|:--------|:---------------------|-------:|:-----------------------------------------------------------------------------------------------------------------|------------:|---------------:|--------------:|--------------:|---------------------:|----------------------:|------------------------:|---------------------------------:|----------------------:|-----------------------------:|------------------------------:|----------------------------:|-----------------------------:|-----------------------------:|----------------------------------:|:------------------------|:------------|
| val     | phc_stp_v2_cosine_lr |      2 | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c4_phc_stp_cosine_lr\models\phc_stp_v2_seed2_best_val.pt |           0 |             24 |            12 |         82960 |              1.9042  |              0.121146 |                0.998517 |                         0.999721 |                   123 |                        64568 |                         64656 |                 2.03792e+06 |                  2.0377e+06  |                         -220 |                              0.98 | True                    | False       |
| new3    | phc_stp_v2_cosine_lr |      2 | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c4_phc_stp_cosine_lr\models\phc_stp_v2_seed2_best_val.pt |           0 |             24 |            12 |        176754 |              5.71704 |              0.134701 |                0.99828  |                         0.999752 |                   304 |                       141208 |                        141442 |                 5.23741e+06 |                  5.2353e+06  |                        -2110 |                              0.98 | True                    | False       |
| new4    | phc_stp_v2_cosine_lr |      2 | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c4_phc_stp_cosine_lr\models\phc_stp_v2_seed2_best_val.pt |           0 |             24 |            12 |         68620 |              2.98841 |              0.133152 |                0.998324 |                         0.999649 |                   115 |                        54183 |                         54260 |                 1.87168e+06 |                  1.87313e+06 |                         1445 |                              0.98 | True                    | False       |

## Causal-fill PnL Equivalence

| split   |   phc_stp_v2_cosine_lr_fp32 |   phc_stp_v2_cosine_lr_q24_12_emulated |   causal_pnl_delta_fixed_minus_fp32 |   causal_pnl_delta_pct_fixed_minus_fp32 | material_degradation   |
|:--------|----------------------------:|---------------------------------------:|------------------------------------:|----------------------------------------:|:-----------------------|
| new3    |                      611095 |                                 610415 |                                -680 |                            -0.00111276  | False                  |
| new4    |                      129160 |                                 129940 |                                 780 |                             0.00603902  | False                  |
| val     |                      171835 |                                 171825 |                                 -10 |                            -5.81954e-05 | False                  |

Material degradation is defined exactly as in the Phase 06B style check: fixed-point causal-fill PnL below 95% of the float32 causal-fill PnL.

## Sanity Checks

| split   | check                                     |         computed |         existing |   delta | pass   |
|:--------|:------------------------------------------|-----------------:|-----------------:|--------:|:-------|
| val     | fp32_action_oos_matches_existing_05c4     |      2.03792e+06 |      2.03792e+06 |       0 | True   |
| val     | fp32_causal_fill_matches_existing_05c4    | 171835           | 171835           |       0 | True   |
| val     | fixed_causal_fill_not_materially_degraded | 171825           | 171835           |     -10 | True   |
| new3    | fp32_action_oos_matches_existing_05c4     |      5.23741e+06 |      5.23741e+06 |       0 | True   |
| new3    | fp32_causal_fill_matches_existing_05c4    | 611095           | 611095           |       0 | True   |
| new3    | fixed_causal_fill_not_materially_degraded | 610415           | 611095           |    -680 | True   |
| new4    | fp32_action_oos_matches_existing_05c4     |      1.87168e+06 |      1.87168e+06 |       0 | True   |
| new4    | fp32_causal_fill_matches_existing_05c4    | 129160           | 129160           |       0 | True   |
| new4    | fixed_causal_fill_not_materially_degraded | 129940           | 129160           |     780 | True   |

## Claim Boundary

- No retraining was performed.
- The 05C4 checkpoint and threshold were not modified.
- `common_05a.py` and `common_05c4.py` were imported only; they were not edited.
- This does not claim true exchange replay, real-world profitability, or hardware-caused PnL improvement.
