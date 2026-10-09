status: stage2_csim_and_csynth_complete_no_cosim_no_pnr

> **CORRECTION NOTE (2026-08-16), see `AUDIT_stage2_checkpoint_selection_fix.md`:** the checkpoint
> selection rule below ranked candidates by `test_macro_f1`, which uses the held-out test set to
> make a model-selection decision and violates the test-isolation discipline Stage 1 itself claims
> (flagged as a real bug in `VALIDATION_METHODOLOGY_SOUNDNESS_AUDIT.md` §3.x). The selection code in
> `05_prepare_stage2_hls.py` has been fixed to rank by `val_macro_f1` only. Under the corrected rule
> the representative checkpoint changes from **seed=1** (below) to **seed=0**
> (`models/phc_fi2010_h50_seed0_raw_v2.pt`). The csim/csynth numbers in this report were produced
> against the old seed=1 checkpoint and are therefore superseded pending a re-run of weight
> export → csim → csynth against seed=0 in a Vitis HLS environment (not available in this
> sandbox). Treat every number below as provisional until that re-run is done.

# FI-2010 PHC-lite Stage 2 HLS Feasibility Report

## Scope

This Stage 2 pass starts only after Stage 1 v2 convergence closure. It uses the public FI-2010 benchmark artifacts under `NoAuction/1.NoAuction_Zscore`; it does not use TXF NEW1-NEW5 data. It does not retrain, reselect thresholds, run cosim, run P&R, create a bitstream, or touch board/JTAG flow.

The selected representative checkpoint is `models/phc_fi2010_h50_seed1_raw_v2.pt`: k=50, seed=1, raw loss. This was chosen as the median seed by Stage 1 v2 test macro F1 among k=50 raw runs, so the HLS feasibility number is tied to one concrete checkpoint rather than a cherry-picked best case.

Important boundary: this FI-2010 checkpoint records `fast_exact_final_step=True` and `fast_equiv_max_abs_diff=0.0` in the Stage 1 metrics, so the HLS step path can reproduce the full-window terminal logits from the last timestep input and zero state. Vitis optimized away unused `state_prev` logic in csynth. This is valid for this FI-2010 classifier checkpoint, but it is not a replacement claim for the TXF production recurrent PHC-lite deployment.

## Checkpoint

| field | value |
| --- | --- |
| checkpoint | `models\phc_fi2010_h50_seed1_raw_v2.pt` |
| horizon | 50 |
| seed | 1 |
| loss_mode | raw |
| best_epoch | 28 |
| trainable_parameters | 2115 |
| val_macro_f1 | 0.374580 |
| test_macro_f1 | 0.424395 |
| new5_used | False |

## Fixed-Point Export

Weights and buffers were exported to Q24.12 (`ap_fixed<24,12>`) with accumulator type `ap_fixed<40,16>`, matching the fixed-point convention already used by the TXF manual HLS path. The tanh/sigmoid path uses the same 65-point tanh LUT style; ReLU is not used in this PHC-lite step.

| item | value |
| --- | ---: |
| exported tensors | 19 |
| saturated values | 0 |
| test vectors | 25600 |
| test vector CSV | `results\fi2010_phc_stage2_test_vectors.csv` |
| csim output CSV | `results\fi2010_phc_stage2_csim_out.csv` |

## CSim

| metric | value |
| --- | ---: |
| status | pass |
| rows | 25600 |
| mismatches | 0 |
| max_abs_diff | 0.0079038175 |
| mean_abs_diff_all_outputs | 0.000998698878 |
| sign_agreement_logit_0 | 1 |
| sign_agreement_logit_1 | 0.999882812 |
| sign_agreement_logit_2 | 0.999921875 |
| joint_sign_agreement | 0.999804687 |

Tie-breaking convention for sign agreement is `0 -> positive`.

## CSynth

| metric | value |
| --- | ---: |
| target_clock_ns | 10.000 |
| estimated_clock_ns | 7.147 |
| estimated_fmax_mhz | 139.919 |
| latency_cycles | 7347 |
| latency_us_at_100mhz | 73.470 |
| interval_cycles | 7348 |
| pipeline_type | no |
| DSP | 33 (15.0%) |
| LUT | 20582 (38.7%) |
| FF | 11711 (11.0%) |
| BRAM_18K | 3 (1.1%) |
| fits_zynq7020_csynth_budget | True |
| report | `hls_component\hls_component\hls\syn\report\fi2010_phc_step_csynth.rpt` |

## Cross-Artifact Comparison

| design | input_dim | latency | frequency | verification_tier | resource | note |
| --- | --- | --- | --- | --- | --- | --- |
| TXF PHC-lite v8 manual HLS | 16 | 8316 cycles / 83.16 us | 100 MHz | routed_pnr_met + cosim | DSP 46, LUT 24836, FF 14518, BRAM_18K 6 | Production TXF baseline; different dataset/task. |
| TXF PHC-lite unroll16 | 16 | 2316 cycles / 25.733 us | 90 MHz | routed_pnr_met | DSP 125, routed 90MHz evidence in 06L/06O | Different dataset/task; included only as HLS methodology reference. |
| FI-2010 PHC-lite Stage2 step | 40 | 7347 cycles / 73.470 us | 100 MHz target | csim_pass + csynth_only | DSP 33 (15.0%), LUT 20582 (38.7%), FF 11711 (11.0%), BRAM_18K 3 (1.1%) | Public FI-2010 feasibility artifact only; no cosim/P&R. |

These rows are deliberately side-by-side only. FI-2010 uses 40 input features and a 3-class public benchmark objective, while TXF uses 16 microstructure features and bid/ask action values. The table supports HLS methodology feasibility; it does not establish a direct speed or PnL comparison between tasks.

## Claim Boundary

- No NEW1-NEW5 data is used in this FI-2010 HLS feasibility phase.
- No model training or checkpoint modification is performed in Stage 2.
- CSim passed and csynth completed; cosim, routed P&R, bitstream, and board validation were not run.
- This does not prove FPGA PnL improvement.
- This does not validate real-world profitability.
- This does not replace the TXF PHC-lite production HLS reports.
