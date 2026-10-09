# TCN PrePad Manual HLS Setup Report

Status: `code_prepared_csim_not_run_by_codex`.

## What Was Built

This phase prepares a manual Vitis HLS implementation for `TCNPrePadRegressor` on the Phase 07A TXF L2 task.

The implementation mirrors `src/models/tcn_prepad.py`:

1. Input is an unpadded `(128, 16)` normalized feature window.
2. HLS internally prepends 124 zero timesteps, creating `(252, 16)`.
3. Five valid-convolution residual TCN blocks run with dilations `[1, 2, 4, 8, 16]`.
4. Each block applies `conv1 -> ReLU -> conv2 -> ReLU -> cropped residual add -> ReLU`.
5. Final `(128, 16)` activations are averaged over time.
6. A linear readout emits bid/ask scores.

The only nonlinearity is ReLU, implemented as a fixed-point comparison. There is no tanh/sigmoid/sqrt LUT in this model.

## Variants

| variant | checkpoint | seed |
| --- | --- | ---: |
| scratch | `n2ew_era/07a_tcn_feedforward_phc_tradeoff/models/tcn_prepad_scratch_16feat_seed2_best_val.pt` | 2 |
| distilled | `n2ew_era/07a_tcn_feedforward_phc_tradeoff/models/tcn_prepad_distilled_16feat_seed1_best_val.pt` | 1 |

## Fixed-Point Contract

| quantity | type |
| --- | --- |
| input | `ap_fixed<24,12>` |
| weights | `ap_fixed<24,12>` |
| output readout | `ap_fixed<24,12>` |
| accumulators | `ap_fixed<40,16>` |

## Latency Definition

The HLS top function answers:

> Given one new event and the latest 128-row feature window, how long does a naive full-window recomputation take before bid/ask scores are emitted?

It does not use sliding-window cached convolution states. This must not be confused with PHC-lite's single recurrent-step latency.

## Files

- `00_export_tcn_weights.py`
- `01_generate_tcn_test_vectors.py`
- `02_prepare_hls_components.py`
- `03_collect_hls_results.py`
- `hls_src/tcn_prepad_hls.hpp`
- `hls_src/tcn_prepad_hls_impl.hpp`
- `hls_src/tcn_prepad_hls_scratch.cpp`
- `hls_src/tcn_prepad_hls_distilled.cpp`
- `hls_src/tcn_prepad_testbench.cpp`

## Commands

See `RUNBOOK.md`.

## Forbidden Claims

- Do not cite TCN latency as HLS-verified until C-sim and C-synthesis have been run.
- Do not cite it as routed timing until P&R has been run.
- Do not claim FPGA improves PnL.
- Do not claim true exchange replay.
- Do not use NEW5.

