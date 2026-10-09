# FI-2010 TCN Scratch Report

status: tcn_scratch_partial_or_complete

## Scope

This report covers only the FI-2010 scratch TCN baseline. It does not modify PHC-lite files, PHC v2/v3 checkpoints, TXF phases, or NEW1-NEW5 data.

## Architecture

- Model file: `fi2010_tcn.py`
- Input: `(batch, time=100, feature=40)`
- Projection: per-timestep `Linear(40 -> 8)`; for `tcn_small`, this shrinks width while preserving the same causal TCN topology.
- TCN blocks: 5 residual causal-conv blocks, kernel=3, dilations `[1, 2, 4, 8, 16]`, ReLU, dropout.
- Receptive field: `1 + 2 * (3 - 1) * sum(dilations) = 125`, covering the full 100-event FI-2010 window.
- Output: final-timestep readout. This matches the benchmark label at the terminal event and avoids pooling future/early-window context into a decision that should be anchored at the last event.
- Input/readout bias enabled: `False`.

| model | trainable parameters |
| --- | ---: |
| TCN scratch | 2344 |
| PHC-lite FI-2010 v2 | 2115 |

## Runtime Discipline

| item | value |
| --- | --- |
| start_time | 2026-08-05 12:47:36 |
| wall_clock_budget_hours | 6.0 |
| cpu_count | 8 |
| torch_threads | 2 |
| timing_train_samples_per_second | 3111.0959371218296 |
| timing_eval_samples_per_second | 15160.68998838594 |
| estimated_combo_seconds_max_epochs | 4138.9871796213165 |
| estimated_all_15_hours_max_epochs | 17.25 |
| full_15_combo_scan_feasible_under_budget | False |
| requested_combos_this_run | 1 |
| completed_combos_total | 1 |
| budget_exhausted | False |
| stopped_before_next_combo |  |

The timing probe is deliberately recorded before training. On this CPU-only run, the full 15-combination scan is not feasible under the 6-hour wall-clock budget, so k=50 was prioritized first.

## Per-Seed Results

| horizon | seed | best_epoch | final_epoch | early_stop | val_macro_F1 | test_macro_F1 | combo_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 0 | 28 | 38 | True | 55.26 | 61.23 | 1362.2 |

## TCN vs PHC-lite v2

PHC-lite v2 is used as the comparison baseline because it is the closed Stage 1 convergence baseline. PHC v3 is intentionally not used here.

| horizon | TCN_completed_seeds | TCN_test_macro_F1 | PHC_v2_test_macro_F1 | TCN_minus_PHC_pct_pt |
| --- | --- | --- | --- | --- |
| 50 | 1 | 61.23 +/- 0.00 | 40.72 | +20.52 |

## Files

- Metrics: `results\tcn_fi2010_metrics_small_raw.csv`
- Curves: `results\tcn_fi2010_training_curves_small_raw.csv`
- Summary: `results\tcn_fi2010_summary_small_raw.csv`
- Timing estimate: `results\tcn_fi2010_timing_estimate_small_raw.csv`
- Checkpoints: `models/tcn_fi2010_h{horizon}_seed{seed}_scratch_best.pt`

## Claim Boundary

- Scratch means direct label training; no distillation was used.
- This is software-only FI-2010 evidence, not HLS, P&R, or board validation.
- This does not touch TXF NEW1-NEW5.
- If fewer than 15 horizon/seed combinations are complete, the report is a partial time-budgeted run, not a full benchmark.
