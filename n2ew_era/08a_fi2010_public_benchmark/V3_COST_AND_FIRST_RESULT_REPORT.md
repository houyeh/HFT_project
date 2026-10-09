# 08A Stage1 v3 Cost And First Result Report

status: `first_v3_evidence_point_complete_full_sweep_deferred`

This note records the speed budget and the first minimal v3 training result
before any 15-combination or hidden-dim expansion sweep. No Stage2 HLS flow was
run. No TXF NEW1-NEW5 data or code path was touched.

## Forward Equivalence Gate

`09_verify_v3_forward.py` was run before training. It wrote
`V3_FORWARD_EQUIVALENCE_CHECK.md`.

| check | value |
|---|---:|
| max_abs_diff, vectorized `forward()` vs repeated `forward_step()` | 1.78813934e-07 |
| threshold | 1.0e-05 |
| pass | True |
| trainable parameters | 2132 |
| initial effective decay min | 0.899999976 |
| initial effective decay max | 0.899999976 |
| initial alpha_scalar | 0.25 |

Interpretation: the v3 full-window forward path preserves the true recurrent
step semantics up to floating-point noise. This clears the architecture sanity
gate before spending training time.

## Speed Benchmark

`10_benchmark_v3_batch.py` was run on the requested grid and wrote
`results/v3_speed_benchmark.csv`.

| threads | batch_size | seconds | samples/sec |
|---:|---:|---:|---:|
| 4 | 128 | 0.1272 | 1006.06 |
| 4 | 256 | 2.5990 | 98.50 |
| 4 | 512 | 2.4244 | 211.19 |
| 4 | 1024 | 5.4008 | 189.60 |
| 8 | 128 | 2.0721 | 61.77 |
| 8 | 256 | 0.6835 | 374.53 |
| 8 | 512 | 2.7338 | 187.28 |
| 8 | 1024 | 13.7645 | 74.39 |

The fastest single-batch microbenchmark was `threads=4, batch_size=128`.
However, the actual training script includes numpy window materialization,
many batches, full validation, checkpointing, and final test evaluation, so the
single-batch number is only a lower bound.

Using the fastest microbenchmark alone:

- train windows per epoch: 203,701
- lower-bound training-only epoch time: 203,701 / 1006.06 = 202.5 seconds
- lower-bound 60-epoch one-combination time: 3.37 hours
- lower-bound 15-combination time: 50.6 hours

Actual calibrated training timing from `k=50, seed=0, raw, max_epochs=1` with
`threads=4, batch_size=128`:

- epoch 1 elapsed time reported by training curve: 1185.0 seconds
- wall time including final test/writeout: about 1305 seconds
- calibrated 60-epoch one-combination time: 19.75 hours
- calibrated 15-combination time: 296.3 hours, or 12.3 days, before any
  weighted fallbacks or hidden_dim expansion

This is why a full 15-combination v3 sweep is not started yet.

## First v3 Training Result

Run:

```text
phase: 08a_stage1_v3_k50_speedprobe
horizon: 50
seed: 0
loss_mode: raw
hidden_dim: 16
max_epochs: 1
patience: 10
batch_size: 128
eval_batch_size: 1024
torch_threads: 4
fast_exact_final_step: False
```

Artifacts:

- `results/training_curves_v3_k50_speedprobe.csv`
- `results/stage1_metrics_v3_k50_speedprobe.csv`
- `results/stage1_summary_v3_k50_speedprobe.csv`
- `results/stage1_primary_metrics_v3_k50_speedprobe.csv`
- `models/phc_fi2010_h50_seed0_raw_v3_k50_speedprobe.pt`

This is a real full-dataset train/validation/test run, but it is intentionally
budget-limited and not converged: `max_epoch_hit=True`,
`convergence_closed=False`.

| metric | v3 k=50/seed=0/raw, 1 epoch |
|---|---:|
| train_loss | 1.0445585 |
| val_loss | 1.0563931 |
| val_macro_f1 | 43.3307% |
| test_macro_f1 | 50.1784% |
| best_epoch | 1 |
| final_epoch | 1 |
| trainable_parameters | 2132 |
| decay_effective_mean | 0.8975116 |
| decay_effective_min | 0.8819652 |
| decay_effective_max | 0.9067095 |
| alpha_scalar_final | 0.5457251 |

The v3 decay distribution did not collapse toward zero after this first epoch.
It remained a nonzero per-channel distribution around the initialized memory
coefficient, while alpha moved from 0.25 to 0.5457.

## v2 Comparison For The Same Combination

Baseline row from `results/stage1_metrics_v2.csv`:

```text
phase: 08a_stage1_v2
horizon: 50
seed: 0
loss_mode: raw
best_epoch: 15
final_epoch: 25
test_macro_f1: 42.4444%
val_macro_f1: 36.6488%
fast_exact_final_step: True
```

| comparison | val macro F1 | test macro F1 |
|---|---:|---:|
| v2 closed-recursion, converged | 36.6488% | 42.4444% |
| v3 true-recursion, 1 epoch budget-limited | 43.3307% | 50.1784% |
| v3 minus v2 | +6.6810 pct-pt | +7.7340 pct-pt |

This is encouraging but not a final v3 claim, because the v3 row has only one
epoch and hit the artificial epoch cap. It is enough to show that the corrected
recurrence is active and can improve the same k=50/seed=0/raw comparison very
early, but not enough to claim v3 convergence or seed-level stability.

## Extension Attempt And Stop Decision

A controlled 5-epoch extension was attempted with
`threads=4, batch_size=128`. Epoch 1 completed with the same F1 as the
speedprobe but was slower under current machine load (`elapsed_seconds` about
1503 seconds), then epoch 2 failed during backward with a CPU allocation error.

A follow-up `batch_size=512, threads=4` attempt failed at torch import with
`WinError 1114` while loading `torch/lib/shm.dll`, consistent with transient
system resource pressure.

On 2026-08-04, the requested fair `batch_size=512` rerun was not launched
because the resource preflight still did not satisfy the "no other heavy load"
condition. Available memory was about `2133 MB`; `OneDrive.Sync.Service` used
about `4148.7 MB` working set, `OneDrive` used about `855.1 MB`, and two
Anaconda Python processes remained CPU-active. The snapshot is saved in
`results/v3_fair512_resource_preflight.csv`. No `_v3_k50_fair512` metric row
exists yet.

Given these resource failures and the calibrated 12.3-day estimate for the full
15-combination sweep, the correct decision is to stop here. The next expansion
should only happen after either freeing memory/CPU contention, adding GPU or a
faster host, or adding resumable checkpointing and a narrower staged protocol.

## Next Decision

The first real v3 evidence point is positive for `k=50/seed=0/raw`, with test
macro F1 moving from v2 `42.44%` to v3 `50.18%` after one epoch. The immediate
next useful run is not the full 15-combination sweep. It is a resumable
continuation of this single combination to 5-10 epochs under stable resources,
then a decision on whether the validation curve is still rising.
