status: resolved
finding: v3_fair512_three_seed_patience_closed_beats_v2_k50_after_removing_batch_size_confound
# V3 Fair512 Report

## Scope

This report covers the FI-2010 PHC v3 fair batch-size control for `k=50`, `loss_mode=raw`.

- Architecture/training code was not modified: `fi2010_phc_lite.py` and `02_train_phc_lite_stage1.py` were read only.
- No `fi2010_tcn.py` or `tcn_*.py` files were touched.
- No TXF NEW1-NEW5 data was used or touched.
- Hidden dimension stayed fixed at 16.
- Batch size stayed at 512, matching the v2 baseline.
- Early stopping used `patience=10`.

## Source Files

| artifact | path |
| --- | --- |
| continuation helper | `n2ew_era/08a_fi2010_public_benchmark/12_continue_v3_fair512.py` |
| per-seed metrics | `n2ew_era/08a_fi2010_public_benchmark/results/stage1_metrics_v3_fair512.csv` |
| primary mean/std | `n2ew_era/08a_fi2010_public_benchmark/results/stage1_primary_metrics_v3_fair512.csv` |
| training curves | `n2ew_era/08a_fi2010_public_benchmark/results/training_curves_v3_fair512.csv` |
| heartbeat log | `n2ew_era/08a_fi2010_public_benchmark/results/training_heartbeat_v3_fair512.csv` |
| first run stdout | `n2ew_era/08a_fi2010_public_benchmark/results/continue_v3_fair512.log` |
| seed1 continuation stdout | `n2ew_era/08a_fi2010_public_benchmark/results/continue_v3_fair512_extend_seed1.log` |
| v2 baseline metrics | `n2ew_era/08a_fi2010_public_benchmark/results/stage1_metrics_v2.csv` |
| v2 baseline primary mean/std | `n2ew_era/08a_fi2010_public_benchmark/results/stage1_primary_metrics_v2.csv` |
| batch128 speedprobe | `n2ew_era/08a_fi2010_public_benchmark/results/stage1_metrics_v3_k50_speedprobe.csv` |

## Preflight

| check | result |
| --- | --- |
| Python launcher | `.venv/Scripts/python.exe` |
| venv base executable | `C:\Users\holbo\AppData\Local\Programs\Python\Python313\python.exe` |
| venv Python version | 3.13.7 |
| Torch version | 2.10.0+cpu, confirmed after completion with `.venv/Scripts/python.exe` |
| visible logical cores | 8 via `[Environment]::ProcessorCount` |
| torch thread limit | 4 for this continuation run |
| `decay_param` is `nn.Parameter` | confirmed in `fi2010_phc_lite.py:55` |
| `alpha_scalar` is `nn.Parameter` | confirmed in `fi2010_phc_lite.py:56` |
| `--fast-exact-final-step` v3 guard | confirmed in `02_train_phc_lite_stage1.py:663`; v3 raises `RuntimeError` |

## Execution Notes

`02_train_phc_lite_stage1.py --resume-existing` skips rows that already exist; it does not continue optimizer training from the prior checkpoint. To avoid changing the frozen training script, `12_continue_v3_fair512.py` imports the existing training helpers and adds a minimal continuation path.

Checkpoint caveat: existing checkpoints store the best model state, but not the optimizer state or the final epoch state. Continuation therefore loads the best checkpoint, restores the epoch counter and patience state from `training_curves_v3_fair512.csv`, and reinitializes Adam. This is recorded in each checkpoint under `resume_note`.

Run sequence:

| step | command intent | result |
| --- | --- | --- |
| first continuation | seed0 resume from epoch30, then seed1/seed2 sequentially, `max_epochs=60` | seed0 closed at epoch32; seed1 hit epoch60 boundary and was not closed; seed2 closed at epoch56 |
| second continuation | seed1 only, resume from epoch60 state, `max_epochs=120` | seed1 closed at epoch89 |

## Per-Seed Results

All values below come from `results/stage1_metrics_v3_fair512.csv`.

| seed | resumed_from_epoch | best_epoch | final_epoch | convergence_closed | val macro-F1 | test macro-F1 | decay mean | decay min | decay max | alpha_scalar |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 30 | 22 | 32 | True | 57.2212% | 62.3107% | 0.888295 | 0.863301 | 0.901553 | 1.177070 |
| 1 | 60 | 79 | 89 | True | 59.8384% | 64.9395% | 0.874837 | 0.834907 | 0.925939 | 1.433792 |
| 2 | 0 | 46 | 56 | True | 59.7130% | 64.6743% | 0.880751 | 0.784226 | 0.917498 | 1.244949 |

## Primary Mean/Std

All mean/std values are three-seed primary rows.

| model/version | batch_size | n_seeds | val macro-F1 mean +/- std | test macro-F1 mean +/- std | source |
| --- | ---: | ---: | ---: | ---: | --- |
| v2 k=50 raw baseline | 512 | 3 | 34.7658% +/- 3.9828% | 40.7194% +/- 2.9836% | `results/stage1_primary_metrics_v2.csv` |
| v3 fair512 k=50 raw | 512 | 3 | 58.9242% +/- 1.4762% | 63.9748% +/- 1.4473% | `results/stage1_primary_metrics_v3_fair512.csv` |
| v3 - v2 | 512 | 3 | +24.1584 percentage points | +23.2554 percentage points | computed from the two primary CSVs |

## Three-Way Seed0 Context

| run | batch_size | best/final epoch | status | val macro-F1 | test macro-F1 | source |
| --- | ---: | --- | --- | ---: | ---: | --- |
| v2 k=50 seed0 raw baseline | 512 | 15 / 25 | convergence_closed | 36.6488% | 42.4444% | `results/stage1_metrics_v2.csv` |
| v3 speedprobe k=50 seed0 raw | 128 | 1 / 1 | one epoch only, not converged | 43.3307% | 50.1784% | `results/stage1_metrics_v3_k50_speedprobe.csv` |
| v3 fair512 k=50 seed0 raw | 512 | 22 / 32 | convergence_closed | 57.2212% | 62.3107% | `results/stage1_metrics_v3_fair512.csv` |

## Decay Channels

Values are effective sigmoid decay values at the selected best checkpoint.

| channel | seed0 | seed1 | seed2 |
| ---: | ---: | ---: | ---: |
| 0 | 0.863301396 | 0.847924769 | 0.868023872 |
| 1 | 0.901552975 | 0.858095288 | 0.914846599 |
| 2 | 0.899511755 | 0.895963132 | 0.917497694 |
| 3 | 0.875013351 | 0.918575644 | 0.885245085 |
| 4 | 0.896808267 | 0.872944593 | 0.899694920 |
| 5 | 0.894015968 | 0.907400668 | 0.888747156 |
| 6 | 0.899114192 | 0.900661767 | 0.784225583 |
| 7 | 0.895924270 | 0.846423149 | 0.858694077 |
| 8 | 0.899886131 | 0.846592128 | 0.858082175 |
| 9 | 0.878883183 | 0.845021009 | 0.896004975 |
| 10 | 0.898501039 | 0.878803730 | 0.903969944 |
| 11 | 0.878882766 | 0.872541249 | 0.861171782 |
| 12 | 0.899145544 | 0.925938964 | 0.910521388 |
| 13 | 0.864589453 | 0.834907353 | 0.871383429 |
| 14 | 0.870543599 | 0.857754886 | 0.862797499 |
| 15 | 0.897040308 | 0.887843013 | 0.911109805 |

## Heartbeat Diagnostics

`training_heartbeat_v3_fair512.csv` contains 147 heartbeat rows. The anomaly detector used the recorded epoch wall times only: baseline epoch time 110.637387 seconds, anomaly threshold 331.912160 seconds, and `duration_anomaly_gt_3x_baseline=True` for 26 epochs.

| seed | heartbeat rows | anomaly rows | median epoch seconds | max epoch seconds |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 0 | 263.058247 | 263.058247 |
| 1 | 89 | 10 | 95.589725 | 1271.366407 |
| 2 | 56 | 16 | 89.461759 | 703.760890 |

Notable anomaly spans:

| seed | epochs | wall-clock range | note |
| ---: | --- | --- | --- |
| 2 | 41-56 | 2026-08-05T15:28:38 to 2026-08-05T18:08:25 | sustained slowdown; each epoch exceeded 3x baseline |
| 1 | 61, 62, 70, 71, 83-88 | 2026-08-05T18:17:22 to 2026-08-05T21:17:23 | intermittent slowdown during continuation |

Additional context: during diagnosis, two pre-existing Anaconda Python processes from 2026-07-20 (`pid=16908`, `pid=30464`, outside this task) were still consuming CPU. They were not terminated because they were unrelated user/system work. The heartbeat anomalies should therefore be treated as wall-clock contamination, not model convergence behavior.

The first 30 seed0 epochs were from the earlier partial run and did not have heartbeat timestamps. The curve CSV does show the previously observed elapsed-time jump from epoch28 to epoch29: `17963.266848` seconds to `33445.398546` seconds, a delta of `15482.131697` seconds.

## Honest Conclusion

The fair512 control is now complete for `k=50/raw` across seeds 0, 1, and 2. With batch size matched to v2 at 512 and natural patience early stopping reached for all three seeds, v3 fair512 achieves val macro-F1 `58.9242% +/- 1.4762%` and test macro-F1 `63.9748% +/- 1.4473%`, compared with v2 `34.7658% +/- 3.9828%` val and `40.7194% +/- 2.9836%` test. The batch-size confound has been removed for this comparison. The main caveat is execution hygiene: heartbeat logging found substantial wall-clock slowdowns from external CPU contention, but these affect elapsed time, not the selected validation/test metrics.
