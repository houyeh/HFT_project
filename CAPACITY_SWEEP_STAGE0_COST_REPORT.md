# FI-2010 PHC-lite Capacity Sweep: Stage 0 Cost Report

Status: `stage0_complete_waiting_for_approval`

No training, evaluation, checkpoint conversion, or hardware work was started for this audit. This report only reads the existing FI-2010 public-benchmark scripts and outputs under `n2ew_era/08a_fi2010_public_benchmark/`.

## A. Fixed Training Recipe

The capacity sweep must hold every item below fixed and vary only `hidden_dim`. The reusable v3 runs use a common optimization rule, although their configured epoch ceilings differ because seed 1 was later extended to close convergence.

| item | v3 fair512 hidden16 | hidden64 capacity run | proposed fixed sweep rule | source |
|---|---:|---:|---:|---|
| horizon | 50 | 50 | 50 | `results/stage1_metrics_v3_fair512.csv`; `13_train_hidden64_capacity.py` |
| loss | raw `CrossEntropyLoss()` | raw `CrossEntropyLoss()` | raw `CrossEntropyLoss()` | `02_train_phc_lite_stage1.py:279-282`; `13_train_hidden64_capacity.py:240` |
| batch size | 512 | 512 | 512 | metrics CSVs; script defaults |
| evaluation batch size | 1024 | 1024 | 1024 | script argument definitions |
| optimizer | Adam | Adam | Adam | both scripts |
| learning rate | 0.001 | 0.001 | 0.001 | script argument definitions |
| LR schedule | none | none | none | no scheduler is constructed in either training path |
| weight decay | 0 | 0 | 0 | script argument definitions |
| gradient clipping | 5.0 | 5.0 | 5.0 | script argument definitions |
| early-stop patience | 10 | 10 | 10 | metrics CSV; `13_train_hidden64_capacity.py` |
| minimum improvement | 1e-5 | 1e-5 | 1e-5 | script argument definitions |
| epoch ceiling | seeds 0/2: 60; seed 1: 120 | 120 | 120 for all new runs | `results/stage1_metrics_v3_fair512.csv`; `results/stage1_metrics_v3_hidden64_capacity.csv` |
| device / Torch threads | CPU; historical thread setting varies by launcher | CPU; 2 threads | must be fixed before Stage 1 | launch records and `13_train_hidden64_capacity.py` |

Using `max_epochs=120` for new runs preserves the same patience-based stopping rule while avoiding the seed-dependent 60/120 ceiling. The existing hidden16 seed 0 and seed 2 runs stopped at epochs 32 and 56, respectively, before their 60-epoch ceilings; seed 1 stopped at epoch 89 after its ceiling was extended to 120. The existing hidden64 seed 0 run stopped at epoch 37. Therefore all four reusable checkpoints closed under the same `patience=10` rule rather than at their epoch boundary. This is compatible with reuse, but Stage 1 should record the historical ceiling difference explicitly.

## B. Reusable Checkpoints

The trainable-parameter formula implied by `fi2010_phc_lite.py` is `5h^2 + 53h + 4`, where `h` is `hidden_dim`.

| hidden | seed | checkpoint | parameters | bytes | reuse status |
|---:|---:|---|---:|---:|---|
| 16 | 0 | `n2ew_era/08a_fi2010_public_benchmark/models/phc_fi2010_h50_seed0_raw_v3_fair512.pt` | 2,132 | 16,614 | reuse |
| 16 | 1 | `n2ew_era/08a_fi2010_public_benchmark/models/phc_fi2010_h50_seed1_raw_v3_fair512.pt` | 2,132 | 16,742 | reuse |
| 16 | 2 | `n2ew_era/08a_fi2010_public_benchmark/models/phc_fi2010_h50_seed2_raw_v3_fair512.pt` | 2,132 | 16,742 | reuse |
| 64 | 0 | `n2ew_era/08a_fi2010_public_benchmark/models/phc_fi2010_h50_seed0_raw_v3_hidden64_capacity.pt` | 23,876 | 104,416 | reuse |

No compatible hidden8, hidden32, hidden64 seed1/2, or hidden128 capacity-sweep checkpoints were found.

The spatial-front-end control already has three PHC seeds. Its total parameter count is 2,420, or 288 more than hidden16. Existing results are in `results/spatial_frontend_probe_metrics.csv`; no additional spatial run is required at Stage 1.

## C. New Runs and Runtime Estimate

| hidden | parameters | reusable runs | new runs required |
|---:|---:|---:|---:|
| 8 | 748 | 0 | 3 |
| 16 | 2,132 | 3 | 0 |
| 32 | 6,820 | 0 | 3 |
| 64 | 23,876 | 1 | 2 |
| 128 | 88,708 | 0 | 3 |
| **total** | | **4** | **11** |

Existing heartbeat measurements are noisy and contain resource-contention outliers:

| run family | representative epoch time | observed anomaly | interpretation |
|---|---:|---:|---|
| hidden16 seed1 | median 95.59 s | max 1,271.37 s | typical clean epoch is about 1.5 minutes; long stalls occurred |
| hidden16 seed2 | median 88.98 s | max 703.76 s | similar clean rate with stalls |
| hidden64 seed0 | median 319.60 s | max 23,844.37 s | the recorded 10 h 10 m wall time is contaminated by a very long stall |

Based on the clean medians, early-stop histories, and quadratic hidden-state work, the pre-run planning range is:

| hidden | estimated wall time per new run | new runs | subtotal |
|---:|---:|---:|---:|
| 8 | 1-2 h | 3 | 3-6 h |
| 32 | 2-4 h | 3 | 6-12 h |
| 64 | 4-6 h clean; up to about 10 h under observed contention | 2 | 8-20 h |
| 128 | 10-20 h | 3 | 30-60 h |
| **total** | | **11** | **47-98 h** |

This is a planning estimate, not a benchmark. The upper range can be exceeded if the prior contention/stall behavior recurs. Runs must be serialized as requested. A short hidden128 speed probe before committing to all three runs would tighten the estimate without changing the training recipe, but it should only be performed after approval.

## D. Disk Estimate

The four reusable checkpoints occupy about 155 KB in total. Parameter-count scaling suggests approximately 10 KB for hidden8, 30-40 KB for hidden32, about 105 KB for hidden64, and roughly 360 KB for hidden128 per checkpoint. Eleven new checkpoints should therefore remain near 1.2-1.5 MB. Curves, metrics, logs, figures, and reports should remain below 10 MB under normal CSV logging. A conservative reservation of **50 MB** is sufficient; the existing FI-2010 cache is reused and is not duplicated.

## Stage 0 Decision Gate

- New training runs required: **11**.
- Expected serialized wall time: **47-98 hours**, with contention risk above that range.
- Expected new storage: **less than 50 MB**.
- Reuse compatibility: **acceptable**, provided Stage 1 fixes batch size 512, raw loss, Adam at 1e-3, no LR scheduler, weight decay 0, gradient clip 5, and patience 10, while documenting the historical 60/120 epoch-ceiling difference.
- Current action: **STOP. Await approval before creating a sweep script or launching any training.**

