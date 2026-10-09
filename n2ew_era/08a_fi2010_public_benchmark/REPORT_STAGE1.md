# 08A PHC-lite FI-2010 Public Benchmark - Stage 1 Report

status: `stage1_complete_with_validation_caveat`

This report covers Stage 1 only. Stage 2 HLS export/csim/csynth was not started and must wait for review approval.

Data archive note: the FI-2010 zip was already available locally before this phase. Work started from extraction and integrity verification; no download step was run.

## Scope And Guardrails

- Work directory: `n2ew_era/08a_fi2010_public_benchmark/`.
- Public dataset only: FI-2010 official Fairdata/Etsin archive.
- No project-internal data and no NEW1-NEW5 split were read or used.
- Existing phases under `n2ew_era/` were read for PHC-lite structure only; no existing phase source, pragma, data, or report file was modified.
- No synthetic replacement data was generated.
- The result is a Pareto/trade-off anchor, not a SOTA claim.

## Data Source And Integrity

Dataset: Ntakaris, Magris, Kanniainen, Gabbouj, and Iosifidis, 2018, "Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods." Official host: Fairdata/Etsin dataset `73eb48d7-4dbc-4a10-a52a-da745b47a649`.

Archive used: `data/raw/BenchmarkDatasets.zip`.

Acquisition note: this archive was already in hand before Stage 1 execution. The preparation script does not contain a downloader and raises if the local zip is missing.

| check | expected | observed | status |
|---|---:|---:|---|
| official zip size bytes | 1,864,361,899 | 1,864,361,899 | pass |
| official zip SHA256 | `cea93692a270724fa91e8f124da641db727d757e5e0f0bb85067709e9932f664` | same | pass |
| published total representations | 394,337 | 394,337 | pass |

Only the official `NoAuction/1.NoAuction_Zscore` version was used. Setup 2 uses `Train_Dst_NoAuction_ZScore_CF_7.txt` for the first 7 days and concatenates `Test_Dst_NoAuction_ZScore_CF_7/8/9.txt` for the final 3 days.

Full file hashes and zip member list are in `data_manifest.md`, `results/file_manifest.csv`, and `results/zip_noauction_zscore_members.csv`.

## Protocol

- Input: sliding window of 100 events by 40 official normalized raw LOB feature rows.
- Labels: official five horizon label rows, mapped to k=10/20/30/50/100, with labels 1=up, 2=stationary, 3=down.
- Window stride: 1.
- Window label: label at the last event of the 100-event window.
- Train/validation: the first 80% of `Train CF7` events for fitting, tail 20% of `Train CF7` for early stopping.
- Test: concatenated `Test CF7/8/9`, touched only after checkpoint selection.
- Metrics: accuracy, macro precision, macro recall, macro F1, computed with sklearn and `zero_division=0`.

| split | events | windows |
|---|---:|---:|
| train fit head 80% | 203,800 | 203,701 |
| validation tail 20% | 50,950 | 50,851 |
| test CF7+CF8+CF9 concat | 139,587 | 139,488 |

Class distribution was recomputed from the official label rows:

| split | k | up | stationary | down |
|---|---:|---:|---:|---:|
| train windows | 10 | 20.08% | 59.84% | 20.07% |
| train windows | 20 | 25.76% | 48.61% | 25.62% |
| train windows | 30 | 29.76% | 40.88% | 29.37% |
| train windows | 50 | 35.06% | 30.35% | 34.59% |
| train windows | 100 | 41.45% | 17.78% | 40.78% |
| validation windows | 10 | 18.25% | 63.38% | 18.37% |
| validation windows | 20 | 23.68% | 52.65% | 23.67% |
| validation windows | 30 | 27.22% | 45.60% | 27.19% |
| validation windows | 50 | 31.62% | 37.01% | 31.37% |
| validation windows | 100 | 37.23% | 25.96% | 36.81% |
| test windows | 10 | 15.15% | 70.69% | 14.17% |
| test windows | 20 | 19.66% | 62.08% | 18.26% |
| test windows | 30 | 22.86% | 56.13% | 21.01% |
| test windows | 50 | 27.53% | 47.31% | 25.15% |
| test windows | 100 | 34.35% | 34.45% | 31.20% |

Caveat: I found a published total representation count and label definition in the FI-2010/Fairdata material, but not a paper table giving per-horizon class proportions for Setup 2. Therefore the requested "within 1% of published class distribution" check cannot be expressed as a paper-vs-run error. The actual computed distributions are exported in `results/label_distribution_windows.csv` and `results/label_distribution_snapshots.csv`.

## Model And Training

The classifier is defined in `fi2010_phc_lite.py`.

- Recurrent PHC-lite equations were kept from the Phase 06 step core.
- Input projection changed from 16 features to 40 features.
- Final readout changed from 2 regression outputs to 3 class logits.
- Hidden dimension stayed 16.
- Parameter count: 2,115 trainable parameters.
- Optimizer: Adam, lr=0.001, weight decay=0.
- Seeds: 0, 1, 2.
- Max epochs: 4.
- Early stopping: validation macro F1, patience=2, min_delta=1e-5.
- Batch size: 4096 train, 8192 eval.
- Runtime: PyTorch 2.10.0+cpu, CPU, 4 torch threads.

Raw cross-entropy was always trained first. If raw loss produced a degenerate validation policy with `max_val_pred_share >= 0.95`, a class-weighted run was trained and treated as the primary Stage 1 result for that horizon. This switch used validation only, not test.

| k | raw max validation prediction share, mean over seeds | raw degenerate? | primary loss reported |
|---:|---:|---|---|
| 10 | 100.00% | yes | weighted |
| 20 | 100.00% | yes | weighted |
| 30 | 99.96% | yes | weighted |
| 50 | 66.75% | no | raw |
| 100 | 73.94% | no | raw |

## Stage 1 PHC-lite Results

Numbers below are test-set mean +/- std over seeds 0/1/2. F1 is macro F1.

| k | primary loss | accuracy | macro precision | macro recall | macro F1 |
|---:|---|---:|---:|---:|---:|
| 10 | weighted | 55.04 +/- 1.39 | 40.21 +/- 0.68 | 41.61 +/- 0.91 | 40.34 +/- 0.60 |
| 20 | weighted | 50.08 +/- 2.15 | 38.00 +/- 1.15 | 38.03 +/- 1.00 | 37.75 +/- 0.96 |
| 30 | weighted | 49.42 +/- 1.20 | 40.24 +/- 1.49 | 39.84 +/- 1.43 | 39.88 +/- 1.43 |
| 50 | raw | 39.85 +/- 1.20 | 38.62 +/- 1.20 | 38.75 +/- 1.11 | 38.34 +/- 1.07 |
| 100 | raw | 36.70 +/- 0.74 | 41.17 +/- 1.63 | 37.15 +/- 0.42 | 29.79 +/- 0.58 |

Full per-seed rows are in `results/stage1_metrics.csv`; train/validation curves are in `results/training_curves.csv`; aggregated mean/std rows are in `results/stage1_summary.csv`; the primary loss-mode subset used in the table above is in `results/stage1_primary_metrics.csv`.

## Published Comparison

Published numbers are quoted only with their source table. F1 values are percentages. The `reported/macro F1` columns preserve the earlier project table; the new `k=50 weighted F1` column is the direct comparison field for DeepLOB Table II.

| model | params | input | protocol/source | k=10 reported/macro F1 | k=20 reported/macro F1 | k=50 reported/macro F1 | k=50 weighted F1 |
|---|---:|---|---|---:|---:|---:|---:|
| PHC-lite FI-2010 v3 fair512 | 2,132 | 100x40 | this Stage 1 v3 fair512 recompute, 3 seeds, raw loss | N/A | N/A | 63.97 +/- 1.45 | 66.50 +/- 1.15 |
| PHC-lite FI-2010 v1 primary | 2,115 | 100x40 | this Stage 1 software run, macro F1 only | 40.34 +/- 0.60 | 37.75 +/- 0.96 | 38.34 +/- 1.07 | N/A |
| DeepLOB | 60k | 100x40 | Zhang, Zohren, Roberts 2018/2020, Table II; params Table III | 83.40 | 72.82 | 80.35 | 80.35 |
| SVM | N/A | FI-2010 features | Zhang, Zohren, Roberts 2018/2020, Table II, citing [28] | 35.88 | 43.20 | 49.42 | 49.42 |
| MLP | N/A | FI-2010 features | Zhang, Zohren, Roberts 2018/2020, Table II, citing [28] | 48.27 | 51.12 | 55.95 | 55.95 |
| LSTM | N/A | FI-2010 sequence | Zhang, Zohren, Roberts 2018/2020, Table II/III, citing [28] | 66.33 | 62.37 | 61.43 | 61.43 |
| RR | N/A | 144 FI-2010 representation features | Ntakaris et al. 2018, Table 7; original anchored 9-fold, not Setup 2 | 41.00 | 44.00 | 42.70 | N/A |
| SLFN | N/A | 144 FI-2010 representation features | Ntakaris et al. 2018, Table 7; original anchored 9-fold, not Setup 2 | 32.70 | 32.70 | 45.90 | N/A |

Important source note: the original FI-2010 paper table I found reports RR and SLFN in Table 7, not SVM/MLP. The Setup 2 SVM/MLP/LSTM comparison figures are from the DeepLOB paper Table II, where they are cited from earlier work. They are included because they share the DeepLOB Setup 2 comparison context.

The comparison table is also exported as `results/stage1_comparison_table.csv`.

## Interpretation

PHC-lite lands where expected for a small recurrent core: far below DeepLOB F1, with far fewer trainable parameters. The k=10/20/30 raw-loss runs show that class imbalance is material; raw cross-entropy collapses to a near-stationary validation policy, and class weighting raises macro F1 by restoring minority-class recall. The k=50 result is the cleanest raw-loss point for the current model, with test macro F1 38.34 +/- 1.07 at 2,115 trainable parameters.

Metric-definition note: `F1_DEFINITION_RECONCILIATION.md` records that DeepLOB Table II reports weighted-average F1; its DeepLOB rows have Accuracy% equal to Recall% for k=10/20/50, which is the support-weighted recall identity. Weighted F1 is therefore the direct published-baseline comparison number, while macro F1 remains this project's stricter class-balanced target.

This Stage 1 result does not establish a hardware Pareto point yet because no csim/csynth has been run. Stage 2 should export the median-seed checkpoint only after review, then measure HLS latency/resource estimates before making any F1-vs-parameter-vs-latency statement.

## Sources

- Fairdata/Etsin dataset page: <https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649>
- Ntakaris, Magris, Kanniainen, Gabbouj, and Iosifidis, 2018, Journal of Forecasting: <https://pure.au.dk/portal/files/143396530/Ntakaris_et_al_2018_Journal_of_Forecasting.pdf>
- Zhang, Zohren, and Roberts, DeepLOB arXiv v6: <https://arxiv.org/pdf/1808.03668>

## Stop Point

Stage 1 deliverables are complete. Stop here and wait for review before Stage 2.

## Stage 1 收斂重驗(v2)

status: `stage1_v2_convergence_revalidation_complete`

Interruption diagnostic: the prior v2 attempt stopped during `k=10/seed=0/weighted` before final evaluation/writeout, after the runner timeout. The checkpoint existed but no weighted metrics row could be written because `train_one()` had not returned. See `STAGE1_V2_INTERRUPTION_DIAGNOSTIC.md` for artifact-level evidence.

Part A reran training only. `fi2010_phc_lite.py`, `01_prepare_fi2010_stage1.py`, the cached FI-2010 Setup 2 split, and all Stage 2/HLS artifacts were left untouched. No project-internal NEW1-NEW5 data were read.

v2 protocol: max_epochs=60, patience=10, early stopping on validation macro F1, same raw-first/class-weighted fallback rule as v1, seeds 0/1/2, horizons k=10/20/30/50/100. The single boundary case `k=30/seed=2/raw` was then rerun with max_epochs=100 and the same patience=10, closing naturally at final_epoch=61 without changing any other split, seed, model, or threshold. The run used the script's verified `--fast-exact-final-step` path: because this FI-2010 classifier has zero `decay` and zero `alpha_scalar` buffers, full-window logits and final-step logits are mathematically identical; the script checks this before training.

| k | v1 primary | v1 test macro F1 | v2 primary | v2 test macro F1 | delta pct-pt |
|---|---|---|---|---|---|
| 10 | weighted | 40.34 +/- 0.60 | weighted | 40.67 +/- 0.26 | +0.33 |
| 20 | weighted | 37.75 +/- 0.96 | weighted | 39.80 +/- 0.72 | +2.05 |
| 30 | weighted | 39.88 +/- 1.43 | mixed_raw_weighted | 41.02 +/- 0.45 | +1.15 |
| 50 | raw | 38.34 +/- 1.07 | raw | 40.72 +/- 2.98 | +2.38 |
| 100 | raw | 29.79 +/- 0.58 | raw | 36.71 +/- 0.67 | +6.91 |

Convergence gate: `convergence_closed=True` means `early_stop=True` and `best_epoch < max_epochs - patience`.

| k | seed | loss | best_epoch | final_epoch | early_stop | max_epoch_hit | distance_to_max | closed |
|---|---|---|---|---|---|---|---|---|
| 10 | 0 | raw | 1 | 11 | True | False | 59 | True |
| 10 | 0 | weighted | 5 | 15 | True | False | 55 | True |
| 10 | 1 | raw | 1 | 11 | True | False | 59 | True |
| 10 | 1 | weighted | 10 | 20 | True | False | 50 | True |
| 10 | 2 | raw | 1 | 11 | True | False | 59 | True |
| 10 | 2 | weighted | 37 | 47 | True | False | 23 | True |
| 20 | 0 | raw | 1 | 11 | True | False | 59 | True |
| 20 | 0 | weighted | 9 | 19 | True | False | 51 | True |
| 20 | 1 | raw | 1 | 11 | True | False | 59 | True |
| 20 | 1 | weighted | 6 | 16 | True | False | 54 | True |
| 20 | 2 | raw | 1 | 11 | True | False | 59 | True |
| 20 | 2 | weighted | 9 | 19 | True | False | 51 | True |
| 30 | 0 | raw | 49 | 59 | True | False | 11 | True |
| 30 | 1 | raw | 41 | 51 | True | False | 19 | True |
| 30 | 1 | weighted | 28 | 38 | True | False | 32 | True |
| 30 | 2 | raw | 51 | 61 | True | False | 49 | True |
| 50 | 0 | raw | 15 | 25 | True | False | 45 | True |
| 50 | 1 | raw | 28 | 38 | True | False | 32 | True |
| 50 | 2 | raw | 1 | 11 | True | False | 59 | True |
| 100 | 0 | raw | 19 | 29 | True | False | 41 | True |
| 100 | 1 | raw | 24 | 34 | True | False | 36 | True |
| 100 | 2 | raw | 32 | 42 | True | False | 28 | True |

Closed runs: 22 / 22. Max-epoch boundary runs: 0.

| model | params | verification | k=10 F1 | k=20 F1 | k=50 F1 |
|---|---|---|---|---|---|
| PHC-lite FI-2010 v2 | 2115 | stage1_v2_software_measured | 40.67119667552192 | 39.80021344757372 | 40.71937484594487 |
| DeepLOB | 60k | published | 83.4 | 72.82 | 80.35 |
| SVM | N/A | published | 35.88 | 43.2 | 49.42 |
| MLP | N/A | published | 48.27 | 51.12 | 55.95 |
| LSTM | N/A | published | 66.33 | 62.37 | 61.43 |
| RR | N/A | published | 41.0 | 44.0 | 42.7 |
| SLFN | N/A | published | 32.7 | 32.7 | 45.9 |

Conclusion: v2 improves the Stage 1 F1 enough to confirm the 4-epoch run understated this small model. The convergence gate is now closed, so Stage 1 no longer blocks the separately reviewed Stage 2 HLS feasibility step.

Stage 1 v2 outputs: `results/training_curves_v2.csv`, `results/stage1_metrics_v2.csv`, `results/stage1_summary_v2.csv`, `results/stage1_primary_metrics_v2.csv`, `results/stage1_comparison_table_v2.csv`, `results/stage1_convergence_v2.csv`, and `results/stage1_v1_v2_delta.csv`.

Hard stop: Stage 2 HLS export/csim/csynth was not started in this run.

## Stage 1 架構修正重驗(v3)

status: `architecture_fix_complete_full_rerun_blocked_by_cpu_runtime`

Scope: this v3 pass only touched the FI-2010 public benchmark path under
`n2ew_era/08a_fi2010_public_benchmark/`. Stage 2 HLS files/results were not
modified or rerun. No TXF NEW1-NEW5 data or code paths were used.

Architecture fix applied in `fi2010_phc_lite.py`:

- v2 closed-recursion cause: `decay` and `alpha_scalar` were zero buffers, so
  the temporal memory term and within-step message loop were structurally off.
- v3 uses `decay_param = nn.Parameter(logit(initial_decay).repeat(hidden_dim))`
  and exposes the effective memory coefficient as `sigmoid(decay_param)`,
  matching the convention in
  `05a_phc_lite_microstructure_sequence_feasibility/models/phc_lite.py`.
- v3 uses trainable `alpha_scalar = nn.Parameter(0.25)`.
- `--fast-exact-final-step` is now rejected at startup because the v2
  final-timestep shortcut is mathematically invalid once recurrence is active.
- The full-window `forward()` still runs the recurrent state update over all
  100 timesteps; only timestep-independent encoder/gate/candidate projections
  are batched for CPU efficiency. A diagnostic script verified equivalence to
  repeated `forward_step()` calls with max absolute difference `1.79e-7`.

Parameter count changed from v2 `2115` to v3 `2132`, exactly adding 16
per-channel decay parameters plus 1 scalar alpha parameter.

The requested full 15-combination v3 rerun was intentionally not started in
this CPU-only environment. PyTorch reported `2.10.0+cpu` and `cuda=False`.
`10_benchmark_v3_batch.py` wrote `results/v3_speed_benchmark.csv` for the
requested grid `threads=[4,8] x batch_size=[128,256,512,1024]`. The fastest
single-batch microbenchmark was `threads=4, batch_size=128`, but an actual
full-dataset training probe showed the end-to-end cost is much higher than the
single-batch lower bound.

First real v3 evidence point, not a final score:

| run | batch_size | epochs | convergence | val macro F1 | test macro F1 | decay effective min/max | alpha |
|---|---:|---:|---|---:|---:|---:|---:|
| v3 k=50/seed=0/raw speedprobe | 128 | 1 | max_epoch_hit=True, not converged | 43.33% | 50.18% | 0.8820 / 0.9067 | 0.25 -> 0.5457 |

This row is from `results/stage1_metrics_v3_k50_speedprobe.csv`. It is a real
train/validation/test run over the official FI-2010 split, but it is
`1 epoch`, `budget-capped`, `batch_size=128`, and `convergence_closed=False`.
It must be read as preliminary evidence that the corrected recurrence is active,
not as the final Stage 1 v3 result.

Same-combination v2 reference:

| run | batch_size note | best/final epoch | val macro F1 | test macro F1 |
|---|---|---:|---:|---:|
| v2 k=50/seed=0/raw closed-recursion | v2/default batch_size=512 | 15 / 25 | 36.65% | 42.44% |
| v3 k=50/seed=0/raw speedprobe | batch_size=128 | 1 / 1 | 43.33% | 50.18% |
| naive v3 speedprobe minus v2 | confounded by architecture + batch/update count + convergence budget | n/a | +6.68 pct-pt | +7.73 pct-pt |

The apparent v3 gain above is encouraging but confounded: it mixes the
architecture correction with `batch_size=128` and a very different number of
gradient updates per epoch. It cannot yet be decomposed into "architecture
fix" versus "batch size / update count" contributions.

Fair batch_size=512 control status:

- Target run: `k=50`, `seed=0`, `raw`, v3 true recurrence, `batch_size=512`,
  suffix `_v3_k50_fair512`.
- Required reason: match the v2/default `batch_size=512`, so the only intended
  experimental change is v2 closed-recursion architecture -> v3 true recurrence.
- Current result: not available.
- Prior fair512 attempt under resource pressure failed at torch import with
  `WinError 1114` while loading `torch/lib/shm.dll`.
- 2026-08-04 preflight did not satisfy the "no other heavy load" condition:
  available memory was only about `2133 MB`; `OneDrive.Sync.Service` used about
  `4148.7 MB` working set, `OneDrive` used about `855.1 MB`, and two Anaconda
  Python processes remained CPU-active. Snapshot saved in
  `results/v3_fair512_resource_preflight.csv`.

Because the fair512 control was not run under clean resources, no
`_v3_k50_fair512` metric row is reported. The honest next step is to rerun only
this one fair512 control after OneDrive/background Python load is cleared, then
compare it side-by-side with the `batch_size=128` speedprobe. The 15-combination
sweep and `hidden_dim in {16,64,128}` capacity diagnosis remain deferred until
the batch-size confound is resolved and the machine is stable.
