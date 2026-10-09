# F1 Definition Reconciliation

Date: 2026-08-07

Updated: 2026-08-08 for PHC v3 hidden64 capacity weighted-F1 recompute.

Scope: FI-2010 only. This audit recomputed metrics from existing checkpoints; no
model was retrained and no TXF / NEW1-NEW5 artifact was read.

## DeepLOB Table II Evidence

Source: Zhang, Zohren, and Roberts, DeepLOB arXiv:1808.03668v6, Table II
("Setup 2: Experiment results for the FI-2010 dataset").

The DeepLOB row has `Accuracy % == Recall %` for every Setup 2 horizon listed
below. That equality is the support-weighted recall identity for single-label
multiclass classification, so the Table II F1 column should be treated as
weighted-average F1 for direct comparisons.

| horizon | DeepLOB accuracy % | DeepLOB precision % | DeepLOB recall % | DeepLOB F1 % | accuracy == recall |
|---:|---:|---:|---:|---:|---|
| 10 | 84.47 | 84.00 | 84.47 | 83.40 | yes |
| 20 | 74.85 | 74.06 | 74.85 | 72.82 | yes |
| 50 | 80.51 | 80.38 | 80.51 | 80.35 | yes |

## PHC-lite v3 Fair512, k=50

Recomputed from existing checkpoints:
`models/phc_fi2010_h50_seed*_raw_v3_fair512.pt`.

Output files:

- `results/stage1_metrics_v3_fair512_weighted_recompute.csv`
- `results/stage1_summary_v3_fair512_weighted_recompute.csv`

| seed | checkpoint | test accuracy % | test macro-F1 % | test weighted-F1 % | weighted recall - accuracy |
|---:|---|---:|---:|---:|---:|
| 0 | `phc_fi2010_h50_seed0_raw_v3_fair512.pt` | 65.24 | 62.31 | 65.18 | 0.0 |
| 1 | `phc_fi2010_h50_seed1_raw_v3_fair512.pt` | 67.28 | 64.94 | 67.30 | 0.0 |
| 2 | `phc_fi2010_h50_seed2_raw_v3_fair512.pt` | 66.84 | 64.67 | 67.01 | 0.0 |
| mean +/- std | 3 seeds | 66.45 +/- 1.07 | 63.97 +/- 1.45 | 66.50 +/- 1.15 | max 0.0 |

## TCN Scratch / Small, k=50

Existing TCN checkpoints were found only for seed 0, so these are per-checkpoint
values, not 3-seed aggregates.

Output files:

- `results/tcn_fi2010_metrics_weighted_recompute.csv`
- `results/tcn_fi2010_summary_weighted_recompute.csv`

| model | checkpoint | test accuracy % | test macro-F1 % | test weighted-F1 % | weighted recall - accuracy |
|---|---|---:|---:|---:|---:|
| TCN scratch | `tcn_fi2010_h50_seed0_scratch_best.pt` | 67.64 | 65.20 | 67.60 | 0.0 |
| TCN small raw | `tcn_fi2010_h50_seed0_small_raw_scratch_best.pt` | 63.92 | 61.23 | 64.02 | 0.0 |

## PHC-lite v3 Hidden64 Capacity, k=50

Recomputed from the existing checkpoint:
`models/phc_fi2010_h50_seed0_raw_v3_hidden64_capacity.pt`.

Output files:

- `results/stage1_metrics_v3_hidden64_capacity_weighted_recompute.csv`
- `results/stage1_summary_v3_hidden64_capacity_weighted_recompute.csv`

| seed | checkpoint | test accuracy % | test macro-F1 % | test weighted-F1 % | weighted recall - accuracy |
|---:|---|---:|---:|---:|---:|
| 0 | `phc_fi2010_h50_seed0_raw_v3_hidden64_capacity.pt` | 66.12 | 64.43 | 66.32 | 0.0 |

## Capacity-Matched Seed0 View, k=50

This table puts the three capacity-comparison rows requested for paper/report
cleanup side by side. All rows are test split, seed 0, existing-checkpoint
recompute only.

| model | trainable parameters | checkpoint | test macro-F1 % | test weighted-F1 % |
|---|---:|---|---:|---:|
| PHC v3 hidden16 fair512 | 2,132 | `phc_fi2010_h50_seed0_raw_v3_fair512.pt` | 62.31 | 65.18 |
| PHC v3 hidden64 capacity | 23,876 | `phc_fi2010_h50_seed0_raw_v3_hidden64_capacity.pt` | 64.43 | 66.32 |
| TCN small raw | 2,344 | `tcn_fi2010_h50_seed0_small_raw_scratch_best.pt` | 61.23 | 64.02 |

## Sanity Check

For every recomputed PHC-lite and TCN checkpoint above, sklearn
`average="weighted"` recall exactly matched `accuracy_score` on the same split
and predictions. The maximum recorded absolute difference was 0.0 on validation
and test.

## Conclusion

Weighted-F1 is the number that can be directly compared with DeepLOB Table II.
Macro-F1 is the project's stricter class-balanced evaluation target. Both should
be reported in the paper; reporting only the one that looks better would be
misleading.
