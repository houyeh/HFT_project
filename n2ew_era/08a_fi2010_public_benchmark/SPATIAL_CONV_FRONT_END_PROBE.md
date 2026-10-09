# Spatial Conv Front-End Probe

status: software_probe_3seed_batch512_natural_early_stop_no_slow_abort

## Scope

This is a k=50 FI-2010 software training probe for PHC+spatial and TCN+spatial, expanded to seed0/seed1/seed2. It does not touch TXF NEW1-NEW5, does not modify `fi2010_phc_lite.py` or `fi2010_tcn.py`, and does not run HLS, csynth, implementation, or board work.

## Front-End Design

- Input reshape: `40 -> 10 levels x 4 values`.
- Pair convolution: `Conv1d(1 -> C, kernel_size=2, stride=2)` within each level, pairing `(ask_price, ask_volume)` and `(bid_price, bid_volume)`.
- Adjacent-level convolution: reshape the two side-pair outputs into `2C` channels over 10 levels, then apply `Conv1d(2C -> C, kernel_size=2, stride=2)` across adjacent price levels.
- Activation: `SiLU` after each convolution.
- Probe setting: `C=8`, so the front end returns `5*C=40` features and the downstream PHC/TCN entry width stays comparable to the existing baselines.

## Preflight

The preflight samples process CPU deltas over 5 seconds and records available memory before training. It is intended to catch Anaconda/OneDrive style background-load contamination before the run starts.

| label | time | avail_MB | pid | process | cpu_delta_5s | working_MB |
| --- | --- | --- | --- | --- | --- | --- |
| phc_post | 2026-10-03T01:12:05-0500 | 9037.5 | 29252 | iCloudPhotos.exe | 11.797 | 62.0 |
| phc_post | 2026-10-03T01:12:05-0500 | 9037.5 | 48576 | SearchIndexer.exe | 9.812 | 162.4 |
| phc_post | 2026-10-03T01:12:05-0500 | 9037.5 | 40796 | python.exe | 7.203 | 1334.4 |
| phc_pre | 2026-10-03T01:03:50-0500 | 8112.0 | 29252 | iCloudPhotos.exe | 8.859 | 62.2 |
| phc_pre | 2026-10-03T01:03:50-0500 | 8112.0 | 1264 | lsass.exe | 5.828 | 639.9 |
| phc_pre | 2026-10-03T01:03:50-0500 | 8112.0 | 40796 | python.exe | 4.453 | 293.4 |
| tcn_post | 2026-08-08T06:17:24+0800 | 13934.5 | 1332 | lsass.exe | 1.859 | 649.0 |
| tcn_post | 2026-08-08T06:17:24+0800 | 13934.5 | 17104 | python.exe | 1.719 | 924.0 |
| tcn_post | 2026-08-08T06:17:24+0800 | 13934.5 | 1752 | dwm.exe | 1.656 | 190.0 |
| tcn_pre | 2026-08-08T05:56:59+0800 | 14399.5 | 1752 | dwm.exe | 1.969 | 190.0 |
| tcn_pre | 2026-08-08T05:56:59+0800 | 14399.5 | 1332 | lsass.exe | 1.797 | 600.3 |
| tcn_pre | 2026-08-08T05:56:59+0800 | 14399.5 | 17104 | python.exe | 1.641 | 294.9 |

The table shows the latest `phc_pre/phc_post/tcn_pre/tcn_post` snapshots. Training-time process checks also found no active Vivado/HLS/xsim process names. Repeated labels can correspond to later seed runs; the full audit trail is in the CSV.

## Parameter Counts

| model | front_channels | front_output_dim | trainable_parameters | baseline_parameters | delta |
| --- | --- | --- | --- | --- | --- |
| phc | 8 | 40 | 2420 | 2132 | +288 |
| tcn | 8 | 40 | 2632 | 2344 | +288 |

## Baselines

| baseline | n | params | test_macro_F1 | test_weighted_F1 | source |
| --- | --- | --- | --- | --- | --- |
| PHC v3 hidden16 fair512 raw | 1 | 2132 | 62.31% +/- n/a | nan% +/- n/a | fallback constants in this script |
| TCN small raw | 1 | 2344 | 61.23% +/- n/a | nan% +/- n/a | fallback constants in this script |

## Per-Seed Results

| model | seed | batch | best | final | early_stop | slow_abort | max_epoch | resumed | val_macro | val_weighted | test_macro | test_weighted | seconds | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PHC+spatial | 0 | 512 | 39 | 49 | True | False | False | False | 60.12% | 60.60% | 65.98% | 68.33% | 5084.4 | models\phc_fi2010_h50_seed0_raw_spatial_c8_probe.pt |
| PHC+spatial | 1 | 512 | 53 | 63 | True | False | False | True | 59.98% | 60.45% | 65.55% | 67.98% | 281.1 | models\phc_fi2010_h50_seed1_raw_spatial_c8_probe.pt |
| PHC+spatial | 2 | 512 | 25 | 35 | True | False | False | False | 58.68% | 59.23% | 63.90% | 66.45% | 3189.5 | models\phc_fi2010_h50_seed2_raw_spatial_c8_probe.pt |
| TCN+spatial | 0 | 512 | 9 | 19 | True | False | False | False | 57.60% | 58.23% | 63.52% | 66.33% | 1213.4 | models\tcn_fi2010_h50_seed0_raw_spatial_c8_probe.pt |
| TCN+spatial | 1 | 512 | 25 | 35 | True | False | False | False | 58.49% | 59.07% | 64.85% | 67.46% | 2209.1 | models\tcn_fi2010_h50_seed1_raw_spatial_c8_probe.pt |
| TCN+spatial | 2 | 512 | 35 | 45 | True | False | False | False | 60.03% | 60.58% | 66.49% | 68.96% | 2830.2 | models\tcn_fi2010_h50_seed2_raw_spatial_c8_probe.pt |

## 3-Seed Summary

| model | n | batch_sizes | params | val_macro_F1 | val_weighted_F1 | test_macro_F1 | test_weighted_F1 | baseline | delta_macro | delta_weighted | all_early_stop | any_slow_abort | any_max_epoch | total_hours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PHC+spatial | 3 | 512 | 2420 | 59.59% +/- 0.79 | 60.09% +/- 0.75 | 65.14% +/- 1.10 | 67.58% +/- 1.00 | PHC v3 hidden16 fair512 raw | +2.83 pp | +nan pp | True | False | False | 2.38 |
| TCN+spatial | 3 | 512 | 2632 | 58.71% +/- 1.23 | 59.29% +/- 1.19 | 64.95% +/- 1.49 | 67.58% +/- 1.32 | TCN small raw | +3.72 pp | +nan pp | True | False | False | 1.74 |

## Archived Superseded Runs

These rows are retained only as provenance. The old slow-abort rows were interrupted by `slow_epoch_abort=True`; the old TCN seed0 clean row used batch_size=2048 and was superseded by the batch512 consistency rerun.

| model | seed | batch | archived_at | reason | best | final | slow_abort | test_macro | test_weighted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PHC+spatial | 0 | 512 | 2026-08-07T16:34:07+0800 | superseded_by_clean_rerun_after_vivado_overlap_slow_abort | 9 | 9 | True | 59.97% | 62.64% |
| TCN+spatial | 0 | 2048 | 2026-08-07T19:51:53+0800 | superseded_by_clean_rerun_after_vivado_overlap_slow_abort | 7 | 7 | True | 59.32% | 62.39% |
| TCN+spatial | 0 | 2048 | 2026-08-08T05:57:08+0800 | superseded_by_batch512_consistency_rerun_for_3seed | 38 | 48 | False | 63.74% | 66.44% |

## Decision Gate

PHC+spatial is positive in a 3-seed vs 3-seed comparison. TCN+spatial is also positive against the existing TCN-small baseline, but that baseline is seed0 only in the current repo, so do not label the TCN comparison as 3-seed-vs-3-seed robust until TCN-small seed1/seed2 baselines exist.

## Files

- Module: `spatial_conv_front_end.py`
- Training script: `14_train_spatial_frontend_probe.py`
- Metrics: `results/spatial_frontend_probe_metrics.csv`
- Archived metrics: `results/spatial_frontend_probe_metrics_history.csv`
- Curves: `results/spatial_frontend_probe_training_curves.csv`
- Preflight: `results/spatial_frontend_probe_resource_preflight.csv`
- Parameter counts: `results/spatial_frontend_probe_parameter_count.csv`

## Claim Boundary

- These numbers are FI-2010 PyTorch/software training results only.
- Weighted F1 uses the reconciled sklearn weighted-average definition: `precision_recall_fscore_support(..., average="weighted")`.
- No HLS/csynth/resource implementation claim is made here; "parameters" is the only resource-like quantity recorded in this probe.
- PHC baseline has a 3-seed fair512 source in this repo. TCN-small raw baseline currently has only seed0 in this repo, so any TCN 3-seed-vs-baseline statement must carry that limitation.
