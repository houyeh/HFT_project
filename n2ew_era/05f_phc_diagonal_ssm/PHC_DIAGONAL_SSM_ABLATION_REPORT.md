# PHC Diagonal SSM Ablation Report

Final verdict: diagonal16 `regresses` vs dense16 baseline and `regresses` vs dense8 baseline on val/NEW3/NEW4.

Verdict status: `diagonal16_regresses_vs_dense16_and_regresses_vs_dense8`.

- vs dense16: At least one val/NEW3/NEW4 action PnL, causal-fill PnL, or oracle capture metric regresses by more than 2% versus dense16.
- vs dense8: At least one val/NEW3/NEW4 action PnL, causal-fill PnL, or oracle capture metric regresses by more than 2% versus dense8.

## Protocol

- Phase: `05f`.
- Model: `PHCDiagonalSSM` defined locally in `n2ew_era/05f_phc_diagonal_ssm/models/phc_diagonal_ssm.py`.
- Architecture: same encoder/readout shape as 05C4, hidden_dim=16, but dense recurrent gate/candidate/synapse/gain matrices are replaced by learned per-channel diagonal decay and gain.
- Training: from-scratch random initialization for each seed. No 05C2/05C3/05C4/05D/05E weights or optimizer state loaded.
- Controlled architecture change versus 05C4: recurrence structure only. Hidden_dim, seeds, max epochs, patience, batch size, LR, LR_MIN, CosineAnnealingLR, weight decay, feature list, target, and causal-fill salt are retained.
- Optimizer and scheduler states are saved in both best and last checkpoints.
- Selection split: validation only.
- NEW3/NEW4: diagnostic OOS only.
- NEW5: not used.
- No files under `n2ew_era/06f_phc_lite_manual_hls/` are modified or evaluated in this phase.
- Material-match criterion in this report: no key metric regresses by more than 2% versus the comparison baseline.
- This phase does not claim or estimate HLS latency for the diagonal model.

## Parameter Count Comparison

| model                        | recurrence           |   hidden_dim |   parameter_count |   feature_dim |   mtl_steps |   delta_vs_dense16 |   ratio_vs_dense16 |   delta_vs_dense8 |   ratio_vs_dense8 |
|:-----------------------------|:---------------------|-------------:|------------------:|--------------:|------------:|-------------------:|-------------------:|------------------:|------------------:|
| phc_diagonal_ssm_hidden16    | diagonal_per_channel |           16 |               370 |            16 |           1 |              -1361 |           0.213749 |              -177 |          0.676417 |
| phc_stp_v2_cosine_lr_dense16 | dense_stp            |           16 |              1731 |            16 |           1 |                  0 |           1        |              1184 |          3.16453  |
| phc_stp_v2_hidden8_dense8    | dense_stp            |            8 |               547 |            16 |           1 |              -1184 |           0.316002 |                 0 |          1        |

## Selected Policy

```json
{
  "model": "phc_diagonal_ssm",
  "variant": "PHC diagonal SSM recurrence ablation clean 200epoch cosine LR",
  "phase": "05f",
  "selection_split": "val",
  "seed": 2,
  "threshold": 0.0,
  "target": "excess_pnl_H20",
  "seq_len": 128,
  "stride": 32,
  "hidden_dim": 16,
  "mtl_steps": 1,
  "max_epochs": 200,
  "patience": 20,
  "lr_initial": 0.001,
  "lr_min": 1e-05,
  "lr_scheduler": "CosineAnnealingLR",
  "optimizer_restarted": false,
  "validation_metrics": {
    "split": "val",
    "policy_id": "phc_diagonal_ssm_seed2",
    "horizon": 20,
    "state_count": 82960,
    "selected_action_count": 61828,
    "selection_rate": 0.745274831243973,
    "bid_count": 32510,
    "ask_count": 29318,
    "no_quote_count": 21132,
    "total_excess_pnl_H20": 1771525.0,
    "avg_selected_excess": 28.65247137219383,
    "win_rate": 0.46461150287895453,
    "negative_rate": 0.5353399754156692,
    "max_drawdown": 16590.0,
    "positive_session_count": 6,
    "session_count": 6,
    "seed": 2,
    "threshold": 0.0,
    "best_epoch": 157,
    "selection_split": "val",
    "phase": "05f",
    "optimizer_restarted": false,
    "new3_used_for_selection": false,
    "new4_used_for_selection": false,
    "new5_used_for_selection": false
  },
  "new3_used_for_selection": false,
  "new4_used_for_selection": false,
  "new5_used_for_selection": false
}
```

## 7a. Convergence Status

|   seed |   best_epoch | early_stop   | max_epoch_hit   | convergence_closed   |
|-------:|-------------:|:-------------|:----------------|:---------------------|
|      0 |          165 | True         | False           | True                 |
|      1 |          184 | False        | True            | False                |
|      2 |          157 | True         | False           | True                 |

Convergence is closed only if `early_stop=True` and `best_epoch<185` for every seed.

## 7b. vs PHC-lite v1 Comparison

| split   |   action_pnl |   phc_lite_v1_action_pnl |   delta_vs_v1_action_pnl |   delta_pct_vs_v1_action_pnl |   causal_fill_pnl |   phc_lite_v1_causal_fill_pnl |   delta_vs_v1_causal_fill_pnl |   delta_pct_vs_v1_causal_fill_pnl |   oracle_capture |   phc_lite_v1_capture_ratio |   delta_vs_v1_oracle_capture |   delta_pct_vs_v1_oracle_capture |
|:--------|-------------:|-------------------------:|-------------------------:|-----------------------------:|------------------:|------------------------------:|------------------------------:|----------------------------------:|-----------------:|----------------------------:|-----------------------------:|---------------------------------:|
| val     |  1.77152e+06 |              1.92717e+06 |                  -155645 |                   -0.0807635 |            157790 |                        170310 |                        -12520 |                        -0.073513  |         0.228521 |                    0.249534 |                   -0.0210134 |                       -0.0842106 |
| new3    |  4.72742e+06 |              5.02152e+06 |                  -294090 |                   -0.058566  |            458525 |                        488870 |                        -30345 |                        -0.0620717 |         0.250463 |                    0.266517 |                   -0.0160543 |                       -0.0602374 |
| new4    |  1.64268e+06 |              1.7625e+06  |                  -119815 |                   -0.0679803 |            114135 |                        121765 |                         -7630 |                        -0.0626617 |         0.222255 |                    0.239057 |                   -0.0168024 |                       -0.0702861 |

## 7c. Diagnostic Gates vs PHC-lite v1

| gate                       | pass   | rule                                         |
|:---------------------------|:-------|:---------------------------------------------|
| Convergence closed         | False  | all seeds early_stop=True and best_epoch<185 |
| Seed stability             | True   | all seeds x splits action PnL > 0            |
| Matched random             | True   | val/NEW3/NEW4 all beat random 95p            |
| OOS improvement            | False  | NEW3/NEW4 action PnL >= PHC-lite v1          |
| Causal-fill improvement    | False  | val/NEW3/NEW4 sequential PnL >= PHC-lite v1  |
| Oracle capture improvement | False  | val/NEW3/NEW4 capture ratio >= PHC-lite v1   |

Diagnostic gate status versus PHC-lite v1: `diagonal_ssm_convergence_still_incomplete`. This is not the final dense-baseline verdict.

## 7d. vs Frozen 05C4 Dense16

| split   |   action_pnl_diagonal16 |   causal_fill_pnl_diagonal16 |   oracle_capture_diagonal16 |   action_pnl_dense16 |   causal_fill_pnl_dense16 |   oracle_capture_dense16 |   action_pnl_delta_diagonal16_minus_dense16 |   action_pnl_delta_pct_diagonal16_minus_dense16 |   causal_fill_pnl_delta_diagonal16_minus_dense16 |   causal_fill_pnl_delta_pct_diagonal16_minus_dense16 |   oracle_capture_delta_diagonal16_minus_dense16 |   oracle_capture_delta_pct_diagonal16_minus_dense16 |
|:--------|------------------------:|-----------------------------:|----------------------------:|---------------------:|--------------------------:|-------------------------:|--------------------------------------------:|------------------------------------------------:|-------------------------------------------------:|-----------------------------------------------------:|------------------------------------------------:|----------------------------------------------------:|
| new3    |             4.72742e+06 |                       458525 |                    0.250463 |          5.23741e+06 |                    611095 |                 0.27863  |                                     -509985 |                                      -0.0973735 |                                          -152570 |                                           -0.249667  |                                      -0.0281667 |                                            -0.10109 |
| new4    |             1.64268e+06 |                       114135 |                    0.222255 |          1.87168e+06 |                    129160 |                 0.254392 |                                     -229005 |                                      -0.122352  |                                           -15025 |                                           -0.116329  |                                      -0.0321373 |                                            -0.12633 |
| val     |             1.77152e+06 |                       157790 |                    0.228521 |          2.03792e+06 |                    171835 |                 0.264287 |                                     -266400 |                                      -0.130721  |                                           -14045 |                                           -0.0817354 |                                      -0.0357659 |                                            -0.13533 |

The dense16 baseline values are read mechanically from `n2ew_era/05c4_phc_stp_cosine_lr/results/phc_stp_v2_vs_phc_lite_v1_summary.csv`.

## 7e. vs Frozen 05E Dense8

| split   |   action_pnl_diagonal16 |   causal_fill_pnl_diagonal16 |   oracle_capture_diagonal16 |   action_pnl_dense8 |   causal_fill_pnl_dense8 |   oracle_capture_dense8 |   action_pnl_delta_diagonal16_minus_dense8 |   action_pnl_delta_pct_diagonal16_minus_dense8 |   causal_fill_pnl_delta_diagonal16_minus_dense8 |   causal_fill_pnl_delta_pct_diagonal16_minus_dense8 |   oracle_capture_delta_diagonal16_minus_dense8 |   oracle_capture_delta_pct_diagonal16_minus_dense8 |
|:--------|------------------------:|-----------------------------:|----------------------------:|--------------------:|-------------------------:|------------------------:|-------------------------------------------:|-----------------------------------------------:|------------------------------------------------:|----------------------------------------------------:|-----------------------------------------------:|---------------------------------------------------:|
| new3    |             4.72742e+06 |                       458525 |                    0.250463 |         5.04471e+06 |                   489630 |                0.26795  |                                    -317285 |                                     -0.0628946 |                                          -31105 |                                          -0.0635276 |                                     -0.0174874 |                                         -0.0652636 |
| new4    |             1.64268e+06 |                       114135 |                    0.222255 |         1.76568e+06 |                   124540 |                0.239626 |                                    -123000 |                                     -0.0696615 |                                          -10405 |                                          -0.0835475 |                                     -0.0173714 |                                         -0.0724937 |
| val     |             1.77152e+06 |                       157790 |                    0.228521 |         1.92998e+06 |                   164205 |                0.250199 |                                    -158460 |                                     -0.0821043 |                                           -6415 |                                          -0.039067  |                                     -0.0216782 |                                         -0.0866439 |

The dense8 baseline values are read mechanically from `n2ew_era/05e_phc_hidden8_ablation/results/phc_stp_v2_vs_phc_lite_v1_summary.csv`.

## 7f. Seed Stability vs Dense16

| split   |   diagonal16_seed_count |   dense16_seed_count |   diagonal16_min_action_pnl |   dense16_min_action_pnl |   diagonal16_mean_action_pnl |   dense16_mean_action_pnl |   mean_action_pnl_delta_diagonal16_minus_dense16 | diagonal16_all_seed_action_positive   | dense16_all_seed_action_positive   |
|:--------|------------------------:|---------------------:|----------------------------:|-------------------------:|-----------------------------:|--------------------------:|-------------------------------------------------:|:--------------------------------------|:-----------------------------------|
| new3    |                       3 |                    3 |                 4.63284e+06 |              5.06118e+06 |                  4.67685e+06 |               5.14966e+06 |                                          -472808 | True                                  | True                               |
| new4    |                       3 |                    3 |                 1.59534e+06 |              1.7851e+06  |                  1.62334e+06 |               1.83389e+06 |                                          -210550 | True                                  | True                               |
| val     |                       3 |                    3 |                 1.67618e+06 |              1.95618e+06 |                  1.72826e+06 |               1.99489e+06 |                                          -266637 | True                                  | True                               |

### Per-Seed Details

|   seed | split   |   best_epoch_diagonal16 |   threshold_diagonal16 |   action_pnl_diagonal16 | action_pnl_positive_diagonal16   |   best_epoch_dense16 |   threshold_dense16 |   action_pnl_dense16 | action_pnl_positive_dense16   |   action_pnl_delta_diagonal16_minus_dense16 |   action_pnl_delta_pct_diagonal16_minus_dense16 |
|-------:|:--------|------------------------:|-----------------------:|------------------------:|:---------------------------------|---------------------:|--------------------:|---------------------:|:------------------------------|--------------------------------------------:|------------------------------------------------:|
|      0 | new3    |                     165 |                      0 |             4.67029e+06 | True                             |                  167 |                   0 |          5.1504e+06  | True                          |                                     -480110 |                                      -0.093218  |
|      0 | new4    |                     165 |                      0 |             1.632e+06   | True                             |                  167 |                   0 |          1.8449e+06  | True                          |                                     -212895 |                                      -0.115396  |
|      0 | val     |                     165 |                      0 |             1.73707e+06 | True                             |                  167 |                   0 |          1.99058e+06 | True                          |                                     -253505 |                                      -0.127353  |
|      1 | new3    |                     184 |                      0 |             4.63284e+06 | True                             |                  137 |                   0 |          5.06118e+06 | True                          |                                     -428330 |                                      -0.0846305 |
|      1 | new4    |                     184 |                      0 |             1.59534e+06 | True                             |                  137 |                   0 |          1.7851e+06  | True                          |                                     -189750 |                                      -0.106297  |
|      1 | val     |                     184 |                      0 |             1.67618e+06 | True                             |                  137 |                   0 |          1.95618e+06 | True                          |                                     -280005 |                                      -0.143139  |
|      2 | new3    |                     157 |                      0 |             4.72742e+06 | True                             |                  171 |                   0 |          5.23741e+06 | True                          |                                     -509985 |                                      -0.0973735 |
|      2 | new4    |                     157 |                      0 |             1.64268e+06 | True                             |                  171 |                   0 |          1.87168e+06 | True                          |                                     -229005 |                                      -0.122352  |
|      2 | val     |                     157 |                      0 |             1.77152e+06 | True                             |                  171 |                   0 |          2.03792e+06 | True                          |                                     -266400 |                                      -0.130721  |

## 7g. Seed Stability vs Dense8

| split   |   diagonal16_seed_count |   dense8_seed_count |   diagonal16_min_action_pnl |   dense8_min_action_pnl |   diagonal16_mean_action_pnl |   dense8_mean_action_pnl |   mean_action_pnl_delta_diagonal16_minus_dense8 | diagonal16_all_seed_action_positive   | dense8_all_seed_action_positive   |
|:--------|------------------------:|--------------------:|----------------------------:|------------------------:|-----------------------------:|-------------------------:|------------------------------------------------:|:--------------------------------------|:----------------------------------|
| new3    |                       3 |                   3 |                 4.63284e+06 |             5.03226e+06 |                  4.67685e+06 |              5.0485e+06  |                                         -371645 | True                                  | True                              |
| new4    |                       3 |                   3 |                 1.59534e+06 |             1.75561e+06 |                  1.62334e+06 |              1.7687e+06  |                                         -145352 | True                                  | True                              |
| val     |                       3 |                   3 |                 1.67618e+06 |             1.90628e+06 |                  1.72826e+06 |              1.91504e+06 |                                         -186780 | True                                  | True                              |

### Per-Seed Details

|   seed | split   |   best_epoch_diagonal16 |   threshold_diagonal16 |   action_pnl_diagonal16 | action_pnl_positive_diagonal16   |   best_epoch_dense8 |   threshold_dense8 |   action_pnl_dense8 | action_pnl_positive_dense8   |   action_pnl_delta_diagonal16_minus_dense8 |   action_pnl_delta_pct_diagonal16_minus_dense8 |
|-------:|:--------|------------------------:|-----------------------:|------------------------:|:---------------------------------|--------------------:|-------------------:|--------------------:|:-----------------------------|-------------------------------------------:|-----------------------------------------------:|
|      0 | new3    |                     165 |                      0 |             4.67029e+06 | True                             |                 171 |            0       |         5.04471e+06 | True                         |                                    -374420 |                                     -0.0742203 |
|      0 | new4    |                     165 |                      0 |             1.632e+06   | True                             |                 171 |            0       |         1.76568e+06 | True                         |                                    -133675 |                                     -0.0757074 |
|      0 | val     |                     165 |                      0 |             1.73707e+06 | True                             |                 171 |            0       |         1.92998e+06 | True                         |                                    -192915 |                                     -0.0999567 |
|      1 | new3    |                     184 |                      0 |             4.63284e+06 | True                             |                 196 |            1.24337 |         5.03226e+06 | True                         |                                    -399410 |                                     -0.07937   |
|      1 | new4    |                     184 |                      0 |             1.59534e+06 | True                             |                 196 |            1.24337 |         1.75561e+06 | True                         |                                    -160265 |                                     -0.0912874 |
|      1 | val     |                     184 |                      0 |             1.67618e+06 | True                             |                 196 |            1.24337 |         1.90884e+06 | True                         |                                    -232670 |                                     -0.12189   |
|      2 | new3    |                     157 |                      0 |             4.72742e+06 | True                             |                 179 |            0       |         5.06853e+06 | True                         |                                    -341105 |                                     -0.0672986 |
|      2 | new4    |                     157 |                      0 |             1.64268e+06 | True                             |                 179 |            0       |         1.7848e+06  | True                         |                                    -142115 |                                     -0.0796254 |
|      2 | val     |                     157 |                      0 |             1.77152e+06 | True                             |                 179 |            0       |         1.90628e+06 | True                         |                                    -134755 |                                     -0.07069   |

## 7h. Matched Random Summary

| split   | policy_id        |   model_total_excess_pnl_H20 |       random_mean |   random_std |        random_95p |        random_99p |   empirical_p_value_fraction_random_ge_model | empirical_p_value_wording   | beats_random_95p   |   n_trials | matching                                                    |   seed_selected_on_val |   threshold_selected_on_val | phase   | optimizer_restarted   |
|:--------|:-----------------|-----------------------------:|------------------:|-------------:|------------------:|------------------:|---------------------------------------------:|:----------------------------|:-------------------|-----------:|:------------------------------------------------------------|-----------------------:|----------------------------:|:--------|:----------------------|
| val     | phc_diagonal_ssm |                  1.77152e+06 | -924651           |      29713.7 | -872337           | -856534           |                                            0 | < 0.001                     | True               |       1000 | count_session_side_spread_bucket_with_session_side_fallback |                      2 |                           0 | 05f     | False                 |
| new3    | phc_diagonal_ssm |                  4.72742e+06 |      -2.03952e+06 |      46693.3 |      -1.96396e+06 |      -1.93384e+06 |                                            0 | < 0.001                     | True               |       1000 | count_session_side_spread_bucket_with_session_side_fallback |                      2 |                           0 | 05f     | False                 |
| new4    | phc_diagonal_ssm |                  1.64268e+06 | -778073           |      29247.7 | -730825           | -709618           |                                            0 | < 0.001                     | True               |       1000 | count_session_side_spread_bucket_with_session_side_fallback |                      2 |                           0 | 05f     | False                 |

## 7i. Explicit Verdict

`diagonal16 regresses vs dense16 baseline and regresses vs dense8 baseline on val/NEW3/NEW4`.

- vs dense16: At least one val/NEW3/NEW4 action PnL, causal-fill PnL, or oracle capture metric regresses by more than 2% versus dense16.
- vs dense8: At least one val/NEW3/NEW4 action PnL, causal-fill PnL, or oracle capture metric regresses by more than 2% versus dense8.

## 7j. Forbidden Claims

- Not PHCLiteSTP dense recurrence
- Not full PHCSSM
- Not fresh NEW5 holdout
- Not true exchange replay
- Not FPGA PnL improvement proof
- Not real-world profitability proof
- Does not estimate or claim new HLS latency
- Does not retrain or replace the deployed 06F HLS weights
- phase = "05f"
- optimizer_restarted = False
- lr_scheduler = "CosineAnnealingLR"
