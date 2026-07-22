# PHC Hidden8 Ablation Report

Final verdict: hidden8 `regresses` vs hidden16 baseline on val/NEW3/NEW4.

Verdict status: `hidden8_regresses_vs_hidden16_baseline`.

At least one val/NEW3/NEW4 action PnL, causal-fill PnL, or oracle capture metric regresses by more than 2% versus hidden16.

## Protocol

- Phase: `05e`.
- Model: `PHCLiteSTP` imported from Phase 05C2 source code; architecture unchanged.
- Training: from-scratch random initialization for each seed. No 05C2/05C3/05C4/05D weights or optimizer state loaded.
- Controlled single-variable change versus 05C4: `HIDDEN_DIM=8` instead of `HIDDEN_DIM=16`.
- Unchanged from 05C4: seeds, max epochs, patience, batch size, LR, LR_MIN, CosineAnnealingLR, weight decay, feature list, target, and causal-fill salt.
- Optimizer and scheduler states are saved in both best and last checkpoints.
- Selection split: validation only.
- NEW3/NEW4: diagnostic OOS only.
- NEW5: not used.
- No files under `n2ew_era/06f_phc_lite_manual_hls/` are modified or evaluated in this phase.
- Material-match criterion in this report: no key metric regresses by more than 2% versus hidden16.

## Parameter Count Comparison

| model                         |   hidden_dim |   parameter_count |   feature_dim |   mtl_steps |   delta_vs_hidden16 |   ratio_vs_hidden16 |
|:------------------------------|-------------:|------------------:|--------------:|------------:|--------------------:|--------------------:|
| phc_stp_v2_hidden8            |            8 |               547 |            16 |           1 |               -1184 |            0.316002 |
| phc_stp_v2_cosine_lr_hidden16 |           16 |              1731 |            16 |           1 |                   0 |            1        |

## Selected Policy

```json
{
  "model": "phc_stp_v2_hidden8",
  "variant": "PHC-lite + STP-like dynamic synaptic gain clean 200epoch cosine LR",
  "phase": "05e",
  "selection_split": "val",
  "seed": 0,
  "threshold": 0.0,
  "target": "excess_pnl_H20",
  "seq_len": 128,
  "stride": 32,
  "hidden_dim": 8,
  "mtl_steps": 1,
  "max_epochs": 200,
  "patience": 20,
  "lr_initial": 0.001,
  "lr_min": 1e-05,
  "lr_scheduler": "CosineAnnealingLR",
  "optimizer_restarted": false,
  "validation_metrics": {
    "split": "val",
    "policy_id": "phc_stp_v2_hidden8_seed0",
    "horizon": 20,
    "state_count": 82960,
    "selected_action_count": 64384,
    "selection_rate": 0.7760848601735776,
    "bid_count": 32404,
    "ask_count": 31980,
    "no_quote_count": 18576,
    "total_excess_pnl_H20": 1929985.0,
    "avg_selected_excess": 29.976158672962228,
    "win_rate": 0.47014786282306165,
    "negative_rate": 0.529805541749503,
    "max_drawdown": 16045.0,
    "positive_session_count": 6,
    "session_count": 6,
    "seed": 0,
    "threshold": 0.0,
    "best_epoch": 171,
    "selection_split": "val",
    "phase": "05e",
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
|      0 |          171 | True         | False           | True                 |
|      1 |          196 | False        | True            | False                |
|      2 |          179 | True         | False           | True                 |

Convergence is closed only if `early_stop=True` and `best_epoch<185` for every seed.

## 7b. vs PHC-lite v1 Comparison

| split   |   action_pnl |   phc_lite_v1_action_pnl |   delta_vs_v1_action_pnl |   delta_pct_vs_v1_action_pnl |   causal_fill_pnl |   phc_lite_v1_causal_fill_pnl |   delta_vs_v1_causal_fill_pnl |   delta_pct_vs_v1_causal_fill_pnl |   oracle_capture |   phc_lite_v1_capture_ratio |   delta_vs_v1_oracle_capture |   delta_pct_vs_v1_oracle_capture |
|:--------|-------------:|-------------------------:|-------------------------:|-----------------------------:|------------------:|------------------------------:|------------------------------:|----------------------------------:|-----------------:|----------------------------:|-----------------------------:|---------------------------------:|
| val     |  1.92998e+06 |              1.92717e+06 |                     2815 |                   0.00146069 |            164205 |                        170310 |                         -6105 |                       -0.0358464  |         0.250199 |                    0.249534 |                  0.000664806 |                       0.00266419 |
| new3    |  5.04471e+06 |              5.02152e+06 |                    23195 |                   0.00461912 |            489630 |                        488870 |                           760 |                        0.00155461 |         0.26795  |                    0.266517 |                  0.00143309  |                       0.00537711 |
| new4    |  1.76568e+06 |              1.7625e+06  |                     3185 |                   0.0018071  |            124540 |                        121765 |                          2775 |                        0.0227898  |         0.239626 |                    0.239057 |                  0.000568995 |                       0.00238016 |

## 7c. Diagnostic Gates vs PHC-lite v1

| gate                       | pass   | rule                                         |
|:---------------------------|:-------|:---------------------------------------------|
| Convergence closed         | False  | all seeds early_stop=True and best_epoch<185 |
| Seed stability             | True   | all seeds x splits action PnL > 0            |
| Matched random             | True   | val/NEW3/NEW4 all beat random 95p            |
| OOS improvement            | True   | NEW3/NEW4 action PnL >= PHC-lite v1          |
| Causal-fill improvement    | False  | val/NEW3/NEW4 sequential PnL >= PHC-lite v1  |
| Oracle capture improvement | True   | val/NEW3/NEW4 capture ratio >= PHC-lite v1   |

Diagnostic gate status versus PHC-lite v1: `stp_v2_convergence_still_incomplete`. This is not the final hidden8-vs-hidden16 verdict.

## 7d. vs Frozen 05C4 Hidden16

| split   |   action_pnl_hidden8 |   causal_fill_pnl_hidden8 |   oracle_capture_hidden8 |   action_pnl_hidden16 |   causal_fill_pnl_hidden16 |   oracle_capture_hidden16 |   action_pnl_delta_hidden8_minus_hidden16 |   action_pnl_delta_pct_hidden8_minus_hidden16 |   causal_fill_pnl_delta_hidden8_minus_hidden16 |   causal_fill_pnl_delta_pct_hidden8_minus_hidden16 |   oracle_capture_delta_hidden8_minus_hidden16 |   oracle_capture_delta_pct_hidden8_minus_hidden16 |
|:--------|---------------------:|--------------------------:|-------------------------:|----------------------:|---------------------------:|--------------------------:|------------------------------------------:|----------------------------------------------:|-----------------------------------------------:|---------------------------------------------------:|----------------------------------------------:|--------------------------------------------------:|
| new3    |          5.04471e+06 |                    489630 |                 0.26795  |           5.23741e+06 |                     611095 |                  0.27863  |                                   -192700 |                                    -0.036793  |                                        -121465 |                                         -0.198766  |                                    -0.0106793 |                                        -0.0383278 |
| new4    |          1.76568e+06 |                    124540 |                 0.239626 |           1.87168e+06 |                     129160 |                  0.254392 |                                   -106005 |                                    -0.0566361 |                                          -4620 |                                         -0.0357696 |                                    -0.0147659 |                                        -0.0580438 |
| val     |          1.92998e+06 |                    164205 |                 0.250199 |           2.03792e+06 |                     171835 |                  0.264287 |                                   -107940 |                                    -0.0529656 |                                          -7630 |                                         -0.0444031 |                                    -0.0140877 |                                        -0.0533045 |

The hidden16 baseline values are read mechanically from `n2ew_era/05c4_phc_stp_cosine_lr/results/phc_stp_v2_vs_phc_lite_v1_summary.csv`.

## 7e. Seed Stability vs Hidden16

| split   |   hidden8_seed_count |   hidden16_seed_count |   hidden8_min_action_pnl |   hidden16_min_action_pnl |   hidden8_mean_action_pnl |   hidden16_mean_action_pnl |   mean_action_pnl_delta_hidden8_minus_hidden16 | hidden8_all_seed_action_positive   | hidden16_all_seed_action_positive   |
|:--------|---------------------:|----------------------:|-------------------------:|--------------------------:|--------------------------:|---------------------------:|-----------------------------------------------:|:-----------------------------------|:------------------------------------|
| new3    |                    3 |                     3 |              5.03226e+06 |               5.06118e+06 |               5.0485e+06  |                5.14966e+06 |                                      -101163   | True                               | True                                |
| new4    |                    3 |                     3 |              1.75561e+06 |               1.7851e+06  |               1.7687e+06  |                1.83389e+06 |                                       -65198.3 | True                               | True                                |
| val     |                    3 |                     3 |              1.90628e+06 |               1.95618e+06 |               1.91504e+06 |                1.99489e+06 |                                       -79856.7 | True                               | True                                |

### Per-Seed Details

|   seed | split   |   best_epoch_hidden8 |   threshold_hidden8 |   action_pnl_hidden8 | action_pnl_positive_hidden8   |   best_epoch_hidden16 |   threshold_hidden16 |   action_pnl_hidden16 | action_pnl_positive_hidden16   |   action_pnl_delta_hidden8_minus_hidden16 |   action_pnl_delta_pct_hidden8_minus_hidden16 |
|-------:|:--------|---------------------:|--------------------:|---------------------:|:------------------------------|----------------------:|---------------------:|----------------------:|:-------------------------------|------------------------------------------:|----------------------------------------------:|
|      0 | new3    |                  171 |             0       |          5.04471e+06 | True                          |                   167 |                    0 |           5.1504e+06  | True                           |                                   -105690 |                                   -0.0205207  |
|      0 | new4    |                  171 |             0       |          1.76568e+06 | True                          |                   167 |                    0 |           1.8449e+06  | True                           |                                    -79220 |                                   -0.04294    |
|      0 | val     |                  171 |             0       |          1.92998e+06 | True                          |                   167 |                    0 |           1.99058e+06 | True                           |                                    -60590 |                                   -0.0304384  |
|      1 | new3    |                  196 |             1.24337 |          5.03226e+06 | True                          |                   137 |                    0 |           5.06118e+06 | True                           |                                    -28920 |                                   -0.00571409 |
|      1 | new4    |                  196 |             1.24337 |          1.75561e+06 | True                          |                   137 |                    0 |           1.7851e+06  | True                           |                                    -29485 |                                   -0.0165173  |
|      1 | val     |                  196 |             1.24337 |          1.90884e+06 | True                          |                   137 |                    0 |           1.95618e+06 | True                           |                                    -47335 |                                   -0.0241977  |
|      2 | new3    |                  179 |             0       |          5.06853e+06 | True                          |                   171 |                    0 |           5.23741e+06 | True                           |                                   -168880 |                                   -0.0322449  |
|      2 | new4    |                  179 |             0       |          1.7848e+06  | True                          |                   171 |                    0 |           1.87168e+06 | True                           |                                    -86890 |                                   -0.0464234  |
|      2 | val     |                  179 |             0       |          1.90628e+06 | True                          |                   171 |                    0 |           2.03792e+06 | True                           |                                   -131645 |                                   -0.0645976  |

## 7f. Matched Random Summary

| split   | policy_id          |   model_total_excess_pnl_H20 |      random_mean |   random_std |        random_95p |        random_99p |   empirical_p_value_fraction_random_ge_model | empirical_p_value_wording   | beats_random_95p   |   n_trials | matching                                                    |   seed_selected_on_val |   threshold_selected_on_val | phase   | optimizer_restarted   |
|:--------|:-------------------|-----------------------------:|-----------------:|-------------:|------------------:|------------------:|---------------------------------------------:|:----------------------------|:-------------------|-----------:|:------------------------------------------------------------|-----------------------:|----------------------------:|:--------|:----------------------|
| val     | phc_stp_v2_hidden8 |                  1.92998e+06 | -968649          |      29046.4 | -919296           | -898724           |                                            0 | < 0.001                     | True               |       1000 | count_session_side_spread_bucket_with_session_side_fallback |                      0 |                           0 | 05e     | False                 |
| new3    | phc_stp_v2_hidden8 |                  5.04471e+06 |      -2.0883e+06 |      44130.8 |      -2.01631e+06 |      -1.98046e+06 |                                            0 | < 0.001                     | True               |       1000 | count_session_side_spread_bucket_with_session_side_fallback |                      0 |                           0 | 05e     | False                 |
| new4    | phc_stp_v2_hidden8 |                  1.76568e+06 | -799529          |      27499.6 | -756015           | -735419           |                                            0 | < 0.001                     | True               |       1000 | count_session_side_spread_bucket_with_session_side_fallback |                      0 |                           0 | 05e     | False                 |

## 7g. Explicit Verdict

`hidden8 regresses vs hidden16 baseline on val/NEW3/NEW4`.

At least one val/NEW3/NEW4 action PnL, causal-fill PnL, or oracle capture metric regresses by more than 2% versus hidden16.

## 7h. Forbidden Claims

- Not full biological STP
- Not fresh NEW5 holdout
- Not true exchange replay
- Not FPGA PnL improvement proof
- Not real-world profitability proof
- Does not estimate or claim new HLS latency
- Does not retrain or replace the deployed 06F HLS weights
- phase = "05e"
- optimizer_restarted = False
- lr_scheduler = "CosineAnnealingLR"
