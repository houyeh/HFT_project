# PHC STP v2 Cosine LR Final Decision Report

Final status: `stp_v2_adopted_as_best_sequence_model`.

## Protocol

- Phase: `05c4`.
- Model: `PHCLiteSTP` imported from Phase 05C2 source code; architecture unchanged.
- Training: from-scratch random initialization for each seed. No 05C2/05C3/05D weights or optimizer state loaded.
- Only change versus 05C3: `CosineAnnealingLR(T_max=200, eta_min=1e-5)` after each epoch.
- Optimizer and scheduler states are saved in both best and last checkpoints.
- Selection split: validation only.
- NEW3/NEW4: diagnostic OOS only.
- NEW5: not used.
- Causal-fill random salt is retained from 05C3 for apples-to-apples causal-fill comparison.

## Selected Policy

```json
{
  "model": "phc_stp_v2_cosine_lr",
  "variant": "PHC-lite + STP-like dynamic synaptic gain clean 200epoch cosine LR",
  "phase": "05c4",
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
    "policy_id": "phc_stp_v2_cosine_lr_seed2",
    "horizon": 20,
    "state_count": 82960,
    "selected_action_count": 64568,
    "selection_rate": 0.7783027965284475,
    "bid_count": 32522,
    "ask_count": 32046,
    "no_quote_count": 18392,
    "total_excess_pnl_H20": 2037925.0,
    "avg_selected_excess": 31.56246128112997,
    "win_rate": 0.47365568083261056,
    "negative_rate": 0.5262978565233553,
    "max_drawdown": 16200.0,
    "positive_session_count": 6,
    "session_count": 6,
    "seed": 2,
    "threshold": 0.0,
    "best_epoch": 171,
    "selection_split": "val",
    "phase": "05c4",
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
|      0 |          167 | True         | False           | True                 |
|      1 |          137 | True         | False           | True                 |
|      2 |          171 | True         | False           | True                 |

Convergence is closed only if `early_stop=True` and `best_epoch<185` for every seed.

## 7b. vs PHC-lite v1 Comparison

| split   |   action_pnl |   phc_lite_v1_action_pnl |   delta_vs_v1_action_pnl |   delta_pct_vs_v1_action_pnl |   causal_fill_pnl |   phc_lite_v1_causal_fill_pnl |   delta_vs_v1_causal_fill_pnl |   delta_pct_vs_v1_causal_fill_pnl |   oracle_capture |   phc_lite_v1_capture_ratio |   delta_vs_v1_oracle_capture |   delta_pct_vs_v1_oracle_capture |
|:--------|-------------:|-------------------------:|-------------------------:|-----------------------------:|------------------:|------------------------------:|------------------------------:|----------------------------------:|-----------------:|----------------------------:|-----------------------------:|---------------------------------:|
| val     |  2.03792e+06 |              1.92717e+06 |                   110755 |                    0.0574703 |            171835 |                        170310 |                          1525 |                        0.00895426 |         0.264287 |                    0.249534 |                    0.0147525 |                        0.05912   |
| new3    |  5.23741e+06 |              5.02152e+06 |                   215895 |                    0.042994  |            611095 |                        488870 |                        122225 |                        0.250015   |         0.27863  |                    0.266517 |                    0.0121124 |                        0.0454468 |
| new4    |  1.87168e+06 |              1.7625e+06  |                   109190 |                    0.0619519 |            129160 |                        121765 |                          7395 |                        0.0607317  |         0.254392 |                    0.239057 |                    0.0153349 |                        0.0641473 |

## 7c. Adoption Gates

| gate                       | pass   | rule                                         |
|:---------------------------|:-------|:---------------------------------------------|
| Convergence closed         | True   | all seeds early_stop=True and best_epoch<185 |
| Seed stability             | True   | all seeds x splits action PnL > 0            |
| Matched random             | True   | val/NEW3/NEW4 all beat random 95p            |
| OOS improvement            | True   | NEW3/NEW4 action PnL >= PHC-lite v1          |
| Causal-fill improvement    | True   | val/NEW3/NEW4 sequential PnL >= PHC-lite v1  |
| Oracle capture improvement | True   | val/NEW3/NEW4 capture ratio >= PHC-lite v1   |

## 7d. vs 05C3 Comparison

| split   |   action_pnl_05c3 |   causal_fill_pnl_05c3 |   oracle_capture_05c3 |   action_pnl_05c4 |   causal_fill_pnl_05c4 |   oracle_capture_05c4 |   action_delta_05c4_minus_05c3 |   action_delta_pct_05c4_minus_05c3 |   causal_fill_delta_05c4_minus_05c3 |   causal_fill_delta_pct_05c4_minus_05c3 |   oracle_capture_delta_05c4_minus_05c3 |   oracle_capture_delta_pct_05c4_minus_05c3 |
|:--------|------------------:|-----------------------:|----------------------:|------------------:|-----------------------:|----------------------:|-------------------------------:|-----------------------------------:|------------------------------------:|----------------------------------------:|---------------------------------------:|-------------------------------------------:|
| new3    |       5.33485e+06 |                 527500 |              0.283069 |       5.23741e+06 |                 611095 |              0.27863  |                         -97440 |                         -0.0182648 |                               83595 |                               0.158474  |                            -0.0044395  |                                 -0.0156834 |
| new4    |       1.90854e+06 |                 134580 |              0.258614 |       1.87168e+06 |                 129160 |              0.254392 |                         -36850 |                         -0.019308  |                               -5420 |                              -0.0402734 |                            -0.00422238 |                                 -0.0163269 |
| val     |       2.08874e+06 |                 191525 |              0.269646 |       2.03792e+06 |                 171835 |              0.264287 |                         -50820 |                         -0.0243304 |                              -19690 |                              -0.102806  |                            -0.00535895 |                                 -0.0198741 |

Cosine LR closes the 05C3 convergence failure. It does not improve or match 05C3 on action-level PnL or oracle capture, and it only improves NEW3 causal-fill among the three causal-fill splits. The adoption decision is therefore based on passing all formal gates versus the frozen PHC-lite v1 reference while 05C3 remained convergence-incomplete.

## 7e. Forbidden Claims

- Not full biological STP
- Not fresh NEW5 holdout
- Not true exchange replay
- Not FPGA PnL improvement proof
- Not real-world profitability proof
- phase = "05c4"
- optimizer_restarted = False
- lr_scheduler = "CosineAnnealingLR"
