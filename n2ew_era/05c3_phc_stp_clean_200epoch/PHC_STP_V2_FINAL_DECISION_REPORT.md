# PHC STP v2 Clean 200-Epoch Final Decision Report

Final status: `stp_v2_convergence_still_incomplete`.

## Protocol

- Phase: `05c3`.
- Model: `PHCLiteSTP` imported from Phase 05C2 source code; architecture unchanged.
- Training: from-scratch random initialization for each seed. No 05C2/05D weights or optimizer state loaded.
- Optimizer state is saved in both best and last checkpoints.
- Selection split: validation only.
- NEW3/NEW4: diagnostic OOS only.
- NEW5: not used.

## Selected Policy

```json
{
  "model": "phc_stp_v2_clean_200epoch",
  "variant": "PHC-lite + STP-like dynamic synaptic gain clean 200epoch",
  "phase": "05c3",
  "selection_split": "val",
  "seed": 2,
  "threshold": 3.301987946033478,
  "target": "excess_pnl_H20",
  "seq_len": 128,
  "stride": 32,
  "hidden_dim": 16,
  "mtl_steps": 1,
  "max_epochs": 200,
  "patience": 20,
  "optimizer_restarted": false,
  "validation_metrics": {
    "split": "val",
    "policy_id": "phc_stp_v2_clean_200epoch_seed2",
    "horizon": 20,
    "state_count": 82960,
    "selected_action_count": 62220,
    "selection_rate": 0.75,
    "bid_count": 32325,
    "ask_count": 29895,
    "no_quote_count": 20740,
    "total_excess_pnl_H20": 2088745.0,
    "avg_selected_excess": 33.5703150112504,
    "win_rate": 0.47942783670845385,
    "negative_rate": 0.5205239472838316,
    "max_drawdown": 16030.0,
    "positive_session_count": 6,
    "session_count": 6,
    "seed": 2,
    "threshold": 3.301987946033478,
    "best_epoch": 190,
    "selection_split": "val",
    "phase": "05c3",
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
|      0 |          196 | False        | True            | False                |
|      1 |          200 | False        | True            | False                |
|      2 |          190 | False        | True            | False                |

Convergence is closed only if `early_stop=True` and `best_epoch<185` for every seed.

## 7b. vs PHC-lite v1 Comparison

| split   |   action_pnl |   phc_lite_v1_action_pnl |   delta_vs_v1_action_pnl |   delta_pct_vs_v1_action_pnl |   causal_fill_pnl |   phc_lite_v1_causal_fill_pnl |   delta_vs_v1_causal_fill_pnl |   delta_pct_vs_v1_causal_fill_pnl |   oracle_capture |   phc_lite_v1_capture_ratio |   delta_vs_v1_oracle_capture |   delta_pct_vs_v1_oracle_capture |
|:--------|-------------:|-------------------------:|-------------------------:|-----------------------------:|------------------:|------------------------------:|------------------------------:|----------------------------------:|-----------------:|----------------------------:|-----------------------------:|---------------------------------:|
| val     |  2.08874e+06 |              1.92717e+06 |                   161575 |                    0.0838406 |            191525 |                        170310 |                         21215 |                          0.124567 |         0.269646 |                    0.249534 |                    0.0201114 |                        0.0805959 |
| new3    |  5.33485e+06 |              5.02152e+06 |                   313335 |                    0.0623985 |            527500 |                        488870 |                         38630 |                          0.079019 |         0.283069 |                    0.266517 |                    0.0165519 |                        0.0621042 |
| new4    |  1.90854e+06 |              1.7625e+06  |                   146040 |                    0.0828598 |            134580 |                        121765 |                         12815 |                          0.105244 |         0.258614 |                    0.239057 |                    0.0195572 |                        0.0818099 |

## 7c. Adoption Gates

| gate                       | pass   | rule                                         |
|:---------------------------|:-------|:---------------------------------------------|
| Convergence closed         | False  | all seeds early_stop=True and best_epoch<185 |
| Seed stability             | True   | all seeds x splits action PnL > 0            |
| Matched random             | True   | val/NEW3/NEW4 all beat random 95p            |
| OOS improvement            | True   | NEW3/NEW4 action PnL >= PHC-lite v1          |
| Causal-fill improvement    | True   | val/NEW3/NEW4 sequential PnL >= PHC-lite v1  |
| Oracle capture improvement | True   | val/NEW3/NEW4 capture ratio >= PHC-lite v1   |

## 7d. Forbidden Claims

- Not full biological STP
- Not fresh NEW5 holdout
- Not true exchange replay
- Not FPGA PnL improvement proof
- Not real-world profitability proof
- phase = "05c3"
- optimizer_restarted = False
