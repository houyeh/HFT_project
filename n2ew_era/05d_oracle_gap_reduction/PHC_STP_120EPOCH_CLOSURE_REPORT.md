# PHC STP 120-Epoch Closure Report

Final status: `stp_not_adopted_convergence_incomplete`.

Protocol: max_epoch=120, patience=15, seeds 0/1/2, validation-only seed/threshold selection. NEW3/NEW4 are diagnostic OOS only. NEW5 is not used.

Implementation note: this closure extends the audited Phase 05C.2 80-epoch STP-like checkpoints to epochs 81-120. Because 05C.2 did not store optimizer state, AdamW is restarted from the 80-epoch weights; this is an epoch-extension sanity closure, not a fresh-from-scratch 120-epoch training claim.

## Selected Policy

```json
{
  "model": "phc_stp_120epoch",
  "selection_split": "val",
  "seed": 2,
  "threshold": 0.0,
  "target": "excess_pnl_H20",
  "seq_len": 128,
  "stride": 32,
  "max_epoch": 120,
  "patience": 15,
  "validation_metrics": {
    "split": "val",
    "policy_id": "phc_stp_120epoch_seed2",
    "horizon": 20,
    "state_count": 82960,
    "selected_action_count": 63146,
    "selection_rate": 0.761162005785921,
    "bid_count": 31445,
    "ask_count": 31701,
    "no_quote_count": 19814,
    "total_excess_pnl_H20": 2061555.0,
    "avg_selected_excess": 32.64743610046559,
    "win_rate": 0.4766414341367624,
    "negative_rate": 0.5233110569157191,
    "max_drawdown": 16215.0,
    "positive_session_count": 6,
    "session_count": 6,
    "seed": 2,
    "threshold": 0.0,
    "best_epoch": 119,
    "selection_split": "val",
    "new3_used_for_selection": false,
    "new4_used_for_selection": false,
    "new5_used_for_selection": false
  },
  "new3_used_for_selection": false,
  "new4_used_for_selection": false,
  "new5_used_for_selection": false
}
```

## Seed Convergence Summary

|   seed |   max_epoch_seen |   best_epoch |   best_val_pnl |   final_val_pnl |
|-------:|-----------------:|-------------:|---------------:|----------------:|
|      0 |              120 |          113 |    1.998e+06   |     1.98556e+06 |
|      1 |              120 |          119 |    1.96586e+06 |     1.9517e+06  |
|      2 |              120 |          119 |    2.06156e+06 |     2.01036e+06 |

## STP 120 vs PHC-lite v1

| split   |   total_excess_pnl_H20 |   selected_action_count |   max_drawdown |   v1_action_pnl |   sequential_excess_pnl |   causal_max_drawdown |   v1_causal_pnl |   capture_ratio |   v1_capture_ratio |   action_delta_vs_v1 |   causal_delta_vs_v1 |   oracle_capture_delta_vs_v1 |
|:--------|-----------------------:|------------------------:|---------------:|----------------:|------------------------:|----------------------:|----------------:|----------------:|-------------------:|---------------------:|---------------------:|-----------------------------:|
| val     |            2.06156e+06 |                   63146 |          16215 |     1.92717e+06 |                  181945 |                  2605 |          170310 |        0.266614 |           0.249534 |               134385 |                11635 |                    0.0170794 |
| new3    |            5.25102e+06 |                  138747 |           4605 |     5.02152e+06 |                  606750 |                  5170 |          488870 |        0.278807 |           0.266517 |               229510 |               117880 |                    0.0122891 |
| new4    |            1.87112e+06 |                   53247 |           5665 |     1.7625e+06  |                  129555 |                  2370 |          121765 |        0.253831 |           0.239057 |               108630 |                 7790 |                    0.0147744 |

Adoption requires action-level, causal-fill, oracle capture, and seed stability gates. This is STP-like dynamic gain only, not full biological STP.
