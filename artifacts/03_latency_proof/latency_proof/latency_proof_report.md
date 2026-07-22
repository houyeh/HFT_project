# Latency Proof Report

## Validation Winner

```text
symbol  score_threshold  one_sided_threshold  quote_off_threshold  toxicity_threshold  one_sided_toxicity_threshold  toxicity_freeze_threshold  toxicity_freeze_steps  inventory_widen_ticks  inventory_tighten_ticks  stationary_cutoff alpha_profile  inventory_penalty_ticks  max_inventory end_time  quote_confirmation_steps  alpha_smoothing_window  stop_loss_ticks  take_profit_ticks  alpha_decay_threshold     regime_profile  fill_rate  adverse_selection_loss_ticks  realized_trades  fills  quote_updates  toxicity_freeze_events  net_pnl_ticks  avg_pnl_ticks  win_rate  avg_holding_steps  sharpe_like
 TXFJ5            0.006                0.008                0.006                0.06                          0.04                       0.08                      5                    0.5                      1.0              0.336   flow_cancel                      0.5              1 12:00:00                         1                       1              4.0                0.0                  0.002               none    0.00000                           0.0                0      0            586                       0            0.0       0.000000   0.00000           0.000000     0.000000
 MXFJ5            0.006                0.008                0.006                0.06                          0.04                       0.08                      5                    0.5                      1.0              0.336    toxic_flow                      0.5              1 12:00:00                         0                       2              4.0                0.0                  0.002 cancel_guard_q0.80    0.39823                          44.0               29     45            113                       0           24.0       0.827586   0.62069           2.517241     0.902932
```

## Test Latency Sweep

```text
 latency_steps  fill_rate  adverse_selection_loss_ticks  realized_trades  fills  quote_updates  toxicity_freeze_events  net_pnl_ticks  avg_pnl_ticks  win_rate  avg_holding_steps  sharpe_like
             0   0.025641                           9.0                2      2             78                       0           -9.0           -4.5       0.0                1.5         -9.0
             1   0.055556                           1.0                1      1             18                       0           -1.0           -1.0       0.0               10.0          0.0
             2   0.000000                           0.0                0      0              0                       0            0.0            0.0       0.0                0.0          0.0
             4   0.000000                           0.0                0      0              0                       0            0.0            0.0       0.0                0.0          0.0
```

## Notes

- Fill rate is computed as `fills / quote_updates`.
- Adverse selection loss is defined here as the magnitude of realized negative pnl ticks on closed trades.
- Monotonic decreasing pnl by latency: `False`

## Frozen Model

- Frozen model path: `artifacts\latency_proof\frozen_tcn_best_model.pt`
- MACs per sample: `460848.0`
- FP32 weight bytes: `19084`
- INT8 weight bytes: `4771`
- Approximate FP32 input bytes per inference: `4800`
- Approximate INT8 streamed weight bytes per inference: `4771`
