# Opportunity Cost — Verified Numbers

來源檔案: `lt=.../eval_outputs/diagnosis/missed_opportunity_diagnostics_fixed.csv`

## Model Cancel Filter (signal = cancel bid or ask)

| thr | signal_rate | cancel_count | missed_profitable_fills | missed_pnl | toxic_avoided | toxic_pnl_avoided | net_cancel_value |
|-----|------------|-------------|------------------------|-----------|---------------|------------------|-----------------|
| 0.55 | 14.68% | 342,385 | 34 | 3,900 | 227 | 32,800 | **+28,900** |
| 0.60 | 10.89% | 253,917 | 25 | 2,900 | 184 | 26,500 | **+23,600** |
| 0.70 | 5.09% | 118,787 | 8 | 1,000 | 98 | 15,800 | **+14,800** |

## Random Cancel Baseline (matched signal_rate, 50 seeds mean)

| thr | cancel_count | missed_profitable | missed_pnl | toxic_avoided | toxic_pnl_avoided | net_cancel_value |
|-----|-------------|-------------------|-----------|---------------|------------------|-----------------|
| 0.55 | 342,384 | 31.2 | 3,558 | 106.4 | 13,108 | **9,550** |
| 0.60 | 253,897 | 22.8 | 2,610 | 79.2 | 9,728 | **7,118** |

## Interpretation
- At thr=0.55: model net_cancel_value (28,900) is **3.0× random** (9,550)
  → TCN selects cancels more intelligently than random at same rate
- Net cancel value positive across all thresholds → avoided toxic > missed profitable
- avg_net_cancel_value per cancel:
  - Model 0.55: 0.084 TWD/cancel
  - Model 0.70: 0.125 TWD/cancel (fewer but more precise cancels)
