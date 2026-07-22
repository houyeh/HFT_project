# Baseline Evidence — Verified Numbers

## Test Dataset
- 20260127Day:   LOB rows=410,696 (raw), samples=82,119
- 20260127Night: LOB rows=755,845 (raw), samples=151,149
- Source: `data/NEW_ALL/20260127Day/` + `data/NEW_ALL/20260127Night/`

## No-Model Baseline
Source: `baseline_summary_fixed.csv`
- net_pnl = **56,100 TWD**, trades=660, max_drawdown=76,232

## Model vs Matched Random (50 seeds)
Source: `baseline_summary_fixed.csv`

| Setup | Model net_pnl | Random mean | Random std | Random mean+1std | Beats mean+1std? |
|-------|--------------|-------------|------------|-----------------|-----------------|
| thr=0.55 | **84,980** | 42,362 | 38,111 | 80,473 | **YES** |
| thr=0.60 | **83,460** | 46,293 | 32,756 | 79,049 | **YES** |
| thr=0.70 (val-selected) | 63,440 | 53,209 | 19,569 | 72,779 | **NO** |

## Threshold Sweep Detail
Source: `threshold_sweep_enhanced.csv`

| thr | signal_rate | trades | net_pnl | toxic_avoided | missed_pnl | net_cancel_value |
|-----|------------|--------|---------|---------------|-----------|-----------------|
| 0.50 | 19.1% | 429 | 75,765 | 269 | 4,900 | 33,100 |
| **0.55** | **14.7%** | **468** | **84,980** | **227** | **3,900** | **28,900** |
| **0.60** | **10.9%** | **516** | **83,460** | **184** | **2,900** | **23,600** |
| 0.65 | 7.7% | 558 | 78,430 | 130 | 1,600 | 17,800 |
| **0.70** | **5.1%** | **584** | **63,440** | **98** | **1,000** | **14,800** |

## Validation-Selected Threshold
Source: `validation_threshold_selection.csv`
- Val dataset: **20260126Night only** (112,698 samples)
- Selected: thr=**0.70** (best composite score on val)
- Val net_pnl @0.70 = 60,740 | val signal_rate = 3.7%
- Test result @0.70 = **63,440 TWD**

## Opportunity Cost (threshold=0.55)
Source: `missed_opportunity_diagnostics_fixed.csv`
- Cancel count: 342,385
- Missed profitable fills: 34 fills = 3,900 TWD lost
- Toxic fills avoided: 227 fills = 32,800 TWD saved
- Net cancel value: **+28,900 TWD**
- Random baseline net cancel value @0.55: 9,550 TWD (mean over 50 seeds)

## Sources
- `lt=.../eval_outputs/diagnosis/baseline_summary_fixed.csv`
- `lt=.../eval_outputs/diagnosis/threshold_sweep_enhanced.csv`
- `lt=.../eval_outputs/diagnosis/validation_threshold_selection.csv`
- `lt=.../eval_outputs/diagnosis/missed_opportunity_diagnostics_fixed.csv`
