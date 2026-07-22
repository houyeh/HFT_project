# Implemented Regime Gate — Status

## Implemented (in regime_gate_sweep.csv)
All gate types applied to NEW and NEW3 retrospectively.

| Gate Type | Parameters Tested | Datasets |
|-----------|------------------|----------|
| spread_filter | spread_ticks_min ∈ {1.0,1.5,2.0} | NEW, NEW3 |
| trend_filter | trend_ticks ∈ {5,10,20,30} | NEW, NEW3 |
| volatility_filter | vol_threshold = 70th/80th pctile of rolling_vol | NEW, NEW3 |
| combined_filter | trend+vol+spread combinations | NEW, NEW3 |

## Not Yet Implemented
- EWMA-based gate (alpha sweep)
- Kalman velocity gate
- Causal auto-gate selector (requires 017_regime_auto_gate experiment)

## Key Limitation
Current gate selection is ORACLE/RETROSPECTIVE:
- spread_filter chosen for NEW because it scored best on NEW test split
- combined_filter chosen for NEW3 because it scored best on NEW3 full
- This is NOT production auto-gating

## Source
- `lt=.../eval_outputs/diagnosis/regime_gate_sweep.csv`
- Implementation: `new_era/15_models_playaround/run_diagnosis_v3_phase1.py`
