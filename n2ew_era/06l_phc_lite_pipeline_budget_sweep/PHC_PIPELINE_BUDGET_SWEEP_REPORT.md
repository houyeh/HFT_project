# PHC-lite Pipeline Budget Sweep Report

This report is generated from user-run Vitis HLS artifacts in this independent 06L phase. It does not modify the frozen 06F production baseline.

## Endpoint References

| endpoint | latency_cycles | latency_us | II | DSP | DSP_pct | LUT | LUT_pct | FF | FF_pct | BRAM_18K | BRAM_pct | fits_budget |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| v7_auto_pipeline_over_budget | 604 | 6.04 | 605 | 268 | 121.0 | 41433 | 77.0 | 55719 | 52.0 | 0 | 0.0 | False |
| v8_pipeline_off_production_baseline | 8316 | 83.16 | 8317 | 46 | 20.9 | 24836 | 46.7 | 14518 | 13.6 | 6 | 2.1 | True |

## Sweep Table

| factor | csim | max_abs_diff | DSP | LUT | FF | BRAM | latency_cycles | latency_us | II | fits_budget | speedup_vs_v8 | cosim |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|---|
| 1 | PASS | 0.00725275 | 46 | 24836 | 14518 | 6 | 8316 | 83.16 | 8317 | True | 1 | PASS |
| 2 | PASS | 0.00725275 | 56 | 25601 | 16162 | 12 | 5036 | 50.36 | 5037 | True | 1.65131 | PASS |
| 4 | PASS | 0.00725275 | 57 | 27456 | 17000 | 0 | 3724 | 37.24 | 3725 | True | 2.23308 | PASS |
| 8 | PASS | 0.00725275 | 70 | 30392 | 19704 | 0 | 3068 | 30.68 | 3069 | True | 2.71056 | PASS |
| 16 | PASS | 0.00725275 | 121 | 36792 | 35774 | 0 | 2316 | 23.16 | 2317 | True | 3.59067 | PASS |

## Boundary Decision

Best budget-fitting factor so far: `16` with latency `2316` cycles (23.16 us), speedup `3.591x` versus v8.

Cosim is only expected for factors that fit all four resource budgets. A missing cosim entry is not a failure by itself unless the factor was budget-fitting and intended for RTL confirmation.

## Claim Boundary

- This is a synthesis-level resource/latency sweep only.
- Do not claim speedup until a factor has passing csim and all four resource budgets fit.
- Do not modify 06F based on this table without a separate freeze decision.
- This does not involve retraining, NEW5, P&R, exchange replay, or PnL claims.
