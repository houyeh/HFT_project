# Regime Feature Report

## Causality / Leakage Check
- All regime features are computed with backward-looking rolling or EWMA operators only.
- No full-session future return, future trend, or cumulative future information is used.
- Event-level masks for downstream backtests are based on current/past event information only.

## Dataset Coverage
| dataset | symbol | sessions | rows | split_summary |
|---|---:|---:|---:|---|
| NEW | TXFB6 | 9 | 5322624 | test:1166333, train:3592802, val:563489 |
| NEW3 | TXFA6 | 6 | 5657398 | oos_full:5657398 |

## Field Availability
| dataset | field | available | missing_ratio |
|---|---|---|---:|
| NEW | timestamp | yes | 0.0000 |
| NEW | imbalance | yes | 0.0000 |
| NEW | rolling_imbalance_mean_100 | yes | 0.0000 |
| NEW | microprice_minus_mid_ticks | yes | 0.0000 |
| NEW3 | timestamp | yes | 0.0000 |
| NEW3 | imbalance | yes | 0.0000 |
| NEW3 | rolling_imbalance_mean_100 | yes | 0.0000 |
| NEW3 | microprice_minus_mid_ticks | yes | 0.0000 |

## Validation Caveat
- NEW validation session count = 1 (20260126Night).
- If this count is 1, validation-driven policy selection has high variance and should be treated cautiously.
