# Break-Even Frontier

Grid definition: `neutral_fill_haircut in {1.0, 0.75, 0.5, 0.25}`, `toxic_multiplier in {1.0, 1.25, 1.5, 2.0}`, and `cancel_delay_ticks=0`.

The frontier is descriptive. It is not a new model-selection pass and it does not tune thresholds on test data.

## Summary

### 015_common
- toxic=1.00: min positive haircut `0.75`; last positive `haircut=0.75, toxic=1, delay=0`; first negative `haircut=0.5, toxic=1, delay=0`.
- toxic=1.25: min positive haircut `0.75`; last positive `haircut=0.75, toxic=1.25, delay=0`; first negative `haircut=0.5, toxic=1.25, delay=0`.
- toxic=1.50: min positive haircut `1.00`; last positive `haircut=1, toxic=1.5, delay=0`; first negative `haircut=0.75, toxic=1.5, delay=0`.
- toxic=2.00: min positive haircut `NONE on this grid`; last positive `NONE`; first negative `haircut=1, toxic=2, delay=0`.

### 016_v2_binary
- toxic=1.00: min positive haircut `0.75`; last positive `haircut=0.75, toxic=1, delay=0`; first negative `haircut=0.5, toxic=1, delay=0`.
- toxic=1.25: min positive haircut `0.75`; last positive `haircut=0.75, toxic=1.25, delay=0`; first negative `haircut=0.5, toxic=1.25, delay=0`.
- toxic=1.50: min positive haircut `0.75`; last positive `haircut=0.75, toxic=1.5, delay=0`; first negative `haircut=0.5, toxic=1.5, delay=0`.
- toxic=2.00: min positive haircut `0.75`; last positive `haircut=0.75, toxic=2, delay=0`; first negative `haircut=0.5, toxic=2, delay=0`.

### no_cancel
- toxic=1.00: min positive haircut `0.75`; last positive `haircut=0.75, toxic=1, delay=0`; first negative `haircut=0.5, toxic=1, delay=0`.
- toxic=1.25: min positive haircut `1.00`; last positive `haircut=1, toxic=1.25, delay=0`; first negative `haircut=0.75, toxic=1.25, delay=0`.
- toxic=1.50: min positive haircut `NONE on this grid`; last positive `NONE`; first negative `haircut=1, toxic=1.5, delay=0`.
- toxic=2.00: min positive haircut `NONE on this grid`; last positive `NONE`; first negative `haircut=1, toxic=2, delay=0`.

### random_v3_mean
- toxic=1.00: min positive haircut `0.75`; last positive `haircut=0.75, toxic=1, delay=0`; first negative `haircut=0.5, toxic=1, delay=0`.
- toxic=1.25: min positive haircut `NONE on this grid`; last positive `NONE`; first negative `haircut=1, toxic=1.25, delay=0`.
- toxic=1.50: min positive haircut `NONE on this grid`; last positive `NONE`; first negative `haircut=1, toxic=1.5, delay=0`.
- toxic=2.00: min positive haircut `NONE on this grid`; last positive `NONE`; first negative `haircut=1, toxic=2, delay=0`.
