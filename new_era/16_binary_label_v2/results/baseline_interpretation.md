# Baseline Interpretation

`016_v2_binary` does not replace `015_smoothed`. They are competing baseline candidates under different operating assumptions.

- `015_smoothed` is a legacy smoothed mid-move TCN strategy with skew/gating behavior and a historically audited universe.
- `016_v2_binary` is a side-specific quote-risk proxy cancel model with validation-frozen thresholds and stronger matched-random audit evidence.
- The Pre-018 comparison treats both as candidates for queue-aware stress testing, not as a winner-takes-all replacement.

The purpose of `018_queue_aware` is to stress both candidates under common execution assumptions and determine which remains robust when fills, adverse selection, and cancel latency are made less favorable.
