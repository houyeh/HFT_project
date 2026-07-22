# 015 Original vs 015 Common Universe

The 015 audit is intentionally reported as two tracks.

- `015_original_universe`: preserves the historical 015 audit result and its original event universe.
- `015_common_universe`: re-scores the frozen 015 model on the 016_v2 quote-eligible test universe for apples-to-apples 018 queue stress.

The PnL moves from `170,625` on `015_original_universe` to `90,845` on `015_common_universe`.
This drop is due to the evaluation universe changing for the apples-to-apples 018 comparison, not necessarily a degradation in the 015 model itself.

Common-universe coverage:

- original_event_count: `233266`
- common_universe_event_count: `1166321`
- overlap_count: `1166123`
- coverage_rate: `0.999830`
- missing_prediction_rate: `0.000170`
- dropped_event_count: `198`
