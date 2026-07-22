# Quote-Eligible Distribution Report

The filter did not remove the hard/toxic event population; retention is nearly complete and the post-filter set still contains fill-proxy and toxic examples in every split.

## Waterfall

| split | raw_rows | after_valid_bid_ask | after_positive_spread | after_complete_depth | after_session_boundary | after_feature_finite | quote_eligible_rows | dropped_invalid_bid_ask | dropped_nonpositive_spread | dropped_missing_depth | dropped_session_boundary | dropped_feature_na | retention_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 1166541 | 1166541 | 1166541 | 1166540 | 1166321 | 1166321 | 1166321 | 0 | 0 | 1 | 219 | 0 | 0.999811 |
| train | 3593426 | 3593426 | 3593426 | 3593424 | 3592764 | 3592764 | 3592764 | 0 | 0 | 2 | 660 | 0 | 0.999816 |
| val | 563593 | 563593 | 563593 | 563587 | 563477 | 563477 | 563477 | 0 | 0 | 6 | 110 | 0 | 0.999794 |

## Toxic and Fill Proxy Distribution After Filtering

| split | side | eligible_rows | fill_proxy_count | fill_proxy_rate | toxic_count | toxic_rate | toxic_rate_given_fill | median_spread_ticks | p95_spread_ticks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | bid | 1166321 | 11783 | 0.010103 | 7748 | 0.006643 | 0.657557 | 2.000000 | 3.000000 |
| test | ask | 1166321 | 12862 | 0.011028 | 8246 | 0.007070 | 0.641113 | 2.000000 | 3.000000 |
| train | bid | 3592764 | 40711 | 0.011331 | 28947 | 0.008057 | 0.711036 | 2.000000 | 4.000000 |
| train | ask | 3592764 | 41321 | 0.011501 | 28796 | 0.008015 | 0.696885 | 2.000000 | 4.000000 |
| val | bid | 563477 | 5491 | 0.009745 | 3873 | 0.006873 | 0.705336 | 2.000000 | 3.000000 |
| val | ask | 563477 | 6215 | 0.011030 | 4200 | 0.007454 | 0.675784 | 2.000000 | 3.000000 |

Interpretation: the eligibility filter is primarily a market-validity and boundary filter. It removes invalid/depth/boundary rows while preserving natural train/val/test distributions for evaluation.