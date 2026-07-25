# AUDIT: Latency PnL Sensitivity

Status: `resolution_limited_for_sub_ms_latency_references`.

## Delay Injection Mechanism

A frozen policy observes the original PHC-window terminal state at event `t` and emits the same decision it emitted in the no-latency evaluation. No model, feature, threshold, seed, or checkpoint is retrained or reselected.

For an injected deterministic latency `L`, the quote is considered eligible for fill testing from the first dense TXF L2 event in the same session whose `timestamp_us >= timestamp_us(t) + L`. The book-state features used by the conservative L2 stress fill-probability model (`top_depth`, `bid_sz_1`, `ask_sz_1`) are taken from that effective dense event. If the lookup runs past the end of the session, the decision is treated as no-fill for latency reasons.

The H20 excess outcome and inventory scheduling remain the existing `simulate_scheduled_by_event_index()` proxy on the PHC-window decision-state universe. In other words, latency shifts the fill-probability book state using the dense L2 stream, but it does not create a new stale-order exchange replay. This preserves the prior causal-fill simulator semantics and should be read as a latency sensitivity proxy, not true exchange matching.

## Timestamp Resolution Statement

The verified minimum non-zero timestamp step in this sequence-simulator scope is `1000us` (`1ms`). Phase 06P independently verified the underlying NEW1-NEW4 TXF event stream also has a 1ms minimum non-zero timestamp difference. Therefore `23.16us` and `83.16us` are below data resolution and are shown only as visual references; this dataset cannot honestly measure separate PnL effects at those two sub-millisecond latencies.

The scanned latency grid is: 0, 1ms, 5ms, 10ms, 50ms, 100ms, 500ms.

NEW5 is not loaded or touched. Splits are limited to val, NEW3, and NEW4.

## Relation To The 2026-06-26 Null Result

The prior val-only check found CPU empirical vs FPGA fixed latency changed selected-row fill probabilities for only `0.09111724%` of decisions under the then-current proxy. This audit extends that null result into a deterministic latency curve. If the curve stays flat through the measurable range, the correct conclusion is a boundary result: under this timestamp resolution and market-density representation, sub-millisecond CPU/FPGA latency differences are not economically distinguishable by the proxy.

## Baseline Consistency Check

The `L=0` rows were compared against the frozen causal-fill CSVs from Phase 05A and 05C4. Exact agreement means the new script did not change the no-latency simulator path.

| split | policy_id | sequential_excess_pnl | fill_count | existing_l0_sequential_excess_pnl | existing_l0_fill_count | delta_existing_l0_pnl | delta_existing_l0_fill_count | pnl_matches_existing | fill_count_matches_existing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | side_by_queue_imbalance_abs_0p1 | 103260.0 | 6416 | 103260.0 | 6416 | 0.0 | 0 | True | True |
| val | ml_hgb_regressor_shallow_convergence_selected | 142435.0 | 6231 | 142435.0 | 6231 | 0.0 | 0 | True | True |
| val | phc_stp_v2_cosine_lr | 171835.0 | 6891 | 171835.0 | 6891 | 0.0 | 0 | True | True |
| new3 | side_by_queue_imbalance_abs_0p1 | 303700.0 | 13821 | 303700.0 | 13821 | 0.0 | 0 | True | True |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 411090.0 | 13955 | 411090.0 | 13955 | 0.0 | 0 | True | True |
| new3 | phc_stp_v2_cosine_lr | 611095.0 | 17907 | 611095.0 | 17907 | 0.0 | 0 | True | True |
| new4 | side_by_queue_imbalance_abs_0p1 | 72415.0 | 3820 | 72415.0 | 3820 | 0.0 | 0 | True | True |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 97680.0 | 3629 | 97680.0 | 3629 | 0.0 | 0 | True | True |
| new4 | phc_stp_v2_cosine_lr | 129160.0 | 4237 | 129160.0 | 4237 | 0.0 | 0 | True | True |

## PnL Sensitivity Table

| split | policy_id | latency_label | sequential_excess_pnl | delta_pnl_vs_0us | delta_pct_pnl_vs_0us | fill_count | delta_fill_count_vs_0us | avg_trade_excess_pnl | nonzero_book_shift_rate_selected | latency_no_fill_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | side_by_queue_imbalance_abs_0p1 | 0 | 103260.0 | 0.0 | 0.000% | 6416 | 0 | 16.09413965087282 | 0.000% | 0.000% |
| val | side_by_queue_imbalance_abs_0p1 | 1ms | 142865.0 | 39605.0 | 38.355% | 7209 | 793 | 19.81758912470523 | 100.000% | 0.001% |
| val | side_by_queue_imbalance_abs_0p1 | 5ms | 134865.0 | 31605.0 | 30.607% | 7449 | 1033 | 18.10511478050745 | 100.000% | 0.003% |
| val | side_by_queue_imbalance_abs_0p1 | 10ms | 137320.0 | 34060.0 | 32.985% | 7492 | 1076 | 18.328884143085958 | 100.000% | 0.003% |
| val | side_by_queue_imbalance_abs_0p1 | 50ms | 137205.0 | 33945.0 | 32.873% | 7553 | 1137 | 18.16562955117172 | 100.000% | 0.004% |
| val | side_by_queue_imbalance_abs_0p1 | 100ms | 136085.0 | 32825.0 | 31.789% | 7681 | 1265 | 17.717094128368704 | 100.000% | 0.004% |
| val | side_by_queue_imbalance_abs_0p1 | 500ms | 143170.0 | 39910.0 | 38.650% | 8122 | 1706 | 17.627431667077076 | 100.000% | 0.007% |
| val | ml_hgb_regressor_shallow_convergence_selected | 0 | 142435.0 | 0.0 | 0.000% | 6231 | 0 | 22.859091638581287 | 0.000% | 0.000% |
| val | ml_hgb_regressor_shallow_convergence_selected | 1ms | 178375.0 | 35940.0 | 25.233% | 6695 | 464 | 26.64301717699776 | 100.000% | 0.002% |
| val | ml_hgb_regressor_shallow_convergence_selected | 5ms | 169070.0 | 26635.0 | 18.700% | 6822 | 591 | 24.78305482263266 | 100.000% | 0.003% |
| val | ml_hgb_regressor_shallow_convergence_selected | 10ms | 170725.0 | 28290.0 | 19.862% | 6845 | 614 | 24.941563184806427 | 100.000% | 0.003% |
| val | ml_hgb_regressor_shallow_convergence_selected | 50ms | 175035.0 | 32600.0 | 22.888% | 6891 | 660 | 25.40052242054854 | 100.000% | 0.005% |
| val | ml_hgb_regressor_shallow_convergence_selected | 100ms | 175820.0 | 33385.0 | 23.439% | 6972 | 741 | 25.218014916810098 | 100.000% | 0.005% |
| val | ml_hgb_regressor_shallow_convergence_selected | 500ms | 190725.0 | 48290.0 | 33.903% | 7345 | 1114 | 25.966643975493533 | 100.000% | 0.006% |
| val | phc_stp_v2_cosine_lr | 0 | 171835.0 | 0.0 | 0.000% | 6891 | 0 | 24.936148599622697 | 0.000% | 0.000% |
| val | phc_stp_v2_cosine_lr | 1ms | 210110.0 | 38275.0 | 22.274% | 7166 | 275 | 29.320401897850964 | 100.000% | 0.002% |
| val | phc_stp_v2_cosine_lr | 5ms | 199945.0 | 28110.0 | 16.359% | 7237 | 346 | 27.628160840127123 | 100.000% | 0.003% |
| val | phc_stp_v2_cosine_lr | 10ms | 203375.0 | 31540.0 | 18.355% | 7275 | 384 | 27.9553264604811 | 100.000% | 0.003% |
| val | phc_stp_v2_cosine_lr | 50ms | 209530.0 | 37695.0 | 21.937% | 7338 | 447 | 28.554101935132188 | 100.000% | 0.005% |
| val | phc_stp_v2_cosine_lr | 100ms | 211695.0 | 39860.0 | 23.197% | 7427 | 536 | 28.50343341860778 | 100.000% | 0.005% |
| val | phc_stp_v2_cosine_lr | 500ms | 221565.0 | 49730.0 | 28.941% | 7709 | 818 | 28.741081852380336 | 100.000% | 0.005% |
| new3 | side_by_queue_imbalance_abs_0p1 | 0 | 303700.0 | 0.0 | 0.000% | 13821 | 0 | 21.97380797337385 | 0.000% | 0.000% |
| new3 | side_by_queue_imbalance_abs_0p1 | 1ms | 346880.0 | 43180.0 | 14.218% | 15429 | 1608 | 22.482338453561475 | 100.000% | 0.000% |
| new3 | side_by_queue_imbalance_abs_0p1 | 5ms | 318935.0 | 15235.0 | 5.016% | 15912 | 2091 | 20.043677727501258 | 100.000% | 0.000% |
| new3 | side_by_queue_imbalance_abs_0p1 | 10ms | 322055.0 | 18355.0 | 6.044% | 16024 | 2203 | 20.098290064902645 | 100.000% | 0.000% |
| new3 | side_by_queue_imbalance_abs_0p1 | 50ms | 340140.0 | 36440.0 | 11.999% | 16545 | 2724 | 20.558476881233002 | 100.000% | 0.001% |
| new3 | side_by_queue_imbalance_abs_0p1 | 100ms | 330880.0 | 27180.0 | 8.950% | 16829 | 3008 | 19.661298948244102 | 100.000% | 0.001% |
| new3 | side_by_queue_imbalance_abs_0p1 | 500ms | 421405.0 | 117705.0 | 38.757% | 20613 | 6792 | 20.443652064231312 | 100.000% | 0.003% |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 0 | 411090.0 | 0.0 | 0.000% | 13955 | 0 | 29.458258688642065 | 0.000% | 0.000% |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 1ms | 455765.0 | 44675.0 | 10.867% | 14950 | 995 | 30.485953177257525 | 100.000% | 0.000% |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 5ms | 437355.0 | 26265.0 | 6.389% | 15144 | 1189 | 28.879754358161648 | 100.000% | 0.000% |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 10ms | 439235.0 | 28145.0 | 6.846% | 15252 | 1297 | 28.798518227117754 | 100.000% | 0.000% |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 50ms | 471430.0 | 60340.0 | 14.678% | 15739 | 1784 | 29.952983035771016 | 100.000% | 0.000% |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 100ms | 461630.0 | 50540.0 | 12.294% | 15999 | 2044 | 28.85367835489718 | 100.000% | 0.001% |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 500ms | 579090.0 | 168000.0 | 40.867% | 19334 | 5379 | 29.951898210406537 | 100.000% | 0.002% |
| new3 | phc_stp_v2_cosine_lr | 0 | 611095.0 | 0.0 | 0.000% | 17907 | 0 | 34.126040096051824 | 0.000% | 0.000% |
| new3 | phc_stp_v2_cosine_lr | 1ms | 681700.0 | 70605.0 | 11.554% | 18560 | 653 | 36.72952586206897 | 100.000% | 0.000% |
| new3 | phc_stp_v2_cosine_lr | 5ms | 667970.0 | 56875.0 | 9.307% | 18682 | 775 | 35.75473718017343 | 100.000% | 0.000% |
| new3 | phc_stp_v2_cosine_lr | 10ms | 663635.0 | 52540.0 | 8.598% | 18751 | 844 | 35.391979094448295 | 100.000% | 0.000% |
| new3 | phc_stp_v2_cosine_lr | 50ms | 695955.0 | 84860.0 | 13.887% | 19323 | 1416 | 36.01692283806862 | 100.000% | 0.001% |
| new3 | phc_stp_v2_cosine_lr | 100ms | 695230.0 | 84135.0 | 13.768% | 19618 | 1711 | 35.43837292282598 | 100.000% | 0.001% |
| new3 | phc_stp_v2_cosine_lr | 500ms | 741500.0 | 130405.0 | 21.340% | 20540 | 2633 | 36.10029211295034 | 100.000% | 0.002% |
| new4 | side_by_queue_imbalance_abs_0p1 | 0 | 72415.0 | 0.0 | 0.000% | 3820 | 0 | 18.956806282722514 | 0.000% | 0.000% |
| new4 | side_by_queue_imbalance_abs_0p1 | 1ms | 75020.0 | 2605.0 | 3.597% | 4313 | 493 | 17.393925341989334 | 100.000% | 0.002% |
| new4 | side_by_queue_imbalance_abs_0p1 | 5ms | 72325.0 | -90.0 | -0.124% | 4466 | 646 | 16.194581280788178 | 100.000% | 0.004% |
| new4 | side_by_queue_imbalance_abs_0p1 | 10ms | 69390.0 | -3025.0 | -4.177% | 4495 | 675 | 15.437152391546162 | 100.000% | 0.004% |
| new4 | side_by_queue_imbalance_abs_0p1 | 50ms | 63770.0 | -8645.0 | -11.938% | 4663 | 843 | 13.675745228393739 | 100.000% | 0.005% |
| new4 | side_by_queue_imbalance_abs_0p1 | 100ms | 72565.0 | 150.0 | 0.207% | 4790 | 970 | 15.149269311064717 | 100.000% | 0.005% |
| new4 | side_by_queue_imbalance_abs_0p1 | 500ms | 134770.0 | 62355.0 | 86.108% | 7863 | 4043 | 17.139768536182117 | 100.000% | 0.032% |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 0 | 97680.0 | 0.0 | 0.000% | 3629 | 0 | 26.916505924497105 | 0.000% | 0.000% |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 1ms | 106450.0 | 8770.0 | 8.978% | 3931 | 302 | 27.079623505469346 | 100.000% | 0.002% |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 5ms | 106470.0 | 8790.0 | 8.999% | 4043 | 414 | 26.334405144694532 | 100.000% | 0.004% |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 10ms | 103950.0 | 6270.0 | 6.419% | 4051 | 422 | 25.660330782522834 | 100.000% | 0.004% |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 50ms | 102200.0 | 4520.0 | 4.627% | 4161 | 532 | 24.56140350877193 | 100.000% | 0.004% |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 100ms | 108620.0 | 10940.0 | 11.200% | 4253 | 624 | 25.53961909240536 | 100.000% | 0.004% |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 500ms | 195465.0 | 97785.0 | 100.107% | 7330 | 3701 | 26.66643929058663 | 100.000% | 0.031% |
| new4 | phc_stp_v2_cosine_lr | 0 | 129160.0 | 0.0 | 0.000% | 4237 | 0 | 30.483832900637243 | 0.000% | 0.000% |
| new4 | phc_stp_v2_cosine_lr | 1ms | 146455.0 | 17295.0 | 13.390% | 4404 | 167 | 33.25499545867393 | 100.000% | 0.004% |
| new4 | phc_stp_v2_cosine_lr | 5ms | 147295.0 | 18135.0 | 14.041% | 4488 | 251 | 32.81974153297683 | 100.000% | 0.006% |
| new4 | phc_stp_v2_cosine_lr | 10ms | 148010.0 | 18850.0 | 14.594% | 4487 | 250 | 32.98640517049253 | 100.000% | 0.006% |
| new4 | phc_stp_v2_cosine_lr | 50ms | 150900.0 | 21740.0 | 16.832% | 4621 | 384 | 32.655269422202984 | 100.000% | 0.007% |
| new4 | phc_stp_v2_cosine_lr | 100ms | 154225.0 | 25065.0 | 19.406% | 4686 | 449 | 32.91186513017499 | 100.000% | 0.007% |
| new4 | phc_stp_v2_cosine_lr | 500ms | 285150.0 | 155990.0 | 120.773% | 7791 | 3554 | 36.59992298806315 | 100.000% | 0.035% |

## Breakpoint Rule

For this audit, a material degradation breakpoint is the first scanned `L > 0` where causal-fill PnL is at least `1.0%` below the same strategy/split's `L=0` PnL. This is a diagnostic threshold for readability, not a tuned trading rule.

| split | policy_id | baseline_pnl | material_drop_threshold_pnl | breakpoint_status | first_material_degradation_latency_us |
| --- | --- | --- | --- | --- | --- |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 411090.0 | 4110.9 | no_material_degradation_in_scanned_range |  |
| new3 | phc_stp_v2_cosine_lr | 611095.0 | 6110.95 | no_material_degradation_in_scanned_range |  |
| new3 | side_by_queue_imbalance_abs_0p1 | 303700.0 | 3037.0 | no_material_degradation_in_scanned_range |  |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 97680.0 | 976.8000000000001 | no_material_degradation_in_scanned_range |  |
| new4 | phc_stp_v2_cosine_lr | 129160.0 | 1291.6000000000001 | no_material_degradation_in_scanned_range |  |
| new4 | side_by_queue_imbalance_abs_0p1 | 72415.0 | 724.15 | material_degradation_found | 10000.0 |
| val | ml_hgb_regressor_shallow_convergence_selected | 142435.0 | 1424.3500000000001 | no_material_degradation_in_scanned_range |  |
| val | phc_stp_v2_cosine_lr | 171835.0 | 1718.3500000000001 | no_material_degradation_in_scanned_range |  |
| val | side_by_queue_imbalance_abs_0p1 | 103260.0 | 1032.6 | no_material_degradation_in_scanned_range |  |

## Conclusion

A material degradation point appears in the scanned range: first observed at `10ms` for `side_by_queue_imbalance_abs_0p1` on `new4` under the 1% diagnostic rule.
However, the project still cannot claim a measured economic distinction between `23.16us` and `83.16us`, because both are below the verified 1ms timestamp resolution.

Safe claim: this is a latency-to-fill proxy on PHC-window sequence states. It does not prove real-world profitability, does not validate exchange queue replay, and does not imply FPGA latency changes PnL unless the measured curve supports that at the available timestamp resolution.

## Outputs

- Calculation script: `n2ew_era/06q_latency_pnl_sensitivity/00_latency_pnl_sensitivity.py`
- Main CSV: `results/latency_pnl_sensitivity.csv`
- Shift CSV: `results/latency_shift_summary.csv`
- Breakpoint CSV: `results/latency_pnl_breakpoints.csv`
- Resolution CSV: `results/timestamp_resolution_sensitivity_scope.csv`
- Baseline check CSV: `results/baseline_l0_consistency_check.csv`
- Figure: `figures/latency_pnl_curve.png`
