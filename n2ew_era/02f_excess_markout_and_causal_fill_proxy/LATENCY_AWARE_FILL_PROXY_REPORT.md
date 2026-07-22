# Latency-Aware Fill Proxy Report

Superseded note: this smoke report was generated before `latency_aware_fill_probabilities()` was switched from the legacy event-rate approximation to timestamp-based `searchsorted` lookup. Use `LATENCY_SHIFT_DIAGNOSTIC_VAL.md` and rerun `07_latency_aware_fill_proxy_analysis.py` before using PnL/matched-random results.

This is a proxy model for latency effects on fill quality. It approximates latency-driven adverse selection by shifting the order-book state used for fill probability forward by a latency-dependent number of events. It is not validated exchange fill data, not true queue replay, and not proof of real-world profitability. If true tick-level order acknowledgements, queue priority, and exchange matching data become available, they should replace this proxy.

Decision features are unchanged: policies still decide from the original event. Only the book state used for fill probability is shifted after the decision latency.

CPU raw drift allowed: `True`.
CPU raw latency source: `C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06f_phc_lite_manual_hls\results\cpu_baseline_timing_libtorch_raw.csv`.
CPU raw summary: mean=63.9025us, median=45.4000us, p99=386.4000us, min=37.3000us, max=1799.4000us.
FPGA latency scenario: fixed 83.16us.

## CPU Summary Drift Check

| metric | original | raw_rerun | relative_delta |
|---|---:|---:|---:|
| mean_us | 74.5444 | 63.9025 | -14.28% |
| median_us | 51 | 45.4 | -10.98% |
| p99_us | 636.7 | 386.4 | -39.31% |
| min_us | 43.8 | 37.3 | -14.84% |

## Matched Random Status

This report was generated with `1` matched-random trial(s). It is a pipeline smoke test, not the final n=1000 matched-random significance run requested for presentation use.

Run `python 07_latency_aware_fill_proxy_analysis.py --allow-raw-drift --trials 1000` to produce the full matched-random output after accepting the recaptured CPU latency distribution caveat.

## Policy Results

|   sequential_excess_pnl |   selected_action_count |   fill_count |   no_fill_count |   latency_no_fill_count |   latency_no_fill_rate |   forced_no_quote_count |   max_abs_inventory |   max_drawdown |   win_rate |   loss_rate |   fill_rate_after_latency |   mean_events_to_skip_selected |   p99_events_to_skip_selected_max_session |   max_events_to_skip_selected | split   | policy_id                       | scenario           |   horizon | fill_model             |   inventory_cap |   fee_multiplier |   matched_random_mean |   matched_random_95p |   matched_random_max | beats_matched_random_95p   |   empirical_p_value |   random_trials |
|------------------------:|------------------------:|-------------:|----------------:|------------------------:|-----------------------:|------------------------:|--------------------:|---------------:|-----------:|------------:|--------------------------:|-------------------------------:|------------------------------------------:|------------------------------:|:--------|:--------------------------------|:-------------------|----------:|:-----------------------|----------------:|-----------------:|----------------------:|---------------------:|---------------------:|:---------------------------|--------------------:|----------------:|
|        378585           |                  468627 |        25861 |          442766 |                       0 |                      0 |                   16087 |                   1 |          15755 |   0.422257 |    0.577743 |                 0.0551846 |                    2.13389e-06 |                                         0 |                             1 | val     | side_by_queue_imbalance_abs_0p1 | cpu_empirical      |        20 | conservative_l2_stress |               1 |                1 |     -845270           |    -845270           |    -845270           | True                       |                   0 |               1 |
|        378585           |                  468627 |        25861 |          442766 |                       0 |                      0 |                   16087 |                   1 |          15755 |   0.422257 |    0.577743 |                 0.0551846 |                    0           |                                         0 |                             0 | val     | side_by_queue_imbalance_abs_0p1 | fpga_fixed_83p16us |        20 | conservative_l2_stress |               1 |                1 |     -795360           |    -795360           |    -795360           | True                       |                   0 |               1 |
|             5.49544e+06 |                 4631249 |       287641 |         4343608 |                       0 |                      0 |                  219954 |                   1 |          12625 |   0.439694 |    0.560292 |                 0.0621087 |                    2.15924e-06 |                                         0 |                             1 | new3    | side_by_queue_imbalance_abs_0p1 | cpu_empirical      |        20 | conservative_l2_stress |               1 |                1 |          -9.50408e+06 |         -9.50408e+06 |         -9.50408e+06 | True                       |                   0 |               1 |
|             5.49544e+06 |                 4631249 |       287641 |         4343608 |                       0 |                      0 |                  219954 |                   1 |          12625 |   0.439694 |    0.560292 |                 0.0621087 |                    0           |                                         0 |                             0 | new3    | side_by_queue_imbalance_abs_0p1 | fpga_fixed_83p16us |        20 | conservative_l2_stress |               1 |                1 |          -9.3513e+06  |         -9.3513e+06  |         -9.3513e+06  | True                       |                   0 |               1 |
|             1.9779e+06  |                 1799304 |       110004 |         1689300 |                       0 |                      0 |                   81976 |                   1 |           6730 |   0.436993 |    0.56297  |                 0.061137  |                    1.11154e-06 |                                         0 |                             1 | new4    | side_by_queue_imbalance_abs_0p1 | cpu_empirical      |        20 | conservative_l2_stress |               1 |                1 |          -3.61266e+06 |         -3.61266e+06 |         -3.61266e+06 | True                       |                   0 |               1 |
|             1.9779e+06  |                 1799304 |       110004 |         1689300 |                       0 |                      0 |                   81976 |                   1 |           6730 |   0.436993 |    0.56297  |                 0.061137  |                    0           |                                         0 |                             0 | new4    | side_by_queue_imbalance_abs_0p1 | fpga_fixed_83p16us |        20 | conservative_l2_stress |               1 |                1 |          -3.67482e+06 |         -3.67482e+06 |         -3.67482e+06 | True                       |                   0 |               1 |

## Matched Random Summary

| split   | scenario           |   matched_random_mean |   matched_random_95p |   matched_random_max |       policy_pnl | beats_matched_random_95p   |   empirical_p_value |   random_trials |
|:--------|:-------------------|----------------------:|---------------------:|---------------------:|-----------------:|:---------------------------|--------------------:|----------------:|
| val     | cpu_empirical      |     -845270           |    -845270           |    -845270           | 378585           | True                       |                   0 |               1 |
| val     | fpga_fixed_83p16us |     -795360           |    -795360           |    -795360           | 378585           | True                       |                   0 |               1 |
| new3    | cpu_empirical      |          -9.50408e+06 |         -9.50408e+06 |         -9.50408e+06 |      5.49544e+06 | True                       |                   0 |               1 |
| new3    | fpga_fixed_83p16us |          -9.3513e+06  |         -9.3513e+06  |         -9.3513e+06  |      5.49544e+06 | True                       |                   0 |               1 |
| new4    | cpu_empirical      |          -3.61266e+06 |         -3.61266e+06 |         -3.61266e+06 |      1.9779e+06  | True                       |                   0 |               1 |
| new4    | fpga_fixed_83p16us |          -3.67482e+06 |         -3.67482e+06 |         -3.67482e+06 |      1.9779e+06  | True                       |                   0 |               1 |
