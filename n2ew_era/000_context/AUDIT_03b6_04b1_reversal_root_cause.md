status: mixed
# Audit: 03B6 vs 04B1 Reversal Root Cause

Primary report location: `n2ew_era/03b6_hgb_convergence_closure/AUDIT_03b6_04b1_reversal_root_cause.md`. A synchronized context copy is also written to `n2ew_era/000_context/AUDIT_03b6_04b1_reversal_root_cause.md`.

This audit did not train or tune any model. It loaded the frozen 03B6 and 04B1 HGB model objects and their frozen validation-selected thresholds, recomputed development-only diagnostics on NEW1/NEW2-derived artifacts, and read existing frozen NEW3/NEW4 diagnostic CSVs only for posthoc comparison.

Hard boundary: no NEW5 path was loaded. The script hard-stops on any file path containing `NEW5`.

Runtime compatibility note: the final script was run under Python 3.13.9 with sklearn 1.7.2. The frozen HGB joblib objects loaded and predictions completed; the recomputed 04B1 TXF validation policy total matched the frozen 04B1 validation total. Treat sklearn/joblib persistence as residual risk if the runtime changes.

## Frozen Objects Used

| object | frozen setting | role | split use |
|:--|:--|:--|:--|
| 03B6 mixed HGB | `hgb_regressor_iter_320`, threshold `0.0` | diagnostic frozen model | selected on mixed validation only |
| 04B1 TXF HGB | `hgb_regressor_iter_120`, threshold `0.22419440373778343` | diagnostic frozen model | selected on TXF validation only |
| simple rule | `side_by_queue_imbalance_abs_0p1` | fixed baseline | no tuned parameters in this audit |

## Development-Only Feature Distribution

Row basis is action-row model input. For `queue_imbalance`, `deep_imbalance`, `spread_ticks`, and `event_rate_past_50`, bid/ask/no-quote triplication does not change distributional values within a state.

| analysis_scope   | universe   | split_scope            | feature            |        n |         mean |       std |       p05 |         p50 |        p95 |   abs_gt_0p1_rate |
|:-----------------|:-----------|:-----------------------|:-------------------|---------:|-------------:|----------:|----------:|------------:|-----------:|------------------:|
| development-only | txf_04b1   | train_plus_val         | deep_imbalance     | 15377784 |  -0.0202863  |  0.203108 | -0.363636 |  -0.0153846 |   0.303371 |          0.602657 |
| development-only | mixed_03b6 | train_sampled_plus_val | deep_imbalance     |  2590431 |  -0.0184984  |  0.193431 | -0.351351 |  -0.012987  |   0.283019 |          0.590064 |
| development-only | mixed_03b6 | val                    | deep_imbalance     |  1690431 |  -0.0222583  |  0.185313 | -0.348837 |  -0.0136986 |   0.263158 |          0.579252 |
| development-only | txf_04b1   | val                    | deep_imbalance     |  7966107 |  -0.0149385  |  0.197215 | -0.354167 |  -0.0120482 |   0.297297 |          0.591249 |
| development-only | txf_04b1   | train_plus_val         | event_rate_past_50 | 15377784 | 149.858      | 48.0044   | 77.3524   | 147.038     | 232.12     |          0.999959 |
| development-only | mixed_03b6 | train_sampled_plus_val | event_rate_past_50 |  2590431 | 153.772      | 46.9974   | 81.6227   | 150.994     | 234.61     |          0.999979 |
| development-only | mixed_03b6 | val                    | event_rate_past_50 |  1690431 | 154.091      | 46.2059   | 82.3811   | 151.428     | 233.764    |          0.999993 |
| development-only | txf_04b1   | val                    | event_rate_past_50 |  7966107 | 147.806      | 46.7235   | 76.0961   | 145.48      | 227.554    |          0.999969 |
| development-only | txf_04b1   | train_plus_val         | queue_imbalance    | 15377784 |  -0.00860315 |  0.43298  | -0.714286 |   0         |   0.684211 |          0.828067 |
| development-only | mixed_03b6 | train_sampled_plus_val | queue_imbalance    |  2590431 |  -0.00843595 |  0.434317 | -0.714286 |   0         |   0.692308 |          0.830252 |
| development-only | mixed_03b6 | val                    | queue_imbalance    |  1690431 |  -0.0108241  |  0.436339 | -0.714286 |   0         |   0.714286 |          0.832712 |
| development-only | txf_04b1   | val                    | queue_imbalance    |  7966107 |  -0.00437254 |  0.438408 | -0.714286 |   0         |   0.714286 |          0.832437 |
| development-only | txf_04b1   | train_plus_val         | spread_ticks       | 15377784 |   2.3981     |  0.82621  |  1        |   2         |   4        |          1        |
| development-only | mixed_03b6 | train_sampled_plus_val | spread_ticks       |  2590431 |   2.31665    |  0.945274 |  1        |   2         |   3        |          1        |
| development-only | mixed_03b6 | val                    | spread_ticks       |  1690431 |   2.2213     |  0.979103 |  1        |   2         |   3        |          1        |
| development-only | txf_04b1   | val                    | spread_ticks       |  7966107 |   2.30638    |  0.77072  |  1        |   2         |   3        |          1        |

## Mixed Universe Composition

| is_txf_like_symbol   |   action_rows |   action_row_fraction |   symbols |
|:---------------------|--------------:|----------------------:|----------:|
| txf_like             |       2590431 |                     1 |         1 |

## Mixed Universe TXF vs Non-TXF Subsets

| analysis_scope   | universe              | split_scope            | feature            |       n |         mean |       std |        p50 |        p95 |   abs_gt_0p1_rate |
|:-----------------|:----------------------|:-----------------------|:-------------------|--------:|-------------:|----------:|-----------:|-----------:|------------------:|
| development-only | mixed_txf_like_subset | train_sampled_plus_val | queue_imbalance    | 2590431 |  -0.00843595 |  0.434317 |   0        |   0.692308 |          0.830252 |
| development-only | mixed_txf_like_subset | train_sampled_plus_val | deep_imbalance     | 2590431 |  -0.0184984  |  0.193431 |  -0.012987 |   0.283019 |          0.590064 |
| development-only | mixed_txf_like_subset | train_sampled_plus_val | spread_ticks       | 2590431 |   2.31665    |  0.945274 |   2        |   3        |          1        |
| development-only | mixed_txf_like_subset | train_sampled_plus_val | event_rate_past_50 | 2590431 | 153.772      | 46.9974   | 150.994    | 234.61     |          0.999979 |

## Recomputed Permutation Importance

Method: one deterministic shuffle per feature on validation action rows, scored by Spearman drop against H20 excess PnL. This is diagnostic only and not used to pick features or thresholds.

| analysis_scope   | model_label                              | split_scope          | feature                    |   base_spearman |   importance_spearman_drop |
|:-----------------|:-----------------------------------------|:---------------------|:---------------------------|----------------:|---------------------------:|
| development-only | 03B6_mixed_hgb_iter_320_threshold_0p0    | val_NEW1_development | side_queue_imbalance       |        0.29928  |                0.168648    |
| development-only | 03B6_mixed_hgb_iter_320_threshold_0p0    | val_NEW1_development | side_imbalance_change_past |        0.29928  |                0.0875273   |
| development-only | 03B6_mixed_hgb_iter_320_threshold_0p0    | val_NEW1_development | spread_change_past         |        0.29928  |                0.0192994   |
| development-only | 03B6_mixed_hgb_iter_320_threshold_0p0    | val_NEW1_development | side_deep_imbalance        |        0.29928  |                0.0174847   |
| development-only | 03B6_mixed_hgb_iter_320_threshold_0p0    | val_NEW1_development | action_is_no_quote         |        0.29928  |                0.00884752  |
| development-only | 03B6_mixed_hgb_iter_320_threshold_0p0    | val_NEW1_development | spread_ticks               |        0.29928  |                0.00747079  |
| development-only | 03B6_mixed_hgb_iter_320_threshold_0p0    | val_NEW1_development | top_depth                  |        0.29928  |                0.00573575  |
| development-only | 03B6_mixed_hgb_iter_320_threshold_0p0    | val_NEW1_development | action_is_ask              |        0.29928  |                0.00351504  |
| development-only | 03B6_mixed_hgb_iter_320_threshold_0p0    | val_NEW1_development | deep_imbalance             |        0.29928  |                0.00111899  |
| development-only | 03B6_mixed_hgb_iter_320_threshold_0p0    | val_NEW1_development | event_rate_past_50         |        0.29928  |                0.000257125 |
| development-only | 03B6_mixed_hgb_iter_320_threshold_0p0    | val_NEW1_development | queue_imbalance            |        0.29928  |               -6.16146e-05 |
| development-only | 04B1_txf_hgb_iter_120_threshold_0p224194 | val_NEW1_development | side_queue_imbalance       |        0.297149 |                0.173668    |
| development-only | 04B1_txf_hgb_iter_120_threshold_0p224194 | val_NEW1_development | side_imbalance_change_past |        0.297149 |                0.0845928   |
| development-only | 04B1_txf_hgb_iter_120_threshold_0p224194 | val_NEW1_development | side_deep_imbalance        |        0.297149 |                0.018603    |
| development-only | 04B1_txf_hgb_iter_120_threshold_0p224194 | val_NEW1_development | action_is_ask              |        0.297149 |                0.0141385   |
| development-only | 04B1_txf_hgb_iter_120_threshold_0p224194 | val_NEW1_development | spread_change_past         |        0.297149 |                0.0132172   |
| development-only | 04B1_txf_hgb_iter_120_threshold_0p224194 | val_NEW1_development | top_depth                  |        0.297149 |                0.00777913  |
| development-only | 04B1_txf_hgb_iter_120_threshold_0p224194 | val_NEW1_development | side_sign                  |        0.297149 |                0.00356498  |
| development-only | 04B1_txf_hgb_iter_120_threshold_0p224194 | val_NEW1_development | volatility_past_50         |        0.297149 |                0.00315014  |
| development-only | 04B1_txf_hgb_iter_120_threshold_0p224194 | val_NEW1_development | spread_ticks               |        0.297149 |                0.00161977  |
| development-only | 04B1_txf_hgb_iter_120_threshold_0p224194 | val_NEW1_development | queue_imbalance            |        0.297149 |                9.23107e-05 |
| development-only | 04B1_txf_hgb_iter_120_threshold_0p224194 | val_NEW1_development | event_rate_past_50         |        0.297149 |                2.10523e-05 |
| development-only | 04B1_txf_hgb_iter_120_threshold_0p224194 | val_NEW1_development | deep_imbalance             |        0.297149 |               -0.000128732 |

Top 03B6 feature by this diagnostic: `side_queue_imbalance` with Spearman drop `0.1686`. Top 04B1 feature: `side_queue_imbalance` with Spearman drop `0.1737`.

## Same TXF Validation Prediction Error

Both frozen models were run on the same pure-TXF NEW1 validation action rows. This is the development-only held-out comparison requested in B; no NEW3/NEW4 row is used here.

| analysis_scope   | split_scope          | heldout_universe   | model_label                        |   frozen_threshold |   action_rows |   rmse_vs_target_excess |   mae_vs_target_excess |   spearman_score_vs_target |   pr_auc_positive_excess |   roc_auc_positive_excess |
|:-----------------|:---------------------|:-------------------|:-----------------------------------|-------------------:|--------------:|------------------------:|-----------------------:|---------------------------:|-------------------------:|--------------------------:|
| development-only | val_NEW1_development | txf_04b_val        | 03B6_mixed_hgb_iter_320_on_txf_val |           0        |       7966107 |                 130.349 |                74.8971 |                   0.261132 |                 0.339797 |                  0.575284 |
| development-only | val_NEW1_development | txf_04b_val        | 04B1_txf_hgb_iter_120_on_txf_val   |           0.224194 |       7966107 |                 121.183 |                71.2254 |                   0.29034  |                 0.36913  |                  0.582421 |

## Same TXF Validation Policy PnL

| analysis_scope   | split                | policy_id                                    |   state_count |   selected_action_count |   selection_rate |   bid_count |   ask_count |   total_excess_pnl_H20 |   avg_excess_pnl_per_selected |   win_rate |   max_drawdown |
|:-----------------|:---------------------|:---------------------------------------------|--------------:|------------------------:|-----------------:|------------:|------------:|-----------------------:|------------------------------:|-----------:|---------------:|
| development-only | val_NEW1_development | side_by_queue_imbalance_abs_0p1              |       2655369 |                 2208424 |         0.831683 |     1096770 |     1111654 |            4.04909e+07 |                       18.3348 |   0.427169 |         293385 |
| development-only | val_NEW1_development | 03B6_mixed_hgb_iter_320_threshold_0p0_on_txf |       2655369 |                 2050994 |         0.772395 |     1020885 |     1030109 |            4.8964e+07  |                       23.8733 |   0.449872 |         189505 |
| development-only | val_NEW1_development | 04B1_txf_hgb_iter_120_threshold_0p224194     |       2655369 |                 1980301 |         0.745772 |     1046055 |      934246 |            5.45502e+07 |                       27.5464 |   0.460805 |         172560 |

On the same TXF validation data, 03B6 frozen ML PnL is `48,964,030`, 04B1 frozen ML PnL is `54,550,215`, and the queue rule is `40,490,925`.

## Same TXF Validation Decision Overlap

| analysis_scope   | split_scope          | heldout_universe   |   state_count |   both_selected |   only_03b6_selected |   only_04b1_selected |   neither_selected |   same_side_selected |   opposite_side_selected |   both_selected_rate |   only_03b6_selected_rate |   only_04b1_selected_rate |   neither_selected_rate |   opposite_side_among_both_selected_rate | test_new3_new4_used_for_selection   | new5_used   |
|:-----------------|:---------------------|:-------------------|--------------:|----------------:|---------------------:|---------------------:|-------------------:|---------------------:|-------------------------:|---------------------:|--------------------------:|--------------------------:|------------------------:|-----------------------------------------:|:------------------------------------|:------------|
| development-only | val_NEW1_development | txf_04b_val        |       2655369 |         1839341 |               211653 |               140960 |             463415 |              1782946 |                    56395 |             0.692688 |                 0.0797076 |                 0.0530849 |                 0.17452 |                                0.0306604 | False                               | False       |

Only-04B1-selected states are `5.31%` of TXF validation states, while only-03B6-selected states are `7.97%`. This is a direct development-only measurement of changed HGB decision boundary behavior.

## TXF Validation Queue-Imbalance Bins

| analysis_scope   | split_scope          | feature_bin         | bin       | policy_or_model                              |   state_count |   selected_action_count |   selection_rate |   total_excess_pnl_H20 |   avg_excess_pnl_per_selected |   win_rate |   score_mean |   score_p50 |   score_p95 |
|:-----------------|:---------------------|:--------------------|:----------|:---------------------------------------------|--------------:|------------------------:|-----------------:|-----------------------:|------------------------------:|-----------:|-------------:|------------:|------------:|
| development-only | val_NEW1_development | abs_queue_imbalance | 0_0p05    | 03B6_mixed_hgb_iter_320_threshold_0p0_on_txf |        341121 |                  198146 |         0.580867 |            1.80502e+06 |                       9.10955 |   0.428532 |      9.2943  |     9.95346 |     40.4011 |
| development-only | val_NEW1_development | abs_queue_imbalance | 0p05_0p10 | 03B6_mixed_hgb_iter_320_threshold_0p0_on_txf |        105824 |                   62054 |         0.586389 |       456405           |                       7.35497 |   0.44479  |      8.75926 |     6.26483 |     34.5138 |
| development-only | val_NEW1_development | abs_queue_imbalance | 0p10_0p20 | 03B6_mixed_hgb_iter_320_threshold_0p0_on_txf |        433560 |                  318541 |         0.73471  |            2.885e+06   |                       9.05693 |   0.41734  |     13.6455  |     5.74486 |     42.5519 |
| development-only | val_NEW1_development | abs_queue_imbalance | 0p20_0p40 | 03B6_mixed_hgb_iter_320_threshold_0p0_on_txf |        646874 |                  484094 |         0.748359 |            8.5311e+06  |                      17.6228  |   0.443015 |     18.8587  |    12.2284  |     56.1536 |
| development-only | val_NEW1_development | abs_queue_imbalance | 0p40_1p00 | 03B6_mixed_hgb_iter_320_threshold_0p0_on_txf |       1127990 |                  988159 |         0.876035 |            3.52865e+07 |                      35.7093  |   0.468315 |     44.6974  |    39.0904  |     94.5022 |
| development-only | val_NEW1_development | abs_queue_imbalance | 0_0p05    | 04B1_txf_hgb_iter_120_threshold_0p224194     |        341121 |                  188863 |         0.553654 |            2.33828e+06 |                      12.3808  |   0.439509 |      4.60434 |     1.38255 |     23.2327 |
| development-only | val_NEW1_development | abs_queue_imbalance | 0p05_0p10 | 04B1_txf_hgb_iter_120_threshold_0p224194     |        105824 |                   56529 |         0.534179 |       742180           |                      13.1292  |   0.463054 |      2.48659 |     2.4908  |     14.3639 |
| development-only | val_NEW1_development | abs_queue_imbalance | 0p10_0p20 | 04B1_txf_hgb_iter_120_threshold_0p224194     |        433560 |                  310807 |         0.716872 |            3.72232e+06 |                      11.9763  |   0.426219 |      6.83292 |     3.65563 |     27.9887 |
| development-only | val_NEW1_development | abs_queue_imbalance | 0p20_0p40 | 04B1_txf_hgb_iter_120_threshold_0p224194     |        646874 |                  463461 |         0.716463 |            9.18298e+06 |                      19.8139  |   0.451322 |     12.4447  |    10.4848  |     43.776  |
| development-only | val_NEW1_development | abs_queue_imbalance | 0p40_1p00 | 04B1_txf_hgb_iter_120_threshold_0p224194     |       1127990 |                  960641 |         0.85164  |            3.85645e+07 |                      40.1445  |   0.480625 |     31.7809  |    36.363   |     63.9053 |
| development-only | val_NEW1_development | abs_queue_imbalance | 0_0p05    | side_by_queue_imbalance_abs_0p1              |        341121 |                       0 |         0        |            0           |                       0       |   0        |    nan       |   nan       |    nan      |
| development-only | val_NEW1_development | abs_queue_imbalance | 0p05_0p10 | side_by_queue_imbalance_abs_0p1              |        105824 |                       0 |         0        |            0           |                       0       |   0        |    nan       |   nan       |    nan      |
| development-only | val_NEW1_development | abs_queue_imbalance | 0p10_0p20 | side_by_queue_imbalance_abs_0p1              |        433560 |                  433560 |         1        |           -1.06306e+06 |                      -2.45194 |   0.378287 |    nan       |   nan       |    nan      |
| development-only | val_NEW1_development | abs_queue_imbalance | 0p20_0p40 | side_by_queue_imbalance_abs_0p1              |        646874 |                  646874 |         1        |            5.22986e+06 |                       8.08483 |   0.410531 |    nan       |   nan       |    nan      |
| development-only | val_NEW1_development | abs_queue_imbalance | 0p40_1p00 | side_by_queue_imbalance_abs_0p1              |       1127990 |                 1127990 |         1        |            3.63241e+07 |                      32.2025  |   0.4555   |    nan       |   nan       |    nan      |

## Simple Rule Development Contrast

This table is development-only and recomputed from existing feature/action artifacts. It checks whether the simple queue rule itself collapses when moving from mixed to TXF. The mixed train sample is intentionally excluded here because the Phase 03B train matrix is an action-row sample and does not preserve complete bid/ask state triplets for policy PnL.

| analysis_scope   | universe   | split_scope   | policy_id                       |   state_count |   selected_action_count |   selection_rate |   total_excess_pnl_H20 |   avg_excess_pnl_per_selected |   win_rate |   max_drawdown |
|:-----------------|:-----------|:--------------|:--------------------------------|--------------:|------------------------:|-----------------:|-----------------------:|------------------------------:|-----------:|---------------:|
| development-only | mixed_03b6 | val           | side_by_queue_imbalance_abs_0p1 |        563477 |                  469214 |         0.832712 |            7.76908e+06 |                       16.5576 |   0.429915 |         701710 |
| development-only | txf_04b1   | train         | side_by_queue_imbalance_abs_0p1 |       2470559 |                 2032673 |         0.822758 |            4.20832e+07 |                       20.7034 |   0.438631 |         318275 |
| development-only | txf_04b1   | val           | side_by_queue_imbalance_abs_0p1 |       2655369 |                 2208424 |         0.831683 |            4.04909e+07 |                       18.3348 |   0.427169 |         293385 |

Validation avg PnL per selected action for the queue rule is `16.5576` in mixed 03B6 validation and `18.3348` in TXF-only validation.

## NEW3/NEW4 Posthoc Diagnostics

These numbers are read from already frozen diagnostic CSVs and used only to check whether the development-only pattern is consistent out of sample. They were not used to pick models, thresholds, features, or hypotheses.

| analysis_scope               | universe   | split   | metric_family          | policy_or_model                               |   ml_total_excess_pnl_H20 |   queue_rule_total_excess_pnl_H20 |   ml_minus_queue_rule | ml_beats_queue_rule   |
|:-----------------------------|:-----------|:--------|:-----------------------|:----------------------------------------------|--------------------------:|----------------------------------:|----------------------:|:----------------------|
| development-only             | txf_04b1   | val     | instant_excess_markout | ml_hgb_regressor_shallow_convergence_selected |               5.45502e+07 |                       4.04909e+07 |           1.40593e+07 | True                  |
| development-only             | txf_04b1   | val     | causal_fill_sequential | ml_hgb_regressor_shallow_convergence_selected |               2.7386e+06  |                       2.0572e+06  |      681390           | True                  |
| NEW3_NEW4_posthoc_diagnostic | mixed_03b6 | new3    | instant_excess_markout | hgb_convergence_selected                      |               7.53909e+07 |                       8.67324e+07 |          -1.13415e+07 | False                 |
| NEW3_NEW4_posthoc_diagnostic | txf_04b1   | new3    | instant_excess_markout | ml_hgb_regressor_shallow_convergence_selected |               1.29325e+08 |                       9.38726e+07 |           3.54525e+07 | True                  |
| NEW3_NEW4_posthoc_diagnostic | txf_04b1   | new3    | causal_fill_sequential | ml_hgb_regressor_shallow_convergence_selected |               7.45686e+06 |                       5.7336e+06  |           1.72326e+06 | True                  |
| NEW3_NEW4_posthoc_diagnostic | mixed_03b6 | new4    | instant_excess_markout | hgb_convergence_selected                      |               2.64169e+07 |                       2.9901e+07  |          -3.4841e+06  | False                 |
| NEW3_NEW4_posthoc_diagnostic | txf_04b1   | new4    | instant_excess_markout | ml_hgb_regressor_shallow_convergence_selected |               4.60936e+07 |                       3.23474e+07 |           1.37462e+07 | True                  |
| NEW3_NEW4_posthoc_diagnostic | txf_04b1   | new4    | causal_fill_sequential | ml_hgb_regressor_shallow_convergence_selected |               2.72002e+06 |                       2.03515e+06 |      684870           | True                  |

## Mechanism Assessment

The development-only evidence supports part of the reversal story but rejects the simplest version of it. The current 03B/03B6 development matrices used by the frozen model contain only `TXFB6`, not a visible mixture of TXF plus non-TXF symbols. Therefore the available artifacts cannot prove that non-TXF products polluted the HGB.

The same-TXF held-out run is still a strong sanity check: when both frozen models are applied to the same TXF validation data, 04B1 has lower error, stronger rank metrics, and higher policy PnL. This points to a changed HGB decision boundary caused by the rebuilt TXF action universe, feature schema, training coverage, and threshold scale. The simple queue rule remains positive and similar in validation average PnL per selected action, so the reversal is ML-specific rather than a collapse of the queue rule itself.

## Honest Conclusion

diagnosed but inconclusive. The audit finds a concrete negative result: the available 03B/03B6 development feature matrices used here contain only the TXFB6 symbol, so the current artifacts do not substantiate the simple explanation that removing non-TXF products alone caused the reversal. On the same TXF validation rows, 04B1 still ranks and trades better than the frozen 03B6 model, so a real ML decision-boundary change exists. The supported mechanism is broader than product filtering: Phase 04B/04B1 changed the TXF action-universe coverage/loader contract and feature schema, then fit and thresholded the HGB on that rebuilt TXF development universe. The exact contribution of coverage, schema, and fit/threshold scale cannot be uniquely separated from the frozen artifacts without rebuilding historical intermediates.

No NEW5 data was loaded, and no NEW3/NEW4 number was used for selection or tuning in this audit.
