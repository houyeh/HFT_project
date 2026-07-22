# ML Readiness Report

Readiness classification: `ready_only_for_simple_linear_models`.

The dataset is large and contains measurable L2 signal, but readiness is constrained by overlapping labels, drift, or missing requested features. Start with simple linear/action-value baselines before larger models.

No ML models were trained in this phase. Test, NEW3, and NEW4 were used only for frozen/OOS diagnostics, not tuning.

## 1. Dataset Size

| split   | action_side   |   row_count |   finite_label_count |   positive_count |   negative_count |   zero_count |   unique_sessions |   positive_rate |   negative_rate |   zero_rate |
|:--------|:--------------|------------:|---------------------:|-----------------:|-----------------:|-------------:|------------------:|----------------:|----------------:|------------:|
| new3    | ask           |     5657362 |              5657242 |          2022479 |          3634763 |            0 |                 6 |        0.357503 |        0.642497 |           0 |
| new3    | bid           |     5657362 |              5657242 |          2020025 |          3637217 |            0 |                 6 |        0.357069 |        0.642931 |           0 |
| new3    | no_quote      |     5657362 |              5657362 |                0 |                0 |      5657362 |                 6 |        0        |        0        |           1 |
| new4    | ask           |     2196640 |              2196560 |           782271 |          1414289 |            0 |                 4 |        0.356135 |        0.643865 |           0 |
| new4    | bid           |     2196640 |              2196560 |           783961 |          1412599 |            0 |                 4 |        0.356904 |        0.643096 |           0 |
| new4    | no_quote      |     2196640 |              2196640 |                0 |                0 |      2196640 |                 4 |        0        |        0        |           1 |
| test    | ask           |     1166321 |              1166281 |           348215 |           818066 |            0 |                 2 |        0.298569 |        0.701431 |           0 |
| test    | bid           |     1166321 |              1166281 |           358529 |           807752 |            0 |                 2 |        0.307412 |        0.692588 |           0 |
| test    | no_quote      |     1166321 |              1166321 |                0 |                0 |      1166321 |                 2 |        0        |        0        |           1 |
| train   | ask           |     3592764 |              3592644 |          1256613 |          2336031 |            0 |                 6 |        0.349774 |        0.650226 |           0 |
| train   | bid           |     3592764 |              3592644 |          1274245 |          2318399 |            0 |                 6 |        0.354682 |        0.645318 |           0 |
| train   | no_quote      |     3592764 |              3592764 |                0 |                0 |      3592764 |                 6 |        0        |        0        |           1 |
| val     | ask           |      563477 |               563457 |           195777 |           367680 |            0 |                 1 |        0.347457 |        0.652543 |           0 |
| val     | bid           |      563477 |               563457 |           198823 |           364634 |            0 |                 1 |        0.352863 |        0.647137 |           0 |
| val     | no_quote      |      563477 |               563477 |                0 |                0 |       563477 |                 1 |        0        |        0        |           1 |

Bid/ask train action rows: `7,185,528`. Train positive excess rate across bid/ask: `0.3522`.

## 2. Session Independence

Autocorrelation is measured by event order within each session/action side. High lag-20 autocorrelation indicates overlapping H20 labels and reduced effective sample independence.

| split   | action_side   |   sessions |   mean_lag1_autocorr |   mean_lag20_autocorr |   median_effective_sample_ratio_lag1 |   high_overlap_rate |
|:--------|:--------------|-----------:|---------------------:|----------------------:|-------------------------------------:|--------------------:|
| new3    | ask           |          6 |             0.911806 |            0.0648286  |                            0.0459089 |                   0 |
| new3    | bid           |          6 |             0.911806 |            0.0648286  |                            0.0459089 |                   0 |
| new3    | no_quote      |          6 |           nan        |          nan          |                          nan         |                   0 |
| new4    | ask           |          4 |             0.910643 |            0.0177486  |                            0.0467383 |                   0 |
| new4    | bid           |          4 |             0.910643 |            0.0177486  |                            0.0467383 |                   0 |
| new4    | no_quote      |          4 |           nan        |          nan          |                          nan         |                   0 |
| test    | ask           |          2 |             0.934895 |           -0.00578334 |                            0.033704  |                   0 |
| test    | bid           |          2 |             0.934895 |           -0.00578334 |                            0.033704  |                   0 |
| test    | no_quote      |          2 |           nan        |          nan          |                          nan         |                   0 |
| train   | ask           |          6 |             0.945354 |            0.0533395  |                            0.02448   |                   0 |
| train   | bid           |          6 |             0.945354 |            0.0533395  |                            0.02448   |                   0 |
| train   | no_quote      |          6 |           nan        |          nan          |                          nan         |                   0 |
| val     | ask           |          1 |             0.900265 |           -0.232319   |                            0.0524846 |                   0 |
| val     | bid           |          1 |             0.900265 |           -0.232319   |                            0.0524846 |                   0 |
| val     | no_quote      |          1 |           nan        |          nan          |                          nan         |                   0 |

## 3. Feature / Label Drift

Max feature PSI vs train across frozen/OOS splits: `1.0188`.

Max label PSI vs train across frozen/OOS splits: `0.0437`.

Feature drift output: `results/feature_drift_for_ml.csv`.

Label drift output: `results/label_drift_for_ml.csv`.

## 4. Single-Feature Signal Diagnostics

| feature         | split   | action_side   | status   |   sample_count |   spearman_corr |   base_mean_excess_pnl |   bottom_decile_mean_excess_pnl |   top_decile_mean_excess_pnl |   top_decile_lift_vs_base |   bottom_decile_lift_vs_base | decile_edges_source          |
|:----------------|:--------|:--------------|:---------|---------------:|----------------:|-----------------------:|--------------------------------:|-----------------------------:|--------------------------:|-----------------------------:|:-----------------------------|
| queue_imbalance | test    | bid           | ok       |         100154 |       0.250438  |               -13.2756 |                       -63.3827  |                     42.3444  |                  55.62    |                     -50.107  | train_split_same_action_side |
| queue_imbalance | test    | ask           | ok       |          99846 |      -0.248116  |               -16.1769 |                        33.5788  |                    -71.4856  |                 -55.3087  |                      49.7556 | train_split_same_action_side |
| queue_imbalance | train   | ask           | ok       |         300258 |      -0.2365    |               -15.7158 |                        42.5615  |                    -76.1514  |                 -60.4357  |                      58.2772 | train_split_same_action_side |
| queue_imbalance | train   | bid           | ok       |         299742 |       0.232362  |               -15.1805 |                       -72.1658  |                     43.1886  |                  58.3691  |                     -56.9853 | train_split_same_action_side |
| queue_imbalance | val     | bid           | ok       |          50192 |       0.230015  |               -16.1914 |                       -67.0271  |                     24.8894  |                  41.0808  |                     -50.8357 | train_split_same_action_side |
| queue_imbalance | val     | ask           | ok       |          49808 |      -0.228924  |               -13.5323 |                        35.3959  |                    -57.5338  |                 -44.0015  |                      48.9282 | train_split_same_action_side |
| queue_imbalance | new3    | ask           | ok       |         299763 |      -0.216206  |               -15.2202 |                        41.0383  |                    -72.387   |                 -57.1668  |                      56.2585 | train_split_same_action_side |
| queue_imbalance | new3    | bid           | ok       |         300237 |       0.215201  |               -14.6173 |                       -69.3234  |                     41.1079  |                  55.7252  |                     -54.7061 | train_split_same_action_side |
| queue_imbalance | new4    | ask           | ok       |         199801 |      -0.198613  |               -16.365  |                        33.5908  |                    -71.4133  |                 -55.0483  |                      49.9558 | train_split_same_action_side |
| queue_imbalance | new4    | bid           | ok       |         200199 |       0.195022  |               -14.0948 |                       -63.7307  |                     41.554   |                  55.6489  |                     -49.6359 | train_split_same_action_side |
| deep_imbalance  | val     | bid           | ok       |          50192 |       0.102826  |               -16.1914 |                       -37.1415  |                      1.18062 |                  17.3721  |                     -20.95   | train_split_same_action_side |
| deep_imbalance  | test    | bid           | ok       |         100154 |       0.102631  |               -13.2756 |                       -32.1219  |                     12.7993  |                  26.0749  |                     -18.8463 | train_split_same_action_side |
| deep_imbalance  | val     | ask           | ok       |          49808 |      -0.101904  |               -13.5323 |                         7.17646 |                    -18.0612  |                  -4.52895 |                      20.7087 | train_split_same_action_side |
| deep_imbalance  | test    | ask           | ok       |          99846 |      -0.100715  |               -16.1769 |                         4.30202 |                    -40.9264  |                 -24.7495  |                      20.4789 | train_split_same_action_side |
| deep_imbalance  | train   | ask           | ok       |         300258 |      -0.0987661 |               -15.7158 |                        11.3008  |                    -46.0426  |                 -30.3268  |                      27.0165 | train_split_same_action_side |
| deep_imbalance  | train   | bid           | ok       |         299742 |       0.096876  |               -15.1805 |                       -42.3727  |                     12.4808  |                  27.6613  |                     -27.1922 | train_split_same_action_side |
| deep_imbalance  | new3    | ask           | ok       |         299763 |      -0.0913604 |               -15.2202 |                        10.1435  |                    -44.0581  |                 -28.8379  |                      25.3637 | train_split_same_action_side |
| deep_imbalance  | new3    | bid           | ok       |         300237 |       0.0902125 |               -14.6173 |                       -40.2003  |                     14.2731  |                  28.8904  |                     -25.583  | train_split_same_action_side |
| deep_imbalance  | new4    | ask           | ok       |         199801 |      -0.0760922 |               -16.365  |                         7.85957 |                    -41.2256  |                 -24.8606  |                      24.2246 | train_split_same_action_side |
| deep_imbalance  | new4    | bid           | ok       |         200199 |       0.0715989 |               -14.0948 |                       -36.7516  |                      8.37801 |                  22.4728  |                     -22.6568 | train_split_same_action_side |

Bucket-lift figures are in `figures/bucket_lift_*.png`.

## 5. Readiness Decision

`ready_only_for_simple_linear_models`: The dataset is large and contains measurable L2 signal, but readiness is constrained by overlapping labels, drift, or missing requested features. Start with simple linear/action-value baselines before larger models.

Important caveat: `event_rate_past_50` is requested for ML readiness but is absent from the Phase 02F parquet schema. It should be restored before any full ML pipeline if the source supports it.

Recommended next step: run a small validation-only baseline suite using linear/logistic action-value models before considering tree or neural models.
