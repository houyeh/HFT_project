# Phase 02D Final Strict Baseline Audit Report

Fixed selected policy: `side_by_queue_imbalance_abs_0p1` at horizon `H20`.

Rule definition: If queue_imbalance > +0.1 quote bid; if queue_imbalance < -0.1 quote ask; otherwise no_quote.

No new policy search, ML training, or threshold tuning was performed in Phase 02D. This package only collects Phase 02B/02C evidence and verifies advisor-safe claims.

## 1. Input Completeness

| phase   | input_name                                 | path                                                                                                                                            | exists   |   row_count |   column_count |   bytes |
|:--------|:-------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|:---------|------------:|---------------:|--------:|
| 02B     | strict_policy_grid                         | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\02b_strict_baseline_and_excess_pnl\results\strict_policy_grid.csv                         | True     |         720 |             52 |  441112 |
| 02B     | random_summary_strict                      | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\02b_strict_baseline_and_excess_pnl\results\random_summary_strict.csv                      | True     |         720 |             19 |  195484 |
| 02B     | random_summary_strict_exact_top_candidates | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\02b_strict_baseline_and_excess_pnl\results\random_summary_strict_exact_top_candidates.csv | True     |           3 |             16 |     995 |
| 02B     | fee_sensitivity                            | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\02b_strict_baseline_and_excess_pnl\results\fee_sensitivity.csv                            | True     |        2880 |             10 |  284889 |
| 02B     | strict_selected_candidates                 | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\02b_strict_baseline_and_excess_pnl\results\strict_selected_candidates.json                | True     |         nan |            nan |    7477 |
| 02C     | queue_stress_grid                          | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\02c_queue_aware_stress\results\queue_stress_grid.csv                                      | True     |         900 |             15 |  140543 |
| 02C     | delay_sensitivity                          | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\02c_queue_aware_stress\results\delay_sensitivity.csv                                      | True     |          45 |             15 |    7179 |
| 02C     | fill_keep_break_even                       | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\02c_queue_aware_stress\results\fill_keep_break_even.csv                                   | True     |           8 |              3 |     168 |

## 2. Selected Policy vs Controls

| split   |   total_pnl |   old_bid_only_pnl |   selected_minus_old_bid_only | beats_old_bid_only   |   matched_random_95p | beats_random_95p   |
|:--------|------------:|-------------------:|------------------------------:|:---------------------|---------------------:|:-------------------|
| new4    | 4.95871e+08 |        5.34754e+08 |                  -3.88835e+07 | False                |          4.39163e+08 | True               |
| test    | 2.3387e+08  |        2.45096e+08 |                  -1.12261e+07 | False                |          2.03052e+08 | True               |
| val     | 1.11461e+08 |        1.1671e+08  |                  -5.24946e+06 | False                |          9.69224e+07 | True               |

**Critical interpretation:** the selected queue-imbalance rule beats matched random, but it does **not** beat `old_bid_only` in absolute total PnL. It should be framed as a non-trivial L2 ranking/excess-PnL signal, not as an absolute PnL replacement for old bid-only.

## 3. Exact Random OOS Check

In 1000 exact random trials, no random trial exceeded the selected policy. Therefore empirical p-value < 0.001.

| split   | policy_id                       |   horizon |   trials |   model_pnl |   selected_action_count |   exact_random_mean |   exact_random_std |   exact_random_90p |   exact_random_95p |   exact_random_99p | empirical_p_value_exact_random_ge_model   | beats_exact_random_mean   | beats_exact_random_95p   | matching_keys                                            | sampling_method                           | requested_match                                                                  | computational_status                |
|:--------|:--------------------------------|----------:|---------:|------------:|------------------------:|--------------------:|-------------------:|-------------------:|-------------------:|-------------------:|:------------------------------------------|:--------------------------|:-------------------------|:---------------------------------------------------------|:------------------------------------------|:---------------------------------------------------------------------------------|:------------------------------------|
| val     | side_by_queue_imbalance_abs_0p1 |        20 |     1000 | 1.11461e+08 |                  468627 |         9.66827e+07 |           156699   |        9.68867e+07 |        9.69288e+07 |        9.70222e+07 | <0.001                                    | True                      | True                     | side + session_id + spread_bucket + exact selected count | actual_without_replacement_matched_trials | nan                                                                              | nan                                 |
| test    | side_by_queue_imbalance_abs_0p1 |        20 |     1000 | 2.3387e+08  |                  971600 |         2.02888e+08 |            95366.2 |        2.03013e+08 |        2.03047e+08 |        2.03094e+08 | <0.001                                    | True                      | True                     | side + session_id + spread_bucket + exact selected count | nan                                       | matched count + side ratio + session; spread bucket included as stricter control | completed_exact_without_replacement |
| new4    | side_by_queue_imbalance_abs_0p1 |        20 |     1000 | 4.95871e+08 |                 1799304 |         4.38869e+08 |           170410   |        4.3909e+08  |        4.39142e+08 |        4.39251e+08 | <0.001                                    | True                      | True                     | side + session_id + spread_bucket + exact selected count | nan                                       | matched count + side ratio + session; spread bucket included as stricter control | completed_exact_without_replacement |

## 4. Fee Sensitivity

`eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection.

| split   | policy_id                       |   horizon |   fee_multiplier |   base_total_pnl_fee_1x |   adjusted_total_pnl |   selected_action_count |   break_even_fee_multiplier | positive_after_fee   | eligible_for_final_selection_at_1x   | eligibility_column_note                                                                                                                           |
|:--------|:--------------------------------|----------:|-----------------:|------------------------:|---------------------:|------------------------:|----------------------------:|:---------------------|:-------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|
| new4    | side_by_queue_imbalance_abs_0p1 |        20 |                1 |             4.95871e+08 |          4.95871e+08 |                 1799304 |                     19.3727 | True                 | False                                | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| new4    | side_by_queue_imbalance_abs_0p1 |        20 |                2 |             4.95871e+08 |          4.68881e+08 |                 1799304 |                     19.3727 | True                 | False                                | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| new4    | side_by_queue_imbalance_abs_0p1 |        20 |                3 |             4.95871e+08 |          4.41892e+08 |                 1799304 |                     19.3727 | True                 | False                                | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| new4    | side_by_queue_imbalance_abs_0p1 |        20 |                5 |             4.95871e+08 |          3.87913e+08 |                 1799304 |                     19.3727 | True                 | False                                | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| test    | side_by_queue_imbalance_abs_0p1 |        20 |                1 |             2.3387e+08  |          2.3387e+08  |                  971600 |                     17.047  | True                 | False                                | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| test    | side_by_queue_imbalance_abs_0p1 |        20 |                2 |             2.3387e+08  |          2.19296e+08 |                  971600 |                     17.047  | True                 | False                                | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| test    | side_by_queue_imbalance_abs_0p1 |        20 |                3 |             2.3387e+08  |          2.04722e+08 |                  971600 |                     17.047  | True                 | False                                | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| test    | side_by_queue_imbalance_abs_0p1 |        20 |                5 |             2.3387e+08  |          1.75574e+08 |                  971600 |                     17.047  | True                 | False                                | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| val     | side_by_queue_imbalance_abs_0p1 |        20 |                1 |             1.11461e+08 |          1.11461e+08 |                  468627 |                     16.8563 | True                 | True                                 | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| val     | side_by_queue_imbalance_abs_0p1 |        20 |                2 |             1.11461e+08 |          1.04431e+08 |                  468627 |                     16.8563 | True                 | True                                 | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| val     | side_by_queue_imbalance_abs_0p1 |        20 |                3 |             1.11461e+08 |          9.74018e+07 |                  468627 |                     16.8563 | True                 | True                                 | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| val     | side_by_queue_imbalance_abs_0p1 |        20 |                5 |             1.11461e+08 |          8.3343e+07  |                  468627 |                     16.8563 | True                 | True                                 | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |

Break-even fee points:

| split   | policy_id                       |   horizon |   base_total_pnl_fee_1x |   selected_action_count |   break_even_fee_multiplier |   max_tested_fee_multiplier_positive | eligibility_column_note                                                                                                                           |
|:--------|:--------------------------------|----------:|------------------------:|------------------------:|----------------------------:|-------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------|
| new4    | side_by_queue_imbalance_abs_0p1 |        20 |             4.95871e+08 |                 1799304 |                     19.3727 |                                    5 | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| test    | side_by_queue_imbalance_abs_0p1 |        20 |             2.3387e+08  |                  971600 |                     17.047  |                                    5 | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| val     | side_by_queue_imbalance_abs_0p1 |        20 |             1.11461e+08 |                  468627 |                     16.8563 |                                    5 | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |

## 5. Queue Stress

Moderate stress is `fill_keep_rate=0.5`, `adverse_multiplier=1.5`, and `delay=0`.

`eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection.

| policy_id                       |   horizon | split   |   fill_keep_rate |   adverse_multiplier |   action_delay_events |   state_count |   selected_action_count |   selection_rate |   stress_total_pnl |   win_rate |   loss_rate |    max_drawdown | positive_under_stress   | moderate_stress   | eligibility_column_note                                                                                                                           |
|:--------------------------------|----------:|:--------|-----------------:|---------------------:|----------------------:|--------------:|------------------------:|-----------------:|-------------------:|-----------:|------------:|----------------:|:------------------------|:------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|
| side_by_queue_imbalance_abs_0p1 |        20 | val     |              0.5 |                  1.5 |                     0 |        563477 |                  468627 |         0.83167  |        5.27034e+07 |   0.924183 |   0.0757767 |     1.03545e+06 | True                    | True              | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| side_by_queue_imbalance_abs_0p1 |        20 | test    |              0.5 |                  1.5 |                     0 |       1166321 |                  971600 |         0.833047 |        1.13652e+08 |   0.948985 |   0.0509757 | 47425           | True                    | True              | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |
| side_by_queue_imbalance_abs_0p1 |        20 | new4    |              0.5 |                  1.5 |                     0 |       2196640 |                 1799304 |         0.819116 |        2.38079e+08 |   0.931123 |   0.0688394 | 90417.5         | True                    | True              | `eligible_for_final_selection_at_1x` is meaningful only on validation. Test and NEW4 are frozen evaluation splits and are not used for selection. |

Fill-keep break-even points:

| split   |   adverse_multiplier |   min_fill_keep_rate_positive |
|:--------|---------------------:|------------------------------:|
| test    |                 1    |                           0.1 |
| test    |                 1.25 |                           0.1 |
| test    |                 1.5  |                           0.1 |
| test    |                 2    |                           0.1 |
| new4    |                 1    |                           0.1 |
| new4    |                 1.25 |                           0.1 |
| new4    |                 1.5  |                           0.1 |
| new4    |                 2    |                           0.1 |

## 6. Claim Safety Table

| claim                                                     | status     | evidence                                                                                                         | safe_wording                                                                                                                   |
|:----------------------------------------------------------|:-----------|:-----------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------|
| Full L2 action universe avoids residual bottleneck        | PASS       | Phase 01 built millions of bid/ask/no_quote candidate actions instead of the old 820-row residual veto universe. | The n2ew_era action universe removes the residual-sample bottleneck for short-horizon proxy audits.                            |
| Selected policy is non-trivial L2 rule                    | PASS       | Policy uses queue_imbalance thresholding and has selection_rate < 0.95 on val/test/NEW4.                         | The selected rule is a non-trivial L2 queue-imbalance quote-ranking primitive.                                                 |
| Selected policy beats matched random 95p on validation    | PASS       | selected_policy_vs_controls.csv, split=val.                                                                      | On validation, the selected queue-imbalance rule beats the matched-random 95p hurdle under proxy markout.                      |
| Selected policy beats matched random 95p on frozen test   | PASS       | selected_policy_vs_controls.csv and exact_random_oos_summary.csv, split=test.                                    | On frozen test, the selected rule beats the matched-random 95p hurdle under proxy markout.                                     |
| Selected policy beats matched random 95p on NEW4          | PASS       | selected_policy_vs_controls.csv and exact_random_oos_summary.csv, split=NEW4.                                    | On NEW4, the selected rule beats the matched-random 95p hurdle under proxy markout.                                            |
| Selected policy beats old bid-only absolute PnL           | FAIL       | selected_minus_old_bid_only is negative on val/test/NEW4                                                         | The selected rule is not an absolute PnL replacement for old bid-only; it is evidence of non-trivial L2 ranking/excess signal. |
| Selected policy remains positive under fee sensitivity    | PASS       | fee_sensitivity_summary.csv across fee multipliers 1x, 2x, 3x, and 5x.                                           | The selected rule remains positive under the audited proxy fee multipliers.                                                    |
| Selected policy remains positive under proxy queue stress | PROXY_PASS | queue_stress_selected_policy_summary.csv at fill_keep_rate=0.5, adverse_multiplier=1.5, delay=0.                 | The selected rule remains positive under the defined proxy queue stress; this is not true queue replay.                        |
| This is true queue replay                                 | FORBIDDEN  | No aggressor_side, true order IDs, queue position, acknowledgements, or exchange matching replay.                | Use `proxy short-horizon markout audit`, not `true exchange replay`.                                                           |
| This proves real-world profitability                      | FORBIDDEN  | No true queue priority, order acknowledgements, partial-fill replay, or production execution model.              | Do not claim real-world profitability.                                                                                         |
| FPGA improves PnL                                         | FORBIDDEN  | No production latency experiment proves an FPGA PnL improvement.                                                 | Frame FPGA only as an HLS-friendly deterministic quote-control implementation path.                                            |

## 7. Safe Conclusion

The selected L2 queue-imbalance rule is a non-trivial ranking/control signal under the current short-horizon proxy markout framework. It clears matched-random hurdles and remains positive under the defined proxy queue stress. However, it does not outperform old bid-only in absolute total PnL, and the evaluation is not true exchange replay or evidence of real-world profitability.

## Figures

- `figures/selected_vs_controls_bar.png`
- `figures/random_hurdle_bar.png`
- `figures/fee_sensitivity_selected_policy.png`
- `figures/queue_stress_heatmap_selected_policy.png`
- `figures/delay_sensitivity_selected_policy.png`
