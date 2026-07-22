# Short-Horizon Sequential Proxy Baseline Report

Phase 02 evaluates deterministic, non-ML quote-control baselines on the Phase 01 full L2 action universe. No ML was trained, no test/NEW4 tuning was performed, and every state maps to at most one action: `bid`, `ask`, or `no_quote`.

## 1. Validation-Selected Policy

Best main policy selected strictly on validation: `ask_all` at horizon `H1`.
Selection split: `val`. Test used for selection: `False`. NEW4 used for selection: `False`.

Selection constraints:

|   min_action_count |   max_drawdown_limit | required_total_pnl_positive   | score                                                                           |
|-------------------:|---------------------:|:------------------------------|:--------------------------------------------------------------------------------|
|               1000 |                2e+06 | True                          | total_pnl - 0.10 * max_drawdown, tie-break toward simpler HLS-friendly policies |

Validation metrics for the selected policy:

| split   |   horizon | policy_id   |   total_pnl |   selected_action_count |   selection_rate |   win_rate |   max_drawdown |   avg_pnl_per_selected_action |
|:--------|----------:|:------------|------------:|------------------------:|-----------------:|-----------:|---------------:|------------------------------:|
| val     |         1 | ask_all     | 1.16712e+08 |                  563477 |                1 |   0.993127 |          26430 |                       207.129 |

Backup policies:

| role     | policy_id                |   selected_horizon | policy_type              | family   |   threshold |
|:---------|:-------------------------|-------------------:|:-------------------------|:---------|------------:|
| backup_1 | bid_all                  |                  1 | bid_all                  | control  |         nan |
| backup_2 | old_bid_only_spread_ge_1 |                  1 | old_bid_only_spread_ge_1 | control  |         nan |

## 2. Frozen Test and NEW4

The selected policies were evaluated on `test` and `new4` without changing thresholds, horizons, policy family, or weights.

| role     | policy_id                |   horizon | split   | selection_split   | test_used_for_selection   | new4_used_for_selection   |   total_pnl |   selected_action_count |   selection_rate |   win_rate |   avg_pnl_per_selected_action |   max_drawdown |   no_quote_pnl |   old_bid_only_pnl | beats_no_quote   | beats_old_bid_only   |
|:---------|:-------------------------|----------:|:--------|:------------------|:--------------------------|:--------------------------|------------:|------------------------:|-----------------:|-----------:|------------------------------:|---------------:|---------------:|-------------------:|:-----------------|:---------------------|
| main     | ask_all                  |         1 | test    | val               | False                     | False                     | 2.43492e+08 |                 1166321 |                1 |   0.992949 |                       208.769 |           1015 |              0 |         2.4365e+08 | True             | False                |
| main     | ask_all                  |         1 | new4    | val               | False                     | False                     | 5.33655e+08 |                 2196640 |                1 |   0.989774 |                       242.942 |           3230 |              0 |         5.3376e+08 | True             | False                |
| backup_1 | bid_all                  |         1 | test    | val               | False                     | False                     | 2.4365e+08  |                 1166321 |                1 |   0.993001 |                       208.905 |           4615 |              0 |         2.4365e+08 | True             | False                |
| backup_1 | bid_all                  |         1 | new4    | val               | False                     | False                     | 5.3376e+08  |                 2196640 |                1 |   0.989834 |                       242.989 |           3860 |              0 |         5.3376e+08 | True             | False                |
| backup_2 | old_bid_only_spread_ge_1 |         1 | test    | val               | False                     | False                     | 2.4365e+08  |                 1166321 |                1 |   0.993001 |                       208.905 |           4615 |              0 |         2.4365e+08 | True             | False                |
| backup_2 | old_bid_only_spread_ge_1 |         1 | new4    | val               | False                     | False                     | 5.3376e+08  |                 2196640 |                1 |   0.989834 |                       242.989 |           3860 |              0 |         5.3376e+08 | True             | False                |

## 3. Baseline and Random Hurdles

Main-policy hurdle answer:

| role   | policy_id   |   horizon | split   | selection_split   | test_used_for_selection   | new4_used_for_selection   |   total_pnl |   selected_action_count |   selection_rate |   win_rate |   avg_pnl_per_selected_action |   max_drawdown |   no_quote_pnl |   old_bid_only_pnl | beats_no_quote   | beats_old_bid_only   |   random_95p |   random_99p |   empirical_p_value_random_ge_model | beats_random_95p   |
|:-------|:------------|----------:|:--------|:------------------|:--------------------------|:--------------------------|------------:|------------------------:|-----------------:|-----------:|------------------------------:|---------------:|---------------:|-------------------:|:-----------------|:---------------------|-------------:|-------------:|------------------------------------:|:-------------------|
| main   | ask_all     |         1 | test    | val               | False                     | False                     | 2.43492e+08 |                 1166321 |                1 |   0.992949 |                       208.769 |           1015 |              0 |         2.4365e+08 | True             | False                |  2.43627e+08 |  2.43653e+08 |                               0.995 | False              |
| main   | ask_all     |         1 | new4    | val               | False                     | False                     | 5.33655e+08 |                 2196640 |                1 |   0.989774 |                       242.942 |           3230 |              0 |         5.3376e+08 | True             | False                |  5.3384e+08  |  5.33893e+08 |                               0.774 | False              |

The matched-count random hurdle is evaluated only for the validation-selected main policy. Backup policies are shown in the frozen table but are not used to rescue the main claim.

| split   | policy_id   |   horizon | random_baseline            |   trials |   model_pnl |   model_selected_action_count |   random_mean |   random_std |   random_90p |   random_95p |   random_99p |   empirical_p_value_random_ge_model | beats_random_95p   |
|:--------|:------------|----------:|:---------------------------|---------:|------------:|------------------------------:|--------------:|-------------:|-------------:|-------------:|-------------:|------------------------------------:|:-------------------|
| val     | ask_all     |         1 | random_action_uniform      |     1000 | 1.16712e+08 |                        563477 |   7.78112e+07 |      99649.8 |  7.79317e+07 |  7.79727e+07 |  7.80405e+07 |                               0     | True               |
| val     | ask_all     |         1 | random_quote_matched_count |     1000 | 1.16712e+08 |                        563477 |   1.16715e+08 |      69159.9 |  1.168e+08   |  1.16822e+08 |  1.16869e+08 |                               0.512 | False              |
| test    | ask_all     |         1 | random_action_uniform      |     1000 | 2.43492e+08 |                       1166321 |   1.62381e+08 |     120188   |  1.6253e+08  |  1.62576e+08 |  1.62664e+08 |                               0     | True               |
| test    | ask_all     |         1 | random_quote_matched_count |     1000 | 2.43492e+08 |                       1166321 |   2.43572e+08 |      33779   |  2.43614e+08 |  2.43627e+08 |  2.43653e+08 |                               0.995 | False              |
| new4    | ask_all     |         1 | random_action_uniform      |     1000 | 5.33655e+08 |                       2196640 |   3.55802e+08 |     195173   |  3.56046e+08 |  3.56117e+08 |  3.56282e+08 |                               0     | True               |
| new4    | ask_all     |         1 | random_quote_matched_count |     1000 | 5.33655e+08 |                       2196640 |   5.33711e+08 |      75521.2 |  5.33807e+08 |  5.3384e+08  |  5.33893e+08 |                               0.774 | False              |

## 4. Leave-One-Session-Out Robustness

| split   |   full_sample_pnl |   min_loso_pnl |   positive_loso_count |   session_count |
|:--------|------------------:|---------------:|----------------------:|----------------:|
| new4    |       5.33655e+08 |    2.85913e+08 |                     4 |               4 |
| test    |       2.43492e+08 |    9.28316e+07 |                     2 |               2 |
| val     |       1.16712e+08 |    0           |                     0 |               1 |

## 5. Interpretation

This phase is a short-horizon non-ML baseline audit. Positive validation PnL is allowed to select a deterministic policy, but frozen test and NEW4 decide whether the rule transfers.

The reported PnL is short-horizon sequential proxy PnL based on future-mid markout labels. It is not true exchange replay, does not model queue priority, and does not prove real-world profitability.

## 6. Output Files

- `results/baseline_grid_all_splits.csv`
- `results/selected_phase02_policy.json`
- `results/frozen_baseline_test_new4.csv`
- `results/random_baseline_summary.csv`
- `results/loso_robustness_by_policy.csv`
- `figures/validation_policy_pnl_heatmap.png`
- `figures/validation_family_pnl_heatmap.png`
- `figures/selected_policy_equity_curves.png`
- `figures/selected_policy_drawdown_curves.png`
- `figures/random_baseline_distributions.png`
