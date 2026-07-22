# FPGA-Friendly Policy Search Report

## 1. Objective

The objective is to convert existing risk signals (`015_common` and `016_v2_binary`) into a fixed, hardware-synthesizable market-making cancel policy. This phase does not add model features, does not tune on test, and does not make a real-world profitability claim.

## 2. Robustness Checks

- Shuffled-label val PR-AUC mean: `0.007782`.
- Multi-seed val PR-AUC range: `0.007907` to `0.009669`.
- Best validation simple-rule baseline: `simple_spread_filter_only_s2` with PnL `33,990`.
- Score bucket economic value audit is available in `robustness/score_bucket_economic_value.csv`.

Validation score buckets show economic meaning: top 1% bid-risk quotes have expected not-cancelled quote PnL `-15.486` versus base `-1.226`; top 1% ask-risk quotes have expected not-cancelled quote PnL `-16.301` versus base `-1.269`.

## 3. Validation-Only Policy Search

Candidate policies were searched on validation only. Policy B depth proxy rows are diagnostic only and were excluded from the main claim.

- selection_split: `validation`
- test_used_for_selection: `False`
- selected policy: `C_ensemble_weighted_w0.75_s3_r0.05`
- policy family: `C_015_016_ensemble`
- validation PnL: `38,440`
- validation trades: `144`
- validation max drawdown: `92,570`
- validation any-cancel rate: `0.1109`

## 4. Frozen Test Evaluation

The selected policy was applied to test using frozen validation-selected parameters.

| Stress | Net PnL | Gap vs No-Cancel | Gap vs random_v3_mean | Trades | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| Base | 77,195 | 13,410 | 52,148.914 | 567 | 58,150 |
| Mild | -5,440 | 4,345 | 10,559.863 | 441 | 66,330 |
| Moderate | -74,805 | 22,465 | 10,764.435 | 297 | 76,345 |

The selected policy is base-positive and remains defensive under mild/moderate stress, but it does not remain absolutely profitable under stress. The stronger test outcome of a backup policy is retained only as a diagnostic appendix result and is not used to reselect the main policy.

## 5. HLS Mapping

The policy maps to threshold comparators, spread tick comparisons, and optional small threshold tables. See `hls_policy_spec.md` for the exact mapping.

## 6. Diagnostic Appendix

Backup policies and diagnostic Policy B rows are retained in `search/policy_search_grid.csv`, `search/policy_selection_ranked.csv`, and `evaluation/frozen_policy_test_results.csv`. They are not the main positive-PnL claim.

## 7. Conservative Interpretation

This phase demonstrates an FPGA-friendly policy that retains defensive value under the current proxy execution framework. It is not proof of real-world HFT profitability, not proof of exchange fill toxicity, and not a substitute for queue-position-aware validation.
