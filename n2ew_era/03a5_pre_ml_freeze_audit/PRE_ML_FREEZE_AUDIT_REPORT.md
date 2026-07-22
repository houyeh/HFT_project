# Pre-ML Freeze Audit Report

Final freeze status: `freeze_pass_with_minor_caveats`.

No ML was trained. No strategy or threshold was tuned. This phase freezes labels, features, splits, baselines, and evaluation rules.

## Hard Gates

| gate                         | pass   |
|:-----------------------------|:-------|
| excess_label_formula         | True   |
| feature_causality            | True   |
| split_duplicate_independence | True   |
| feature_schema_freeze        | True   |
| baseline_hurdle_freeze       | True   |
| evaluation_contract_freeze   | True   |

## Gate 1: Excess Label Formula

|   horizon |   max_abs_bid_reconstruction_error |   max_abs_ask_reconstruction_error | formula_pass   |
|----------:|-----------------------------------:|-----------------------------------:|:---------------|
|         1 |                                  0 |                                  0 | True           |
|         5 |                                  0 |                                  0 | True           |
|        10 |                                  0 |                                  0 | True           |
|        20 |                                  0 |                                  0 | True           |
|        50 |                                  0 |                                  0 | True           |
|       100 |                                  0 |                                  0 | True           |

## Gate 2: Feature Causality

Allowed feature count: `14`. Future-leaking allowed feature violations: `0`.

## Gate 3: Split / Duplicate Independence

| split_a   | split_b   |   session_id_overlap_count |   timestamp_sequence_hash_overlap_count |   price_sequence_hash_overlap_count | session_id_overlap_values   | independence_pass   | classification   |
|:----------|:----------|---------------------------:|----------------------------------------:|------------------------------------:|:----------------------------|:--------------------|:-----------------|
| train     | val       |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| train     | test      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| train     | new4      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| train     | new3      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| val       | test      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| val       | new4      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| val       | new3      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| test      | new4      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| test      | new3      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| new4      | new3      |                          0 |                                       0 |                                   0 |                             | True                | independent      |

## Gate 4: ML Feature Schema

Schema frozen in `allowed_features.json` and `forbidden_features.json`.

Minor caveat: requested features missing from Phase 02F parquet are not allowed until rebuilt causally:

| column                | notes                                                              |
|:----------------------|:-------------------------------------------------------------------|
| event_rate_past_50    | Missing from Phase 02F parquet; do not use until rebuilt causally. |
| spread_change_past    | Missing from Phase 02F parquet; do not use until rebuilt causally. |
| imbalance_change_past | Missing from Phase 02F parquet; do not use until rebuilt causally. |

## Gate 5: Baseline Hurdles

Frozen hurdles are saved to `results/frozen_baseline_hurdles.csv` and described in `BASELINE_HURDLE_CONTRACT.md`.

## Gate 6: Evaluation Contract

`ML_EVALUATION_CONTRACT.md` freezes primary targets, validation-only selection, and frozen test/NEW3/NEW4 evaluation.

## Final Decision

`freeze_pass_with_minor_caveats`.
