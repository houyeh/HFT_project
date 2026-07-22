status: resolved
finding: mixed
# Audit: Causal-Fill Sequential Oracle Derivation

## 10.1 Is it globally optimal?

The action-level full oracle is pointwise optimal for the dense label objective: source lines `263-282` select `argmax(excess_bid, excess_ask, 0)` per state. The matched-count oracle sorts best state values (`top_best_state_oracle`, lines `284-295`). The matched-count+side oracle uses side-specific greedy sorting (`greedy_count_side_oracle`, lines `297-327`).

The causal-fill sequential oracle is more limited: source lines `393-433` first create oracle decisions from future labels, then applies the same `conservative_l2_stress` fill probabilities, stable hash fills, H20 scheduled close, and inventory cap 1 simulator. It is therefore a policy-constrained oracle passed through the proxy simulator, not a dynamic-programming proof of global optimum under stochastic fills and inventory constraints.

## 10.2 Constraint sensitivity and observed values

| oracle_id                                                                   | split              | fill_model             |   inventory_cap |   horizon |   sequential_excess_pnl |   fill_count |   forced_no_quote_count |   max_abs_inventory |   max_drawdown |   win_rate |   loss_rate | oracle_uses_future_labels   | diagnostic_only   | true_queue_replay   |
|:----------------------------------------------------------------------------|:-------------------|:-----------------------|----------------:|----------:|------------------------:|-------------:|------------------------:|--------------------:|---------------:|-----------:|------------:|:----------------------------|:------------------|:--------------------|
| oracle_full_action                                                          | new5_final_holdout | conservative_l2_stress |               1 |        20 |             3.92313e+06 |        29418 |                   50100 |                   1 |              0 |   1        |   0         | True                        | True              | False               |
| oracle_matched_count_for_side_by_queue_imbalance_abs_0p1                    | new5_final_holdout | conservative_l2_stress |               1 |        20 |             3.83256e+06 |        34630 |                   57964 |                   1 |             90 |   0.839705 |   0.160266  | True                        | True              | False               |
| oracle_matched_count_side_for_side_by_queue_imbalance_abs_0p1               | new5_final_holdout | conservative_l2_stress |               1 |        20 |             3.82995e+06 |        34750 |                   57792 |                   1 |            105 |   0.835281 |   0.164719  | True                        | True              | False               |
| oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected      | new5_final_holdout | conservative_l2_stress |               1 |        20 |             3.87439e+06 |        31696 |                   53642 |                   1 |            105 |   0.923871 |   0.0760664 | True                        | True              | False               |
| oracle_matched_count_side_for_ml_hgb_regressor_shallow_convergence_selected | new5_final_holdout | conservative_l2_stress |               1 |        20 |             3.86321e+06 |        31668 |                   53618 |                   1 |            105 |   0.922572 |   0.0773652 | True                        | True              | False               |

Dense action-level oracle context:
| split              | policy_id          |   horizon |   state_count |   selected_action_count |   selection_rate |   bid_count |   ask_count |   no_quote_count |   total_excess_pnl_H20 |   avg_selected_excess |   win_rate |   loss_rate |   max_drawdown | diagnostic_only   | oracle_uses_future_labels   |
|:-------------------|:-------------------|----------:|--------------:|------------------------:|-----------------:|------------:|------------:|-----------------:|-----------------------:|----------------------:|-----------:|------------:|---------------:|:------------------|:----------------------------|
| new5_final_holdout | oracle_full_action |        20 |        815912 |                  577627 |         0.707953 |      291947 |      285680 |           238285 |             8.4518e+07 |               146.319 |          1 |           0 |              0 | True              | True                        |

Matched-count oracle context:
| policy_id                                     |   policy_selected_count |   policy_pnl |   matched_count_oracle_selected_count |   matched_count_oracle_pnl |   positive_only_oracle_selected_count |   positive_only_oracle_pnl |   forced_nonpositive_actions_for_exact_count |   capture_ratio |   opportunity_gap | oracle_uses_future_labels   | diagnostic_only   |
|:----------------------------------------------|------------------------:|-------------:|--------------------------------------:|---------------------------:|--------------------------------------:|---------------------------:|---------------------------------------------:|----------------:|------------------:|:----------------------------|:------------------|
| side_by_queue_imbalance_abs_0p1               |                  672049 |  1.14398e+07 |                                672049 |                8.3102e+07  |                                577627 |                 8.4518e+07 |                                        94422 |        0.13766  |       7.16622e+07 | True                        | True              |
| ml_hgb_regressor_shallow_convergence_selected |                  617738 |  1.56481e+07 |                                617738 |                8.39166e+07 |                                577627 |                 8.4518e+07 |                                        40111 |        0.186472 |       6.82686e+07 | True                        | True              |

Matched-count+side oracle context:
| policy_id                                     |   policy_selected_count |   policy_bid_count |   policy_ask_count |   policy_pnl |   oracle_selected_count |   oracle_bid_count |   oracle_ask_count |   matched_count_side_oracle_pnl |   positive_only_oracle_selected_count |   positive_only_oracle_bid_count |   positive_only_oracle_ask_count |   positive_only_oracle_pnl |   forced_nonpositive_actions_for_exact_count |   capture_ratio |   opportunity_gap | oracle_uses_future_labels   | diagnostic_only   |
|:----------------------------------------------|------------------------:|-------------------:|-------------------:|-------------:|------------------------:|-------------------:|-------------------:|--------------------------------:|--------------------------------------:|---------------------------------:|---------------------------------:|---------------------------:|---------------------------------------------:|----------------:|------------------:|:----------------------------|:------------------|
| side_by_queue_imbalance_abs_0p1               |                  672049 |             325153 |             346896 |  1.14398e+07 |                  672049 |             325153 |             346896 |                     8.3102e+07  |                                577627 |                           291947 |                           285680 |                 8.4518e+07 |                                        94422 |        0.13766  |       7.16622e+07 | True                        | True              |
| ml_hgb_regressor_shallow_convergence_selected |                  617738 |             318294 |             299444 |  1.56481e+07 |                  617738 |             318294 |             299444 |                     8.39166e+07 |                                577627 |                           291947 |                           285680 |                 8.4518e+07 |                                        40111 |        0.186472 |       6.82686e+07 | True                        | True              |

The existing artifact fixes `fill_model=conservative_l2_stress`, `inventory_cap=1`, and `horizon=20`. It does not sweep inventory caps 2/3 or alternate fill models in 04C.1; sensitivity outside those settings remains an explicit limitation.

## 10.3 Corrected wording

Replace the stronger wording in `B_CLASS_RETROSPECTIVE_ANALYSIS.md` §D.2 with:

> The causal-fill sequential oracle is a diagnostic, future-label upper-bound under the same proxy execution contract (`conservative_l2_stress`, H20 exit, inventory cap 1). It is not a tradable strategy and not a formal global optimum over all possible inventory-aware policies. Its purpose is to measure headroom under fixed proxy assumptions and to bound how much of the observable future-label opportunity causal models captured.

Forbidden claims remain: true exchange replay, live profitability, or a proven executable oracle.
