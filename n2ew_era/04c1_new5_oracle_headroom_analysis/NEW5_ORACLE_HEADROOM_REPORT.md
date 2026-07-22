# NEW5 Oracle Headroom Report

Oracle uses future labels and is non-causal. Oracle is diagnostic only. Oracle cannot be used to tune policy. Oracle does not imply real-world profitability.

## Unconstrained Oracle Upper Bound

| split              | policy_id          |   horizon |   state_count |   selected_action_count |   selection_rate |   bid_count |   ask_count |   no_quote_count |   total_excess_pnl_H20 |   avg_selected_excess |   win_rate |   loss_rate |   max_drawdown | diagnostic_only   | oracle_uses_future_labels   |
|:-------------------|:-------------------|----------:|--------------:|------------------------:|-----------------:|------------:|------------:|-----------------:|-----------------------:|----------------------:|-----------:|------------:|---------------:|:------------------|:----------------------------|
| new5_final_holdout | oracle_full_action |        20 |        815912 |                  577627 |         0.707953 |      291947 |      285680 |           238285 |             8.4518e+07 |               146.319 |          1 |           0 |              0 | True              | True                        |

## Matched-Count Oracle

| policy_id                                     |   policy_selected_count |   policy_pnl |   matched_count_oracle_selected_count |   matched_count_oracle_pnl |   positive_only_oracle_selected_count |   positive_only_oracle_pnl |   forced_nonpositive_actions_for_exact_count |   capture_ratio |   opportunity_gap | oracle_uses_future_labels   | diagnostic_only   |
|:----------------------------------------------|------------------------:|-------------:|--------------------------------------:|---------------------------:|--------------------------------------:|---------------------------:|---------------------------------------------:|----------------:|------------------:|:----------------------------|:------------------|
| side_by_queue_imbalance_abs_0p1               |                  672049 |  1.14398e+07 |                                672049 |                8.3102e+07  |                                577627 |                 8.4518e+07 |                                        94422 |        0.13766  |       7.16622e+07 | True                        | True              |
| ml_hgb_regressor_shallow_convergence_selected |                  617738 |  1.56481e+07 |                                617738 |                8.39166e+07 |                                577627 |                 8.4518e+07 |                                        40111 |        0.186472 |       6.82686e+07 | True                        | True              |

## Matched-Count + Side Oracle

| policy_id                                     |   policy_selected_count |   policy_bid_count |   policy_ask_count |   policy_pnl |   oracle_selected_count |   oracle_bid_count |   oracle_ask_count |   matched_count_side_oracle_pnl |   positive_only_oracle_selected_count |   positive_only_oracle_bid_count |   positive_only_oracle_ask_count |   positive_only_oracle_pnl |   forced_nonpositive_actions_for_exact_count |   capture_ratio |   opportunity_gap | oracle_uses_future_labels   | diagnostic_only   |
|:----------------------------------------------|------------------------:|-------------------:|-------------------:|-------------:|------------------------:|-------------------:|-------------------:|--------------------------------:|--------------------------------------:|---------------------------------:|---------------------------------:|---------------------------:|---------------------------------------------:|----------------:|------------------:|:----------------------------|:------------------|
| side_by_queue_imbalance_abs_0p1               |                  672049 |             325153 |             346896 |  1.14398e+07 |                  672049 |             325153 |             346896 |                     8.3102e+07  |                                577627 |                           291947 |                           285680 |                 8.4518e+07 |                                        94422 |        0.13766  |       7.16622e+07 | True                        | True              |
| ml_hgb_regressor_shallow_convergence_selected |                  617738 |             318294 |             299444 |  1.56481e+07 |                  617738 |             318294 |             299444 |                     8.39166e+07 |                                577627 |                           291947 |                           285680 |                 8.4518e+07 |                                        40111 |        0.186472 |       6.82686e+07 | True                        | True              |

## Causal-Fill Sequential Oracle

| oracle_id                                                                   | split              | fill_model             |   inventory_cap |   horizon |   sequential_excess_pnl |   fill_count |   forced_no_quote_count |   max_abs_inventory |   max_drawdown |   win_rate |   loss_rate | oracle_uses_future_labels   | diagnostic_only   | true_queue_replay   |
|:----------------------------------------------------------------------------|:-------------------|:-----------------------|----------------:|----------:|------------------------:|-------------:|------------------------:|--------------------:|---------------:|-----------:|------------:|:----------------------------|:------------------|:--------------------|
| oracle_full_action                                                          | new5_final_holdout | conservative_l2_stress |               1 |        20 |             3.92313e+06 |        29418 |                   50100 |                   1 |              0 |   1        |   0         | True                        | True              | False               |
| oracle_matched_count_for_side_by_queue_imbalance_abs_0p1                    | new5_final_holdout | conservative_l2_stress |               1 |        20 |             3.83256e+06 |        34630 |                   57964 |                   1 |             90 |   0.839705 |   0.160266  | True                        | True              | False               |
| oracle_matched_count_side_for_side_by_queue_imbalance_abs_0p1               | new5_final_holdout | conservative_l2_stress |               1 |        20 |             3.82995e+06 |        34750 |                   57792 |                   1 |            105 |   0.835281 |   0.164719  | True                        | True              | False               |
| oracle_matched_count_for_ml_hgb_regressor_shallow_convergence_selected      | new5_final_holdout | conservative_l2_stress |               1 |        20 |             3.87439e+06 |        31696 |                   53642 |                   1 |            105 |   0.923871 |   0.0760664 | True                        | True              | False               |
| oracle_matched_count_side_for_ml_hgb_regressor_shallow_convergence_selected | new5_final_holdout | conservative_l2_stress |               1 |        20 |             3.86321e+06 |        31668 |                   53618 |                   1 |            105 |   0.922572 |   0.0773652 | True                        | True              | False               |

## Required Answers

Unconstrained full-action oracle PnL: `84,517,995.00`.
HGB matched-count oracle PnL: `83,916,630.00`; HGB capture ratio: `0.1865`; gap: `68,268,560.00`.
HGB matched-count+side oracle PnL: `83,916,630.00`; HGB capture ratio: `0.1865`; gap: `68,268,560.00`.
Queue matched-count oracle PnL: `83,101,965.00`; queue capture ratio: `0.1377`.
Because the full no-quote-allowed oracle finds fewer positive states than the frozen policies selected, the CSV also reports `positive_only_oracle_*` and `forced_nonpositive_actions_for_exact_count`. The main matched-count columns enforce the exact policy count by including the least-bad remaining actions when necessary.
The gap between HGB/queue and oracle indicates remaining theoretical headroom for sequence models such as PHC-lite, but only fresh unseen data can support a formal claim.

Why oracle is not tradable: it directly selects using `excess_pnl_H20_bid` and `excess_pnl_H20_ask`, which are future outcome labels unavailable at quote decision time.
