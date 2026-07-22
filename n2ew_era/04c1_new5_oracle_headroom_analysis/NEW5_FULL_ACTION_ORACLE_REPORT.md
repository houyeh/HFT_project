# NEW5 Full Action Oracle Report

Oracle action is `argmax(excess_bid, excess_ask, 0)` for every NEW5 state. It uses future H20 labels and is non-causal.

| split              | policy_id          |   horizon |   state_count |   selected_action_count |   selection_rate |   bid_count |   ask_count |   no_quote_count |   total_excess_pnl_H20 |   avg_selected_excess |   win_rate |   loss_rate |   max_drawdown | diagnostic_only   | oracle_uses_future_labels   |
|:-------------------|:-------------------|----------:|--------------:|------------------------:|-----------------:|------------:|------------:|-----------------:|-----------------------:|----------------------:|-----------:|------------:|---------------:|:------------------|:----------------------------|
| new5_final_holdout | oracle_full_action |        20 |        815912 |                  577627 |         0.707953 |      291947 |      285680 |           238285 |             8.4518e+07 |               146.319 |          1 |           0 |              0 | True              | True                        |

This oracle is diagnostic only and cannot be used to tune policy.
