# Backtest Overview (This Run)

- Experiments: 32

## Top 10 (by pnl_improvement_flatten)

| experiment_id                                                                      | label_type                    |   feature_count | val_selected_strategy   | val_selected_threshold   |   baseline_test_pnl_flatten |   model_test_pnl_flatten |   pnl_improvement_flatten |   flatten_delta |   max_drawdown |
|:-----------------------------------------------------------------------------------|:------------------------------|----------------:|:------------------------|:-------------------------|----------------------------:|-------------------------:|--------------------------:|----------------:|---------------:|
| lt=smoothed_mid_move_3class__fc=8__w=100__H=10__thr=1                              | smoothed_mid_move_3class      |               8 | tcn_skew                | pthr=0.55,ethr=0.4       |                       90490 |                   169875 |                     79385 |            -250 |          25420 |
| lt=mid_move_3class__fc=8__w=100__H=10__thr=1                                       | mid_move_3class               |               8 | tcn_skew                | pthr=0.5,ethr=0.05       |                       90490 |                   144925 |                     54435 |            -250 |          31550 |
| lt=mid_move_3class__fc=9__w=100__H=10__thr=1                                       | mid_move_3class               |               9 | tcn_skew                | pthr=0.55,ethr=0.05      |                       90490 |                   120925 |                     30435 |            -250 |          25385 |
| lt=active_buy_profitable_binary__fc=7__w=100__H=10__thr=1__tfee=50__exit=0__mar=0  | active_buy_profitable_binary  |               7 | tcn_gated               | pthr=0.65,ethr=0.0       |                       90490 |                   102520 |                     12030 |               0 |          37960 |
| lt=active_sell_profitable_binary__fc=8__w=100__H=10__thr=1__tfee=50__exit=0__mar=0 | active_sell_profitable_binary |               8 | tcn_gated               | pthr=0.55,ethr=0.0       |                       90490 |                    94285 |                      3795 |             250 |          29220 |
| lt=mid_move_3class__fc=7__w=100__H=10__thr=1                                       | mid_move_3class               |               7 | tcn_skew                | pthr=0.5,ethr=0.05       |                       90490 |                    92610 |                      2120 |             500 |          45440 |
| lt=adverse_ask_fill_binary__fc=6__w=100__H=10__thr=1__adv=1                        | adverse_ask_fill_binary       |               6 | tcn_gated               | pthr=0.5,ethr=0.0        |                       90490 |                    90490 |                         0 |               0 |          31420 |
| lt=adverse_ask_fill_binary__fc=7__w=100__H=10__thr=1__adv=1                        | adverse_ask_fill_binary       |               7 | tcn_gated               | pthr=0.5,ethr=0.0        |                       90490 |                    90490 |                         0 |               0 |          31420 |
| lt=adverse_ask_fill_binary__fc=8__w=100__H=10__thr=1__adv=1                        | adverse_ask_fill_binary       |               8 | tcn_gated               | pthr=0.5,ethr=0.0        |                       90490 |                    90490 |                         0 |               0 |          31420 |
| lt=adverse_ask_fill_binary__fc=9__w=100__H=10__thr=1__adv=1                        | adverse_ask_fill_binary       |               9 | tcn_gated               | pthr=0.5,ethr=0.0        |                       90490 |                    90490 |                         0 |               0 |          31420 |

## Best Per Label (by pnl_improvement_flatten)

| label_type                    |   feature_count | experiment_id                                                                      |   pnl_improvement_flatten | val_selected_strategy   | val_selected_threshold   |   max_drawdown |
|:------------------------------|----------------:|:-----------------------------------------------------------------------------------|--------------------------:|:------------------------|:-------------------------|---------------:|
| smoothed_mid_move_3class      |               8 | lt=smoothed_mid_move_3class__fc=8__w=100__H=10__thr=1                              |                     79385 | tcn_skew                | pthr=0.55,ethr=0.4       |          25420 |
| mid_move_3class               |               8 | lt=mid_move_3class__fc=8__w=100__H=10__thr=1                                       |                     54435 | tcn_skew                | pthr=0.5,ethr=0.05       |          31550 |
| active_buy_profitable_binary  |               7 | lt=active_buy_profitable_binary__fc=7__w=100__H=10__thr=1__tfee=50__exit=0__mar=0  |                     12030 | tcn_gated               | pthr=0.65,ethr=0.0       |          37960 |
| active_sell_profitable_binary |               8 | lt=active_sell_profitable_binary__fc=8__w=100__H=10__thr=1__tfee=50__exit=0__mar=0 |                      3795 | tcn_gated               | pthr=0.55,ethr=0.0       |          29220 |
| adverse_ask_fill_binary       |               6 | lt=adverse_ask_fill_binary__fc=6__w=100__H=10__thr=1__adv=1                        |                         0 | tcn_gated               | pthr=0.5,ethr=0.0        |          31420 |
| adverse_bid_fill_binary       |               6 | lt=adverse_bid_fill_binary__fc=6__w=100__H=10__thr=1__adv=1                        |                         0 | tcn_gated               | pthr=0.5,ethr=0.0        |          31420 |
| toxic_ask_binary              |               7 | lt=toxic_ask_binary__fc=7__w=100__H=10__thr=1__adv=1                               |                         0 | tcn_gated               | pthr=0.5,ethr=0.0        |          31420 |
| toxic_bid_binary              |               6 | lt=toxic_bid_binary__fc=6__w=100__H=10__thr=1__adv=1                               |                         0 | tcn_gated               | pthr=0.7,ethr=0.0        |          31420 |

## Pivot: label_type ? feature_count (pnl_improvement_flatten)

| label_type                    |      6 |     7 |      8 |      9 |
|:------------------------------|-------:|------:|-------:|-------:|
| active_buy_profitable_binary  |  -3595 | 12030 | -28020 | -27850 |
| active_sell_profitable_binary | -22240 | -4170 |   3795 | -18160 |
| adverse_ask_fill_binary       |      0 |     0 |      0 |      0 |
| adverse_bid_fill_binary       |      0 |     0 |      0 |      0 |
| mid_move_3class               | -22875 |  2120 |  54435 |  30435 |
| smoothed_mid_move_3class      | -29460 | -6205 |  79385 |  -4590 |
| toxic_ask_binary              |  -2170 |     0 |   -570 |      0 |
| toxic_bid_binary              |      0 |     0 |      0 |      0 |

## Inventory-Drift Sensitivity

`flatten_delta = pnl_improvement_flatten - pnl_improvement` (non-zero means ending inventory affected the un-flattened comparison).

| experiment_id                                                                      | label_type                    |   feature_count |   pnl_improvement |   pnl_improvement_flatten |   flatten_delta |
|:-----------------------------------------------------------------------------------|:------------------------------|----------------:|------------------:|--------------------------:|----------------:|
| lt=mid_move_3class__fc=7__w=100__H=10__thr=1                                       | mid_move_3class               |               7 |              1620 |                      2120 |             500 |
| lt=active_sell_profitable_binary__fc=6__w=100__H=10__thr=1__tfee=50__exit=0__mar=0 | active_sell_profitable_binary |               6 |            -22740 |                    -22240 |             500 |
| lt=active_buy_profitable_binary__fc=6__w=100__H=10__thr=1__tfee=50__exit=0__mar=0  | active_buy_profitable_binary  |               6 |             -3345 |                     -3595 |            -250 |
| lt=smoothed_mid_move_3class__fc=8__w=100__H=10__thr=1                              | smoothed_mid_move_3class      |               8 |             79635 |                     79385 |            -250 |
| lt=smoothed_mid_move_3class__fc=7__w=100__H=10__thr=1                              | smoothed_mid_move_3class      |               7 |             -6455 |                     -6205 |             250 |
| lt=mid_move_3class__fc=9__w=100__H=10__thr=1                                       | mid_move_3class               |               9 |             30685 |                     30435 |            -250 |
| lt=mid_move_3class__fc=8__w=100__H=10__thr=1                                       | mid_move_3class               |               8 |             54685 |                     54435 |            -250 |
| lt=mid_move_3class__fc=6__w=100__H=10__thr=1                                       | mid_move_3class               |               6 |            -23125 |                    -22875 |             250 |
| lt=active_sell_profitable_binary__fc=8__w=100__H=10__thr=1__tfee=50__exit=0__mar=0 | active_sell_profitable_binary |               8 |              3545 |                      3795 |             250 |
| lt=adverse_ask_fill_binary__fc=6__w=100__H=10__thr=1__adv=1                        | adverse_ask_fill_binary       |               6 |                 0 |                         0 |               0 |
