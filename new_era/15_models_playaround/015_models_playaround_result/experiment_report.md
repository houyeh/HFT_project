# Experiment Report (015_models_playaround)

- Experiments: 33

## Class Distributions (by label_type)

### active_buy_profitable_binary

```
 feature_count class_distribution_train  class_distribution_val  class_distribution_test
             6 {"0": 331318, "1": 5767} {"0": 80490, "1": 1628} {"0": 148576, "1": 2572}
             7 {"0": 331318, "1": 5767} {"0": 80490, "1": 1628} {"0": 148576, "1": 2572}
             8 {"0": 331318, "1": 5767} {"0": 80490, "1": 1628} {"0": 148576, "1": 2572}
             9 {"0": 331318, "1": 5767} {"0": 80490, "1": 1628} {"0": 148576, "1": 2572}
```

### active_sell_profitable_binary

```
 feature_count class_distribution_train  class_distribution_val  class_distribution_test
             6 {"0": 331031, "1": 6054} {"0": 80521, "1": 1597} {"0": 148598, "1": 2550}
             7 {"0": 331031, "1": 6054} {"0": 80521, "1": 1597} {"0": 148598, "1": 2550}
             8 {"0": 331031, "1": 6054} {"0": 80521, "1": 1597} {"0": 148598, "1": 2550}
             9 {"0": 331031, "1": 6054} {"0": 80521, "1": 1597} {"0": 148598, "1": 2550}
```

### adverse_ask_fill_binary

```
 feature_count class_distribution_train class_distribution_val class_distribution_test
             6    {"0": 337076, "1": 9}   {"0": 82116, "1": 2}   {"0": 151146, "1": 2}
             7    {"0": 337076, "1": 9}   {"0": 82116, "1": 2}   {"0": 151146, "1": 2}
             8    {"0": 337076, "1": 9}   {"0": 82116, "1": 2}   {"0": 151146, "1": 2}
             9    {"0": 337076, "1": 9}   {"0": 82116, "1": 2}   {"0": 151146, "1": 2}
```

### adverse_bid_fill_binary

```
 feature_count class_distribution_train class_distribution_val class_distribution_test
             6   {"0": 337072, "1": 13}   {"0": 82115, "1": 3}   {"0": 151145, "1": 3}
             7   {"0": 337072, "1": 13}   {"0": 82115, "1": 3}   {"0": 151145, "1": 3}
             8   {"0": 337072, "1": 13}   {"0": 82115, "1": 3}   {"0": 151145, "1": 3}
             9   {"0": 337072, "1": 13}   {"0": 82115, "1": 3}   {"0": 151145, "1": 3}
```

### mid_move_3class

```
 feature_count            class_distribution_train           class_distribution_val           class_distribution_test
             6 {"0": 2518, "1": 332139, "2": 2428} {"0": 802, "1": 80446, "2": 870} {"0": 748, "1": 149611, "2": 789}
             7 {"0": 2518, "1": 332139, "2": 2428} {"0": 802, "1": 80446, "2": 870} {"0": 748, "1": 149611, "2": 789}
             8 {"0": 2518, "1": 332139, "2": 2428} {"0": 802, "1": 80446, "2": 870} {"0": 748, "1": 149611, "2": 789}
             9 {"0": 2518, "1": 332139, "2": 2428} {"0": 802, "1": 80446, "2": 870} {"0": 748, "1": 149611, "2": 789}
```

### smoothed_mid_move_3class

```
 feature_count              class_distribution_train              class_distribution_val             class_distribution_test
             6     {"0": 934, "1": 335273, "2": 878}    {"0": 325, "1": 81444, "2": 349}   {"0": 285, "1": 150567, "2": 296}
             7     {"0": 934, "1": 335273, "2": 878}    {"0": 325, "1": 81444, "2": 349}   {"0": 285, "1": 150567, "2": 296}
             8     {"0": 934, "1": 335273, "2": 878}    {"0": 325, "1": 81444, "2": 349}   {"0": 285, "1": 150567, "2": 296}
             8 {"0": 10702, "1": 697212, "2": 10648} {"0": 1370, "1": 109950, "2": 1378} {"0": 2762, "1": 227711, "2": 2795}
             9     {"0": 934, "1": 335273, "2": 878}    {"0": 325, "1": 81444, "2": 349}   {"0": 285, "1": 150567, "2": 296}
```

### toxic_ask_binary

```
 feature_count class_distribution_train class_distribution_val class_distribution_test
             6  {"0": 336895, "1": 190}  {"0": 82019, "1": 99}  {"0": 151076, "1": 72}
             7  {"0": 336895, "1": 190}  {"0": 82019, "1": 99}  {"0": 151076, "1": 72}
             8  {"0": 336895, "1": 190}  {"0": 82019, "1": 99}  {"0": 151076, "1": 72}
             9  {"0": 336895, "1": 190}  {"0": 82019, "1": 99}  {"0": 151076, "1": 72}
```

### toxic_bid_binary

```
 feature_count class_distribution_train class_distribution_val class_distribution_test
             6  {"0": 336877, "1": 208} {"0": 82014, "1": 104}  {"0": 151116, "1": 32}
             7  {"0": 336877, "1": 208} {"0": 82014, "1": 104}  {"0": 151116, "1": 32}
             8  {"0": 336877, "1": 208} {"0": 82014, "1": 104}  {"0": 151116, "1": 32}
             9  {"0": 336877, "1": 208} {"0": 82014, "1": 104}  {"0": 151116, "1": 32}
```

## Feature Count Comparison (Batch Summary)

```
 feature_count  avg_test_macro_f1  avg_pnl_improvement  median_pnl_improvement  experiments
             6           0.460226        -10105.000000                 -2757.5            8
             7           0.465557           378.125000                     0.0            8
             8           0.438649         17782.222222                     0.0            9
             9           0.466123         -2489.375000                     0.0            8
```

## Best Candidates (Heuristic)

- Best PnL improvement: `lt=smoothed_mid_move_3class__fc=8__w=100__H=10__thr=1` (label=smoothed_mid_move_3class, feat=8, +79635.00 TWD)
- Best PnL improvement (forced flatten): `lt=smoothed_mid_move_3class__fc=8__w=100__H=10__thr=1` (label=smoothed_mid_move_3class, feat=8, +79385.00 TWD)
