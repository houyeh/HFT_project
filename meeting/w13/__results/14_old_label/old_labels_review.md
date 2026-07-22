# Old Label Definitions and Training Results Review

This document extracts the exact definitions of the old binary labels from `src/features/label_generation.py` and compares their training and backtest performance.

## 1. Label Definitions

```python
# 1. toxic_bid_binary
# Definition: If the minimum future mid price (over the horizon) drops below the current bid by a threshold, it's toxic.
future_mid_min = future_mid.min(axis=1)
toxic = (future_mid_min < (current_bid - threshold_price)).astype(np.int64)

# 2. toxic_ask_binary
# Definition: If the maximum future mid price rises above the current ask by a threshold, it's toxic.
future_mid_max = future_mid.max(axis=1)
toxic = (future_mid_max > (current_ask + threshold_price)).astype(np.int64)

# 3. adverse_bid_fill_binary
# Definition: Same as toxic_bid, but ALSO requires an immediate fill proxy (next_ask <= current_bid).
bid_fill = next_ask <= current_bid
future_mid_min = future_mid.min(axis=1)
adverse = bid_fill & (future_mid_min < (current_bid - threshold_price))

# 4. adverse_ask_fill_binary
# Definition: Same as toxic_ask, but ALSO requires an immediate fill proxy (next_bid >= current_ask).
ask_fill = next_bid >= current_ask
future_mid_max = future_mid.max(axis=1)
adverse = ask_fill & (future_mid_max > (current_ask + threshold_price))

# 5. active_buy_profitable_binary
# Definition: If actively buying (crossing the spread to buy at ask) yields a profit > total cost (taker fee + margin).
future_mid_max = future_mid.max(axis=1)
profit_ticks = (future_mid_max - current_ask) / tick_size
out = (profit_ticks > total_cost_ticks).astype(np.int64)

# 6. active_sell_profitable_binary
# Definition: If actively selling (crossing the spread to sell at bid) yields a profit > total cost.
future_mid_min = future_mid.min(axis=1)
profit_ticks = (current_bid - future_mid_min) / tick_size
out = (profit_ticks > total_cost_ticks).astype(np.int64)
```

## 2. Performance Summary

Here is the aggregated performance of these old labels across different feature counts (fc=6,7,8,9):

### toxic_bid_binary
- **Experiments**: 4
- **Mean Best Val Loss**: 0.6971
- **Mean Test Macro F1**: 0.4993
- **Mean PnL Improvement**: 0.00 TWD

### toxic_ask_binary
- **Experiments**: 4
- **Mean Best Val Loss**: 0.6903
- **Mean Test Macro F1**: 0.5020
- **Mean PnL Improvement**: -685.00 TWD

### adverse_bid_fill_binary
- **Experiments**: 4
- **Mean Best Val Loss**: 0.2244
- **Mean Test Macro F1**: 0.5000
- **Mean PnL Improvement**: 0.00 TWD

### adverse_ask_fill_binary
- **Experiments**: 4
- **Mean Best Val Loss**: 0.1234
- **Mean Test Macro F1**: 0.5000
- **Mean PnL Improvement**: 0.00 TWD

### active_buy_profitable_binary
- **Experiments**: 4
- **Mean Best Val Loss**: 0.5904
- **Mean Test Macro F1**: 0.5536
- **Mean PnL Improvement**: -11,796.25 TWD

### active_sell_profitable_binary
- **Experiments**: 4
- **Mean Best Val Loss**: 0.5594
- **Mean Test Macro F1**: 0.5469
- **Mean PnL Improvement**: -10,381.25 TWD

## 3. Why They Failed (Diagnostic Verdict)

- **`adverse_bid/ask_fill_binary`**: The `best_val_loss` is extremely low (e.g., 0.12 to 0.22). This is a strong indicator of severe class imbalance. The `bid_fill` and `ask_fill` conditions (`next_ask <= current_bid` or `next_bid >= current_ask`) almost never happen in reality because prices rarely cross like that instantly without wiping out the level. Thus, the positive class is practically non-existent, the model learns to predict 0 constantly, giving F1 ≈ 0.5 and PnL Improvement = 0.
- **`toxic_bid/ask_binary`**: Does not check for fill, meaning it simply measures if the mid price moves beyond the bid/ask within the horizon. While there are more positive samples here, the test Macro F1 is near 0.5 (random guessing). The model failed to find a stable edge, resulting in zero or slightly negative PnL improvement. F1 score being stuck around 0.5 shows the model is just predicting the majority class because it cannot distinguish toxic from non-toxic using order book alone.
- **`active_buy/sell_profitable_binary`**: Designed for aggressive Taker trading. Requires the future move to overcome the spread AND the taker fee (which is 50 TWD per point). These labels were too strict. The model attempts to predict them but mostly triggers false positives, resulting in negative PnL improvements (loss of -10K to -12K on average).

In summary, the unconditional `mid_move` and `smoothed_mid_move` labels ended up being the most balanced target for the model to capture short-term microstructure alpha. That is why `smoothed_mid_move_3class` was selected as the final candidate for the cancel filter setup.
