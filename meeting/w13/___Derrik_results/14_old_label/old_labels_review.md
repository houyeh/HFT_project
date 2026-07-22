# 舊標籤定義與訓練結果回顧 (Old Labels Review)

這份文件從 `src/features/label_generation.py` 中擷取了舊版二元標籤 (Binary labels) 的確切定義，並比較了它們的訓練與回測表現。

## 1. 標籤定義 (Label Definitions)

```python
# 1. toxic_bid_binary
# 定義：如果在預測區間內，未來的最低 Mid Price 跌破了當前的 Bid 減去設定的 threshold_price，即標記為 toxic。
future_mid_min = future_mid.min(axis=1)
toxic = (future_mid_min < (current_bid - threshold_price)).astype(np.int64)

# 2. toxic_ask_binary
# 定義：如果在預測區間內，未來的最高 Mid Price 突破了當前的 Ask 加上設定的 threshold_price，即標記為 toxic。
future_mid_max = future_mid.max(axis=1)
toxic = (future_mid_max > (current_ask + threshold_price)).astype(np.int64)

# 3. adverse_bid_fill_binary
# 定義：與 toxic_bid 相同，但額外要求必須有立即成交的替代指標 (next_ask <= current_bid)。
bid_fill = next_ask <= current_bid
future_mid_min = future_mid.min(axis=1)
adverse = bid_fill & (future_mid_min < (current_bid - threshold_price))

# 4. adverse_ask_fill_binary
# 定義：與 toxic_ask 相同，但額外要求必須有立即成交的替代指標 (next_bid >= current_ask)。
ask_fill = next_bid >= current_ask
future_mid_max = future_mid.max(axis=1)
adverse = ask_fill & (future_mid_max > (current_ask + threshold_price))

# 5. active_buy_profitable_binary
# 定義：如果主動買入 (跨越價差吃 Ask) 能產生大於總成本 (Taker fee + margin) 的利潤。
future_mid_max = future_mid.max(axis=1)
profit_ticks = (future_mid_max - current_ask) / tick_size
out = (profit_ticks > total_cost_ticks).astype(np.int64)

# 6. active_sell_profitable_binary
# 定義：如果主動賣出 (跨越價差吃 Bid) 能產生大於總成本的利潤。
future_mid_min = future_mid.min(axis=1)
profit_ticks = (current_bid - future_mid_min) / tick_size
out = (profit_ticks > total_cost_ticks).astype(np.int64)
```

## 2. 表現總結 (Performance Summary)

以下是這些舊標籤在不同特徵數量 (fc=6,7,8,9) 下的平均表現：

### toxic_bid_binary
- **實驗次數**: 4
- **平均最佳 Val Loss**: 0.6971
- **平均測試集 Macro F1**: 0.4993
- **平均 PnL 提升**: 0.00 TWD

### toxic_ask_binary
- **實驗次數**: 4
- **平均最佳 Val Loss**: 0.6903
- **平均測試集 Macro F1**: 0.5020
- **平均 PnL 提升**: -685.00 TWD

### adverse_bid_fill_binary
- **實驗次數**: 4
- **平均最佳 Val Loss**: 0.2244
- **平均測試集 Macro F1**: 0.5000
- **平均 PnL 提升**: 0.00 TWD

### adverse_ask_fill_binary
- **實驗次數**: 4
- **平均最佳 Val Loss**: 0.1234
- **平均測試集 Macro F1**: 0.5000
- **平均 PnL 提升**: 0.00 TWD

### active_buy_profitable_binary
- **實驗次數**: 4
- **平均最佳 Val Loss**: 0.5904
- **平均測試集 Macro F1**: 0.5536
- **平均 PnL 提升**: -11,796.25 TWD

### active_sell_profitable_binary
- **實驗次數**: 4
- **平均最佳 Val Loss**: 0.5594
- **平均測試集 Macro F1**: 0.5469
- **平均 PnL 提升**: -10,381.25 TWD

## 3. 失敗原因分析 (Diagnostic Verdict)

- **`adverse_bid/ask_fill_binary`**: `best_val_loss` 極度偏低 (約 0.12 到 0.22)。這是嚴重類別不平衡 (Class Imbalance) 的強烈指標。`bid_fill` 和 `ask_fill` 條件 (`next_ask <= current_bid` 或 `next_bid >= current_ask`) 在現實中極少發生，因為價格很少會在不刷破檔位的情況下瞬間交叉。因此，正類別 (Positive class) 幾乎不存在，模型學會了不斷預測 0，導致 F1 ≈ 0.5 且 PnL 提升為 0。
- **`toxic_bid/ask_binary`**: 沒有檢查是否成交，只衡量預測區間內的 Mid Price 是否超出 Bid/Ask。雖然這裡有更多的正樣本，但測試集 Macro F1 接近 0.5 (隨機亂猜)。模型無法找到穩定的優勢，導致 PnL 提升為零或微幅負數。F1 卡在 0.5 顯示模型只是在預測多數類別，因為它無法單憑 Order book 區分出毒性與非毒性。
- **`active_buy/sell_profitable_binary`**: 專為積極的 Taker 交易設計。要求未來的價格移動必須克服價差 AND Taker 手續費 (每點 50 TWD)。這些標籤的要求太過嚴苛。模型試圖預測它們，但大多數觸發的都是假陽性 (False positives)，導致嚴重的負 PnL (平均虧損 1萬 到 1.2萬 TWD)。

總結來說，無條件的 `mid_move` 與 `smoothed_mid_move` 標籤最終被證明是讓模型捕捉短期微觀 Alpha 最平衡的目標。這也就是為什麼 `smoothed_mid_move_3class` 會被選為目前撤單過濾器 (Cancel filter) 的最終候選基準。
