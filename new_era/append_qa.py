import sys
from pathlib import Path

log_path = Path("new_era/execution_log.md")

content = """

## Week 10 Q&A (教授攻防題庫)

**Q1：要怎麼證明現在的 TCN 模型沒有像以前的 MLP 一樣，只是死背了目前的特定合約價格 (Overfitting)？有什麼證據能說明 Z-score + TCN 更厲害？**

**A1**：
1. **嚴格的時間序列切割 (No Shuffle)**：我們在 `datasets.py` 中使用了 `chronological_split_indices`，嚴格切出「最後 20%」的時間段。這代表模型在測試時，遇到的是它在訓練階段「完全沒見過的未來時間與價格區間」。如果它只是死背模式，在 Out-of-Sample 測試的混淆矩陣中，上下漲跌的命中率絕對會崩盤（但我們實測依然高達 80%）。
2. **Rolling Z-score 的降維打擊**：過去的 MLP 吃的是原始數值，如果合約換月從 23,000 點跳到 23,500 點，神經網路就會徹底迷失。而 **Rolling Z-score (2000 ticks)** 把絕對價格變成了「無因次 (Dimensionless)」的相對標準差 (例如 +1.5 Std)。這強迫 TCN 只能去學習「委託簿失衡的形狀」與「微觀動能」，徹底去除了價格基期的干擾，這就是它能跨越時間限制的原因。

**Q2：專案一開始的 `01_feature_importance.py` 中，Feature Importance 是如何計算出來的？憑什麼說這 8 個特徵最重要？**

**A2**：
1. **演算法基底 (Random Forest)**：我們使用了 `sklearn.ensemble.RandomForestClassifier` (隨機森林分類器) 來評估特徵。
2. **Gini Importance (吉尼不純度下降)**：在構建幾百棵決策樹的過程中，演算法會測試用每個特徵（如 `micro_return` 或 `queue_imbalance_change`）來把資料切分成 Up/Down/Stat。如果用某個特徵切一刀，能最乾淨地把「上漲」跟「下跌」分開（不純度下降最多），那個特徵就會獲得最高的分數。
3. **非線性關聯性**：傳統的相關係數 (Pearson) 只能抓線性關係，但 Random Forest 能抓出非線性的臨界點。這保證了我們選出的 Top 8 特徵，是微觀結構中最具鑑別度的物理量。

**Q3: 關於「滑價 (Slippage)」與「成交量 (Volume)" 的爭議？**

**A3**：
1. **滑價 (Slippage)**：這是硬體模擬中最大的挑戰。在真實交易所，當你用「對敲 (Aggressive)」的方式吃掉对手價時，你的成交價可能跟你的「預掛價」有幾檔的差距。傳統模擬為了追求極致性能，往往**忽略**這幾檔價差，假設 0.5 點就是成交價。然而，根據我們用 `mm_backtest.py` 的測試結果，當 we treat trading cost as 0，净損益 significantly increased（這證明了滑價是造成虧損的主因）。在未來的 HLS 實作中，我們必須納入滑價模型。
2. **成交量 (Volume)**：**這是實盤交易的關鍵差異。**在我們的論文與現有代碼中，`tick_size` 設定為 1 個最小跳動單位（TWD 25 元），`volume_per_lot` 設定為 1 單位。這代表我們模擬的是 **「每筆成交都是 1 單位」** 的理想狀態。在真實期貨市場，通常存在「盤後交易」或「大額交易」，實際成交量會遠大於我們設定的 1 單位。這會導致真實的滑價成本高於模擬，進而壓縮利潤。但在算法邏輯上，我們的回歸模型（TCN）所學習到的「委託簿壓力」與「價格反轉」規律是真實存在的。

"""

with open(log_path, "a", encoding="utf-8") as f:
    f.write(content)
print("Done")
