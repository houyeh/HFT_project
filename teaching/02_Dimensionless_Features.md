# 02. 無因次特徵工程 (Dimensionless Features)

在將金融數據丟進神經網路 (如 TCN) 之前，我們必須將原始價格轉換為模型容易理解的形式。我們稱之為「無因次特徵 (Dimensionless Features)」。

## 為什麼不能直接用原始價格？
假設台指期 (TXF) 價格從 18000 漲到 20000。如果你的神經網路吃的是絕對價格（20000），當合約結算或是市場換到另一個商品（例如價格只有 50 的股票）時，模型看到完全不同的數值範圍，就會崩潰。
為了解決這個問題，我們必須剝離「絕對價格區間」的影響，也就是**去因次化 (Dimensionless)**。

## 常用的核心無因次特徵

### 1. 微觀報酬率 (Micro Return)
不看絕對價格，而是看「報酬率」。
*   `Return = (Current_Mid_Price / Previous_Mid_Price) - 1`
透過這樣的轉換，不管商品是一股 1000 元還是 10 元，變化的比例（例如漲了 0.1%）是一致的，模型學到的是相對動能。

### 2. 正規化價差 (Normalized Spread)
市場流動性好時價差小，流動性差時價差大。
*   `Normalized Spread = (Best_Ask - Best_Bid) / Tick_Size` (跳動點值)
這樣我們可以把價差標準化為「幾個跳動點 (ticks)」。Spread 越大，通常代表市場越恐慌或流動性枯竭。

### 3. 委託簿失衡指標 (Order Book Imbalance, OBI)
這是高頻交易最重要的特徵之一！衡量目前買氣與賣氣的傾斜程度。
*   `OBI = (Bid_Volume - Ask_Volume) / (Bid_Volume + Ask_Volume)`
*   **範圍**：`-1.0` 到 `+1.0`
*   **意義**：當 OBI 接近 +1.0，代表買單（Bid）遠超賣單（Ask），上漲壓力極大；反之接近 -1.0 則是下跌壓力大。

透過將 LOB 轉換這類 Dimensionless features，模型就能跨商品 (Cross-symbol)、跨時間段學習市場的微觀結構模式。
