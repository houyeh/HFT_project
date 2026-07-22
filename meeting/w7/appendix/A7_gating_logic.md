# A7：Gating 訊號過濾邏輯

> 對應 Slide 8b「Gating 訊號過濾邏輯」。
> 本文件詳細說明模型輸出如何轉換為執行面訊號。

---

## 為什麼需要 Gating？

在高頻交易環境中，並非模型的所有預測都值得執行。Gating 的目的在於：
1.  **降低手續費損耗**：排除信心不足、預期漲幅不夠覆蓋成本的訊號。
2.  **管理風險**：在市場極端波動或流動性不足時強制停止交易。
3.  **提高勝率**：只在模型呈現強烈統計優勢時進場。

---

## 核心數學公式

### 1. 動態機率門檻 (Dynamic Probability Threshold)

我們定義一個方向性的機率 $P_{dir} = \max(P_{up}, P_{down})$。該訊號必須滿足：

$$P_{dir} \ge \text{Threshold}_{base} + (\lambda_{spread} \cdot \text{Spread}) + (\lambda_{vol} \cdot \text{Volatility})$$

- $\text{Threshold}_{base}$：基礎門檻（預設 0.5）。
- $\lambda_{spread}$：價差懲罰係數。
- $\lambda_{vol}$：波動率懲罰係數。

**意義**：當買賣價差變寬或波動變大（風險增加）時，系統會自動要求模型必須有更高的信心度才能進場。

---

### 2. 領先優勢門檻 (Margin Threshold)

訊號必須顯著優於「盤整」狀態（Stationary）：

$$P_{dir} - P_{stationary} \ge \text{Margin}$$

**意義**：防止在模型對「漲」與「盤整」區分不明顯時（例如 0.36 vs 0.35）誤入市場。

---

## 市場狀態守衛 (Regime Guards)

這是一組「硬性開關」，只要有一項不滿足，該樣本就會被標註為 `trade_taken_by_gate = False`：

| 守衛名稱 | 判定公式 | 物理意義 |
|---|---|---|
| **Volatility Guard** | $Vol_{rolling} \le \Theta_{vol}$ | 波動率過高時，模型歷史統計規律往往失效。 |
| **Intensity Guard** | $Count_{trade} \le \Theta_{intensity}$ | 成交過於頻繁可能代表發生極端事件，先暫停觀望。 |
| **Liquidity Guard** | $V_{L1} \ge \Theta_{liquidity}$ | 當最佳一檔掛單太薄，進場會產生巨大的 Price Impact。 |
| **Stability Guard** | $P_{cancel} \le \Theta_{cancel}$ | 撤單壓力太大代表市場極度不安穩，隱藏流動性正在消失。 |

---

## 在 CME 驗證中的體現

在 `W6 CME Logic20` 的驗證中，我們觀察到：
- **Raw Accuracy = 0.82**，但多數預測落入 Stationary。
- 當我們強制要求 $P_{dir} > 0.5$ 時，**多數交易訊號消失**。
- 這代表模型在 CME 上的「信心度」分佈極度向中間靠攏，缺乏強力的方向預估能力。

---

## 未來優化方向：參數掃描 (Parameter Sweep)

下一步我們將執行 `scripts/03_experiments/sweep_thresholds.py`，針對不同商品尋找最佳的 $(Threshold, Margin)$ 組合，以追求在保有一定交易頻率的前提下，最大化 Sharpe Ratio。
