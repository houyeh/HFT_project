# A3：12 個特徵的數學定義

> 對應 Slide 8「模型技術細節」。
> 以下定義為「追根究柢教授」版本，含數學符號、物理意義、以及與其他特徵的差異說明。

---

## 符號約定

- $B_i$：第 $i$ 檔買價（bid price at level $i$）
- $A_i$：第 $i$ 檔賣價（ask price at level $i$）
- $V^{bid}_i$：第 $i$ 檔買量（bid volume at level $i$）
- $V^{ask}_i$：第 $i$ 檔賣量（ask volume at level $i$）
- $tick$：商品最小跳動單位（tick size）
- $t$：當前事件時間步
- $N$：滾動視窗長度（通常 = 20 或 50 events）

---

## 特徵 1：`normalized_spread`（正規化買賣價差）

$$\text{normalized\_spread} = \frac{A_1 - B_1}{tick}$$

**物理意義**：以 tick 為單位的最佳買賣報價差，代表立即流動性成本。無因次化後跨商品可比較。

**典型值域**：整數，台指期通常 = 1, 偶爾 = 2

---

## 特徵 2：`micro_return`（微觀報酬）

$$\text{micro\_return} = \frac{mid_t - mid_{t-1}}{tick}, \quad mid = \frac{A_1 + B_1}{2}$$

**物理意義**：相鄰兩個 event 之間的中間價變動，以 tick 為單位。捕捉事件級別的短期動能。

**注意**：因為是 event-level（非 clock-time），所以這個 micro_return 對應的時間間隔是不固定的。

---

## 特徵 3：`obi`（委託簿失衡，Order Book Imbalance）

$$\text{OBI} = \frac{\sum_{i=1}^{5} w_i V^{bid}_i - \sum_{i=1}^{5} w_i V^{ask}_i}{\sum_{i=1}^{5} w_i (V^{bid}_i + V^{ask}_i)}$$

目前實作中 $w_i = 1$（等權重），即：

$$\text{OBI} = \frac{\sum_{i=1}^{5} V^{bid}_i - \sum_{i=1}^{5} V^{ask}_i}{\sum_{i=1}^{5} (V^{bid}_i + V^{ask}_i)}$$

**物理意義**：整個委託簿（5 檔）的買賣力量對比。正值代表買방占優（看漲壓力），負值代表賣方占優。

**值域**：$[-1, +1]$

---

## 特徵 4：`weighted_mid_deviation`（加權中間價偏離）

$$\text{weighted\_mid} = \frac{V^{ask}_1 \cdot B_1 + V^{bid}_1 \cdot A_1}{V^{bid}_1 + V^{ask}_1}$$

$$\text{weighted\_mid\_deviation} = \frac{\text{weighted\_mid} - mid}{tick}$$

**物理意義**：VWAP 中間價（成交量加權）相對於算術中間價的偏離程度，以 tick 為單位。反映最佳一檔的 bid/ask 量不對稱性。

**與 OBI 的差別**：OBI 看多檔量的相對大小；weighted_mid_deviation 只看 L1 量的相對比例如何拉偏中間價。

---

## 特徵 5：`deep_obi`（深度委託簿失衡）

$$\text{deep\_OBI} = \frac{\sum_{i=2}^{5} V^{bid}_i - \sum_{i=2}^{5} V^{ask}_i}{\sum_{i=2}^{5} (V^{bid}_i + V^{ask}_i)}$$

**物理意義**：去除最佳一檔後的委託簿失衡，代表「背後的支撐力」。有時可作為 OBI 的 leading indicator（深層先積單，最佳一檔才跟著動）。

**與 OBI 的差別**：OBI 含 L1；deep_OBI 只看 L2-L5，代表流動性深度方向的偏移。

---

## 特徵 6：`queue_imbalance`（最佳一檔委託失衡）

$$\text{queue\_imbalance} = \frac{V^{bid}_1 - V^{ask}_1}{V^{bid}_1 + V^{ask}_1}$$

**物理意義**：只看 L1 最佳買/賣價的掛單量，代表「最前線的即時壓力」。相較於 OBI（全深度），queue_imbalance 更反映下一筆成交前的短期方向壓力。

**與 OBI 的差別**：OBI = 全深度版本；queue_imbalance = L1 單層版本。研究顯示 queue_imbalance 對下一個 tick 方向的預測力往往強於 OBI（Cont et al., 2014）。

---

## 特徵 7：`queue_imbalance_change`（最佳一檔失衡變化量）

$$\text{qi\_change}_t = \text{queue\_imbalance}_t - \text{queue\_imbalance}_{t-1}$$

**物理意義**：委託失衡的一階差分，捕捉「委託壓力方向是在增強還是減弱」，反映市場意圖的動態變化。

---

## 特徵 8-12：執行感知特徵（Execution-Aware Features）

以下特徵由 `src/features/execution_features.py` 中的 `align_trade_features_to_quotes()` 函式計算。

### 特徵 8：`aggressor_imbalance_interval`

$$\text{agg\_imb} = \frac{\sum_{\Delta t} \text{trade\_sign} \cdot V_{trade}}{\sum_{\Delta t} V_{trade}}$$

其中 trade\_sign 由以下層級規則推斷（Quote Rule → Tick Rule）：
1. 若成交價 > mid → +1（主動買）
2. 若成交價 < mid → -1（主動賣）
3. 若成交價 = mid → 看前一個 tick 方向

**物理意義**：在相鄰兩個 quote event 之間，所有成交的淨主動方向（-1 到 +1）。

---

### 特徵 9：`rolling_signed_flow_ratio`

$$\text{signed\_flow} = \frac{\sum_{i \in \text{window\_N}} sign_i \cdot V_i}{\sum_{i \in \text{window\_N}} V_i}$$

**物理意義**：過去 N 個成交事件的淨成交流向比率（Order Flow Imbalance, OFI）。正值代表近期主動買方強，負值代表賣方強。

---

### 特徵 10：`trade_intensity_short`

$$\text{trade\_intensity} = \text{rolling count of trades in recent N quote events}$$

**物理意義**：短期成交活躍程度，代表市場的「激烈程度」。高活躍度通常伴隨高波動率。

---

### 特徵 11：`net_cancel_pressure`

$$\text{cancel} = \frac{(V^{bid}_{t-1,1} - V^{bid}_{t,1}) + (V^{ask}_{t-1,1} - V^{ask}_{t,1})}{\text{normalization factor}}$$

**物理意義**：最佳一檔掛單量的減少量，作為撤單壓力的估計值。撤單壓力高代表市場流動性在消失，往往是price impact 的前兆。

**注意**：這是 proxy（代理指標），因為 TW replay 格式不直接提供成交 vs 撤單的標記。

---

### 特徵 12：`cancel_pressure_short`

**物理意義**：與 `net_cancel_pressure` 類似，但只用更短的滾動視窗（短期撤單壓力），捕捉更即時的流動性消失訊號。

---

## Label 建構說明

$$y_t = \begin{cases} +1 & \text{if } mid_{t+horizon} - mid_t > threshold \cdot tick \\ -1 & \text{if } mid_{t+horizon} - mid_t < -threshold \cdot tick \\ 0 & \text{otherwise (Stationary)} \end{cases}$$

預設參數：
- `horizon = 10`（往後看 10 個 quote event）
- `threshold_ticks = 1.0`

---

## 感受野（Receptive Field）計算

TCN 架構：3 個 ResidualBlock，每塊含 2 個 CausalConv1d

$$RF = 1 + \sum_{b=1}^{3} 2 \times (kernel\_size - 1) \times dilation_b$$

$$RF = 1 + 2 \times (3-1) \times 1 + 2 \times (3-1) \times 2 + 2 \times (3-1) \times 4$$

$$RF = 1 + 4 + 8 + 16 = 29 \text{ events}$$

**意義**：每個樣本的 100 個 event 中，網路的「有效視野」約為最近 29 個 event。透過膨脹卷積（dilation），在不增加參數的情況下擴大感受野。
