# 05. 資料前處理 (Data Preprocessing)

取得交易所的原始高頻資料 (Raw Data) 後，你不能直接把資料拿來算特徵或訓練模型。真實世界的資料充滿了雜訊 (Noise)、缺失值 (Missing values) 以及各種極端異常狀況 (Outliers)。

## 1. 處理缺失值 (Handing Missing Values)
在委託簿 (LOB) 中，有時候由於網路封包掉包，或者某個買賣檔位完全空了，會出現 `NaN` (Not a Number)。
*   **價格填補 (Forward Fill, ffill)**：如果最新一筆資料沒有給價格，通常我們會假設「價格停留在上一個有效狀態」，因此使用前向填充 (`ffill()`)。
*   **數量填補**：如果某一檔的委託量是 NaN，我們通常要補零 (`fillna(0)`)，因為那代表簿子上目前沒有人在排隊。

## 2.  Tick Size 對齊 (Tick Size Alignment)
在期貨或股票中，價格不是連續的，而是一格一格跳動的。這個最小跳動單位稱為 **Tick Size**。
例如台指期 (TXF) 的 tick size 是 1，而小型台指 (MXF) 也是 1 點。但加密貨幣可能是 0.01。
在算特徵（如 Spread）時，一定要使用 `tick_size` 正規化：
`Spread = (Ask_Px - Bid_Px) / Tick_Size`，這樣才能統一度量衡，確保模型在不同商品間能一視同仁。

## 3. 去除離群值或髒資料 (Outlier Rejection)
有時候原始資料會出現 `Bid_Price > Ask_Price` 的情況，這在正常的撮合引擎下是荒謬的（因為買價高於賣價應該立刻成交）。這通常是行情轉發系統 (Market Data Gateway) 亂序或錯亂造成的。
在 Preprocessing 階段，我們必須過濾掉這些髒資料：
```python
df = df[df['bid_px_1'] < df['ask_px_1']]
```

只有經過嚴格清洗的 Clean Data，才能萃取出乾淨的 Alpha 訊號，這就是業界常說的 **Garbage In, Garbage Out (GIGO)** 準則。
