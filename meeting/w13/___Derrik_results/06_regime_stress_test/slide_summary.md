# 為什麼需要狀態濾網 (Why Regime Gate)

- 在 NEW3 這個充滿強烈趨勢的壓力測試集中，如果不加 Gate，模型會帶來劇烈虧損（大約虧損 55 萬）。
- NEW3 代表了不適合做市的極端市場狀態 (Stress / Unfavorable Regime)。
- 這證明了一件事：**微觀的 100-tick TCN 撤單邏輯，無法單獨拯救宏觀趨勢下的虧損，必須加上 Regime Gate 來防護。**
