# 12. 混合造市與 FPGA 硬體加速 (Hybrid Market Making and FPGA)

本專題的核心策略是 **混合造市 (Hybrid Market Making)**。這有別於單純看漲跌的「方向性交易 (Directional Alpha)」。

## 1. 什麼是混合造市？
在傳統造市中，造市商 (Maker) 永遠在最好的買假與賣價掛出對稱的單子 (Symmetric Quoting)，不管漲跌，就是純吃點差。
而 **混合造市 (Hybrid)**，是把神經網路預測的「微弱方向性 Alpha」融合進造市邏輯中：
*   **掛單偏斜 (Quote Skewing)**：如果 TCN 強烈預測下一秒會上漲 (Alpha Score 高)，我們就不掛賣單了 (因為會被買方大單碾過吃虧)，我們只掛買單。這叫做「偏一邊造市 (One-sided quoting)」。
*   **有把屋才造市 (Quote-off Threshold)**：如果 TCN 覺得盤勢混沌不明，我們就**全部抽單 (Cross the spread or Quote off)**，什麼都不做。只在「勝率極高」且「排隊隊伍夠大」的時候，才冒險當造市商。

## 2. 為什麼混合造市需要 FPGA？
我們已經知道 Hybrid MM 依賴 TCN 去預測何時該「不掛單 (Quote-off)」。這其實是一場**與時間賽跑的搶單大戰**。

### A. 逆向選擇的痛苦 (Adverse Selection)
市場行情暴跌的前兆，是市場上所有的聰明錢 (Informed Traders) 瘋狂以市價丟出龐大賣單，瞬間吃光第一層、第二層甚至第三層的買單。
身為造市商，如果你掛在買方第一層，你就是被吃的那塊肉。
因此，當 TCN 透過委託簿失衡 (OBI) 預料到「大砸單要來了」的瞬間，你必須比大單更早一步 **撤單 (Cancel Order)**！

### B. 軟體延遲 (Software Latency) vs 硬體極速 (FPGA Zero-Latency)
*   **一般 Python / C++ Server**：收到交易所封包 ➔ 經過網路卡 ➔ 作業系統中斷 ➔ 搬到 CPU 記憶體 ➔ 通過 Python 直譯器或 C++ TCN 模型算出結果 ➔ 發送撤單封包 ➔ 經過 OS ➔ 送回交易所。這個過程大約需要 **幾百微秒 到 數毫秒 (Milliseconds)**。
    在這個漫長的時間裡，聰明錢的大單早就穿透你的掛單了。**撤單失敗，暴虧！**
*   **FPGA (Field-Programmable Gate Array)**：把神經網路寫死在晶片的實體電路上。收到封包 ➔ 直接拉進晶片的 DSP 物理連線運算 ➔ TCN 計算完成 ➔ 直接由網卡晶片發送撤單。不需要經過 CPU，不需要 OS 中斷。總耗時：**小於 1 微秒 (Microsecond)**。
    在聰明錢的封包還在光纖裡跑的時候，你的 FPGA 就已經撤單完畢。**撤單成功，全身而退！**

**專題終極目標**：利用軟體的 Event-Driven 回測，模擬 Python 延遲 (`delay=4`) 時被大單貫穿虧損，而 FPGA 零延遲 (`delay=0`) 成功逃逸並獲利的圖表，以此完美證明硬體加速的巨大商業價值。
