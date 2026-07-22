# 16_Old. 實驗方法 A, B, C 對比 (Method A, B, C Experiments)

> **注意：此文件為早期開發階段的紀錄，用以展示我們如何摸索造市商的防禦機制。請參考最新的 `17_HFT_First_Principles.md` 獲取最終結論。**

在尋找能讓 `delay=0` 賺錢的路上，我們嘗試了幾種不同的過濾器控制方法：

## 實驗 A：強引用 Quote-off 與 Freeze 雙重過濾
*   **邏輯**：當 TCN 分數 (Alpha Score) 很小時，啟動 Quote-off 不報價。當發現毒性 (Toxicity) 飆高時，啟動 Toxicity Freeze (凍結並全撤)。
*   **現象**：在 Validation 集上跑出了 `+20 ticks` 的好成績。但在 Test 上的 `delay=0` 卻虧了 `-15 ticks`。
*   **原因**：這代表這種複雜的雙重過濾在歷史資料上產生了 **過度擬合 (Overfitting)**。它挑選出了特定的時間段，但在未知的市場中失去了防禦力。

## 實驗 B：極端收緊 Signal 訊號門檻 (高 Precision 路線)
*   **邏輯**：不要管 Quote-off 了，我們單純要求 TCN 給出極度強烈的方向性機率 (例如 `Prob_Up > 0.40`) 才掛買單。
*   **現象**：交易次數銳減。即便是在 `delay=0` 的情況下，因為出手次數太少，只要被大單碾過一兩次，就再也賺不回來。
*   **原因**：這不是造市商的做法，這是投機客。造市商靠高頻的 Turnover 來賺錢，一味收緊訊號只會餓死。

## 實驗 C：大破大立，無腦報價 (Dumb Quoting Baseline)
*   **邏輯**：將 `quote_off_threshold` 設為常開 (也就是只要沒危險，必定雙邊報價)。把防禦的重責大任全部交給 TCN 觸發的 `Toxicity Freeze`。
*   **現象**：成交量暴增。這時我們終於能在 `delay=0` 看到與 `delay=2` 截然不同的表現。
*   **原因**：這就是第一性原理的核心。用高頻報價累積 Spread，並把這累積下來的資本，用來在 `delay=0` 發生毒性時極速撤防，展現了延遲的真正價值。
