# Week 13 講者逐字稿 (Speaker Script)

## 00 ｜ 模型選擇過程
「各位教授好。在過去一週，我們徹底評估了 8 種不同的 Label Type。我們發現過去使用的 toxic_bid 或 adverse labels 存在結構性的失敗，回測 PnL 幾乎為零或是負數。
相比之下，smoothed_mid_move_3class 成為了目前 PnL improvement 最穩定且為正的 baseline。需要強調的是，這不代表它是我們最終理想的 Label，但它是我們目前做 risk-score 防禦最好的基石。」

## 01 ｜ 最終選擇的模型
「我們目前的 Final Model 採用了 8 個微觀特徵，觀察視窗為 100 ticks。
請看 Confusion Matrix，大家會發現它預測 Down 或 Up 的 Precision 非常低，這主要是因為市場絕大多數時間都在盤整。但是！它的 Recall 相當高。
這讓我們確認了一件事：這個模型不是拿來做主動方向性交易的，它極度適合被用作高敏感度的『毒性訂單撤單過濾器 (Toxic-Flow Cancel Filter)』。」

## 02 ｜ 回測策略與結果
「我們採用了 Maker-only 的策略，只要預測機率大於 Threshold 就撤銷掛單。
在 NEW 的 Test Split 中，Threshold 0.55 與 0.60 展示了非常好的 local edge，明顯擊敗了隨機撤單 baseline。
不過我們要誠實地說，在 Validation-selected 的 0.70 上，表現尚未大幅贏過 Random + 1 std，因此 0.55 目前只能算測試候選，還無法宣稱為 Production-ready。」

## 03 ｜ 為什麼需要 Regime
「為什麼光有 Cancel Filter 不夠？因為 TCN 的 window 只有 100 步，只能處理微觀結構。
我們將模型放進 NEW3 這個充滿強烈趨勢的壓力測試資料集中，發現模型大幅虧損了 55 萬，幾乎跟沒用模型一樣。
這證明了一件事：微觀的過濾器救不了宏觀的 Regime 崩盤。NEW3 絕不是無效數據，而是一個必須透過事前 Causal Gate 進行防禦的極端市場狀態。」

## 04 ｜ Regime 的加入與成效
「因此我們加入了 Rolling Trend 與 Spread Filter 作為 Regime Gate。
加入後，NEW 和 NEW3 都有專屬的最佳 Gate 配置，且 Active Rate 依然保持在 82% 左右，證明 PnL 的改善不是單純因為我們不交易了。
更重要的是，這兩個資料集需要不同的 Gate，這成為了 Regime Dependency 最直接的證據。下一步，我們必須做出能夠自動切換的 Auto-Gate。」

## 05 ｜ 新發現：Label 與 Latency
「在進行了深度診斷後，我們有了兩個新發現。
首先是 Label，我們發現 fill-conditional 的 toxic rate 遠比 unconditional 來得高，這指出了下週模型優化的明確方向。當然，這目前是基於 proxy 資料。
第二是 Latency，我們發現 Cancel Delay 超過 10 ticks 後表現明顯變差。這證明了我們朝向 FPGA 硬體加速的必要性：它是為了極致的 Risk-Control，以防禦延遲造成的巨大虧損。」

## 06 ｜ 下週行動事項
「下週我們將專注於三點：
第一，放棄舊有 toxic labels，改訓練 Binary Quote-Risk Model，並與目前的 Baseline 進行嚴謹對比。
第二，實作 Auto-Gating，用 Causal Feature 自動辨識市場狀態，絕不使用未來函數或在 NEW3 上作弊調參。
最後，開始規劃 FPGA 的 MVP Path，專注於降低 Cancel Latency。」
