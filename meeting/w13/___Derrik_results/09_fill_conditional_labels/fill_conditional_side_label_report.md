# 條件成交 (Fill-Conditional) 側邊標籤毒性報告

## 1. 審查背景 (Audit Context)
- **資料集 (Dataset)**: NEW
- **資料切分 (Split)**: test
- **預測區間 (Horizon)**: 10
- **替代指標類型 (Proxy Type)**: price_level_removed_proxy
- **成交視窗 (Fill Window)**: 10

## 2. 條件毒性提升幅度 (Conditional Toxicity Lift)
- **Bid 毒性率提升幅度 (Bid Toxic Rate Lift)**: 3.32x
- **Ask 毒性率提升幅度 (Ask Toxic Rate Lift)**: 2.14x

## 3. 分析與結論 (Interpretation)
- 使用替代指標 C (`price_level_removed_proxy`) 在 horizon=10 且 fill_window=10 的設定下，我們觀察到 **毒性率有顯著的倍數提升 (Lift)**。
- 相比於無條件 (Unconditional) 基準，有成交條件 (Fill-conditional) 下的毒性風險明顯變高。
- 然而，條件毒性率的絕對值仍落在低個位數百分比 (Low single-digit percent) 區間。
- **極重要聲明**：由於缺乏對手單 (Aggressor-side) 數據，這份報告是基於替代指標 (Proxy evidence)，並非交易所真實成交的絕對真相 (Ground truth)。
- 這項發現為後續針對 `bid_toxic` 與 `ask_toxic` 進行模型重訓的研究，提供了強而有力的動機與方向。
