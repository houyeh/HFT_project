# 00 模型選擇過程 (Model Selection Process)

## 為什麼選擇 smoothed_mid_move_3class?
- 檢視了 `experiment_summary.csv` 中的 8 種 label 類型。
- **舊版 Label 的失敗**：`toxic_bid`, `toxic_ask` 與 `adverse` labels 有結構性問題，導致 PnL 接近 0 或是負數。
- **目前的選擇**：`smoothed_mid_move_3class` 是唯一能穩定帶來正向 PnL improvement 的 baseline。
- **重點強調**：這不是最終理想的 label，但它是目前最好、最穩定的 risk-score baseline。
