import os
from pathlib import Path
import re

DERRIK_ROOT = Path("c:/Users/holbo/00VS_projjects/00_HFT_project/meeting/w13/___Derrik_results")

# Fix 04_result/00_slide_summary.md renaming back to slide_summary.md
bad_file = DERRIK_ROOT / "04_result" / "00_slide_summary.md"
if bad_file.exists():
    good_file = DERRIK_ROOT / "04_result" / "slide_summary.md"
    bad_file.rename(good_file)

# List of all md files
md_files = list(DERRIK_ROOT.rglob("*.md"))

# Manual mapping for specific files
files_to_rewrite = {
    "week13_meeting_pack_summary.md": """# Week13 會議資料包總覽 (Meeting Pack Summary)

## 輸出資訊
- 輸出目錄: `week13_meeting_pack`
- 簡報資料夾: 00_result 到 12_result
- 檔案清單: `week13_output_manifest.csv`

## 缺失檔案 (Missing Inputs)
- 無

## 警告與備註 (Warnings)
- 未執行新的回測。
- 未執行模型重新訓練。
- Threshold 0.55 並未被標示為可上線生產 (production) 的閾值。
- Fill-conditional (有成交條件) 的標籤審查是基於替代指標 (Proxy)，並非真實交易所成交數據。
- 延遲 (Latency) 的解釋避免了單調崩潰的過度宣稱。

## 須避免的宣稱 (Claims To Avoid)
- 絕對不要宣稱具備主動方向性交易 (Directional trading) 的能力。
- 絕對不要宣稱延遲大於 3 ticks 策略就會完全崩盤。
- 絕對不要宣稱替代指標 (Proxy) 估算的成交是真實成交。
- 絕對不要宣稱已經解決了統一的生產環境 Gate 選擇器問題。
""",
    "14_old_label/old_labels_review.md": """# 舊標籤定義與訓練結果回顧 (Old Labels Review)

這份文件從 `src/features/label_generation.py` 中擷取了舊版二元標籤 (Binary labels) 的確切定義，並比較了它們的訓練與回測表現。

## 1. 標籤定義 (Label Definitions)

```python
# 1. toxic_bid_binary
# 定義：如果在預測區間內，未來的最低 Mid Price 跌破了當前的 Bid 減去設定的 threshold_price，即標記為 toxic。
future_mid_min = future_mid.min(axis=1)
toxic = (future_mid_min < (current_bid - threshold_price)).astype(np.int64)

# 2. toxic_ask_binary
# 定義：如果在預測區間內，未來的最高 Mid Price 突破了當前的 Ask 加上設定的 threshold_price，即標記為 toxic。
future_mid_max = future_mid.max(axis=1)
toxic = (future_mid_max > (current_ask + threshold_price)).astype(np.int64)

# 3. adverse_bid_fill_binary
# 定義：與 toxic_bid 相同，但額外要求必須有立即成交的替代指標 (next_ask <= current_bid)。
bid_fill = next_ask <= current_bid
future_mid_min = future_mid.min(axis=1)
adverse = bid_fill & (future_mid_min < (current_bid - threshold_price))

# 4. adverse_ask_fill_binary
# 定義：與 toxic_ask 相同，但額外要求必須有立即成交的替代指標 (next_bid >= current_ask)。
ask_fill = next_bid >= current_ask
future_mid_max = future_mid.max(axis=1)
adverse = ask_fill & (future_mid_max > (current_ask + threshold_price))

# 5. active_buy_profitable_binary
# 定義：如果主動買入 (跨越價差吃 Ask) 能產生大於總成本 (Taker fee + margin) 的利潤。
future_mid_max = future_mid.max(axis=1)
profit_ticks = (future_mid_max - current_ask) / tick_size
out = (profit_ticks > total_cost_ticks).astype(np.int64)

# 6. active_sell_profitable_binary
# 定義：如果主動賣出 (跨越價差吃 Bid) 能產生大於總成本的利潤。
future_mid_min = future_mid.min(axis=1)
profit_ticks = (current_bid - future_mid_min) / tick_size
out = (profit_ticks > total_cost_ticks).astype(np.int64)
```

## 2. 表現總結 (Performance Summary)

以下是這些舊標籤在不同特徵數量 (fc=6,7,8,9) 下的平均表現：

### toxic_bid_binary
- **實驗次數**: 4
- **平均最佳 Val Loss**: 0.6971
- **平均測試集 Macro F1**: 0.4993
- **平均 PnL 提升**: 0.00 TWD

### toxic_ask_binary
- **實驗次數**: 4
- **平均最佳 Val Loss**: 0.6903
- **平均測試集 Macro F1**: 0.5020
- **平均 PnL 提升**: -685.00 TWD

### adverse_bid_fill_binary
- **實驗次數**: 4
- **平均最佳 Val Loss**: 0.2244
- **平均測試集 Macro F1**: 0.5000
- **平均 PnL 提升**: 0.00 TWD

### adverse_ask_fill_binary
- **實驗次數**: 4
- **平均最佳 Val Loss**: 0.1234
- **平均測試集 Macro F1**: 0.5000
- **平均 PnL 提升**: 0.00 TWD

### active_buy_profitable_binary
- **實驗次數**: 4
- **平均最佳 Val Loss**: 0.5904
- **平均測試集 Macro F1**: 0.5536
- **平均 PnL 提升**: -11,796.25 TWD

### active_sell_profitable_binary
- **實驗次數**: 4
- **平均最佳 Val Loss**: 0.5594
- **平均測試集 Macro F1**: 0.5469
- **平均 PnL 提升**: -10,381.25 TWD

## 3. 失敗原因分析 (Diagnostic Verdict)

- **`adverse_bid/ask_fill_binary`**: `best_val_loss` 極度偏低 (約 0.12 到 0.22)。這是嚴重類別不平衡 (Class Imbalance) 的強烈指標。`bid_fill` 和 `ask_fill` 條件 (`next_ask <= current_bid` 或 `next_bid >= current_ask`) 在現實中極少發生，因為價格很少會在不刷破檔位的情況下瞬間交叉。因此，正類別 (Positive class) 幾乎不存在，模型學會了不斷預測 0，導致 F1 ≈ 0.5 且 PnL 提升為 0。
- **`toxic_bid/ask_binary`**: 沒有檢查是否成交，只衡量預測區間內的 Mid Price 是否超出 Bid/Ask。雖然這裡有更多的正樣本，但測試集 Macro F1 接近 0.5 (隨機亂猜)。模型無法找到穩定的優勢，導致 PnL 提升為零或微幅負數。F1 卡在 0.5 顯示模型只是在預測多數類別，因為它無法單憑 Order book 區分出毒性與非毒性。
- **`active_buy/sell_profitable_binary`**: 專為積極的 Taker 交易設計。要求未來的價格移動必須克服價差 AND Taker 手續費 (每點 50 TWD)。這些標籤的要求太過嚴苛。模型試圖預測它們，但大多數觸發的都是假陽性 (False positives)，導致嚴重的負 PnL (平均虧損 1萬 到 1.2萬 TWD)。

總結來說，無條件的 `mid_move` 與 `smoothed_mid_move` 標籤最終被證明是讓模型捕捉短期微觀 Alpha 最平衡的目標。這也就是為什麼 `smoothed_mid_move_3class` 會被選為目前撤單過濾器 (Cancel filter) 的最終候選基準。
""",
    "09_result/fill_conditional_side_label_report.md": """# 條件成交 (Fill-Conditional) 側邊標籤毒性報告

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
"""
}

translations = {
    "Training and Classification — Verified Numbers": "訓練與分類 — 驗證數據",
    "Training Convergence": "訓練收斂情況",
    "Source:": "來源檔案:",
    "Best epoch by val_loss": "由 val_loss 選出的最佳 epoch",
    "min val_loss": "最低 val_loss",
    "train_loss monotonically decreasing": "train_loss 單調遞減",
    "no divergence": "無發散",
    "val_loss also generally decreasing": "val_loss 也整體呈現遞減",
    "no overfitting": "無過度擬合",
    "epoch 30 is best": "epoch 30 為最佳",
    "Classification Report": "分類報告",
    "Test samples:": "測試樣本數:",
    "Cancel Filter Interpretation": "撤單過濾器解讀",
    "Down/Up precision ≈": "Down/Up 精準度 (Precision) ≈",
    "NOT usable as directional alpha": "不可用作主動方向性訊號",
    "Down/Up recall ≈": "Down/Up 召回率 (Recall) ≈",
    "captures majority of adverse moves": "捕捉了多數的不利價格移動",
    "useful as cancel trigger": "適合作為撤單觸發器",
    "Strategy: use as high-sensitivity cancel filter, not directional entry": "策略定位：作為高敏感度的撤單過濾器，而非方向性進場點",
    "Sources": "資料來源",
    
    # Generic generic
    "Dataset Details": "資料集詳細使用清單",
    "Train Split": "訓練集 (Train Split)",
    "Validation Split": "驗證集 (Validation Split)",
    "Test Split": "測試集 (Test Split)",
}

for md_path in md_files:
    rel_path = md_path.relative_to(DERRIK_ROOT).as_posix()
    if rel_path in files_to_rewrite:
        md_path.write_text(files_to_rewrite[rel_path], encoding="utf-8")
        print(f"Rewrote exact file: {rel_path}")
    elif "slide_data_detailed" in rel_path:
        content = md_path.read_text(encoding="utf-8")
        changed = False
        for k, v in translations.items():
            if k in content:
                content = content.replace(k, v)
                changed = True
        if changed:
            md_path.write_text(content, encoding="utf-8")
            print(f"Translated generic strings in: {rel_path}")

print("Done finalizing Chinese translations.")
