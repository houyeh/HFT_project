import os
from pathlib import Path
import pandas as pd
import shutil

DERRIK_ROOT = Path("c:/Users/holbo/00VS_projjects/00_HFT_project/meeting/w13/___Derrik_results")

# 1. Rename data detailed files
for root, dirs, files in os.walk(DERRIK_ROOT):
    for f in files:
        if f.startswith("data_source_details") or "data_detailed" in f:
            if not f.startswith("00_"):
                old_path = Path(root) / f
                new_path = Path(root) / ("00_" + f)
                old_path.rename(new_path)
                print(f"Renamed: {f} -> 00_{f}")

# 2. Translate CSVs
replacements = {
    # label_type_comparison
    "SELECTED: highest mean pnl_improvement": "已選取：平均 PnL 提升最高",
    "runner-up; no smoothing, slightly lower pnl_improvement": "候補：無平滑處理，PnL 提升稍低",
    "REJECTED: mean pnl_improvement negative": "拒絕：平均 PnL 提升為負",
    "REJECTED: zero improvement, model rarely signals": "拒絕：零提升，模型極少觸發訊號",
    "REJECTED: near-random F1, zero improvement": "拒絕：F1 分數近乎隨機，零提升",
    "REJECTED: near-random F1, zero or negative improvement": "拒絕：F1 分數近乎隨機，零或負提升",
    # phase2
    "final selected: H=5 thr=0.5 for cancel filter framework": "最終選取：H=5 thr=0.5 用於撤單過濾器框架",
    "H=10 thr=1: highest raw pnl_improvement but uses older backtest framework": "H=10 thr=1：原始 PnL 提升最高，但使用舊版回測框架",
    "simulate_with_metrics (cancel filter)": "simulate_with_metrics (撤單過濾器)",
    "tcn_skew (original backtest)": "tcn_skew (原始回測)",
    # phase3
    "SELECTED: stable zscore_window=2000, consistent val_loss": "已選取：穩定的 zscore_window=2000，val_loss 穩定",
    "REJECTED: zscore_window=10 too short, OOS stability concern": "拒絕：zscore_window=10 太短，有樣本外穩定性疑慮",
    "REJECTED: zscore_window=5 unstable": "拒絕：zscore_window=5 不穩定",
    "baseline run": "基準運行 (baseline)",
    "initial freeze test": "初始權重凍結測試",
    # classification
    "low precision + high recall: useful as cancel-risk trigger, not directional alpha": "低精準度 + 高召回率：適合做為撤單風險觸發器，不適合做為主動交易訊號",
    "dominant class with high precision and moderate recall": "主導類別，具備高精準度與中等召回率",
    # baseline
    "beats matched random mean+1std in NEW test split": "在 NEW 測試集中，擊敗了隨機基準的 平均值+1標準差",
    "beats random mean but not mean+1std": "擊敗隨機基準的平均值，但未超過 平均值+1標準差",
    "does not beat random benchmark robustly": "未能穩定擊敗隨機基準",
    # opportunity
    "avoided toxic loss exceeds missed profitable opportunity": "避開的毒性虧損 大於 錯失的獲利機會",
    "missed profitable opportunity dominates": "錯失的獲利機會 佔主導地位",
    # regime gate
    "best gate by net_pnl for dataset": "該資料集下 Net PnL 最佳的 Gate 設定",
    "reference near no-gate behavior (spread_filter=1.0)": "接近無 Gate 狀態的參考基準 (spread_filter=1.0)",
    # fill conditional
    "proxy conditional toxicity lift observed; proxy only, not true fill label": "觀察到條件毒性率上升；但此為替代指標，非真實交易所成交紀錄",
    # latency
    "possible false-positive cancel reduction; non-monotonic gain": "可能減少了假陽性撤單；非單調遞減的獲利",
    "deterioration at long cancel delay": "在長撤單延遲下表現明顯惡化",
    "cancel delay trade-off": "撤單延遲的權衡 (Trade-off)",
    "prediction delay changes signal timing; non-monotonic response possible": "預測延遲改變了訊號時機；可能出現非單調反應",
    "combined delay effect": "綜合延遲效應",
}

for root, dirs, files in os.walk(DERRIK_ROOT):
    for f in files:
        if f.endswith(".csv"):
            csv_path = Path(root) / f
            try:
                df = pd.read_csv(csv_path)
                changed = False
                for col in df.columns:
                    if df[col].dtype == object:
                        # try replacing exact matches or substrings
                        for k, v in replacements.items():
                            if df[col].astype(str).str.contains(k, regex=False).any():
                                df[col] = df[col].apply(lambda x: str(x).replace(k, v) if pd.notnull(x) else x)
                                changed = True
                if changed:
                    df.to_csv(csv_path, index=False)
                    print(f"Translated CSV: {csv_path.name}")
            except Exception as e:
                pass

# 3. Overwrite slide_summary.md with Chinese content
markdowns = {
    "00_result": """# 模型選擇過程 (Model Selection Process)

## 階段一：Label 類型掃描 (8 種 label 類型 x 特徵數量在 {6,7,8,9}, H=10, thr=1)
- 測試了 8 種 label 類型。
- 只有 `smoothed_mid_move_3class` 和 `mid_move_3class` 顯示正的 PnL 改善平均值。
- **最終選取**：`smoothed_mid_move_3class`。
- **拒絕**：所有二元分類（toxic / adverse / active），因為無法穩定超越 baseline。

## 階段二：特徵數量與預測區間 (Horizon) 選擇
- 確認使用 fc=8, window=100。
- H=10 thr=1 在原始回測框架中給出了最高的原始 PnL 提升。
- 最終轉向使用 H=5 thr=0.5 配合新的撤單過濾器診斷框架 (`simulate_with_metrics`)。

## 階段三：expA/B/C Snapshot 選擇 (H=5, thr=0.5, fc=8)
- expB_w10 的 val_loss 最低，但 zscore_window=10 太短，樣本外不穩定。
- **最終選取**：`expC_snapshot`，zscore_window=2000，val_loss=0.781，表現穩定。
- expC 將作為後續所有實驗的基準模型。
""",
    
    "01_result": """# 從主動方向性預測 轉向 毒性訂單撤單過濾器

- TCN **並非**用作主動方向性交易訊號。
- TCN 是用作**毒性流量風險評分 (Toxic-flow risk score)**，用於防禦性撤單。
- 策略邏輯：
  - `P(Down)` 機率高 -> 撤銷買單 (Cancel Bid)
  - `P(Up)` 機率高 -> 撤銷賣單 (Cancel Ask)
- 目前狀態：為研究候選 (research candidate)，尚未達到可直接上線生產 (production-ready) 的狀態。
""",

    "02_result": """# 資料集與模型基本資訊

- **資料集來源**：見 00_data_source_details.md
- **交易對 (Symbol)**：TXF
- **資料切分 (Train/Val/Test)**：採時間序列切分，詳細筆數見 00_data_source_details.md
- **特徵與視窗**：特徵數 (feature_count)=8，正規化視窗 (zscore_window)=2000
- **模型視窗**：輸入視窗 (window_size)=100，預測區間 (horizon)=5
- **目標標籤**：`smoothed_mid_move_3class`
""",

    "03_result": """# 訓練收斂與分類報告

- **Loss Curve**：模型在訓練集與驗證集上皆有正常收斂，沒有嚴重 overfitting。
- **Confusion Matrix (混淆矩陣)**：
  - Up/Down 類別呈現「低精準度 (Precision) + 高召回率 (Recall)」的特性。
  - 這樣的特性不適合用來做方向性下單，但非常適合做為高敏感度的「撤單防禦觸發器」。
""",

    "04_result": """# Baseline 回測證據 (Threshold 比較)

- **0.55 局部優勢 (Local Edge)**：在 NEW 測試集中，0.55 的 PnL 擊敗了「隨機撤單平均值 + 1 個標準差」，具備防禦價值。
- **0.60 局部優勢 (Local Edge)**：同樣擊敗了「隨機撤單平均值 + 1 個標準差」。
- **0.70 (驗證集選出)**：這是我們依照 Validation loss 選出的參數，但在測試集中表現並未穩健超過隨機+1std。因此我們誠實說明：目前的最佳 threshold 偏向是事後掃出來的候選。
""",

    "05_result": """# 機會成本分析 (Opportunity Cost)

- 在目前的診斷回測假設下，**「成功避開的毒性虧損」大於「因撤單而錯失的獲利機會」**。
- 因此，Net Cancel Value (淨撤單價值) 在這些診斷切片中保持為正數。
""",

    "06_result": """# 為什麼需要狀態濾網 (Why Regime Gate)

- 在 NEW3 這個充滿強烈趨勢的壓力測試集中，如果不加 Gate，模型會帶來劇烈虧損（大約虧損 55 萬）。
- NEW3 代表了不適合做市的極端市場狀態 (Stress / Unfavorable Regime)。
- 這證明了一件事：**微觀的 100-tick TCN 撤單邏輯，無法單獨拯救宏觀趨勢下的虧損，必須加上 Regime Gate 來防護。**
""",

    "07_result": """# 狀態濾網成果 (Regime Gate Results)

- **NEW 最佳組合**：Rolling Trend 配合 TCN，能顯著改善 PnL。
- **NEW3 最佳組合**：Spread Filter 配合 TCN，能將原本的大幅虧損拉回到接近平盤，甚至微幅正數。
- **結論**：這顯示了「市場狀態依賴性 (Regime Dependency)」。不同市場狀態需要不同的 Gate，目前尚未找出統一的生產環境 Gate 選擇器，這將是 Auto-Gating 的下一步目標。
""",

    "08_result": """# 已實作的 Regime Gate

- **已實作**：
  - 滾動趨勢濾網 (Rolling trend filter)
  - 滾動波動率濾網 (Rolling volatility filter)
  - 價差濾網 (Spread filter)
  - 綜合濾網 (Combined_filter)
- **尚未實作 (未來工作)**：
  - EWMA Gate
  - 基於 Kalman 的 Gate
""",

    "09_result": """# 條件成交標籤審查 (Fill-Conditional Label Audit)

- 使用了 Proxy C (`price_level_removed_proxy`) 在 horizon=10 的條件下進行審查。
- 發現「有成交條件下」的毒性率（Toxic Rate）相比「無條件下」有顯著的倍數提升 (Lift)。
- ⚠️ 注意：目前沒有對手單 (Aggressor-side) 數據，所以這只是替代指標 (Proxy) 證據，不是交易所真實成交的 Ground Truth。
- 這個結果給了我們後續訓練 `bid_toxic` / `ask_toxic` binary model 極大的信心與方向。
""",

    "10_result": """# 延遲敏感度分析 (Latency Sensitivity)

- 隨著撤單延遲 (Cancel delay) 增加，診斷出的「延遲毒性虧損 (Delayed toxic loss)」也會隨之增加。
- Net PnL 對延遲的反應不是單調的：
  - 在 Delay 3~5 時，PnL 有時反而會上升，這可能是因為過濾掉了假陽性撤單。
  - **但在 Delay 10 時，所有關鍵切片的表現都會明顯惡化。**
- 結論：推動 FPGA 硬體加速的主要學術理由，在於**極致的風險控管 (Risk-control)**，防止延遲過大造成防禦崩潰，而不是單純為了賺更多錢。
""",

    "11_result": """# 目前狀態與限制 (Current Status and Limitations)

目前進度大約在 C+ / B- 的階段。

**已達成的里程碑：**
- 證明 TCN 在 NEW 測試集上有 Local edge。
- 模型 (Threshold 0.55 / 0.60) 成功擊敗 Random+1std。
- Regime Gate 成功在 NEW3 壓力測試中發揮救援作用。
- Fill-conditional 審查發現了毒性率的提升方向。
- 具體量化了 Cancel Delay 帶來的虧損風險。

**尚未達成的目標：**
- Validation 所選出的 Threshold 尚未能在測試集穩健擊敗 Random+1std。
- 尚未完成 `bid_toxic` / `ask_toxic` 的獨立重新訓練。
- 尚缺基於真實 Queue-aware 的回測驗證 (HftBacktest-style)。
- 尚未開發出動態適應的 Auto-Gating 機制。
- 尚未實作 FPGA 的硬體獲利證明。
""",

    "12_result": """# 下週行動計畫 (Next Steps)

1. **標籤重建**：建立專屬的 `bid_toxic` 與 `ask_toxic` 二元標籤。
2. **模型重訓**：使用新的 Side-specific 毒性標籤重新訓練輕量級模型。
3. **閾值選擇**：在有 Regime Gate 防護的前提下，透過 Validation 集嚴謹選擇 Threshold。
4. **效能比對**：與隨機撤單及基於規則 (Rule-based) 的基準進行嚴格比對。
5. **回測升級**：在宣稱硬體價值前，導入 Queue-aware (類似 HftBacktest) 的真實搓合驗證。
6. **FPGA MVP**：規劃簡單的 Rolling/EWMA Gate + 輕量級 TCN 撤單邏輯進入硬體。
"""
}

for folder, content in markdowns.items():
    p = DERRIK_ROOT / folder
    if p.exists():
        # find slide_summary.md or the other md files and overwrite
        md_file = p / "slide_summary.md"
        if md_file.exists():
            md_file.write_text(content, encoding="utf-8")
            print(f"Updated: {md_file}")
            
        # Also translate specific files
        if folder == "08_result":
            (p / "implemented_regime_gate.md").write_text(content, encoding="utf-8")
        if folder == "11_result":
            (p / "current_status.md").write_text(content, encoding="utf-8")
        if folder == "12_result":
            (p / "next_steps.md").write_text(content, encoding="utf-8")

print("Done translating all summaries to Traditional Chinese.")
