import os
import shutil
import pandas as pd
from pathlib import Path
import json

def read_csv_safe(path):
    if not path.exists():
        return "MISSING"
    try:
        return pd.read_csv(path)
    except Exception as e:
        return f"ERROR: {e}"

def write_md(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def copy_safe(src, dst):
    if src.exists():
        shutil.copy(src, dst)

def safe_to_csv(df, path, index=False):
    try:
        df.to_csv(path, index=index)
    except PermissionError:
        print(f"Permission denied for {path}, writing to {path.stem}_backup.csv")
        df.to_csv(path.parent / f"{path.stem}_backup.csv", index=index)

def generate_deck():
    project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
    out_dir = project_root / "meeting/w13/___Nancy_meeting_deck"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = project_root / "new_era/15_models_playaround/015_models_playaround_result/lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot"
    eval_dir = model_dir / "eval_outputs"
    diag_dir = eval_dir / "diagnosis"
    diag_v3_dir = eval_dir / "diagnosis_v3"
    
    # 00 Model Selection
    d00 = out_dir / "00_model_selection"
    d00.mkdir(exist_ok=True)
    write_md(d00 / "slide.md", """# 00 模型選擇過程 (Model Selection Process)

## 為什麼選擇 smoothed_mid_move_3class?
- 檢視了 `experiment_summary.csv` 中的 8 種 label 類型。
- **舊版 Label 的失敗**：`toxic_bid`, `toxic_ask` 與 `adverse` labels 有結構性問題，導致 PnL 接近 0 或是負數。
- **目前的選擇**：`smoothed_mid_move_3class` 是唯一能穩定帶來正向 PnL improvement 的 baseline。
- **重點強調**：這不是最終理想的 label，但它是目前最好、最穩定的 risk-score baseline。
""")
    # Clean label_sweep_summary
    exp_csv = project_root / "new_era/15_models_playaround/015_models_playaround_result/experiment_summary.csv"
    if exp_csv.exists():
        df_exp = pd.read_csv(exp_csv)
        if "label_type" in df_exp.columns:
            df_label = df_exp.groupby("label_type")[["best_val_loss", "test_macro_f1", "model_test_pnl", "pnl_improvement"]].mean().round(4).reset_index()
            df_label = df_label.sort_values("pnl_improvement", ascending=False)
            df_label["Verdict"] = df_label["label_type"].apply(lambda x: "SELECTED" if "smoothed_mid_move_3class" in x else "REJECTED (Poor PnL/F1)")
            safe_to_csv(df_label, d00 / "label_sweep_summary.csv")
        else:
            copy_safe(exp_csv, d00 / "label_sweep_summary.csv")
    
    # 01 Final Model
    d01 = out_dir / "01_final_model"
    d01.mkdir(exist_ok=True)
    write_md(d01 / "slide.md", """# 01 最終選擇的模型 (Final Model Identity)

## 架構與資料 (Architecture & Data)
- **資料集 (Dataset)**: NEW_ALL / TXFB6
- **資料切分 (Split)**: 嚴格依照時間順序切分 (Chronological Split: 70% Train / 15% Val / 15% Test)
- **特徵 (Features)**: 8 個微觀特徵 (Z-score window: 2000)
- **時間視窗 (Window)**: 100 ticks | **預測視距 (Horizon)**: 5 ticks
- **目標標籤 (Label)**: `smoothed_mid_move_3class` (Threshold: 0.5 ticks)
- **模型結構 (Model)**: TCN (channel_list=[16,16,16], kernel_size=3, dropout=0.1)

## 分類指標洞察 (Classification Insights)
- 在預測 Down/Up 類別上，模型呈現**極低的 Precision 但極高的 Recall**，這是因為 Stationary (盤整) 樣本佔據壓倒性多數。
- **結論**：這不是一個主動預測方向的交易模型 (Directional Trading Model)，而是極高敏感度的 **toxic-flow 撤單過濾器 (Cancel Filter)**。
""")
    copy_safe(project_root / "meeting/w13/01_docs/train_history_plot.png", d01 / "training_loss_plot.png")
    copy_safe(eval_dir / "confusion_matrix_percent_test.png", d01 / "confusion_matrix_percent_test.png")
    
    config_path = model_dir / "experiment_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        
        split_cfg = cfg.get("split", {})
        train_frac = split_cfg.get("train_session_frac", 0.7)
        val_frac = split_cfg.get("val_session_frac", 0.15)
        test_frac = round(1.0 - train_frac - val_frac, 2)
        
        identity_data = [
            {"Parameter": "Dataset", "Value": "NEW_ALL / TXFB6"},
            {"Parameter": "Data Split", "Value": f"Chronological (Train {train_frac*100}%, Val {val_frac*100}%, Test {test_frac*100}%)"},
            {"Parameter": "Model Architecture", "Value": f"TCN (channels={cfg.get('model', {}).get('channel_list', [16,16,16])}, kernel={cfg.get('model', {}).get('kernel_size', 3)}, dropout={cfg.get('model', {}).get('dropout', 0.1)})"},
            {"Parameter": "Feature Count", "Value": str(cfg.get("features", {}).get("feature_count", 8))},
            {"Parameter": "Z-score Window", "Value": "2000"},
            {"Parameter": "Lookback Window (Ticks)", "Value": str(cfg.get("window", {}).get("window_size", 100))},
            {"Parameter": "Prediction Horizon (Ticks)", "Value": str(cfg.get("window", {}).get("horizon", 5))},
            {"Parameter": "Label Type", "Value": cfg.get("labels", {}).get("label_type", "smoothed_mid_move_3class")},
            {"Parameter": "Default Threshold", "Value": "0.5"}
        ]
        df_identity = pd.DataFrame(identity_data)
        safe_to_csv(df_identity, d01 / "model_identity.csv")
    else:
        df_identity = pd.DataFrame([
            {"Parameter": "Dataset", "Value": "NEW_ALL / TXFB6"},
            {"Parameter": "Data Split", "Value": "Chronological (Train 70.0%, Val 15.0%, Test 15.0%)"},
            {"Parameter": "Model Architecture", "Value": "TCN (channels=[16,16,16], kernel=3, dropout=0.1)"},
            {"Parameter": "Feature Count", "Value": "8"},
            {"Parameter": "Z-score Window", "Value": "2000"},
            {"Parameter": "Lookback Window (Ticks)", "Value": "100"},
            {"Parameter": "Prediction Horizon (Ticks)", "Value": "5"},
            {"Parameter": "Label Type", "Value": "smoothed_mid_move_3class"},
            {"Parameter": "Default Threshold", "Value": "0.5"}
        ])
        safe_to_csv(df_identity, d01 / "model_identity.csv")
        
    cr_path = eval_dir / "classification_report_test.csv"
    if cr_path.exists():
        cr_df = pd.read_csv(cr_path)
        col_name = cr_df.columns[0]
        classes_df = cr_df[cr_df[col_name].astype(str).isin(['0', '1', '2'])].copy()
        classes_df[col_name] = classes_df[col_name].astype(str).map({'0':'Down', '1':'Stationary', '2':'Up'})
        classes_df = classes_df.rename(columns={col_name: "Class"})
        safe_to_csv(classes_df, d01 / "classification_summary.csv")
    
    # 02 Backtest
    d02 = out_dir / "02_backtest_strategy_result"
    d02.mkdir(exist_ok=True)
    write_md(d02 / "slide.md", """# 02 回測策略與結果 (Maker-only Cancel Filter Backtest)

## 策略邏輯 (Strategy Logic)
- **被動造市 (Maker-only)**：預設掛單在最佳 Bid/Ask。
- **撤單邏輯**：`P(Down) > threshold` → 撤銷 Bid | `P(Up) > threshold` → 撤銷 Ask。
- **參數設定**：Maker Fee = 15 TWD, Taker Fee = 50 TWD, Point Value = 200, Max Inventory = 3, allow_active_alpha = false。

## 關鍵結果 (Results)
- 評估的 Threshold：0.55 / 0.60 (Test-sweep candidates), 0.70 (Validation-selected)。
- **重點洞察**：0.55 與 0.60 在 NEW test split 中相較於無模型與隨機撤單 baseline 有著明顯的 local edge。
- **注意事項**：0.70 尚未顯著擊敗 `Random+1std`，因此 0.55 不能宣稱為 production threshold，策略尚不可稱為 production-ready。
""")
    base_path = diag_dir / "baseline_summary_fixed.csv"
    if base_path.exists():
        df_base = pd.read_csv(base_path)
        if "threshold" in df_base.columns:
            df_base = df_base[df_base["threshold"].isin([0.55, 0.60, 0.70])].copy()
            df_base = df_base.round(2)
            safe_to_csv(df_base, d02 / "baseline_comparison.csv")
        else:
            copy_safe(base_path, d02 / "baseline_comparison.csv")
            
    thr_path = diag_dir / "threshold_sweep_enhanced.csv"
    if thr_path.exists():
        df_thr = pd.read_csv(thr_path)
        df_thr = df_thr[["threshold", "signal_rate", "trade_count", "net_pnl", "toxic_fill_avoided", "max_drawdown"]].round(4)
        safe_to_csv(df_thr, d02 / "threshold_result.csv")
        
    opp_path = diag_dir / "missed_opportunity_diagnostics_fixed.csv"
    if opp_path.exists():
        df_opp = pd.read_csv(opp_path)
        if "threshold" in df_opp.columns:
            df_opp = df_opp[df_opp["threshold"].isin([0.55, 0.60, 0.70])].copy()
        safe_to_csv(df_opp, d02 / "opportunity_cost.csv")
    
    # 03 Regime Motivation
    d03 = out_dir / "03_regime_motivation"
    d03.mkdir(exist_ok=True)
    write_md(d03 / "slide.md", """# 03 為何需要 Regime Gate? (Why Regime Gate Is Needed)

## 原始微觀模型的極限 (Micro vs Macro)
- TCN (window=100) 只能處理微觀的 adverse selection (毒性訂單)。
- **NEW3 壓力測試原始結果**：Model PnL (-551,705) vs Baseline (-557,165)，改善僅微幅 +5,460。
- **結論**：微觀的 cancel filter 無法拯救宏觀的 regime failure。Maker-only 策略在強烈趨勢下會如同接刀，累積致命的 inventory risk。

**重點提醒**：NEW3 是一個絕佳的壓力測試市場狀態 (Stress Regime)，不能錯誤地將其排除，而是需要事前辨識的 causal gate 來防禦。
""")
    new_audit = diag_v3_dir / "new_session_audit.csv"
    new3_audit = diag_v3_dir / "new3_session_audit.csv"
    if new_audit.exists() and new3_audit.exists():
        df_new = pd.read_csv(new_audit)
        df_new3 = pd.read_csv(new3_audit)
        summary = pd.DataFrame([
            {"Dataset": "NEW", "Session Count": len(df_new), "Avg Trend Strength": df_new["trend_strength"].mean(), "Avg Volatility": df_new["realized_vol_ticks"].mean(), "Avg Baseline PnL": df_new["baseline_net_pnl"].mean()},
            {"Dataset": "NEW3", "Session Count": len(df_new3), "Avg Trend Strength": df_new3["trend_strength"].mean(), "Avg Volatility": df_new3["realized_vol_ticks"].mean(), "Avg Baseline PnL": df_new3["baseline_net_pnl"].mean()}
        ]).round(2)
        safe_to_csv(summary, d03 / "new_vs_new3_regime_summary.csv")
        
        df_new3_worst = df_new3.sort_values("baseline_net_pnl").head(5)
        df_new3_worst = df_new3_worst[["session_id", "trend_strength", "realized_vol_ticks", "baseline_net_pnl", "trend_regime"]]
        safe_to_csv(df_new3_worst, d03 / "new3_failure_summary.csv")

    # 04 Regime Added
    d04 = out_dir / "04_regime_added_result"
    d04.mkdir(exist_ok=True)
    write_md(d04 / "slide.md", """# 04 加入 Regime Gate 的結果 (Regime Gate Added)

## 實作與成效 (Implementation & Results)
- **目前實作**：Rolling trend + Rolling volatility + Spread filter (註：Kalman/EWMA 為 Future Work)。
- **NEW 最佳 Gate**：`spread_filter + TCN 0.55` (Net PnL 約 101,455，Active Event Rate 約 82.4%)。
- **NEW3 最佳 Gate**：`combined_filter + TCN 0.55` (Net PnL 約 72,520，Active Event Rate 約 82.6%)。

## 關鍵洞察 (Key Takeaways)
- **並非因減少交易次數而獲利**：Active rate 仍維持在 82% 左右的高水準。
- **Regime 的獨立性**：NEW 和 NEW3 分別需要不同的最佳 Gate，這是 Regime Dependency 的直接證據！
- **下一步**：目前尚未做到 production-ready 的 auto-gating，接下來需要開發 Causal Auto-gate Selector。
""")
    gate_path = diag_dir / "regime_gate_sweep.csv"
    if gate_path.exists():
        df_gate = pd.read_csv(gate_path)
        best_gates = []
        for dataset in df_gate["dataset"].unique():
            df_ds = df_gate[df_gate["dataset"] == dataset]
            best_row = df_ds.loc[df_ds["net_pnl"].idxmax()]
            best_gates.append(best_row)
        df_best = pd.DataFrame(best_gates)
        df_best = df_best[["dataset", "gate_type", "model_threshold", "net_pnl", "active_event_rate"]].round(4)
        safe_to_csv(df_best, d04 / "regime_gate_best_by_dataset.csv")
    
    # 05 New Findings
    d05 = out_dir / "05_new_findings"
    d05.mkdir(exist_ok=True)
    write_md(d05 / "slide.md", """# 05 新發現：Label 與 Latency 敏感度 (New Findings)

## A. Label 的發現
- **舊版 Label**：已經測試過 `toxic_bid/ask`，其 Macro F1 約為 0.5，但 PnL 卻是 0 或負數，這代表不需要再重試舊版 Label。
- **新方向 (Fill-conditional)**：建立 fill-conditional quote-risk label。根據 Proxy C 的審查，conditional toxic rate 高達數個 percent，明顯高於 unconditional 的 0.3%，展現出強烈潛力。
- **限制聲明**：這是基於 proxy 的分析，並非來自真實的 exchange fill（因缺乏 aggressor-side 數據）。

## B. Latency 的發現
- **延遲與虧損**：Cancel delay 會導致 delayed toxic loss 增加，但整體 Net PnL 的衰退是非單調的 (non-monotonic)。
- **重大衰退點**：當 Delay=10 時，表現明顯變差。
- **FPGA 加速的定位**：我們採用 FPGA 的合理性，是基於「Risk-control-oriented (風險控制)」，防堵延遲造成的巨額虧損，而非純粹為了證明 PnL 的絕對必要性。
""")
    lat_path = diag_dir / "latency_sensitivity_enhanced.csv"
    if lat_path.exists():
        df_lat = pd.read_csv(lat_path)
        df_lat = df_lat.groupby("cancel_delay_events")[["net_pnl", "toxic_fill_avoided_count"]].mean().round(2).reset_index()
        safe_to_csv(df_lat, d05 / "latency_summary.csv")
        
    old_new_path = diag_v3_dir / "side_label_unconditional_vs_fill_conditional.csv"
    if old_new_path.exists():
        df_old_new = pd.read_csv(old_new_path)
        df_old_new = df_old_new.round(4)
        safe_to_csv(df_old_new, d05 / "label_old_vs_new.csv")
        
    fill_path = diag_v3_dir / "side_label_distribution_fill_conditional.csv"
    if fill_path.exists():
        df_fill = pd.read_csv(fill_path)
        df_fill = df_fill.round(4)
        safe_to_csv(df_fill, d05 / "fill_conditional_summary.csv")

    # 06 Next Week
    d06 = out_dir / "06_next_week_todos"
    d06.mkdir(exist_ok=True)
    write_md(d06 / "slide.md", """# 06 下週行動事項 (Next Week Todos)

## 1. New Binary Quote-Risk Model
- **不再重試舊版 toxic labels**，而是建立 fill-conditional bid/ask quote-risk label。
- 訓練小型 Binary TCN 模型。
- 保留完整的 Loss curve, PR-AUC, Confusion Matrix 與 Threshold diagnostics。
- 與現有的 `smoothed_mid_move_3class` baseline 進行最公平的比較。

## 2. Regime Auto-Gating
- 驗證能否使用 causal rolling features 自動選擇最佳的 Filter (`spread_filter`, `combined_filter`, 或是 `risk-off`)。
- **不使用** full-session future trend，並確保**不在 NEW3 上調參**。
- 將 no-gate, always-spread, always-combined 與 auto-gate 的表現進行橫向對比。

## 3. FPGA MVP Path
- 若模型的 Edge (優勢) 不夠，將優先實作 rolling/EWMA gate 搭配 tiny TCN cancel filter 作為 MVP (最小可行性產品)。
- FPGA 的終極目標在於：降低 Cancel Latency 並消除 Delayed Toxic Loss。
""")
    write_md(d06 / "todo_table.csv", "Task,Priority,Description\nQuote-Risk Model,High,Binary TCN with conditional labels\nAuto-Gating,High,Causal rolling regime selector\nFPGA MVP,Medium,Baseline implementation for latency mitigation")
    write_md(d06 / "decision_tree.csv", "Decision,Condition,Action\nEdge Weak,True,Fallback to tiny TCN MVP\nRegime Shift,True,Trigger Auto-Gating")

    # Speaker Script
    write_md(out_dir / "speaker_script.md", """# Week 13 講者逐字稿 (Speaker Script)

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
""")

    # Deck Outline
    write_md(out_dir / "deck_outline.md", """# Week 13 簡報大綱 (Deck Outline)

1. **00 模型選擇過程**：為什麼最後選 smoothed_mid_move_3class。
2. **01 最終選擇的模型**：架構定義與 Cancel Filter 的重新定調。
3. **02 回測策略與結果**：Maker-only 撤單邏輯，以及 0.55 具備的 local edge。
4. **03 為什麼需要 Regime**：微觀模型在 NEW3 強趨勢下的崩潰證明。
5. **04 Regime 加入與成效**：Spread filter 展現的 Regime Dependency。
6. **05 新發現**：Fill-conditional label 潛力與 Latency 對虧損的影響。
7. **06 下週行動事項**：Binary 模型、Auto-Gating、FPGA MVP。
""")

    # Claims to Avoid
    write_md(out_dir / "claims_to_avoid.md", """# 不能過度宣稱的避雷清單 (Claims to Avoid)

- **絕對不要說** 模型是 directional trader (主動方向性交易)。
- **絕對不要說** 0.55 已經是 production threshold (它只是 test-sweep candidate)。
- **絕對不要說** 策略目前已經 production-ready。
- **絕對不要說** Delay > 3 ticks 策略就會完全崩盤 (正確說法：Net PnL 非單調，在 delay=10 時明顯變差)。
- **絕對不要說** FPGA 已經被完全證明在絕對獲利上的必要性 (正確說法：FPGA 是 risk-control-oriented)。
- **絕對不要說** fill-conditional label 是真實的交易所 fill (正確說法：基於 proxy 且無 aggressor-side data)。
- **絕對不要說** 舊版的 toxic labels 還有重試的價值。
- **絕對不要說** NEW3 表現不好所以被排除不算 (正確說法：NEW3 是 stress regime，需要 causal gate)。
""")

if __name__ == "__main__":
    generate_deck()
    print("Deck generated successfully.")
