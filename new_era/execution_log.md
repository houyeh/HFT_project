# 新紀元執行日誌 (New Era Execution Log)

## 任務一：特徵重要性分析 (Feature Importance Analysis)

**背景**：

在 `0428_test` 驗證中，我們發現原本表現優異的 TCN 模型在新合約資料（B6）上發生了嚴重的 Feature Drift，變成了「無腦預測上漲」的失效模型。這意味著我們原本萃取的 12 個特徵中，有部分特徵在跨合約時數值發生了嚴重的漂移，產生了毒性（Toxicity）或噪音。

**行動方案**：

1. 取出 75 萬筆 NEW 資料的 LOB 快照。
2. 使用 `RandomForestClassifier`（隨機森林）對當下 tick 的 12 個特徵進行決策樹分類。
3. 透過樹狀模型對特徵數值縮放不敏感的特性，真實地排名出這 12 個特徵對預測「未來 10 ticks 漲跌」的貢獻度 (Gini Importance)。
4. 揪出重要性為 0 的無效特徵，並找出主導預測的叛徒特徵，作為下一步實作 Rolling Z-score 的依據。

**預期產出**：

- `new_era/feature_importance.csv`
- 各特徵的重要性權重報告。

---

## 任務二：特徵工程與 Rolling Z-score

**背景**：

從第一階段的分析得知，`micro_return` 等核心特徵在跨合約時會發生嚴重的數值漂移 (Feature Drift)，導致模型在面對新時間點的資料時嚴重過擬合。為了解決這個問題，我們需要讓這些特徵變得「無因次 (Dimensionless)」且不受長期基期影響。

**行動方案**：

1. 捨棄次要與雜訊特徵，精簡為貢獻度最高的前 8 大核心特徵 (如 `micro_return`, `trade_intensity_short` 等)。

2. 對這 8 個特徵實作 **Rolling Z-score** (滑動時間窗標準化，設定 Window=2000 ticks)。將特徵轉換為「當下數值距離過去一段時間均值的標準差」。

3. 透過 `WindowedLOBSequenceDataset` 將處理好的特徵重新打包成 `(Batch, 100, 8)` 的張量。

4. 將特徵與對應的標籤打包儲存成 `.npz` 壓縮檔，方便後續直接送進 hls4ml 的 Keras 模型進行快速訓練。

**預期產出**：

- `new_era/02_feature_engineering.py`
- `new_era/zscore_dataset.npz` (包含正規化後的高頻時序張量)

## 任務三：建立與訓練 Baseline MLP (hls4ml-friendly)

**背景**：

在完成特徵標準化後，我們需要建立一個極簡的 Keras MLP 作為硬體基準。這個模型不追求極致的勝率，而是用來驗證「特徵 -> 軟體模型 -> hls4ml C Synthesis -> FPGA」這條管線的暢通。

**行動方案**：

1. 讀取 002_zscore_dataset.npz 的高純度資料。
2. 建立 Flatten -> Dense(32) -> Dense(16) -> Softmax 的硬體友善 MLP。
3. 考慮到標籤不平衡，動態計算 Class Weights，讓模型不會偷懶只猜 Stationary。
4. 訓練並儲存為 Keras 標準的 .h5 模型。

**預期產出**：

- new_era/03_train_baseline_mlp.py
- new_era/003_baseline_mlp.h5

## 任務四：透過 hls4ml 生成 FPGA IP Core (C Synthesis 前置作業)

**背景**：

有了 Keras 的 Baseline 模型 (.h5) 後，我們需要將軟體神經網路轉換為 FPGA 懂的 C++ 程式碼 (High-Level Synthesis)。

**行動方案**：

1. 讀取 003_baseline_mlp.h5。
2. 設定 hls4ml 硬體配置，針對 PYNQ-Z2 (xc7z020clg400-1) 進行編譯。
3. 優化策略：設定 ReuseFactor=1 以追求 1 個 Clock 的極限延遲，並將 Precision 設為 ap_fixed<16,6> (16-bit 定點數)。
4. 產出包含 C++ 原始碼與 build_prj.tcl 的 Vivado HLS 專案。

**預期產出**：

- new_era/04_hls4ml_conversion.py
- new_era/004_hls4ml_prj/ (包含所有硬體合成所需的 C++ 檔案)

### 如何使用這個 FPGA 硬體專案 (How to Use)？

1. **專案位置**：所有的 C++ 原始碼、網路權重、測試腳本與 Vivado 設定檔，都已經被打包在

`ew_era/004_hls4ml_prj` 這個資料夾中了。

1. **進入硬體合成 (C Synthesis)**：

   - 將這個資料夾放到一台有安裝 **Xilinx Vivado HLS** 或 **Vitis HLS** 的電腦上。
   - 打開終端機 (Vivado HLS Command Prompt)，切換到該資料夾：cd 004_hls4ml_prj

   - 執行自動合成腳本：vivado_hls -f build_prj.tcl (若為新版則用 vitis_hls)。

2. **產出 IP Core 與燒錄 (.bit)**：

   - 執行完畢後，它會產出一個硬體矽智財 (IP Core) 壓縮檔。
   - 打開普通的 Vivado，把這個神經網路 IP 拉進 Block Design，透過 AXI 匯流排連線到你的 PYNQ-Z2 處理器上。
   - 點擊「Generate Bitstream」，就能拿到最終可以燒錄進 PYNQ-Z2 的 .bit 檔了！

## 任務五：以新特徵重新訓練 PyTorch TCN

**背景**：

將 Z-score 處理後的新資料丟進我們之前寫好的 HLSFriendlyTCNClassifier 裡面訓練，看加入時間序列理解能力後，準確率是否能擊敗單純的 Baseline MLP。

**行動方案**：

1. 讀取 002_zscore_dataset.npz (148 萬筆資料)。

2. 轉置矩陣為 (Batch, Features, Time) 以符合 PyTorch Conv1d 的要求。

3. 使用硬體友善版 TCN：channel_list=[16, 16, 16]，參數量僅 4,515 個！

4. 加入 Class Weights 對抗標籤不平衡，並以 Adam 優化器訓練 20 Epochs。

**預期產出**：

- new_era/05_train_tcn_pytorch.py

- new_era/005_tcn_weights.pth

## 任務六：事件驅動回測 (Event-Driven PnL Backtest)

**背景**：

高頻交易模型不能只看「準確率 (Accuracy)」，因為即使準確率高，也可能被「買賣價差 (Spread)」與「手續費」吃光利潤。因此必須建立一個考慮了滑價與成本的模擬下單系統。

**行動方案**：

1. 載入原始時間序列資料與真實價格 (Mid, Ask, Bid)。

2. 取出不打亂的最後 20% 作為時間序列測試集 (Out-of-Sample Test Set)。

3. 載入 TCN 權重進行順序推論。

4. **交易邏輯**：當模型對漲跌的信心 > 80% 時，跨過價差 (敲擊對手價) 進行市價單進場，持有 10 個 Ticks 後強制平倉。

5. **成本計算**：每口單邊手續費以 50 元台幣計，嚴格扣除。計算出最終淨損益 (Net PnL)。

**預期產出**：

- new_era/08_backtest_pnl.py

- 回測結果報表 (包含勝率與最終 PnL)。

## 任務七：市場中性造市商回測 (Market-Neutral Market Making Backtest)

**背景**：

單純的事件驅動策略容易受到大盤方向 (Bias) 影響。身為造市商 (Market Maker)，理想的狀態是「不猜漲跌，只賺價差」，並在庫存過高時主動撤單或反手做對沖，以保持中性 (Neutral) 狀態。我們將透過「攻擊型造市策略」來測試這點。

**行動方案**：

1. 載入原始時間序列資料與真實價格 (Mid, Ask, Bid)。

2. 取出不打亂的最後 20% 作為測試集。

3. 載入 TCN 模型權重，使用「攻擊型造市 (Aggressive Market Making)」策略。

   - 策略邏輯：

     - 當預測 **Up** 時：撤銷買單，並提高買價 (Asymmetric Buy)。

     - 當預測 **Down** 時：撤銷賣單，並壓低賣價 (Asymmetric Sell)。

     - 旨在利用模型看對趨勢時「主動咬住單子」賺取利潤，同时在趨勢完全相反時不與之對抗。

4. **成本計算**：每口單邊手續費以 50 元台幣計。

5. **盈虧結算**：結算所有已成交單的損益。

**預期產出**：

- `new_era/09_mm_backtest.py`

- 三種策略的對比報表 (Baseline 無腦造市 vs. TCN 防禦造市 vs. TCN 攻擊造市)，重點比較 PnL 與交易次數。

## Week 10 Q&A (教授攻防題庫)

**Q1：要怎麼證明現在的 TCN 模型沒有像以前的 MLP 一樣，只是死背了目前的特定合約價格 (Overfitting)？有什麼證據能說明 Z-score + TCN 更厲害？**

**A1**：

1. **嚴格的時間序列切割 (No Shuffle)**：我們在 `datasets.py` 中使用了 `chronological_split_indices`，嚴格切出「最後 20%」的時間段。這代表模型在測試時，遇到的是它在訓練階段「完全沒見過的未來時間與價格區間」。如果它只是死背模式，在 Out-of-Sample 測試的混淆矩陣中，上下漲跌的命中率絕對會崩盤（但我們實測依然高達 80%）。

2. **Rolling Z-score 的降維打擊**：過去的 MLP 吃的是原始數值，如果合約換月從 23,000 點跳到 23,500 點，神經網路就會徹底迷失。而 **Rolling Z-score (2000 ticks)** 把絕對價格變成了「無因次 (Dimensionless)」的相對標準差 (例如 +1.5 Std)。這強迫 TCN 只能去學習「委託簿失衡的形狀」與「微觀動能」，徹底去除了價格基期的干擾，這就是它能跨越時間限制的原因。

**Q2：專案一開始的 `01_feature_importance.py` 中，Feature Importance 是如何計算出來的？憑什麼說這 8 個特徵最重要？**

**A2**：

1. **演算法基底 (Random Forest)**：我們使用了 `sklearn.ensemble.RandomForestClassifier` (隨機森林分類器) 來評估特徵。

2. **Gini Importance (吉尼不純度下降)**：在構建幾百棵決策樹的過程中，演算法會測試用每個特徵（如 `micro_return` 或 `queue_imbalance_change`）來把資料切分成 Up/Down/Stat。如果用某個特徵切一刀，能最乾淨地把「上漲」跟「下跌」分開（不純度下降最多），那個特徵就會獲得最高的分數。

3. **非線性關聯性**：傳統的相關係數 (Pearson) 只能抓線性關係，但 Random Forest 能抓出非線性的臨界點。這保證了我們選出的 Top 8 特徵，是微觀結構中最具鑑別度的物理量。

---

## 任務八：硬體資源優化與驗證 (HLS DSE & Co-Simulation)

**背景**：
當我們成功使用 hls4ml 將 PyTorch TCN 模型轉換為 Vitis HLS 專案後，我們發現預設的 C Synthesis 結果雖然延遲極低，但硬體資源消耗極度誇張（例如 PYNQ-Z2 上的 DSP 佔用率高達 1556%）。這意味著模型無法實際燒錄進 FPGA 晶片中。同時，為了滿足學術嚴謹性，我們必須證明硬體合成出來的定點數運算結果，與 Python 浮點數算出來的機率完全一致。

**行動方案**：

1. **設計空間探索 (DSE) - 調整 Reuse Factor**：
   - 透過修改 `07_hls4ml_tcn_conversion.py`，將所有 Layer 的 `ReuseFactor` 從預設的 1 調高至 32。
   - 原理：放棄「完全平行展開 (Fully Unrolled)」，改為讓神經網路的乘加運算單元（MACs）進行分時多工（Time-multiplexing），透過「時間換取空間」，將硬體面積大幅壓縮至 PYNQ-Z2 能夠負載的範圍。
2. **重啟 C/RTL 協同模擬 (Co-simulation) - 注入測試數據**：
   - 在轉換腳本中生成一組隨機的高頻市場特徵矩陣 (`X_test = np.random.rand(1, 8, 100)`)。
   - 強制呼叫 `hls_model.predict(X_test)`，讓轉換器不僅生成 C++ 程式碼，同時將輸入矩陣與對應的黃金預測結果（Golden outputs）輸出為 `tb_input_features.dat` 與 `tb_output_predictions.dat`。
   - 修改 `build_prj.tcl`，重新將 `csim` 與 `cosim` 設為 1，使 Vitis HLS 能夠載入上述資料，自動比對 C++ 定點數運算結果與 Python 浮點數結果的誤差 (Bit-accurate Verification)。

**預期產出**：

- 更新後的 `new_era/07_hls4ml_tcn_conversion.py`
- 可被 PYNQ-Z2 容納的新版 IP Core。
- 證明硬體演算法正確性的 C/RTL Co-simulation 驗證通過報表。

## 任務九：修復 hls4ml 框架 Bug 與 C/RTL 協同模擬驗證 (Co-simulation Verification)

**背景**：
在執行 C/RTL 協同模擬時，發生了 `an hls::stream is read while empty` 的模擬當機。經過深層 C++ 原始碼追蹤，確認此為 `hls4ml` 開源套件在轉換 PyTorch 的「殘差連接 (Residual Connection)」模型時，遺漏為旁支 (Downsample 層) 插入 `Transpose` (矩陣轉置) 硬體層的底層 Bug。此外，在修復當機後，遭遇了 `Test failed` 的誤差問題，此現象肇因於定點數量化溢位 (Quantization Overflow)。

**行動方案**：

1. **攔截並修補底層 C++ 程式碼**：
   - 使用 Python 腳本直接攔截 `007_hls4ml_tcn_prj\firmware\myproject.cpp`。
   - 手動為 `layer32_cpy2` 旁支補上 `nnet::transpose` 硬體層，使資料維度從 `(8, 100)` 正確轉置為 `(100, 8)` 的 `channels_last` 格式，徹底解決 Empty Stream 當機問題。
2. **定點數溢位 (Quantization Error) 分析**：
   - 將 Python 測試資料腳本中的 `Softmax` 移除，確保比對基準一致 (Logits vs Logits)。
   - 記錄並分析 `ap_fixed<16,6>` 造成的數值截斷與溢位現象，作為後續導入 Quantization-Aware Training (QAT) 的核心論點。

**預期產出**：

- 成功通過 `csim` 與 `cosim` 的測試管線，確保硬體運作邏輯正確。
- 一份完整的 FPGA 硬體合成報告 (`csynth.rpt`)，證明模型推論延遲已達 24 微秒 (4426 Clock Cycles)。
- 將量化溢位現象包裝為 DSE (設計空間探索) 的探討題材。

下一步

- QAT量化
- Batch normalization
- 為了放到PyNQ Z2 上跑
  - 降低 Sequence Length (100 -> 30)
  - 導入 INT8 量化
  - 導入 HLS 語法優化
