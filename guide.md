# HFT 專題規劃

## 🚀 專題計畫書：基於 FPGA 硬體加速之跨資產高頻微觀結構通用預測模型(Universal Microstructure Prediction across Asset Classes with FPGA Hardware Acceleration)📍 總體目標 (Project Objectives)本專題旨在探討並解決高頻交易（HFT）中「AI 預測精準度」與「硬體推論延遲」之間的極限權衡（Trade-off）

## 財工端（通用泛化模型）： 摒棄傳統單一商品的過擬合訓練，提取無因次（Dimensionless）微觀結構特徵，結合時間卷積網路（TCN），打造能跨資產（如股指、原油）泛化的微秒級方向預測模型

## 硬體端（極低延遲架構）： 將 PyTorch 模型進行 INT8 量化與 C++ 底層重構，導入 Vitis HLS 進行管線化與記憶體架構優化（Shift Register），達成 50 微秒內的極致推論延遲

## 回測端（延遲敏感度分析）： 建立事件驅動回測引擎，量化「硬體加速帶來的物理時間優勢」如何降低逆向選擇與滑價成本，實質提升策略夏普值

## 專案導覽（2026-04 結構）

目前專案的主模組維持穩定語意：

- `src/`：核心研究與策略邏輯
- `cpp_hls_prep/`：C++ / HLS handoff 與 testbench
- `artifacts/`：實驗與硬體交付產出
- `meeting/`：週會材料
- `teaching/`：教學與方法論文件

而 `scripts/` 已改成依照使用時機排序：

- `scripts/01_data_prep/`：資料前處理與 mock data
- `scripts/02_train_eval/`：訓練與直接驗證
- `scripts/03_experiments/`：sweep 與策略實驗
- `scripts/04_analysis/`：分析、模擬、延遲估算
- `scripts/05_hardware_handoff/`：INT8 匯出與硬體驗證
- `scripts/06_reporting/`：圖表、報告、投影片

如果要從頭理解專案，建議先看：

1. `scripts/02_train_eval/train_tw_baseline.py`
2. `scripts/03_experiments/sweep_latency.py`
3. `scripts/05_hardware_handoff/export_tcn_int8_artifacts.py`
4. `cpp_hls_prep/`

## 📊 資料規格需求 (Data Requirements)資料源： 芝加哥商品交易所（CME）歷史資料

### 標的清單： E-mini S&P 500 (ES)、E-mini NASDAQ (NQ)、Crude Oil (CL)

### 資料層級： Level-2 Market-by-Price (MBP) Order Book Snapshots，至少 5 檔深度

### 時間解析度： Tick-level（逐筆更新），時間戳記精確至奈秒（ns）

### 時間跨度： 1 至 3 個月的連續交易日

## 🟢 第一階段：通用特徵工程與 TCN 架構設計 (軟體與財工期)目標：處理高頻資料，萃取跨資產通用特徵，並完成 TCN 基準模型訓練

### Week 1-2 (Sprint Week)：資料清理與無因次特徵工程

核心任務：讀取 ES, NQ, CL 的 Tick-level LOB 資料，處理時間對齊與缺失值。
實作向量化運算，計算跨資產通用的「無因次特徵」：
正規化價差：$Spread_{norm} = \frac{Ask_1 - Bid_1}{Tick\_Size}$
微觀動能報酬：$Return_{mid} = \frac{Mid_t - Mid_{t-1}}{Tick\_Size}$
委託簿失衡：$OBI = \frac{V_{bid1} - V_{ask1}}{V_{bid1} + V_{ask1}}$
建立滑動視窗（Sliding Window），將 2D 表格轉換為 3D 張量 (Batch, Sequence, Features)，並標記未來 N 個 Tick 的價格方向（漲、平、跌）。

檢核點 (Deliverables)： 產出可用於訓練的三維特徵矩陣與特徵相關性熱力圖。

### Week 3：硬體友善型 TCN 模型架構設計

核心任務：放棄傳統 RNN/LSTM，設計基於 PyTorch 的 時間卷積網路 (TCN)。導入「膨脹因果卷積（Dilated Causal Convolution）」，在不增加乘加運算（MAC）數量的前提下，擴大模型的感受野（Receptive Field），以符合 FPGA 的嚴苛資源限制。

檢核點 (Deliverables)： 完成 model.py 架構設計，並通過 Dummy Tensor 的 Forward Pass 測試。

### Week 4：跨資產模型訓練與 Baseline 驗證

核心任務：建立 Training Loop，混合 ES 與 NQ 的資料進行模型訓練。使用 Validation Set 微調超參數，防止過擬合。

檢核點 (Deliverables)： 產出訓練 Loss 曲線與混淆矩陣，確立 FP32 模型的最高預測準確率（Baseline Accuracy）。

## 🟡 第二階段：模型量化與 C++ 核心重構 (軟硬體交界期)目標：脫離 Python 環境，將模型轉換為硬體能理解的 INT8 定點數格式與 C++ 底層語言

### Week 5：訓練後量化 (Post-Training Quantization, PTQ)

核心任務：實作 BatchNorm 摺疊（Folding），將 BN 層參數吸收進前一層的 Convolution 權重中，消除硬體執行時的除法運算。將 FP32 的權重與激勵函數量化為 INT8，並評估精準度損失（Accuracy Drop）。

檢核點 (Deliverables)： FP32 vs. INT8 模型的記憶體佔用與準確率 Trade-off 比較表。

### Week 6-7：C++ 手刻 TCN 核心運算

核心任務：建立 C++ 專案，讀取 INT8 權重。不依賴任何外部數學庫（如 Eigen），純手工以 C++ 實作 1D 卷積、ReLU 與全連接層矩陣乘法。串接完整的 Inference 前向傳播流程。

檢核點 (Deliverables)： 成功編譯並可於終端機執行的 C++ 推論程式。

prompt tried: `檢視我現在的計畫，手刻C++對於我這個專題以及未來履歷的優勢在哪？難度大嗎(從沒做過用C++做HLS)？可以直接將現在的東西丟給hls4ml嗎`

### Week 8：Bit-accurate (位元級精確) 驗證

核心任務：將相同的測試資料分別輸入 Python INT8 模型與 C++ 模型。逐筆比對輸出結果，確保軟體與 C++ 邏輯達到 100% 位元級別的一致。

檢核點 (Deliverables)： 零誤差驗證通過報告（進入 HLS 前的最終防線）。

## 🟠 第三階段：hls4ml 導入與設計空間探索 (硬體期) 目標：為迎戰 FPGA 競賽，導入 hls4ml 自動合成工具，並透過 MLP 與 TCN 的對比，建立完整的「準確率 vs 延遲」權衡論述

### Week 9：建立 hls4ml-friendly 基準模型與隨插即用管線 (Pipeline Template)

核心任務：暫時放下複雜的 TCN，因應嚴重過擬合問題，先實作一個極簡的全連接層 (MLP) 作為防禦性基準。更重要的是建立「硬體自動化管線」：只要模型能匯出特定格式（如 Keras h5 或 ONNX），管線就能自動透過 hls4ml 轉成 IP Core 並生成 Bitstream。讓未來的模型都能「隨插即用 (Plug-and-Play)」。

檢核點 (Deliverables)： MLP 基準模型的 Synthesis Report，以及一套可重複使用的 FPGA 硬體轉換腳本 (Template)。

### Week 10：TCN 架構重構與滾動式訓練 (Walk-Forward Optimization)

核心任務：解決前期 hls4ml 報錯的問題，將 TCN 移植至 Keras 以原生支援 Causal Padding。同時，為解決 TCN 對特定合約過擬合的問題，導入「滾動式訓練 (Walk-Forward) 與跨月驗證」原則。在軟體端持續吸納新資料訓練直到 PnL 收斂穩定；在硬體端則利用 Week 9 建立的管線，將每一次收斂的新模型快速套用模板燒錄至 FPGA 測試。

檢核點 (Deliverables)： 具備時序記憶能力的 Keras TCN 模型，以及解決 Overfitting 的跨月滾動回測報告。

### Week 11：設計空間探索 (Design Space Exploration, DSE)

核心任務：在 hls4ml 參數 (如 Reuse Factor, Precision) 上進行調優。最重要的是，產出 MLP 與 TCN 的「準確率 (Accuracy) vs. 推論延遲 (Latency) vs. 硬體資源消耗 (DSP/LUT)」對比圖。論證在不同交易頻率與策略下，兩種架構的各自優勢。

檢核點 (Deliverables)： 準備用於研究所申請與 FPGA 競賽的「Trade-off 核心視覺化圖表」。

### Week 12：PYNQ-Z2 實體驗證與 Bitstream 燒錄 (Board Deployment)

**目前卡在** 現在的架構太大了，上不了ZYNQ。

核心任務：結束純軟體模擬，進入真機驗證階段。將 hls4ml 產出的 IP Core 匯入 Vivado，透過 AXI DMA 與 Zynq 處理器 (PS) 連接。跑完 Synthesis 與 Implementation 後生成 `.bit` 檔。將硬體設定檔匯入 PYNQ-Z2 開發板，並用 Jupyter Notebook 透過 Python 驅動 FPGA 進行實際推論，測量「真正的物理延遲」。

檢核點 (Deliverables)： PYNQ-Z2 板上執行畫面、Jupyter Notebook 互動推論展示、End-to-End 硬體實際延遲量測數據。

## 🔴 第四階段：泛化回測與延遲敏感度分析 (大乘期)目標：將硬體的微秒級優勢轉化為財務指標，並驗證模型的跨資產能力

### Week 13：事件驅動回測引擎構建

加入解網路/數據拆解的晶片?

核心任務：開發模擬「時間流動」與「委託簿排隊機制（Queue Priority）」的 Python 回測引擎。加入手續費與跳動點價值（Tick Value）計算，設定市價單吃單邏輯。

檢核點 (Deliverables)： 具備完整 Trade Log 紀錄能力的回測系統。

### Week 14：延遲敏感度與滑價分析 (Latency Sensitivity)

核心任務：引入「滑價（Slippage）」與「逆向選擇（Adverse Selection）」模型。模擬並對比兩種情境下的獲利表現：Python 軟體推論延遲（如 2 毫秒）。FPGA 硬體推論延遲（如 50 微秒）。

檢核點 (Deliverables)： 核心視覺化圖表：延遲時間 (X軸) vs. 夏普值/累積報酬 (Y軸) 曲線。

### Week 15：Zero-shot 跨資產泛化驗證

核心任務：將僅用股指（ES, NQ）訓練出的 TCN 模型，直接應用於原油（CL）的資料上進行推論與回測（Zero-shot Inference）。驗證「無因次特徵」在不同市場微觀結構下的通用有效性。

檢核點 (Deliverables)： 泛化能力驗證報告與跨資產績效對比。

### Week 16：最終報告撰寫與答辯準備

核心任務：統整專題的 Trade-off 分析（AI 容量 vs. 硬體資源，運算延遲 vs. 交易滑價）。製作 5-10 分鐘推甄/專題發表簡報。

檢核點 (Deliverables)： 專題結案，產出具備頂級量化所/IC 設計所申請水準的書面論文與簡報。
