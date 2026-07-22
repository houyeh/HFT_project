# 🚀 新人快速上手指南 — HFT × FPGA 專題
> *更新於 2026-04-14 | Week 7 進度快照*

---

## 0. 這個指南的目的

你剛拿到這個專案，想在最短時間內搞清楚：
1. **這個專題在做什麼？主線在哪裡？**
2. **模型是怎麼設計的？怎麼訓練的？**
3. **程式碼架構長什麼樣？**
4. **現在做到哪裡了？下一步是什麼？**

照著這份文件讀，預計 2-4 小時可以掌握全貌。

---

## 1. 專題一句話總結

> **用小型 TCN（時間卷積網路）預測高頻交易中 LOB（委託簿）的短期方向，並把這顆模型從 Python → INT8 C++ → Vitis HLS，讓它能在 FPGA 上以 < 50 微秒的延遲完成推論。**

這個專題同時橫跨三個領域：
- **財工研究**：高頻微觀結構特徵、模型訓練
- **量化回測**：事件驅動模擬、延遲敏感度分析
- **數位 IC 設計**：INT8 量化、C++ 手刻、Vitis HLS 高階合成

---

## 2. 當前主線（Week 7，最優先讀這條）

```
TW replay data (data/raw/TW/replay1_TW.csv)
      ↓
  12 維特徵工程 (src/features/)
  + Sliding Window: 最近 100 events → (100, 12) 樣本
      ↓
  TCNClassifier 訓練 (PyTorch, FP32)
  scripts/02_train_eval/train_tw_baseline.py
      ↓
  INT8 量化 + 權重匯出 (.wts + .hpp)
  scripts/05_hardware_handoff/export_tcn_int8_artifacts.py
      ↓
  C++ fixed-shape 推論核心
  cpp_hls_prep/src/ (fixed_tcn_top.cpp 等)
      ↓
  Vitis HLS csim ✅  →  csynth 🔄進行中
      ↓
  FPGA synthesis report（目標 < 50 µs latency）
```

**次線（暫時不阻塞主線）**：
- CME Logic20 原始資料 → 分商品模型（ES / CL / NQ 各自訓練）

---

## 3. 現在到哪了（Week 7 檢查點）

| 階段 | 狀態 | 關鍵數字 |
|------|------|---------|
| TW replay data pipeline | ✅ 完整 | 台指 replay CSV |
| 12 維特徵計算 | ✅ | execution_aware feature set |
| TCNClassifier 訓練（FP32）| ✅ | 4,771 params |
| INT8 量化 + 權重 header 生成 | ✅ | ~4.8 KB weights |
| C++ fixed-shape TCN | ✅ | fixed_tcn_top |
| CMake 編譯 + HLS testbench PASS | ✅ | max prob diff = 0 |
| Vitis HLS **csim** | ✅ PASS | |
| Vitis HLS **csynth** | 🔄 已可啟動 | 目標 < 50 µs |
| Pragma 優化（pipeline / unroll）| ⬜ 下一步 | |
| CME 分商品模型 | ⬜ 次線 | ES / CL / NQ |

---

## 4. 10 分鐘掌握模型

### 4.1 輸入資料

不是 K 棒，而是 **委託簿事件（LOB snapshot event）**：
- 每次 bid/ask 報價更新 = 一個 event
- 每個樣本 = 最近 **100 個 events** 的視窗
- 每個 event → 計算 **12 維 feature**

→ 每個 sample 形狀：`(100, 12)` → 送進 TCN 時轉成 `(Batch, 12, 100)`

### 4.2 12 維 Feature 是哪些？

| 類別 | Feature |
|------|---------|
| 價格/價差 | `normalized_spread`, `micro_return`, `weighted_mid_deviation` |
| 委託簿不平衡 | `obi`, `deep_obi`, `queue_imbalance`, `queue_imbalance_change` |
| 成交/撤單流 | `aggressor_imbalance_interval`, `rolling_signed_flow_ratio`, `trade_intensity_short`, `net_cancel_pressure`, `cancel_pressure_short` |

關鍵設計：全部特徵都是**無因次（dimensionless）**，原則上不需要 z-score 就可以跨商品比較。
程式碼：`src/features/dimensionless_features.py` + `src/features/execution_features.py`

### 4.3 預測目標（Label）

- **看未來 10 個 events 後的 mid-price**
- 上漲 > 1 tick → `Up`（class 2）
- 下跌 > 1 tick → `Down`（class 0）
- 否則 → `Stationary`（class 1，通常最多）

> 注意：Stationary 類別遠多於 Up/Down，所以 accuracy 高不代表模型好！要看 macro-F1。

### 4.4 TCN 架構（`src/models/tcn.py`）

```
Input (B, 12, 100)  — B=Batch, 12 features, 100 timesteps
  ↓
TemporalBlock × 3（堆疊 3 層）
  Block 0: CausalConv1d(12→16, kernel=3, dilation=1) + Residual
  Block 1: CausalConv1d(16→16, kernel=3, dilation=2) + Residual
  Block 2: CausalConv1d(16→16, kernel=3, dilation=4) + Residual
  ↓
Global Average Pooling → (B, 16)
  ↓
Linear(16 → 3) → softmax → [P(Down), P(Stationary), P(Up)]
```

**關鍵設計：因果卷積（Causal Convolution）**
- 只對時間軸「左邊」做 zero padding → 絕不偷看未來
- dilation 1/2/4 → 感受野 ≈ 29 events（用 3 層就能「看到」29 步歷史）

**模型規格**：
- 參數量：**4,771**（極小，適合 FPGA）
- 每樣本 MACs：460,848
- FP32 → INT8 後：約 4.8 KB 權重

**兩個版本**（同一個檔案）：
- `TCNClassifier`：原版，訓練用，輸入 `(B, T, F)` 會自動 permute
- `HLSFriendlyTCNClassifier`：HLS 友善版，輸入 `(B, F, T)` channel-first，不用 `F.pad`，用 `torch.cat` 做 causal padding

### 4.5 訓練細節

| 設定 | 數值 |
|------|------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 256 |
| Loss | Weighted CrossEntropyLoss |
| Early stopping | patience=2（看 val loss）|
| 通常 epochs | ~6（early stop）|

**Class imbalance 處理**（高頻資料的必要設計）：
1. `class_weight`：依 train set 類別頻率反比計算，讓 Up/Down 在 loss 裡佔更大比重
2. `sample_weight`：dataset 層再加權，可針對非 Stationary 樣本放大影響

---

## 5. 程式碼架構快速導覽

### 5.1 最重要的五個檔案（按優先順序）

| 優先度 | 檔案 | 說明 |
|--------|------|------|
| ⭐⭐⭐ | `src/models/tcn.py` | TCN 模型定義，有完整中文 docstring |
| ⭐⭐⭐ | `scripts/02_train_eval/train_tw_baseline.py` | 主訓練腳本 |
| ⭐⭐⭐ | `scripts/05_hardware_handoff/export_tcn_int8_artifacts.py` | INT8 量化 + 產生 C++ header |
| ⭐⭐ | `cpp_hls_prep/src/fixed_tcn_top.cpp` | HLS top function |
| ⭐⭐ | `cpp_hls_prep/hls/run_vitis_hls_2025_2.ps1` | 一鍵執行 HLS |

### 5.2 目錄樹速查

```
00_HFT_project/
├── src/                       ← 核心研究邏輯
│   ├── models/tcn.py          ← ⭐ TCN 模型（最重要）
│   ├── features/
│   │   ├── dimensionless_features.py  ← OBI, spread 等
│   │   ├── execution_features.py      ← flow, cancel 等
│   │   └── sliding_window.py          ← 100-event window → tensor
│   ├── data/
│   │   ├── tw_replay.py       ← 台指 replay 資料載入
│   │   ├── cme_logic20.py     ← CME Logic20 格式解析
│   │   └── loader.py          ← 通用清理 / forward-fill
│   ├── training/
│   │   ├── datasets.py        ← Dataset + sample weight
│   │   ├── trainer.py         ← Training loop + early stopping
│   │   └── metrics.py         ← F1, confusion matrix
│   └── backtest/
│       ├── event_driven.py    ← 事件驅動回測引擎
│       ├── hybrid_market_maker.py  ← MM 策略（最複雜）
│       └── signals.py         ← Probability gating 邏輯
│
├── scripts/                   ← 可執行入口（依工作流程排序）
│   ├── 01_data_prep/          ← mock data 與前處理
│   ├── 02_train_eval/         ← 訓練與評估  ⭐ 先看這
│   ├── 03_experiments/        ← sweep 實驗
│   ├── 04_analysis/           ← 分析、延遲估算
│   ├── 05_hardware_handoff/   ← INT8 匯出、bit-accurate 驗證 ⭐
│   └── 06_reporting/          ← 圖表、投影片
│
├── cpp_hls_prep/              ← C++ / HLS 主戰場 ⭐
│   ├── include/
│   │   ├── fixed_tcn.hpp
│   │   ├── fixed_tcn_modules.hpp  ← 各模組宣告
│   │   ├── fixed_tcn_top.hpp
│   │   ├── fixed_tcn_host.hpp
│   │   ├── int8_kernels.hpp
│   │   └── generated/
│   │       ├── fixed_tcn_weights.hpp      ← 自動產生的 INT8 權重
│   │       └── fixed_tcn_test_vectors.hpp ← golden input/output
│   ├── src/
│   │   ├── fixed_tcn.cpp          ← 整合版推論核心
│   │   ├── fixed_tcn_modules.cpp  ← causal_conv, GAP, fc 等
│   │   ├── fixed_tcn_top.cpp      ← HLS top function ⭐
│   │   ├── fixed_tcn_host.cpp     ← host-only 載入邏輯（不進 HLS）
│   │   ├── hls_testbench.cpp      ← HLS testbench
│   │   ├── int8_kernels.cpp       ← INT8 卷積核
│   │   └── main.cpp               ← 完整 host 執行入口
│   ├── hls/
│   │   ├── run_hls.tcl            ← Vivado Tcl 腳本
│   │   ├── run_vitis_hls_2025_2.ps1  ← PowerShell 封裝 ⭐
│   │   └── project/fixed_tcn_hls/ ← HLS 專案產物
│   └── build/                     ← CMake 產物
│       ├── tcn_hls_prep.exe
│       └── tcn_hls_testbench.exe
│
├── artifacts/                 ← 實驗產出
│   ├── 03_latency_proof/
│   │   └── latency_proof/
│   │       └── frozen_tcn_best_model.pt  ← ⭐ 主力凍結模型
│   └── 04_hardware_handoff/
│       ├── int8_export/           ← .wts + reference I/O
│       └── latency_estimate/      ← latency_estimate.json
│
├── data/
│   ├── raw/TW/replay1_TW.csv      ← 主訓練資料（台指 replay）
│   ├── raw/CME/                   ← CME 原始資料（次線）
│   └── processed/*.npz            ← mock 資料做的張量
│
├── teaching/                  ← 01 到 22 的教學文件
├── Weekly_log/                ← 逐週進度記錄
└── tests/                     ← pytest 單元測試（51 passed）
```

---

## 6. 建議閱讀路徑（照順序）

### Level 0：背景概念（沒做過 HFT 先看）
1. `teaching/01_High_Frequency_Trading_Basics.md`
2. `teaching/02_Dimensionless_Features.md`
3. `teaching/07_TCN_in_HFT.md`
4. `teaching/08_TCN_Architecture_Deep_Dive.md`

### Level 1：了解模型與訓練
5. `teaching/20_Model_Features_Params_and_Dataflow.md` ← **最完整技術總覽**
6. `src/models/tcn.py`（有詳細中文注釋，從頭讀到尾）
7. `scripts/02_train_eval/train_tw_baseline.py`（訓練流程）

### Level 2：了解硬體化流程
8. `teaching/19_TW_TCN_CPP_to_HLS_Mainline.md`
9. `teaching/21_Hardware_Handoff_hls4ml_CPP_and_Latency.md`
10. `teaching/22_Vitis_HLS_2025_2_Step_By_Step.md` ← **實際操作步驟**

### Level 3：動手跑
11. 按照下面第 7 節的步驟實際操作

---

## 7. 動手跑的最快路徑

### Step A：確認環境

```powershell
cd C:\Users\holbo\00VS_projjects\00_HFT_project
python -m pytest -q    # 應看到 51 passed
```

### Step B：跑訓練（TW Baseline）

```powershell
python scripts\02_train_eval\train_tw_baseline.py
```

### Step C：匯出 INT8 + 產生 C++ headers

```powershell
python scripts\05_hardware_handoff\export_tcn_int8_artifacts.py
```

輸出：
- `artifacts/04_hardware_handoff/int8_export/tcn_int8_weights.wts`
- `cpp_hls_prep/include/generated/fixed_tcn_weights.hpp`
- `cpp_hls_prep/include/generated/fixed_tcn_test_vectors.hpp`

### Step D：在本機 C++ 驗證

```powershell
cmake -S cpp_hls_prep -B cpp_hls_prep/build
cmake --build cpp_hls_prep/build --config Release
.\cpp_hls_prep\build\tcn_hls_testbench.exe
# 應看到: [PASS] HLS testbench matched fixed golden logits/probabilities.
```

### Step E：進 Vitis HLS

```powershell
# Step 1: C simulation（快，驗證邏輯正確性）
.\cpp_hls_prep\hls\run_vitis_hls_2025_2.ps1 -Stage csim -Part xc7z020clg400-1

# Step 2: C synthesis（慢，產生 RTL 與 resource report）
.\cpp_hls_prep\hls\run_vitis_hls_2025_2.ps1 -Stage csynth -Part xc7z020clg400-1
```

Synthesis report 會在：`cpp_hls_prep/hls/project/fixed_tcn_hls/solution1/syn/report/`

---

## 8. 關鍵技術 Q&A

**Q：為什麼不用 LSTM 或 Transformer？**
A：TCN 結構固定（無動態遞迴），Conv1d 在 C++/FPGA 很好對應，推論延遲更可預測。RNN 的 hidden state 在硬體上需要反覆的時序依賴，難以 pipeline。

**Q：INT8 量化是怎麼做的？**
A：Post-Training Quantization（PTQ）。先訓練好 FP32 模型，再用 calibration data 決定 scale factor，把 weight 壓成 INT8。BatchNorm 先 fold 進 Conv 的 weight 裡（BN folding），消除硬體執行時的除法。相關腳本：`scripts/05_hardware_handoff/export_tcn_int8_artifacts.py`

**Q：`hls4ml` 試過了嗎？為什麼不用？**
A：試過，卡在 `permute` 和 `causal pad` 等 hls4ml 不好解析的算子（相關記錄見 `artifacts/04_hardware_handoff/hls4ml_attempt/`）。後來改成手刻 C++（`fixed_tcn_modules.cpp`），完全掌控 HLS 友善度。

**Q：TW 訓練的模型能直接用在 CME 嗎？**
A：Week 6 做了直接驗證，gated accuracy 0.82 但 net PnL = -36316.5 ticks，結論：不行。原因是 TW 和 CME 的微觀結構有 domain shift（成交量、流動性特性不同）。後續計畫分商品訓練各自的 ES / CL 模型。

**Q：csim 和 csynth 的差別？**
A：`csim`（C Simulation）只驗證軟體行為是否正確（幾分鐘）。`csynth`（C Synthesis）才是真正把 C++ 合成成 RTL（Verilog/VHDL），會輸出 latency / resource 報告（幾十分鐘）。

**Q：HLS top function 是什麼？**
A：`tcn_fixed_top`（定義在 `fixed_tcn_top.cpp`）。HLS 只能 synthesize 從這個 top function 開始往下呼叫的函式，其餘 host-side 邏輯（檔案讀取等）必須隔離。

---

## 9. 關鍵下一步（Week 8 目標）

### HLS 主線（最優先）
1. **`csynth` 跑完 → 收集第一版 Synthesis Report**
   - 重點指標：LUT / FF / BRAM / DSP 使用率、Latency cycles、Initiation Interval (II)
2. **加 HLS pragma 優化**：
   - `#pragma HLS PIPELINE` → 流水線化（讓 II 接近 1）
   - `#pragma HLS UNROLL` → 迴圈展開（換更多 LUT 換時間）
   - `#pragma HLS ARRAY_PARTITION` → 消除 BRAM read bottleneck
3. **目標**：讓 Latency < 50 µs（預估 FPGA latency ~9 µs，有餘裕）

### CME 次線
- 開始訓練 ES 專屬模型 + CL 專屬模型
- 每個各自走 train → calibrate → eval 流程

---

## 10. 容易踩到的坑

| 坑 | 說明 | 解法 |
|----|------|------|
| Synthesis 用 `std::vector` | HLS 不支援動態容器 | 改用固定大小 C 陣列 |
| Synthesis 用 `std::string` / `cout` | HLS 不支援 | 移到 host-only 檔案 |
| `std::copy` 引發 pointer 問題 | 已踩過 | 改成明確 for loop |
| Class imbalance | accuracy 高 ≠ 能交易 | 看 macro-F1 和 PnL |
| Label leakage | rolling 特徵 `bfill` 開頭有輕微影響 | 已知，後續改前向填補 |
| TW → CME transfer | 特徵分布 domain shift | 分商品各自訓練 |
| Part not installed | HLS 需已安裝的 FPGA part | 用 `xc7z020clg400-1` |

---

## 11. 重要路徑速查

| 項目 | 路徑 |
|------|------|
| **最完整技術文件** | `teaching/20_Model_Features_Params_and_Dataflow.md` |
| **HLS 操作步驟** | `teaching/22_Vitis_HLS_2025_2_Step_By_Step.md` |
| **主力凍結模型** | `artifacts/03_latency_proof/latency_proof/frozen_tcn_best_model.pt` |
| **INT8 權重** | `artifacts/04_hardware_handoff/int8_export/tcn_int8_weights.wts` |
| **C++ HLS top function** | `cpp_hls_prep/src/fixed_tcn_top.cpp` |
| **生成的 C++ 權重 header** | `cpp_hls_prep/include/generated/fixed_tcn_weights.hpp` |
| **最新週次記錄** | `Weekly_log/week7.md` |
| **下週規劃** | `Weekly_log/w7_next_step.md` |
| **專案全貌簡介** | `guide.md` |
| **目錄結構** | `file_structure.md` |
