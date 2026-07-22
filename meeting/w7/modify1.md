# W7 Meeting 投影片修改指引 (modify1.md)

> **以 W7_meeting.pptx / W7_meeting.pdf 的實際投影片架構為基礎。**
> 圖表腳本：`meeting/w7/appendix/scripts/gen_all_figures.py`
> 生成圖片路徑：`meeting/w7/appendix/figures/`

---

## 📊 圖表索引

| 圖片檔名 | 對應投影片 | 說明 |
|---|---|---|
| `fig_pipeline.png` | Slide 3 | 完整 Pipeline 方塊圖（TW主線 + CME次線 + 硬體主線） |
| `fig_tcn_arch.png` | Slide 7 | TCN 架構示意圖（3 ResidualBlocks, dilation=1/2/4） |
| `fig_gating_diagram.png` | Slide 9 | Gating 訊號過濾流程圖 |
| `fig_cme_results.png` | Slide 16/17 | CME 驗證結果（PnL + Confusion Matrix + Macro-F1對比） |
| `fig_hls_resources_and_latency.png` | Slide 14/15 | HLS csynth 資源超限 + 延遲三層次對比 |

> 重新跑圖表指令：
> ```powershell
> python meeting\w7\appendix\scripts\gen_all_figures.py
> ```

---

## 📋 逐頁修改指引

---

### Slide 1：封面（HFT TCN Hardware Inference 系統）
**現有內容：** 標題、日期（Week 7 Report）

**問題：** 週次已是 Week 8，封面沒有更新。

**修改建議：**
- 標題改為：**「HFT TCN Hardware Inference 系統 — Week 7-8 進度報告」**
- 副標題加上本次的三個核心里程碑：
  1. CME Logic20 直接驗證（Transfer 失敗）
  2. C++ Fixed-Shape HLS Core（Bit-Accurate 通過）
  3. Vitis HLS csim ✅ / csynth ⚠️（資源超限）

---

### Slide 2：Motivation（為什麼要做 FPGA 低延遲推論）
**現有內容：** 「Why FPGA?」的動機說明

**問題：** 缺少量化數字支撐動機。

**修改建議：**
- 加入延遲對比數字：
  - 競爭者 C++ INT8 目標：< 100 µs
  - 目前 PyTorch CPU：446 µs（p50）
  - 目標 FPGA：< 50 µs
- 加入一句話說明：「差距就是 edge，這就是為什麼 FPGA 每微秒都值錢。」

---

### Slide 3：Pipeline Overview
**現有內容：** 文字描述流程，架構不清楚

**問題：** 沒有視覺化的流程圖，教授很難在 2 分鐘內理解整個系統。

**修改建議：**
- 替換為 `fig_pipeline.png`（Pipeline 方塊圖）
- 說明三條線：TW 主線（訓練） → 硬體主線（INT8/C++/HLS） → CME 次線（Transfer 驗證）
- 重點標注：Frozen Model 是三條線的核心「交交接」樞紐

---

### Slide 4/5：Features（特徵列表與定義）
**現有內容：** 只有特徵名稱列表，沒有定義

**問題：** 教授必問「這些特徵是什麼、怎麼計算的」，且圖表生成可能不夠清晰。

**修改建議：**
直接在投影片中放入以下 **12 個特徵定義總表**：

| 名稱 | 公式（簡化版） | 物理意義 | 類別 |
|---|---|---|---|
| `normalized_spread` | (Ask₁ − Bid₁) / tick | 買賣價差（成本代理） | 基礎微結構 |
| `micro_return` | (mid_t − mid_{t−1}) / tick | 事件級中間價變動（動能） | 基礎微結構 |
| `obi` | (ΣV_bid − ΣV_ask) / (ΣV_bid + ΣV_ask) | 全深度委託簿失衡 (L1-5) | 基礎微結構 |
| `weighted_mid_deviation` | (VWAP_mid − mid) / tick | 量加權中間價偏離 | 基礎微結構 |
| `deep_obi` | 同 OBI 但只用 L2-5 | 深層委託簿失衡（背後力量） | 基礎微結構 |
| `queue_imbalance` | (V_bid1 − V_ask1) / (V_bid1 + V_ask1) | 最佳一檔即時壓力 | 執行感知 |
| `queue_imbalance_change` | q_imb_t − q_imb_{t−1} | 委託壓力方向變化 | 執行感知 |
| `aggressor_imbalance` | Σ(sign_i·V_i) / ΣV_i | 淨主動方向 (interval OFI) | 執行感知 |
| `signed_flow_ratio` | 滾動 N 筆淨成交流向比率 | 近期 Order Flow Imbalance | 執行感知 |
| `trade_intensity_short` | 近期 N quote 內成交數 | 短期市場活躍度 | 執行感知 |
| `net_cancel_pressure` | 最佳一檔掛單量減少代理值 | 撤單壓力（流動性消失） | 執行感知 |
| `cancel_pressure_short` | 短視窗撤單壓力滾動值 | 即時流動性不穩定訊號 | 執行感知 |

- 口頭說明：前 5 個特徵是有效的「委託簿快照狀態」，後 7 個需要對齊 trade event（執行感知）。

---

### Slide 6：Model Params（模型參數）
**現有內容：** 部分已填，但有**錯誤數值**

**問題（需立即修正）：**
- ❌ `channels = 32` → 應為 **`channels = 16`**
- ❌ `sequence length = 10` → 應為 **`sequence length = 100`**（10 是 horizon）
- 參數量、MACs 是否有填？

**修改建議（全部正確數值）：**

| 參數 | 正確值 |
|---|---|
| TCN Blocks | 3 |
| Channels per block | **16** |
| Kernel size | 3 |
| Dilations | 1 / 2 / 4 |
| Window size (sequence length) | **100** |
| Horizon | **10** |
| Output classes | 3（Down / Stationary / Up） |
| Total params | **4,771** |
| MACs / sample | **460,848** |
| Receptive Field | **29 events** |
| INT8 weight size | **4.8 KB** |

---

### Slide 7：Model Architecture 示意圖（若有）
**修改建議：**
- 加入 `fig_tcn_arch.png`（TCN 架構方塊圖）
- 重點標注 Receptive Field = 29、感受野計算公式
- 加入：「這顆模型的設計出發點是『夠小（4.8 KB）、夠快（固定 shape）、夠正確（bit-accurate）』」

---

### Slide 8：Label & Sampling Method
**現有內容：** 說明 event-based model

**修改建議：補充以下圖解式說明**

```
[t0][t1][t2]...[t99]  ← 100 個 quote events → Model Input (100×12)
                                                     ↓ TCN
                                               3-class prediction
                                                     ↓
                         觀察 [t100...t109] 的 mid-price 方向
                         超過 1 tick → Up/Down；否則 → Stationary

※ 不是每個 raw market update 都跑一次：
   ・離線訓練：quote_stride=5, sample_stride=10, dedupe=True
   ・線上推論（未來）：每個新 event 更新 ring buffer，立即推論
```

---

### Slide 9：Gating Mechanism（訊號過濾）
**現有內容：** 有標題，內容不完整

**修改建議：**
- 加入 `fig_gating_diagram.png`（Gating 流程示意圖）
- 三層過濾條件：
  1. **P_dir ≥ Threshold**（基礎 0.5 + Spread penalty + Vol penalty）
  2. **P_dir − P_stationary ≥ Margin**（目前 0.0）
  3. **Regime Guards**（波動率 / 成交強度 / 掛單量 / 撤單壓力）
- 加入 CME 的教訓：套用 Gating 後訊號幾乎全被過濾 → 表示模型對 CME 幾乎沒有 directional edge

---

### Slide 10：INT8 Quantization（量化說明）
**現有內容：** 有標題，內容不完整

**修改建議：**
- 加入 PTQ（Post-Training Quantization）說明：
  - 對稱量化：`scale = max(|w|) / 127`
  - Activation 固定 scale：`1/64`
- 加入量化後的關鍵數字：

| 指標 | FP32 | INT8 | 差異 |
|---|---|---|---|
| Weight bytes | 19,084 B | **4,771 B** | 4× 壓縮 |
| Latency (CPU, p50) | 446.5 µs | ~57.6 µs (est.) | ~8× 加速 |
| Max logit diff (Python↔C++) | — | **0（Bit-Accurate ✅）** | — |

- Golden logits：`[-3947, 220, -4732]`（Python = C++，逐 bit 相等）

---

### Slide 11：Software Inference（C++ Dynamic Core）
**現有內容：** 有標題，內容不完整

**修改建議：**
- 說明 Dynamic C++ 的角色（讀 .wts 在執行期做推論，用來驗證量化）
- 加入 Bit-Accurate 驗證的邏輯流程：

```
reference_input.txt (int8) → [Python INT8 Simulator] ─┐
                           → [C++ tcn_hls_prep.exe]  ─┴→ logits_int32 相等? → PASS ✅
```

- 說明踩過的兩個坑：
  1. **ReLU 放置位置**：conv2 後有一個 ReLU，漏掉就 logits 全偏
  2. **Rounding 規則**：Python `floor(x+0.5)` 對負數與 C++ `round()` 行為不同，需對齊 half-away-from-zero

---

### Slide 12：Hardware Inference（Fixed-Shape C++ HLS Core）
**現有內容：** 有標題，疑似空白

**修改建議：**
- 加入 Dynamic vs Fixed-shape 差異表：

| 特性 | Dynamic C++ | Fixed-shape HLS-ready |
|---|---|---|
| 形狀 | 執行期讀 .wts | Compile-time 常數 |
| 記憶體 | heap / vector | 靜態陣列（適合 BRAM）|
| HLS pragma | 難加 | 完全支援 |
| 驗證 | 動態輸出 | Golden header 靜態比對 |

- 加入 HLS Top-level 介面（關鍵程式碼，4 行）：
  ```cpp
  void tcn_fixed_top(
      const int8_t input[12 * 100],   // 1200 bytes
      int32_t  logits[3],
      float    probabilities[3]) {
  #pragma HLS INTERFACE ap_ctrl_hs port=return
  ```

---

### Slide 13：hls4ml 嘗試（為什麼改走手刻）
**現有內容：** 有提及 hls4ml，但未說明失敗原因

**修改建議：**
- 加入三種 wrapper 嘗試結果表：

| Wrapper | 失敗原因 |
|---|---|
| `original` | hls4ml 不支援 `permute`（channel-last → first） |
| `channel_first_mean` | `CausalConv1d` 的 `F.pad` 無對應 HLS 算子 |
| `channel_first_pool` | 相同問題 |

- 強調手刻的三個優勢：完全控制資料流、精確 pragma 控制、逐層 bit-accurate 驗證

---

### Slide 14：HLS Synthesis Results（Vitis HLS csynth）
**現有內容：** 可能有舊數字，需更新

**修改建議：**
- 插入 `fig_hls_resources_and_latency.png`（左：資源使用率 / 右：延遲對比）
- 附上正確的 csynth 數字表：

| 指標 | 結果 | 目標 | 狀態 |
|---|---|---|---|
| Estimated Clock | **6.978 ns** | 5.00 ns | ⚠️ |
| Latency | **~133,612 cycles ≈ 0.932 ms** | < 50 µs | ❌ |
| BRAM | 9% | ≤100% | ✅ |
| DSP48 | **194%** | ≤100% | ❌ |
| FF | **105%** | ≤100% | ❌ |
| LUT | **410%** | ≤100% | ❌ |

- 加入根本原因：「目前是 fully-parallel 實作，每個 MAC 都實例化 → LUT/DSP 炸開」
- 加入解決路徑：loop pipeline pragma + reuse factor → latency ↓，資源 ↓

---

### Slide 15：Latency Estimation（延遲估算）
**現有內容：** 可能只有 9.2 µs 這個數字

**問題：** 9.2 µs 是工程估算，但 csynth 是 0.932 ms，差 100 倍，不解釋會被追問。

**修改建議：**
- 使用 `fig_hls_resources_and_latency.png`（右側子圖，延遲對比）
- 加入解釋表：

| 來源 | 延遲 | 性質 |
|---|---|---|
| PyTorch CPU (p50) | 446.5 µs | 軟體實測 |
| Eng. Estimate (FPGA) | **9.2 µs** | MACs / 50 GMAC/s（理論下界）|
| Vitis HLS csynth（未優化）| **0.932 ms** | 完全展開、未加 pragma |
| 目標 | < 50 µs | 加 pipeline pragma 後可達 |

- 說明：9.2 µs 是「假設設計充分 pipeline（II=1）」的下界；csynth 0.932 ms 是「什麼優化都沒做的基準」。兩者不矛盾。

---

### Slide 16：CME Validation Results（CME 驗證結果）
**現有內容：** 數字可能存在但無圖表

**修改建議：**
- 插入 `fig_cme_results.png`（三子圖：PnL / Confusion Matrix / Macro-F1對比）
- 加入關鍵結論文字：
  - ✅ Pipeline 打通（adapter 正常運作）
  - ❌ Transfer 失敗（gated PnL = -36,316 ticks）
  - ❌ Directional edge 不足（macro-F1 = 0.387，僅略優於 random 0.333）

---

### Slide 17：CME by Symbol（各商品表現）
**現有內容：** 有數字但可能缺圖

**修改建議：**
- 使用 `fig_cme_results.png`（左側子圖）已含各商品 PnL
- 補充分析：

| Symbol | Trades | Net PnL | 結論 |
|---|---|---|---|
| ESM6（標普 500 期貨）| 466 | -1,181 ticks | ⚠️ 差但還有商量空間 |
| CLK6（原油期貨）| 11,481 | **-35,135 ticks** | ❌ 完全不適用 |

- 說明 CLK6 差的根本原因：台指期與原油期貨的 tick size、成交密度、波動特性完全不同

---

### Slide 18：Next Steps（下週計畫）
**現有內容：** 有規劃，可能需更新

**修改建議：**

**HLS 主線（優先）：**
1. 加 `#pragma HLS PIPELINE` 到主要 loop → 讓 LUT/DSP 降下來
2. 加 `#pragma HLS ARRAY_PARTITION` 改善 BRAM 存取
3. 每次修改後重跑 `verify_cpp_python_bit_accurate.py` 確保 correctness 不退步

**CME 次線（並行）：**
- 從共用模型改為 **ES 一顆、CL 一顆** 的 symbol-specific model
- 評估標準：symbol-specific macro-F1 > 0.50 才宣稱有 edge

---

### Slides 19-21：Backup Q&A（備用問答）
**現有內容：** 有問答，可能不完整

**建議補上以下高頻追問：**

**Q1：CSynth 超限這麼嚴重，你的 9.2 µs 目標還成立嗎？**
> A：成立。9.2 µs 是理論下界（假設 pipeline II=1）。目前 0.932 ms 是沒加任何優化的基準。下一步加 pipeline pragma 後，理論上可以降到 ~10-50 µs 區間。首先要讓設計 fit（不超限），才能談延遲優化。

**Q2：Macro-F1 = 0.387，比 random（0.333）好一點點，這有意義嗎？**
> A：在 CME transfer 的情況下幾乎沒有意義。交易 PnL 為負就是最直接的判斷依據。下一步做 symbol-specific model，目標 F1 > 0.5。

**Q3：你怎麼確定手刻 C++ 沒把模型改壞？**
> A：兩層驗證：(1) fixed core vs dynamic core 機率 diff = 0；(2) `logits_int32 = [-3947, 220, -4732]`，Python 和 C++ 完全一致（Bit-Accurate PASS）。

**Q4：每個 tick 都跑一次模型嗎？**
> A：在設計上，每個新的 quote event 可以更新 ring buffer 並立即推論（O(1) update）。但目前離線訓練有 `quote_stride=5, sample_stride=10` 做採樣，不是每筆資料都進模型。線上部署時才會接近「每 event 跑一次」。

---

## 🔴 最緊急修正（必須在報告前完成）

1. **Slide 6**：`channels = 32` → **`16`**，`sequence length = 10` → **`100`**（這是最顯眼的錯誤）
2. **Slide 4/5**：用 `fig_feature_table.png` 或至少補上特徵的數學定義
3. **Slide 14**：更新 csynth 數字，並插入 `fig_hls_resources_and_latency.png`
4. **Slide 15**：解釋 9.2 µs vs 0.932 ms 的差距（否則教授一定追問）
5. **Slide 16/17**：插入 `fig_cme_results.png`，說清楚 Transfer 失敗的根本原因
