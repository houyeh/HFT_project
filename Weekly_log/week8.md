# Week 8：Bit-accurate（位元級精確）驗證完成 + Week 6–8 報告整合

> 日期範圍：2026-04-07 ～ 2026-04-16（對齊 Week6–8 的 repo 狀態）

本週的核心目標是完成：

1. **Week 6（CME 支線）**：把 CME Logic20 資料接上現有 pipeline，做「TW frozen model 直接丟 CME」的最直接驗證，得到清楚結論與數據。
2. **Week 7（HLS 主線前置）**：把 TCN INT8 推論重寫成 HLS-friendly 的固定 shape C++ core，建立 testbench，並跑通 Vitis HLS `csim`/`csynth` 產生報告。
3. **Week 8（本週重點）**：完成 **Python INT8 模擬 ↔ C++ INT8 推論的 bit-accurate（逐 bit）對齊驗證**，作為進入 HLS 優化前的最終 correctness gate。

---

## 0. 一頁式摘要（可直接拿去報告）

### 0.1 現在專案主線/次線
- 主線：`TW replay -> features -> TCN -> frozen model -> INT8 export -> fixed-shape C++ -> Vitis HLS`
- 次線：`CME Logic20 adapter -> direct validation（TW frozen model） -> 分商品模型（規劃中）`

### 0.2 Week6–8 最關鍵結論
- **CME 直接搬用 TW 模型：分類看似 OK，但交易 PnL 大幅為負**，因此不能宣稱跨市場通用；合理下一步是走 **symbol-specific training / calibration**。
- **硬體交付（handoff）鏈已端到端打通**：凍結模型 -> INT8 量化匯出 -> C++ fixed core -> HLS testbench -> Vitis HLS `csim` + `csynth`。
- **bit-accurate 已完成**：同一份 `reference_input`，Python INT8 simulator 與 C++ logits **逐 bit 相等**；這代表 C++ 整數數學與量化規則已被鎖定，可作為後續 HLS pragma / microarchitecture 優化的 correctness 防線。

### 0.3 目前最大卡點（不是 correctness，是資源/時脈/延遲）
以 `xc7z020clg400-1` 目標器件為例（Vitis HLS `csynth`）：
- Estimated clock：`6.978 ns`（target `5.00 ns`，未達標）
- Latency：`~133,612 cycles`，約 `0.932 ms`
- Utilization：DSP `194%`、FF `105%`、LUT `410%`（嚴重超限）

結論：
- 已證明「可合成、可跑通」
- 下一步重點是：**縮 DSP/LUT、降低並行度、增加重用（reuse）、改善 memory banking/partition 與 pipeline/II**；或評估換更大器件/調整模型結構。

### 0.4 口頭報告建議順序（3–5 分鐘版本）
1. **一句話題目**：把 event-level TCN 從 Python 移植到 FPGA 流程（INT8 + C++ + Vitis HLS）。
2. **Week 6**：CME 接軌 + 直接驗證：分類還行但交易 PnL 不行 → 結論：不能宣稱跨市場通用，要走分商品。
3. **Week 7**：硬體交付鏈打通：INT8 匯出、固定 shape C++ core、HLS testbench、csim/csynth report。
4. **Week 8**：bit-accurate：Python INT8 simulator 與 C++ logits_int32 逐 bit 對齊（correctness gate 完成）。
5. **現在卡點**：不是邏輯，是 HLS 資源/時脈/延遲超限 → 請老師/廠商給 microarchitecture 建議。

---

## 1. 整體資料處理與硬體交付流程圖

### 1.1 軟體/資料流（TW + CME）

```mermaid
flowchart LR
  subgraph TW[TW replay 主線]
    A[Raw replay] --> B[LOB 重建/事件流]
    B --> C[Features
(12 dims execution_aware)]
    C --> D[TCN training/eval]
    D --> E[Frozen model
frozen_tcn_best_model.pt]
  end

  subgraph CME[CME 次線]
    M[Logic20 CSV] --> N[CME adapter
Logic20 -> 標準 LOB schema]
    N --> C
    E --> O[Direct validation
(TW frozen -> CME)]
  end

  subgraph HW[硬體交付]
    E --> P[INT8 export
.wts + reference I/O]
    P --> Q[C++ dynamic INT8 inference]
    P --> R[C++ fixed-shape core
(HLS-friendly)]
    R --> S[HLS testbench]
    S --> T[Vitis HLS
csim/csynth]
  end
```

### 1.2 correctness gate（Week 8）

```mermaid
flowchart TD
  X[reference_input.txt
int8 (12x100)] --> PY[Python INT8 simulator
逐層整數運算]
  X --> CPP[C++ inference
(dynamic/fixed)]
  PY --> CMP{logits_int32 bit-equal?}
  CPP --> CMP
  CMP -- yes --> PASS[PASS: bit-accurate
可進 HLS 優化]
  CMP -- no --> DBG[DEBUG: 對齊
rounding/scale/ReLU/residual]
```

---

## 2. Week 6：CME 直接驗證（TW frozen model -> CME）

### 2.1 目的
用最直接的方式回答：「用台灣期貨資料訓練的 frozen TCN，能不能直接在 CME 上交易？」

### 2.2 方法（對齊既有 pipeline）
- CME adapter：`src/data/cme_logic20.py`
- 先用保守策略把 CME 2-level book 對齊到既有 feature pipeline（3–5 檔用 tick size 向外展開、量補 0），目的不是造假，而是 **先讓 pipeline 可重用、可跑通**。

### 2.3 結果（可報告的數據）
來源：`artifacts/02_model_runs/cme_eval_transfer/20260407_161938/summary.json`

#### (A) 主要 gated 指標
- `gated accuracy = 0.8202`
- `gated macro_f1 = 0.3868`
- `gated signal net pnl = -36,316.5 ticks`

#### (B) 混淆矩陣（gated）
> 類別順序：Down / Stationary / Up

| True\Pred | Down | Stationary | Up |
|---:|---:|---:|---:|
| Down | 370 | 512 | 591 |
| Stationary | 5903 | 54259 | 4140 |
| Up | 392 | 557 | 551 |

#### (C) 解讀
- adapter/pipeline **是通的**，模型可以在 CME 上完整跑完。
- `Stationary` 佔比非常高（support 約 64k/67k），因此 accuracy 看似不差。
- 但在交易面，net PnL 大幅為負，代表 **directional edge 與交易可用性不足**。

### 2.4 本週要問教授的重點（CME 線）
1. 你希望我們「先把 HLS 主線跑到可部署」，還是「同時把 CME 分商品模型也推進」？（時間資源如何分配）
2. CME 資料只有 2-level book 的情況下，用補齊 3–5 檔的保守策略是否合理？還是應該改做「只用 L1/L2 的 feature 子集」更誠實？
3. 評估標準：報告時以分類指標（macro-F1）為主，還是以交易指標（net PnL / hit rate / sharpe-like）為主？

---

## 3. Week 7：C++/HLS-friendly TCN core（端到端已打通）

### 3.1 已完成的 C++ 結構（給廠商/老師看得懂的版本）
- Dynamic INT8 reference inference（可讀 `.wts`）
  - `cpp_hls_prep/include/int8_kernels.hpp`
  - `cpp_hls_prep/src/int8_kernels.cpp`
- Fixed-shape HLS-friendly core（固定 12x100、3 blocks、channels=16、k=3）
  - `cpp_hls_prep/include/fixed_tcn.hpp`
  - `cpp_hls_prep/src/fixed_tcn.cpp`
  - `cpp_hls_prep/include/fixed_tcn_modules.hpp`
  - `cpp_hls_prep/src/fixed_tcn_modules.cpp`
  - HLS top：`cpp_hls_prep/src/fixed_tcn_top.cpp`

### 3.2 HLS testbench 與 golden vectors
- golden vectors（固定 input、golden logits/probabilities）：
  - `cpp_hls_prep/include/generated/fixed_tcn_test_vectors.hpp`
- HLS testbench：
  - `cpp_hls_prep/src/hls_testbench.cpp`

### 3.3 Vitis HLS：csim/csynth 已產出
- csim log（PASS）：
  - `cpp_hls_prep/hls/project/fixed_tcn_hls/solution1/csim/report/tcn_fixed_top_csim.log`
- csynth report（關鍵指標）：
  - `cpp_hls_prep/hls/project/fixed_tcn_hls/solution1/syn/report/tcn_fixed_top_csynth.rpt`

---

## 4. Week 8：bit-accurate（Python INT8 ↔ C++）驗證

### 4.1 目標定義
「bit-accurate」在本專案的意思是：
- 同一份 `reference_input.txt`（int8，shape=12x100）
- Python 端用 numpy 實作 **完全相同** 的整數推論流程
- C++ 端用 `.wts` 跑推論
- 最終 `logits_int32` 三個值 **逐 bit 完全相等**（`np.array_equal`）

### 4.2 驗證腳本與可重現指令
- 驗證腳本：`scripts/05_hardware_handoff/verify_cpp_python_bit_accurate.py`
- 參考 Python INT8 simulator：`scripts/04_analysis/simulate_int8_inference.py`

從 repo root 執行：
```bash
python scripts\05_hardware_handoff\verify_cpp_python_bit_accurate.py
```
成功時會輸出：
```text
[SUCCESS] Layer-by-layer mathematical logic is BIT-ACCURATE!
```

### 4.3 Golden 對齊證據（給廠商/老師看的「硬證據」）
- C++/HLS 的 golden logits（int32）：
  - `cpp_hls_prep/include/generated/fixed_tcn_test_vectors.hpp`
  - `kGoldenLogitsInt32 = {-3947, 220, -4732}`
- 此 logits 也與 `cpp_hls_prep/build/tcn_hls_prep.exe` 的輸出一致。

### 4.4 本週踩過的坑（重要：避免之後又對不齊）
1. **ReLU placement 必須一致**
   - `conv2` 後如果 C++/固定 reference 有 `ReLU`，Python simulator 也必須有；少一次 ReLU 就會讓 logits 全部偏掉。
2. **rounding 規則要對齊 C++**
   - 量化/反量化（requantize）與 bias 對齊時會涉及 rounding。
   - 若 Python 用 `floor(x + 0.5)`，對負數會錯；需要「half-away-from-zero」的行為才會對齊。

### 4.5 關鍵程式碼片段（給老師/廠商快速對焦）

#### (A) HLS Top-level 介面與 pragma
檔案：`cpp_hls_prep/src/fixed_tcn_top.cpp`

```cpp
void tcn_fixed_top(
  const int8_t input[kFixedNumInputs * kFixedSeqLen],
  int32_t logits[kFixedNumClasses],
  float probabilities[kFixedNumClasses]) {
#pragma HLS INTERFACE ap_memory port=input
#pragma HLS INTERFACE ap_memory port=logits
#pragma HLS INTERFACE ap_memory port=probabilities
#pragma HLS INTERFACE ap_ctrl_hs port=return
  ...
}
```

#### (B) Fixed core 裡 conv2 後的 ReLU（會影響 bit-accurate）
檔案：`cpp_hls_prep/src/fixed_tcn_modules.cpp`

```cpp
for (auto& value : acc2) {
  if (value < 0) {
    value = 0;
  }
}
```

#### (C) bit-accurate 驗證腳本：同一份 reference input 同時餵 Python/C++
檔案：`scripts/05_hardware_handoff/verify_cpp_python_bit_accurate.py`

```python
cmd = [
  "cpp_hls_prep/build/tcn_hls_prep.exe",
  "--weights", "artifacts/04_hardware_handoff/int8_export/tcn_int8_weights.wts",
  "--ref-input", "artifacts/04_hardware_handoff/int8_export/reference_input.txt",
]
```

---

## 5. 本週圖表整理（可直接貼投影片）

### 5.1 模型規模與延遲估計（工程估算）
來源：`artifacts/04_hardware_handoff/latency_estimate/latency_estimate.json`

| 指標 | 數值 |
|---|---:|
| params | 4,771 |
| MACs/sample | 460,848 |
| FP32 weight bytes | 19,084 |
| INT8 weight bytes | 4,771 (~4.8KB) |
| PyTorch CPU latency p50 | 0.4465 ms |
| Estimated C++ INT8 latency | 57.606 µs |
| Estimated FPGA latency | 9.217 µs |

> 註：這是 order-of-magnitude engineering estimate，非 post-route。

### 5.2 HLS 綜合結果（csynth）— 目前超限摘要
來源：`cpp_hls_prep/hls/project/fixed_tcn_hls/solution1/syn/report/tcn_fixed_top_csynth.rpt`

| 項目 | 結果 |
|---|---|
| Estimated clock | 6.978 ns（target 5.00 ns） |
| Latency | ~133,612 cycles（~0.932 ms） |
| BRAM | 9% |
| DSP | 194%（超限） |
| FF | 105%（超限） |
| LUT | 410%（超限） |

---

## 6. 給廠商看的「技術交接摘要」（像簡報一樣）

### 6.1 我目前做法（technical details）
1. **模型**：TCNClassifier（3 blocks，channels=16，kernel=3，dilation=1/2/4，classes=3）
2. **輸入**：event-level LOB snapshot，window=100 events，features=12（execution-aware）
3. **量化**：PTQ（對稱 int8），activation 固定 scale = 1/64
4. **匯出物**：
   - INT8 weights：`artifacts/04_hardware_handoff/int8_export/tcn_int8_weights.wts`
   - reference input：`reference_input.txt`（int8）
   - reference output：`reference_output.txt`（FP32 logits/probabilities，用於 sanity check）
   - fixed headers：`cpp_hls_prep/include/generated/*`
5. **C++ 推論**：
   - dynamic core（讀 `.wts`）
   - fixed core（完全固定 shape，HLS-friendly）
6. **correctness**：Python INT8 simulator ↔ C++ logits_int32 bit-accurate 已通過。
7. **HLS**：csim pass、csynth 出 report，證明可合成，但目前資源與時脈不符合 xc7z020。

### 6.2 我希望廠商給建議的問題清單（請你幫我看）
1. 在 `xc7z020` 這種等級的 FPGA，上述 TCN 結構是否「理論上就不該做 fully-parallel」？
   - 若要 fit，通常的設計策略是什麼：
     - 限制 DSP 使用（resource constraint）
     - 時間複用 MAC（reuse factor）
     - 小型 systolic / MAC array？
2. 對於 `conv1d causal` + dilation 的資料流，你會建議我用哪一種 microarchitecture：
   - shift register/line buffer 的做法（每 tick O(1) 更新）
   - 或是直接展開時間維度做 pipeline（但會爆 LUT/DSP）
3. 權重只有 4.8KB，是否應該 **全部常駐 on-chip**？
   - BRAM vs LUTRAM 的選擇建議？banking/partition 的典型做法？
4. 為了過 timing（5ns），你會建議優先：
   - 降低 combinational path（切 pipeline stage）
   - 還是先降低資源（降低 congestion/route complexity）
5. 若目標是「展示 HLS 能做低延遲」，你會建議：
   - (A) 先換更大器件把 II 拉到 1，再回頭縮
   - (B) 先在 xc7z020 上做資源縮減，接受較大 latency

---

## 7. 下週（Week 9）最具體下一步

### 7.1 HLS 主線（優先）
- 目標：讓設計 **先 fit（資源不超限）**，再談 latency < 50 µs。
- 具體工作：
  1. 對 `tcn_fixed_top` / `tcn_inference_fixed_int8` 加入資源/重用策略（限制 DSP、避免過度 unroll）。
  2. 做 memory partition/banking，讓內部 buffer 不被單埠 BRAM 卡住。
  3. 每次修改都要跑：`csim` + `verify_cpp_python_bit_accurate.py`，確保 correctness 不倒退。

### 7.2 CME 次線（不阻塞主線）
- 開始訓練 symbol-specific：ES / CL（資料到再加 NQ）。
- 並決定 CME book 深度不足時，feature 應該「補齊」還是「重做子集」。

---

## 8. 附錄：重要路徑與檔案索引

### 8.1 CME direct validation
- `src/data/cme_logic20.py`
- `scripts/02_train_eval/evaluate_cme_frozen_model.py`
- `artifacts/02_model_runs/cme_eval_transfer/20260407_161938/summary.json`

### 8.2 INT8 export / handoff artifacts
- `scripts/05_hardware_handoff/export_tcn_int8_artifacts.py`
- `artifacts/04_hardware_handoff/int8_export/`

### 8.3 bit-accurate verification
- `scripts/05_hardware_handoff/verify_cpp_python_bit_accurate.py`
- `scripts/04_analysis/simulate_int8_inference.py`

### 8.4 C++ fixed core / HLS
- `cpp_hls_prep/src/fixed_tcn.cpp`
- `cpp_hls_prep/src/fixed_tcn_modules.cpp`
- `cpp_hls_prep/src/fixed_tcn_top.cpp`
- `cpp_hls_prep/src/hls_testbench.cpp`
- `cpp_hls_prep/hls/project/fixed_tcn_hls/solution1/syn/report/tcn_fixed_top_csynth.rpt`

### 8.5 可重現指令（建議報告時附上）

#### (A) 匯出 INT8 artifacts（.wts + reference I/O + generated headers）
```bash
python scripts\05_hardware_handoff\export_tcn_int8_artifacts.py
```

#### (B) bit-accurate：Python INT8 ↔ C++（logits_int32 逐 bit 對齊）
```bash
python scripts\05_hardware_handoff\verify_cpp_python_bit_accurate.py
```

#### (C) C++ 推論執行（dynamic core，讀 .wts）
```bash
cpp_hls_prep\build\tcn_hls_prep.exe --weights artifacts\04_hardware_handoff\int8_export\tcn_int8_weights.wts --ref-input artifacts\04_hardware_handoff\int8_export\reference_input.txt
```

#### (D) HLS：csim / csynth
（若已安裝 Vitis HLS 2025.2 且環境變數/路徑 OK）
```powershell
.\cpp_hls_prep\hls\run_vitis_hls_2025_2.ps1 -Stage csim -Part xc7z020clg400-1
.\cpp_hls_prep\hls\run_vitis_hls_2025_2.ps1 -Stage csynth -Part xc7z020clg400-1
```

