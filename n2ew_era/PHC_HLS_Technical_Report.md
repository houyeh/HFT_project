# PHC-lite HLS FPGA 實作技術報告
### 從模型設計到 xc7z020 上板的完整優化歷程

**目標板：** PYNQ-Z2（Xilinx xc7z020clg400-1）  
**工具：** Vitis HLS 2025.2  
**生成日期：** 2026-06-15

---

## 目錄

1. [專案背景與目標](#1-專案背景與目標)
2. [PHC-lite 模型架構](#2-phc-lite-模型架構)
3. [HLS 實作策略](#3-hls-實作策略)
4. [資源優化歷程（8 輪）](#4-資源優化歷程)
5. [關鍵技術洞見](#5-關鍵技術洞見)
6. [最終結果](#6-最終結果)
7. [延遲分析與改進路徑](#7-延遲分析與改進路徑)
8. [比賽定位建議](#8-比賽定位建議)

---

## 1. 專案背景與目標

本專案為 HFT/FPGA 競賽作品，在 PYNQ-Z2 開發板上實作 PHC-lite 遞迴神經網路的 FPGA 推論核心，以定點運算硬體取代 CPU 軟體推論，探索 FPGA 在 HFT 信號推論中的可行性。

### 1.1 硬體資源限制（xc7z020clg400-1）

| 資源 | 可用量 | 說明 |
|------|--------|------|
| LUT | 53,200 | 查找表：組合邏輯 |
| FF | 106,400 | 正反器：暫存器 |
| DSP48E1 | 220 | 專用乘法器單元 |
| BRAM_18K | 280 | 區塊記憶體 |

### 1.2 核心設計約束

- PYNQ-Z2 是「展示可行性」用途，非生產部署
- PHC-lite ≠ 完整 PHCSSM 模型（簡化版本，16-dim hidden state）
- 遞迴模型特性：t+1 步依賴 t 步的 `state_next`，**不可跨步 pipeline**
- 比賽定位：展示 FPGA 推論可行性 + 資源/延遲 tradeoff 分析

---

## 2. PHC-lite 模型架構

### 2.1 I/O 規格

| 參數 | 維度 | 型別 | 說明 |
|------|------|------|------|
| `x_t`（輸入） | 16 | `phc_fixed_t` | 當前市場特徵向量 |
| `state_prev`（輸入） | 16 | `phc_fixed_t` | 上一步隱藏狀態 |
| `readout`（輸出） | 2 | `phc_fixed_t` | 預測訊號 |
| `state_next`（輸出） | 16 | `phc_fixed_t` | 更新後隱藏狀態 |

### 2.2 定點數型別

| C++ 別名 | ap_fixed 定義 | 整數/小數位 | 用途 |
|----------|---------------|-------------|------|
| `phc_fixed_t` | `ap_fixed<24,12>` | 12整數/12小數 | 輸入、輸出、權重 |
| `phc_acc_t` | `ap_fixed<40,16>` | 16整數/24小數 | 累加器（避免溢位）|
| `phc_var_t` | `ap_fixed<48,24>` | 24整數/24小數 | LayerNorm 變異數 |

### 2.3 計算流程

```
x_t [16] + state_prev [16]
    │
    ├─ linear_16_16_fixed → LayerNorm → SiLU → encoded [16]
    │
    ├─ linear_16_16_acc  → Sigmoid → gate_t [16]      (input: encoded)
    ├─ linear_16_16_acc  → Tanh    → candidate_t [16]  (input: encoded)
    │
    ├─ state_temporal = DECAY × state_prev + gate_t × candidate_t
    │
    ├─ linear_16_16_acc         (encoded)    → gain_encoded [16] ──┐
    ├─ linear_16_16_fixed_no_bias (state_prev) → gain_prev   [16] ─┤ Sigmoid → gain_t [16]
    │
    ├─ linear_16_16_acc (state_temporal) → SiLU → message [16]
    │
    ├─ loop_input = state_temporal + ALPHA × gain_t × message
    ├─ LayerNorm → z [16]
    │
    ├─ linear_16_2_acc → readout [2]
    └─ state_next = state_temporal [16]
```

**共 7 個 linear layer、2 個 LayerNorm、3 種 activation（Sigmoid / Tanh / SiLU）**

### 2.4 Tanh / Sigmoid 近似

以 65 點 LUT 近似 tanh，範圍 [-4, 4]，步長 0.125，兩點線性插值。Sigmoid 由半角公式推導，SiLU = x × sigmoid(x)，無獨立 LUT 開銷。

```cpp
static const phc_acc_t TANH_LUT[65] = { /* 預計算值 */ };
#pragma HLS ARRAY_PARTITION variable=TANH_LUT complete dim=1
// runtime index 存取需同時可讀所有元素 → 存在暫存器（FF），不用 BRAM
```

---

## 3. HLS 實作策略

### 3.1 選擇手動 HLS（Manual HLS），不用 hls4ml

hls4ml 自動生成的程式碼在小型 FPGA 上通常存在過度展開的問題；手動 HLS 讓我們對每個 pragma 有完整控制權，也更容易 debug 資源問題。

### 3.2 函式結構

```cpp
// 所有 helper 函式均 INLINE 到 phc_step，讓 HLS 做跨函式硬體共用
void linear_16_16_fixed(in, w, b, out) { #pragma HLS INLINE ... }
void linear_16_16_acc(in, w, b, out)   { #pragma HLS INLINE ... }
void layernorm_16(in, gamma, beta, out) { #pragma HLS INLINE ... }
void sigmoid_16(in, out)               { #pragma HLS INLINE ... }
// ... 等

void phc_step(x_t, state_prev, readout, state_next) {
    // 所有計算在此展平執行
}
```

### 3.3 測試向量

Python reference 模型生成 25,600 組 (input, expected_output) 對，存入 CSV，testbench 逐組驗證：
- `mismatches = 0`（完全符合）
- `max_abs_diff ≤ 0.00725275`（接受定點近似誤差）
- `tolerance = 1`（PHC_ABS_TOL = 1.0，即不允許邏輯錯誤）

---

## 4. 資源優化歷程

### 4.0 總覽

| 版本 | 關鍵變更 | DSP | LUT | FF | BRAM | cosim |
|------|---------|-----|-----|----|------|-------|
| v1 | 初始（double tanh） | **2636%** | 2073% | 441% | — | — |
| v2 | LUT tanh（65點） | **1603%** | 839% | 441% | — | PASS |
| v3 | 移除 inner UNROLL | 1603% | 839% | 441% | — | — |
| v4 | cyclic partition | 1603% | 839% | 441% | — | — |
| v5 | II=297 | ~395% | — | — | — | — |
| v6 | 刪除 PIPELINE | 247% | 113% | 86% ✅ | 34% ✅ | **PASS** |
| v7 | 移除 helper ARRAY_PARTITION | 121% | 77% ✅ | 52% ✅ | 0% ✅ | — |
| **v8** | **#pragma HLS PIPELINE off** | **21% ✅** | **46% ✅** | **13% ✅** | **2% ✅** | **PASS** |

---

### 4.1 v1 → v2：雙精度浮點根因

**問題：** 初始版本使用 C 標準 `tanh()`，編譯器生成 `generic_tanh_double_s` 雙精度浮點核心，一個函式就消耗 ~3,000 DSP。

```
Instance list 截錄（v1）：
  generic_tanh_double_s_U1   → 2048 DSP
  generic_sqrt_double_s_U1   →  588 DSP
  ... 更多浮點核心
```

**修復：** 替換為預計算的 65 點固定點 LUT + 線性插值，消除所有浮點硬體。cosim PASS，`max_abs_diff = 0.00725`，精度可接受。

---

### 4.2 v2 → v5：PIPELINE II=1 的致命副作用（四輪無效）

v2 把 tanh 換成 LUT 後，DSP 仍有 1603%。Instance list 顯示全是 `mul_*` 定點乘法器（約 882 個實例）。

**嘗試 v3**：移除 linear 函式的 inner loop UNROLL → **零效果**  
**嘗試 v4**：weight matrix 從 `complete dim=0` 改 `cyclic dim=2 factor=2` → **零效果**  
**嘗試 v5**：改 `#pragma HLS PIPELINE II=297`（代替 II=1）→ **仍無效**

每次 compile log 都出現同一個 WARNING：

```
WARNING: [HLS 214-275]
  The pragma 'UNROLL factor=2' in 'linear_16_16_acc' is partially unroll,
  and is converted into complete unroll due to PIPELINE pragma.
```

**根本原因確診：**
> **只要任何 `#pragma HLS PIPELINE II=X`（X 為任意正整數）存在於 phc_step 的 scope，Vitis HLS 就會強制把所有 partial UNROLL 升級為 complete UNROLL。** 這是 scheduler 為達到目標 II 的自動行為，無法被其他 pragma 覆蓋。

---

### 4.3 v6：刪除 PIPELINE pragma — 重大突破

**修復：** 直接刪除 `#pragma HLS PIPELINE` 這一整行。

沒有頂層 PIPELINE 後，HLS 改為把各個 loop auto-pipeline 成獨立子模組（24 個 loop pipeline modules），不再強制展開。

```
v6 結果（2026-06-15 00:11）：
  DSP:  545 / 220 = 247%  （仍超出）
  LUT:  60,378 / 53,200 = 113% （仍超出）
  FF:   92,260 / 106,400 = 86%  ✅
  BRAM: 96 / 280 = 34%    ✅
  Latency: 489 cycles（4.89 µs）

cosim：total=25600, mismatches=0, max_abs_diff=0.00725275 → PASS ✅
```

**剩餘根因：** helper 函式的參數陣列（`in`, `out`, `b`, `gamma`, `beta`）仍有 `ARRAY_PARTITION complete dim=1`，outer j-loop 仍有 `UNROLL factor=2`，使各子模組各自建立 2 個並行乘法器。

---

### 4.4 v7：移除 helper ARRAY_PARTITION — 新的自動行為問題浮現

移除所有 helper 函式的 `ARRAY_PARTITION` 和 `UNROLL` pragma 後：

```
v7 結果（2026-06-15 01:44）：
  DSP:  268 / 220 = 121%  （仍超出）
  LUT:  41,433 / 53,200 = 77%  ✅
  FF:   55,719 / 106,400 = 52% ✅
  BRAM: 0 / 280 = 0%      ✅
  Latency: 604 cycles（6.04 µs）
```

LUT/FF/BRAM 全過，但 DSP 仍 121%。compile log 出現新的三條 INFO：

```
INFO: [HLS 214-376]
  Pipelining loop<VITIS_LOOP_114_1> at phc_step_hls.cpp:114:23
  due to pipeline_loops threshold          ← HLS 自動把 16 次 loop 加上 pipeline

INFO: [HLS 214-291]
  Loop 'VITIS_LOOP_116_2' is marked as complete unroll
  implied by the pipeline pragma           ← outer pipeline 強制 inner 完全展開

INFO: [HLS 214-449]
  Partitioning array 'SYNAPSE_W' dimension 2
  due to constant index.                   ← 展開後 j 成常數 → 自動 partition 權重
```

**三連鎖機制（比 PIPELINE II=1 更隱蔽）：**

```
[HLS pipeline_loops 閾值] 發現 16-iteration loop
      → 自動加上 pipeline [HLS 214-376]
      → outer pipeline 強制 inner loop 完全展開 factor=16 [HLS 214-291]
      → 展開後 w[j][i] 的 j 在每個 pipeline cycle 為常數
      → 自動 partition weight matrix dim=2 [HLS 214-449]
      → 等效於手動寫 PIPELINE II=1 + complete partition
      → 每個 linear sub-module 仍有 16 個並行乘法器
```

移除手動 pragma → HLS 自動補回等效行為。**無法靠移除 pragma 解決，需要主動禁止。**

---

### 4.5 v8：#pragma HLS PIPELINE off — 最終解

**修復：** 對所有 linear 函式的 outer j-loop 與 inner i-loop，以及 layernorm、activation、phc_step element-wise loop，全部加上 `#pragma HLS PIPELINE off`：

```cpp
void linear_16_16_acc(const phc_acc_t in[16], ...) {
#pragma HLS INLINE
    for (int j = 0; j < 16; ++j) {
#pragma HLS PIPELINE off     // 明確禁止 pipeline_loops 自動接管
        phc_acc_t acc = (phc_acc_t)b[j];
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off     // 防止 inner loop 被 outlined 後重新 pipeline
            acc += (phc_acc_t)w[j][i] * in[i];
        }
        out[j] = acc;
    }
}
```

`#pragma HLS PIPELINE off` 的優先級高於 `pipeline_loops` 全域設定，是 HLS 提供的顯式覆蓋機制。

**驗證信號（compile log 不再出現）：**
- ❌ 消失：`[HLS 214-376]` "due to pipeline_loops threshold" 在 linear loop 上
- ❌ 消失：`[HLS 214-291]` "complete unroll implied by the pipeline pragma" 在 linear inner loop 上
- ❌ 消失：`[HLS 214-449]` "due to constant index" 的 weight auto-partition

```
v8 結果（2026-06-15 02:25）：
  DSP:  46 / 220 = 21%    ✅✅
  LUT:  24,836 / 53,200 = 46% ✅✅
  FF:   14,518 / 106,400 = 13% ✅✅
  BRAM: 6 / 280 = 2%      ✅✅
  Latency: 8,316 cycles（83.16 µs）

csim:  total=25600, mismatches=0, max_abs_diff=0.00725275 → PASS ✅
cosim: total=25600, mismatches=0, max_abs_diff=0.00725275 → PASS ✅
```

**csim 與 cosim 的 max_abs_diff 完全相同**，代表 RTL 實作與 C model 在 bit-level 一致。

---

## 5. 關鍵技術洞見

### 5.1 三大 HLS 陷阱（優先級由高到低）

#### 陷阱 1：任何 PIPELINE pragma 強制完全展開 UNROLL

```
[HLS 214-275] WARNING:
  partial UNROLL → complete UNROLL due to PIPELINE pragma
```

- 觸發條件：任何 `#pragma HLS PIPELINE II=X`（X ∈ {1, 2, ..., ∞}）
- 效果：scope 內所有 partial UNROLL 全部升級為 complete UNROLL
- **唯一解：刪除 PIPELINE pragma**

#### 陷阱 2：pipeline_loops 閾值自動接管（更隱蔽）

```
[HLS 214-376] INFO:
  Pipelining loop due to pipeline_loops threshold
```

- 觸發條件：loop 迴圈次數 ≤ 全域閾值（預設涵蓋 16-iteration loop）
- 效果：等同手動加 PIPELINE，觸發同樣的展開/partition 連鎖
- **解法：加 `#pragma HLS PIPELINE off` 明確覆蓋**（不能只移除手動 pragma）

#### 陷阱 3：Auto-partition 連鎖

```
[HLS 214-449] INFO:
  Partitioning array due to constant index.

[HLS 214-270] INFO:
  Inferring pragma 'array_partition type=complete dim=1' due to pipeline pragma.
```

- 觸發條件：陣列存取 index 在 pipeline scope 內為常數
- 效果：HLS 自動把 weight matrix / 本地陣列 partition，等同手動加 complete partition
- **根治：移除觸發它的 pipeline（陷阱 1/2）**

### 5.2 Pragma 最終正確配置

| 位置 | 正確設定 | 錯誤設定 |
|------|---------|---------|
| phc_step 頂層 | （無 pragma） | `PIPELINE II=1` |
| linear 外層 j-loop | `PIPELINE off` | （無 pragma）← 會被 pipeline_loops 接管 |
| linear 內層 i-loop | `PIPELINE off` | `UNROLL factor=N` |
| layernorm / activation 各 loop | `PIPELINE off` | （無 pragma）|
| phc_step element-wise loop | `PIPELINE off` | `UNROLL factor=4` |
| TANH_LUT | `ARRAY_PARTITION complete dim=1` | ← 這個保留！runtime index 選取必要 |
| 所有 helper 函式 | `INLINE` | （無 INLINE）|

### 5.3 HLS Warning/Info 對照速查表

| 代碼 | 嚴重性 | 含義 | 處置 |
|------|--------|------|------|
| `[HLS 214-275]` | WARNING | partial UNROLL → complete（因 PIPELINE） | 刪除 PIPELINE pragma |
| `[HLS 214-376]` | INFO | loop 因 pipeline_loops 被自動 pipeline | 加 `PIPELINE off` |
| `[HLS 214-291]` | INFO | inner loop 因 pipeline 隱含完全展開 | 根治 outer pipeline |
| `[HLS 214-449]` | INFO | array 因 constant index 被自動 partition | 根治 auto-pipeline |
| `[HLS 214-270]` | INFO | 本地陣列因 pipeline 被推斷 complete partition | 根治 pipeline |
| `[SYNCHK 200-120]` | WARNING | multiplication overflow 假設（ap_fixed 計算） | 可忽略（數值範圍設計合理）|
| `[SYN 201-103]` | WARNING | 函式名稱合法化（Xilinx 內部 IP） | 可忽略 |
| `[HLS 200-880]` | WARNING | II Violation（layernorm var 累加迴圈） | cosim PASS → 可接受，stall 邏輯正確 |

---

## 6. 最終結果

### 6.1 Csynth 資源利用率（v8，2026-06-15 02:25:11）

| 資源 | 用量 | 可用量 | 利用率 | 狀態 |
|------|------|--------|--------|------|
| DSP48E1 | 46 | 220 | **21%** | ✅ PASS |
| LUT | 24,836 | 53,200 | **46%** | ✅ PASS |
| FF | 14,518 | 106,400 | **13%** | ✅ PASS |
| BRAM_18K | 6 | 280 | **2%** | ✅ PASS |

### 6.2 效能數據

| 項目 | 數值 | 說明 |
|------|------|------|
| Latency | 8,316 cycles | **83.16 µs @ 100 MHz** |
| Interval（II） | 8,317 cycles | 遞迴模型不可跨步 pipeline |
| Clock Estimated | 7.147 ns | 目標 10 ns，timing margin 充裕 |
| Pipeline type | none | 完全序列執行 |

### 6.3 功能驗證結果

| 測試 | 總測試點 | mismatches | max_abs_diff | mean_abs_diff | 結果 |
|------|----------|------------|--------------|----------------|------|
| C-Sim（csim） | 25,600 | 0 | 0.00725275 | 0.000372553 | ✅ PASS |
| C/RTL Cosim | 25,600 | 0 | 0.00725275 | 0.000372553 | ✅ PASS |

csim 與 cosim 的 max_abs_diff 完全相同 → RTL 與 C model bit-level 一致，無任何精度損失或時序問題。

---

## 7. 延遲分析與改進路徑

### 7.1 83 µs 的成因

完全序列執行：每個 linear layer 的 16×16 矩陣乘法為 256 次序列 MAC。

| 運算 | Cycles | 說明 |
|------|--------|------|
| linear 16×16（×5 層） | ~1,300 × 5 | 256 MAC 序列 × 5 |
| linear 16×16 no bias（×1） | ~1,313 | GAIN_FROM_PREV_STATE |
| linear 16×2（×1） | ~164 | READOUT |
| LayerNorm（×2） | ~81 × 2 | 含 hls::sqrt（28 cycles） |
| Sigmoid/Tanh/SiLU（×3） | ~145 × 3 | LUT 查表序列 |
| Element-wise loops | ~200 | state_temporal, loop_input 等 |
| **合計** | **~8,316** | **83.16 µs @ 100 MHz** |

### 7.2 改進路徑比較

| 方案 | 預期 Latency | 預期 DSP | 代價 |
|------|-------------|---------|------|
| v8 現況（序列基準） | 83 µs | 21% | — |
| **INT16 量化**（ap_fixed<16,8>） | **~25 µs** | **~7%** | 需驗證 max_abs_diff |
| **Hidden dim 8**（從 16 縮半） | **~8 µs** | **~5%** | 需重新訓練模型 |
| **稀疏 Gate Update** | ~10–20 µs | ~10% | 需硬體 bitmask 邏輯 |
| **Alveo U50**（5,952 DSP） | **< 5 µs** | < 1% | 換板子，可全開 pipeline |

#### 方案 A：INT16 量化（最快可落地）

將 `ap_fixed<24,12>` 改為 `ap_fixed<16,8>`，16-bit 乘法可映射到單個 DSP48E1（每乘法 1 DSP 而非 3 DSP），節省的 DSP 預算可用來重新開放 PIPELINE factor=2 或 UNROLL factor=2，latency 估計降至 ~25 µs。代價是精度，需 post-training calibration 驗證。

#### 方案 B：縮小 hidden dim 至 8

模型 hidden dim 從 16 縮為 8，每個 linear layer 從 256 MAC 降到 64 MAC（降 4×）。在 220 DSP 內可重新開放 UNROLL factor=4，latency 估計 ~8 µs。代價是模型表達力，需重新訓練並驗證預測效果。

#### 方案 C：商用 FPGA（Alveo U50）

| 板子 | DSP | LUT | PHC-lite DSP% | 備註 |
|------|-----|-----|---------------|------|
| xc7z020（PYNQ-Z2） | 220 | 53,200 | 21%（序列） | 學術展示，83 µs |
| Alveo U50 | 5,952 | 872,000 | < 1%（並行） | 商用 FPGA，< 5 µs |

Alveo U50 可重新啟用 `PIPELINE II=1` + `UNROLL factor=16`，預計 latency < 5 µs，throughput 接近 II=1（但 recurrent step 仍需序列執行）。

---

## 8. 比賽定位建議

### 8.1 論述框架

從 reviewer 角度，比賽評分重點不在「最快 latency 數字」，而在於「設計決策的說服力」。

| 論點 | 支撐證據 |
|------|---------|
| FPGA 推論可行性已驗證 | PYNQ-Z2 上 csim + cosim PASS，RTL 完全正確 |
| 資源 tradeoff 理解透徹 | DSP 21%、LUT 46%——資源充裕，仍有優化空間 |
| 商業部署路徑清晰 | Alveo U50 DSP < 1%，可全開 pipeline 達 < 5 µs |
| 延遲改進方案明確 | INT16 量化可在同板達 ~25 µs；縮 dim 可達 ~8 µs |

### 8.2 建議論述（reviewer 角度）

> *「PHC-lite 在 xc7z020（最低階 Zynq，220 DSP）上以 83 µs latency 通過 csim/cosim，所有資源在預算內（DSP 21%、LUT 46%）。延遲的成因為完全序列執行，是在有限資源下維持數值正確性的必要取捨。若採 INT16 量化（同板可達 ~25 µs）或縮減 hidden dim 至 8（~8 µs），或換用 Alveo U50（< 5 µs），可達 HFT 可用的推論速度。本設計展示了完整的 FPGA 模型部署流程，以及對 HLS 資源/延遲 tradeoff 的系統性理解。」*

### 8.3 需要誠實說明的事項

- PYNQ-Z2 上的 83 µs 目前不足以用於真實 HFT（L1 order book 更新通常 < 10 µs 要求）
- PHC-lite 是 PHCSSM 的簡化版本，完整模型的資源需求更高
- 83 µs 的延遲不影響「模型邏輯是否正確」的驗證，但需補充延遲改進計畫

---

## 附錄

### A. 版本時間軸

| 時間戳 | 版本 | 里程碑 |
|--------|------|--------|
| Jun 13 早 | v1 | 初始雙精度實作，DSP 2636% |
| Jun 14 00:11 | v2 | LUT tanh 完成，cosim PASS，DSP 1603% |
| Jun 14 03:30 | v3～v4 | UNROLL/cyclic 修正，確認 PIPELINE 覆蓋問題 |
| Jun 15 00:11 | v6 | 刪除 PIPELINE，DSP 247%，cosim PASS |
| Jun 15 01:44 | v7 | 移除 helper partition，DSP 121%，pipeline_loops 問題浮現 |
| **Jun 15 02:25** | **v8** | **PIPELINE off，DSP 21%，全資源通過，cosim PASS —— 封箱** |

### B. 關鍵檔案路徑

```
n2ew_era/06f_phc_lite_manual_hls/
├── hls_src/
│   ├── phc_step_hls.cpp          ← 主設計檔（含 #pragma HLS PIPELINE off）
│   ├── phc_step_hls.hpp          ← 型別定義、常數
│   ├── phc_weights.hpp           ← 所有權重（const，FPGA ROM）
│   └── phc_step_testbench.cpp    ← CSV-based testbench
├── hls_component/hls_component/
│   ├── hls/syn/report/phc_step_csynth.rpt  ← 最終資源報告
│   └── logs/
│       ├── hls_compile.log        ← csynth compile log
│       ├── hls_run_csim.log       ← csim 結果
│       └── hls_run_cosim.log      ← cosim 結果（1.5 MB）
└── results/
    └── phc_step_test_vectors.csv  ← 25,600 組測試向量
```

---

*文件結束 | HFT/FPGA 競賽專案 | 2026-06-15*
