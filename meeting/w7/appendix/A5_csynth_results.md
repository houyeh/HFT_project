# A5：Vitis HLS csynth 結果詳細分析

> 對應 Slide 10「驗證進度」與 Slide 11「數字彙整」。
> 這是教授最可能仔細追問的頁面。

---

## csynth 結果摘要

目標器件：`xc7z020clg400-1`（Zynq-7020，常見於嵌入式 FPGA 開發板）

| 指標 | 實際結果 | 目標 | 狀況 |
|---|---|---|---|
| Estimated Clock | **6.978 ns** | 5.00 ns (200 MHz) | ⚠️ 未達標（+40%） |
| Latency（cycles） | **~133,612 cycles** | < 10,000 cycles | ❌ 嚴重超限（13× 以上）|
| Latency（時間） | **≈ 0.932 ms** | < 50 µs | ❌ 嚴重超限（18×） |
| BRAM | 9% | ≤ 100% | ✅ 合理（幾乎未用）|
| DSP48 | **194%** | ≤ 100% | ❌ 超限（近 2×） |
| FF（Flip-Flop） | **105%** | ≤ 100% | ❌ 輕微超限 |
| LUT | **410%** | ≤ 100% | ❌ 嚴重超限（4× 以上） |

---

## 超限根本原因分析

### 問題 1：嚴重超限的 LUT/DSP（根本原因）

**根因：目前 C++ 是「完全並行（fully-parallel）」的寫法**

VItis HLS 預設行為：對於一個 `for` 迴圈，如果沒有任何 pragma，HLS 會嘗試把迴圈裡的所有操作**展開並平行執行**。

舉例：

```cpp
// 目前的 causal_conv1d（Pseudo-code）
for (int t = 0; t < SEQ_LEN; t++) {
    for (int o = 0; o < OUT_CH; o++) {
        int32_t acc = 0;
        for (int i = 0; i < IN_CH; i++) {
            for (int k = 0; k < KERNEL; k++) {
                acc += input[i][t-k*dil] * weight[o][i][k];
            }
        }
        output[o][t] = requantize(acc + bias[o]);
    }
}
```

如果所有維度都被展開：
- 乘法次數 = SEQ_LEN × OUT_CH × IN_CH × KERNEL = 100×16×12×3 = **57,600 個 MAC 同時進行**
- 每個 MAC 消耗 ≥ 1 DSP48 block
- `xc7z020` 只有 **220 個 DSP block**（工具估算需要 **220 × 194% ≈ 427 個**）

### 問題 2：Clock 6.978 ns（組合邏輯路徑過長）

當大量計算被攤開在同一個 clock cycle，組合邏輯路徑（combinational path）太長，導致 clock 無法達到 200 MHz（5 ns）。

---

## 解決路徑（下週要做的事）

### 策略 A：加入 Reuse Factor（loop folding/time-multiplexing）

```cpp
// 在 for 迴圈上加約束：限制同時展開數
#pragma HLS ALLOCATION instances=mul limit=20 operation
```

或直接控制 loop：

```cpp
LOOP_OUT_CH: for (int o = 0; o < OUT_CH; o++) {
    #pragma HLS PIPELINE II=1   // ← 每個 output channel 占一個 cycle，循環
    LOOP_IN_CH: for (int i = 0; i < IN_CH; i++) {
        #pragma HLS UNROLL       // ← 輸入維度完全展開（MAC 平行）
        ...
    }
}
```

**效果**：DSP 用量從 ~427 降到 ~12（只展開 IN_CH × KERNEL = 12×3 = 36 個 MAC），換來的代價是 latency 從 1 cycle 提高到 OUT_CH cycles。

### 策略 B：Array Partition（改善 BRAM 存取頻寬）

```cpp
// 當前問題：多個 for 迴圈想同時讀同一個 BRAM，會有 port conflict
// 解決：把 weight 陣列分割成多個 partition

#pragma HLS ARRAY_PARTITION variable=weights dim=1 complete
//                                              ^^^  
// 把 out_channel 維度完全分割（每個 output channel 有獨立的 BRAM port）
```

### 策略 C：Shift Register（線上推論的終極優化）

當系統轉為「每個 event 更新一次」的線上模式：

```cpp
// 維護一個環形緩衝區（ring buffer）
// 每來一個新 event，只更新最新的 slice，其餘 99 個 slice 不動
// 輸入緩衝區只需 O(1) 寫入時間

int8_t ring_buffer[kFixedNumInputs][kFixedSeqLen];
int write_ptr = 0;  // 指向最新 event 的位置
```

---

## 三層延遲對比（必須在投影片中說清楚）

```
┌─────────────────────────────────────────────────────────────────┐
│                    延遲估算的三個層次                            │
│                                                                 │
│  1. PyTorch CPU (p50)           0.4465 ms  ←  軟體推論實測     │
│  2. Engineering estimate FPGA   9.2  µs   ←  MAC數/吞吐理論值  │
│  3. csynth 未優化               0.932 ms  ←  HLS合成實際報告   │
│  4. 專案目標                    < 50 µs   ←  設計目標          │
│                                                                 │
│  問：為什麼 9.2 µs 估算和 0.932 ms csynth 差了 100 倍？         │
│                                                                 │
│  答：9.2 µs 是「假設 FPGA 能以 50 GMAC/s 的有效吞吐率          │
│       連續執行」的理論下界；                                     │
│       0.932 ms 是「目前 fully-parallel 實作，                   │
│       沒有加任何優化、沒有 pipeline」的結果。                    │
│                                                                 │
│  要從 0.932 ms → < 50 µs，需要：                                │
│  ① 讓設計先 fit（LUT/DSP 不超限）                               │
│  ② 加 pipeline pragma，使 II = 1（每個 clock 產出一個結果）     │
│  ③ 評估是否需要換更大 FPGA（如 xc7z045 或 UltraScale+）         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 給教授的追問備案

**Q：你現在 0.932 ms 遠超過 50 µs，這是否代表你的目標根本無法達成？**

A：
> 不代表無法達成。目前 0.932 ms 是沒有加任何優化指令（pragma）的「基準」。
> 若加入 loop pipeline（Initiation Interval=1）並使設計 fit 進 xc7z020，
> 計算量（460,848 MACs）在 200 MHz 下理論只需 **約 2,305 cycles ≈ 11.5 µs**（假設 II=1）。
> 更積極的方向（MACs partial unroll + pipeline）可以進一步降低，
> 但首要工作是先讓設計 fit（不超限），因為超限的設計無法實際部署。

**Q：BRAM 只用 9%，但 LUT 用了 410%，矛盾嗎？**

A：
> 不矛盾。BRAM 低代表「資料儲存的空間夠用」（只有 4.8KB 權重 + 少量 buffer）。
> LUT 高代表「計算邏輯被完全攤開」——每個乘法和加法都對應一個 LUT 組合邏輯電路。
> 解決方式是讓計算「串行化」（time-multiplex），同一組 LUT 被複用多次，換取 latency 增加。
