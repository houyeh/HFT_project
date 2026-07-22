# A4：C++ HLS 模組架構與程式碼

> 對應 Slide 9「C++/HLS-friendly 重構」。
> 給廠商或教授快速理解 C++ 架構的參考文件。

---

## 模組架構方塊圖

```
╔══════════════════════════════════════════════════════════════════════╗
║                    tcn_fixed_top()   [HLS Top-level]                 ║
║   Input: int8_t[12 × 100]    Output: int32_t[3] + float[3]           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────────────────────────────────────────────────────┐     ║
║  │  tcn_inference_fixed_int8()   [src/fixed_tcn.cpp]           │     ║
║  │                                                             │     ║
║  │  Block 0 (dilation=1)                                       │     ║
║  │  ┌────────────────────────────────────────────────────┐     │     ║
║  │  │  residual_block_fixed()   [fixed_tcn_modules.cpp]  │     │     ║
║  │  │  ┌─────────────────┐   ┌─────────────────────┐     │     │     ║
║  │  │  │causal_conv1d    │   │ pointwise_conv1d    │     │     │     ║
║  │  │  │(in=12, out=16   │   │ (in=16, out=16,     │     │     │     ║
║  │  │  │ k=3, dil=1)     │   │  k=1)               │     │     │     ║
║  │  │  └────────┬────────┘   └──────────┬──────────┘     │     │     ║
║  │  │           │ ReLU + requantize     │ ReLU           │     │     ║
║  │  │           └──────────┬────────────┘                │     │     ║
║  │  │                      │ + residual (downsample if needed) │     ║
║  │  └──────────────────────┼────────────────────────────┘      │     ║
║  │                         │                                   │     ║
║  │  Block 1 (dilation=2)   │→  [相同結構]                       │    ║
║  │  Block 2 (dilation=4)   │→  [相同結構]                       │    ║
║  │                         │                                   │     ║
║  │            global_average_pool_fixed()                      │     ║
║  │            (對 seq_len=100 維度平均)                         │    ║
║  │                         │                                   │     ║
║  │              fc_fixed()                                     │     ║
║  │            (in=16, out=3)                                   │     ║
║  └─────────────────────────────────────────────────────────────┘     ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## HLS Top-level 介面（關鍵程式碼）

檔案：`cpp_hls_prep/src/fixed_tcn_top.cpp`

```cpp
// ============================================================
// HLS Top-level function：定義與 FPGA 世界的介面
// input:  int8 量化後的 12×100 特徵矩陣（攤平成一維陣列）
// logits: int32 原始邏輯輸出（三類別）
// probabilities: float softmax 輸出概率
// ============================================================
void tcn_fixed_top(
    const int8_t input[kFixedNumInputs * kFixedSeqLen],  // 12×100 = 1200 bytes
    int32_t  logits[kFixedNumClasses],                    // 3×4 = 12 bytes
    float    probabilities[kFixedNumClasses])             // 3×4 = 12 bytes
{
#pragma HLS INTERFACE ap_memory port=input        // 記憶體介面（BRAM 讀者）
#pragma HLS INTERFACE ap_memory port=logits       // 記憶體介面（BRAM 寫者）
#pragma HLS INTERFACE ap_memory port=probabilities
#pragma HLS INTERFACE ap_ctrl_hs port=return      // handshake 控制（start/done 信號）

    tcn_inference_fixed_int8(input, logits, probabilities);
}
```

---

## INT8 量化數學路徑

```
每一層的計算路徑：

int8_t input[N]
    × int8_t weight[K]
    ─────────────────────
    int32_t accumulator    ← 不允許溢位！int8 × int8 最大 = 127²×N = 需足夠位寬

    + int32_t scaled_bias  ← bias 已被縮放到 accumulator 域
    ─────────────────────
    int32_t pre_relu

    ──ReLU──► max(0, pre_relu)

    ──requantize──►  round( pre_relu × (scale_in × scale_w) / scale_out )

    ──saturate──► clamp to [-128, 127]
    ─────────────────────
    int8_t output[M]
```

**關鍵位元寬規則**：
- `int8 × int8 → int16`（乘積）
- `Σ int16 → int32`（累加，最多加 N 項）
- 在 HLS 中必須明確宣告累加器為 `int32_t` 以防止截斷

---

## Bit-accurate 驗證的關鍵細節（Week 8 踩過的坑）

### 坑 1：ReLU 放置位置
```cpp
// ❌ 錯誤（少一次 ReLU）：
causal_conv1d(...);  // conv1
pointwise_conv1d(...);  // conv2
// 直接進 residual merge → logits 全部偏掉

// ✅ 正確（conv2 後也要 ReLU）：
causal_conv1d(...);  // conv1 → ReLU
pointwise_conv1d(...);  // conv2
for (auto& value : acc2) {
    if (value < 0) value = 0;  // ← conv2 後的 ReLU
}
// 然後才做 residual merge
```

### 坑 2：Rounding 規則
```python
# ❌ 錯誤（Python floor 對負數行為不同）：
quantized = int(np.floor(value + 0.5))

# ✅ 正確（對齊 C++ 的 round()，即 half-away-from-zero）：
quantized = int(np.sign(value) * np.floor(np.abs(value) + 0.5))
```

**驗證成功標準**：
```
Python logits_int32 = [-3947, 220, -4732]
C++ logits_int32    = [-3947, 220, -4732]
np.array_equal(py_logits, cpp_logits) → True  ✅
```

---

## 固定常數定義（模板參數）

```cpp
// 來自 cpp_hls_prep/include/fixed_tcn.hpp

static constexpr int kFixedNumInputs = 12;    // 特徵數（features）
static constexpr int kFixedSeqLen    = 100;   // 序列長度（window size）
static constexpr int kFixedChannels  = 16;    // TCN 內部通道數
static constexpr int kFixedKernel    = 3;     // 卷積核大小
static constexpr int kFixedNumClasses = 3;   // 輸出類別（Down/Stationary/Up）
static constexpr int kFixedNumBlocks  = 3;   // TCN blocks 數量

// 各 block 的膨脹係數
static constexpr int kDilations[kFixedNumBlocks] = {1, 2, 4};
```

---

## 記憶體用量估算

| 元素 | 大小 |
|---|---|
| 輸入 (12×100 int8) | 1,200 bytes |
| 權重 (.wts, int8) | ~4,771 bytes ≈ 4.8 KB |
| 中間激活 (int8, 16×100) | 1,600 bytes |
| 最終輸出 (int32×3 + float×3) | 24 bytes |
| **總 on-chip 需求（估計）** | **< 10 KB** |

> 備註：4.8 KB 的權重完全適合放在 FPGA 的 BRAM 中（xc7z020 有 4.9 MB BRAM）。BRAM 超限的問題不在這裡，而在於中間激活計算時的多個 buffer。
