# Prompt：PHC-lite HLS 第七輪資源修正 — 移除 helper 函式的 ARRAY_PARTITION 和 UNROLL

> 給 coding agent 的自包含任務說明。直接複製整份貼給 agent，不需要補充對話脈絡。

---

## 背景與現況

目前 csynth 數字（v6，已刪除頂層 PIPELINE pragma）：
- DSP: 545 / 220 = **247%**（超出，根因在 helper 函式）
- LUT: 60,378 / 53,200 = **113%**（差 7,178）
- FF: 92,260 / 106,400 = **86%** ✅
- BRAM: 96 / 280 = **34%** ✅
- Latency: 489 cycles（4.89 µs），II=490

cosim 已通過：`total=25600 mismatches=0, max_abs_diff=0.00725275, PASS`

DSP 247% 的根因：helper 函式（linear、layernorm、sigmoid_16 等）的
`#pragma HLS ARRAY_PARTITION complete dim=1` 讓所有參數陣列同時可被存取，
結合各 loop 仍有 `#pragma HLS UNROLL factor=2` 或 `factor=4`，
HLS 為每個 call site 建立獨立的 loop pipeline 模組，各有自己的乘法器，無法跨模組共用。

---

## 需要修改的地方

### 修改 1：四個 linear 函式（`linear_16_16_fixed` / `linear_16_16_acc` / `linear_16_16_fixed_no_bias` / `linear_16_2_acc`）

對每一個函式，刪除以下所有 ARRAY_PARTITION 行（共 3~4 行）：
```cpp
// 以下全部刪除：
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=w cyclic dim=2 factor=2
#pragma HLS ARRAY_PARTITION variable=b complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
```

刪除外層迴圈的 UNROLL：
```cpp
// 刪除（linear_16_16_* 的外層 j 迴圈）：
#pragma HLS UNROLL factor=2

// 刪除（linear_16_2_acc 的外層 j 迴圈）：
#pragma HLS UNROLL
```

保留：`#pragma HLS INLINE`，保留內層 `for (int i = 0; i < 16; ++i)` 迴圈（已無 UNROLL）。

### 修改 2：`layernorm_16`

刪除所有 ARRAY_PARTITION：
```cpp
// 以下全部刪除：
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=gamma complete dim=1
#pragma HLS ARRAY_PARTITION variable=beta complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
```

刪除三個迴圈的 UNROLL（mean 累加、var 累加、正規化輸出）：
```cpp
// 以下全部刪除：
#pragma HLS UNROLL factor=4   // mean 累加迴圈
#pragma HLS UNROLL factor=4   // var 累加迴圈
#pragma HLS UNROLL factor=4   // 正規化輸出迴圈
```

保留：`#pragma HLS INLINE`，保留迴圈本身及 `hls::sqrt` 呼叫。

### 修改 3：`sigmoid_16`、`tanh_16`、`silu_16`

刪除 ARRAY_PARTITION：
```cpp
// 以下全部刪除（三個函式各自）：
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
```

若仍有 `#pragma HLS UNROLL factor=4` 也一併刪除。

保留：`#pragma HLS INLINE`。

### 修改 4：`phc_step` 函式主體內的 element-wise 迴圈

phc_step 主體內有多個 element-wise loop 帶有 `#pragma HLS UNROLL factor=4`，全部刪除：

```cpp
// 以下所有位置（約 line 281–312）的 #pragma HLS UNROLL factor=4 全部刪除：
for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL factor=4    // ← 刪除這行
    state_temporal[i] = ...;
}

for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL factor=4    // ← 刪除這行
    gain_pre[i] = ...;
}

for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL factor=4    // ← 刪除這行
    loop_input[i] = ...;
}

for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL             // ← 刪除這行（j<2 的 readout 迴圈）
    readout[i] = ...;
}

for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL factor=4    // ← 刪除這行
    state_next[i] = ...;
}
```

---

## 絕對不動的部分

- `tanh_lut_scalar` 內的 `#pragma HLS ARRAY_PARTITION variable=TANH_LUT complete dim=1`（保留——這是 runtime index 選取所需的）
- `tanh_lut_scalar`、`sigmoid_scalar`、`tanh_scalar` 的 `#pragma HLS INLINE`（保留）
- TANH_LUT[65] 的數值（不動）
- `phc_weights.hpp` 的所有權重（不動）
- `layernorm_16` 的 `hls::sqrt(var + LN_EPS)`（不動）

---

## 執行流程

從 `n2ew_era/06f_phc_lite_manual_hls/hls_component/` 目錄執行：

```bash
vitis-run --mode hls --syn  --config hls_config.cfg --work_dir hls_component
vitis-run --mode hls --csim --config hls_config.cfg --work_dir hls_component
```

若 csynth DSP < 220 且 LUT < 53,200，再跑：
```bash
vitis-run --mode hls --cosim --config hls_config.cfg --work_dir hls_component
```

---

## 驗收標準

1. DSP < 220（100%）、LUT < 53,200（100%）、FF < 106,400（100%）
2. csim PASS，`mismatches=0`，`max_abs_diff` ≤ 0.015
3. cosim PASS（資源通過後才跑）
4. compile log 無 "partially UNROLL is converted into complete unroll" warning

## 回報格式

- LUT / FF / DSP / BRAM 利用率（數字 + 百分比）
- Latency / II
- csim 的 max_abs_diff / mean_abs_diff
- compile log 有無新的 ERROR 或關鍵 WARNING
