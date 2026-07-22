# Prompt：修正 PHC-lite HLS 矩陣乘法展開過度問題（第二輪資源修正）

> 給 coding agent 的自包含任務說明。直接複製整份貼給 agent，不需要額外補充對話脈絡。

---

## 背景與現況（已驗證，非推測）

`n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.cpp` 已在第一輪修正中將
雙精度浮點活化函數（`(double)` 轉型 + `generic_tanh_double_s`）替換為 65 點 LUT-based
tanh 近似，csim/cosim 均通過：

- C-Sim: `total=25600 mismatches=0, max_abs_diff=0.00647, mean_abs_diff=0.00025`
- Cosim (Jun 14): `PASS, total=25600 mismatches=0, max_abs_diff=0.00725, mean_abs_diff=0.000373`
- Csynth latency: 297 cycles @ 100 MHz, II=1

但第二次 csynth 的資源仍超出 xc7z020clg400-1 的實體容量：

| 資源 | 用量 | 可用量 | 利用率 |
|---|---|---|---|
| LUT | 446,494 | 53,200 | **839%** |
| FF | 469,913 | 106,400 | **441%** |
| DSP | 3,528 | 220 | **1,603%** |

Instance list 已**不含任何** `generic_tanh_double_s` 或 `generic_sqrt_double_s`（浮點核心問題已解決），
全部都是 `mul_*` 定點乘法器（例如 `mul_26s_40s_64_2_1_U1` ~ `_U80` 共 80 個，×4 DSP 各 = 320 DSP 僅此一型）。

## 新的根本原因

問題在 `linear_16_16_fixed`、`linear_16_16_acc`、`linear_16_16_fixed_no_bias` 三個函式，
以及 `linear_16_2_acc`，**outer 與 inner 兩層迴圈都加了 `#pragma HLS UNROLL factor=4`**：

```cpp
// 目前寫法（有問題）
for (int j = 0; j < 16; ++j) {
#pragma HLS UNROLL factor=4    // outer: 4 個輸出神經元並行
    phc_acc_t acc = (phc_acc_t)b[j];
    for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL factor=4    // inner: 4 個輸入元素並行 ← 這裡是主要問題
        acc += (phc_acc_t)w[j][i] * (phc_acc_t)in[i];
    }
    out[j] = acc;
}
```

兩層各展開 4 倍 = 每次函式呼叫產生 **4×4=16 個並行乘法器**。
所有 linear 函式都 `#pragma HLS INLINE`，全部展平到 `phc_step` 的同一排程空間中。
6~7 個 linear layer 呼叫 × 16 個並行乘法器 = **100+ 個同時存在的乘法器單元**，
加上 layernorm、silu_16 等 element-wise 運算的乘法 → csynth 合計 882 個乘法器實例。

## 你需要做的修正

### 步驟 1：修改 `phc_step_hls.cpp` — 移除所有 linear 函式的 inner loop UNROLL

對 **所有四個** linear 函式（`linear_16_16_fixed`、`linear_16_16_acc`、
`linear_16_16_fixed_no_bias`、`linear_16_2_acc`）：

**刪除** inner loop 的 `#pragma HLS UNROLL factor=4`：

```cpp
// 修正後寫法
for (int j = 0; j < 16; ++j) {
#pragma HLS UNROLL factor=2    // outer 從 factor=4 降為 factor=2（保留少量並行）
    phc_acc_t acc = (phc_acc_t)b[j];
    for (int i = 0; i < 16; ++i) {
        // 不加任何 UNROLL pragma（讓 HLS scheduler 決定，sequential）
        acc += (phc_acc_t)w[j][i] * (phc_acc_t)in[i];
    }
    out[j] = acc;
}
```

- Outer loop: `factor=4` → **`factor=2`**（減半並行輸出）
- Inner loop: 完全**移除** `#pragma HLS UNROLL factor=4`

對 `linear_16_2_acc`（只有 2 個輸出）：
```cpp
for (int j = 0; j < 2; ++j) {
#pragma HLS UNROLL    // j 只有 2，完全展開 2 個沒問題
    phc_acc_t acc = (phc_acc_t)b[j];
    for (int i = 0; i < 16; ++i) {
        // 不加任何 UNROLL（移除 inner factor=4）
        acc += (phc_acc_t)w[j][i] * in[i];
    }
    out[j] = acc;
}
```

### 步驟 2：降低 element-wise 函式的 UNROLL

`sigmoid_16`、`tanh_16`、`silu_16` 三個函式裡的 `for (int i = 0; i < 16; ++i)` 迴圈目前是 `factor=4`，改為不加 UNROLL（sequential）：

```cpp
void sigmoid_16(const phc_acc_t in[16], phc_acc_t out[16]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
    for (int i = 0; i < 16; ++i) {
        // 移除 #pragma HLS UNROLL factor=4
        out[i] = sigmoid_scalar(in[i]);
    }
}
// tanh_16 和 silu_16 同理
```

### 步驟 3：保留不動的部分

以下這些**不要改**：

- `#pragma HLS PIPELINE II=1` 在頂層 `phc_step`（保留）
- 所有 `#pragma HLS ARRAY_PARTITION variable=... complete dim=1`（保留，這是資料路徑需要的）
- 所有 `#pragma HLS INLINE`（保留）
- `layernorm_16` 內部的三個迴圈（`mean`/`var`/`output`）各自的 `#pragma HLS UNROLL factor=4` 改成不加（sequential），但 `var` 的加法迴圈保留（它是 phc_var_t 加法，不是乘法，代價小）
- tanh_lut_scalar 內的 `#pragma HLS ARRAY_PARTITION variable=TANH_LUT complete dim=1`（保留）
- TANH_LUT[65] 的值不要動

### 步驟 4：重新跑 csim → csynth

從 **`n2ew_era/06f_phc_lite_manual_hls/hls_component/`** 目錄執行：

```bash
# C-Sim
vitis-run --mode hls --csim --config hls_config.cfg --work_dir hls_component

# 確認 csim PASS 且 max_abs_diff 數字在合理範圍後，跑 csynth
vitis-run --mode hls --syn --config hls_config.cfg --work_dir hls_component
```

hls_config.cfg 的 `csim.argv` 與 `cosim.argv` 已設定好絕對路徑，不需要修改。

### 步驟 5：若 csynth 資源通過，再跑 cosim

```bash
vitis-run --mode hls --cosim --config hls_config.cfg --work_dir hls_component
```

## 驗收標準（DONE 的定義）

1. **資源通過**：`phc_step_csynth.rpt` 顯示 DSP < 220（100%），LUT < 53,200（100%）。
   理想目標：DSP < 110（50%），LUT < 26,600（50%）。
2. **數值正確性未退化**：C-Sim PASS，`max_abs_diff` ≤ 0.010（前次基準 0.00725），`mismatches=0`。
3. **Cosim PASS**（若資源通過才跑）。
4. **回報前後對照**：
   - LUT / FF / DSP 利用率（修正前 vs 修正後）
   - latency / II（新的 cycle 數，預期會從 297 增加到 1000~3000 cycles，這是預期的取捨）
   - max_abs_diff / mean_abs_diff

## 紅線提醒

- **不要**升高 `PHC_ABS_TOL` 來遮蓋誤差（testbench 有明確警告）。
- **不要**改動 TANH_LUT 的值或 tanh_lut_scalar / sigmoid_scalar 的邏輯。
- **不要**改動 phc_weights.hpp（權重值不動）。
- 如果 csim `max_abs_diff` 超過 0.02，回報給我而不是自行調參，先找到原因。
- Latency 增加是**預期的**，不是 bug。只要 II=1 維持，對 HFT pipeline throughput 無影響。

## 資源估算（供參考，修完後以實際 csynth 為準）

移除 inner UNROLL 後，每個 linear 函式從 16 個並行乘法器變為 2 個（outer factor=2），
6 個 linear layer × 2 = 12 個並行乘法器，加上其他運算估計總 DSP ~30~80，
應可落在 xc7z020 的 220 DSP 以內。
