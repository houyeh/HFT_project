# Prompt：PHC-lite HLS 第三輪資源修正 — 改用 cyclic ARRAY_PARTITION（保留 PIPELINE II=1）

> 給 coding agent 的自包含任務說明。直接複製整份貼給 agent，不需要額外補充對話脈絡。

---

## 背景與現況

`n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.cpp` 已經過兩輪修正：

- **第一輪**：雙精度浮點 → 65 點 LUT tanh，DSP 從 2636% → 1603%
- **第二輪**：移除 inner loop UNROLL、outer 從 factor=4 → factor=2，但資源**完全沒有改變**

第二輪失敗的根本原因已確診：**UNROLL pragma 對已完全 partition 的陣列無效**。

在 `linear_16_16_*` 等函式裡，16×16 的權重矩陣有 `#pragma HLS ARRAY_PARTITION variable=w complete dim=0`，這讓所有 256 個元素同時引出獨立硬體埠。只要頂層函式有 `#pragma HLS PIPELINE II=1`，Vitis HLS 就**自動把所有 256 個乘法都排進同一個 pipeline**，UNROLL 與否完全不影響結果。

目前 csynth 結果（第三次，Jun 14 03:30:57）仍為：

| 資源 | 用量 | 可用量 | 利用率 |
|---|---|---|---|
| LUT | 446,494 | 53,200 | **839%** |
| FF | 469,913 | 106,400 | **441%** |
| DSP | 3,528 | 220 | **1,603%** |

---

## 真正有效的修法

### 核心修改：weight 矩陣從 `complete dim=0` 改為 `cyclic dim=2 factor=2`

`complete dim=0` = 所有維度全部打開 = 256 路並行讀取 = 256 個同時運作的乘法器。

`cyclic dim=2 factor=2` = 只沿第 2 維（列方向，即 `w[j][i]` 的 `i` 軸）做 2-bank 輪流分配：
- Bank 0：偶數列（i=0,2,4,...,14）
- Bank 1：奇數列（i=1,3,5,...,15）
- 每個 cycle 只能讀 2 個元素 → inner loop 需要 8 cycles 而非 1 cycle
- HLS 因此只需要 2 個並行乘法器（+ outer UNROLL factor=2 = 共 4 個/函式）

---

## 需要修改的地方（逐一列出）

### 修改 1：`linear_16_16_fixed`（約 line 100–116）

```cpp
// 修改前
#pragma HLS ARRAY_PARTITION variable=w complete dim=0

// 修改後
#pragma HLS ARRAY_PARTITION variable=w cyclic dim=2 factor=2
```

outer loop 維持 `#pragma HLS UNROLL factor=2`，inner loop 不加任何 UNROLL。

### 修改 2：`linear_16_16_acc`（約 line 118–137）

同上，`w` 的 ARRAY_PARTITION 由 `complete dim=0` 改為 `cyclic dim=2 factor=2`。

### 修改 3：`linear_16_16_fixed_no_bias`（約 line 139–162）

同上。

### 修改 4：`linear_16_2_acc`（約 line 164–184）

此函式的 weight 是 `w[2][16]`，dim=2 同樣是 i 軸：

```cpp
// 修改前
#pragma HLS ARRAY_PARTITION variable=w complete dim=0

// 修改後
#pragma HLS ARRAY_PARTITION variable=w cyclic dim=2 factor=2
```

outer loop（`j < 2`）保留 `#pragma HLS UNROLL`（完全展開 2 個輸出），inner loop 不加 UNROLL。

### 修改 5：`sigmoid_16`、`tanh_16`、`silu_16` 的 element-wise loop

這三個函式目前還有 `#pragma HLS UNROLL factor=4`（上一輪沒改到）。
雖然這些不是 DSP 的來源，但是 tanh_lut_scalar 的 65 點 LUT 在 4 路並行下會複製 4 份，
大幅增加 Expression LUT。改為不加 UNROLL（sequential 16 iterations）：

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
// tanh_16、silu_16 同理
```

---

## 保留不動的部分

- `#pragma HLS PIPELINE II=1` 在頂層 `phc_step`（**必須保留，這是延遲最小化的關鍵**）
- 所有 1D 陣列（`in[16]`、`out[16]`、`b[16]` 等）的 `ARRAY_PARTITION complete dim=1`（保留）
- `tanh_lut_scalar` 內的 `ARRAY_PARTITION variable=TANH_LUT complete dim=1`（保留，但因 sigmoid_16 改為 sequential，只有 1 份 LUT 被同時用到）
- `#pragma HLS INLINE` 在所有 helper 函式（保留）
- TANH_LUT[65] 的數值（不動）
- `phc_weights.hpp` 中的所有權重（不動）
- `layernorm_16` 內的 UNROLL（保留 factor=4 在 mean/var/output 的 loop，因為加法不用 DSP）

---

## 執行流程

從 **`n2ew_era/06f_phc_lite_manual_hls/hls_component/`** 目錄執行：

```bash
# Step 1: C-Sim（先驗功能正確性）
vitis-run --mode hls --csim --config hls_config.cfg --work_dir hls_component

# Step 2: C-Synthesis（驗資源）
vitis-run --mode hls --syn --config hls_config.cfg --work_dir hls_component

# Step 3: 若 LUT < 100% 且 DSP < 100%，才跑 Cosim
vitis-run --mode hls --cosim --config hls_config.cfg --work_dir hls_component
```

hls_config.cfg 的 csim.argv / cosim.argv 已設好絕對路徑，不需要修改。

---

## 預期結果（供參考，以實際 csynth 為準）

| 資源 | 修改前（v3） | 預期修改後（v4） | 目標 |
|---|---|---|---|
| DSP | 3,528（1,603%） | **~40–80（18–36%）** | < 220（100%） |
| LUT | 446,494（839%） | **~20,000–40,000（38–75%）** | < 53,200（100%） |
| FF | 469,913（441%） | **~30,000–80,000（28–75%）** | < 106,400（100%） |
| Latency | 297 cycles（2.97 µs） | **~600–1,000 cycles（6–10 µs）** | < 15 µs |
| II | 1 cycle | **1–4 cycles** | 盡量小 |

Latency 增加是**預期的**（inner loop 從並行 → sequential），換來的是晶片實際可放入。

---

## 驗收標準

1. **csynth 資源通過**：DSP < 220（100%）、LUT < 53,200（100%）
2. **csim PASS**：`mismatches=0`，`max_abs_diff` ≤ 0.015（基準 0.00725，允許小幅退化）
3. **cosim PASS**（資源通過後才跑）
4. **回報前後對照**：
   - LUT / FF / DSP 利用率（v1 double / v2 LUT-tanh / v3 UNROLL fix（無效）/ v4 cyclic）
   - Latency / II（新的數字）
   - max_abs_diff / mean_abs_diff

---

## 紅線提醒

- **不要**移除 `#pragma HLS PIPELINE II=1` 在頂層 `phc_step`（這是保持低延遲的核心）
- **不要**升高 `PHC_ABS_TOL`
- 如果 csynth 顯示 II > 1（例如 II=2 或 II=4），**如實回報**，不要隱瞞
- 如果 HLS 警告 array partition bank conflict，**如實回報**，不要自行調整 factor
- `layernorm_16` 裡的 `hls::sqrt(var + LN_EPS)` 用的是 `hls::sqrt` on `phc_var_t`（ap_fixed），**不要改成 double**
