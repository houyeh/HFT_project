# Prompt：PHC-lite HLS 第八輪資源修正 — 加 `#pragma HLS PIPELINE off` 阻止自動 pipeline

> 給 coding agent 的自包含任務說明。直接複製整份貼給 agent，不需要補充對話脈絡。

---

## 背景與現況

目前 csynth 數字（v7，Mon Jun 15 01:44:28，已移除全部 ARRAY_PARTITION 與 UNROLL pragma）：
- DSP: 268 / 220 = **121%**（仍超出）
- LUT: 41,433 / 53,200 = **77%** ✅
- FF: 55,719 / 106,400 = **52%** ✅
- BRAM: 0 / 280 = **0%** ✅
- Latency: 604 cycles（6.04 µs），II=605

cosim 狀態：尚未跑（DSP 超出先不跑）

---

## 根本原因（已確診，不要猜）

compile log 顯示三條 INFO，形成連鎖反應：

```
INFO: [HLS 214-376] Pipelining loop<VITIS_LOOP_114_1> at phc_step_hls.cpp:114:23 due to pipeline_loops threshold
INFO: [HLS 214-291] Loop 'VITIS_LOOP_116_2' is marked as complete unroll implied by the pipeline pragma (phc_step_hls.cpp:116:27)
INFO: [HLS 214-449] Partitioning array 'phc_step_weights::SYNAPSE_W' dimension 2 due to constant index.
```

步驟：
1. HLS `pipeline_loops` 閾值把 16 次的 outer j-loop (line 114) **自動 pipeline**
2. outer pipeline 強制 inner i-loop (line 116) **完全展開 factor=16** → 16 個並行乘法器
3. 完全展開後 w[j][i] 的 j 在每個 pipeline cycle 為常數，HLS **自動 partition 權重矩陣 dim=2 complete**

這與之前手動加 `PIPELINE II=1` 的效果完全一樣，只是現在由工具自動觸發。
手動移除 ARRAY_PARTITION / UNROLL 無效，因為工具會自己把它們加回來。

---

## 修改方式

### 唯一有效的修法：`#pragma HLS PIPELINE off`

在每個受影響的迴圈的第一行加上這個 pragma，**明確告訴 HLS 不要 pipeline 這個 loop**。

### 需要修改的位置（逐一列出）

#### 1. `linear_16_16_fixed`（約 line 98–104）

```cpp
void linear_16_16_fixed(...) {
#pragma HLS INLINE
    for (int j = 0; j < 16; ++j) {
#pragma HLS PIPELINE off         // ← 新增這行
        phc_acc_t acc = (phc_acc_t)b[j];
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off         // ← 新增這行
            acc += (phc_acc_t)w[j][i] * (phc_acc_t)in[i];
        }
        out[j] = acc;
    }
}
```

#### 2. `linear_16_16_acc`（約 line 114–120）

同上，outer j-loop 與 inner i-loop 各加一行 `#pragma HLS PIPELINE off`。

#### 3. `linear_16_16_fixed_no_bias`（約 line 129–135）

同上。

#### 4. `linear_16_2_acc`（約 line 145–151）

```cpp
void linear_16_2_acc(...) {
#pragma HLS INLINE
    for (int j = 0; j < 2; ++j) {
#pragma HLS PIPELINE off         // ← 新增這行
        phc_acc_t acc = (phc_acc_t)b[j];
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off         // ← 新增這行
            acc += (phc_acc_t)w[j][i] * in[i];
        }
        out[j] = acc;
    }
}
```

#### 5. `layernorm_16`（三個 for loop，約 line 162, 168, 175）

```cpp
void layernorm_16(...) {
#pragma HLS INLINE
    phc_acc_t mean = 0;
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off         // ← 新增這行（mean 累加）
        mean += in[i];
    }
    // ... mean /= 16 ...
    phc_var_t var = 0;
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off         // ← 新增這行（var 累加）
        phc_acc_t d = in[i] - mean;
        var += (phc_var_t)d * (phc_var_t)d;
    }
    // ... var /= 16, sqrt ...
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off         // ← 新增這行（normalize 輸出）
        out[i] = (phc_acc_t)gamma[i] * (in[i] - mean) * inv_std + (phc_acc_t)beta[i];
    }
}
```

#### 6. `sigmoid_16`、`tanh_16`、`silu_16`（各一個 for loop）

```cpp
void sigmoid_16(...) {
#pragma HLS INLINE
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off         // ← 新增這行
        out[i] = sigmoid_scalar(in[i]);
    }
}
// tanh_16 和 silu_16 同理，各自的 for loop 第一行加 #pragma HLS PIPELINE off
```

#### 7. `phc_step` 主體內的 element-wise 迴圈（約 line 219–245）

```cpp
// state_temporal 計算迴圈
for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off         // ← 新增這行
    state_temporal[i] = (phc_acc_t)DECAY[i] * (phc_acc_t)state_prev[i] + gate_t[i] * candidate_t[i];
}

// gain_pre 加法迴圈（沒有乘法，加不加都可，但加了更保險）
for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off         // ← 新增這行
    gain_pre[i] = gain_encoded[i] + gain_prev_state[i];
}

// loop_input 計算迴圈
for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off         // ← 新增這行
    loop_input[i] = state_temporal[i] + (phc_acc_t)ALPHA_SCALAR[0] * gain_t[i] * message[i];
}

// readout 轉型迴圈
for (int i = 0; i < 2; ++i) {
#pragma HLS PIPELINE off         // ← 新增這行
    readout[i] = (phc_fixed_t)readout_acc[i];
}

// state_next 寫回迴圈
for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off         // ← 新增這行
    state_next[i] = (phc_fixed_t)state_temporal[i];
}
```

---

## 絕對不動的部分

- `tanh_lut_scalar` 內的 `#pragma HLS ARRAY_PARTITION variable=TANH_LUT complete dim=1`（保留）
- `tanh_lut_scalar`、`sigmoid_scalar`、`tanh_scalar` 的 `#pragma HLS INLINE`（保留）
- 所有函式的 `#pragma HLS INLINE`（保留）
- TANH_LUT[65] 的數值、phc_weights.hpp 的所有權重（不動）
- `hls::sqrt(var + LN_EPS)`（不動）

---

## 執行流程

從 `n2ew_era/06f_phc_lite_manual_hls/hls_component/` 目錄：

```bash
vitis-run --mode hls --syn  --config hls_config.cfg --work_dir hls_component
vitis-run --mode hls --csim --config hls_config.cfg --work_dir hls_component
```

若 DSP < 220 且 LUT < 53,200，再跑：
```bash
vitis-run --mode hls --cosim --config hls_config.cfg --work_dir hls_component
```

---

## 驗收標準

1. **compile log 中不再出現 `[HLS 214-376]` "due to pipeline_loops threshold" 在 linear 的 loop 上**（這是修正生效的信號）
2. **compile log 中不再出現 `[HLS 214-291]` "complete unroll implied by the pipeline pragma" 在 linear 的 inner loop 上**
3. **DSP < 220（100%）**，目標 < 110（50%）
4. **LUT < 53,200（100%）**
5. **csim PASS**，`mismatches=0`，`max_abs_diff` ≤ 0.015
6. **cosim PASS**（資源通過後才跑）

## 回報格式

- LUT / FF / DSP / BRAM 利用率
- Latency / II（預期增加到 ~2,000–5,000 cycles，這是正常的）
- csim max_abs_diff / mean_abs_diff
- compile log 有無新的 ERROR 或關鍵 WARNING
- 有無 `[HLS 214-376]` 在 linear loop 上的 pipeline_loops 自動行為
