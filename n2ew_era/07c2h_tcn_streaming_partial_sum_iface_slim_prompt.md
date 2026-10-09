# Prompt：TCN streaming 第四輪 — 把 partial-sum 變體套上 07C2D 的介面瘦身，解決 cosim OOM

> 給 coding agent 的自包含任務說明。直接複製整份貼給 agent，不需要補充對話脈絡。
> 工作目錄：`n2ew_era/07c2b_tcn_streaming_speed/`。這是 `AUDIT_tcn_stage2_inner_pipeline_partialsum.md`（round 3, `innerpipe_partialsum`）的後續修正，csynth 已經過關（50.36us, 1.48x），這輪只解決 cosim OOM，不碰 accumulator 結構本身。

---

## 上一輪結果（round 3, partial-sum）——csynth 是真的過了，但 cosim 記憶體爆了

`distilled_shared_u4_innerpipe_partialsum`：把 accumulator 的 3 級連續 `acc +=` 依賴鏈拆成部分和之後，csynth **通過**，估計時脈 7.040ns（Fmax 142.05MHz），latency/interval 從 baseline 的 7,438/7,439 cycles 降到 5,036/5,037 cycles（74.38us → 50.36us，1.48x）。DSP/LUT/FF/BRAM 都在預算內。這是這條實驗線第一次 csynth 真的過關。

但 RTL cosim 沒有跑完：XSIM 跑到 918/1012 個 transaction 時記憶體耗盡：

```text
Out of memory on request for a fresh 8388608 bytes
Total memory consumed so far :3931333132 bytes
ERROR: [COSIM-4] *** C/RTL co-simulation finished: FAIL ***
```

**這不是功能錯誤**，csim 數值結果跟前兩輪完全一致（sign_agreement_joint=1），純粹是 XSIM 模擬器記憶體不夠撐完整段 cosim。

## 根因已經查到：這個 variant 用的是舊的、沒瘦身過的頂層介面

已經直接比對過原始碼確認：

`07c2b_tcn_streaming_speed/hls_src/tcn_stage2_resource_impl.hpp` 目前的 `tcn_streaming_step` 頂層函式簽名，除了 `x` / `reset` / `readout` / `readout_valid` 之外，還把 5 個 block 的中間輸出（`block0_out`…`block4_out`）和對應的 valid flag（`block0_valid`…`block4_valid`）**全部暴露成頂層 RTL 埠**（10 個額外 `ap_memory` 介面 + `ARRAY_PARTITION`）。這正是 `07c2c_tcn_streaming_cosim_diagnostic/COSIM_SLOWDOWN_DIAGNOSTIC.md` 診斷出來、並在 `07c2d_tcn_streaming_iface_slim/AUDIT_tcn_streaming_iface_slim.md` 裡實測修好的同一種介面膨脹問題：

| design | interface rows | `ap_ovld` rows | autowrap bytes |
|---|---:|---:|---:|
| 07C2C 診斷版頂層（跟目前 07C2B 這個介面同一類，都暴露 5 個 block 診斷輸出） | 279 | 240 | 116,334 |
| 07C2D 瘦身後頂層（只留 x/reset/readout/readout_valid） | 29 | 0 | 39,664 |

07C2D 把診斷輸出從頂層介面拿掉之後，`ap_ovld` 相關列從 240 降到 0，autowrap 體積縮小 2.93 倍，cosim 記憶體壓力大幅下降，成功跑完並進到 P&R。

**這一輪的 round 3 partial-sum 變體是直接接在 07C2B 這條線上做的，從來沒有套用過 07C2D 的介面瘦身**——所以它在 cosim 階段撞到跟 07C2C 同一種記憶體爆炸，只是因為內部結構已經不同（partial-sum 排程改變了硬體），跑到 918/1012 才爆，不是一開始就爆。這不是新問題，是舊問題還沒補上。

## 這次要做什麼：把 07C2D 的介面瘦身，套用到 round 3 的 partial-sum 版本上

**只改介面，不改 accumulator 結構**（round 3 的 partial-sum 拆解全部保留，不要動）。做法完全比照 `07c2d_tcn_streaming_iface_slim/hls_src/tcn_stage2_resource_impl.hpp` 已經驗證過的模式：

### 1. 頂層函式簽名收斂成只有 4 個參數

原本（`hls_src/tcn_stage2_resource_impl.hpp:250-264`）：

```cpp
void tcn_streaming_step(
    const tcn_fixed_t x[TCN_CHANNELS],
    bool reset,
    tcn_fixed_t readout[TCN_READOUT_DIM],
    bool readout_valid[1],
    tcn_fixed_t block0_out[TCN_CHANNELS],
    bool block0_valid[1],
    tcn_fixed_t block1_out[TCN_CHANNELS],
    bool block1_valid[1],
    tcn_fixed_t block2_out[TCN_CHANNELS],
    bool block2_valid[1],
    tcn_fixed_t block3_out[TCN_CHANNELS],
    bool block3_valid[1],
    tcn_fixed_t block4_out[TCN_CHANNELS],
    bool block4_valid[1]
) {
```

改成（照抄 07C2D 的簽名）：

```cpp
void tcn_streaming_step(
    const tcn_fixed_t x[TCN_CHANNELS],
    bool reset,
    tcn_fixed_t readout[TCN_READOUT_DIM],
    bool readout_valid[1]
) {
```

### 2. `block0_out`…`block4_out` 改成函式內部的區域陣列，不再是參數

在函式開頭（原本 `readout_ring` 宣告附近）新增區域陣列宣告，取代原本從參數傳進來的版本：

```cpp
    tcn_fixed_t block0_out[TCN_CHANNELS];
    tcn_fixed_t block1_out[TCN_CHANNELS];
    tcn_fixed_t block2_out[TCN_CHANNELS];
    tcn_fixed_t block3_out[TCN_CHANNELS];
    tcn_fixed_t block4_out[TCN_CHANNELS];
#pragma HLS ARRAY_PARTITION variable=block0_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=block1_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=block2_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=block3_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=block4_out complete dim=1
```

`v0`…`v4`（原本回寫到 `block0_valid[0]`…`block4_valid[0]` 的那五個 bool）維持原本的區域變數用法（`stage2_resource_stream_block_step` 呼叫方式完全不變），只是最後不要再寫回任何頂層輸出埠——直接把 `block0_valid[0] = v0;` 到 `block4_valid[0] = v4;` 這五行，以及 `reset` 分支裡對應的 `block0_valid[0] = false;`…`block4_valid[0] = false;` 五行整段刪掉（`hls_src/tcn_stage2_resource_impl.hpp:330-334` 與 `:350-354` 兩處）。

### 3. INTERFACE / ARRAY_PARTITION pragma 收斂

原本 `:266-288` 那整段，刪掉所有 `port=block0_out` 到 `port=block4_valid` 的 `INTERFACE ap_memory` 行，以及對應的 `ARRAY_PARTITION variable=block0_out`…`block4_out` 行（這幾行改放到步驟 2 新增的區域陣列宣告下方，如上）。收斂後只留：

```cpp
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS INTERFACE ap_memory port=x
#pragma HLS INTERFACE ap_none port=reset
#pragma HLS INTERFACE ap_memory port=readout
#pragma HLS INTERFACE ap_memory port=readout_valid

#pragma HLS ARRAY_PARTITION variable=x complete dim=1
#pragma HLS ARRAY_PARTITION variable=readout complete dim=1
```

### 4. CSim 除錯用的 block 輸出比對：比照 07C2D 的 `#ifndef __SYNTHESIS__` 做法

07C2D 用一個只在 CSim 模式編譯的 helper（`stage2_resource_store_debug_block`，guard 在 `#ifndef __SYNTHESIS__`）把 block 中間值寫到非 RTL 介面的除錯用結構，讓 csim 還能比對 block0-4 的數值（跟 round 2/3 之前的 per-stage 比對表一樣），但不會把這些變成 RTL 頂層埠。**先看 `07c2d_tcn_streaming_iface_slim/hls_src/tcn_stage2_resource_impl.hpp` 完整的這個 helper 定義和呼叫方式，原封不動搬過來用**，確保這輪的 per-stage CSim 比對（block0…block4 的 max_abs_diff/rmse 那張表）不會因為介面瘦身而消失。

### 5. Testbench / wrapper 呼叫端同步修正

`hls_src/tcn_s2_dsu4_ip_ps.cpp`（round 3 用的短檔名 wrapper）本身不用改（它只是 `#define` + `#include`）。但任何直接呼叫 `tcn_streaming_step(x, reset, readout, readout_valid, block0_out, block0_valid, ...)` 十四參數版本的 testbench/harness 檔案，都要跟著改成四參數呼叫——**先搜尋 `07c2b_tcn_streaming_speed/` 底下所有呼叫 `tcn_streaming_step(` 的地方**，比照 07C2D 的 testbench 改法（07C2D 的 testbench 已經是四參數呼叫，可以直接參考它怎麼拿 block 中間值做比對——應該是透過步驟 4 的 CSim-only debug helper，不是透過函式參數）。

---

## 絕對不動的部分

- Round 3 已經驗證過關的 partial-sum accumulator 重構（`tap_m2/tap_m1/tap_cur/tap_sum` 那段，CONV1 在 `:165-169`、CONV2 在 `:227-231`）——一個字元都不要動。
- `TCN_STAGE2_INNER_PIPELINE`、`TCN_STAGE2_CHANNEL_UNROLL`、`TCN_STAGE2_SHARED_HISTORY_STORAGE`、`INLINE off`、history ring buffer——都不動。
- 外層 oc 迴圈維持 `PIPELINE off`，不要嘗試 pipeline。
- 不要動 `07c2d_tcn_streaming_iface_slim/` 既有凍結產物，只能讀取參考，不能修改。
- 不要重新訓練、不要碰 NEW5。

## 執行流程

```bash
cd n2ew_era/07c2b_tcn_streaming_speed/
python 01_prepare_speed_components.py --variant distilled_shared_u4_innerpipe_partialsum_ifaceslim --clean
```

（沿用 round 3 一樣的短路徑/短檔名處理方式，避免 Windows `mingw32-make no rule` 路徑問題——如果新檔名還是太長導致同樣的 make 錯誤，比照 round 3 做法另外建一個短檔名 wrapper。）

依序：csim → csynth（`vitis-run.bat --mode hls --tcl --input_file run_csynth_only.tcl`）→ 確認時序/資源沒有劣化（跟 round 3 的 7.040ns / 5,036 cycles / DSP 100 / LUT 34,089 / FF 23,579 / BRAM 180 比對，介面瘦身理論上不該讓這些數字變差，甚至可能因為少了診斷埠的 partition/routing 負擔而略為改善）→ cosim。

**這輪的重點驗收在 cosim 階段**：如果介面瘦身假說正確，cosim 應該可以跑完全部 1012 個 transaction 而不是卡在 918。如果還是 OOM（無論卡在哪個 transaction），代表記憶體問題不是介面埠數量造成的，需要另外查（例如降低 testbench 一次跑的 event 數量，比照 07C2C 曾經用過的 174-event 縮小規模 cosim 當作中繼驗證點）。

## 驗收標準

1. 介面確認：新版 `tcn_streaming_step` 是否真的只剩 4 個參數，`ap_ovld`/診斷埠是否從 synthesis log 或 interface report 裡消失（可以比照 07C2D audit 用的 harness complexity 表格，數 autowrap bytes / TV markers / interface rows / `ap_ovld` rows）。
2. csynth 數字是否維持 round 3 水準（5,036/5,037 cycles, 7.040ns, DSP 100/LUT 34,089/FF 23,579/BRAM 180 附近），不應該因為介面瘦身而變差；如果變好或變差都要如實回報並解釋可能原因。
3. cosim 是否真的跑完 1012/1012 transaction，還是仍在某個 transaction 卡住記憶體不足——兩種結果都要回報，不要只回報成功案例。
4. csim per-stage 數值比對（block0-4 + readout 的 max_abs_diff/rmse/sign_agreement）是否跟 round 3 完全一致，確認介面瘦身沒有動到任何計算邏輯。

## 回報格式

- 介面瘦身前後的埠數量/`ap_ovld` 數量對比表。
- csynth 對比表（round 3 vs 這輪）。
- cosim 結果：完整跑完/卡在第幾個 transaction/記憶體用量，附 log 證據路徑。
- 明確結論句：「介面瘦身（有/沒有）解決 partial-sum 變體的 cosim OOM」，如果解決了，這個 variant 就會是 07C2B 這條速度優化線第一個同時通過 csynth 與完整 cosim 的版本，離下一步（送 07C2D 路線做完整 P&R 封版）只差 implementation 這一關。
