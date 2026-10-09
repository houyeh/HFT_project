# Prompt：TCN streaming — 只 pipeline 區塊內 oc 迴圈（block-level pipeline），不碰頂層 PIPELINE

> 給 coding agent 的自包含任務說明。直接複製整份貼給 agent，不需要補充對話脈絡。
> 工作目錄：`n2ew_era/07c2b_tcn_streaming_speed/`（沿用既有 speed-scan 基礎設施，不動 07c2d 的最終 closure 產物）。

---

## 背景與現況

07C2D（`n2ew_era/07c2d_tcn_streaming_iface_slim/AUDIT_tcn_streaming_iface_slim_pnr.md`）目前的封版數字：

| variant | clock | latency cycles | interval cycles | latency@clock | DSP | LUT | FF | BRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| distilled shared_u4 | 100MHz | 7438 | 7439 | 74.38µs | 49.09% | 32.00% | 14.44% | 23.57% |
| scratch shared_u4 | 90MHz | 7438 | 7439 | 82.64µs | 51.36% | 31.39% | 14.36% | 23.57% |

`interval ≈ latency`（II≈latency），代表這個電路目前完全沒有被 pipeline，行為上跟全序列執行沒有差別——即使 TCN 演算法本身是前饋、沒有 t 依賴 t-1 的問題。

## 根本原因（已確診，不要重猜）

`07c2b_tcn_streaming_speed/results/stage2_speed_pipeline_scan.csv` 與 `AUDIT_tcn_streaming_speed.md` 已經測過一次「整個函式套頂層 PIPELINE」（`TCN_STAGE2_TOP_PIPELINE=1`），結果被 Vitis **忽略**，hls.log 明確寫：

```
WARNING: [HLS 214-275] In loop 'VITIS_LOOP_100_3'(../hls_src/tcn_stage2_resource_impl.hpp:100:23),
pragma conflict between PIPELINE (../hls_src/tcn_stage2_resource_impl.hpp:210:1) and PIPELINE off
(../hls_src/tcn_stage2_resource_impl.hpp:102:1) in a sub-region. Ignoring PIPELINE to honor PIPELINE off
```

原因：`tcn_streaming_step` 頂層的 `#pragma HLS PIPELINE`（原檔案約 line 210 附近）跟它呼叫的 `stage2_resource_stream_block_step`（`#pragma HLS INLINE off`，函式本身不會被展開進呼叫端）內部大量的 `#pragma HLS PIPELINE off` 衝突，Vitis 選擇尊重內層的 `off`，頂層 pipeline 從未真正生效。csynth top report 仍是 `PipelineType=no`，interval 維持 7439 不變。**這是一次"pragma 衝突診斷"，不是"pipeline 沒有效果"的證據**——AUDIT_tcn_streaming_speed.md 原文自己也這樣寫。

## 這次要測什麼、為什麼選這個範圍

不要再嘗試頂層 PIPELINE（會撞同一個衝突）。改成**只在 `stage2_resource_stream_block_step` 內部，把兩個熱區迴圈（CONV1 的 oc 迴圈、CONV2 的 oc 迴圈）從 `PIPELINE off` 換成 `PIPELINE II=1`**，其餘所有迴圈（history ring buffer 搬移、inner ic 迴圈的 UNROLL factor）維持原樣不動。理由：

- 這兩個 oc 迴圈（各 16 次迭代）是目前唯一「外層迴圈完全序列執行、內層已經 partial-unroll factor=4」的熱區，也是最可能貢獻大部分 cycle 數的地方。
- Inner ic 迴圈已經是 `UNROLL factor=TCN_STAGE2_CHANNEL_UNROLL`，本身不需要也不應該疊加 PIPELINE（疊加會複製 PHC v1-v5 時代「PIPELINE 疊 UNROLL 造成資源爆炸」的已知陷阱）。
- 只動這兩處，改動範圍小、風險可控，出問題時容易回滾，也方便做「資源 vs 延遲」的乾淨 A/B 對照。

## ⚠️ 資源風險（PHC 的教訓，這次一定要複製同樣的紀律）

TCN 目前 LUT 已經在 ~32%（DSP 45–51%），跟 PHC v1 一開始 PIPELINE 導致 DSP 從正常值暴衝到 2636% 是同一個機制風險：**任何在有陣列索引的迴圈上加 PIPELINE，都可能被 HLS 自動強制展開週邊迴圈或 partition 陣列，資源可能非線性暴增，不是線性增加**。所以：

1. **先只對 `distilled` 一個 variant 跑**，不要一開始就兩個 model 一起跑。
2. csynth 完成後**先看資源數字再決定要不要跑 cosim**（跟 PHC 06f-06k 的紀律一致）：DSP > 220 或 LUT > 53,200 就不要跑 cosim，直接回報數字並停止，不要嘗試"再調一下看看"。
3. 如果資源爆炸：不要放棄這個方向，改成**降低 `TCN_STAGE2_CHANNEL_UNROLL`（4→2 或 4→1）換取 PIPELINE 空間**，當作第二輪嘗試（見下方「如果爆資源怎麼辦」）。

---

## 修改方式

### 1. 在 `hls_src/tcn_stage2_resource_impl.hpp` 加一個新的開關巨集

在既有的巨集群組（約檔案開頭 line 12–23，`TCN_STAGE2_SHARED_HISTORY_STORAGE` / `TCN_STAGE2_TOP_PIPELINE` / `TCN_STAGE2_LOCAL_TAP_BUFFER` 定義的地方）新增：

```cpp
#ifndef TCN_STAGE2_BLOCK_PIPELINE
#define TCN_STAGE2_BLOCK_PIPELINE 0
#endif
```

### 2. 修改 CONV1 的 oc 迴圈（約 line 134–135）

原本：

```cpp
        for (int oc = 0; oc < TCN_CHANNELS; ++oc) {
#pragma HLS PIPELINE off
            tcn_acc_t acc = (tcn_acc_t)tcn_weights::CONV1_B[BLOCK][oc];
```

改成：

```cpp
        for (int oc = 0; oc < TCN_CHANNELS; ++oc) {
#if TCN_STAGE2_BLOCK_PIPELINE
#pragma HLS PIPELINE II=1
#else
#pragma HLS PIPELINE off
#endif
            tcn_acc_t acc = (tcn_acc_t)tcn_weights::CONV1_B[BLOCK][oc];
```

### 3. 修改 CONV2 的 oc 迴圈（約 line 186–187）

原本：

```cpp
            for (int oc = 0; oc < TCN_CHANNELS; ++oc) {
#pragma HLS PIPELINE off
                tcn_acc_t acc = (tcn_acc_t)tcn_weights::CONV2_B[BLOCK][oc];
```

改成：

```cpp
            for (int oc = 0; oc < TCN_CHANNELS; ++oc) {
#if TCN_STAGE2_BLOCK_PIPELINE
#pragma HLS PIPELINE II=1
#else
#pragma HLS PIPELINE off
#endif
                tcn_acc_t acc = (tcn_acc_t)tcn_weights::CONV2_B[BLOCK][oc];
```

### 4. 新增一個 variant wrapper：`hls_src/tcn_stage2_resource_distilled_shared_u4_blockpipe.cpp`

內容仿照既有的 `tcn_stage2_resource_distilled_shared_u4.cpp`，只多加一行：

```cpp
#define TCN_MODEL_DISTILLED
#define TCN_STAGE2_CHANNEL_UNROLL 4
#define TCN_STAGE2_SHARED_HISTORY_STORAGE 1
#define TCN_STAGE2_BLOCK_PIPELINE 1
#include "tcn_stage2_resource_impl.hpp"
```

### 5. 在 `01_prepare_speed_components.py` 的 `VARIANTS` dict 新增一筆

```python
    "distilled_shared_u4_blockpipe": (
        "distilled",
        "shared_u4_blockpipe",
        "tcn_stage2_resource_distilled_shared_u4_blockpipe.cpp",
    ),
```

---

## 絕對不動的部分

- Inner `ic` 迴圈的 `#pragma HLS PIPELINE off` + `#pragma HLS UNROLL factor=TCN_STAGE2_CHANNEL_UNROLL`（line 137–139、189–191）——不要動，不要疊加 PIPELINE。
- History ring buffer 搬移迴圈（line 159–165、176–181 附近）的 pragma——不要動。
- `#pragma HLS INLINE off`（line 58）——不要移除，這次實驗不測試 inline 的影響。
- `TCN_STAGE2_SHARED_HISTORY_STORAGE`、`TCN_STAGE2_LOCAL_TAP_BUFFER` 這兩個巨集的既有設定值——沿用 shared_u4 的設定（`SHARED_HISTORY_STORAGE=1`），不要一次改兩個變數。
- 權重檔（`tcn_weights_distilled.hpp`）、testbench、測試向量——不動。
- 不要重新訓練、不要碰 NEW5、不要動 TXF 主線任何東西——這是純 HLS 工具實驗。

---

## 執行流程

```bash
cd n2ew_era/07c2b_tcn_streaming_speed/
python 01_prepare_speed_components.py --name distilled_shared_u4_blockpipe --clean
```

（若腳本是用其他方式呼叫 Vitis，沿用腳本既有的 csynth 執行方式；若需要手動下指令，比照既有 variant 用同一套 `hls_config.cfg` 跑：）

```bash
cd C:/tmp/07c2b_tcn_streaming_speed_shortpath/distilled_shared_u4_blockpipe/hls_component_stage2_resource_distilled_shared_u4_blockpipe
vitis-run --mode hls --csim --config hls_config.cfg --work_dir hls_component
vitis-run --mode hls --syn   --config hls_config.cfg --work_dir hls_component
```

**先看 csynth 的資源數字，DSP < 220 且 LUT < 53,200 才繼續跑 cosim：**

```bash
vitis-run --mode hls --cosim --config hls_config.cfg --work_dir hls_component
```

---

## 如果爆資源怎麼辦（第二輪，只在第一輪爆資源時才做）

把 `TCN_STAGE2_CHANNEL_UNROLL` 從 4 降到 2，同時保留 `TCN_STAGE2_BLOCK_PIPELINE=1`，新增 `tcn_stage2_resource_distilled_shared_u2_blockpipe.cpp`：

```cpp
#define TCN_MODEL_DISTILLED
#define TCN_STAGE2_CHANNEL_UNROLL 2
#define TCN_STAGE2_SHARED_HISTORY_STORAGE 1
#define TCN_STAGE2_BLOCK_PIPELINE 1
#include "tcn_stage2_resource_impl.hpp"
```

這是在測「unroll 換 pipeline」這條資源—延遲的新權衡軸，跟 PHC v8 用「關 PIPELINE 換資源」是同一種取捨邏輯的鏡像版本。如果這輪還是爆資源，就先停下回報，不要再往下猜新配置。

---

## 驗收標準

1. `hls.log` 裡**不再出現** `pragma conflict between PIPELINE ... and PIPELINE off` 警告（在 CONV1/CONV2 oc 迴圈上）——這是這次改動真正生效的訊號。
2. csynth top report 的 `interval cycles` 是否**顯著低於 7439**（回報實際數字，不要預設一定會降）。
3. `latency cycles` 是否也有變化（可能降，也可能因為 pipeline fill/drain 而略增，兩者都要報）。
4. 資源：DSP / LUT / FF / BRAM 利用率，跟 07c2d 的基準（49.09% / 32.00% / 14.44% / 23.57%）逐項對比。
5. csim（若資源過關而跑了 cosim）：mismatches / max_abs_diff，跟 07c2d 的容忍度（sign_agreement=1.0）比對是否劣化。

## 回報格式

- 一張表：baseline（shared_u4, PIPELINE off）vs 這次（shared_u4_blockpipe, PIPELINE II=1）的 interval / latency / DSP / LUT / FF / BRAM 逐項對比。
- hls.log 裡有沒有新的 pragma 衝突或 ERROR。
- 如果進了第二輪（unroll降級），一併附上第二輪的同一張對比表。
- 明確結論句：「block-level pipeline 在這塊板子上（有效/無效/需要用unroll換）」，附上支持這句話的具體數字。
