# Prompt：TCN streaming 第二輪 — 改成直接 pipeline 內層 ic 迴圈，不再碰外層

> 給 coding agent 的自包含任務說明。直接複製整份貼給 agent，不需要補充對話脈絡。
> 工作目錄：`n2ew_era/07c2b_tcn_streaming_speed/`。這是 07C2E prompt 第一輪（`distilled_shared_u4_blockpipe`，外層 oc 迴圈套 PIPELINE）撞到新的 pragma conflict 之後的修正版。

---

## 上一輪發生了什麼（人工回報，尚未附完整 log）

`distilled_shared_u4_blockpipe`（外層 oc 迴圈從 `PIPELINE off` 改成 `PIPELINE II=1`，內層 ic 迴圈維持原本的 `PIPELINE off` + `UNROLL factor=4` 不動）**又撞到 pragma conflict，不是資源爆炸**。降 unroll factor（4→2）沒有意義，因為衝突是語法層級的（pragma 互相矛盾），不是資源層級的（用量超標）——這點判斷是對的，第二輪 `u2_blockpipe` 不需要跑。

**如果手上還留著這次的 hls.log，請先把完整的 conflict warning 貼出來確認** ── 但不要因為沒有 log 就卡住，下面的修正方向不依賴知道 log 的精確措辭，直接針對結構性原因下手。

## 推測的結構性原因

CONV1／CONV2 的迴圈長這樣（不是「完美巢狀」）：

```cpp
for (int oc = 0; oc < TCN_CHANNELS; ++oc) {
#pragma HLS PIPELINE II=1              // 外層新加的
    tcn_acc_t acc = ...B[oc];          // ← 迴圈外、迴圈內都有的額外敘述
    for (int ic = 0; ic < TCN_CHANNELS; ++ic) {
#pragma HLS PIPELINE off               // 內層原本就有的
#pragma HLS UNROLL factor=...
        acc += ...;
    }
    out[oc] = acc;                     // ← 額外敘述
}
```

Vitis 的「外層 PIPELINE 自動攤平巢狀迴圈」機制，只在**完美巢狀**（外層迴圈的 body 只有內層迴圈，沒有其他敘述）才會自動生效。這裡外層迴圈在內層迴圈前後都有額外的初始化／寫回敘述，屬於**不完美巢狀**，所以 Vitis 沒辦法直接攤平，只好把外層當成一個 pipeline 排程去包住內層——這時候內層自己的 `PIPELINE off` 就會跟外層的 pipeline 排程矛盾，衝突發生的位置從「函式層級」下移到「迴圈層級」，但本質是同一種病。

## 這次改用更保守的做法：只 pipeline 內層 ic 迴圈，外層完全不動

不要再嘗試讓外層 oc 迴圈 pipeline。外層維持 `#pragma HLS PIPELINE off`（跟 07C2D 封版狀態完全一樣，不做任何改動）。**只針對內層 ic 迴圈**（一個單純、沒有巢狀子迴圈的 reduction 迴圈）直接套用 `PIPELINE II=1`，取代原本的 `PIPELINE off` + `UNROLL factor=...`。這個迴圈本身不再有巢狀結構，不存在「內層迴圈自己又有子迴圈」的衝突可能性，風險比第一輪低很多，也更可能真的量到 pipeline 的效果。

代價：這樣做預期只會讓「單一輸出通道 oc 的 16 次乘加」變快，16 個 oc 之間仍然是序列執行（因為外層沒動）——所以就算成功，latency 的改善幅度會比原本設想的「整個函式都 pipeline」小很多。**這是刻意的取捨：先確認"這個 codebase 裡 pipeline 到底能不能生效"這個更基本的問題，再考慮要不要進一步處理外層的不完美巢狀問題**（那需要把迴圈重構成三段式：先填 acc 陣列初始值、再跑一個純二維迴圈做 reduction、最後寫回 out——這是更大的改動，先不要在這輪做）。

---

## 修改方式

### 1. 新增巢巢集開關（`hls_src/tcn_stage2_resource_impl.hpp`，跟既有巨集放一起）

```cpp
#ifndef TCN_STAGE2_INNER_PIPELINE
#define TCN_STAGE2_INNER_PIPELINE 0
#endif
```

### 2. CONV1 的 ic 迴圈（約 line 137–139）

原本：

```cpp
                for (int ic = 0; ic < TCN_CHANNELS; ++ic) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=TCN_STAGE2_CHANNEL_UNROLL
```

改成：

```cpp
                for (int ic = 0; ic < TCN_CHANNELS; ++ic) {
#if TCN_STAGE2_INNER_PIPELINE
#pragma HLS PIPELINE II=1
#else
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=TCN_STAGE2_CHANNEL_UNROLL
#endif
```

**注意：`TCN_STAGE2_INNER_PIPELINE=1` 時完全不套 UNROLL，讓 PIPELINE 自己決定怎麼排程這個 reduction，不要兩個一起套。**

### 3. CONV2 的 ic 迴圈（約 line 189–191），同樣改法

原本：

```cpp
                for (int ic = 0; ic < TCN_CHANNELS; ++ic) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=TCN_STAGE2_CHANNEL_UNROLL
```

改成：

```cpp
                for (int ic = 0; ic < TCN_CHANNELS; ++ic) {
#if TCN_STAGE2_INNER_PIPELINE
#pragma HLS PIPELINE II=1
#else
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=TCN_STAGE2_CHANNEL_UNROLL
#endif
```

### 4. 外層 oc 迴圈（CONV1、CONV2 各一處）：**改回原本的 `PIPELINE off`，不要沿用第一輪的 `TCN_STAGE2_BLOCK_PIPELINE`**

這次不用 07C2E 那個開關了，外層兩處都固定寫死：

```cpp
        for (int oc = 0; oc < TCN_CHANNELS; ++oc) {
#pragma HLS PIPELINE off
```

（如果 `TCN_STAGE2_BLOCK_PIPELINE` 巨集還留在檔案裡沒關係，這次的新 variant 不要定義它、也不要設成 1。）

### 5. 新增 variant wrapper：`hls_src/tcn_stage2_resource_distilled_shared_u4_innerpipe.cpp`

```cpp
#define TCN_MODEL_DISTILLED
#define TCN_STAGE2_CHANNEL_UNROLL 4
#define TCN_STAGE2_SHARED_HISTORY_STORAGE 1
#define TCN_STAGE2_INNER_PIPELINE 1
#include "tcn_stage2_resource_impl.hpp"
```

（`TCN_STAGE2_CHANNEL_UNROLL 4` 這行在 `TCN_STAGE2_INNER_PIPELINE=1` 時實際上不會被用到，留著只是為了不動既有 macro 判斷邏輯，不影響行為。）

### 6. `01_prepare_speed_components.py` 的 `VARIANTS` dict 新增

```python
    "distilled_shared_u4_innerpipe": (
        "distilled",
        "shared_u4_innerpipe",
        "tcn_stage2_resource_distilled_shared_u4_innerpipe.cpp",
    ),
```

---

## 絕對不動的部分

- 外層 oc 迴圈——這輪固定是 `PIPELINE off`，不要嘗試任何 pipeline。
- History ring buffer 搬移迴圈、`#pragma HLS INLINE off`、`TCN_STAGE2_SHARED_HISTORY_STORAGE`——不動。
- 07C2E 那個 `TCN_STAGE2_BLOCK_PIPELINE` 巨集本身不用刪，但這個新 variant 不要開它。
- 不要重新訓練、不要碰 NEW5。

## 執行流程

跟 07C2E 相同模式：

```bash
cd n2ew_era/07c2b_tcn_streaming_speed/
python 01_prepare_speed_components.py --name distilled_shared_u4_innerpipe --clean
```

先跑 csim + csynth，**看 hls.log 有沒有 pragma conflict warning、資源有沒有超標，兩者都過才跑 cosim**。

## 驗收標準

1. hls.log **不再出現任何 "pragma conflict" 或 "Ignoring PIPELINE" 字樣**——這是判斷這輪是否真的繞開問題的第一道關卡，比延遲數字更優先確認。
2. 如果還是有 pragma conflict：把完整 warning 貼出來，這代表連單一無巢狀迴圈都不給 pipeline，問題可能出在 reduction 迴圈裡 `acc +=` 這個累加造成的迴圈帶依賴（loop-carried dependency），需要另外處理（例如拆成部分和樹狀合併），先回報不要自己延伸嘗試。
3. 若過關：報 interval / latency cycles 相較 baseline（7439/7438）的變化——預期只會小幅改善（因為外層仍序列），不要預期跟第一輪設想的大幅改善。
4. 資源：DSP/LUT/FF/BRAM，跟 07c2d 基準（49.09%/32.00%/14.44%/23.57%）比對。
5. csim（過關才跑 cosim）：sign agreement 是否維持 1.0。

## 回報格式

- 有沒有出現 pragma conflict，完整 warning 文字。
- 若無衝突：interval/latency/資源對比表。
- 明確結論句：「單一無巢狀 reduction 迴圈可以/不可以被 pipeline」，這個結論本身比數字更重要——它會決定這條路要不要繼續往外層迴圈重構（三段式拆迴圈）投入時間，還是這個 codebase 在目前的迴圈寫法下就是無法 pipeline，該把力氣轉去別的地方（例如 06e 的 hls4ml 版本升級路線）。
