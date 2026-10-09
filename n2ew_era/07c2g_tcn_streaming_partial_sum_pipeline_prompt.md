# Prompt：TCN streaming 第三輪 — 拆 accumulator 依賴鏈（partial sum），解決 12.148ns 時序問題

> 給 coding agent 的自包含任務說明。直接複製整份貼給 agent，不需要補充對話脈絡。
> 工作目錄：`n2ew_era/07c2b_tcn_streaming_speed/`。這是 `AUDIT_tcn_stage2_inner_pipeline.md`（round 2, `innerpipe`）的後續修正，不是回頭嘗試 round 1（block pipeline）。

---

## 上一輪結果（round 2, innerpipe）——不是 pragma conflict，是純時序問題

`distilled_shared_u4_innerpipe`：內層 ic reduction 迴圈套 `PIPELINE II=1` 被 Vitis **接受**（`Target II = 1, Final II = 1, Depth = 4`，沒有任何 pragma conflict warning）。但 csynth 在排程階段失敗，因為單一 ic 迭代內三個 tap（m2/m1/cur）連續三次 `acc += ...` 形成一條依賴鏈：

```
acc_load → add_ln225(3.52ns) → add_ln226(3.52ns) → add_ln227(3.52ns) → acc_write(1.588ns)
總計 12.148ns，超過 100MHz 目標的 10ns 週期（扣 2.7ns clock uncertainty，實際預算 7.3ns）
```

Evidence：`n2ew_era/07c2b_tcn_streaming_speed/AUDIT_tcn_stage2_inner_pipeline.md`（完整逐行時序證據）。

## 這次要做什麼：把 3 級連續 accumulator 依賴砍成 1 級

原本（`tcn_stage2_resource_impl.hpp` 約 line 149–151，CONV1；約 line 201–203，CONV2）：

```cpp
                acc += (tcn_acc_t)tcn_weights::CONV1_W[BLOCK][oc][ic][0] * in_m2;
                acc += (tcn_acc_t)tcn_weights::CONV1_W[BLOCK][oc][ic][1] * in_m1;
                acc += (tcn_acc_t)tcn_weights::CONV1_W[BLOCK][oc][ic][2] * in_cur;
```

三個 add 全部讀寫同一個 `acc`，是連續依賴鏈。改成先算三個 tap 各自獨立的乘積、兩兩合併成一個跟 `acc` 無關的部分和，最後只對 `acc` 做**一次**加法：

```cpp
                const tcn_acc_t tap_m2 = (tcn_acc_t)tcn_weights::CONV1_W[BLOCK][oc][ic][0] * in_m2;
                const tcn_acc_t tap_m1 = (tcn_acc_t)tcn_weights::CONV1_W[BLOCK][oc][ic][1] * in_m1;
                const tcn_acc_t tap_cur = (tcn_acc_t)tcn_weights::CONV1_W[BLOCK][oc][ic][2] * in_cur;
                const tcn_acc_t tap_sum = (tap_m2 + tap_m1) + tap_cur;
                acc += tap_sum;
```

`tap_m2 + tap_m1` 這一步不依賴 `acc`（跟迴圈的 loop-carried accumulator 無關），HLS 排程時可以跟前一輪迭代的 `acc` 寫回並行處理；只有最後 `acc += tap_sum` 這一次加法真正在 loop-carried 的關鍵路徑上——把依賴鏈從 3 級 add 砍到 1 級。

CONV2 那個迴圈（約 line 201–203，變數名是 `c1_m2/c1_m1/c1_cur` 不是 `in_m2/in_m1/in_cur`，其餘結構相同）做一樣的改法：

```cpp
                    const tcn_acc_t tap_c1_m2 = (tcn_acc_t)tcn_weights::CONV2_W[BLOCK][oc][ic][0] * c1_m2;
                    const tcn_acc_t tap_c1_m1 = (tcn_acc_t)tcn_weights::CONV2_W[BLOCK][oc][ic][1] * c1_m1;
                    const tcn_acc_t tap_c1_cur = (tcn_acc_t)tcn_weights::CONV2_W[BLOCK][oc][ic][2] * c1_cur;
                    const tcn_acc_t tap_sum2 = (tap_c1_m2 + tap_c1_m1) + tap_c1_cur;
                    acc += tap_sum2;
```

**這是唯二要改的地方，其他都不動**——沿用 round 2 的 `TCN_STAGE2_INNER_PIPELINE=1` 開關（不用新增巨集），一樣讓內層 ic 迴圈套 `PIPELINE II=1`，外層 oc 迴圈維持 `PIPELINE off` 不變。

### 新增 variant wrapper：`hls_src/tcn_stage2_resource_distilled_shared_u4_innerpipe_partialsum.cpp`

```cpp
#define TCN_MODEL_DISTILLED
#define TCN_STAGE2_CHANNEL_UNROLL 4
#define TCN_STAGE2_SHARED_HISTORY_STORAGE 1
#define TCN_STAGE2_INNER_PIPELINE 1
#include "tcn_stage2_resource_impl.hpp"
```

（跟 round 2 的 wrapper 完全一樣，因為這次改的是 `tcn_stage2_resource_impl.hpp` 本體的加法結構，不是新開一個巨集分支——直接修改 round 2 已經在用 `TCN_STAGE2_INNER_PIPELINE` 保護的那段程式碼即可，不需要再包一層新的 `#if`。）

### `01_prepare_speed_components.py` 新增

```python
    "distilled_shared_u4_innerpipe_partialsum": (
        "distilled",
        "shared_u4_innerpipe_partialsum",
        "tcn_stage2_resource_distilled_shared_u4_innerpipe_partialsum.cpp",
    ),
```

---

## 絕對不動的部分

- 外層 oc 迴圈——維持 `PIPELINE off`，跟 round 2 一樣不碰。
- History ring buffer、`INLINE off`、`SHARED_HISTORY_STORAGE`——不動。
- 不要引入新的巨集開關，直接改 `TCN_STAGE2_INNER_PIPELINE=1` 分支底下的加法結構本身。
- 不要重新訓練、不要碰 NEW5。

## 執行流程

```bash
cd n2ew_era/07c2b_tcn_streaming_speed/
python 01_prepare_speed_components.py --variant distilled_shared_u4_innerpipe_partialsum --clean
```

跟前兩輪一樣：先 csim，再用 `vitis-run.bat --mode hls --tcl --input_file run_csynth_only.tcl`（round 2 用的就是這個路徑，`--syn` 旗標在這台機器上不支援）跑 csynth。**先看 hls.log 的時序警告有沒有消失，csynth report 有沒有真的產生，再決定要不要跑 cosim。**

## 驗收標準

1. `hls.log` 裡 `Cannot meet target clock period` 這類警告是否消失，或者估計時脈週期是否降到 10ns（含 2.7ns uncertainty 後的 7.3ns 預算）以內——這是最直接的判斷依據，比看有沒有產生 report 更精確（report 有可能因為其他原因也產生不出來）。
2. 若時序過關：`csynth_report_count` 是否 > 0，latency/interval cycles 是多少（跟 round 2 失敗前、以及 07C2D baseline 的 7438/7439 對比）。
3. 資源：DSP/LUT/FF/BRAM，因為多了兩次額外的加法（tap_m2+tap_m1 這一步），LUT/FF 可能小幅增加，屬預期內，不是異常。
4. csim（若資源與時序都過關才跑 cosim）：sign agreement 是否維持 1.0，數值結果應該跟 round 2 的 csim 完全一致（重排運算順序不應該改變浮點/定點計算結果的功能正確性，只是硬體排程不同）。

## 回報格式

- 時序警告是否消失，若還有，附上新的關鍵路徑逐段延遲（格式比照 `AUDIT_tcn_stage2_inner_pipeline.md` 的 critical path 表格）。
- 若過關：interval/latency/資源對比表，比照前兩輪 audit 的表格格式。
- 明確結論句：「partial-sum 重構（有/沒有）解決 12.148ns 時序問題」，如果解決了，順便估算一下如果把這個結構套用到全部 5 個 block 的 CONV1+CONV2（目前這個實驗只測了函式模板的一個 instantiation，但 `stage2_resource_stream_block_step` 是被 5 個 BLOCK 各自 instantiate 一次，理論上資源會乘 5 倍，需要確認 DSP 總量還在 220 預算內）。
