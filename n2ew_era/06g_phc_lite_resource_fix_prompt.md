# Prompt：修正 PHC-lite manual HLS 的資源爆炸問題（活化函數重寫）

> 給 coding agent 的自包含任務說明。直接複製整份貼給 agent 即可，不需要額外補充對話脈絡。

---

##背景與問題（已驗證，非推測）

`n2ew_era/06f_phc_lite_manual_hls/` 的 PHC-lite manual HLS pipeline 已經跑完整個
csim → csynth → cosim 流程，數值/功能驗證**通過**：

- C-Sim: `total=25600 mismatches=0, max_abs_diff=0.00647461, mean_abs_diff=0.00025022`
- Cosim (Verilog/xsim): `*** C/RTL co-simulation finished: PASS ***`，latency=638 cycles, II=1

但讀取 `hlsComponent/hlsComponent/hls/syn/report/phc_step_csynth.rpt` 後發現
**C-Synthesis 階段的資源用量遠超出 target FPGA (xc7z020clg400-1 / PYNQ-Z2) 的實體容量**：

| 資源 | PHC-lite 用量 | xc7z020 可用量 | 利用率 |
|---|---|---|---|
| LUT | 1,049,148 | 53,200 | 1972%（超出 ~20倍） |
| FF | 1,213,268 | 106,400 | 1140%（超出 ~11倍） |
| DSP | 5,800 | 220 | 2636%（超出 ~26倍） |
| BRAM_18K | 80 | 280 | 28%（正常） |

**這代表目前這版 RTL 物理上不可能放進這顆晶片**，連嘗試 Place & Route 都沒有意義。
必須先把這個問題修好，重新合成確認資源用量回到合理範圍（目標：跟同專題的
`06a_hls_queue_rule_baseline` 一樣是個位數百分比，例如 LUT 0.7% / FF 0.1% / DSP 0%），
之後才可以考慮跑 Implementation (P&R)。

## 根本原因（已追到原始碼，精確到行號）

問題出在 `n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.cpp`：

1. **檔案開頭（約 line 8-14）** 為了讓程式碼能在沒有 `<hls_math.h>` 的環境下編譯，
   定義了一個 fallback `namespace hls`，把 `exp`/`sqrt`/`tanh` 都實作成
   **`double` → `double`** 的函式（直接呼叫 `std::exp/sqrt/tanh`）：
   ```cpp
   namespace hls {
   inline double exp(double x) { return std::exp(x); }
   inline double sqrt(double x) { return std::sqrt(x); }
   inline double tanh(double x) { return std::tanh(x); }
   }
   ```

2. **三個呼叫點都把 `ap_fixed` 定點數先轉型成 `double`再呼叫**（line ~25, ~30, ~174）：
   ```cpp
   // sigmoid_scalar (約 line 25)
   return (phc_acc_t)1.0 / ((phc_acc_t)1.0 + (phc_acc_t)hls::exp((double)(-x)));

   // tanh_scalar (約 line 30)
   return (phc_acc_t)hls::tanh((double)x);

   // LayerNorm 的 inv_std (約 line 174)
   phc_acc_t inv_std = (phc_acc_t)1.0 / (phc_acc_t)hls::sqrt((double)(var + (phc_var_t)LN_EPS));
   ```

3. **Vitis HLS 因此合成出完整 IEEE-754 雙精度浮點運算硬體核心**
   （`generic_tanh_double_s` 等），csynth 報告顯示**每一個實例就要吃掉
   ~11,602 LUT + ~20,171 FF**。

4. **疊加因素**：`sigmoid_16`/`tanh_16`/LayerNorm 內的迴圈都套了
   `#pragma HLS UNROLL` + `#pragma HLS ARRAY_PARTITION variable=... complete`
   （見 `phc_step_hls.cpp` line 35-38, 45-48, 75/78, 97/100, 117/120, 139/142,
   162, 169, 177 等多處），把 16 個 hidden units 的迴圈完全展開、不共用運算單元。
   結果是 csynth 報告顯示 **16 個獨立的 `generic_tanh_double_s` 實例同時存在**
   （`grp_generic_tanh_double_s_fu_2576` ~ `_fu_2741`），把本來就過大的成本再乘以 16。

兩個因素疊加 → LUT/FF/DSP 暴增到晶片容量的 11~26 倍。

## 你需要做的修正（依序，逐步驗證）

### 步驟 1：移除 `(double)` 轉型，改用定點數原生的數學函式
- 修改 `phc_step_hls.cpp` 開頭的 fallback `namespace hls` 區塊，**不要**再提供
  `double → double` 版本的 `exp/sqrt/tanh`。
- 確認編譯環境是否真的有 Vitis HLS 提供的 `<hls_math.h>`（它對 `ap_fixed<W,I>`
  類型有原生的 overload，運算全程在定點數網域內完成，不會生成雙精度浮點核心）。
  如果有，直接 `#include <hls_math.h>` 並讓 `hls::exp/hls::tanh/hls::sqrt`
  直接作用在 `phc_acc_t`/`phc_var_t`（定點數）上，**不要做 `(double)` 轉型**：
  ```cpp
  // 修正後（範例，實際型別請以 phc_step_hls.hpp 中 phc_acc_t/phc_var_t 定義為準）
  return (phc_acc_t)1.0 / ((phc_acc_t)1.0 + hls::exp(-x));   // x, 回傳值皆為 phc_acc_t
  return hls::tanh(x);                                        // 不轉 double
  phc_acc_t inv_std = (phc_acc_t)1.0 / hls::sqrt(var + (phc_var_t)LN_EPS);
  ```
- 如果合成環境真的沒有 `<hls_math.h>` 對 `ap_fixed` 的 overload（少見，但要實際測試
  確認，不要用猜的），改用 **piecewise-linear / 查表近似**（這是 hls4ml 等成熟流程
  對 sigmoid/tanh 的標準作法），自行用 `ap_fixed` 實作分段線性近似函式取代
  `hls::exp/tanh/sqrt`，徹底避免任何浮點運算硬體被合成出來。

### 步驟 2：降低/重新評估展開倍率
- 即使換成定點數版本的 `tanh`/`sigmoid`/`sqrt`，16 份完全展開的運算單元仍然偏貴。
  檢視 `sigmoid_16`、`tanh_16`、LayerNorm 內的 `#pragma HLS UNROLL` 與
  `#pragma HLS ARRAY_PARTITION variable=... complete`（見上方根因第 4 點列出的
  行號），評估是否需要改成部分展開（`#pragma HLS UNROLL factor=N`，N < 16）或
  時間共享單一運算單元，在資源與 latency/II 之間取得合理平衡。
  **注意**：頂層 `phc_step` 函式本身有 `#pragma HLS PIPELINE II=1`（line 190），
  改動內部展開策略時要確保 II=1 仍然可以達成（或至少明確記錄新的 II 數字），
  不要在沒有說明的情況下讓 pipeline 退化。

### 步驟 3：重新跑 csim → csynth → cosim，驗證兩件事都成立
1. **功能/數值正確性沒有退化**：C-Sim 與 Cosim 仍須 PASS，且
   `max_abs_diff`/`mean_abs_diff` 數字應與修正前相近或更好（修正前基準：
   `max_abs_diff=0.00647461, mean_abs_diff=0.00025022`，這是 Q24.12 精度方案
   已驗證可接受的誤差量級，不應該因為換了活化函數實作而顯著惡化）。
2. **資源用量回到可放入晶片的範圍**：目標是讓 `phc_step_csynth.rpt` 的
   Utilization Estimates 表格中 LUT/FF/DSP 利用率降到個位數百分比量級
   （對照組 `06a_hls_queue_rule_baseline` 是 LUT 0.7% / FF 0.1% / DSP 0%；
   PHC-lite 因為運算量遠大於 queue_rule，數字一定會比它高，但至少要遠低於
   100%，理想上落在 5~30% 之間，這樣才談得上放上 PYNQ-Z2）。

跑完後，把新的 csynth Utilization Estimates 表格（LUT/FF/DSP/BRAM 的用量、
可用量、百分比）、csim/cosim 的 PASS 狀態與誤差數字，整理成一段簡短的
前後對照（修正前 vs 修正後），方便我檢視。

### 步驟 4：修正過時且錯誤的報告文件
`n2ew_era/06f_phc_lite_manual_hls/PHC_HLS_SYNTHESIS_REPORT.md` 目前內容寫
「C simulation: not_run, C synthesis: not_run, C/RTL co-simulation: not_run」，
但實際上這些模擬都已經跑完且有結果（log 時間戳可證明），這份報告**內容與
事實矛盾、不可信**。請在完成上述修正並重新合成後，**重寫**這份報告，
使其內容與 `hls_run_csim.log` / `hls_run_cosim.log` / `phc_step_csynth.rpt`
的實際內容一致：明確記錄
- C-Sim/Cosim 的 PASS 狀態與 `max_abs_diff`/`mean_abs_diff`（不要只寫
  "0 mismatches, PASS"，因為目前 testbench 用的判定容忍度
  `PHC_ABS_TOL = 1.0` 偏寬鬆，單獨引用 PASS 標籤說服力不足，必須附上
  實測的誤差數字）
- csynth 的 latency/II/時脈估計與資源利用率表格（修正前後都列出，並標明
  哪一份是最終採用的版本）
- 是否已嘗試/計劃 Implementation (P&R)，以及理由

## 驗收標準（done 的定義）
1. `phc_step_csynth.rpt` 顯示 LUT/FF/DSP 利用率遠低於 100%（具體數字回報給我）。
2. C-Sim 與 Cosim 仍然 PASS，且誤差數字與修正前相近或更佳。
3. `PHC_HLS_SYNTHESIS_REPORT.md` 內容與實際 log/report 一致，無矛盾敘述。
4. 提供修正前後的數字對照（至少包含 LUT/FF/DSP 利用率、max_abs_diff/mean_abs_diff、
   latency/II），方便我直接拿來檢視與寫入構想書。

## 紅線提醒（請務必遵守）
- **不要**為了讓利用率數字變好看而調高 `PHC_ABS_TOL` 來掩蓋誤差變大的事實——
  testbench 裡已經有註解明確寫著「Do not silently raise this tolerance to
  hide failures」。
- **不要**只回報「PASS」就結案，務必附上 `max_abs_diff`/`mean_abs_diff` 等
  實測數字，這是審稿時一定會被追問的細節。
- 若 `<hls_math.h>` 對 `ap_fixed` 確實沒有所需的 overload，**如實說明**並改走
  piecewise-linear 近似這條路，不要硬塞一個會繼續產生浮點硬體的變通寫法。
