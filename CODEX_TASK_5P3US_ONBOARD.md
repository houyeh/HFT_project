# Codex 任務包：把 5.300 µs 設計點（06U V3 diagonal）燒上 PYNQ-Z2

**開立日期**：2026-10-04
**目標**：讓本專案最快的設計點從 `routed` 驗證等級升級為 `on-board`，消除成果報告 §6.3 的第一條限制。
**預估**：4 個階段，每階段都有停止點。不要一次跑到底。

---

## 0. 驗收標準（唯一的成功定義）

在 PYNQ-Z2 實體板上，以 06U V3 diagonal SSM 核心重播 **4,096 筆** NEW4 事件，逐筆比對板上讀出的 bid/ask 分數與軟體參考模型輸出，並產出：

| 必須回報的數字 | 比較基準（現有 25.733 µs 版本的對應值） |
|---|---|
| 總比對筆數 | 4,096 |
| 出現差異的筆數 | 27 |
| 最大差異（LSB） | 1 |
| **最終決策翻轉筆數** | **0** |
| 實際收斂頻率與推導延遲 | 90 MHz → 25.733 µs |

**決策翻轉筆數必須為 0。** 若非 0，立即停止並回報，不要嘗試修正——那是一個需要討論的結果，不是一個 bug。

---

## 1. 已查證的背景事實（不要重新調查，直接用）

### 1.1 V3 的原始碼完整存在，但 RTL 不存在

```
n2ew_era/06u_phc_latency_breakthrough/variants/v3_diagonal16_unroll8_localpart_100mhz/
├── hls_src/
│   ├── phc_diag_step_hls.cpp        7,333 B
│   ├── phc_diag_step_hls.hpp          880 B
│   ├── phc_diag_step_testbench.cpp  5,555 B
│   └── phc_diag_weights.hpp        14,351 B   <- 05F diagonal 權重，已匯出，不要重新產生
└── hls_component/                   <- 空目錄，HLS 專案已被清除
```

`artifacts/v3_diagonal16_unroll8_localpart_100mhz_shortpath/` 底下**只有 .rpt 與 .xml 報告，沒有任何 .v / .vhd / IP zip**（已確認：`find` 的 `*.v` 結果為 0 個檔案）。

> **所以這不是「接線」任務。第一步是重跑 Vitis HLS 產生 RTL。**

### 1.2 必須重現的既有數字（這是階段 A 的驗收門）

來源：`artifacts/v3_diagonal16_unroll8_localpart_100mhz_shortpath/impl/phc_diag_step_export.rpt`

| 項目 | 值 |
|---|---|
| 工具版本 | Xilinx Vivado v.2025.2 |
| 目標元件 | `xc7z020-clg400-1` |
| RTL 週期數 | **530** |
| SLICE / LUT / FF | 4,463 / 11,348 / 12,760 |
| **DSP** | **196** |
| BRAM / URAM / SRL | 0 / 0 / 26 |
| CP required | 10.000 ns |
| CP achieved post-synthesis | 9.259 ns |
| CP achieved post-implementation | 9.870 ns（即 WNS +0.130 ns） |
| 協同模擬 | 1,000 向量通過；另有 25,600 向量 CSim 零不符 |

### 1.3 V3 的 C 介面與現有 `phc_step` 完全相同

```cpp
typedef ap_fixed<24, 12> phc_fixed_t;
typedef ap_fixed<40, 16> phc_acc_t;
typedef ap_fixed<48, 24> phc_var_t;
static const int PHC_FEATURE_DIM = 16;
static const int PHC_HIDDEN_DIM  = 16;
static const int PHC_READOUT_DIM = 2;

void phc_diag_step(
    const phc_fixed_t x_t[PHC_FEATURE_DIM],
    const phc_fixed_t state_prev[PHC_HIDDEN_DIM],
    phc_fixed_t readout[PHC_READOUT_DIM],
    phc_fixed_t state_next[PHC_HIDDEN_DIM]
);
```

維度、型別、引數順序與 `phc_step` 一致。**但模組名稱不同（`phc_diag_step`），且生成的 RTL 埠數可能不同**——見 §9 陷阱 1。

### 1.4 現有板級 wrapper 的位置與介面

```
n2ew_era/06s2_pynq_z2_wrapper_bringup/
├── rtl/
│   ├── pynq_z2_phc_demo_wrapper.sv      <- 實例化 HLS 核心（L194 `phc_step u_phc_step`）
│   ├── pynq_z2_phc_demo_board_top.sv    <- 板級頂層
│   └── pynq_z2_clock_90mhz.sv           <- 目前是 90 MHz
├── bram/
│   ├── BRAM_VECTOR_PACKING.md           <- 先讀這份
│   ├── demo_replay_x_readout_q24_12.{coe,mem}
│   └── demo_replay_expected_state_q24_12.{coe,mem}
├── xdc/pynq_z2_demo_wrapper_board.xdc
├── tcl/{program_pynq_z2_demo.tcl, run_pnr_after_xsim_template.tcl, run_xsim_demo_wrapper.tcl}
├── scripts/{program_pynq_z2_demo.cmd, run_xsim_demo_wrapper_direct.ps1}
├── 01_generate_replay_reference_vectors.py
├── 02_update_bram_expected_from_hls_csim.py
├── 03_analyze_wrapper_readout_drift.py
├── 1LSB_DRIFT_ROOT_CAUSE_AND_IMPACT.md  <- 先讀這份
└── WRAPPER_BRINGUP_STATUS.md            <- 先讀這份
```

wrapper 已有 `PHC_HAS_STATE_PREV_PORT1` 這個 `ifdef`，顯示過去已處理過埠數變動。沿用這個模式。

### 1.5 兩個模型來源不同，期望向量必須重算

| 設計點 | 訊號來源模型 | NEW3 markout 動作損益 |
|---|---|---|
| 25.733 µs（目前已上板） | **05C4 dense** PHC/STP seed2 | 5,237,410 |
| 5.300 µs（本任務） | **05F diagonal SSM** seed2, best_epoch 157 | 4,727,425 |

> `bram/` 目前的 `.coe` / `.mem` 是 **05C4** 的期望值。**必須以 V3 的 C 模擬重新產生**，否則比對會全數不符。

---

## 2. 絕對不可以做的事

1. **不要覆寫任何既有 artifact。** 本專案近期才因為「腳本在 import 時計算共用輸出路徑」而污染過共用 CSV。所有新產物寫到**新目錄**：
   - HLS 專案 → `variants/v3_diagonal16_unroll8_localpart_100mhz/hls_component/`（目前為空，可用）
   - wrapper P&R → `06s2_pynq_z2_wrapper_bringup/vivado_pnr_work_v3_diag_100mhz/`（新建）
   - BRAM 向量 → `06s2_pynq_z2_wrapper_bringup/bram_v3_diag/`（新建，**不要寫進 `bram/`**）
   - 新 RTL → `06s2_pynq_z2_wrapper_bringup/rtl/pynq_z2_phc_demo_wrapper_v3diag.sv`（新檔，**不要改原檔**）
2. **不要修改** `hls_src/` 底下任何檔案，包括 `phc_diag_weights.hpp`。權重已凍結。
3. **不要修改** 05F 的 checkpoint 或任何 `models/` 內容。
4. **不要碰** `n2ew_era/08a_fi2010_public_benchmark/`（另有實驗在跑）。
5. **不要用 `--skip-preflight` 之類的旗標繞過檢查**。若有檢查阻擋，回報阻擋原因。
6. **不要在任何 .cmd / .bat 檔裡寫中文**。本機 console 是 cp950，UTF-8 中文會變亂碼並破壞控制流（本專案已踩過）。`.cmd` 一律純 ASCII + CRLF。
7. **Python print 不要用 emoji 或 `≈`**，同樣會在 cp950 觸發 `UnicodeEncodeError`。用 `[OK]` / `[WARN]` / `[FAIL]`，並在腳本開頭加 `sys.stdout.reconfigure(errors="replace")`。
8. **不要宣稱沒有量到的東西。** 延遲是「週期數 ÷ 實際收斂頻率」推導值，不是晶片上量測值——報告措辞必須維持這個區分。

---

## 3. 階段 A：重建 V3 的 RTL 並重現既有數字

**工作目錄**：`n2ew_era/06u_phc_latency_breakthrough/variants/v3_diagonal16_unroll8_localpart_100mhz/`

1. 讀 `hls_src/phc_diag_step_hls.cpp`，找出其中的 pragma 設定（應包含 unroll factor 8 與 local `ARRAY_PARTITION`）。**原封不動使用**。
2. 建立 Vitis HLS 專案：part `xc7z020-clg400-1`，target clock period **10.000 ns**，top function `phc_diag_step`，testbench `phc_diag_step_testbench.cpp`。
3. 依序執行：`csim` → `csynth` → `cosim` → `export RTL`（含 Vivado 實作以取得 routed 數字）。
4. **把產生的 RTL 複製到一個明確的位置**並列出所有 `.v` 檔，例如
   `variants/v3_diagonal16_unroll8_localpart_100mhz/rtl_export/`。

### 階段 A 的驗收門

> 🔧 **2026-10-04 修正**：本表原先把三份不同報告的數字混在一起（532/198 來自 **csynth**、530 來自 **RTL cosim**、196/11,348/12,760/9.870 ns 來自 **post-implementation export**）。Codex 於階段 A 正確指出此規格錯誤。以下為修正後的正確驗收門，已對照存檔報告逐項驗證。

**csynth 層級**（對照 `artifacts/.../syn/phc_diag_step_csynth.rpt`）：

| 檢查 | 期望 | 容許差異 |
|---|---|---|
| Latency / Interval（cycles） | **532 / 533** | 必須完全相同 |
| DSP | **198** | 必須完全相同 |
| BRAM_18K | 0 | 必須完全相同 |
| LUT / FF | **35,617 / 28,485** | 必須完全相同 |

**cosim 層級**（對照 `artifacts/.../cosim/phc_diag_step_cosim.rpt`）：

| 檢查 | 期望 |
|---|---|
| Latency（cycles） | **530** |
| Interval（cycles） | **531** |
| Total Execution Time | **530999** |
| 結果 | Pass，0 不符 |

**csim 層級**：25,600 向量，0 不符。

**post-implementation 層級**（DSP 196 / LUT 11,348 / FF 12,760 / CP 9.870 ns）：
⚠️ **此層級不是本任務的必要條件**，見 §6 的說明——板級 P&R（階段 D）會產生我們真正需要的 routed 數字。HLS 核心單獨的 implementation 僅用於重現存檔數字，**不應阻擋階段 B**。

> **若週期數不是 530 或 DSP 不是 196，停止。** 這表示環境或 pragma 與原始執行不同，必須先查清楚，而不是繼續往下做。工具版本差異是最可能原因——原始執行為 Vivado **v.2025.2**，請回報你實際使用的版本。

### 階段 A 回報格式

```
[A] Vivado 版本: <...>
[A] csynth: cycles=<...> DSP=<...> BRAM=<...> LUT=<...> FF=<...>
[A] cosim: <向量數> vectors, <不符筆數> mismatches
[A] export: CP required=<...> post-synth=<...> post-impl=<...>
[A] 生成的 RTL .v 檔清單:
<逐行列出>
[A] phc_diag_step 的完整 port 列表（從生成的 .v 的 module 宣告逐字複製）:
<逐行列出>
```

**最後那個 port 列表是階段 B 的前提，不可省略。**

---

## 4. 階段 B：wrapper 整合

1. 先讀 `WRAPPER_BRINGUP_STATUS.md`、`bram/BRAM_VECTOR_PACKING.md`、`1LSB_DRIFT_ROOT_CAUSE_AND_IMPACT.md` 三份。
2. 複製 `rtl/pynq_z2_phc_demo_wrapper.sv` → `rtl/pynq_z2_phc_demo_wrapper_v3diag.sv`，修改：
   - 模組名改為 `pynq_z2_phc_demo_wrapper_v3diag`
   - `phc_step u_phc_step` → `phc_diag_step u_phc_diag_step`
   - **依階段 A 回報的實際 port 列表調整連接**。若 `x_t` / `state_prev` 因 unroll8 被展開為多埠或純量埠，沿用既有的 `ifdef` 模式新增對應條件。
3. 同樣方式複製板級頂層為 `pynq_z2_phc_demo_board_top_v3diag.sv`。
4. **時脈**：目標 100 MHz。檢視 `pynq_z2_clock_90mhz.sv` 的 MMCM/PLL 參數後，新建 `pynq_z2_clock_100mhz.sv`。確認 PYNQ-Z2 的輸入時脈來源與既有檔案一致。
5. 以 `tcl/run_xsim_demo_wrapper.tcl` 為範本跑 XSim 行為模擬，**先在模擬中確認 FSM 交握正確**（`ap_start` / `ap_done` / `ap_idle` / `ap_ready`）再進 P&R。

### 階段 B 回報格式

```
[B] port 連接差異（相對原 wrapper）:
<逐項說明哪個 port 新增/移除/改名>
[B] 時脈: 輸入 <...> MHz -> MMCM 參數 <...> -> 輸出 100 MHz
[B] XSim: <通過/失敗>，若失敗附最後 30 行 log
```

---

## 5. 階段 C：重算 BRAM 期望向量

`bram/` 內現有的期望值屬於 **05C4 dense**，對 V3 無效。

1. 讀 `01_generate_replay_reference_vectors.py` 與 `02_update_bram_expected_from_hls_csim.py`，確認它們如何取得參考值。
2. 以**同一批 4,096 筆 NEW4 事件**（場次 `NEW4_20260112_TXFA6`，選取規則為「val/NEW3/NEW4 全部場次中 1 秒滾動事件數最高的窗口」，峰值 1,469 事件／秒）作為輸入，**不要重新挑選窗口**。
3. 以 V3 的 C 模擬（階段 A 的 csim）產生期望 readout 與 expected state。
4. 輸出到**新目錄** `bram_v3_diag/`，檔名沿用原命名慣例。
5. **自我檢查**：新舊 `.mem` 必須不同（因為模型不同）。若相同，表示你誤用了舊的參考值——停止並回報。

### 階段 C 回報格式

```
[C] 輸入事件數: <...>（必須是 4096）
[C] 來源場次: <...>（必須是 NEW4_20260112_TXFA6）
[C] 新舊 readout .mem 相異行數: <...> / 4096   <- 必須 > 0
[C] 輸出檔案: <逐行列出路徑與大小>
```

---

## 6. 階段 D：P&R 與時序決策點

**這一階段有一個預期中的風險，請先讀完再開始。**

5.300 µs 這個數字來自 **HLS 核心單獨**在 100 MHz 下的實作，WNS 僅 **+0.130 ns**。板級 wrapper 會額外加入重播 FSM、BRAM、比對邏輯、LED 與時脈產生，這些都會吃掉餘裕。

**既有前例**：06L unroll16 在 100 MHz 下 baseline / aggressive / fanout 三種布局繞線策略**全數失敗**（根因為 FSM → DSP48E1 `PCIN` 路徑高扇出繞線），最終退頻至 90 MHz 才收斂。

### 決策規則（照這個走，不要自行判斷）

1. 先在 **100 MHz** 嘗試。若 WNS ≥ 0 且 TNS = 0 → 繼續階段 E，延遲 = 530 / 100 MHz = **5.300 µs**。
2. 若 100 MHz 失敗，依序嘗試 `baseline` / `aggressive` / `fanout` 三種策略（沿用 06L 的做法）。
3. 三種都失敗 → **退頻至 90 MHz**，延遲 = 530 / 90 MHz = **5.889 µs**。這是一個完全可接受的結果，**不是失敗**。
4. 90 MHz 仍失敗 → 停止回報，附上失敗路徑的 timing report。**不要繼續降頻**（會失去與既有設計點的可比性）。

> **不要為了湊到 5.300 µs 而修改 HLS 核心或移除 wrapper 的比對邏輯。** 報告會誠實記載實際收斂頻率。

### 階段 D 回報格式

```
[D] 嘗試序列:
  100 MHz / baseline   : WNS=<...> TNS=<...> <met/failed>
  100 MHz / aggressive : ...
  100 MHz / fanout     : ...
  90 MHz  / baseline   : ...（僅在需要時）
[D] 採用: <...> MHz，WNS=<...> ns，TNS=<...>
[D] 推導延遲: 530 cycles / <...> MHz = <...> µs
[D] 板級 routed 資源: LUT=<...>% FF=<...>% DSP=<...>% BRAM=<...>%
[D] 若有失敗: 最差路徑的起點/終點/延遲明細
```

⚠️ 注意 [D] 的資源數字是**板級全設計**（含 wrapper），會高於 §1.2 的 HLS 核心單獨數字（LUT 21.33% / DSP 89.09%）。兩者不可混用，回報時請明確標示是板級。

---

## 7. 階段 E：燒錄與 4,096 筆比對

1. 以 `tcl/program_pynq_z2_demo.tcl` / `scripts/program_pynq_z2_demo.cmd` 為範本，建立 v3diag 版本（新檔）。
2. JTAG 燒錄至 `xc7z020_1`。
3. 執行完整 4,096 筆重播，讀出 `mismatch_count` 與 `max_readout_lsb_diff`。
4. 以 `03_analyze_wrapper_readout_drift.py` 為範本分析差異分布（新檔，輸出到新目錄）。
5. **關鍵**：計算**決策翻轉筆數**——即差異是否使任何一筆的 `argmax(bid, ask, no_quote)` 或門檻判定改變。既有 25.733 µs 版本為 **0 / 27**。

### 階段 E 回報格式

```
[E] 燒錄: <成功/失敗>，device=<...>
[E] 總筆數: 4096
[E] 差異筆數: <...>
[E] 最大差異: <...> LSB
[E] 差異方向分布: <n> 筆 -1, <n> 筆 +1
[E] 決策翻轉筆數: <...> / <差異筆數>     <- 必須為 0
[E] 與門檻的最小安全邊界: <...> LSB
```

---

## 8. 停止點（這是我監控的介面）

**每個階段結束後停下來，貼出該階段的回報區塊，等我覆核後再進下一階段。**

不要連續跑完 A→E。原因：階段 A 的 port 列表決定階段 B 的做法；階段 D 的頻率決定報告措辭。中間任一步偏離都會讓後續工作白做。

若任一階段的驗收門未通過，**停止並回報，不要自行修補**。

---

## 9. 已知陷阱清單

| # | 陷阱 | 徵兆 | 做法 |
|---|---|---|---|
| 1 | **生成的 RTL 埠數與 `phc_step` 不同** | Vivado 報 port 不存在或未連接 | unroll8 + local partition 可能使 HLS 為 `state_prev` / `x_t` 生成更多讀埠，或把陣列展開為純量埠。**從生成的 `.v` 的 module 宣告逐字讀取**，不要假設與 `phc_step` 相同 |
| 2 | **沿用舊的 BRAM 期望向量** | 4,096 筆全部不符 | 期望值綁定模型；V3 用 05F，必須重算（階段 C 自我檢查會抓到） |
| 3 | **工具版本不同導致週期數變動** | csynth 不是 530 | 原始執行為 Vivado v.2025.2。回報你的版本，不要硬調 pragma 去湊 530 |
| 4 | **100 MHz 板級時序失敗** | WNS 為負 | 這是預期中的風險，走 §6 的決策規則退到 90 MHz |
| 5 | **`.cmd` 寫中文** | 批次檔控制流錯亂、亂碼 | 純 ASCII + CRLF |
| 6 | **Python print 用 emoji / `≈`** | `UnicodeEncodeError` | 用 `[OK]`/`[WARN]`/`[FAIL]` |
| 7 | **腳本在 import 時計算輸出路徑** | 新執行污染既有共用檔案 | 本專案已被此坑傷過一次（`08a` 的 `spatial_frontend_probe_metrics.csv`）。所有輸出路徑必須是**函式參數或 CLI 旗標**，不可在模組層級用常數決定 |
| 8 | **把推導延遲說成量測延遲** | 報告措辭錯誤 | 530 / f 是從 routed 靜態時序推導的；本專案未實作晶片內週期計數器或 ILA 延遲量測 |
| 9 | **1-LSB 漂移的既有根因未定** | 出現少量 ±1 LSB 差異 | 這是**已知且預期**的（25.733 µs 版本為 27/4096，影響上界 0 筆決策翻轉）。出現類似數量級屬正常，不要試圖消除；只需量化並確認決策翻轉為 0 |
| 10 | **Vivado 與其他負載搶 CPU** | 其他實驗被 slow-abort 砍掉 | 本機目前可能有 Python 訓練在跑。開始前確認，或等它結束 |

---

## 10. 完成後我會做的覆核

你回報完階段 E 之後，我會：

1. 核對階段 A 的數字與 `phc_diag_step_export.rpt` 是否一致（防止環境漂移被忽略）
2. 檢查 `bram_v3_diag/` 的 `.mem` 確實不同於 `bram/`（防止陷阱 2）
3. 檢查所有新產物都在新目錄、既有 artifact 的 mtime 未變（防止陷阱 7）
4. 核對 [D] 的板級資源與 [E] 的比對結果是否自相一致
5. 把結果寫入成果報告 §3.7、§4.3.1、§4.6 與 §6.3，並據實調整驗證等級標註
