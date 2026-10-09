# Codex 任務包：Stage E — 5.300 µs 設計點上板驗證

**接續**：`CODEX_TASK_5P3US_ONBOARD.md`（Stage A–D 已通過）
**開立日期**：2026-10-04
**前置條件**：`results_v3_diag/pnr_100mhz_baseline/phc_v3diag_demo_100mhz_baseline.bit`
SHA-256 `A4661EB5D94481596040E6586C12AB5E1F11246F7A3A698CA703BE408BE61085`

---

## 0. 本任務包修正了前一包的一項誤判

前一包說「Tcl Store 建議修但不要擋著，對上板沒有影響」。**這句是錯的。**

`tcl/run_pnr_after_xsim_template.tcl` 第 102 行是：

```tcl
create_ip -name ila -vendor xilinx.com -library ip -module_name phc_demo_ila
```

`create_ip` 依賴 Vivado Tcl Store——與 Stage A 阻擋 HLS export 的是**同一個損壞**。所以 ILA 不可用的根因就是它。而 ILA 是唯一能把 `mismatch_count` / `first_mismatch_index` / `max_readout_lsb_diff` 搬出板子的路徑（`led` 只有 4 個位元）。

**但這不代表要先修它。** 見下節。

---

## 1. 一個已查證的事實改變了做法順序

`rtl/pynq_z2_phc_demo_wrapper_v3diag.sv` 的參數預設值：

```systemverilog
parameter integer INJECT_NEGATIVE_TEST = 0,
parameter integer NEGATIVE_TEST_INDEX  = 17,
parameter integer READOUT_TOL_LSB      = 0,      // <-- 零容差
parameter [23:0]  NEGATIVE_TEST_MASK   = 24'h000010,
```

`READOUT_TOL_LSB = 0` 表示比較器做的是**精確比對**（舊的 05C4 wrapper 用的是 1-LSB 容差）。第 507 行：

```systemverilog
row_mismatch = (diff0_lsb > READOUT_TOL_LSB) || (diff1_lsb > READOUT_TOL_LSB);
```

**後果：正向測試本身就是容量測試。** 如果 BRAM 真的只存了部分列（見 §2 的算術），超出範圍的列會讀到零或繞回資料，在零容差下必然不符，`fail` 就會亮。

所以**先跑現有 bitstream 的正向測試**，不要先重建。它同時回答兩個問題，而且零成本。

---

## 2. 待解決的容量疑問（這是 E-0 要回答的）

`v3diag_board_utilization_routed.rpt`：28 × RAMB36E1 + 1 × RAMB18E1。

```
板上 BRAM 位元數 = 28 × 36,864 + 1 × 18,432 = 1,050,624 bits
4096 列 × 432 bits                         = 1,769,472 bits
比例                                        =      59.4%
1,050,624 / 432                             =   2,432.0 列（剛好整數，可疑）
```

已排除的良性解釋：

| 解釋 | 檢驗結果 |
|---|---|
| 字級常數摺疊 | ❌ 18 個字（16 特徵 + 2 readout）全部變動 |
| 位元級常數摺疊 | ❌ 432 位元中 430 個變動，只有 2 個恆為 0 |
| 分散式 ROM（LUT） | ❌ 總 LUT 12,285，核心佔 11,348，只剩約 937 |

合成日誌對 replay 記憶體沒有任何裁剪警告，`$readmemh` 讀取成功、`VECTOR_COUNT bound to: 4096`。

> **不要去逆推合成行為。** 用 E-0 的硬體結果回答：`pass` 亮就代表資料是完整且正確的（某個我們沒看懂的 Vivado 實作細節，但實證上成立）；`fail` 亮就代表確實有問題，再用 ILA 定位。

---

## 階段 E-0：用現有 bitstream 跑正向測試（不重建）

1. 以 `tcl/program_pynq_z2_demo.tcl` / `scripts/program_pynq_z2_demo.cmd` 為範本，建立 v3diag 版本（**新檔**），燒錄 `phc_v3diag_demo_100mhz_baseline.bit` 至 `xc7z020_1`。
2. 燒錄前先確認 bitstream 的 SHA-256 與 §0 所列一致。
3. 觀察 4 個 LED（定義在 `rtl/pynq_z2_phc_demo_board_top_v3diag.sv` L113–116）：

   | LED | 訊號 | 意義 |
   |---|---|---|
   | `led[0]` | `done` | 4,096 筆重播完成 |
   | `led[1]` | `pass` | 全數通過（零容差） |
   | `led[2]` | `fail` | 至少一筆不符 |
   | `led[3]` | `heartbeat[25]` | 時脈活著（應持續閃爍） |

4. **等 `led[0]` 亮起後**再判讀 `led[1]` / `led[2]`。重播時間約 4,096 × 530 cycles / 100 MHz ≈ **21.7 ms**，所以上電後幾乎立刻完成——`led[0]` 應該一直是亮的。

### E-0 的兩種結果與各自的意義

**情況 A：`pass` 亮、`fail` 暗** → 這是最好的結果，而且**直接完成本任務**：

> 4,096 筆、**零容差**、全數位元精確相符。因為 readout 位元精確，決策翻轉**必然為 0**（不需另外計算）。

這比 25.733 µs 版本的「27 筆 1-LSB 漂移 / 0 筆翻轉」**更強**。而且 `pass` 這一個位元就足以支撐全部主張，**不需要 ILA**。此時 §2 的容量疑問在實證上消解。

**情況 B：`fail` 亮** → 需要定位，此時才往 E-1 走。

### E-0 回報格式

```
[E-0] bitstream SHA-256 比對: <相符/不符>
[E-0] 燒錄: <成功/失敗>, device=<...>
[E-0] led[3] heartbeat: <閃爍/不動>
[E-0] led[0] done: <亮/暗>
[E-0] led[1] pass: <亮/暗>
[E-0] led[2] fail: <亮/暗>
[E-0] 判定: <情況 A / 情況 B>
```

**跑完 E-0 停下來回報。** 情況 A 的話後面全部不用做。

---

## 階段 E-1：負向對照（只有情況 A 需要，且只需一次重建）

情況 A 的 `pass` 必須排除「比較器形同虛設」這個可能。做法是刻意注入一個錯誤，確認它被抓到。

**問題**：`INJECT_NEGATIVE_TEST` / `NEGATIVE_TEST_INDEX` 是 wrapper 的參數，而 `pynq_z2_phc_demo_board_top_v3diag.sv` 只對外暴露 `MEM_FILE`（L10）。因此目前無法用 `synth_design -generic` 從外部設定。

**要做的**：

1. 在板級頂層新增四個參數並**原樣傳遞**給 wrapper：
   ```systemverilog
   parameter integer INJECT_NEGATIVE_TEST = 0,
   parameter integer NEGATIVE_TEST_INDEX  = 17,
   parameter integer READOUT_TOL_LSB      = 0,
   parameter [23:0]  NEGATIVE_TEST_MASK   = 24'h000010
   ```
2. 建立**一份**負向 bitstream，參數為
   `INJECT_NEGATIVE_TEST=1`、`NEGATIVE_TEST_INDEX=4095`。
   **選 4095 而不是 17**：它同時驗證比較器有效**且**記憶體確實存到最後一列——一次回答兩件事。
3. 燒錄，確認 `led[2] fail` **亮**、`led[1] pass` **暗**。

### E-1 回報格式

```
[E-1] 板級頂層新增參數: <是/否>，原檔 pynq_z2_phc_demo_board_top_v3diag.sv 是否另存新檔: <...>
[E-1] 負向 bitstream 參數: INJECT=1 INDEX=4095 MASK=24'h000010 TOL=0
[E-1] 負向 P&R: WNS=<...> ns, <met/failed>
[E-1] led[1] pass: <亮/暗>   <- 必須為「暗」
[E-1] led[2] fail: <亮/暗>   <- 必須為「亮」
[E-1] 判定: <比較器有效 / 比較器無效>
```

> ⚠️ 若負向測試在 index 4095 **抓不到**錯誤，表示記憶體沒有存到那一列，而正向的 `pass` 是假的。此時回到情況 B。

---

## 階段 E-2：只有情況 B 才做——修 Tcl Store 並加入 ILA

### E-2.1 修 Tcl Store

```
執行 tcl/reset_vivado_tclstore.tcl（呼叫 tclapp::reset_tclstore）
```

修好後驗證：在 Vivado tcl console 執行一次 `create_ip -name ila -vendor xilinx.com -library ip -module_name test_ila_probe` 確認不再報錯，然後刪掉該測試 IP。

### E-2.2 重建含 ILA 的 bitstream

ILA 探針已在板級頂層宣告（L122–125，`probe1=done`、`probe2=pass`、`probe4=mismatch_count`）。請確認也接上 `first_mismatch_index` 與 `max_readout_lsb_diff`。

⚠️ **時序風險**：ILA 會吃 LUT/FF，而目前 WNS 僅 **+0.148 ns**、DSP 已達 **89.09%**。

**決策規則**：

1. 100 MHz + ILA / baseline → 通過就繼續
2. 失敗 → aggressive / fanout
3. 全失敗 → 改時脈設定 `DIVCLK=1, MULT=8.000, CLKOUT_DIVIDE=10`（PFD 125 MHz，抖動更低，同樣輸出 100 MHz）再試
4. 仍失敗 → **降至 90 MHz**，延遲 = 530 / 90 MHz = **5.889 µs**

> **有數字的 5.889 µs 比沒數字的 5.300 µs 有價值得多。** 不要為了保住 5.300 這個數字而放棄 ILA。

### E-2.3 備援：若 ILA 無論如何裝不進去

repo 內已有 HDMI 數字顯示的實作：

```
rtl/pynq_z2_hdmi_numeric_overlay.sv
rtl/pynq_z2_phc_demo_board_top_hdmi_numeric.sv
tcl/run_pnr_hdmi_numeric_variant.tcl
```

把 `mismatch_count` / `first_mismatch_index` / `max_readout_lsb_diff` 顯示在 HDMI 上，用擷取卡讀出。**先問使用者再走這條路**，因為它是另一個完整的建置週期。

### E-2.4 跑完整比對

ILA 可用後，讀出三個計數器並回報：

```
[E-2] 實際頻率: <...> MHz, WNS=<...> ns
[E-2] 推導延遲: 530 cycles / <...> MHz = <...> µs
[E-2] 板級資源（含 ILA）: LUT=<...>% FF=<...>% DSP=<...>% BRAM=<...>%
[E-2] 零容差（READOUT_TOL_LSB=0）:
        mismatch_count       = <...>
        first_mismatch_index = <...>
        max_readout_lsb_diff = <...>
[E-2] 1-LSB 容差（READOUT_TOL_LSB=1）: mismatch_count = <...>
[E-2] 決策翻轉筆數 = <...> / <差異筆數>     <- 必須為 0
[E-2] 負向對照 index=4095: fail=<亮/暗>
```

> **不要預期 27 這個數字重現。** 25.733 µs 版本是 05C4 dense 的算術路徑；V3 是對角化（逐元素，跨通道混合消失）加上 `grp_sqrt_fixed_49_25_s_fu`（LayerNorm 平方根，28 cycles，`ap_fixed<48,24>`）。漂移筆數可能明顯更多或更少，**兩者都是誠實的結果**。唯一的硬門是**決策翻轉必須為 0**。

---

## 護欄（延續前一包）

1. **不要覆寫既有產物**。新 bitstream 與報告寫到 `results_v3_diag/<新子目錄>/`。
2. **不要修改**原 `rtl/pynq_z2_phc_demo_wrapper.sv`、`pynq_z2_phc_demo_board_top.sv`、`pynq_z2_clock_90mhz.sv`、`bram/`。
3. **不要修改** `hls_src/`、`phc_diag_weights.hpp`、`bram_v3_diag/`。
4. **不要為了讓 `pass` 亮而放寬 `READOUT_TOL_LSB`**。零容差是這版的設計選擇，也是比 05C4 版本更強的主張來源。若零容差下有漂移，就據實回報漂移數字。
5. `.cmd` 一律純 ASCII + CRLF；Python print 不用 emoji 或 `≈`。
6. **延遲一律標示為推導值**：530 cycles ÷ 實際收斂頻率，來自 routed 靜態時序與 RTL cosim 週期數，**不是晶片內實測**。本專案未實作晶片內週期計數器。

---

## 停止點

```
E-0 → 回報 → 若情況 A → E-1 → 回報 → 完成
                 若情況 B → E-2.1 → 回報 → E-2.2 → 回報 → E-2.4 → 回報
```

每個階段結束停下來等覆核。不要連跑。

---

## 我會做的覆核

1. 比對 bitstream SHA-256
2. 情況 A 時：確認 E-1 的負向對照在 index 4095 被抓到（否則 `pass` 無效）
3. 情況 B 時：核對 ILA 讀出的三個計數器與 XSim 的 4,096 筆結果是否自相一致
4. 確認既有 artifact 的 mtime 未變
5. 把結果寫入成果報告 §3.7、§4.3.1、§4.6、§6.3，並據實調整驗證等級標註
