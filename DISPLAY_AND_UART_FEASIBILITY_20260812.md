# 板端數字輸出方案評估：擷取卡路徑 vs UART

日期：2026-08-12
背景：HDMI 數字顯示已標記 `visual_failed_final_abandoned`；使用者現有擷取卡；欲評估 UART 可行性與可讀性。

---

## 0. 結論先講

1. **先做擷取卡測試，不要先做 UART。** 已建好、已通過時序（WNS +0.100ns）、已燒錄過的 numeric bitstream 就在 `results/phc_demo_wrapper_hdmi_numeric.bit`。翻查失敗紀錄後，**失敗徵狀強烈指向「顯示器端的鏈路穩定性」而不是「我們的資料或邏輯」** —— 擷取卡是一個完全不同的 sink，有實質機會直接把這個功能救回來。成本約 20 分鐘，零工程。
2. **UART 可行，但擷取卡幫不上忙 —— 你還缺一條 USB-TTL 轉接線。** PYNQ-Z2 板上的 FT2232HL USB-UART 是接在 **PS 的 MIO14/15**，純 PL 設計碰不到。要從 PL 送資料出來，必須走 Pmod 腳位 + 外部 USB 轉序列線（PmodUSBUART / FTDI TTL cable / CP2102 / CH340，約新台幣 100–400 元）。
3. **「全部 4096 筆逐筆刷在終端機上」在人類可讀性上是無效的**（見 §3 計算）。UART 若要做，正確設計是 **done 之後送一個摘要封包 + 抽樣逐筆資料**，而不是全量即時串流。
4. **無論走哪條路，資料傳輸都必須在 `done` 之後才啟動，絕對不能插進推論路徑** —— 否則 25.733µs 這個數字就作廢了。

我沒有辦法替你實機測試（我這邊沒有板子、沒有 Vivado、沒有 USB 裝置存取權）。以下是可行性推算、判讀依據，以及可直接執行的測試程序與 Codex 任務包。

---

## 1. 為什麼擷取卡有機會救回 HDMI 數字顯示

翻 `AUDIT_hdmi_numeric_display_bringup.md` 的三次嘗試紀錄：

| 嘗試 | 做法 | 時序 | 結果 |
|---|---|---:|---|
| 第一次 | 多位元匯流排直接 double-flop 跨 90MHz→25MHz | WNS +0.213ns（met） | 畫面出現類 hex 字元 + 閃爍轉黑 |
| 第二次 | 改成 source-domain packet latch + toggle 同步（CDC 修好） | **WNS −0.040ns（failed）** | 未燒錄（安全規則擋下） |
| 第三次 | 只在 `done` 上升緣鎖存**一個最終封包**並保持到 reset | **WNS +0.100ns（met）** | 已燒錄；**短暫正確顯示 `4095 / −0.012 / −0.136 / decision 0` 後閃爍轉黑** |

**關鍵推論：第三次的畫面內容在 `done` 之後是完全靜態的。** 沒有逐事件更新、沒有 CDC 活動、沒有變動的像素資料 —— 但畫面**仍然**閃爍轉黑。

> **靜態畫面還會閃 ⇒ 問題幾乎不可能出在我們的資料路徑或 CDC，而是出在視訊鏈路本身能不能被 sink 穩定接受。**

而且第一次的「類 hex 字元」已經查明是 renderer 刻意加的 `E/B/A/D` 標籤造成的視覺誤判（`tb_numeric_bcd` 全數 PASS，BCD 轉換無誤），已修掉。所以資料正確性是有獨立證據的。

**常見的 sink 端失敗原因**（都與擷取卡是否較寬容直接相關）：

- Digilent `rgb2dvi` 送出的是 **DVI 訊號，沒有 HDMI infoframe、沒有音訊封包**。部分消費級螢幕／電視會週期性重新協商或直接判定訊號無效。
- EDID 協商：螢幕回報的支援模式與我們輸出的視訊模式不完全吻合。
- TMDS 時脈邊界：加了 overlay 之後像素區域的繞線壓力上升。

**擷取卡通常對 DVI-mode HDMI 的容忍度高於消費級螢幕**（EDID 較單純、不做花俏的模式協商、也不要求音訊）。所以這是一個真實的、非零機率的翻盤機會，而且**你不需要重新合成任何東西**。

### 測試程序（約 20 分鐘，照做即可）

**前置（無論結果如何都要做，這是 §1.3 一直欠的目視確認）**

1. 先燒 **純色 HDMI fallback**（`scripts/program_pynq_z2_hdmi_demo.cmd`），接**擷取卡**，確認藍(reset)/黃(running)/綠(pass)/紅(fail) 四色都正確切換 → **錄影存證**。
2. 同時確認 LD3 心跳、LD1 pass、LD0 done 三顆燈可見 → **拍照存證**。

> 這一步本身就有價值：`hdmi_status_display` 目前仍是 `visual_observation_pending`，而 numeric 的報告卻寫「已還原成 verified 純色版本」。**紀錄本身有矛盾，必須用實機確認消掉。**

**主測試**

3. `results/phc_demo_wrapper_hdmi_numeric.bit`（第三次、WNS +0.100ns 那版）燒進去。
   - 注意：`scripts/program_pynq_z2_hdmi_numeric_demo.cmd` 內建安全檢查，只有在 routed timing report 顯示 met 時才會燒。第三次是 met，應可通過；**若它拒絕，代表磁碟上的 timing report 是第二次的失敗版本，此時停手並回報**，不要繞過安全檢查。
4. HDMI 輸出接**擷取卡**（不要接螢幕），在筆電上開擷取軟體（OBS / 擷取卡附的預覽程式）。
5. 按按鈕跑完一次，觀察：
   - 是否穩定顯示最終封包 `4095 / bid / ask / decision`？
   - 持續觀察 **60 秒**，是否仍會閃爍或轉黑？
6. **連按三次**重跑，確認三次顯示的最終封包完全相同。

**判讀**

| 結果 | 結論 | 下一步 |
|---|---|---|
| 擷取卡穩定顯示 60 秒以上 | 原失敗是 sink 端問題，**功能救回** | ✅ 直接用。demo 用擷取卡投影，UART **不必做** |
| 仍然閃爍／轉黑 | 問題在板端視訊鏈路本身 | 走 §2 的 UART 路線；HDMI 只留純色狀態 |
| 擷取卡連純色都收不到 | 擷取卡與 rgb2dvi 不相容 | HDMI 全線放棄，UART 成為唯一選項 |

---

## 1B. 擷取卡實測結果與診斷（2026-08-12 更新）

**實測回報**：HDMI 經擷取卡進 Windows 相機，呈現「約三拍顯示、休一拍」的週期性斷續（約 75% duty）。

### 診斷：幾乎可以確定是**非標準的視訊時序**，不是我們的邏輯

查 `rtl/pynq_z2_clock_hdmi_640x480.sv`：

- 輸出模式為 **640×480**，像素時脈 **25.000 MHz**（MMCM 從 125MHz 產生，CLKOUT1_DIVIDE=40 → VCO 1000MHz / 40 = 25.000 MHz，模組註解也明寫）。
- **但 VGA 640×480@60 的標準像素時脈是 25.175 MHz。**
- 25.000 / 25.175 = 0.99305 → 實際更新率約 **59.58 Hz**，比標準低約 **0.7%**。

一般螢幕通常吃得下這 0.7% 偏差；**便宜的 4K30 UVC 擷取棒通常吃不下** —— 它們鎖定標準格式清單，遇到非標準更新率會週期性失鎖再重新鎖定，表現出來正好就是**規律的「顯示幾拍、掉一拍」**。這與觀察到的徵狀完全吻合。

同時：**Windows 相機 App 本身是很差的診斷工具**，它會把掉幀直接畫成空白。**先換 OBS Studio** 再說，因為 OBS 會顯示它實際協商到的解析度與格式 —— 那個資訊本身就是關鍵證據。

### 🎉 好消息：診斷用的 bitstream 早就建好了

`results/` 裡已經有一整組**獨立的 HDMI smoke-test bitstream**（不含 PHC，專門用來隔離視訊鏈路問題）：

| bitstream | 差異 | 為什麼要測 |
|---|---|---|
| `hdmi_fixed_color.bit` | 純色，無數字，無 PHC | **基準線**：連靜態純色都斷續 → 問題與 numeric overlay 完全無關 |
| `hdmi_numeric_test_regpipe_vga.bit` | **使用 25.175 MHz 標準 VGA 時脈** | 🎯 **最可能的解藥**，直接針對上面的診斷 |
| `hdmi_numeric_test_regpipe_bufio.bit` | 序列時脈改走 BUFIO | 若時序邊際才是主因 |
| `hdmi_numeric_test_regpipe.bit` | 加了暫存器管線 | 中間對照組 |
| `hdmi_numeric_test.bit` | 最初版 | 對照組 |

### ✅ 診斷已交叉驗證（reviewer 覆核，2026-08-12）

我逐檔覆核了 §1B 的推論，**結論成立，而且比原本寫的更強**：

| 覆核項 | 實際內容 | 判定 |
|---|---|---|
| `pynq_z2_clock_hdmi_640x480.sv` | `DIVCLK_DIVIDE=5` → PFD 25MHz；`CLKFBOUT_MULT_F=40.0` → VCO 1000MHz；`CLKOUT1_DIVIDE=40` → 像素 **25.000 MHz**；`CLKOUT0_DIVIDE_F=8` → 序列 125.000 MHz | ✅ 25.000 MHz 屬實 |
| `pynq_z2_clock_hdmi_640x480_vga.sv` | `CLKFBOUT_MULT_F=40.25` → VCO 1006.25MHz → 像素 **25.15625 MHz**、序列 125.78125 MHz。註解已載明：25.2MHz 需非法的 MMCM 分數乘數，故取最接近的合法值 | ✅ 對策存在且設計合理 |
| `pynq_z2_hdmi_timing_640x480.sv` | H: 640+16+96+48 = **H_TOTAL 800**；V: 480+10+2+33 = **V_TOTAL 525**；hsync/vsync 皆為負極性 | ✅ **完全符合 VESA 640×480@60 標準** |
| 五個 smoke-test bitstream | `hdmi_fixed_color.bit`、`hdmi_numeric_test.bit`、`..._regpipe.bit`、`..._regpipe_vga.bit`、`..._regpipe_bufio.bit` 全部存在 | ✅ |

**這個覆核讓診斷更乾淨**：H/V 總數、消隱、同步極性**全部是標準值**，唯一偏離標準的就是像素時脈。

- 現況：25.000 MHz / (800×525) = **59.524 Hz**（標準 59.940 Hz，低 **0.69%**）
- vga 版：25.15625 MHz / (800×525) = **59.896 Hz**（低 **0.073%**，等同標準）

**所以「換 vga bitstream」是一個精準對症的單變數實驗，不是亂試。**

另外一個正向訊號：接螢幕是「瘋狂閃爍」，接擷取卡變成「三拍顯示、休一拍」（約 75% duty）。**換 sink 帶來大幅改善，本身就是「問題在鏈路相容性而非板端邏輯」的證據。** 方向對了，繼續往這條路走。

### 測試階梯（照順序，每一步約 5 分鐘）

**Step 0｜先換工具，不要動硬體**
用 **OBS Studio** 取代 Windows 相機，把擷取卡加成 Video Capture Device，記下 **OBS 回報的輸入解析度與幀率**。若 OBS 顯示穩定，問題只是 Windows 相機 → 直接結案。

**Step 0.5｜🔴 先排除 USB 端，這是最容易被忽略、又完全免費的一項**

「規律地顯示幾拍、掉一拍」是 **USB 頻寬不足**的典型徵狀，跟時脈偏差的徵狀幾乎一樣。**在燒任何新 bitstream 之前先排除它**：

1. **擷取卡直插 USB 3.0 埠**（藍色接口），**不要經過 hub、不要用延長線、不要插 USB 2.0 埠**。
   4K30 擷取棒若落到 USB 2.0（480 Mbps），連 1080p30 未壓縮都撐不住，一定週期性掉幀。
2. 在 OBS 的裝置屬性裡把**視訊格式改成 MJPEG**（不要用 YUY2 / 未壓縮）。YUY2 在 USB 2.0 上必掉幀。
3. 在 OBS 裡把**解析度／FPS 設成「裝置預設」**，不要強制指定，讓它照實協商。
4. 關掉其他佔用 USB 頻寬的裝置（外接硬碟、webcam）。

做完這三件事再看一次。**如果 duty 從 75% 變成 100%，就結案了，連 bitstream 都不用換。**

**Step 1｜基準線**：燒 `hdmi_fixed_color.bit`，觀察 60 秒。

- 純色也斷續 → 視訊鏈路本身的問題，與 numeric overlay 無關 → 往 Step 2
- 純色穩定、numeric 才斷續 → 是 overlay 造成的額外壓力 → 往 Step 3

**Step 2｜換標準時脈（最可能有效）**：燒 `hdmi_numeric_test_regpipe_vga.bit`，觀察 60 秒。

- 穩定 → **找到解藥**。接著唯一的工程工作是：把 vga 時脈模組移植進完整的 PHC numeric top，重跑 P&R（必須 routed timing met），再重燒。
- 仍斷續 → 往 Step 3

**Step 3｜換序列時脈路由**：燒 `hdmi_numeric_test_regpipe_bufio.bit`，觀察 60 秒。

**Step 4｜若以上全數失敗**
代表這張擷取卡根本不接受 640×480 這個等級的輸入（很多 4K30 擷取棒的最低支援輸入是 720p）。此時只有兩條路：

- 把視訊模式改成 **720p60**（像素 74.25 MHz、序列 371.25 MHz）。**這是真正的工程任務**：新 MMCM、新 timing generator、字型要放大、重新 P&R，而且序列時脈接近 z7020 -1 速度等級的上限。**在剩餘時間內風險偏高。**
- 放棄 HDMI 數字，走 §2 的 UART。

### 判讀重點

無論結果如何，**Step 1 的答案要記錄下來並修正專案紀錄**：目前 `AUDIT_hdmi_numeric_display_bringup.md` 稱純色版本為 "verified"，但 `AUDIT_hdmi_status_display_bringup.md` 的 status 仍是 `visual_observation_pending`。**這兩份紀錄互相矛盾，必須用這次實測消掉。**

### 🛑 硬性停損（一定要遵守）

**Step 0.5 + Step 2 做完就停。** 這兩步涵蓋了兩個最可能的根因（USB 頻寬、像素時脈偏差），而且成本都極低。

- 若 Step 0.5 或 Step 2 之後穩定 → ✅ 收工，把 vga 時脈模組移植進完整 PHC numeric top、重跑 P&R（**必須 routed timing met，不 met 就退回**）、重燒、再目視確認一次。
- 若兩步都做完仍然斷續 → **不要再往 Step 3/4 鑽**。720p60 改造需要新 MMCM、新 timing generator、放大字型、重新 P&R，而且 371.25 MHz 序列時脈接近 z7020 -1 速度等級上限，在剩餘時間內是不可接受的風險。**直接切到 UART，HDMI 只保留已驗證的純色狀態顯示。**

理由：HDMI 數字顯示是**加分項**，不是必需品。§3 的保底組合（連按三次的重複性展示 + 筆電同步視窗 + deadline miss 畫面）不依賴它。把時間押在一個已經失敗三次的功能上，是這個階段最容易犯的錯。

---

## 2. UART 可行性（已查證接腳事實）

### 2.1 硬體事實（查證結果）

| 項目 | 事實 | 對我們的影響 |
|---|---|---|
| 板載 USB-UART | FT2232HL，連到 **PS 的 MIO14 / MIO15**（PS UART0） | ❌ **純 PL 設計無法使用**。要用它就得把 PS 拉進設計，那是大改動 |
| Pmod JA（PL 腳位） | JA1_P/N = Y18/Y19，JA2_P/N = Y16/Y17，JA3_P/N = U18/U19，JA4_P/N = W18/W19 | ✅ 可用。**只需要 1 支腳當 TX** |
| Pmod JB（PL 腳位） | JB1_P/N = W14/Y14，JB2_P/N = T11/T10 | ✅ 備用 |
| Pmod 電氣 | 每個 Pmod 提供 2×3.3V（pin 6/12）、2×GND（pin 5/11）、8 支邏輯訊號 | 3.3V TTL，與一般 USB-TTL 轉接器相容 |

**所以你還缺的東西是：一條 3.3V USB-TTL 序列轉接線**（Digilent PmodUSBUART、FTDI TTL-232R-3V3、CP2102、CH340 皆可）。
⚠️ **必須是 3.3V 版本**，5V 的會打壞 Zynq I/O。
⚠️ **擷取卡對 UART 完全沒有幫助**，這是兩條獨立的路。

### 2.2 傳輸量計算

每個事件要送的欄位：event index（0–4095）、bid readout、ask readout、decision code。

| 編碼 | 每筆位元組 | 4096 筆總量 | @115200 baud (11.52 KB/s) | @921600 baud (92.16 KB/s) |
|---|---:|---:|---:|---:|
| 二進位打包（idx 2B + bid 3B + ask 3B + dec 1B + 分隔 1B） | 10 B | 40,960 B | **3.56 s** | **0.44 s** |
| ASCII CSV（`4095,-0.012,-0.136,0\r\n`） | ~24 B | 98,304 B | **8.53 s** | **1.07 s** |
| **僅摘要封包（done 後 1 行）** | ~80 B | 80 B | **7 ms** | **<1 ms** |
| 抽樣逐筆（每 64 筆送 1 筆，共 64 行 ASCII） | 24 B | 1,536 B | **0.13 s** | **0.017 s** |

（8N1 每位元組 10 bits；115200 / 10 = 11,520 B/s）

**結論：頻寬完全不是問題。** 115200 就綽綽有餘，不需要冒險拉高 baud。

### 2.3 可讀性分析（這才是真正的限制）

**全量 4096 筆逐筆刷在終端機上，人類讀不了。** 即使 8.5 秒跑完，那是每秒 480 行捲動 —— 評審只會看到一片模糊的文字瀑布，資訊量等於零，而且會顯得雜亂。

**正確的設計是分三層：**

| 層 | 內容 | 傳輸量 | 人類可讀性 |
|---|---|---:|---|
| **L1 摘要（必做）** | `DONE idx=4095 pass=1 bid_cnt=1430 ask_cnt=1701 noq_cnt=965 last_bid=-0.012 last_ask=-0.136 dec=0` | 80 B | ✅ **極佳**。一行，證明數字來自板子，可直接投影 |
| **L2 抽樣逐筆（建議）** | 每 64 筆送一筆 → 64 行 | 1.5 KB | ✅ 良好。0.13 秒送完，之後可慢速捲動或畫成圖 |
| **L3 全量逐筆（選配）** | 4096 筆全送 | 40 KB | ⚠️ 給筆電程式吃、畫成即時圖表，**不要直接印在終端機** |

**建議：L1 + L2 為必做，L3 視時間決定。**
L1 是關鍵 —— 它一行就消滅了「你們畫面上的數字是筆電算的，不是板子出來的」這個質疑，而這正是目前 demo 最大的軟肋。

### 2.4 🔴 最重要的設計約束

> **UART 傳輸必須在 `done` 之後才啟動，絕對不能插進推論的資料路徑。**

理由：整段 4096 筆重播的建模時間是 **105.377 ms**，而 UART 傳完要 0.13–8.5 秒。如果 UART 與推論交錯，就會改變 replay FSM 的時序，**25.733µs/事件這個數字立刻失去意義**，整個專案的核心主張跟著垮。

**正確做法**：replay 跑完、`done` 拉起之後，另一個獨立的 FSM 去走既有的 readout BRAM，把資料串出去。推論路徑一個 cycle 都不動。

---

## 3. 風險比較

| 方案 | 工程量 | 需要重新 P&R | 需要新硬體 | 風險 | 對「數字來自板子」的說服力 |
|---|---|---|---|---|---|
| 擷取卡測既有 numeric bitstream | **0** | ❌ | 已有 | **極低** | ✅ 高（板子自己畫的） |
| UART L1 摘要 | 小（~80 行 RTL） | ✅ | USB-TTL 線 | **中低** | ✅ 高 |
| UART L1+L2+L3 | 中 | ✅ | USB-TTL 線 | 中 | ✅ 高 |
| 重做 HDMI overlay | 大 | ✅ | — | **高（已三次失敗）** | — |

UART 相對 HDMI overlay 風險低很多的具體理由：**沒有像素時脈、沒有視訊時序、沒有 TMDS、沒有 CDC 多位元匯流排、沒有 sink 相容性問題**。就是一個 baud 除頻器加一個移位暫存器，單一輸出腳位。歷史上三次 HDMI 失敗有兩次的根因（CDC 多位元、時序邊際）在 UART 方案裡根本不存在。

---

## 4. Codex 任務包（擷取卡測試失敗才發）

```
任務：新增 PL 端 UART TX，在 replay done 之後把板上結果串回筆電
（不得修改推論路徑、不得修改任何已驗證的 bitstream / RTL top，全部走 sibling 新路徑）

背景：
PYNQ-Z2 板載 FT2232HL USB-UART 接在 PS MIO14/15，本專案是純 PL 設計，無法使用。
因此改用 Pmod JA 的一支 PL 腳位當 UART TX，外接 3.3V USB-TTL 轉接線。
目的：讓 demo 畫面上的數字有「確實由 FPGA 輸出」的證據，取代目前只能在筆電端同步播放
已知結果的做法。

🔴 絕對禁止（違反其中任一項，這次工作作廢）：
1. 不得修改、覆寫既有已驗證產物：
   - rtl/pynq_z2_phc_demo_board_top.sv（LED-only 已驗證版本）
   - rtl/pynq_z2_phc_demo_board_top_hdmi.sv（純色 HDMI 已驗證版本）
   - results/phc_demo_wrapper.bit、results/phc_demo_wrapper_hdmi.bit
   要新增請開 sibling 檔案與 sibling bitstream 路徑。
2. 不得修改 PHC HLS core RTL、BRAM payload（.mem/.coe）、model weights、threshold、
   checkpoint、任何 NEW1–NEW5 資料。
3. 🔴 **UART 傳輸不得插入推論資料路徑。** 必須在 wrapper `done` 上升緣之後，
   由一個獨立 FSM 讀取既有 readout BRAM 再串出。replay FSM 的 cycle 數必須完全不變 ——
   請在報告中明確證明這一點（附 XSim 的 done 時間點 cycle 計數，改動前後相同）。
4. 不得為了塞進 UART 而放寬時脈或改變 90MHz 設定。
5. 不得同時執行多個 Vivado 流程，全部序列化。

具體要做的事：

Step 1（先做，做完停下來等我確認）：獨立 UART TX 模組 + XSim 驗證
- 寫一個 `rtl/pynq_z2_uart_tx.sv`：參數化 CLK_HZ 與 BAUD，8N1，單一 tx 輸出，
  含 busy/ready 握手。時脈域用既有的 90MHz replay 域，不新增時脈。
- 寫 testbench `sim/tb_uart_tx.sv`，用 XSim 驗證：
  (a) 送出已知位元組序列後，逐位元取樣還原，比對完全相同；
  (b) baud 週期誤差 < 2%（8N1 的容忍上限約 ±5%，留安全邊界）；
  (c) 連續多位元組不掉資料。
- **這一步不需要板子、不需要 P&R**，先把模組本身證明對。

Step 2：整合到 sibling top + 三層輸出協定
- 新增 `rtl/pynq_z2_phc_demo_board_top_uart.sv`（sibling，不動既有 top）。
- 在 done 之後依序送出：
  L1 摘要一行：`DONE idx=<最終index> pass=<0/1> bid=<筆數> ask=<筆數> noq=<筆數>
                last_bid=<有號十進位> last_ask=<有號十進位> dec=<0/1/2>\r\n`
  L2 抽樣：每 64 筆送一行 `<idx>,<bid>,<ask>,<dec>\r\n`（共 64 行）
  L3（用編譯期 ifdef 開關，預設關閉）：全量 4096 行
- baud 固定 115200。**不要為了快而拉高 baud**，頻寬計算顯示 115200 已足夠
  （L1+L2 合計約 1.6 KB，0.14 秒送完）。
- Q24.12 轉有號十進位請重用既有 `phc_numeric_binary_to_bcd`（已有 tb_numeric_bcd 全數 PASS 的
  驗證紀錄），不要重寫一套。

Step 3：接腳約束與 P&R
- 新增 sibling XDC，TX 指派到 Pmod JA 的一支腳（建議 JA1_P = Y18），
  IOSTANDARD LVCMOS33。**不要改既有 XDC 檔案。**
- 跑 P&R，必須 routed timing met；若 WNS 為負，停下來回報，不要自行放寬約束。
- 報告中列出：WNS / TNS / failing endpoints、LUT/FF/DSP/BRAM 相對純色 HDMI 版本的增量。

Step 4：驗證證據
- XSim 跑 sibling top，抓 UART TX 波形解碼出實際字串，與預期字串逐字比對。
- 證明 replay FSM 的 done cycle 計數與改動前完全相同。

完成後給我：
1. Step 1 的 tb 結果（PASS/FAIL 與 baud 誤差數字）
2. sibling top / XDC / bitstream 的路徑
3. routed WNS/TNS 與資源增量表
4. XSim 解碼出的實際 UART 字串
5. done cycle 計數改動前後對照
6. 明確聲明哪些既有檔案「完全沒有被修改」
```

---

## 5. 建議執行順序

1. **今天**：擷取卡測試（§1）。同時把欠很久的 LED / 純色 HDMI 目視確認一次做完並存證。
2. **測試結果決定分支**：
   - 成功 → HDMI 數字顯示救回，UART 不做，把時間全部投到簡報與 frontier 圖。
   - 失敗 → 買 3.3V USB-TTL 轉接線，發 §4 的 Codex 任務，**先只做 Step 1**（純模擬，零板端風險），確認模組正確後再決定要不要走 Step 2–4。
3. **無論哪個分支**：選項 A（連按三次的重複性展示）都要做，那是零成本且無法造假的確定性證明。

---

## 附：資料來源

- PYNQ-Z2 USB-UART（FT2232HL）接 PS MIO14/15、Pmod JA/JB 為 PL 腳位與腳位編號：Digilent / PYNQ-Z2 Reference Manual v1.0、v1.1
- HDMI numeric 三次嘗試紀錄與時序數字：`n2ew_era/06s2_pynq_z2_wrapper_bringup/AUDIT_hdmi_numeric_display_bringup.md`
- 4096 筆決策分布（1430 bid / 1701 ask / 965 no_quote）與 105.377ms 建模播放時間：`n2ew_era/06s2_pynq_z2_wrapper_bringup/FPGA_KNOWN_RESULT_REPLAY_VIEWER.md`
