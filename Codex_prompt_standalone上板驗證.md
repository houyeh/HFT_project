# 給 Codex Agent 的任務說明：PHC-lite Standalone 核心上板驗證（v2，經 reviewer 修訂）

## 背景（你需要知道但不要自己腦補的事）

- 已存在、**禁止修改**的東西：`PHC-lite` 的 Vitis HLS IP（手動撰寫，非 hls4ml 產出），已通過 csim → cosim → csynth → Vivado P&R，在 100MHz 下 WNS=+0.020ns、TNS=0.000ns（22,941 endpoints 全過）。Post-route 資源：LUT 14.20%(7,555/53,200)、FF 7.72%(8,217/106,400)、DSP48E1 21.36%(47/220)、BRAM_18K 2.14%(6/280)。目標板卡：PYNQ-Z2（Zynq-7020）。
- 這次任務的目的：**不是**做 PS–PL 系統整合（不接 AXI、不寫 PS 端軟體、不用 PYNQ Python API）。目的是用 JTAG 介面（ILA/VIO）在實體晶片上驗證這顆 IP 本身的功能與時序是否與 csim/cosim 的參考向量一致——即「P&R 時序收斂在矽片上是否真的成立」。
- 既有測試向量：csim 階段已有完整測試集（約 25,600 步的輸入/輸出參考向量），格式與位置請向我確認後告知你，**不要自己假設格式**。
- **比對原則（重要，先講清楚）**：參考向量若是同一份定點 csim/cosim 的輸出，則硬體必須 **bit-exact**，不存在「容許小誤差」——因為有 state_next→state_prev 遞迴回授，任何非零誤差都會逐步累積，tolerance 比較與遞迴回授在邏輯上不相容。只有在參考向量來自浮點 golden model 時才談 tolerance。請先向我確認向量來源屬於哪一種，再決定比較器的形式；預設走 bit-exact 等值比較。

## 你要做的事（依序）

1. **盤點現有 IP 介面**：讀取現有 HLS 產出的 RTL/IP，回報以下內容後**停下來等我確認**，不要直接動手寫 wrapper：
   - block-level 控制協定是 ap_ctrl_hs 還是 ap_ctrl_chain（ap_start/ap_done/ap_idle/ap_ready）。
   - 所有輸入/輸出 port 的位寬與 port-level 介面型態（ap_none / ap_vld / ap_hs）。
   - ap_rst 的極性與同步/非同步性質。
   - state_next→state_prev 跨時間步回授的接法，以及 **state 的初始值如何載入**——reset 後 wrapper 必須把 state 初始化成與 csim 第 0 步一致的值，否則從第 1 步開始就全錯。這個初始化機制要寫進你的介面盤點報告。

2. **設計 BRAM 自檢測試向量播放器（FSM + BRAM IP）**：
   - 用 Vivado Block Memory Generator 產生 BRAM 存放測試向量。**先根據第 1 步查到的實際 port 位寬計算**：每步需要多少 bits（輸入 + 期望輸出）、25,600 步總共多少，對照目前 BRAM 餘裕（274 個 BRAM_18K ≈ 5.05 Mbit，還要扣掉 ILA 佔用）。**塞得下就全塞，塞不下才取子集**——回報計算過程與最終筆數，不要未經計算就假設塞不下，也不要假裝塞得下。
   - 向量轉檔（csim 輸出 → .coe）需要一個腳本，由你撰寫；**packing 格式（欄位順序、bit 對齊、endianness）必須寫成文件**，否則 mismatch 時無法排查是 DUT 錯還是打包錯。
   - FSM：依序從 BRAM 讀出輸入向量 → 驅動 ap_ctrl 介面 → 等 ap_done → 讀出輸出 → 與期望輸出做 bit-exact 等值比較（除非我另行確認 tolerance）→ 累計 mismatch 計數器、記錄 first-mismatch index，並**把第一筆不一致的 actual 與 expected 輸出鎖存到暫存器**——只有 index 沒有值，上板後 debug 是瞎的。
   - VIO 輸出：done flag、mismatch count、first-mismatch index、first-mismatch actual/expected 鎖存值、目前處理到第幾筆。
   - ILA 要有明確用途，不是裝飾：**trigger 條件設在 mismatch 首次發生**，capture 當步的輸入、DUT 實際輸出、期望輸出、state_prev。回報 ILA 的深度設定與其 BRAM 佔用。

3. **時脈來源要先查清楚，不要假設**：查 PYNQ-Z2 Reference Manual（報告參考文獻 7）確認。線索：PYNQ-Z2 板上有一路 **125MHz 時脈由 Ethernet PHY 供給 PL（pin H16）**，若確認屬實則走此路徑：125MHz 進 MMCM/PLL 降到 100MHz，加上正確的 create_clock 約束，**完全不碰 PS7**。若查證後此路不可行，退而求其次用 PS7 FCLK0：
   - PS7 IP 只能配置成「輸出 FCLK0 給 PL，不連接任何 AXI 周邊、不跑任何 PS 端程式」。
   - 交付文件裡要寫清楚實際採用哪條路徑；若用 PS7，寫「PS7 僅用於提供 PL 時脈，無 AXI 連接、無軟體」，不可含糊說成「完全沒用到 PS」，也不可被誤會成做了 PS–PL 整合。

4. **wrapper 整體行為模擬（強制步驟，在任何 P&R 之前）**：把 wrapper + PHC-lite IP + BRAM 初始化內容放進 XSim 跑整體模擬：
   - 正向測試：至少完整跑過將載入 BRAM 的全部向量，確認 mismatch count = 0、done flag 正確拉起。
   - **負向測試（必做）**：故意竄改 BRAM 中一筆期望輸出，確認比較器確實抓到、mismatch count 與 first-mismatch index/鎖存值正確。沒有負向測試，上板後 mismatch=0 無法排除「比較器根本沒在動」——這是 review 時必被追問的一題。
   - 模擬通過前不得進入 P&R；單次完整 P&R 成本不低，不要拿 P&R 當 debug 手段。

5. **重新跑一次完整 P&R**：wrapper（FSM + BRAM + VIO + ILA + 時脈/重置邏輯）+ 原 PHC-lite IP 一起合成、實作。**這是強制步驟**——既有 WNS=+0.020ns 只含 IP 本身，加上 wrapper 後必須重新確認，不能假設 wrapper 是免費的。回報新的 WNS/TNS 與資源使用率，與原數字並列成表。

6. **產出 bitstream 與上板操作文件**：你的交付邊界到 bitstream + Tcl 腳本 + 一份**上板操作 checklist**（Hardware Manager 步驟、VIO 各 probe 的預期讀值、pass criteria）為止；實體上板由我執行。**不要自己「模擬」或臆造上板結果**，交付物裡不得出現任何未實測的 VIO 讀值。
   - Pass criteria 寫明：mismatch count = 0 且 done = 1，且**至少重跑 3 次（含一次斷電重上電）結果一致**。
   - 時脈目標是「100MHz 功能驗證通過/失敗」的二元結論；若失敗，退回報告提過的 90MHz 備案重新 build，如實記錄。**不要宣稱量測到「最高可穩定運作時脈」**——單顆晶片、單一溫度電壓下的通過頻率不構成 Fmax 論述，reviewer 會直接打掉。
   - 測試向量覆蓋率（實際載入筆數 / 25,600）必須明確寫出。

## 你絕對不要做的事（踩坑提醒）

- **不要修改**現有已驗證的 PHC-lite HLS IP 原始碼或其 RTL 輸出。任何介接需求都在 wrapper 層解決。
- **不要**用人工方式（手動單步觸發 ILA trigger 25,600 次）做測試——必須是自檢 FSM 硬體迴圈。
- **不要**接 AXI、不要寫 PS 端 C/Python、不要用 PYNQ overlay/API——混進去會讓報告的範圍邊界論述失去意義。
- **不要**跳過第 4 步的 XSim 整體模擬與負向測試就進 P&R。
- **不要**假設誤差容許範圍、向量格式、BRAM 可塞筆數、PYNQ-Z2 時脈架構——先查證/詢問。假設錯了浪費一輪 P&R 的成本不小。
- **不要**臆造、模擬、或「預期性地填入」任何上板實測數字。

## 完成後的簡短報告（缺一不可）

1. 新的 P&R 時序（WNS/TNS）與資源數字，與原 IP-only 數字並列。
2. XSim 整體模擬結果：正向（mismatch=0）與負向（注錯必抓到）皆須附證據。
3. 測試向量覆蓋率、BRAM 容量計算過程、packing 格式文件。
4. 時脈路徑最終結論（125MHz PL 直入 vs PS7 FCLK0）與對應的報告措辭。
5. 上板操作 checklist 與 pass criteria（實測欄位留白，由我上板後回填）。

---
完成後請回報，我會用教授/reviewer 角度檢視你的 P&R 結果與 wrapper 邏輯（特別是 state 初始化與回授接線、比較器的負向測試證據、以及新一輪 P&R 是否維持時序收斂）。
