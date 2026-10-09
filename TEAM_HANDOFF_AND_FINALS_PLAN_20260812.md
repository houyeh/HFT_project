# 隊友交接 + 決賽收尾行動計畫

日期：2026-08-12
用途：給隊友快速接手，以及決賽最後階段的分工與定位
姊妹文件：`COMPETITIVE_STRATEGY_FINALS_20260809.md`（對手分析與評分策略）

---

## 0. 三十秒版本（先看這段）

我們做的**不是**一台 HFT 機器。我們做的是：

> **在一顆 PYNQ-Z2（Zynq-7020）上，把一個輕量時序神經網路的「單筆推論延遲」從 83.16µs 壓到 5.30µs（15.69×），期間產生 5 個全部通過布局繞線時序收斂的設計點，並且替每一個點標上它在真實台指期委託簿上的樣本外 action-level H20 markout 代價。**

一句話賣點：**延遲是有價格的，我們把那個價格量出來了。**

我們**不是**因為遞迴模型比較準才選它 —— 我們自己的實驗證明前饋 TCN 一樣好甚至更好。我們選遞迴，是因為**只有遞迴的 O(1) 狀態能在這顆晶片上壓到 5.3µs**；TCN 的 O(window) 歷史緩衝要吃 23.57% BRAM，每事件 7,439 cycles。**這個「為了硬體而選架構」的反轉，是我們最有記憶點的東西。**

---

## 1. 現在的真實狀態

### 1.1 已經完成、可以拿去講的（全部有報告可查）

| # | 成果 | 證據等級 | 來源 |
|---|---|---|---|
| R1 | **延遲–action-level H20 markout Pareto frontier，5 個 routed 點，83.160µs → 5.300µs（15.69×）** | CSim + Cosim + routed P&R | `06u_phc_latency_breakthrough/AUDIT_*.md` |
| R2 | **實體板燒錄成功**，25.733µs 版本，LED 心跳/pass/done + HDMI 純色狀態 | JTAG programmed，routed timing met | `06s2_pynq_z2_wrapper_bringup/WRAPPER_BRINGUP_STATUS.md` |
| R3 | **位元級比對：4096 筆重播，27 筆 1-LSB 漂移，0 筆決策翻轉**，最小安全邊界 72 LSB | XSim 全量 dump 逐筆比對 | `06s2.../1LSB_DRIFT_ROOT_CAUSE_AND_IMPACT.md` |
| R4 | **FI-2010 對 DeepLOB：weighted-F1 66.50±1.15%（2,132 params）vs 80.35%（60k–142k params）** | 3-seed，定義已對齊並通過 sanity check | `08a_fi2010_public_benchmark/F1_DEFINITION_RECONCILIATION.md` |
| R5 | **微秒級 stale exposure 實測**：2.233%@25.733µs／3.712%@83.16µs／7.199%@262.6µs(CPU p99)，差 4.965pp | LOBSTER ns 級時戳，整數奈秒解析 | `06r_lobster_us_latency_budget/AUDIT_*.md` |
| R6 | **統計紀律**：NEW5 鎖定 holdout 全程只開封一次；matched-random 三個 split 皆 empirical p<0.001 | 凍結 artifact + 一致性 gate 全過 | `05c4`, `06t`, `07a` |
| R7 | **TCN 對照線也做到 routed 收斂**（distilled 100MHz WNS +0.187，1008-event cosim pass） | 完整驗證鏈 | `07c2d_tcn_streaming_iface_slim/AUDIT_*_pnr.md` |

### 1.2 已經放棄或卡住的（不要再投入）

| 項目 | 狀態 | 說明 |
|---|---|---|
| **HDMI 數字顯示** | `visual_failed_final_abandoned` | 三次嘗試，最後一版 routed WNS +0.100ns 也燒進去了，但畫面閃爍後轉黑。**已正式放棄**，板子已還原成純色 HDMI 版本。 |
| 07b2 TCN naive full-window | `closure_blocked` | cosim timeout + 100MHz/90MHz P&R 皆失敗。 |
| 07c2 / 07c2b TCN | cosim timeout | 已被 07c2d 取代，07c2d 才是有效的 TCN 結論。 |
| 06L unroll16 @100MHz | 路由失敗 | baseline/aggressive/fanout 三種策略全失敗，根因為 FSM→DSP48E1 PCIN 高扇出繞線。**這是有價值的失敗紀錄，要留在報告裡。** |
| 07d 決策蒸餾 | `resolved_negative` | 最佳樹需 251 leaves，最弱 split 只保留 33.3% 優勢。目標是可解釋性，與延遲無關。 |
| 06p / 06q TXF 微秒分析 | `resolution_limited` | TXF 時戳 1ms 粒度，物理上無法解析微秒。**不要再試。** |

### 1.3 🔴 兩件現在就能做、零成本、但一直卡著的事

| 項目 | 狀態 | 要做什麼 |
|---|---|---|
| LED 目視確認 | `board_programmed_with_done_led_latch_fix_observation_pending` | **把板子接上，按按鈕，親眼確認 LD3 心跳 / LD1 pass / LD0 done 都看得到**，然後拍照存證。 |
| HDMI 純色狀態目視確認 | `routed_pnr_met_jtag_programmed_visual_observation_pending` | **接上螢幕，確認藍(reset)/黃(running)/綠(pass)/紅(fail) 四種顏色都正確切換**，錄一段影片。 |

> 這兩件事拖了很久，而且是決賽 demo 的地基。**在做任何新東西之前先做完這兩件。** 如果其中任何一項實際上是壞的，整個 demo 計畫要重排 —— 現在知道還來得及，決賽當天知道就完了。

---

## 2. 核心邏輯鏈（給隊友的「一套說法」）

這是整份專案抽出來的推理骨架。**照這個順序講，每一步都有實驗支撐。**

**步驟 1｜問題設定**
交易決策的成本來自**最壞情況延遲**，不是平均延遲。延遲越長，決策抵達時委託簿已經變動的機率越高。
→ 支撐：R5（stale exposure 隨延遲單調上升，實測）

**步驟 2｜時序資訊是必要的**
把時間順序打亂或關掉記憶，PnL 直接由正轉負（reverse_sequence −2,764,500；memory_off −803,235，val split）。
→ 支撐：`05a/AUDIT_learned_memory_vs_fixed_window.md`

**步驟 3｜但「遞迴」不是唯一實現方式 —— 這是我們誠實的反轉**
前饋 TCN 在 causal-fill PnL 上打平甚至勝過遞迴教師（distilled TCN：val +3,340 / NEW3 +98,925 / NEW4 +11,955，3-seed，matched-random p<0.001）。
→ **所以我們不能說「我們選遞迴因為它比較準」。**
→ 支撐：`07a/AUDIT_recursion_necessity_evidence_chain.md`（status 明寫 `recurrence_not_required_by_current_diagnostics`）

**步驟 4｜遞迴的真正優勢在硬體，不在準確率**

| | PHC-lite（遞迴） | TCN streaming（前饋） |
|---|---|---|
| 狀態儲存 | **O(1)**，固定大小狀態向量 | **O(window)**，需歷史緩衝 |
| BRAM | **0.00%** | **23.57%** |
| 每事件 interval | **530 cycles**（V3 diagonal） | **7,439 cycles** |
| 最佳 routed 延遲 | **5.300µs** @100MHz | 74.38µs @100MHz（distilled） |
| Cosim 數值保真 | 最大 **1 LSB** | readout max diff 0.150 / 0.420 |

> ⚠️ 誠實邊界（**簡報必須標註**）：07c2d 報告自己聲明「TCN streaming latency 是 amortized streaming-step 物件，與 PHC recurrent single-step 不是同一個 latency 定義」。**可以比的是 interval（每事件 7,439 vs 530 cycles）與資源（BRAM 23.57% vs 0%）**，不要直接把 74.38µs 對 5.30µs 講成「快 14 倍」而不加註解。

#### 步驟 4 補充｜為什麼前饋反而做不出更短的 throughput？（評審一定會問）

直覺上前饋沒有 loop-carried dependency，II 應該可以逼近 1，理論上該比遞迴快。**這個直覺在資源無限時是對的，在 z7020 上是錯的**，原因有三層：

**(a) 每事件的計算量本質上就比較大。**
遞迴每事件只更新一個大小 h=16 的狀態；diagonal 版本更是把 16×16 dense 退化成 element-wise。串流式 TCN 每事件仍要跑 5 個 block 的 dilated conv（channels × kernel × blocks），**每事件的 MAC 數是一個數量級的差距**。

**(b) 真正的瓶頸是 BRAM port，不是相依性 —— 而且有量化證據。**
O(window) 的歷史緩衝放不進 register，只能放 BRAM（**23.57%**）。BRAM 每塊只有 2 個 port，所以每事件要讀的歷史值被 port 數卡死，計算被迫序列化。

> **決定性證據：unroll 從 1 加到 4，cycles 只從 16,510 降到 7,438（2.22×，遠低於理想的 4×）。**
> 這種明顯的次線性 scaling，是 memory-port bound 的典型特徵。如果瓶頸是算力，加 unroll 應該接近線性；如果瓶頸是 port，加再多平行單元也搶不到頻寬。

**(c) 遞迴的 O(1) 狀態小到可以完全 array-partition，這是它唯一的、但決定性的優勢。**
PHC 的 **BRAM 用量是 0.00%** —— 狀態全部在 register/LUTRAM，可以完全展開平行存取。06U V3 能做到 530 cycles，關鍵 pragma 就是 **local array partition**；而且**沒有這個 pragma 的版本 cosim 直接失敗（268/1000 mismatch）**，證明它不是可有可無的微調，是這個設計點成立的前提。

**（也因此，TCN 的 interval 7,439 ≈ latency 7,438，代表它根本沒有跨事件 pipeline。** 要 pipeline 就得複製計算單元或做更深的 dataflow，需要的 DSP/BRAM 超過 z7020 的量。又回到資源牆。）

**一句話總結（可直接當簡報講稿）：**
> 「在資源無限的機器上，前饋會贏。但在一顆固定大小的晶片上，勝負是由『狀態放得進 register，還是必須放進 BRAM』決定的 —— 前饋的歷史緩衝被 BRAM 的兩個 port 卡住，遞迴的 O(1) 狀態可以完全展開。**這就是我們選遞迴的真正理由，而它是一個硬體理由，不是準確率理由。**」

**步驟 5｜於是我們做延遲的設計空間探索，並用 PnL 標價**
→ 支撐：R1 的 5 個 routed 點（見 §4 數字表）

**步驟 6｜結論**
> 我們選遞迴不是因為它比較準，是因為它的狀態是 O(1)，只有它能在這顆晶片上壓到 5.3µs。這是一個**演算法—硬體協同設計**的決策，而且我們有前饋對照組的完整證據可以證明這個取捨是真的。

**這條鏈的價值**：它把一個「我們的模型沒有比較準」的弱點，翻轉成一個「我們知道為什麼要選它」的強點。評審對誠實的取捨分析給分，遠高於對膨脹的準確率宣稱。

---

## 3. 決賽定位：怎麼瞄準第一名

### 3.1 誠實的處境判斷

對手裡有 Alveo U280、KV260、RFSoC 4x2、HBM-PIM、112/224Gbps SerDes。**在「規格看起來多硬」這條軸上，我們拿不到第一，不要去比。**

創意 35 + 難易度 25 = **60 分是勝負手**。我們的路只有一條：

> **成為全場唯一一件「結果是一個科學發現」而不是「一個能動的實作」的作品。**

30 隊裡有 15 隊在做 AI 推論加速。他們的成果句型都是「我們把 X 加速了 N 倍」。我們的成果句型是：

> **「我們發現：在這個問題上，遞迴架構的價值不在準確率，而在它的狀態複雜度是 O(1)；我們用一條量化的延遲–損益曲線證明了這件事。」**

這是評審講評時會想引用的句子。**「念念不忘」不是靠畫面炫，是靠一句別人沒講過、而且你有證據的話。**

### 3.2 四個評分項各自的主打

| 評分項 | 主打 | 一句話 |
|---|---|---|
| 創意 35 | **用 PnL 當單位做 FPGA design-space exploration** + 架構選擇的反轉結論 | 「沒有人用『這個設計值多少錢』去畫 Pareto frontier。」 |
| 難易度 25 | 5 routed 點 + **DSP 89.09%** + 有紀錄的 P&R 失敗 + 前饋對照組也做到收斂 | 「我們不是做出一個能跑的版本，是做出六個並且知道每一個的邊界在哪。」 |
| 設計品質 25 | 位元級比對 0 決策翻轉 + WNS 全列 + 驗證等級標籤 | 「每個數字都標了它是估計、模擬、路由後、還是板上。」 |
| 報告完整性 15 | 68 個 phase，每個主張可回溯到報告路徑 | 「包含我們證偽的假設。」 |

### 3.3 動態展示問題 —— 正面處理

**先接受事實**：我們的 demo 不會比「低光影像增強前後對比」或「絞肉機停下來」更好看。**在視覺刺激這條軸上競爭是輸定了。**

但「動態」有兩種：**功能會動**（他們）vs **結果可預測**（我們）。要主動把定義搶過來：

> 「其他作品的展示，是證明它**會動**。我們的展示，是證明它**每次都一樣**。」

#### 選項比較（依風險排序）

| 選項 | 內容 | 風險 | 建議 |
|---|---|---|---|
| **A. 重複性展示** | 現場**連按三次**按鈕跑同一段重播，三次 LD1 pass 全亮、結果位元相同；同時筆電上 CPU 跑三次，三條延遲曲線都不一樣 | **零**（不改任何東西） | ✅ **一定要做**。這是確定性論述唯一正確的展示形式，而且無法造假 |
| **B. 已完成的筆電同步視窗** | `FPGA_KNOWN_RESULT_REPLAY_VIEWER` 已 ready，已驗證與 BRAM 讀出 0 LSB 不符、4096 筆決策 100% 吻合（1430 bid / 1701 ask / 965 no_quote） | **零**（已完成） | ✅ 直接用。**畫面必須標明「FPGA 已知結果同步播放，非即時讀回」** |
| **C. Deadline miss 即時畫面** | CPU 15.125% 的點超過 83.16µs deadline 會閃紅；FPGA 全部在線下 | **零**（純軟體） | ✅ 做。這是最有金融意義又不依賴市場時戳解析度的指標 |
| **D. UART 把數字從板子傳回筆電** | PL 端一個 115200 baud 的 UART TX，每事件送 event_idx + bid + ask + decision（約 16 bytes，4096 筆約 5.7 秒） | **中**（要新 P&R + 重燒 + bring-up），但**遠低於 HDMI overlay**（無 pixel clock、無 video timing、無 TMDS） | ⚠️ **唯一能真正解決「數字是筆電算的」這個質疑的方法**。前提：**要先確認有沒有 Pmod-to-USB 序列轉接器**（板上 USB-UART 接在 PS MIO，純 PL 設計用不到）。**有轉接器才啟動，沒有就放棄** |
| **E. HDMI 數字顯示** | — | — | ❌ **已放棄，不要重啟**。三次失敗，時間不夠 |

**建議組合：A + B + C 是保底（零風險，現在就能定案）。D 只有在「LED/HDMI 目視確認完成」且「確認有轉接器」兩個條件都成立時才啟動。**

---

## 4. 統一數字口徑（所有人只能用這張表）

### 4.1 延遲–訊號品質 frontier

| 點 | 延遲 | cycles @ freq | routed WNS | DSP | 驗證等級 | 訊號來源 | val / NEW3 / NEW4 action-level H20 markout 加總 |
|---|---:|---|---:|---:|---|---|---:|
| 06F v8 | 83.160µs | 8316 @100MHz | +0.020ns | 21.36% | cosim + routed | 05C4 dense | 2,037,925 / 5,237,410 / 1,871,685 |
| 06L unroll16 **（板上）** | **25.733µs** | 2316 @90MHz | met | 56.8% | cosim + routed + **實體板** | 05C4 dense | 同上 |
| 06U V1 @90MHz | 24.778µs | 2230 @90MHz | +0.352ns | 68.64% | cosim + routed | 05C4 dense | 同上 |
| 06U V1 @100MHz | 22.360µs | 2236 @100MHz | +0.004ns | 68.64% | 1000-vec cosim + routed | 05C4 dense | 同上 |
| 06U V2 hidden8 | 12.020µs | 1202 @100MHz | +0.445ns | 59.55% | 1000-vec cosim + routed | 05E hidden8 | 1,908,845 / 5,032,255 / 1,755,610 |
| 06U V3 diagonal | **5.300µs** | 530 @100MHz | +0.130ns | **89.09%** | 1000-vec cosim + routed | 05F diagonal | 1,771,525 / 4,727,425 / 1,642,680 |
| （V3 unroll16） | 3.830µs | 383 @100MHz | — | **452/220 超標** | csynth only | — | — |

三句解讀：83.160 → 22.360µs（3.72×）**同模型同權重、action-level H20 markout 加總完全不變**；22.360 → 5.300µs（再 4.22×）代價是 NEW3 action-level H20 markout −9.74%、val −13.07%；再往下 3.830µs 卡在 DSP 452 > 220。

### 4.2 FI-2010（F1 定義已對齊 DeepLOB Table II 的 weighted 口徑）

| 模型 | 參數量 | accuracy % | weighted-F1 % | macro-F1 % |
|---|---:|---:|---:|---:|
| PHC-lite h16 fair512（3 seeds） | **2,132** | 66.45±1.07 | **66.50±1.15** | 63.97±1.45 |
| PHC-lite h64 capacity（seed0） | 23,876 | 66.12 | 66.32 | 64.43 |
| TCN small raw（seed0） | 2,344 | 63.92 | 64.02 | 61.23 |
| DeepLOB（Table II, k=50） | 60,000 / 142,435 | 80.51 | **80.35** | 未報告 |

差距 13.85pp weighted-F1，參數比 1/28.1 ～ 1/66.8。**兩個 F1 都要報，只報好看的那個就是誤導。**

### 4.3 CPU 對照（單一口徑：06F N=4 round 1）

mean 72.199µs / p50 65.600µs / p95 126.300µs / **p99 262.600µs** / deadline 83.16µs / **miss rate 15.125%**。
06S 的 N=4（p99 646.3µs、6299.4µs）**只能**標示為「污染／壓力情境」引用，不得與 06F 混段落。

---

## 5. 禁講清單（給全隊，違反會被當場拆穿）

| ✗ 不能講 | ✓ 改成 | 為什麼 |
|---|---|---|
| 「我們做的是 HFT 系統」 | 「確定性推論核心」 | 25.7µs 不是 HFT 等級，Boutros 2017 是 870ns |
| 「CPU 和 FPGA 在同一段 TXF 重播跑出不同的錢」 | 不提；改用 deadline miss 與 LOBSTER | 實測 burst 窗 CPU 1334 vs FPGA 1331，**4096 筆只差 3 筆** |
| 「我們選遞迴因為它比較準」 | 「因為它的狀態是 O(1)」 | 07a 自己的結論是 TCN 打平或更好 |
| 「hidden64 反而略輸」 | 「落在 hidden16 三 seed 分布 65.18–67.30% 之內」 | 原講法是單 seed 對三 seed 平均，不乾淨 |
| 把 LOBSTER 數字說成台指期數字 | 標明「2012-06-21 NASDAQ 五檔公開樣本，非 TXF」 | 資料集不同 |
| 「TCN 比我們慢 14 倍」 | 「每事件 interval 7,439 vs 530 cycles，且兩者 latency 定義不同」 | 07c2d 自己聲明不可直接排名 |
| 「我們的 CPU 展示比優化過的 C++ 快」 | 「同一數量級」 | 已驗證不成立 |
| 「1-LSB 已找到根因」 | 「影響上界為 0 筆決策翻轉，根因未定，定位在 wrapper RTL 回授邊界」 | S_DRAIN 假設是負結果 |

---

## 6. 分工建議

### 你（主導者）
1. 🔴 **今天就做**：LED + HDMI 目視確認 + 拍照錄影（§1.3）
2. 把 §2 的六步邏輯鏈寫成簡報骨架
3. 決定 D（UART）啟不啟動 —— 先確認有沒有 Pmod-to-USB 轉接器

### 隊友 A｜FPGA / 板端
- 先把 §1.3 兩項目視確認做完並存證
- 練習選項 A（連按三次的重複性展示），確認三次都穩定 pass，記錄任何一次異常
- **若且唯若**轉接器到位 → 評估 UART TX 方案；**不准碰**：HLS core、BRAM `.mem`、threshold、checkpoint、既有已驗證 bitstream（要新增就開 sibling 路徑）
- 提醒：HDMI 數字顯示已正式放棄，不要重啟

### 隊友 B｜Demo 軟體 / 視覺
- 整合 B（`FPGA_KNOWN_RESULT_REPLAY_VIEWER`）+ C（deadline miss 閃紅）+ CPU 抖動散點與 running-max 曲線
- 畫 frontier 圖：橫軸「單筆推論延遲 µs（log）」、縱軸「NEW3 action-level H20 markout 加總」，並註明非 causal-fill sequential PnL；5 點標 WNS / DSP / 驗證等級
- **所有圖都要有軸標籤與單位**，且每張圖要能回答「這條線的數字從哪來」
- **不准改** `03_cpu_live_replay_demo.py` 的 CPU 側量測邏輯

### 隊友 C｜報告 / 簡報
- 用 §4 三張表取代所有舊數字
- 做「我們證偽了什麼」投影片：hidden64 容量、learned vs fixed decay、07d 蒸餾、06p/06q 解析度限制、06L 100MHz 路由失敗
- 做「驗證深度階梯」表：L1 演算法 → L2 因果回測 → L3 鎖定 holdout → L4 HLS 合成 → L5 cosim 數值一致 → L6 P&R 時序收斂 → L7 實體板閉環
- 匿名規則掃描：檔名、圖檔、參考文獻不得出現校名／系所／LOGO／指導教授姓名
- FPGA 功能佔比切分圖（賽規要求 FPGA ≥ 60% 功能性）

---

## 7. 已知的小洞（決賽前補或準備好答案）

| 洞 | 現況 | 處理 |
|---|---|---|
| 手續費假設無推導文件 | `02b/AUDIT_fee_sensitivity_figure.md` status `mixed`：maker fee 15 沒有 derivation document | 補一段來源說明，或準備「這是敏感度分析的其中一個設定值」的答法 |
| 03b6/04b1 反轉根因 | status `mixed`，未完全收斂 | 準備一句話說明，不要主動提 |
| 1-LSB 根因未定 | 影響已界定為 0 筆翻轉 | 照 §5 的講法，被問才展開 |
| 隊友的 22-feature 方案 | 細節不明 | 直接跟隊友要特徵清單與訓練腳本，不要假設它等於既有的 12-feature HGB |
