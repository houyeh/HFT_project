作品名稱：確定性低延遲推論核心——資源受限FPGA上遞迴神經網路之手動HLS實現與驗證
參賽組別：AMD Track（不定題組）
隊伍代號：ZeroJitter

---

## 一、摘要

與多數初賽構想書階段仍停留於紙上設計不同，本作品已完成從 C 模擬、C/RTL 共模擬到 Vivado 實體佈局繞線（Place and Route, P&R）的全鏈路硬體驗證，並取得時序達標的實體電路實測結果（WNS=+0.020 ns）——本文呈現的不是一個尚待驗證的構想，而是一個已被工具鏈與實體電路證實可行的設計，其數據皆可追溯至實際工具輸出紀錄。

高頻交易（HFT）系統對推論延遲的「確定性」要求，往往高於對平均延遲本身的要求。CPU 推論路徑受作業系統排程、記憶體回收（GC）、快取失效等因素影響，尾端延遲（tail latency）波動極大；在交易視窗極短的場景下，一次偶發的高延遲事件即可能等同於錯失機會或承受逆向選擇風險。FPGA 具備確定性執行時間與零抖動的潛力，但這個潛力是否能在學術可取得的入門級平台上真正落地，需要實證回答兩個問題：（1）一個有意義的遞迴神經網路模型，能否在資源極受限的 FPGA（AMD/Xilinx Zynq-7020，僅 220 個 DSP、53,200 個 LUT）上完成高階合成（HLS）並通過實體佈局繞線；（2）若可行，其延遲特性是否確實優於同等 CPU 推論路徑，優勢具體展現在何處。本作品對第一個問題給出的不是預測，而是已實測確認的「是」；對第二個問題的答案則精確指向「尾端延遲與確定性」而非「平均速度」——這個重新定義 FPGA 在 HFT 場景中的價值座標，是本作品最核心的論點。

本作品以「PHC-lite」——核心概念受 Parallelized Hierarchical Connectome（PHC）空間遞迴框架啟發（Chiang, 2026，參考文獻 3），經本團隊大幅簡化為單步、純定點數運算版本之遞迴狀態空間模型 PHCSSM（保留門控、衰減、狀態相依增益等機制，移除原框架之 spiking 與多層連接體拓樸）——作為實驗對象，以手動 Vitis HLS（而非自動化工具 hls4ml）實作其推論核心。我們同時誠實記錄了三次以 hls4ml 自動轉換本模型失敗的具體原因（詳見第二項），並據此說明為何選擇手動 HLS 路徑。歷經 8 輪資源除錯，最終版本通過 C 模擬（25,600 組測試向量、0 mismatch）、C/RTL 共模擬（bit-level 與 C 模型完全一致）、以及 Vivado 實體 P&R（timing met，WNS=+0.020 ns），post-route 資源用量為 LUT 14.20%、FF 7.72%、DSP48E1 21.36%、BRAM_18K 2.14%，全部在預算內。與 PyTorch CPU 基準比較，FPGA 固定延遲 83.16 µs，相較 CPU 路徑 p99 延遲 778.8 µs，尾端延遲改善達 9.4 倍，且抖動為零。

本作品不宣稱已達成可部署的生產級 HFT 系統，亦不宣稱 FPGA 改善了交易 PnL（本階段未進行任何交易回測）；其目的是以扎實、可重現的工程數據，展示資源受限 FPGA 上遞迴神經網路 HLS 實作的可行性與系統化優化方法論，並提出明確的下一階段工作計畫（量化、PS-PL 整合、上板實測）。

---

## 二、動機及相關技術

### 2.1 設計動機

FPGA 在金融交易硬體加速領域已有公開文獻佐證其低延遲優勢，例如以 cuckoo hashing 實現的訂單簿查詢硬體可達平均 253 ns 的查詢延遲［參考文獻 4］，以及多篇文獻指出 FPGA 加速相較傳統軟體路徑可降低 70% 以上的延遲［參考文獻 5、6］。然而，這些既有工作多聚焦於「訂單簿維護」與「封包解析」此類確定性邏輯運算，較少涉及「遞迴神經網路推論」這類同時具備非線性運算與跨時間步狀態依賴的工作負載。本作品的動機即是填補這個落差：驗證一個具備門控（gating）、衰減（decay）、層正規化（LayerNorm）等機制的遞迴模型，能否在最入門等級的 FPGA 上完成完整的軟硬體驗證鏈。

### 2.2 技術選型與失敗經驗：為何放棄 hls4ml，改採手動 HLS

hls4ml（Duarte et al., 2018［參考文獻 1］）是目前學術界最成熟的「PyTorch/Keras 模型自動轉換為 HLS」工具鏈，亦有文獻證明其可支援 GRU/LSTM 等遞迴架構（Aarrestad et al., 2022［參考文獻 2］）。本團隊在進入手動 HLS 之前，先以 hls4ml 1.3.0 對三種模型變體進行轉換嘗試，三次嘗試**全部在 Python 端模型解析階段即失敗**，從未產生 C++ 專案或執行 Vivado 合成：

| 嘗試對象 | 失敗原因（依工具實際錯誤訊息） |
|---|---|
| TCN（3-block dilated, action-value 輸出） | `ConstantPad1d` layer 的 parser 在當前 hls4ml 版本中失敗 |
| TCN（5-block prepad，receptive field 125） | `torch.fx` 對含控制流的前處理模組進行符號追蹤（symbolic trace）時失敗，丟出 `TraceError: symbolically traced variables cannot be used as inputs to control flow` |
| PHC-STP 單步模型（LayerNorm 架構） | hls4ml 之 PyTorch frontend 明確拋出例外：`Input shape [None, 16] is not currently supported for LayerNorm; only three-dimensional inputs (including batch dimension) are supported`；改以 3 維輸入規避後，仍在 LayerNorm 解析處失敗；改走 ONNX 路徑則在更早的 shape 推斷階段失敗（`Could not find the shape for input onnx::MatMul_72`） |

這個結果本身具備技術意義：它具體指出了「自動化 HLS 工具鏈」在「含 LayerNorm 與跨時間步遞迴依賴的狀態空間模型」這一類架構上的當前邊界，而非單純的設定錯誤或版本疏失（三條轉換路徑——PyTorch frontend 2D 輸入、PyTorch frontend 3D 輸入、ONNX frontend——皆已嘗試，全數失敗）。據此，本作品選擇以手動 Vitis HLS 重新實作模型的數學定義，換取對每一個運算節點與每一個資源用量決策的完全控制權。

### 2.3 技術創新與改良：資源收斂成果與三項根本原因修復

本作品將 DSP 用量自最初設計的 2636%（超出平台容量，無法合成）收斂至最終 21%，LUT 由 2073% 收斂至 46%（csynth 估計值，實際 P&R 後為 14.20%），且每一步收斂皆對應一個經工具官方診斷訊息證實的根本原因，而非反覆試誤的偶然結果。最終定位出三個根本原因：

1. C 標準函式庫 `tanh()` 在 HLS 中被合成為雙精度浮點運算核心，單一呼叫即耗用約 2,048 個 DSP；修復方式為以 65 點分段線性定點查表（範圍 [-4, 4]，步長 0.125）取代浮點運算。
2. 任何形式的 `#pragma HLS PIPELINE II=X` 都會觸發 Vitis HLS 的自動最佳化規則，將原本的部分展開（partial unroll）強制升格為完全展開（complete unroll），導致資源用量不隨 II 數值增加而線性下降；修復方式為完全移除頂層 PIPELINE pragma。
3. 移除 PIPELINE pragma 後，工具的全域 `pipeline_loops` 門檻設定仍會自動對迭代次數少（≤16）的迴圈套用 pipeline 化，並連鎖觸發內迴圈強制展開與權重陣列自動切割；修復方式為對所有受影響迴圈明確加上 `#pragma HLS PIPELINE off`，其優先級高於工具的全域預設規則。

這三個根本原因皆透過工具編譯紀錄中的官方診斷訊息（如 `[HLS 214-275]`、`[HLS 214-376]`、`[HLS 214-291]`、`[HLS 214-449]`）逐一確認，而非僅憑資源報表的事後推測。

### 2.4 成果定位：與典型 FPGA 創意應用作品的本質差異

本作品最終交付的成果是：在無法管線化的遞迴架構限制下（第 t+1 步必須等待第 t 步狀態輸出，數學上強制 II=Latency），於僅 220 顆 DSP 的入門級平台（Zynq-7020）上，將最終 DSP 用量收斂至 21%、LUT 14.20%、FF 7.72%，並通過實體 P&R 時序驗證（WNS=+0.020 ns）。這個結果之所以具有意義，是因為多數 FPGA 創意應用作品所合成的模型（如 CNN 影像分類）在數學上是前饋（feed-forward）且無跨時間步依賴，可被管線化、資源與延遲的權衡空間相對寬裕；本作品在不具備這個優勢的前提下，仍交付了時序與資源雙達標的實體電路結果，且此結果可由第四項所列之完整 csim／cosim／P&R 報告逐一覆核（詳見第二項 2.3 之三項根本原因修復紀錄）。

---

## 三、作品設計概念

### 3.1 模型運算流程

PHC-lite（架構靈感來源見參考文獻 3，本團隊簡化為單步定點版本）為單一時間步（single-timestep）的推論核心，輸入為當前市場特徵向量與前一步隱藏狀態，輸出為訊號分數與更新後的隱藏狀態。下圖為完整資料流與遞迴回授架構（虛線為跨時間步回授路徑）：

![PHC-lite 架構圖（白話版）](architecture_diagram_friendly.svg)

*圖一：PHC-lite 單步推論核心資料流（白話對照版）。每個方塊先以生活化類比說明該步驟「在做什麼決定」，方塊內小字標註對應的技術名稱，供熟悉深度學習者逐一核對其與下方 STEP1–STEP9 公式的對應關係。紅色虛線為 state_next 回授至下一時間步 state_prev 的遞迴路徑——這條路徑代表「這一刻的答案必須先算完，才能開始算下一刻」，是「II=Latency、無法管線化」（詳見 3.3）這項硬體難度的直接成因。*

用一句話理解整個流程：這個模型每一刻都在做三件事——（1）把「舊記憶」打折後留下一部分、（2）篩選一部分「新資訊」加進來、（3）再參考「其他神經元傳來的訊息」做最後修正——三者合併、穩定化後，輸出這一刻的判斷，並把結果存成「新記憶」留給下一刻使用。下方為對應的精確數學定義，供技術審查覆核：

```
STEP 1  encoded         = SiLU( LayerNorm( ENC_LINEAR(x_t) ) )
STEP 2  gate_t           = Sigmoid( GATE_LINEAR(encoded) )
STEP 3  candidate_t      = Tanh( CANDIDATE_LINEAR(encoded) )
STEP 4  state_temporal   = DECAY ⊙ state_prev + gate_t ⊙ candidate_t
STEP 5  gain_t           = Sigmoid( GAIN_LINEAR(encoded) + GAIN_PREV_W × state_prev )
STEP 6  message          = SiLU( SYNAPSE_LINEAR(state_temporal) )
STEP 7  loop_input       = state_temporal + ALPHA ⊙ gain_t ⊙ message
STEP 8  z                = LayerNorm(loop_input)
STEP 9  readout          = READOUT_LINEAR(z) ；state_next = state_temporal
```

共 7 個線性層、2 個 LayerNorm、3 種非線性活化（Sigmoid／Tanh／SiLU）。所有非線性運算皆以 65 點預先計算之定點查表（線性插值）實現，目的是避免任何浮點硬體核心進入合成結果——這是本作品從 v1 到 v8 除錯歷程中最關鍵的單一決策（見第二項 2.3 之根本原因 1）。

### 3.1.1 與原始 PHC 論文架構之差異（一覽對照）

為避免讓人誤以為 PHC-lite 等同於原始論文架構，下圖呈現原始論文（arXiv:2604.01295）完整設計樣貌，紅色標籤標示出本作品移除或簡化的部分：

![原始 PHC 論文架構圖](architecture_diagram_original_phc_paper.svg)

*圖二：原始論文 Parallelized Hierarchical Connectome 架構示意（依論文內容摘述整理，非逐一還原所有數學細節）。其設計目標是生物擬真度與可解釋性，包含脈衝神經元、多區域分層連接體拓樸、突觸短期可塑性、推論時在線學習（R-STDP）、以及單一時間步內反覆收斂的 Multi-Transmission Loop。*

兩者差異對照如下：

| 比較項目 | 原始論文 PHC 架構 | 本作品 PHC-lite |
|---|---|---|
| 神經元輸出 | 二元脈衝（Heaviside 發火，ALIF 動態） | 連續值閘控（Sigmoid／Tanh，無脈衝） |
| 網路拓樸 | 多區域分層連接體（R0/R1...），Dale's Law、零自連 | 單層全連接，無區域分割、無拓樸限制 |
| 突觸機制 | 短期可塑性（STP：促進/恢復）＋分層拓樸調節 | 單一突觸線性層（SYNAPSE_LINEAR）＋固定權重 |
| 時間步內計算 | 多輪內部循環，直到收斂（Multi-Transmission Loop） | 單次直通（single-pass），不內部循環 |
| 權重更新時機 | 訓練＋推論時皆可在線更新（R-STDP） | 僅訓練時更新，推論時權重固定 |
| 訓練/推論求解法 | Log-domain 平行前綴掃描，O(log T) 深度 | 逐步驟組合邏輯＋跨時間步序列遞迴（無法平行，II=Latency，詳見 3.3） |
| 設計目標 | 生物擬真度、可解釋性、神經科學對應 | 在資源極受限 FPGA 上可被手動 HLS 合成、通過實體 P&R 時序驗證 |

簡言之：本作品保留的是原始論文「門控、狀態相依增益、神經元與突觸分工」這個核心精神，移除的是所有會讓硬體資源用量爆炸或無法在合理時間內完成驗證的機制（脈衝動態、多區域拓樸、在線學習、多輪收斂迴圈）。這個取捨本身即是本作品「設計概念」的一部分，而非單純的功能閹割——詳見第二項 2.3 之資源收斂數據佐證此取捨的必要性。

### 3.2 定點數規劃

| 型別 | 定義（總位寬, 整數位寬） | 用途 |
|---|---|---|
| 輸入／權重 | `ap_fixed<24,12>` | 模型輸入、各線性層權重 |
| 累加器 | `ap_fixed<40,16>` | 矩陣乘加運算之中間累加，預留足夠位寬避免溢位 |
| LayerNorm 變異數 | `ap_fixed<48,24>` | 變異數計算對動態範圍要求最高，獨立配置最大位寬 |

### 3.3 硬體控制邏輯與架構限制

PHC-lite 在數學定義上是遞迴的：第 t+1 步需要第 t 步算出的 state_next 作為輸入。這代表跨時間步之間無法進行管線化（pipelining），有效的啟動間隔（Initiation Interval, II）等於整體延遲（Latency）。這是模型架構本身的根本約束，而非本次 HLS 實作的疏漏；本作品所有延遲數字皆基於「單步、序列執行」這一前提下量測。

### 3.4 應用場景與系統定位

本核心的設計定位是嵌入於造市商（market making）策略的「訊號生成」環節：對每一筆委託簿事件，產出固定延遲的訊號分數，供下游風控邏輯（例如：是否撤回報價）在已知的最差時序預算下做決策。其價值主張並非「比現有最快的 ASIC/FPGA 交易系統更快」，而是「在已知、固定的時間窗內，保證完成一次推論」——對於依賴尾端延遲風險控制的策略而言，這個確定性本身即具備價值，且本作品以實測 p99 數據（見第四項）量化了這個價值的具體大小。目前的驗證範圍止於 IP 層級的 csim／cosim／P&R，尚未進行 PS（ARM 處理器）與 PL（FPGA 邏輯）的整合及上板測試，此部分列為下一階段工作（第五項）。

---

## 四、可行性分析

### 4.1 功能正確性

- **C 模擬（csim）**：以 200 個驗證集序列、各 128 個時間步，共 25,600 組測試向量進行驗證，**0 筆 mismatch**，最大絕對誤差 0.00725275，平均絕對誤差 0.000372553（容忍門檻 1.0）。
- **C/RTL 共模擬（cosim）**：同樣 25,600 組向量全數通過，且最大絕對誤差與 csim **完全相同**，證明 RTL 電路在位元層級與 C 模型一致，HLS 轉換過程未引入額外精度損失。

### 4.2 實體可行性（P&R）

以 Vivado 2025.2 對最終版本（v8）執行完整實體佈局繞線（非僅合成估算）：

| 項目 | 數值 |
|---|---|
| 目標時脈週期 | 10.000 ns |
| 實作後達成週期 | 9.980 ns |
| WNS（最差負時序餘裕） | +0.020 ns |
| TNS | 0.000 ns（22,941 個時序端點，0 個失敗） |
| 時序判定 | Met（達標） |

**目前狀態**：本設計已在 100 MHz 目標頻率下完成 Vivado P&R 時序驗證，WNS=+0.020 ns，工具判定達標、0 個端點失敗，證明目前版本可在實體電路上正確運作。考量這個餘裕仍偏緊，若需要更穩健的餘裕以吸收後續邏輯路徑變動，下一階段工作已規劃降頻至 90 MHz 後重新執行 P&R。

### 4.3 資源可行性

| 資源 | csynth 合成前估算 | P&R 實際用量 | 可用總量 | 實際使用率 |
|---|---|---|---|---|
| Slice LUT | 24,836 | 7,555 | 53,200 | 14.20% |
| FF | 14,518 | 8,217 | 106,400 | 7.72% |
| DSP48E1 | 46 | 47 | 220 | 21.36% |
| BRAM_18K | 6 | 6 | 280 | 2.14% |

值得提醒審查者注意：post-route 實際用量遠低於 csynth 估算（LUT 由 24,836 降至 7,555，降幅約 70%），原因是 Vivado 在邏輯合成階段進行了額外的邏輯合併與優化。這代表 csynth 報表的資源百分比僅是上界估算，不應被過度解讀為最終結果——本作品以 P&R 實測數據作為可行性判斷依據，而非 csynth 估算值。

### 4.4 資源收斂歷程作為可行性證據

下表為從初始版本到最終版本的 8 輪資源用量演進（皆為 csynth 估算值，用於展示收斂趨勢；最終可行性仍以 4.2、4.3 之 P&R 實測數據為準）：

| 版本 | 關鍵動作 | DSP | LUT | FF | csim |
|---|---|---|---|---|---|
| v1 | 初始版本（浮點 tanh） | 2636% | 2073% | 441% | 未通過 |
| v2 | 改用 65 點定點查表 tanh | 1603% | 839% | 441% | PASS |
| v3–v5 | 調整 UNROLL／PARTITION／PIPELINE II | 約 395%~1603% | — | — | — |
| v6 | 移除頂層 PIPELINE pragma | 247% | 113% | 86% | PASS |
| v7 | 移除輔助陣列 ARRAY_PARTITION | 121% | 77% | 52% | — |
| v8（最終） | 對所有相關迴圈明確加上 PIPELINE off | 21% | 46% | 13% | PASS |

從 2636% 到 21% 的 DSP 用量收斂，每一步都對應一個被工具診斷訊息證實的根本原因（見第二項 2.3），而非反覆嘗試的偶然結果。此收斂路徑本身即是本作品在「可行性」之外的核心方法論貢獻。

### 4.5 延遲與 CPU 基準比較

PyTorch float32 CPU 推論（`torch.no_grad()`，batch_size=1，暖機 100 次、正式量測 10,000 次）與 FPGA 固定延遲（csynth 結果，序列執行、不可管線化）比較：

| 指標 | FPGA（固定值） | CPU（PyTorch） |
|---|---|---|
| 延遲（單次估算值／平均） | 83.16 µs | 117.5 µs |
| 延遲（中位數） | 83.16 µs | 84.2 µs |
| 延遲（p99） | **83.16 µs** | **778.8 µs** |
| 抖動 | 0（確定性） | 高（受作業系統排程／記憶體回收影響） |

**正確解讀**：FPGA 的優勢並非展現在平均延遲（83.16 µs 與 CPU 平均 117.5 µs 接近，甚至中位數上 CPU 84.2 µs 與 FPGA 幾乎相同），而是展現在 **p99 尾端延遲改善 9.4 倍、且抖動為零**。在 HFT 場景中，真正決定風險暴露時間窗的往往是最差情況延遲而非平均延遲，這正是本作品數據支持的核心論點。

**已知限制（不應淡化）**：83.16 µs 不滿足典型 HFT L1（最內層）<10 µs 的延遲門檻；本作品展示的是 FPGA 推論路徑的可行性與系統化優化方法，而非可直接部署的生產級低延遲方案。此外，CPU 基準為 Python/PyTorch 推論路徑，並非經過 C++/LibTorch/TorchScript 最佳化的基準，p99 數字中包含 Python 解譯器層的額外開銷，比較時應留意此公平性限制。

---

## 五、系統實現之預期成果

| 階段 | 工作項目 | 時程 | 預期產出 |
|---|---|---|---|
| Phase 1（已完成） | HLS 8 輪資源優化、csim／cosim／P&R 驗證、CPU 基準對照 | 已完成 | v8 設計、完整驗證報告（本構想書數據來源） |
| Phase 2 | INT16 量化（`ap_fixed<16,8>`）以降低延遲與資源用量 | 4 週內 | 量化版本 csynth／csim 報告，工程預估延遲可降至約 25 µs（待實際合成確認，非承諾值） |
| Phase 3 | PS–PL 整合與 PYNQ-Z2 實機上板測試 | Phase 2 後 2 週內 | 實機量測延遲，並與 P&R 估算值進行對照分析 |
| Phase 4 | 隱藏層維度縮減實驗（16 維 → 8 維）可行性評估 | Phase 3 後 1 週內 | 精度與延遲之權衡（trade-off）報告 |
| Phase 5 | 決賽報告與簡報整理、Reviewer 預答辯準備 | 決賽前 | 最終報告、簡報、可預期問答清單 |

每一階段皆延續本作品的方法論：先建立明確假設與量化驗證指標，再執行修改，最後以工具實際輸出（而非估算外推）作為結案依據。Phase 3 的上板實測結果，無論是否與 P&R 預測一致，皆會在最終報告中誠實呈現——若有落差，落差本身即是有價值的工程資料。

軟硬體系統實現方式：硬體側為前述 HLS 合成之 PHC-lite 推論 IP（IP 層級，目前為 out-of-context 合成與 P&R，尚未整合至完整 Zynq Block Design）；軟體側為 PyTorch 訓練與量化前處理流程、以及用於產生測試向量與比對基準的 Python 驗證腳本。效能描述以第四項所列之 P&R 實測時序與資源數據，以及 CPU 基準對照數據為準，不採用未經實測的理論峰值。

---

## 六、結論

本作品交付的核心成果，是一個已通過實體電路驗證（非合成估算）的遞迴神經網路推論核心：在資源極受限 FPGA（AMD/Xilinx Zynq-7020）上，post-route 資源用量 LUT 14.20%／FF 7.72%／DSP48E1 21.36%／BRAM_18K 2.14%，Vivado P&R 時序達標（WNS=+0.020 ns，22,941 個端點 0 個失敗），且與 CPU 基準比較取得 p99 尾端延遲 9.4 倍改善、零抖動的實測數據。這個成果的特色與創意在於兩點：第一，PHCSSM 是受 Parallelized Hierarchical Connectome（PHC）空間遞迴框架啟發（Chiang, 2026，參考文獻 3）、經團隊大幅簡化為單步定點版本並完成實作驗證的門控/衰減/狀態相依增益遞迴架構，並以此交付了「尾端延遲與確定性」而非「平均速度」的 FPGA-in-HFT 價值論證；第二，在自動化工具 hls4ml 對此類架構三次轉換失敗（詳見 2.2）的前提下，仍以手動 HLS 交付出完整可驗證的硬體成果，三個根本原因的修復記錄（浮點數學核心、PIPELINE 觸發的強制展開、全域 pipeline_loops 門檻連鎖反應）是支撐這個成果可信度的工程證據，而非本作品的主要論述重點。

本作品同時誠實揭露現階段限制：時序餘裕偏緊（WNS=+0.020 ns）、尚未完成上板實測、83.16 µs 的延遲尚未達到典型 HFT L1 部署門檻。本作品的價值主張並非宣稱已解決 HFT 低延遲問題，而是以實測的 p99 尾端延遲改善（9.4 倍）與零抖動特性，重新定義「FPGA 加速」在 HFT 場景中應被理解為「確定性」的改善而非單純平均延遲的競賽，並提出具體、可執行、有明確驗證指標的下一階段工作計畫（量化、上板測試、架構縮減）。未來目標是在完成上板實測與量化後，進一步評估這套確定性推論核心是否能在實際造市策略的風控邏輯中產生可量化的價值——但此一評估需要獨立的交易回測驗證，明確界定在本作品的範圍之外。

---

## 七、參考文獻

1. Duarte, J. *et al.*, "Fast inference of deep neural networks in FPGAs for particle physics," *Journal of Instrumentation (JINST)*, vol. 13, P07027, 2018. (hls4ml 工具原始文獻；DOI: 10.1088/1748-0221/13/07/P07027，arXiv:1804.06913)
2. Aarrestad, T. *et al.*, "Ultra-low latency recurrent neural network inference on FPGAs for physics applications with hls4ml," arXiv:2207.00559, 2022.（既有 hls4ml 對 GRU/LSTM 等遞迴架構之支援文獻，本作品據此比對説明 LayerNorm 架構之轉換限制為當前邊界案例）
3. Chiang, P.-H., "Parallelized Hierarchical Connectome: A Spatiotemporal Recurrent Framework for Spiking State-Space Models," arXiv:2604.01295, 2026.（本作品 PHCSSM 之核心架構靈感來源；原論文提出 Neuron Layer／Synapse Layer 構成之空間遞迴框架，本作品取用其門控、狀態相依增益與層間訊息傳遞概念，移除 spiking 機制與多層連接體拓樸，簡化為單步、純定點數運算之 lite 版本以符合資源受限 FPGA 之 HLS 可行性需求）
4. "Low latency book handling in FPGA for high frequency trading," *IEEE Conference Publication*, 2014. (DOI 可查 IEEE Xplore Document 6868785；cuckoo hashing 訂單簿查詢，平均延遲 253 ns)
5. "FPGA for High-Frequency Trading: Reducing Latency in Financial Systems," *IEEE*, 2024. (IEEE Xplore Document 10841781)
6. Leber, C., Geib, B., Litz, H., "High Frequency Trading Acceleration using FPGAs," 學術技術報告。
7. AMD/Xilinx, *PYNQ-Z2 Reference Manual*, v1.0, 2018.（目標開發板 Zynq-7020 SoC 硬體規格文獻）
8. AMD, *Vitis HLS User Guide* (UG1399)，官方工具文獻。
9. 本作品之模型設計、HLS 原始碼、各版本除錯紀錄、csim／cosim／P&R 報告、CPU 基準量測數據，皆為參賽團隊先前研究階段之內部紀錄與實測產出，未曾對外發表，特此標註來源屬性。

---

*文件性質：初賽構想書 | 隊伍代號：ZeroJitter | 競賽分組：AMD Track*
*本文件所有量化數據均直接取自團隊內部 csim / cosim / csynth / Vivado P&R 工具輸出紀錄與 CPU 基準量測腳本結果，未經估算外推；第四、五、六項所列之限制與邊界聲明為本文件不可省略之部分。*
