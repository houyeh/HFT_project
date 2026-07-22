# 21. Hardware Handoff、hls4ml 與延遲估算（Verilog 初學者版）

## 這章的目標

你現在會基本 Verilog，但剛接觸 AI 模型硬體化。  
這章希望你看完可以回答：

1. **什麼是 hardware handoff**，為什麼不是「把 `.pt` 丟進 FPGA」就結束？
2. **hls4ml 在做什麼**，為什麼有時會卡住？
3. 我們這個專案從 **PyTorch -> INT8 -> C++/HLS** 的每一環到底在做什麼？
4. 還沒上板時，如何用工程方法先估延遲與風險？

---

## 0. 先建立心智模型：你熟悉的 Verilog vs 這個流程

如果你平常寫 Verilog，會習慣：

- 先定義介面（input/output）
- 決定資料寬度（bit-width）
- 寫資料路徑（乘法器、加法樹、FSM、pipeline）
- 寫 testbench 比對 golden

AI 模型硬體化其實做同一件事，只是多了兩個前置工作：

- **把訓練出的浮點參數轉成固定點/INT8**
- **確認軟體圖（graph）可以被硬體友善地展開**

你可以把這件事想成：

> PyTorch 模型 = 很高階的演算法規格  
> HLS/C++/Verilog = 要把規格落地成可時序收斂、可驗證的資料路徑

---

## 1. 什麼是 Hardware Handoff

Hardware handoff = 把「可訓練模型」交接成「可硬體驗證資產」。

在本專案，交接資產包含：

- 權重檔（給 C++/HLS 載入）
- 量化比例（scale）
- 參考輸入（test vector）
- 參考輸出（golden output）
- 固定 shape 的 header（讓 HLS testbench 直接 include）

對應腳本：
- `scripts/05_hardware_handoff/export_tcn_int8_artifacts.py`

主要輸出：
- `artifacts/04_hardware_handoff/int8_export/tcn_int8_weights.wts`
- `artifacts/04_hardware_handoff/int8_export/tensor_manifest.json`
- `artifacts/04_hardware_handoff/int8_export/reference_input.txt`
- `artifacts/04_hardware_handoff/int8_export/reference_output.txt`
- `cpp_hls_prep/include/generated/fixed_tcn_weights.hpp`
- `cpp_hls_prep/include/generated/fixed_tcn_test_vectors.hpp`

---

## 2. hls4ml 是什麼？它幫你做了哪些事？

`hls4ml` 的核心價值：  
把常見神經網路層（Dense/Conv/Activation）從訓練框架轉成 HLS C++。

你可以把它想成「神經網路版高階語法編譯器」，自動幫你：

- 做 layer graph 轉換
- 插入固定點型別
- 產生可綜合的 HLS 程式骨架
- 產生一組基本測試流程

### 2.1 為什麼不是所有模型都能一鍵轉？

因為它只支援「某些圖操作組合」。  
當模型包含某些 operation（例如特定 `permute/pad` 組合）時，可能無法直接映射到它支援的 HLS layer template。

這就像：

- Verilog 你可以寫任意行為，但綜合器不一定都吃得下或能有效實現
- hls4ml 也有自己的「可轉換子集合」

---

## 3. 本專案 hls4ml 卡點與含義

對應腳本：
- `scripts/05_hardware_handoff/attempt_hls4ml_conversion.py`

目前觀察：
- 原始模型卡在 `permute`
- channel-first 版本卡在 `pad`

這代表：

- 問題不是模型太大
- 問題是 **圖結構不夠 hls4ml-friendly**
- 尤其是 TCN 的 causal padding，在工具鏈裡常是高風險點

所以專案策略是：

1. 保留 hls4ml 嘗試（工具路線）
2. 主線先走手刻 fixed-shape C++（可控路線）

這是很工程化的決策：先確保可交付，再追求自動化。

---

## 4. `export_tcn_int8_artifacts.py` 在做什麼（按邏輯分層）

### 4.1 固定規格定義

腳本先定義固定常數（input channels、seq len、classes、kernel size）。  
這相當於硬體設計時先定 `parameter`：

- 介面寬度固定
- buffer 大小固定
- 陣列 shape 固定

好處：
- 容易做時序與資源估算
- 減少動態控制複雜度
- testbench 更穩定

### 4.2 量化函式

核心觀念：把浮點權重轉成 `int8 + scale`。

- `quantize_tensor_symmetric()`：對稱量化
- `bias_to_accumulator()`：把 bias 對齊到 accumulator 尺度
- `requantize_to_fixed_scale()`：將中間 `int32` 累加結果縮回固定 activation scale

這三步的「目的與程式實現」可這樣對照：

1. `quantize_tensor_symmetric()`  
   - 目的：把 FP32 權重壓到 `[-127,127]`，降低記憶體/帶寬與運算成本。  
   - 寫法：先找 `max_abs`，算 `scale=max_abs/127`，再做 `round(x/scale)` 並 clamp 成 int8。  
2. `bias_to_accumulator()`  
   - 目的：卷積乘加在 int32 累加器中進行，bias 必須換到同一尺度才能直接相加。  
   - 寫法：乘上 `bias_scale/(input_scale*weight_scale)` 後四捨五入成 int32。  
3. `requantize_to_fixed_scale()`  
   - 目的：把 int32 中間結果重新壓回固定 activation 尺度，讓下一層能用同一規格接續。  
   - 寫法：乘 `acc_scale/FIXED_ACT_SCALE`，四捨五入後再做 int8 飽和截斷。

> 直觀上有點像 ADC 都在做「連續值 -> 離散值」映射，  
> 但本專案量化是數位域的數學重編碼，不是類比電壓取樣電路本身。

### 4.2.1 量化誤差會不會有？

會，而且一定有。來源包含：

- rounding（四捨五入）
- clipping/saturation（超出範圍被夾住）
- 每層重複 requantize 累積的誤差

控制方法：

- 選合理的 scale 與位寬
- 用 `reference_output` + `fixed_reference` 做分層比對
- 看最終任務指標（分類 + 回測）是否仍在可接受範圍

Verilog 對應思維：
- 乘加在較寬位元（accumulator）做
- 層與層之間做縮放與截位（含飽和）

### 4.3 固定整數推論核心

腳本裡有純 numpy 的整數版推論：

- `causal_conv_accumulate()`
- `pointwise_conv_accumulate()`
- `residual_block()`
- `global_average_pool()`
- `fully_connected()`

這一段很重要，因為它是「硬體行為的軟體鏡像」。  
也就是你在 HLS/RTL 實作前，就先用同一套整數數學把 golden 跑出來。

### 4.4 產生固定 header

- `write_fixed_weights_header()`：把權重和 scale 寫成 C++ 常數結構
- `write_fixed_test_vectors_header()`：寫入固定輸入與 golden logits/probabilities

這在硬體驗證上等價於：
- 不依賴外部檔案 parser
- testbench 可直接 compile 後跑
- debug 可重現

---

## 5. 為什麼要同時保留「FP32 reference」與「INT8 fixed reference」？

`reference_output.txt`（FP32）與 `fixed_reference`（INT8）各有用途：

- **FP32 reference**：確認模型語意（訓練框架的真值）
- **INT8 fixed reference**：確認硬體數學路徑（量化後真值）

在硬體 bring-up 時，常用三層比對：

1. PyTorch FP32
2. C++/numpy INT8 fixed
3. HLS C sim / RTL sim

如果第 1 層就錯，通常是模型或輸入處理問題。  
如果第 1 層對、第 2 層錯，通常是量化或縮放邏輯。  
如果前兩層對、第 3 層錯，通常是硬體實作或介面接線問題。

---

## 6. 模型大小與運算量（如何轉成硬體語言）

參考：
- `artifacts/04_hardware_handoff/latency_estimate/latency_estimate.json`

目前模型：
- 參數量：`4,771`
- MACs / sample：`460,848`
- FP32 權重大小：`19,084 bytes`
- INT8 權重大小：`4,771 bytes`

硬體解讀：

### 6.1 參數量小
- 權重有機會全部放片上（BRAM/URAM）
- 可避免大量外部記憶體存取

### 6.2 MAC 數中等
- 決定理論計算量
- 可以估 pipeline 需要多少 DSP 或時脈週期

### 6.3 INT8 權重小
- 帶寬壓力大幅下降
- 記憶體子系統更簡化

---

## 7. 延遲估算要怎麼看（避免過度樂觀）

目前數字：
- PyTorch CPU p50：`0.4465 ms`
- PyTorch CPU p99：`1.7499 ms`
- 工程估算 C++ INT8：`57.6 us`
- 工程估算 FPGA：`9.2 us`

這些不是最終答案，因為還缺：

- post-synthesis timing
- post-route timing
- 真正板上量測（I/O、DMA、host 開銷）

但它有價值：  
可以先判斷這條硬體路線「大方向」值不值得投資。

### 7.1 這些數字怎麼決定出來？

- PyTorch CPU p50 / p99：  
  由 CPU 推論 benchmark 重複量測後取分位數。  
  - `p50` = 中位延遲（一般情況）  
  - `p99` = 尾延遲（最壞 1% 情況）
- C++ INT8 工程估算：  
  由 MAC 數、資料型別（int8/int32）、預期實作方式推估的軟體側低延遲上限級別。
- FPGA 工程估算：  
  由固定資料流架構、可 pipeline 程度與目標時脈做初估。

### 7.2 代表什麼意義？

- 表示「同一模型語意」在不同實作層的潛在延遲級距：
  - CPU：毫秒級
  - C++ INT8：數十微秒級
  - FPGA：單位到十位微秒級（目標）
- 這不是簽核數字，而是投資決策用的早期訊號。

### 7.3 大方向可不可以？

以目前數字看，大方向是 **可以且值得做**，原因是：

1. 延遲級距有明顯改善空間（ms -> us）  
2. 模型小（4,771 params）且 INT8 權重體積低，硬體落地風險可控  
3. 已建立 golden 驗證鏈，後續除錯成本可管理  

但最終 go/no-go 仍要等：
- post-synth / post-route timing
- 板上端到端延遲（含 I/O、DMA、host）

---

## 8. 從 Verilog 學生角度看每個環節該學什麼

### 8.1 你現在就能掌握的關鍵

1. **資料格式一致性**
   - 輸入 shape、channel 排列、時間軸方向一定要固定
2. **量化一致性**
   - scale、rounding、saturation 規則一定要一致
3. **golden 驗證鏈**
   - 每層都能比對，不要只看最終分類結果

### 8.2 下一步能力地圖

1. 先看懂 C++ 固定點推論核心（等價行為）
2. 再看 HLS pragma 如何改 latency/throughput
3. 最後才進 RTL 細節優化（資源共享、時序收斂）

---

## 9. 這個專案目前最合理的主線

1. 保留 hls4ml 嘗試（持續追蹤工具支援）
2. 主線採手刻 fixed-shape C++ core（可控且可驗證）
3. 用固定權重與固定測試向量推進 HLS testbench
4. 拿到穩定結果後再做 kernel 化與上板量測

這條路線的優點：
- 進度可控
- 風險可分段隔離
- 每一段都能有可觀測的驗證成果

---

## 10. 一頁版總結

如果你只記住三件事：

1. **硬體化不只是在意模型準確率，還要在意資料流、位寬、可驗證性。**
2. **hls4ml 很強，但它不是萬能；圖結構不友善時，要準備手刻路線。**
3. **先建立 FP32 -> INT8 -> HLS/RTL 的 golden 鏈，才有辦法快速 debug 和收斂。**

---

## 11. ML 新手補充：這個專案會用到的機器學習觀念（由淺入深）

這段是給第一次接觸 ML 的同學。先不用背公式，先懂「每一步在解決什麼問題」。

### 11.1 什麼是模型？（把它當成可學習的函數）

在這個專案中，模型吃進一段時間窗內的市場特徵（例如 100 個時間點，每點 12 維特徵），輸出三類機率：

- Down（下跌）
- Stationary（盤整）
- Up（上漲）

你可以把它當成：

`輸入特徵序列 -> 分類分數(logits) -> 機率(probabilities)`

### 11.2 什麼是特徵（Feature）

特徵就是把原始市場資料轉成模型可處理的數值欄位。  
例如本專案會用到：

- spread/obi（買賣盤結構）
- micro_return（微小報酬）
- queue imbalance（排隊不平衡）
- trade flow/cancel pressure（成交與撤單壓力）

重點不是「特徵越多越好」，而是「特徵是否有訊號，且可在部署時穩定計算」。

### 11.3 什麼是標籤（Label）

訓練時需要「正確答案」，本專案用未來一段 horizon 的價格變化做三分類標籤。  
概念上：

- 未來變化 > 門檻 -> Up
- 未來變化 < -門檻 -> Down
- 介於中間 -> Stationary

這裡的門檻設計，會直接影響：

- 類別分佈是否平衡
- 訊號是否容易交易

### 11.4 為什麼要 train/val/test 切分

- **train**：讓模型學習
- **val**：調參和 early stopping
- **test**：最後才看，模擬未知資料

時間序列任務特別要注意：  
切分必須遵守時間先後，避免未來資訊洩漏（look-ahead bias）。

### 11.5 loss、優化器、epoch 在做什麼

- **loss**：衡量「模型答案離正確答案多遠」
- **optimizer**：根據 loss 調整權重（例如 Adam）
- **epoch**：整份訓練資料跑完一次

early stopping 的想法是：  
如果 val 表現不再進步，就停下來，避免過擬合。

### 11.5.1 在本專案程式中是怎麼做的

對應檔案：`src/training/trainer.py`

- `loss`：`nn.CrossEntropyLoss(weight=class_weights, reduction="none")`  
  - 先得到每筆樣本的 loss，再乘 `sample_weight` 做加權平均。  
- `optimizer`：`torch.optim.Adam(model.parameters(), lr=..., weight_decay=...)`  
  - 每個 batch 執行 `zero_grad -> forward -> backward -> step`。  
- `epoch`：`for epoch in range(1, config.epochs + 1):`  
  - 每圈呼叫 `_run_train_epoch(...)` 跑完整個 train loader。  
- `early stopping`：  
  - 若 `val_loss` 連續 `patience` 次沒有改善就停止，並回復最佳權重。  

### 11.6 指標怎麼看（不要只看 accuracy）

三分類常用：

- accuracy：整體答對率
- macro-F1：對每一類都公平看待（不被大類別壓過）
- confusion matrix：看錯在什麼類別之間

在交易問題中，分類指標只是第一層，還要看回測績效（PnL、類 Sharpe）。

### 11.7 為什麼「分類準」不等於「可交易」

因為交易會碰到：

- spread 成本
- 滑價/成交不確定
- 過度交易造成成本吞噬

所以專案會再做：

- probability gating（只做高信心單）
- signal backtest + event-driven backtest（檢查可交易性）

### 11.8 量化（Quantization）是什麼

ML 模型原本通常是 FP32。  
硬體上常改成 INT8，目的是：

- 降低記憶體與頻寬
- 提升吞吐與延遲表現
- 更接近 FPGA/ASIC 的實作習慣

但量化不是「壓縮檔案」而已，而是改變數值表示方式，所以一定要做 golden 驗證。

### 11.9 為什麼這個專案選 TCN

TCN（時間卷積網路）對序列資料友善，且結構規律，通常比 RNN 更容易做平行化與硬體 pipeline。  
對硬體端來說，規律結構通常比較好做資源規劃。

---

## 12. 附錄：1D Conv 的 INT8 手算範例（對照程式與硬體）

這個小範例對照 `export_tcn_int8_artifacts.py` 的想法，手算一次：

- `int8 x int8` 乘加
- `int32` accumulator
- requantize 回 `int8`

### 12.1 設定

假設：

- input（一個 channel，長度 5）：`x = [10, -4, 7, 3, -2]`（int8）
- kernel（長度 3）：`w = [2, -1, 3]`（int8）
- bias（已轉成 accumulator 尺度）：`b_acc = 5`（int32）
- dilation = 1，causal conv
- `acc_scale = 1/256`
- `FIXED_ACT_SCALE = 1/64`

### 12.2 在時間點 `t=2` 的 accumulator

causal 的索引對齊後，可用到的輸入是 `x[0], x[1], x[2]`：

- 乘加：`10*2 + (-4)*(-1) + 7*3`
- 乘加：`20 + 4 + 21 = 45`
- 加 bias：`acc = 45 + 5 = 50`（int32）

### 12.3 requantize 回固定 activation scale

比值：

`ratio = acc_scale / FIXED_ACT_SCALE = (1/256) / (1/64) = 1/4`

所以：

- `q = round(acc * ratio) = round(50 * 1/4) = round(12.5) = 13`
- clamp 到 `[-127,127]` 仍是 `13`

最後該時間點輸出是 `13`（int8）。

### 12.4 這對硬體設計代表什麼

你在 RTL/HLS 端至少要保證四件事一致：

1. 乘加順序和索引規則（causal + dilation）
2. bias 尺度轉換方式
3. rounding 規則（四捨五入細節）
4. saturation/clamp 邊界

只要其中一個不同，就會看到 C++/RTL 與 golden 不一致。

### 12.5 對照本專案程式

對照關係如下：

- accumulator 乘加：`causal_conv_accumulate()`
- bias 對齊：`bias_to_accumulator()`
- 回縮放：`requantize_to_fixed_scale()`
- 最終夾位：`clamp_int8_array()`

這就是「軟體端先做硬體等價數學」的核心價值：  
先把數學定死，再進入 HLS/RTL，可以大幅減少 debug 時間。
