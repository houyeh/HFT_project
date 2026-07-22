# Week 7 Vendor Brief

## 1. 專案目前狀態
- 我們已完成台股 replay 與 CME Logic20 的資料接軌
- 已有可重現的 TCN baseline、event-driven backtest、hybrid MM baseline
- 已做過 INT8 PTQ、MAC/參數量分析
- 已實際嘗試 `hls4ml`

## 2. 目前模型摘要

### 架構：TCN classifier
目前主模型是 `Temporal Convolutional Network (TCN)` 分類器，用來做超短期方向預測。

它的核心結構是：
- 3 個一維時間卷積 block
- causal convolution
- dilation = `1, 2, 4`
- 每層 channel 數為 `16`
- 最後接 global average pooling 與一層全連接分類頭

這顆模型的功能不是做長期趨勢預測，而是：

> 根據最近一小段 order-book 與成交流的變化，判斷未來極短期價格更可能往下、橫盤、或往上。

對硬體來說，它的優點是：
- 結構固定
- 輸入 shape 固定
- 沒有 attention 或可變長路徑
- 比 LSTM / Transformer 更容易做 pipeline 化

目前卡住 `hls4ml` 的主要原因不是模型太大，而是 PyTorch graph 中的：
- `permute`
- `causal pad`

### 輸入 window：`100`
這裡的 `100` 不是 100 秒，也不是 100 根 K 棒，而是：

> 最近 `100` 個 event-level snapshot

也就是說，模型一次看的不是固定時間長度，而是最近 100 筆委託簿事件。

這樣設計的原因是：
- 高頻資料本來就是 event-driven
- 固定事件窗比固定秒數更能反映 microstructure 的即時變化
- 對硬體端也比較好做固定長度 buffer

在實作上，這代表前端只要維持一個固定大小的 ring buffer，就可以持續把最新事件推進模型。

### 特徵數：`12`
目前主力使用的是 `execution_aware` feature set，共 12 維。

#### 12 個特徵
1. `normalized_spread`
2. `micro_return`
3. `obi`
4. `weighted_mid_deviation`
5. `deep_obi`
6. `queue_imbalance`
7. `queue_imbalance_change`
8. `aggressor_imbalance_interval`
9. `rolling_signed_flow_ratio`
10. `trade_intensity_short`
11. `net_cancel_pressure`
12. `cancel_pressure_short`

#### 這些特徵怎麼來
特徵來源可分成三類：

1. 價格/價差特徵
   - 例如 `normalized_spread`, `micro_return`
   - 用來描述當下價格結構與短期變動

2. 委託簿不平衡特徵
   - 例如 `obi`, `deep_obi`, `queue_imbalance`
   - 用來描述哪一邊掛單壓力比較大

3. 成交與撤單流特徵
   - 例如 `aggressor_imbalance_interval`, `trade_intensity_short`, `net_cancel_pressure`
   - 用來描述是否有主動買賣壓力、毒性流或流動性抽離

#### 這些特徵是怎麼訓練的
目前訓練方式是：
- 把每個 event-level snapshot 先算成一列 12 維 feature
- 再用 sliding window 組成 `(100, 12)` 的樣本
- label 定義為未來 `10` 個事件後的價格方向
- threshold 設定為 `1 tick`
  - 上漲超過 1 tick -> `Up`
  - 下跌超過 1 tick -> `Down`
  - 否則 -> `Stationary`

#### 為什麼採用這 12 個特徵
原因有三個：

1. 它們大多與市場微觀結構直接相關
   - 而不是純技術指標

2. 它們大多可在 streaming 條件下即時計算
   - 這點對未來 FPGA 非常重要

3. 它們涵蓋了三種關鍵資訊
   - price/spread
   - book imbalance
   - trade/cancel flow

也就是說，這 12 維不是亂挑的，而是刻意組成一個對 HFT execution 問題比較實用、又相對容易硬體化的 feature 集合。

### 輸出類別：`3`（Down / Stationary / Up）
模型輸出是三分類：
- `Down`
- `Stationary`
- `Up`

它不是直接預測報酬值，而是預測方向類別。

這樣做的原因是：
- 對資料噪音比較穩
- 對 early-stage baseline 比較容易訓練
- 後面可以再用 probability gating 把分類結果轉成交易信號

其中 `Stationary` 很重要，因為它讓模型可以學會：

> 大部分時候不要亂交易

### 參數量：`4,771`
這顆模型只有 `4,771` 個參數。

這個數字的意義不是「越小越好」，而是：
- 它已經小到很適合作為 FPGA 候選模型
- 權重可以輕鬆放進 on-chip memory
- 不會有大型模型常見的頻寬壓力

對研究來說，它代表：
- 我們不是想拿一顆很大的神經網路硬塞進 FPGA
- 而是用一顆刻意簡化、面向低延遲的模型，測試硬體是否真的能提供價值

### 每樣本 MAC：`460,848`
這個數字表示：

> 每做一次單樣本推論，大約需要 460,848 次乘加運算

這是非常重要的硬體指標，因為它直接影響：
- 理論吞吐
- 延遲下界
- pipeline 規模
- reuse / unroll 策略

目前這個量級不算大，代表：
- 用 C++ INT8 優化有機會進入 `數十 us`
- 用 FPGA pipeline 化後，有機會更進一步壓到 `個位數到十幾 us`

### INT8 權重大小：約 `4.8 KB`
目前這顆模型做完 INT8 後，權重大小約為 `4,771 bytes`，也就是大約 `4.8 KB`。

它的意義非常直接：

1. 權重很小
   - 幾乎可以完全留在片上記憶體

2. 記憶體頻寬壓力低
   - 不需要每次推論都依賴外部 DRAM 搬大量參數

3. 很適合做 deterministic low-latency inference
   - 因為權重讀取不會成為主要瓶頸

這也是目前我們認為這顆模型有 FPGA 研究價值的核心原因之一：

> 它不是只有「能跑」，而是從模型規模上看，真的有機會被做成低延遲硬體路徑。

## 3. 目前硬體轉換觀察
- 直接丟 `hls4ml`
  - 卡在 `permute`
- 改成 channel-first wrapper 後
  - 卡在 `pad`

這表示目前的主要障礙不是權重大小，而是：
- causal padding 的實作方式
- 前處理 / layout 轉換是否要搬到硬體前或硬體內

## 4. 想請廠商協助確認的問題
1. 對於這種 `causal Conv1D + dilation` 結構，建議：
   - 直接手寫 HLS kernel
   - 還是先改寫成 tool-friendly 結構再丟工具？
2. `permute + pad + global average pooling` 最適合切在哪一層處理？
3. 如果輸入固定為 `12 x 100`，是否建議：
   - full unroll
   - partial unroll
   - line buffer / circular buffer
4. 對於目前約 `4.8 KB` 的 INT8 權重，BRAM / URAM 配置有沒有明顯建議？
5. 如果目標是低延遲而不是高吞吐，reuse factor 應如何優先設定？

## 5. 可交付給廠商的檔案
- frozen model：
  - `artifacts/03_latency_proof/latency_proof/frozen_tcn_best_model.pt`
- INT8 權重：
  - `artifacts/04_hardware_handoff/int8_export/tcn_int8_weights.wts`
- tensor manifest：
  - `artifacts/04_hardware_handoff/int8_export/tensor_manifest.json`
- `hls4ml` 嘗試報告：
  - `artifacts/04_hardware_handoff/hls4ml_attempt/attempt_report.json`
- latency estimate：
  - `artifacts/04_hardware_handoff/latency_estimate/latency_estimate.json`
