# Updated Schedule

## Current Strategy
專案目前分成兩條清楚的主次線：

1. `主線：TW replay -> TCN -> INT8 -> C++ -> Vitis HLS`
2. `次線：CME Logic20 adapter -> direct validation -> 分商品模型`

這樣安排的原因是：
- `TW replay` 已有完整且可運作的 end-to-end pipeline
- `CME` 已完成資料接軌，但 direct validation 成效不好
- 若讓 `CME` 支線繼續擴散，會拖慢 HLS 主線與比賽準備時程

---

## Week 4 to Week 5
### 已完成
- TW replay baseline
- event-driven backtest
- execution-aware features
- hybrid market making baseline
- latency proof harness

### 代表性結果
- directional realistic baseline 最接近：
  - `val-calibrated test event pnl = -3 ticks`
- hybrid MM 顯示明顯 latency sensitivity：
  - `delay=0 -> -19`
  - `delay=1 -> +7`
  - `delay=2 -> -41`
  - `delay=4 -> -84`

---

## Week 6
### 已完成
- CME data 解壓與整理
- CME Logic20 adapter
- frozen TW model 直接丟 CME 驗證
- 專案口徑收斂為：
  - 資料已接上
  - 直接驗證已完成
  - 效果不好
  - 下一步改做分商品模型

### Week 6 checkpoint
- `CME direct validation`：
  - `gated accuracy = 0.8202`
  - `gated macro_f1 = 0.3868`
  - `gated signal net pnl = -36316.5 ticks`
- 結論：
  - pipeline 是通的
  - 但目前不能宣稱 `TW trained model` 可直接通用到 `CME`

---

## Week 7
### 已完成
- 將 TW TCN 重寫成 fixed-shape C++ core
- 拆出更清楚的 HLS 模組：
  - `conv1d`
  - `residual block`
  - `global average pooling`
  - `fc`
  - `top`
- 生成固定權重與固定測試向量 header
- 建立 HLS testbench
- 實際跑通 Vitis HLS `csim`

### Week 7 checkpoint
- fixed core 與 dynamic core：
  - `max probability diff = 0`
- `tcn_hls_testbench.exe`：
  - `PASS`
- `Vitis HLS`：
  - `csim pass`
  - `csynth` 已可進入正式 synthesis

### Deliverable
- fixed-shape TCN C++ core
- generated fixed-weight headers
- HLS testbench
- Vitis HLS step-by-step 教學文件

---

## Next Week
### CME 線
- 開始訓練 `ES` 與 `CL` 各自模型
- `NQ` 保留在同一流程內，資料一到即可接上
- 每個商品各自：
  - 訓練
  - calibration
  - 評估

### HLS 線
- 跑完 `csynth`
- 蒐集 `LUT / FF / BRAM / DSP / latency / II`
- 加入 pragma 與 memory partition
- 準備 `export_design`

---

## Current Decision
現在最重要的決策不是「要不要繼續做 CME」，而是：

- `TW replay` 先把硬體主線做完整
- `CME` 當作下一階段的分商品模型與外部驗證

這是目前最能同時兼顧：
- 明天報告清楚
- 下週工作聚焦
- 趕上數位晶片設計比賽 HLS 組時程
