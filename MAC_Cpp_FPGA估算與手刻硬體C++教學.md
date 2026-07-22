# MAC / C++ / FPGA 估算與手刻硬體 C++ 教學

適用對象：會基礎 Verilog、剛接觸 ML 與 HLS 的同學  
目標：看懂專案現有 AI 程式，並能自己手刻出可驗證的 C++ 硬體推論核心

---

## 1. 先回答你最關心的：MAC、C++、FPGA 估算怎麼來？

## 1.1 來源程式在哪裡

- 估算腳本：`scripts/04_analysis/estimate_tcn_latency.py`
- MAC 計算邏輯：`src/models/hardware.py` 的 `profile_model_macs()`
- 當前輸出結果：`artifacts/04_hardware_handoff/latency_estimate/latency_estimate.json`

---

## 1.2 MAC 數（460,848）怎麼算

在 `profile_model_macs()` 裡，對 `Conv1d` 和 `Linear` 掛 forward hook，逐層累加 MAC：

- Conv1d：  
  `MAC = batch * out_channels * out_steps * (in_channels/groups) * kernel_size`
- Linear：  
  `MAC = batch * in_features * out_features`

最後把每層 MAC 加總成 `total_macs`，得到：

- `macs_per_sample = 460,848`

重點：這是「數學操作量」，不是實際硬體週期數。

---

## 1.3 C++ INT8 估算 57.6 us 怎麼來

在 `estimate_tcn_latency.py`：

- 假設 C++ 有效吞吐：`8.0e9 MAC/s`
- 公式：`latency_us = macs_per_sample / throughput * 1e6`

代入：

- `460,848 / 8e9 * 1e6 = 57.606 us`

這是工程估算（order-of-magnitude），代表「如果軟體端 int8 實作達到這個有效吞吐，延遲大概在這個量級」。

---

## 1.4 FPGA 估算 9.2 us 怎麼來

同檔案裡假設：

- FPGA 有效吞吐：`50.0e9 MAC/s`
- 同公式：
  `460,848 / 50e9 * 1e6 = 9.21696 us`

重點：這不是 timing report，而是早期投資判斷用的工程預估。

---

## 1.5 這些數字代表什麼意義？大方向可行嗎？

現有數字：

- PyTorch CPU p50：`0.4465 ms`
- PyTorch CPU p99：`1.7499 ms`
- C++ INT8 工程估算：`57.6 us`
- FPGA 工程估算：`9.2 us`

解讀：

1. 量級呈現 `ms -> 十位 us -> 個位/十位 us` 的明顯改善空間。  
2. 模型參數小（`4,771`），INT8 權重小（`4,771 bytes`），硬體化門檻相對低。  
3. 代表「大方向可行且值得做」，但不能當最終承諾數字。

為什麼不能當最終數字？因為還缺：

- post-synthesis timing
- post-route timing
- 板上端到端量測（I/O、DMA、host 開銷）

---

## 2. 手刻 C++ 硬體 code：到底在做什麼？

你可以把手刻 C++ 當作「RTL 前的行為金模（golden model）」，目標是：

- 數學正確（先對）
- 結構可綜合（再快）
- 規格固定（可驗證、可交接）

在本專案對應檔：

- `scripts/05_hardware_handoff/export_tcn_int8_artifacts.py`

它做了三件核心事：

1. 量化權重與輸入（int8 + scale）  
2. 實作固定整數推論路徑（conv/residual/fc）  
3. 產生 HLS 可直接 include 的 header 與 golden test vectors  

---

## 3. 手刻前要建立的「硬體思維」

## 3.1 固定規格思維（先把變動拿掉）

先定死：

- `num_inputs`
- `seq_len`
- `channels`
- `kernel_size`
- `num_classes`
- activation scale

這等價於硬體設計中的 `parameter` 固化，目的是降低控制複雜度與驗證成本。

## 3.2 數值路徑思維（dtype 與尺度）

典型路徑：

- `int8 input` x `int8 weight` -> `int32 accumulator`
- 加上轉尺度後 bias（仍在 int32 域）
- requantize 回 int8（round + saturate）

你要能清楚回答每一層：

- 現在的數值尺度是什麼？
- 何時升位寬？何時降位寬？
- rounding 與 saturate 規則是什麼？

## 3.3 驗證鏈思維（分層比對）

至少三層：

1. FP32 PyTorch reference
2. C++/numpy fixed-int reference
3. HLS/RTL simulation result

若 mismatch，要能定位是：

- 前處理/shape 錯
- 量化尺度錯
- kernel 索引或 padding 錯
- rounding/saturation 規則錯

---

## 4. 手刻 C++ 的實作步驟（可當 checklist）

## Step 1：先凍結模型介面

- 固定輸入 shape（例如 `12 x 100`）
- 固定每層通道數與 dilation
- 固定輸出類別數

## Step 2：定義資料型別與尺度

- `int8_t`：權重/啟動值
- `int32_t`：累加器
- `float`：僅在分析或最後轉概率時使用（硬體路徑可改定點）

## Step 3：先寫「純正確版」核心

- `causal_conv_accumulate`
- `pointwise_conv_accumulate`
- `residual_block`
- `global_average_pool`
- `fully_connected`

先不要急著展開/平行化，先保證和 golden 對齊。

## Step 4：建立外部交付格式

- 權重檔（如 `.wts`）
- manifest（tensor name、shape、scale）
- 固定 header（weights/test vectors）

## Step 5：做 regression

每次改碼都跑：

- 同一組 `reference_input`
- 比對 `logits_int32`, `logits_float`, `probabilities`
- 設 tolerance 與 pass/fail

## Step 6：再做性能優化

- Loop 順序調整（cache locality）
- loop unroll/pipeline（HLS pragma）
- 記憶體分割（array partition）
- 資源共享與吞吐取捨

---

## 5. 你需要的能力地圖（從現在到能獨立完成）

## 5.1 必備數學/ML能力

- 理解 logits / softmax / cross-entropy
- 會看 confusion matrix、macro-F1
- 理解量化尺度與誤差來源

## 5.2 必備系統能力

- 會追 dataflow（CSV -> feature -> tensor -> model -> output）
- 會做 reproducibility（固定 seed、固定輸入向量）
- 會做檔案交接規格（manifest/header）

## 5.3 必備硬體能力

- 位寬與溢位管理
- pipeline/unroll 基本概念
- timing vs resource trade-off
- testbench 驗證與 mismatch 定位

---

## 6. 如何看懂 AI 給你的 C++/HLS 程式（實戰閱讀順序）

建議永遠照這個順序看：

1. **介面**：輸入/輸出 shape 與 dtype  
2. **尺度**：scale 常數、requantize 位置  
3. **核心 loop**：卷積索引是否符合 causal/dilation  
4. **殘差路徑**：downsample 條件與 merge 規則  
5. **驗證輸出**：是否有 golden 比對  

如果 AI 程式沒寫清楚這 5 件事，不要直接信，先補驗證。

---

## 7. 訓練時 PyTorch 到底怎麼 call？（你這個專案）

你目前專案的 call 鏈：

`scripts/02_train_eval/train_tw_baseline.py`
-> `src/training/trainer.py`

---

## 7.1 在 `train_tw_baseline.py` 做了什麼

1. 建資料集與 `DataLoader`
- `train_loader = DataLoader(..., shuffle=True)`
- `val_loader/test_loader = DataLoader(..., shuffle=False)`

2. 建模型
- `model = build_default_model(num_inputs=...)`

3. 建訓練設定
- `TrainerConfig(epochs, batch_size, learning_rate, weight_decay, patience, device)`

4. 呼叫訓練
- `train_with_early_stopping(...)`

5. 推論與評估
- `predict_model(...)`
- `classification_metrics(...)`
- 再進行 gating/backtest/calibration

---

## 7.2 在 `trainer.py` loss / optimizer / epoch 怎麼實作

### loss

```python
criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
```

意義：

- `CrossEntropyLoss`：多分類常用 loss  
- `weight=class_weights`：處理類別不平衡  
- `reduction="none"`：先保留每筆 loss，讓你可以乘 `sample_weight` 再做平均

實際加權：

```python
per_sample_loss = criterion(logits, y)
loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
```

### optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)
```

意義：

- `model.parameters()`：要更新的權重集合  
- `lr`：每步更新幅度  
- `weight_decay`：L2 正則，抑制過擬合

### 每個 batch 的更新語法

```python
optimizer.zero_grad(set_to_none=True)
logits = model(x)
loss.backward()
optimizer.step()
```

意義：

- 清空舊梯度 -> forward -> backward 求梯度 -> step 更新參數

### epoch

```python
for epoch in range(1, config.epochs + 1):
    train_loss = _run_train_epoch(...)
    val_metrics = evaluate_model(...)
```

意義：

- 一個 epoch = 把整份 train data 跑過一次
- 每個 epoch 後都在 val set 檢查泛化表現

### early stopping

邏輯：

- 若 `val_loss` 下降 -> 存最佳權重  
- 連續 `patience` 次沒改善 -> 提前停止  

意義：

- 減少過擬合
- 省下無效訓練時間

---

## 8. 量化像 ADC 嗎？會有 quantization error 嗎？

簡短答案：

- **有點像，但不完全相同**
- **一定會有 quantization error**

像的地方：

- 都是在做「連續值到離散格點」映射。

不同的地方：

- ADC 是類比電壓的硬體取樣/量化流程。
- 本專案量化是數位域中，將 FP32 重新編碼為 int8+scale 的數學流程。

誤差來源：

1. rounding  
2. clipping/saturation  
3. 多層 repeated requantize 累積

控制方法：

- 合理 scale 設計
- 固定 rounding/saturation 規則
- 分層 golden 比對（FP32 vs INT8 vs HLS/RTL）

---

## 9. 給你的實作路線（可以直接照做）

## Week 1：看懂與重現

1. 跑 `estimate_tcn_latency.py`，確認數字可重現  
2. 跑 `export_tcn_int8_artifacts.py`，確認輸出 artifact  
3. 手算一個小 conv 範例，對照 `bias_to_accumulator` / `requantize`

## Week 2：做自己的最小核心

1. 手刻 1-block int8 conv+relu+requantize C++  
2. 讀 `.wts` 或 header 載入固定權重  
3. 寫單元測試比對 golden

## Week 3：往 HLS 推進

1. 加 HLS pragma（pipeline/unroll）  
2. 看資源與延遲報告  
3. 找一組可接受的 latency/resource 平衡點

---

## 10. 最後總結（你該記住的 4 件事）

1. MAC 與 latency 估算是「工程早期決策工具」，不是最終 tapeout 數字。  
2. 手刻 C++ 的核心是「固定規格 + 正確尺度 + 可驗證鏈」。  
3. PyTorch 訓練邏輯在你專案已很標準：CrossEntropy + Adam + epoch + early stopping。  
4. 量化一定有誤差，但可透過規範化流程與 golden 驗證把風險控制住。  
