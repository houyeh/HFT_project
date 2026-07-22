# Week 7: TW replay TCN 的 C++ / Vitis HLS 前置實作

## 1. 本週目標
這週的重點是把專案主線正式切回：

> `TW replay -> TCN -> INT8 -> C++ -> HLS`

也就是說，這週不是再擴充策略實驗，而是把已經在 `TW replay` 上成熟的模型，重寫成真正更接近硬體資料流的版本。

---

## 2. 為什麼這週先做 TW -> HLS
目前的專案狀態是：

- `TW replay` 已有完整資料前處理、feature、訓練、回測與 INT8 匯出
- `CME` 已完成資料接軌與 direct validation，但尚未證明可交易
- `hls4ml` 已實測過，主要卡在 `permute` 與 `causal pad`

因此，若目標是：
- 明天報告能講得清楚
- 下週能繼續穩定推進
- 來得及準備數位晶片設計比賽 HLS 組

最合理的做法就是：

> 先把已成熟的 `TW` 模型寫成 HLS-friendly 的 C++ 主線。

---

## 3. 目前這顆模型的技術規格

### 3.1 Sample 與資料粒度
這顆模型不是用秒 K 或分鐘 K，而是用：

> event-level / tick-level LOB snapshot

每個 sample 由最近 `100` 個事件組成。

### 3.2 Feature set
目前主力模型使用 `execution_aware` 共 `12` 維特徵：

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

### 3.3 TCN 模型結構
目前主模型是 `TCNClassifier`：
- 輸入 shape：`(Batch, 100, 12)`
- 類別數：`3`（`Down / Stationary / Up`）
- channels：`[16, 16, 16]`
- kernel size：`3`
- dilation：`1, 2, 4`
- dropout：`0.1`
- receptive field：約 `29 events`

### 3.4 模型大小與硬體意義
- 參數量：`4,771`
- 每樣本 MAC：`460,848`
- FP32 權重：`19,084 bytes`
- INT8 權重：`4,771 bytes`，約 `4.8 KB`

這代表：
- 模型夠小
- 權重可望完全留在片上記憶體
- 很適合做固定 shape、低延遲的硬體推論實驗

---

## 4. 本週完成的 C++ / HLS-friendly 重構

### 4.1 固定 shape 的推論主線
本週完成的核心檔案：
- [cpp_hls_prep/include/fixed_tcn.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/fixed_tcn.hpp)
- [cpp_hls_prep/src/fixed_tcn.cpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/fixed_tcn.cpp)

重點是把原本較偏 host/runtime 的寫法，改成：
- 固定輸入 shape：`12 x 100`
- 固定 block 數：`3`
- 固定 channels：`16`
- 固定 kernel size：`3`
- 固定大小陣列
- 更明確的 loop 結構

### 4.2 進一步拆成更清楚的 HLS 模組
新增的模組化檔案：
- [cpp_hls_prep/include/fixed_tcn_modules.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/fixed_tcn_modules.hpp)
- [cpp_hls_prep/src/fixed_tcn_modules.cpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/fixed_tcn_modules.cpp)
- [cpp_hls_prep/include/fixed_tcn_top.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/fixed_tcn_top.hpp)
- [cpp_hls_prep/src/fixed_tcn_top.cpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/fixed_tcn_top.cpp)

目前已拆出的核心模組有：
- `causal_conv1d_fixed`
- `pointwise_conv1d_fixed`
- `residual_block_fixed`
- `global_average_pool_fixed`
- `fc_fixed`
- `softmax3_fixed`
- top function：`tcn_fixed_top`

這一步的價值是：

> 之後進 Vitis HLS 時，不再只是單一巨大函式，而是已經有接近硬體模組邊界的結構。

### 4.3 Host-only 解析邏輯與 synthesis 路徑分離
新增：
- [cpp_hls_prep/include/fixed_tcn_host.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/fixed_tcn_host.hpp)
- [cpp_hls_prep/src/fixed_tcn_host.cpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/fixed_tcn_host.cpp)

這代表 `.wts` 載入、runtime parsing 這種 host 邏輯，已經和真正要送去 synthesis 的路徑分開。

---

## 5. 本週完成的固定權重與 testbench

### 5.1 自動匯出固定 header
本週把匯出腳本推進成：
- [scripts/05_hardware_handoff/export_tcn_int8_artifacts.py](/C:/Users/holbo/00VS_projjects/00_HFT_project/scripts/05_hardware_handoff/export_tcn_int8_artifacts.py)

除了原本的 `.wts` 外，現在還會自動生成：
- [cpp_hls_prep/include/generated/fixed_tcn_weights.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/generated/fixed_tcn_weights.hpp)
- [cpp_hls_prep/include/generated/fixed_tcn_test_vectors.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/generated/fixed_tcn_test_vectors.hpp)

這代表：
- 權重已能以固定陣列形式存在
- test vector 與 golden output 已被固化
- HLS testbench 不再依賴 host 端解析 `.wts`

### 5.2 HLS testbench
新增：
- [cpp_hls_prep/src/hls_testbench.cpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/hls_testbench.cpp)

這個 testbench 會直接：
- 餵固定輸入
- 比對 golden int32 logits
- 比對 golden float probabilities

這已經是很標準的 HLS 前置驗證型態。

---

## 6. 驗證結果

### 6.1 Python 端整體測試
- `python -m pytest -q`
- 目前結果：`51 passed`

### 6.2 C++ fixed core 與 dynamic core 比對
本週已用 `tcn_hls_prep.exe --compare-fixed` 驗證：
- fixed core logits 與 dynamic core 一致
- fixed core probabilities 與 dynamic core 一致
- `max probability diff = 0`

### 6.3 HLS testbench
本週已完成：
- `tcn_hls_testbench.exe`
- 結果：`[PASS] HLS testbench matched fixed golden logits/probabilities.`

這代表現在不是只有：
- 能編譯

而是已經做到：
- 有固定輸入
- 有固定 golden output
- 可重現比對結果

---

## 7. Vitis HLS 實際進度

### 7.1 工具鏈與腳本
本週已補好：
- [cpp_hls_prep/hls/run_hls.tcl](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/hls/run_hls.tcl)
- [cpp_hls_prep/hls/run_vitis_hls_2025_2.ps1](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/hls/run_vitis_hls_2025_2.ps1)
- [teaching/22_Vitis_HLS_2025_2_Step_By_Step.md](/C:/Users/holbo/00VS_projjects/00_HFT_project/teaching/22_Vitis_HLS_2025_2_Step_By_Step.md)

### 7.2 已確認的本機環境
- 安裝路徑：`C:\\AMDDesignTools\\2025.2`
- 已確認可用 part：`xc7z020clg400-1`

### 7.3 已做到哪裡
目前已完成：
- `vitis-run --mode hls` 可正常啟動
- `csim` 已實際通過
- `csynth` 已能正式進入 synthesis 流程

也就是說，我們已經從：
- 只是在 Windows 上有安裝工具

往前推到：
- 能真的開始跑 HLS flow

### 7.4 已處理掉的阻礙
這週實際處理過的 HLS 前置問題包括：
- 原本預設 part 沒有安裝
- `.wts` 與 runtime parsing 會干擾 synthesis
- `std::copy` 會引出 HLS 不喜歡的 pointer comparison 問題

這些都已經被拆開或改寫，因此目前流程不再卡在最前面。

---

## 8. 目前的延遲與硬體估計
目前保存的硬體估計在：
- [artifacts/04_hardware_handoff/latency_estimate/latency_estimate.json](/C:/Users/holbo/00VS_projjects/00_HFT_project/artifacts/04_hardware_handoff/latency_estimate/latency_estimate.json)

目前數字為：
- `macs_per_sample = 460,848`
- `params = 4,771`
- `measured_pytorch_cpu_latency p50 = 0.4465 ms`
- `estimated_cpp_int8_latency = 57.606 us`
- `estimated_fpga_latency = 9.217 us`

這些還不是 post-route 結果，但它們的意義很明確：

> 這顆模型的規模已經小到值得進一步做 HLS 與 FPGA 延遲驗證。

---

## 9. 目前還沒完成、下週要做什麼

### 9.1 HLS 主線
接下來最重要的是：
1. 跑完 `csynth`，拿到第一版正式報告
2. 整理 `LUT / FF / BRAM / DSP / latency / II`
3. 開始加 pragma 與 memory partition
4. 規劃 `export_design`

### 9.2 CME 次線
`CME` 會改成：
- `ES` 專屬模型
- `CL` 專屬模型
- `NQ` 專屬模型

但這一條線暫時不再阻塞 HLS 主線。

---

## 10. 本週一句話總結
Week 7 的關鍵成果是：

> 專案已從「Python 版 TCN」正式跨到「固定 shape、固定權重、可進 Vitis HLS 的 C++ TCN 前置版本」，而且 `csim` 已經真的跑起來了。
