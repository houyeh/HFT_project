# 給老師的 HLS 進度摘要

## 專案目標

本專案研究：事件層級（event-level）的 HFT 預測管線，是否能從研究用程式碼移植到 FPGA 導向的流程。
目前主線如下：

`TW replay / CME adapter -> features -> TCN -> gating / backtest -> frozen model -> INT8 export -> fixed-shape C++ -> Vitis HLS`

## 截至 2026-04-09 的現況

### 1. 軟體端研究管線已完成

- 台指 replay 資料已可解析成正規化的限價委託簿（LOB）資料結構。
- 特徵管線同時包含基礎 order-book 特徵與 execution-aware 特徵。
- TCN 訓練、驗證、機率 gating、事件驅動回測、以及 hybrid market-maker 回測皆已實作完成。
- 專案已可在 `artifacts/` 產生可重複使用的產出物，用於驗證、延遲分析與硬體交付。

### 2. 硬體交付路徑在功能面已完成

- 已有凍結（frozen）的勝出模型：`artifacts/03_latency_proof/latency_proof/frozen_tcn_best_model.pt`。
- INT8 匯出已在 `scripts/05_hardware_handoff/export_tcn_int8_artifacts.py` 實作完成。
- 匯出內容包含：
  - `tcn_int8_weights.wts`
  - `reference_input.txt`
  - `reference_output.txt`
  - 產生的 HLS headers：`cpp_hls_prep/include/generated/`
- 目前有兩條 C++ 路徑：
  - 動態 INT8 參考推論（reference inference）
  - 固定形狀、HLS 友善的核心（fixed-shape core）
- 已具備 bit-accurate 對齊檢查與 testbench 基礎設施。

### 3. repo 內已具備可驗證的證據

- `cpp_hls_prep/hls/project/fixed_tcn_hls/solution1/csim/report/tcn_fixed_top_csim.log`
  - C simulation 已於 2026-04-09 通過。
- `cpp_hls_prep/hls/project/fixed_tcn_hls/solution1/syn/report/tcn_fixed_top_csynth.rpt`
  - 針對 `xc7z020-clg400-1` 的 C synthesis 已於 2026-04-09 完成。
- `artifacts/04_hardware_handoff/latency_estimate/latency_estimate.json`
  - parameters: `4,771`
  - MACs per sample: `460,848`
  - INT8 weight footprint: `4,771 bytes`
  - estimated C++ INT8 latency: `57.606 us`
  - estimated FPGA latency: `9.21696 us`

### 4. 目前 HLS 結果的意義

目前的 HLS 流程已經超過「只有軟體模擬」的階段。
它已經走到：

1. 凍結模型（frozen model）選擇
2. INT8 交付產出物（handoff artifacts）匯出
3. 固定形狀的 C++ 改寫（fixed-shape rewrite）
4. HLS testbench 生成
5. Vitis HLS `csim`
6. Vitis HLS `csynth`

這代表整條功能鏈（functional chain）已經建立完成。
目前的瓶頸不是「能不能合成」，而是「如何把資源用量與延遲優化到足以符合目標器件」。

### 5. 綜合 synthesis report 的現況

以目前目標 `xc7z020-clg400-1` 來看，綜合後的設計尚未達到可部署（deployment-ready）：

- estimated clock: `6.978 ns` vs target `5.00 ns`
- latency: about `133,612` cycles, about `0.932 ms`
- BRAM utilization: `9%`
- DSP utilization: `194%`
- FF utilization: `105%`
- LUT utilization: `410%`

解讀：

- HLS 管線在功能面是真實可行的
- 目前 microarchitecture 對這顆 FPGA 來說過大
- 下一階段應聚焦在 pruning、重用（reuse）、記憶體分割策略（memory partition），以及／或改用更大的器件目標

### 6. Transfer-learning 現況

repo 內也包含一條 CME 的直接驗證路徑，可用於展示學術進展：

- `artifacts/02_model_runs/cme_eval_transfer/20260407_161938/summary.json`
- gated accuracy: `0.8202`
- gated macro F1: `0.3868`
- gated signal net pnl: `-36,316.5 ticks`

解讀：

- 軟體管線可功能性地轉移到 CME 資料
- 分類表現仍顯著優於隨機
- 轉移後的模型若沒有做 symbol-specific calibration，仍未達到交易可用（trading-ready）

## 建議的一分鐘老師版摘要

本專案已完成從軟體到 HLS 的端到端流程。
模型已完成訓練、凍結、INT8 量化，並匯出成固定形狀的 C++，再包裝進 Vitis HLS，且已由 HLS 的 C simulation 驗證通過。
目前的 synthesis report 證明設計可被合成，但也顯示現階段的架構尚未符合既定的 FPGA 目標，因此下一階段工作重點會是硬體端的資源與延遲優化，而不是再建管線。
