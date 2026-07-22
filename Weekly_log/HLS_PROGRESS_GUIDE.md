# HLS 進度導覽

## 如果你想要快速說明這個專案

建議用以下順序：

1. 閱讀 `HLS_PROGRESS_FOR_TEACHER.md`
2. 開啟 `scripts/02_train_eval/train_tw_baseline.py`
3. 開啟 `scripts/05_hardware_handoff/export_tcn_int8_artifacts.py`
4. 開啟 `cpp_hls_prep/src/fixed_tcn.cpp`
5. 開啟 `cpp_hls_prep/hls/project/fixed_tcn_hls/solution1/csim/report/tcn_fixed_top_csim.log`
6. 開啟 `cpp_hls_prep/hls/project/fixed_tcn_hls/solution1/syn/report/tcn_fixed_top_csynth.rpt`

這一串就能把故事從模型訓練一路講到 HLS 合成。

## 你已經完成的工作

### 軟體端

- 完成台指 replay 與 CME 資料 adapter
- 建立基礎特徵與 execution-aware 特徵
- 訓練 TCN 分類器
- 加入機率 gating
- 實作 signal backtest 與 event-driven backtest
- 建立 queue-aware 的 hybrid market-maker replay

### 硬體端

- 選擇並凍結（freeze）模型 checkpoint
- 匯出 INT8 weights 與 reference vectors
- 撰寫固定形狀的 C++ TCN cort5rt5e
- 生成 HLS 測試向量與 headers
- 執行 Vitis HLS `csim`
- 執行 Vitis HLS `csynth`

## 各主要資料夾的意義

- `src/`：核心研究程式碼
- `scripts/`：可執行的訓練、評估、匯出、實驗入口
- `cpp_hls_prep/`：C++ 與 Vitis HLS 的橋接區
- `artifacts/`：可直接展示的實驗產出物
- `meeting/`：每週更新用的投影片與圖表
- `teaching/`：概念筆記與教學型說明

## 目前的 HLS 階段

目前的 HLS 進度屬於「功能已整合完成，但尚未做資源最佳化」的階段。

也就是：

- 是的，模型已轉成 HLS 導向的形式
- 是的，HLS 的 simulation 與 synthesis 報告都已存在
- 不是，現階段架構還不能很舒服地塞進目前的 FPGA 目標

## 最短且誠實的一句現況描述

我已完成端到端的 HLS 交付流程，接下來的主要工作是硬體端的資源與延遲最佳化，以及針對目標器件的 fit-to-device 改善。
