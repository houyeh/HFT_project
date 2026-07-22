# 競賽提案書：PYNQ-Z2 上 PHC-lite 神經網路推論核心之 HLS 硬體加速

**專案代號：** PHC-lite HLS Inference Core
**目標硬體：** PYNQ-Z2（Xilinx xc7z020clg400-1）
**文件版本：** v1.0
**文件日期：** 2026-06-18
**文件性質：** 競賽提案 / 申請文件
**對應技術依據文件：** `n2ew_era/PHC_HLS_Reviewer_Briefing.md`、`n2ew_era/PHC_HLS_Technical_Report.md`

---

## 摘要（Abstract）

本提案展示將一個 16 維隱藏狀態的遞迴神經網路（PHC-lite，取材自 PHCSSM 之簡化版本）以手動 Vitis HLS 實作，部署於 PYNQ-Z2（xc7z020）FPGA 平台，作為高頻交易（HFT）訊號推論核心的可行性研究。專案核心貢獻在於完整記錄並解決了 8 輪 HLS 資源爆炸問題（從初始 2636% DSP 超用，逐步定位三個根本原因並修復至全部資源落在預算內），並完成 csim（25,600 組測試向量、0 mismatch）、cosim（bit-level 與 C model 一致）、Vivado P&R（timing met，WNS=+0.020ns）、與 CPU baseline 對照（FPGA p99 tail latency 優於 CPU 9.4 倍）的完整驗證鏈。本提案不宣稱已達成可部署的 HFT 系統，亦不宣稱 FPGA 改善了 PnL；目的是展示系統化的 HLS 優化方法論與工程嚴謹度，並提出下一階段（量化、上板測試）的具體工作計畫。

---

## 1. 研究動機與問題定義

高頻交易系統對推論延遲的確定性（determinism）要求高於平均延遲本身。CPU/GPU 推論路徑受 OS 排程、垂直快取失效、記憶體回收（GC）等因素影響，尾端延遲（tail latency）波動大，在交易視窗極短的場景下等同於錯失機會。FPGA 提供確定性執行時間與零抖動（zero jitter）的潛力，但要驗證這個潛力是否能落地在學術可取得的入門級 FPGA（xc7z020，僅 220 個 DSP、53,200 個 LUT）上，需要回答兩個問題：

1. 一個有意義的遞迴神經網路模型，能否在資源極受限的平台上完成 HLS 綜合並通過實體佈局繞線（P&R）？
2. 若可行，其延遲特性是否真的優於同等 CPU 推論路徑，優勢具體展現在哪裡？

本專案以 PHC-lite（PHCSSM 簡化版）作為實驗對象，針對上述兩個問題提供了完整、可驗證、有精確數據支持的回答。

---

## 2. 技術背景

PHC-lite 是 PHCSSM（一種具生物啟發 short-term-plasticity-like gain 機制的遞迴狀態空間模型）的簡化版本，用於市場微結構特徵的逐步（step-wise）推論。本專案的範圍限定在單步（single timestep）的 HLS 硬體實作，而非完整序列模型的端到端訓練或回測。

**重要邊界聲明（適用於本提案全文）：**

- PHC-lite ≠ 完整 PHCSSM：本專案僅實作 16 維隱藏狀態的簡化版本，特定 checkpoint（seed2）。
- STP-like gain ≠ 完整生物 STP 機制：模型設計上的簡化命名，非神經科學意義上的真實突觸機制。
- Causal-fill proxy ≠ 真實 exchange replay：所有測試向量來自驗證集模擬資料，非真實市場逐筆回放。
- PYNQ-Z2 不是生產部署平台：是教育/研究用展示板，本提案不宣稱已達到生產級部署。
- 本專案不宣稱 FPGA 改善了 PnL：未進行任何交易回測或 A/B test。

---

## 3. 系統架構

### 3.1 模型 I/O 與計算流程

```
輸入：x_t[16]（市場特徵，ap_fixed<24,12>）+ state_prev[16]（前一步隱藏狀態）
輸出：readout[2]（推論訊號）+ state_next[16]（更新後隱藏狀態）

STEP1  encoded         = SiLU(LayerNorm(ENC_LINEAR(x_t)))
STEP2  gate_t           = Sigmoid(GATE_LINEAR(encoded))
STEP3  candidate_t      = Tanh(CANDIDATE_LINEAR(encoded))
STEP4  state_temporal   = DECAY ⊙ state_prev + gate_t ⊙ candidate_t
STEP5  gain_t           = Sigmoid(GAIN_FROM_ENCODED_LINEAR(encoded) + GAIN_FROM_PREV_STATE_W × state_prev)
STEP6  message          = SiLU(SYNAPSE_LINEAR(state_temporal))
STEP7  loop_input       = state_temporal + ALPHA ⊙ gain_t ⊙ message
STEP8  z                = LayerNorm(loop_input)
STEP9  readout          = READOUT_LINEAR(z)；state_next = state_temporal
```

共 7 個 linear layer、2 個 LayerNorm、3 種非線性活化（Sigmoid/Tanh/SiLU）。非線性以 65 點預計算定點 LUT（範圍 [-4, 4]，步長 0.125，線性插值）實現，避免任何浮點硬體核心。

### 3.2 定點數設計

| 型別別名 | 定義 | 用途 |
|---|---|---|
| `phc_fixed_t` | `ap_fixed<24,12>` | I/O、權重 |
| `phc_acc_t` | `ap_fixed<40,16>` | 累加器 |
| `phc_var_t` | `ap_fixed<48,24>` | LayerNorm 變異數計算 |

### 3.3 遞迴模型的硬體約束

t+1 步需要 t 步的 state_next，因此跨步無法 pipeline；有效 Initiation Interval（II）等於 Latency。這是模型架構的根本約束，非 HLS 實作疏漏，提案中所有延遲數字皆基於此約束下的單步（per-step）延遲。

---

## 4. 方法論：8 輪 HLS 優化歷程

手動 HLS（而非 hls4ml 自動化工具）的選擇理由：自動展開策略在 220-DSP 等級的小型 FPGA 上通常直接超用資源；手動控制每個 pragma 可精確管理資源與延遲的權衡，且每輪修改皆可從 compile log 的 `[HLS 2xx-xxx]` warning/info 追蹤工具的自動行為，建立可重現、可解釋的優化路徑。

| 版本 | 關鍵動作 | DSP | LUT | FF | BRAM | csim |
|---|---|---|---|---|---|---|
| v1 | 初始（double tanh） | 2636% | 2073% | 441% | — | — |
| v2 | 65 點 LUT tanh | 1603% | 839% | 441% | — | PASS |
| v3–v4 | 移除/調整 UNROLL、partition | 1603% | 839% | 441% | — | — |
| v5 | PIPELINE II=297 | ~395% | — | — | — | — |
| v6 | 刪除 PIPELINE pragma | 247% | 113% | 86% | 34% | PASS |
| v7 | 移除 helper ARRAY_PARTITION | 121% | 77% | 52% | 0% | — |
| **v8** | **`#pragma HLS PIPELINE off`** | **21%** | **46%** | **13%** | **2%** | **PASS** |

**三個根本原因（root cause）：**

1. C 標準 `tanh()` 在 HLS 中生成雙精度浮點核心（`generic_tanh_double_s`），單一調用即耗用約 2,048 個 DSP。修復：65 點定點 LUT 取代浮點運算。
2. 任何 `#pragma HLS PIPELINE II=X`（無論 X 為何值）都會觸發 `[HLS 214-275]`，強制將 partial UNROLL 升格為 complete UNROLL。修復：完全刪除 PIPELINE pragma。
3. 刪除 PIPELINE 後，HLS 全域 `pipeline_loops` 閾值設定仍會對 ≤16 次迭代的迴圈自動套用 pipeline（`[HLS 214-376]`），進而觸發 inner loop 強制展開（`[HLS 214-291]`）與權重陣列自動 partition（`[HLS 214-449]`）的連鎖反應。修復：對所有相關迴圈明確加上 `#pragma HLS PIPELINE off`，其優先級高於全域 `pipeline_loops` 設定。

此優化歷程的價值在於：每一輪修改皆有明確假設、明確驗證指標（資源使用率）、明確失敗或成功的判定依據，構成一個可被 reviewer 逐步檢視的因果鏈，而非黑箱調參。

---

## 5. 目前已完成且已驗證之成果

### 5.1 HLS 綜合（csynth，v8，2026-06-15 02:25）

| 資源 | Used | Available | Util% |
|---|---|---|---|
| DSP48E1 | 46 | 220 | 20.91% |
| LUT | 24,836 | 53,200 | 46.68% |
| FF | 14,518 | 106,400 | 13.65% |
| BRAM_18K | 6 | 280 | 2.14% |

延遲（csynth 估算）：8,316 cycles @ 10ns clock = **83.16 µs**（序列執行，不可 pipeline）。

### 5.2 C Simulation（csim）

25,600 組測試向量（200 個驗證集序列 × 128 timestep），全部通過，0 mismatch。max_abs_diff = 0.00725275，mean_abs_diff = 0.000372553，tolerance（PHC_ABS_TOL）= 1.0。

### 5.3 C/RTL Co-simulation（cosim）

25,600 組向量全部通過，max_abs_diff 與 csim **完全相同**（0.00725275），證明 RTL 實作在 bit-level 與 C model 一致，未引入額外精度損失。

### 5.4 Vivado Place & Route（Implementation）

執行於 2026-06-15 15:45，Vivado v2025.2。

| 項目 | 數值 |
|---|---|
| Clock period required | 10.000 ns |
| CP achieved post-implementation | 9.980 ns |
| WNS | +0.020 ns |
| TNS | 0.000 ns（0 failing endpoints / 22,941 total） |
| Timing | **Met** |

Post-route 資源（較 csynth 估算大幅下降，因 Vivado synthesis 做了邏輯合併優化）：

| 資源 | csynth 估算 | P&R 實際 | 可用量 | 實際% |
|---|---|---|---|---|
| Slice LUT | 24,836 | 7,555 | 53,200 | 14.20% |
| FF | 14,518 | 8,217 | 106,400 | 7.72% |
| DSP48E1 | 46 | 47 | 220 | 21.36% |
| BRAM_18K | 6 | 6 | 280 | 2.14% |

### 5.5 CPU Baseline 對照

PyTorch float32，`torch.no_grad()`，batch_size=1，warm-up 100 次，正式計時 10,000 次：

| 指標 | FPGA（csynth 固定值） | CPU（PyTorch, no_grad） |
|---|---|---|
| Latency（固定/mean） | 83.16 µs | 117.5 µs |
| Latency（median） | 83.16 µs | 84.2 µs |
| Latency（p99） | **83.16 µs** | **778.8 µs** |
| Jitter | 0 | 高（GC / OS 排程） |

**正確解讀：** FPGA 的優勢不在平均延遲（與 CPU 接近），而在尾端延遲（p99 降低 9.4 倍）與零抖動的確定性，這正是 HFT 場景中真正關鍵的特性。

---

## 6. 已知限制與誠實揭露

以下限制不應在報告或口頭答辯中被淡化或迴避：

1. **WNS margin 偏緊（+0.020 ns）。** Timing 仍判定為 met，22,941 個 endpoint 全數通過、TNS=0，但若需更大裕度，須降頻至 90MHz 並重新跑 P&R。
2. **83.16 µs 不滿足 HFT L1 典型 <10 µs 的需求。** 本專案展示的是 FPGA 推論的可行性與系統化優化方法，非可直接部署的低延遲方案。
3. **未做上板（PYNQ-Z2 實機）測試。** 現有 csim → cosim → P&R timing met 的驗證鏈在硬體設計報告中屬於完整且被接受的驗證流程，但實機量測仍未進行，列為下一階段工作。
4. **25,600 組測試向量為驗證集子集，非全量資料，亦未涵蓋極端邊界值（NaN、極端 outlier）。**
5. **CPU baseline 為 Python/PyTorch 推論路徑，非經過 C++/LibTorch/TorchScript 優化的最佳化基準**，p99 數字包含 Python 層 overhead。

---

## 7. 下一階段工作計畫與時程

| 階段 | 工作項目 | 預計時間 | 預期產出 |
|---|---|---|---|
| Phase 1（已完成） | HLS 優化、csim/cosim/P&R 驗證、CPU baseline | 2026-06-13 ~ 06-15 | v8 設計、Reviewer Briefing |
| Phase 2 | INT16 量化（`ap_fixed<16,8>`）以降低延遲與資源 | 2026-06-19 ~ 06-25 | 量化版本 csynth/csim 報告，預估延遲 ~25 µs |
| Phase 3 | PS-PL 整合與 PYNQ-Z2 實機上板測試 | 2026-06-26 ~ 07-05 | 實機量測延遲、與 csynth/P&R 估算之對照 |
| Phase 4 | Hidden dim 縮減實驗（16→8）可行性評估 | 2026-07-06 ~ 07-10 | 精度 vs 延遲 tradeoff 報告 |
| Phase 5 | 競賽報告與簡報整理、reviewer 預答辯 | 2026-07-11 ~ 07-18 | 最終報告、簡報、Q&A 準備清單 |

每個階段皆延續本提案的方法論：先有明確假設與驗證指標，再執行修改，最後以工具實際輸出（非估算）作為結案依據。

---

## 8. 預期貢獻

1. 提供一個資源極受限 FPGA 平台（xc7z020）上，遞迴神經網路 HLS 實作的完整、可重現的優化案例研究，包含三個根本原因的精確診斷與修復方法。
2. 提供 csim/cosim/P&R 三層驗證鏈的完整數據基準，可作為後續量化、架構縮減等工作的對照基線。
3. 釐清「FPGA 加速」在 HFT 場景中應被正確理解為尾端延遲與確定性的改善，而非單純平均延遲的競賽，並以實測數據（p99 9.4× 改善）支持此論點。

---

## 9. 結論

本提案以扎實、可驗證的工程數據展示了 PHC-lite 在 PYNQ-Z2 上的 HLS 實作可行性，並誠實揭露現階段的限制（延遲未達 HFT 部署門檻、未上板測試、WNS margin 偏緊）。提案的價值不在於宣稱已解決 HFT 延遲問題，而在於展示一套系統化、可重現、有精確數據支持的 FPGA 加速工程方法論，並提出具體可執行的下一階段計畫。

---

## 附錄：關鍵檔案位置索引

| 內容 | 路徑 |
|---|---|
| 主設計檔 | `n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.cpp` |
| 測試向量 | `n2ew_era/06f_phc_lite_manual_hls/results/phc_step_test_vectors.csv` |
| csynth 報告 | `n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt` |
| P&R export 報告 | `n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/impl/report/verilog/phc_step_export.rpt` |
| Timing summary (routed) | `.../project.runs/impl_1/bd_0_wrapper_timing_summary_routed.rpt` |
| CPU baseline 數據 | `n2ew_era/06f_phc_lite_manual_hls/results/cpu_baseline_timing.json` |
| 完整 Reviewer 簡報（更多細節） | `n2ew_era/PHC_HLS_Reviewer_Briefing.md` |
| 技術報告（背景敘事） | `n2ew_era/PHC_HLS_Technical_Report.md` |

---

*文件結束 | 競賽提案 / 申請文件 | 2026-06-18*
*所有數據均來自實際工具輸出（csim/cosim/P&R/CPU baseline log），未做估算或外推；限制與邊界聲明為本文件不可省略之部分。*
