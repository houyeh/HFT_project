# N2EW Era：演進史與繼承關係（2026-06-19 全面重寫版）

> **重寫說明**：本文取代所有先前版本。先前版本在 §9/§11/§12 經歷三輪「補丁式」修正，讀者需要交叉比對才能拿到目前為真的結論。本版直接把最終結論寫進對應主節，不再分輪次補丁。所有結論均標註證據來源（file path + line number 或 audit 檔案），可追溯、可重現。
>
> 補充文件：`B_CLASS_RETROSPECTIVE_ANALYSIS.md`（PHC/PHCSSM 學理根基、reset 證據鏈、oracle 方法論）、`C_CLASS_HLS_STATUS_AND_VERIFY_STEPS.md`（HLS 現況糾正 + 手動複現步驟）。本文已將兩者的最終結論吸收進主文，不需要再回頭查補充文件才能拿到答案——除非要看更細的原始 log 引文。

---

## 0. 一句話總覽

n2ew_era 是 TXF（台指期）L2 order book 高頻造市策略研究線，採用「sequential proxy PnL（causal-fill excess PnL）」作為主要評估目標，而非單純的終端報價 EV。專案在 new_era（舊代）因多個方法論問題（見 §1）被全面重置，n2ew_era 重新建立了凍結資料切分（NEW1-NEW5）、基準階層（No quote → Old baseline → Simple L2 rule → Best Val non-ML rule → ML full-action model）與單次鎖定的最終 holdout（NEW5）。目前的最佳候選是 HGB（樹模型）在 04C 的 NEW5 one-shot 結果，PHC-lite（簡化版遞迴模型）在 05A/05C4 也通過了凍結切分的正號一致性檢查，並已完整跑過 HLS C++ → csim → csynth → cosim → Vivado P&R 全流程且 timing met。仍未完成的是 PYNQ-Z2 實機上板測試，以及若干統計嚴謹度上的已知限制（詳見 §11）。

```
new_era (舊代，已重置)
   │  3 個官方重置理由（§1.2，已用 B_CLASS §C 驗證）
   ▼
n2ew_era (現役)
 00 ── 資料契約與凍結切分 (NEW1-NEW5)
 01-02 ── 基準建立 (No quote / Old baseline)
 03A-03B.6 ── Simple L2 rule + 特徵/早停根因封閉
 04A.2-04C.1 ── ML 全動作模型 (HGB) + NEW5 one-shot + Oracle headroom
 05A ── PHC-lite 序列模型 (簡化版 PHCSSM)
 05C2-05C4 ── PHC + STP（突觸短期可塑性）變體
 06f ── PHC-lite 手寫 HLS C++ → csim/csynth/cosim/P&R（timing met）
 [待辦] PYNQ-Z2 實機上板
```

---

## 1. Root Era 歷史與 Reset 必要性

### 1.1 為什麼會有 new_era → n2ew_era 的重置

`PROJECT_RESET.md` 列出三個官方理由。`B_CLASS_RETROSPECTIVE_ANALYSIS.md` §C 已用 new_era 第 16-38 階段的原始檔案逐一核對，確認三個理由皆有真實證據支撐（非事後合理化）：

1. 評估目標從「終端報價 EV」誤用，轉為「sequential proxy PnL / causal-fill excess PnL」才能正確反映造市策略的逐筆損益。
2. 資料切分曾經存在可被質疑的洩漏風險與切分不一致，n2ew_era 重新定義並凍結 NEW1-NEW5。
3. 基準階層不完整，導致「ML 模型打贏的對象」定義不清楚；n2ew_era 補上 No quote → Old baseline → Simple L2 rule → Best Val non-ML rule → ML 全動作模型的完整階層。

### 1.2 凍結資料切分角色（不可逆規則）

| 角色 | 資料 | 用途 | 規則 |
|---|---|---|---|
| development | NEW1 + NEW2 | 訓練 + 驗證 | 可反覆迭代 |
| used_oos_diagnostic | NEW3 + NEW4 | 開發期診斷用 OOS | 可看，但**不可用於選模型/調參** |
| final_holdout_locked | NEW5 | 最終單次測試 | 一旦用過即鎖定，**只能用一次** |

04C 是目前唯一已經對 NEW5 開過一次的階段（HGB 候選），05A/05C4（PHC 系列）**尚未**對 NEW5 開鎖（見 §8 Task 9 的 9.1 表格，`new5_used: False`）。

---

## 2-5. 00-04C 階段摘要（沿用既有結論，已用本輪 audit 重新核對未變動）

> 此區間結論在本輪重新跑的 `close_gap_audit_generator.py`（Task 1/3/4/5/6）中皆得到「resolved」且與先前文件記載一致，未發現需要修正之處，故保留原表格內容、僅更新證據出處。

### 2.1 00 系列：資料契約

TXF L2 order book 全量原始事件 14,256,951 rows，依官方規則切分為 NEW1-NEW5。`raw_txf_rows` 在各下游階段重新讀取時數字一致（見 §10 Task 3 表格的 `target_valid_minus_raw_txf_rows = 0` 全列）。

### 2.2 非 ML 階段（01-03B.6）

| 階段 | 內容 | 結論 |
|---|---|---|
| 01 | No quote baseline | 對照組，恆為 0 PnL |
| 02 | Old baseline rule | 沿用 new_era 舊規則重新評測 |
| 03A | Simple L2 rule（queue imbalance） | `side_by_queue_imbalance_abs_0p1` 為最終候選 |
| 03A.6 | 特徵/session boundary 根因封閉 | 見 §10 Task 5：session 邊界重置確認無洩漏 |
| 03B6 | HGB 收斂封閉 | 見 §10 Task 4：`n_iter=121` 為 early-stopping 自然結果，非隱藏硬編碼 |

### 2.3 ML 全動作模型（04A.2-04C.1）

04B 選出 `ml_hgb_regressor_shallow`（後於 04B.1 經收斂封閉更名為 `ml_hgb_regressor_shallow_convergence_selected`），於 04C 對 NEW5 進行**唯一一次**最終測試：causal-fill sequential excess PnL = **893,555**（正值，見 §9 Task 9 表格第 5 列）。04C.1 對此結果做 oracle headroom 分析（§7）。

---

## 6. PHC-lite 深度解析（已整併，不再分輪次補丁）

### 6.1 學理根基：PHCSSM（已驗證）

PHC-lite 的命名與架構靈感來自 arXiv 2604.01295《Parallelized Hierarchical Connectome: A Spatiotemporal Recurrent Framework for Spiking State-Space Models》（Po-Han Chiang, NYCU）。`B_CLASS_RETROSPECTIVE_ANALYSIS.md` §A 已逐項核對原論文架構：

- Neuron Layer：ALIF（Adaptive Leaky Integrate-and-Fire）+ surrogate gradient
- Synapse Layer：突觸延遲 + Tsodyks-Markram STP（短期可塑性）+ Dale's Law 約束權重 + R=2 階層遮罩
- Multi-Transmission Loop：M 步循環，深度 O(M log T)
- R-STDP + RMSNorm 讀出層

原論文 ablation（供參考基準，非 n2ew_era 自己的數字）：移除 ALIF −5.23pp、移除 R-STDP −4.37pp、移除 Dale's Law −3.70pp、移除 STP −3.43pp。

**重要紅線（必須在任何報告/答辯場合維持）**：PHC-lite **不是**完整 PHCSSM。它只實作了「可學習 decay/gate 遞迴狀態 + within-timestep multi-transmission loop」，**沒有** spike 機制、沒有 STDP、沒有原論文的 surrogate gradient spiking neuron。這個自我聲明已寫在 `PHC_HLS_Technical_Report.md` §1.2，本文重申以避免被誤讀為「實作了完整論文架構」。原論文本身也**沒有**與 HGB/樹模型的對照實驗——PHC vs HGB 的比較是 n2ew_era 自己設計的，不是延續論文設定。

### 6.2 為什麼會嘗試 PHC：誠實的動機說明

`B_CLASS_RETROSPECTIVE_ANALYSIS.md` §B 明確記載：選擇嘗試 PHCSSM 的最初動機是與指導教授閒談中得知這個架構，屬於機會性嘗試，**不是**先有「HGB 的固定窗口特徵有理論缺陷，所以需要可學習記憶」這個技術論證才去找 PHC。技術論證（B.2：可學習 per-channel decay vs. HGB 的固定窗口手工特徵）是**事後補上**的合理化敘事。本輪 Task 8（§6.4）首次提供了支持「序列/記憶機制確實有貢獻」的實證證據，使這個事後論證從「未經測試的假說」升級為「有條件成立的結論」，但仍建議在答辯時誠實交代動機的時間順序。

### 6.3 訓練與跨種子穩定性

PHC-lite v1（05A）三個種子（seed 0/1/2）獨立訓練，最佳 epoch 47-50。跨種子穩定性（`phc_lite_seed_stability.csv`）：

| seed | split | causal_fill_pnl | beats_queue_rule | beats_hgb |
|---:|---|---:|---|---|
| 0 | val | 170,310 | True | True |
| 0 | new3 | 488,870 | True | True |
| 0 | new4 | 121,765 | True | True |
| 1 | val | 151,530 | True | True |
| 1 | new3 | 454,620 | True | True |
| 1 | new4 | 111,995 | True | True |
| 2 | val | 162,625 | True | True |
| 2 | new3 | 482,590 | True | True |
| 2 | new4 | 116,845 | True | True |

三個種子在 val/NEW3/NEW4 上全部贏過 queue rule 與 HGB，方向一致，數值有 10-15% 量級的波動，屬合理範圍。

### 6.4 記憶機制驗證（本輪最重要的新結論，封閉了先前「孤兒 CSV」缺口）

**先前缺口**：舊版文件 §12.1 標記「memory ablation 數字來自一份找不到生成腳本的 CSV，無法溯源」。本輪重新檢視 `close_gap_audit_generator.py` 的 `task8_memory()` 函式，確認其本身仍只是**讀取**既有 CSV（`phc_lite_memory_ablation.csv`、`phc_lite_seed_stability.csv`），不負責重新計算。但本輪新增的 `AUDIT_learned_memory_vs_fixed_window.md`（05a 資料夾內，與訓練腳本同目錄）明確記載了方法論：**沒有重新訓練**，而是用既有凍結 checkpoint，在推論階段做反事實（counterfactual）變更（固定 decay=0.9、memory-off decay≈0、序列反轉、時間步打亂），這是可重現、可審查的實驗設計，缺口視為**已封閉**。

#### 6.4.1 可學習 decay 結構（三個種子）

| seed | best_epoch | decay_mean | decay_std | half_life_mean_events | alpha |
|---:|---:|---:|---:|---:|---:|
| 0 | 48 | 0.899067 | 0.00477 | 6.530 | 0.300 |
| 1 | 50 | 0.900460 | 0.00378 | 6.621 | 0.302 |
| 2 | 47 | 0.899532 | 0.00361 | 6.555 | 0.356 |

來源：`n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/models/phc_lite.py:52-64`，遞迴更新式 `state_temporal = decay * state + gate_t * candidate_t`。三個種子的 decay 收斂到幾乎相同的窄區間（半衰期 ≈ 6.5-6.6 個事件），本身就是一個值得在答辯中提到的穩定性證據。

#### 6.4.2 反事實 ablation 結果（val / new3 / new4）

| split | variant | total_excess_pnl_H20 | Δ vs baseline | spearman |
|---|---|---:|---:|---:|
| val | baseline | 1,927,170 | 0 | 0.312 |
| val | fixed_decay_0.9 | 1,927,640 | +470 | 0.312 |
| val | memory_off (decay≈0) | 1,123,940 | −803,235 | 0.229 |
| val | reverse_sequence | −837,330 | −2,764,500 | 0.010 |
| val | deterministic_shuffle | −642,420 | −2,569,590 | 0.022 |
| new3 | baseline | 5,021,520 | 0 | 0.330 |
| new3 | memory_off | 3,121,780 | −1,899,740 | 0.240 |
| new3 | reverse_sequence | −1,926,160 | −6,947,680 | 0.004 |
| new4 | baseline | 1,762,500 | 0 | 0.308 |
| new4 | memory_off | 980,800 | −781,695 | 0.214 |
| new4 | reverse_sequence | −767,660 | −2,530,160 | 0.002 |

**誠實結論（直接引用 audit 檔案的「最終答案」，建議答辯時逐字使用）**：

> PHC-lite 並未證明實作了完整的生物遞迴機制，但凍結推論的健全性檢查顯示模型確實使用了序列順序：反轉或打亂時間步會讓相關性崩潰、PnL 轉負；關閉記憶（memory-off）會讓 PnL 明顯轉弱但仍為正。固定 decay=0.9 的結果與學習到的 decay 幾乎打平，所以誠實的說法是「序列順序 + 閘控狀態本身有價值」，而不是「每一個學習到的 decay channel 都各自必不可少」。

**審查者會追問、本文主動標註的限制**：

- 這是**反事實推論**測試，不是**重新訓練**式的 ablation（例如「從頭訓練一個強制 decay=0.9 的模型」）。兩者結論可能不同：本測試只能說明「目前這組已訓練好的權重，在序列被破壞時會失效」，不能完全排除「如果從頭就用固定 decay 訓練，模型可能會學到別的方式補償」的可能性。
- fixed_decay_0.9 與 baseline 幾乎打平（Δ 僅 +470 / −7,505 量級），這個數字本身就在削弱「可學習 decay 是關鍵創新」的敘事，必須誠實呈現，不能選擇性引用「memory-off 大幅下降」而隱藏這一行。

### 6.5 PHC + STP 變體（05C2-05C4）

05C4 的 `phc_stp_v2_cosine_lr` 通過驗證集 seed/threshold 選擇 + 六項採用門檻（adoption gates），在 NEW3/NEW4 frozen split 上維持正值（§8 Task 9 表格）。NEW5 尚未對此候選開鎖。

---

## 7. Oracle Headroom 分析（已用修正後措辭，不需要再次更正）

### 7.1 三層 Oracle 定義

| Oracle | 定義 | 可交易性 |
|---|---|---|
| 全動作 oracle (`oracle_full_action`) | 逐 state 取 `argmax(excess_bid, excess_ask, 0)`，pointwise 最優 | 不可交易，純診斷 |
| matched-count oracle | 依策略選中數量，貪婪排序最佳 state 值 | 不可交易 |
| causal-fill sequential oracle | 用未來標籤產生決策，**但**套用與真實策略相同的 `conservative_l2_stress` 成交機率、H20 排程平倉、inventory cap=1 模擬器 | 不可交易，是「policy-constrained 診斷上界」 |

### 7.2 NEW5 final holdout 數字

- `oracle_full_action`：84,517,995（dense action-level，state_count=815,912，selection_rate=0.708）
- causal-fill sequential oracle：3,923,130（同樣的 fill 模擬器約束下）
- 兩者比值 ≈ 21.5x，差距主要來自 inventory cap=1、H20 強制平倉、隨機 fill 機率等約束，**不是**因為策略「笨」。

### 7.3 Capture ratio（策略實際捕捉到的機會比例）

| policy | policy_pnl | matched_count_oracle_pnl | capture_ratio | opportunity_gap |
|---|---:|---:|---:|---:|
| side_by_queue_imbalance_abs_0p1 (Simple L2) | 11,439,800 | 83,102,000 | 0.138 | 71,662,200 |
| ml_hgb_regressor_shallow_convergence_selected | 15,648,100 | 83,916,600 | 0.186 | 68,268,600 |

**這兩個數字應該主動在報告中呈現，而不是只展示「PnL 為正」**：HGB 候選只捕捉到 matched-count oracle 機會的 18.6%，Simple L2 規則只捕捉到 13.8%。這代表理論上限與目前實作之間仍有巨大空間，這既是誠實的限制揭露，也是「未來工作」的論述素材。

### 7.4 修正後的標準措辭（取代 B_CLASS §D.2 的較強說法）

> causal-fill sequential oracle 是在固定的 proxy 執行假設（`conservative_l2_stress` 成交模型、H20 平倉、inventory cap=1）下，使用未來標籤產生的診斷性上界。它**不是**可交易策略，**也不是**對所有可能的存貨約束策略空間的正式全域最優解（沒有經過動態規劃證明）。它的用途是衡量在固定 proxy 假設下還有多少機會空間，以及因果模型實際捕捉到了多少可觀測的未來標籤機會。

**禁止使用的說法**：真實交易所回放（true exchange replay）、實際可上線獲利保證、已證明的可執行最優策略。`true_queue_replay` 在所有 oracle 列中皆為 `False`，必須維持這個事實。

### 7.5 尚未做的敏感度分析（主動揭露，不要等被問到）

`04C.1` 目前固定 `fill_model=conservative_l2_stress`、`inventory_cap=1`、`horizon=20`，**沒有**掃過 inventory cap 2/3 或其他 fill model 假設。如果上限本身對這些假設高度敏感，目前的 21.5x gap 數字可能會隨假設改變而大幅變動，這是一個尚待補的敏感度分析。

---

## 8. 評估方法論的外部驗證（Task 9）

### 8.1 Val-selected 策略在 frozen split 上的正號一致性

| phase | policy | val pnl | frozen split | frozen pnl | sign 一致 |
|---|---|---:|---|---:|---|
| 04B | side_by_queue_imbalance_abs_0p1 | 2,040,260 | NEW3 | 5,731,540 | True |
| 04B | side_by_queue_imbalance_abs_0p1 | 2,040,260 | NEW4 | 2,075,260 | True |
| 04B.1 | ml_hgb_regressor_shallow | 2,863,500 | NEW3 | 7,652,780 | True |
| 04B.1 | ml_hgb_regressor_shallow | 2,863,500 | NEW4 | 2,791,850 | True |
| 04C | ml_hgb_regressor_shallow_convergence_selected | 2,863,500 | **NEW5 one-shot** | 893,555 | True |
| 05A | phc_lite_v1 | 170,310 | NEW3 | 488,870 | True |
| 05A | phc_lite_v1 | 170,310 | NEW4 | 121,765 | True |
| 05C4 | phc_stp_v2_cosine_lr | 171,835 | NEW3 | 611,095 | True |
| 05C4 | phc_stp_v2_cosine_lr | 171,835 | NEW4 | 129,160 | True |

- n = 9 個 policy/frozen-split 組合，sign 一致率 = 1.000，frozen 觀測中正值比例 = 1.000。

### 8.2 答辯用標準措辭（建議逐字引用）

> n2ew_era 的評估方法論有正面的實證支持，但不是證明。在被審查的 val-selected 策略中，frozen NEW3/NEW4 檢查與 04C 的 NEW5 one-shot HGB holdout 都維持了正值的 causal-fill sequential PnL。樣本仍然小且跨階段相關，所以這個說法應該限定為：這個評估協議看起來對篩選 proxy 策略有用，但還不是一個獨立證明過的實際獲利保證。

**主動揭露的統計限制**：n=9 不是獨立樣本——05A 與 05C4 的 NEW3/NEW4 觀測，以及 04B/04B.1/04C 之間，共享同一段市場資料與高度相關的策略邏輯，不能套用標準二項檢定算信賴區間（audit 檔案本身也只計算比例，沒有計算信賴區間，這是正確的保守做法，本文沿用，不額外加碼宣稱統計顯著性）。

---

## 9. HLS / FPGA 硬體現況（已用 C_CLASS 糾正，取代舊版的「未在本次調查範圍內」說法）

### 9.1 重大糾正：phc_stp_v2_cosine_lr 早已跑過完整 P&R

舊版文件（與原始 `CODEX_PROMPT_close_documentation_gaps.md` 的 Task 7）誤認為 `phc_stp_v2_cosine_lr` 從未跑過 06 系列 HLS validation。`C_CLASS_HLS_STATUS_AND_VERIFY_STEPS.md` 用原始 log（非僅讀 markdown 摘要）逐一核對後確認此前提錯誤：

- `PHC_WEIGHT_EXPORT_REPORT.md:7`：source checkpoint 為 `phc_stp_v2_seed2_best_val.pt`。
- `hls_run_cosim.log:25925-25927`：`total=25600 mismatches=0, max_abs_diff=0.00725275`，cosim **PASS**。
- `bd_0_wrapper_timing_summary_routed.rpt:145,148`：`WNS=+0.020ns, TNS=0.000ns`，22,941 個 endpoint 全過，timing met。

**結論**：`phc_stp_v2_cosine_lr`（seed2）已完整跑過 csim → csynth → cosim → **Vivado P&R（Implementation）**全流程且 timing met，比 Task 7 原本要求的範圍（僅 csim/cosim + 資源數字）還完整。

### 9.2 PHC-lite 資源數字演進（v1→v8，含 P&R 實測對照）

舊有 memory 記載「PHC csynth 資源爆炸 11-26 倍」是 v1-v7 迭代過程中的**真實但過時**狀態（v1: LUT 2073%／DSP 2636%；v7: DSP 仍 121%）。v8（2026-06-15，加上 `#pragma HLS PIPELINE off` 修正 pipeline_loops 自動接管問題）後：

| 資源 | csynth 估計 | P&R 實測 | xc7z020 budget |
|---|---:|---:|---|
| DSP | 21% (46/220) | — | 220 |
| LUT | 46% (24,836/53,200) | **14.20%**（synthesis 邏輯優化後更低） | 53,200 |
| FF | 13% (14,518/106,400) | — | 106,400 |
| BRAM | 2% (6/280) | — | 280 |
| Latency | 8,316 cycles ≈ 83.16µs @ 100MHz | — | — |

P&R 實測 LUT 比 csynth 估計低（46.68% → 14.20%）是 Vivado 邏輯優化的正常現象，**不是**數字矛盾。

### 9.3 仍未解決的三個缺口（主動列出，答辯前務必心裡有底）

1. **跨種子 HLS 穩定性未測**：目前 HLS 全流程只驗證了 `phc_stp_v2_cosine_lr` 的 seed2 checkpoint。05c4 若訓練了其他 seed，理論上資源數字應幾乎不變（架構相同，Q24.12 量化只要不飽和不影響資源），但 `max_abs_diff` 可能因權重分布不同而略有差異——**這點未經實測**。修補成本低：只需把 `00_export_weights.py` 指向其他 seed checkpoint 再跑 csim，不需重新設計 HLS C++。
2. **PYNQ-Z2 實機上板未做**：`PHC_HLS_Reviewer_Briefing.md` §7.1 Q7 已自承列為 future work。csim/cosim/P&R timing met 是業界認可的完整驗證鏈，但離「插上板子量到 83µs」還有一步，需要 Vivado block design 整合 PS-PL 介面。
3. **測試向量可能被質疑挑選了好算的子集**：目前 HLS 驗證用的 25,600 組測試向量是「validation split 前 200 個 sequence」，不是完整 validation set，也不是獨立第二份 Python reference 重新生成的向量。如果審查者問「你怎麼知道沒有挑到特別好算的子集」，目前沒有反駁證據。

---

## 10. 其他已封閉缺口一覽（Task 1, 3, 4, 5, 6）

| Task | 主題 | 結論 | 殘留限制 |
|---|---|---|---|
| 1 | 圖表座標軸語義 | 多數圖表座標軸標籤可從原始碼回溯（自動掃描 `plt.title/xlabel/ylabel`） | 部分 subplot 共用同一個最近的 `plt.title()` 呼叫，造成標題字串重複；**會議用投影片需手動逐圖核對標題與實際內容是否一致**，不能直接信任自動掃描結果 |
| 3 | `state_count` vs `target_valid_rows` 數字差異 | 差異來自分母轉換鏈：`target_valid_rows → raw_txf_rows → valid_book_rows → boundary_kept_rows(=state_count) → action_rows(×3) → PHC window state_count` | 純解釋性，無資料遺失 |
| 4 | HGB 三個 `max_iter`（320/500/800）皆停在 `n_iter=121` | 三者皆用相同 `early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, tol=1e-7`，內部驗證曲線在第 121 輪不再改善 | 非隱藏硬編碼，亦非用了 frozen OOS 來選代數 |
| 5 | Session boundary 特徵洩漏 | 抽樣 12 個 session 的邊界第一筆 row，inter-arrival/event-rate/imbalance-change/spread-change 皆為 0；rolling volatility 重新計算誤差 = 0 | 僅為靜態 artifact 抽樣檢查，非對所有 session 全面重算 |
| 6 | Frozen split 是否被用於調參 | 8 個最終 artifact（policy JSON、threshold grid、matched-random summary、NEW5 frozen eval、seed-stability CSV）皆標記 val-only 選擇、NEW3/4/5 未用於選擇 | **這是靜態 flag 檢查，不是不可篡改的 git 版本歷史證明**；如果審查者堅持要看 commit 級別的時間序列證據，目前沒有 |

---

## 11. 答辯前必須心裡有底的完整限制清單（教授/審查者視角彙整）

> 這是本文最重要的一節：把全文分散提到的限制集中列出，方便答辯前快速複查。每一項都已在上文標明出處，這裡只做索引式重申。

1. **PHC-lite 不是完整 PHCSSM**（§6.1）——缺 spike/STDP 機制，這是自我聲明的紅線，不能被誤讀。
2. **PHC 嘗試動機是事後合理化**（§6.2）——技術論證在嘗試之後才補上，誠實交代時間順序。
3. **記憶 ablation 是反事實推論，非重新訓練**（§6.4）——`fixed_decay_0.9` 幾乎打平 baseline，不能只強調 memory-off 的下降。
4. **Oracle 是診斷上界，非全域最優、非可交易**（§7.4）——`true_queue_replay=False` 在所有列中皆為事實。
5. **Oracle 敏感度分析未做**（§7.5）——inventory cap、fill model 都只測了一組設定。
6. **方法論外部驗證樣本小且相關**（§8.2）——n=9，跨階段共享市場資料，沒有算信賴區間是對的，不要額外宣稱統計顯著。
7. **HLS 跨種子穩定性未測**（§9.3-1）——只驗證了 seed2。
8. **PYNQ-Z2 實機上板未做**（§9.3-2）——目前最後一步未完成。
9. **HLS 測試向量可能不具代表性**（§9.3-3）——只用了 validation split 前 200 組，非完整集合或獨立 reference。
10. **圖表標題自動掃描可能有誤**（§10 Task 1）——會議投影片務必逐圖手動核對。
11. **Frozen split 未調參的證據是靜態 flag，非 git 版本歷史**（§10 Task 6）。

---

## 12. 若需要進一步補強：給 Codex agent 的工作提示

> 以下兩項是目前性價比最高、可在不需要實機硬體的情況下封閉的缺口（對應 §11 第 7、9 項）。如果使用者要準備答辯且時間有限，建議優先做 12.1（成本低、立即可做）；12.2 視答辯嚴謹度需求決定是否要做。PYNQ-Z2 上板（§11 第 8 項）需要實體硬體操作，不適合純程式碼任務，不在此提示範圍內。

### 12.1 跨種子 HLS 一致性檢查（低成本，建議優先）

```
任務：對 05c4_phc_stp_cosine_lr 的其他已訓練 seed checkpoint（非 seed2），重跑
06f_phc_lite_manual_hls 的 weight export + csim 兩步，確認 max_abs_diff 量級是否與
seed2 的 0.00725275 一致。

鐵則（沿用 CODEX_PROMPT_close_documentation_gaps.md 的規則）：
1. 絕對不要修改任何已凍結檔案：hls_config.cfg、phc_step_hls.cpp、現有的 seed2
   weights/log/report。只允許新增檔案（例如新的 weight header 與新的 log 輸出到
   獨立子目錄，不要覆寫 06f 既有的 phc_weights.hpp）。
2. 先確認 05c4_phc_stp_cosine_lr/models/ 底下實際有哪些 seed 的
   best_val.pt（用 ls 或等價指令列出，不要假設檔名），再決定要跑哪個 seed。
3. 必須真的執行 00_export_weights.py 與 vitis-run --mode hls --csim，不能只讀
   程式碼推論結果。把 csim log 的 mismatches / max_abs_diff / mean_abs_diff
   完整貼出來。
4. 輸出一份新檔案 AUDIT_hls_cross_seed_consistency.md，格式仿照既有
   AUDIT_*.md（status: resolved/gap_confirmed_unresolvable 開頭，附 file path +
   line number 引用）。明確寫出每個測試 seed 的 mismatches 與 max_abs_diff，
   並與 seed2 的 0.00725275 比較，誠實標註是否在同一量級。
5. 若 max_abs_diff 差異超過一個量級（例如從 1e-3 變成 1e-1），不要美化或省略，
   要在 finding 欄位標記 negative 並具體寫出觀察到的數字。
6. 不要跑 csynth/cosim/P&R（不需要重複驗證資源與 timing，那部分已知架構不變
   不會改變；只需確認數值精度層面的跨種子一致性）。
```

### 12.2 Oracle 約束敏感度掃描（中等成本，視答辯嚴謹度需求）

```
任務：在 04c1_new5_oracle_headroom_analysis 中，對 causal-fill sequential
oracle 補做 inventory_cap ∈ {1, 2, 3} 與至少一個替代 fill_model 假設的敏感度
掃描，量化 21.5x 的 oracle/policy gap 對這些假設的敏感程度。

鐵則：
1. 不要修改任何現有的 04C/04C.1 凍結報告或 CSV，新增獨立輸出檔案。
2. 沿用既有的 conservative_l2_stress 模擬器邏輯作為基準，新增的 fill_model
   變體要明確寫出與基準的差異（例如成交機率函數的具體改動），不要含糊地說
   「更寬鬆的假設」。
3. NEW5 是 final_holdout_locked，本次掃描如果要用到 NEW5 的標籤生成 oracle
   決策，必須在輸出檔案最上方明確標註「本次屬於診斷性 oracle 敏感度分析，
   不構成對 NEW5 的二次策略選擇或調參」，並確認 04C 既有的 HGB 候選策略本身
   不會因此被重新挑選或微調。
4. 輸出 AUDIT_oracle_constraint_sensitivity.md，必須包含一個表格：
   {inventory_cap, fill_model} × {oracle_pnl, capture_ratio} 的完整矩陣，
   並寫出 21.5x 這個比值在掃描範圍內的最大/最小值。
5. 如果掃描結果顯示 gap 對假設高度敏感（例如某個合理假設下 gap 縮小到 5x 以下），
   要誠實寫出來，不要只報告讓故事好看的那組設定。
```

---

## 13. 來源索引（本次重寫直接引用的檔案）

- `n2ew_era/000_context/close_gap_audit_generator.py`（739 行，生成下列所有 AUDIT_*.md 與 GAP_CLOSURE_SUMMARY.md / N2EW_GAP_CLOSURE_TOTAL_RESULT.md 的腳本）
- `n2ew_era/000_context/GAP_CLOSURE_SUMMARY.md`、`N2EW_GAP_CLOSURE_TOTAL_RESULT.md`
- `n2ew_era/000_context/AUDIT_chart_axis_semantics.md`（Task 1）
- `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_phc_lite_transform_features.md`（Task 2）
- `n2ew_era/04b_txf_only_final_development_rerun/AUDIT_state_count_vs_target_valid_rows.md`（Task 3）
- `n2ew_era/03b6_hgb_convergence_closure/AUDIT_identical_n_iter_121.md`（Task 4）
- `n2ew_era/03a6_root_cause_feature_and_contract_closure/AUDIT_session_boundary_leakage.md`（Task 5）
- `n2ew_era/000_context/AUDIT_frozen_split_no_tuning_check.md`（Task 6）
- `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md`（Task 8）
- `n2ew_era/000_context/AUDIT_evaluation_methodology_external_validation.md`（Task 9）
- `n2ew_era/04c1_new5_oracle_headroom_analysis/AUDIT_causal_fill_oracle_derivation.md`（Task 10）
- `n2ew_era/000_context/B_CLASS_RETROSPECTIVE_ANALYSIS.md`（PHCSSM 學理根基、reset 證據鏈、oracle 方法論）
- `n2ew_era/000_context/C_CLASS_HLS_STATUS_AND_VERIFY_STEPS.md`（HLS 現況糾正、手動複現步驟）
- `n2ew_era/000_context/CODEX_PROMPT_close_documentation_gaps.md`（本輪 audit 的原始任務指派提示，含鐵則）
- `n2ew_era/06f_phc_lite_manual_hls/PHC_WEIGHT_EXPORT_REPORT.md`、`hls_component/hls_component/logs/hls_run_cosim.log`、`.../impl_1/bd_0_wrapper_timing_summary_routed.rpt`（HLS 原始 log，C_CLASS 引用）
- arXiv 2604.01295（Po-Han Chiang, NYCU，PHCSSM 原始論文，僅供架構對照，n2ew_era 未重現其完整實驗）

---

*本文件由 2026-06-19 重新跑過的 `close_gap_audit_generator.py` 輸出（Task 1-6, 8-10，Task 7 已正式撤回並由 C_CLASS 文件取代）與 B_CLASS / C_CLASS 補充文件全面重寫產出，取代所有先前版本。*
