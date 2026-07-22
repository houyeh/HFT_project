# B類缺口回溯分析（教授/Reviewer視角，2026-06-19）

> 本文件處理4個「文件本身承認查無依據、需要靠現有實作/外部文獻去推斷或建構」的缺口。
> 每一節開頭標注是**已驗證事實**還是**推論/事後建構的論述**，後者絕不能在答辯時說成「原始動機」。

---

## A. PHC / PHCSSM 全名與架構溯源（對應原請求第7點）

**狀態：已驗證（grounded in arXiv 2604.01295）**

來源：[2604.01295] *Parallelized Hierarchical Connectome: A Spatiotemporal Recurrent Framework for Spiking State-Space Models*（Po-Han Chiang, NYCU），https://arxiv.org/abs/2604.01295

- **PHC = Parallelized Hierarchical Connectome**；**PHCSSM** = 該架構作為一種 Spiking State-Space Model 的別稱（論文標題本身就是 "...Framework for Spiking State-Space Models"）。
- 完整架構三層：
  1. **Neuron Layer**：每個神經元獨立對角遞迴（膜電位 V_f、適應性閾值 θ_adapt、refractory 抑制 V_res），ALIF spike 產生機制 + surrogate gradient 訓練。
  2. **Synapse Layer**：Pre-synapse（突觸延遲 + Tsodyks-Markram STP，含 facilitation u / recovery R）+ Post-synapse（權重矩陣，zero-diagonal、Dale's Law 符號約束、R=2 hierarchical mask M_topo）。
  3. **Multi-Transmission Loop**：M 次 transmission step 在 NL/SL 間反覆，直到 Cauchy 收斂或 k=N_max，換取 O(M log T) 總深度（相對純時序 scan 的 O(log T)，用層數換時間平行性）。
  4. 加上 R-STDP 線上學習 + RMSNorm/linear readout。
- 論文主張：第一個保留時序平行掃描效率、同時具備可學習側向連接（lateral connection）的 SSM 框架；參數複雜度從 Θ(D²L) 降到 Θ(D²)。
- Ablation（論文原始數字）：移除 ALIF -5.23pp、移除 R-STDP -4.37pp、移除 Dale's Law -3.70pp、移除 STP -3.43pp——四者皆非冗餘設計。

**重要邊界（必須在答辯時主動聲明，否則會被抓到）：**
- 論文**完全沒有**與樹模型（GBM/HGB/XGBoost等）的任何比較實驗。任何PHC-vs-HGB的論述都是**我們自己的推論**，不是引用論文結論。
- 你專題裡的「PHC-lite」是簡化版（16-dim hidden state、移除spike/STDP/多步transmission等大量機制），**不等於**論文的完整PHCSSM。這點`PHC_HLS_Technical_Report.md` §1.2已自己承認（"PHC-lite ≠ 完整 PHCSSM 模型"），是好事——代表這條防線你已經設好了，答辯時直接引用即可。

**Reviewer會追問的點**：「你的PHC-lite砍掉了spike機制、STDP、多步transmission，那論文ablation顯示的`-5.23pp/-4.37pp`這些數字跟你的PHC-lite還有關係嗎？」——誠實答案是：**沒有直接關係**。論文ablation是在完整PHCSSM上做的，PHC-lite既然已經砍掉這些機制，就不能引用論文ablation數字來argue PHC-lite的設計合理性。這是目前論述鏈最大的一個跳躍，必須在文件裡明確切開「論文層級的合理性」與「PHC-lite簡化後是否仍保留優勢」兩件事。

---

## B. PHC vs HGB 的回溯式技術理由（對應原請求第1點）

**狀態：明確標注為「事後建構的論述」，非原始動機**

### B.1 誠實的原始動機（不要在正式報告隱藏，但要框在對的位置）

你提供的原話：「我當初使用PHC的原因是因為在跟導師聊天的時候意外得知PHCSSM這個東西，剛好在做這個專題就想說用上去試試看。」

這是**機會性選擇（opportunistic adoption）**，不是先有問題分析再選方法。誠實面對這件事比硬掰一個「我們系統性比較了LSTM/GRU/Transformer後選了PHC」更安全——後者一旦被問到比較表/實驗記錄拿不出來，可信度會整段崩掉。CODEX_PROMPT文件§「明確不要做的事」第2點已經把「為什麼一開始決定先試PHC而非LSTM/GRU/Transformer」列為已確認查無依據的B類缺口，這跟你的原話完全吻合——**不要試圖事後生出一個比較分析來填這個洞**。

### B.2 回溯式技術理由（建構於現有HGB實作 + PHC架構特性，標注：推論）

HGB（Histogram-based Gradient Boosting，sklearn `HistGradientBoostingRegressor`，依`03b6_hgb_convergence_closure`等phase的實作）的結構性限制：

1. **無跨時間狀態（no cross-time state）**：HGB對每一筆事件的預測，只看該筆事件當下的特徵向量，樹的split完全基於當前特徵值，**不存在任何遞迴/記憶機制**讓t時刻的預測知道t-1時刻發生過什麼，除非你顯式把lag特徵（如`event_rate_past_50`）工程進特徵向量。
2. **特徵工程的記憶是固定窗口、人工設計的**：`03a6_root_cause_feature_and_contract_closure`裡的動態特徵（`event_rate_past_50`/`spread_change_past`/`imbalance_change_past`）都是**人工指定的固定回看窗口**統計量，不是模型自己學出來該記多久、記什麼。

PHC（即便是簡化的PHC-lite）的對應特性：
- `state_temporal[i] = decay[i]*state_prev[i] + gate_t[i]*candidate_t[i]`：這是一個**可學習的、per-channel衰減率**的遞迴狀態更新——衰減速度（記多久）是訓練出來的，不是人工設定窗口大小。
- `gain_t`（synapse gain）由`encoded`和`state_prev`共同決定，代表**訊息傳遞強度本身依賴歷史狀態**，這是HGB的固定特徵工程做不到的「狀態依賴的動態門控」。

**回溯式論述（明確聲明：這是現在才補建構的理由，不是2026年初選PHC時的真實思考過程）：**

> 若將HGB在L2/order-flow微結構資料上的已知弱點定義為「短期記憶完全依賴人工特徵窗口、無法學習狀態相關的動態衰減」，則PHC-lite的`decay[i]`可學習衰減 + `gain_t`狀態依賴門控，理論上提供了一種**讓模型自己學記憶長度與強度**的機制，而非依賴你在`03a6`手動指定的固定窗口（如`_past_50`）。這是一個有理論依據、但**從未在你的專題裡用實驗直接驗證**的假說——即「PHC-lite的可學習記憶機制 vs HGB的固定窗口特徵工程，在你的資料上誰的捕捉oracle headroom的能力更強」這件事，目前的`04c1`等oracle headroom報告並未做這個對照分析。

**Reviewer會追問**：「你怎麼知道PHC-lite真的學到了比固定窗口更好的記憶模式，而不是過擬合或學到雜訊？」——目前手上沒有直接證據能反駁，這是一個誠實的開放問題，建議在文件裡列為「未來工作」而非假裝已解決。

---

## C. Reset必要性：new_era「死胡同」的具體瓶頸是什麼（對應原請求第2點）

**狀態：已驗證，逐phase溯源**

你的原話：「最一開始reset的原因是因為我覺得new_era已經遇到一個瓶頸，進到死胡同的感覺，但這個具體的瓶頸我沒有很清楚。」——以下是用`new_era/`實際phase記錄重建出來的具體技術瓶頸鏈，並對應到`n2ew_era/00_project_reset_and_data_contract/README.md`的四條官方理由。

### C.1 官方四條理由 ↔ 實際phase證據對照表

| README官方理由 | 對應new_era phase | 具體數字證據 |
|---|---|---|
| 「Positive quote EV was dominated by terminal/session-final markout rather than short-horizon executable edge」 | **Phase 38** `sequential_proxy_inventory_backtest` | 同一份資料：`quote_level_ev_sum_moderate` = 6,348,770（NEW4）/ 2,312,590（test）為正且很大；但inventory-tracked、H=10的`sequential_proxy_pnl_H_mid` = **-123,110**（NEW4）/ **-22,885**（test）——同一資料、換一個更貼近真實成交約束的算法，符號直接翻轉 |
| 「Sequential proxy PnL was negative at short horizons such as H=1, H=5, H=10, and H=20」 | **Phase 38** | 上述-123,110 / -22,885即為H=10的數字；這是reset README這句話的直接出處 |
| 「The residual ML veto experiment used a very narrow baseline-selected universe, with only 820 train samples」 | **Phase 28** `residual_quote_ev_veto` | `FINAL_ARCHITECTURE_REPORT.md`：incremental PnL = 0（veto被拒絕），test集Spearman = **-0.1385**（負相關，代表這個殘差模型的排序方向是反的）|
| 「The bid_only_spread_ge_1 controller... is not a complete strategy」 | **Phase 32/33** `final_falsification_audits` / `corrected_final_falsification_audits` | `matched_random_saturation_report.md`：`bid_only_spread_ge_1`在自己的限制宇宙（`side=bid and spread>=1`）裡選中**100%**自己的候選池（new4: selected=714=universe=714；test: selected=419=universe=419），所以「打贏matched random」這個檢定本質上p=1.0、std=0，**數學上無法證明也無法否證技能**——這條規則只是一個always-true的filter，不是一個可驗證的策略 |

### C.2 「死胡同感」的更完整因果鏈（你主觀感覺到、但沒講清楚的部分）

實際phase inventory顯示，這個瓶頸不是單一事件，而是**從phase 16一路重複到phase 38的同一個模式反覆出現**：

```
16_binary_label_v2      → 過濾器無法打贏 random+1std（verdict: C）
17_regime_auto_gate     → 動態regime gate打不贏靜態combined_filter（verdict: C）
18_queue_aware          → 缺aggressor_side，只能proxy-only驗證，無法完整驗證
20_policy_search        → FPGA友善的policy search：防禦性價值存活，但中等壓力下絕對盈利性失敗
21_stress_robust_policy → oracle在壓力情境下是正的（217,660），但所有causal policy選出來的都是負的（-68,495 / -83,715）
22_signal_generalization_ev → Quote-EV proxy標籤的證據混雜，不夠乾淨
23_root_cause_stress_profitability → 專門開一個phase去解釋「為什麼不可交易的oracle是正的，但causal policy一直無法泛化」——這正是你說的「死胡同感」第一次被正式命名為研究問題
28_residual_quote_ev_veto → ML veto被否決（見上）
32/33_final_falsification_audits → 發現核心統計檢定本身是退化的（p=1.0飽和）+ NEW/NEW1資料集是近乎重複的（6/6價格序列hash相同）
38_sequential_proxy_inventory_backtest → 最後一根稻草：把整條因果鏈用inventory-aware、短horizon的方式重算一次，PnL直接翻負
```

**推論（標注）**：這個「死胡同感」的本質，是**評估指標（session-final markout / quote-level EV）系統性地高估了可執行edge**，而專題在phase 16-37之間一直在「換policy/換regime/換特徵」試圖解決一個本質上是「換評估方法才能解決」的問題——直到phase 38才真正換對評估方法，當下立刻看到PnL翻負，這才是真正的死胡同根因：**用錯誤的尺去量了20多個phase，而不是策略本身一直不夠好**。這個推論本身有一定的詮釋成分，但每個環節都有對應的`*_REPORT.md`數字佐證，不是空想。

### C.3 Reset必要性的論述（可直接用於答辯）

> new_era從phase 16到38，反覆嘗試了過濾器、regime gate、policy search、stress-robust優化、訊號泛化分析、殘差ML veto等至少7種不同的策略修正方向，但全部在「session-final/quote-level markout」這個評估指標下顯示正向結果，卻在phase 38用inventory-tracked、短horizon的sequential proxy重算後全部翻負。更進一步，phase 32/33發現這條研究線唯一一個「打贏」的對照組（matched random saturation test）在統計上是退化的（p=1.0，自身選了100%自己的候選宇宙），且兩個獨立驗證集（NEW/NEW1）被發現是近乎重複的資料。三條證據鏈獨立指向同一個結論：**問題不在於哪個策略最好，而在於評估方法本身系統性地產生了虛假的正向訊號**。這是reset的必要性所在——n2ew_era從一開始就把「causal-fill sequential proxy PnL」設為主要優化目標（而非診斷用的quote-level EV），並在`04b`的candidate eligibility gate中明確要求新策略必須同時通過matched-random檢定與causal-fill sequential檢定才能被視為候選，這正是直接從new_era的失敗中提取出的方法論修正。

---

## D. Oracle方法論評估（對應原請求「B類缺口透過現有實作推斷，oracle缺口現在評估」）

**狀態：已驗證數字 + 推論性建議**

### D.1 兩種Oracle ceiling的落差（已驗證，來自`04c1_new5_oracle_headroom_analysis/NEW5_ORACLE_HEADROOM_REPORT.md`）

| Oracle類型 | 數值 | 假設 |
|---|---|---|
| Unconstrained full-action oracle | 84,517,995 | 無inventory限制、無fill model，直接用未來的`excess_pnl_H20_bid/ask`標籤選最優動作 |
| HGB matched-count oracle | 83,916,630（HGB capture ratio = 0.1865） | 同上，但限制選中數量跟HGB一致 |
| Queue matched-count oracle | 83,101,965（queue capture ratio = 0.1377） | 同上，跟queue policy對齊數量 |
| **Causal-Fill Sequential Oracle**（`conservative_l2_stress` fill model, inventory_cap=1） | **3,823,000 ~ 3,923,130**（依policy對齊方式不同） | 有inventory約束、有fill model、horizon=20，是目前唯一**同時**具備「未來資訊」與「執行可行性」雙重檢驗的oracle |

**落差規模**：unconstrained oracle（~84.5M）是causal-fill sequential oracle（~3.86-3.92M）的**約21.5倍**。目前文件「Required Answers」一節（NEW5_ORACLE_HEADROOM_REPORT.md第35-42行）的headline capture_ratio敘述（HGB=0.1865, queue=0.1377）**全部用的是unconstrained/matched-count分母**，causal-fill sequential oracle只是report裡的一張附屬表格，沒有被當作主要headroom敘事。

### D.2 方法論問題（推論，但有new_era的前車之鑑佐證）

這個落差本身**不是錯誤**——unconstrained oracle本來就是設計成「diagnostic only，不可交易」的上限（report開頭已自己聲明這點）。但有兩個reviewer一定會抓的問題：

1. **單位混淆風險**：如果你的口頭報告/海報只說「HGB capture_ratio = 0.1865」而不解釋分母是84.5M的unconstrained oracle，聽眾很容易誤以為這是相對「可執行真實上限」算出來的比例，但真實可執行上限（causal-fill sequential）只有~3.9M——換算下來HGB實際的policy_pnl（11,439,800，從同一份report的Matched-Count Oracle表）相對causal-fill oracle (~3.86M-3.92M) 的比例會是**遠大於1**（因為HGB的quote-level policy_pnl本身也是用同一套未經inventory約束的markout算的，數字虛高），這正好呼應了new_era整條reset敘事的核心病灶——**用session-final/quote-level markout算出來的「正向PnL」系統性地高估可執行edge**。也就是說：n2ew_era目前oracle headroom報告裡，HGB/queue的`policy_pnl`欄位本身（1.14M/1.56M級別）很可能仍然是quote-level markout口徑，而不是causal-fill sequential口徑——這代表**reset所要修正的問題，在oracle headroom這份報告裡可能還沒有被完全修正**，需要去檢查這兩個policy_pnl數字的計算口徑（這點我尚未在`04c1`目錄裡找到逐行程式碼確認，標注為**未明確說明，建議下一步去`04c1_new5_oracle_headroom_analysis/`原始程式碼確認`policy_pnl`欄位的計算函式用的是哪種markout定義**）。

2. **headline數字選擇的說服力問題**：如果reviewer問「你的真實可執行headroom到底是84.5M還是3.9M？」，目前文件結構容易讓人覺得你在挑對自己有利的數字當headline。

### D.3 建議（推論性建議，需你自己決定是否採納）

- 建議在所有對外報告（海報、口頭報告、底氣文件）裡，**把causal-fill sequential oracle提升為與unconstrained oracle並列的「雙主指標」**，而不是只在附屬表格出現一次。具體寫法：「Unconstrained oracle定義理論天花板（diagnostic only），Causal-fill sequential oracle定義在inventory/fill約束下的真實可執行上限；本專題的策略開發以後者為主要優化目標，前者僅用於說明剩餘的理論空間。」
- 建議優先查清楚D.2第1點提到的「HGB/queue的policy_pnl欄位計算口徑」這個未明確說明的點——如果證實HGB的1.14M policy_pnl也是quote-level口徑而非causal-fill口徑，這是一個需要在文件裡明確揭露的重要caveat，而不是留給reviewer自己發現。
- 是否要「引入更好的指標」：目前causal-fill sequential oracle本身已經是相對嚴格的指標（有inventory_cap=1、有stress fill model），暫時看不出有立即必要再引入第三種oracle定義；更值得投入的是上面兩點的口徑澄清與報告呈現方式調整，而非定義新指標。

---

## 待辦（標注未完成事項，供後續追蹤）

1. 查證`04c1_new5_oracle_headroom_analysis/`原始程式碼，確認`policy_pnl`欄位（HGB=11,439,800 / queue=15,648,100量級）的markout口徑定義。
2. 若確認policy_pnl是quote-level口徑，需要在`N2EW_ERA_EVOLUTION_AND_INHERITANCE.md`與任何對外報告中加註caveat。
3. 本文件§A-D的所有「推論」段落，建議在正式答辯前再過一次教授視角的review，確認措辭沒有把推論講成事實。
