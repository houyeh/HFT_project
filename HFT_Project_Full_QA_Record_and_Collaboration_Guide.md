# HFT / FPGA / ML-PHC 專題：本對話串逐輪問答紀錄與協作模式整理

> 整理日期：2026-06-04  
> 目的：把本對話串中關於 **TXF-only microstructure quote-gating、HGB final model、PHC-lite、Oracle headroom、Dale/STP ablation、HLS/FPGA 方向** 的每個主要問答轉折整理成可交接的 Markdown 紀錄。  
> 誠實限制：部分早期上傳檔案在系統中已過期，因此本檔以目前對話中仍可追溯的文字、檔名、數值、判斷與後續 prompt 為主；若要做逐行 code-level audit，過期檔案需要重新上傳。

---

## A. 全局結論

本對話完成了一條從混亂探索到嚴謹證據鏈的研究線：

1. **資料治理完成**：NEW1–NEW5 去重、角色分配、TXF-only universe、NEW5 lock、loader contract。
2. **label 系統固定**：raw PnL 降為診斷，`excess_pnl_H20` 成為主 target，causal-fill sequential proxy 作為策略層評估。
3. **non-ML baseline 被釐清**：old bid-only 是 spread artifact；queue imbalance rule 是強 baseline。
4. **HGB action-value model 成為 official final model**：在 04C NEW5 one-shot final holdout 打贏 queue rule、matched random、causal-fill proxy。
5. **PHC-lite 成為 advanced model success**：在 val / NEW3 / NEW4 以及 NEW5 post-hoc diagnostic 上均贏 HGB / queue，但不是 fresh final holdout claim。
6. **Oracle headroom 建立尺度感**：HGB、PHC 都遠離 Oracle，但 PHC 比 HGB 更接近 same-universe oracle。
7. **Dale ablation 完成 negative/neutral result**：實作正確但不改善 PHC-lite v1。
8. **STP-like dynamic gain 成為 promising candidate**：80 epoch patch 後 action/oracle 超過 v1，但 causal-fill NEW4 與 convergence 尚未 clean closed。
9. **後續方向明確**：Oracle error decomposition、ranking head、uncertainty gating、streaming dense PHC、STP 120 epoch closure、HLS queue baseline、PHC fixed-point/streaming equivalence。

---

# B. 逐組問答紀錄


## 1. residual EV veto / logistic 模型沒有行為

### 你的問題
你貼出 `residual_ev_veto__logistic_l2_balanced__thr0.83338837` 結果：

```text
veto_count = 0
veto_rate = 0
baseline_pnl = 1,218,095
after_veto_pnl = 1,218,095
incremental_vs_baseline = 0
```

你問這是否代表模型沒用，或只是訓練太爛。

### 我的回答
我判斷這**不能證明 ML 路線不行**，只能說當前 residual veto 設計沒有產生任何策略行為。可能原因包括：

- residual universe 太小；
- threshold 太高；
- label / target 不合適；
- baseline 已篩掉許多機會；
- feature 不足；
- 模型太弱；
- 沒有 loss / convergence / confusion / stability diagnostics。

### 轉折
你提出「資料被前面策略篩後太少」與「baseline 可能擋掉機會」這兩個關鍵問題，後來推動我們改做 full action universe 與 oracle headroom。

---

## 2. 你質疑訓練時間太短，是否模型太弱

### 你的問題
你問二十幾分鐘訓練完是否不合理，是否只是因為模型太簡單、沒有 GPU、沒有真正學到。

### 我的回答
我回答：

- logistic / ridge / HGB 這種 tabular model 在 CPU 上跑很快是正常的。
- 但快不代表訓練成功。
- 必須補：
  - loss / staged score
  - convergence flag
  - train vs val metrics
  - confusion / decision matrix
  - feature importance
  - sample stability
  - exact matched random
  - no_quote ablation

### 產生任務
後來形成 03B.5 training diagnostics / implementation sanity patch。

---

## 3. 你問是否在 28_ 上擴充、該加什麼模型

### 你的問題
你問下一步是不是直接在既有 28_ 上擴充，嘗試 baseline + ML，並參考論文避免打牆。

### 我的回答
我建議不要直接堆大模型，而是先建立可辯護的 action-value framework：

- full bid / ask / no_quote action universe
- target = `excess_pnl_H20`
- validation-only threshold selection
- frozen OOS
- matched random
- causal-fill sequential proxy
- simple ML first: ridge / logistic / shallow HGB

### 轉折
這成為後來 03B / 04B simple ML action-value 的主線。

---

## 4. 你問 GPU 筆電是否值得用

### 你的問題
你問 16GB RAM + RTX 4050 Laptop GPU 是否比 32GB 無 GPU 快很多，以及搬資料 / Python 環境是否複雜。

### 我的回答
我回答：

- 對 sklearn HGB / logistic，GPU 幫助有限。
- 對 TCN / PHC / neural sequence model，GPU 有幫助。
- 搬環境時要固定：
  - Python version
  - requirements
  - path config
  - random seeds
  - split / threshold / feature contract
  - output manifest

### 產生任務
我給了 PM / coding agent 的搬移與訓練 prompt，強調不能在新機器上偷偷改 protocol。

---

## 5. 你要求 PM / coding agent 完整指令

### 你的問題
你要給 PM 的所有指令，包含 coding agent 產出成果、程式碼與搬到另一台電腦的完整步驟。

### 我的回答
我提供了一份完整 prompt，要求：

- 不碰 NEW5；
- 不用 test / OOS tuning；
- 不直接上 TCN / 大模型；
- 先補訓練診斷；
- 保存 model artifact / config / metrics；
- 用 deterministic paths 和 seeds；
- 每個結果都有 report / CSV / figure。

### 重要坑
我特別列出：

- 不要用 residual 820 rows 當完整訓練；
- 不要只報 PnL；
- 不要把 proxy 當 true replay；
- 不要讓 coding agent 為了好看改 success criteria。

---

## 6. 你問資料少與 opportunity cost

### 你的問題
你指出有毒大單本來就少，資料一直很少；前面策略可能擋掉很多賺錢機會，問能否驗證。

### 我的回答
我說可以，而且應該驗證：

- full valid universe vs baseline-selected universe；
- missed positive EV actions；
- oracle opportunity；
- matched-count oracle；
- matched-count+side oracle；
- model capture ratio。

### 轉折
這成為後來 dense Oracle 與 PHC-window Oracle 的思想來源。

---

## 7. 多批 results / 圖表 / code review 問答

### 你的問題
你多次上傳圖、CSV、report、code，問：

- 這些結果怎麼看？
- 有符合要求嗎？
- 程式有好好寫嗎？
- 是否需要看原始碼？

### 我的回答模式
每次都用以下標準：

1. **protocol 是否正確**
   - train / val / OOS / NEW5 分工
   - no NEW5 tuning
   - no OOS selection

2. **label 是否正確**
   - raw vs excess
   - label decomposition
   - target 是否是 `excess_pnl_H20`

3. **模型是否可採用**
   - action-level PnL
   - matched random
   - causal-fill sequential
   - convergence
   - seed stability
   - feature importance / artifact check

4. **claim 是否安全**
   - official final vs diagnostic
   - proxy vs true replay
   - fresh holdout vs post-hoc

### 固定分類
我常用這些 status：

- Pass
- Partial pass
- Patch required
- Blocked by convergence
- Signal exists but no strategy gain
- Diagnostic only
- Candidate but not clean adopt
- Not adopted

---

## 8. non-ML 策略與 FPGA claim

### 你的問題
你問是否已同意 non-ML 策略放到 FPGA 上績效會變好，以及之前驗證是否做完。

### 我的回答
我澄清：

- 沒有證明 FPGA 會讓 PnL 變好。
- 只能說低 latency 理論上可能改善 fill / adverse selection。
- 目前缺 CPU vs FPGA latency-sensitive fill simulation。
- 所以 **FPGA improves PnL 是 forbidden claim**。

### 可說 claim
可以說：

- queue rule 很 HLS-friendly；
- FPGA 可做 functional equivalence / latency / resource prototype；
- 不能說實際獲利改善。

---

## 9. non-ML 策略來源與實作

### 你的問題
你要求只根據本專案前面對話介紹 non-ML 策略來源、執行、數據。

### 我的回答
我整理：

- **old bid-only**：
  - raw PnL 看似強；
  - excess 後負；
  - 被判定為 half-spread artifact。

- **queue imbalance rule**：
  - `queue_imbalance = (bid_sz_1 - ask_sz_1)/(bid_sz_1 + ask_sz_1)`
  - `> +0.1` 掛 bid
  - `< -0.1` 掛 ask
  - 否則 no_quote。
  - causal, simple, HLS-friendly。

### caveat
它仍不是真實策略，只是 proxy evaluation 下的 baseline。

---

## 10. 你質疑 queue rule 太簡單是否有偷看未來

### 你的問題
你問簡單 queue rule 為何可以強，是否有 delay / 偷看未來 / simulator 問題。

### 我的回答
我說這是合理質疑，必須檢查：

- causal feature only；
- label not input；
- bid/ask pairing assert；
- delay sensitivity；
- fee sensitivity；
- queue-aware stress；
- product universe；
- matched random；
- oracle headroom；
- causal-fill proxy。

### 後續
這推動了 02F / 04A / 04B / 04C 的嚴謹化。

---


# C. Phase 03B：Simple ML action-value model

## 11. 03B 訓練診斷不足

### 你的問題
你上傳程式碼，問模型訓練過程是否好、是否收斂、是否有 loss-epoch / confusion matrix。

### 我的回答
我判定：

- 流程大致正確：
  - target = `excess_pnl_H20`
  - allowed features
  - validation-only selection
  - frozen OOS
- 但 training diagnostics 不足：
  - 無完整 loss / staged score
  - 無 convergence log
  - 無 confusion / oracle action matrix
  - 無 no_quote ablation
  - matched random 是 approximation
  - bid/ask pairing assert 不完整

### 產生任務
03B.5 patch：

- bid/ask pairing hard assert
- training diagnostics
- confusion / decision matrix
- no_quote ablation
- exact matched random
- train sample stability
- HGB feature importance

---

## 12. 03B.5 結果

### 你的問題
你給 03B.5 成果，問是否符合。

### 我的回答
我判定大多數 sanity pass：

- bid/ask pairing pass
- exact sampled matched random pass
- no_quote ablation 顯示 no_quote 不是主因
- train sample stability pass
- HGB feature importance 合理

但最大 blocker：

- HGB hit max_iter = 80
- convergence 尚未關閉

### 結論
ML 有 signal，但沒有穩定打贏 queue rule；還要做 03B.6 convergence closure。

---

## 13. 03B.6 HGB iteration sweep

### 你的問題
你貼 validation PnL by iteration 圖。

### 我的回答
我判定：

- 80 iterations 確實偏早；
- 80 → 320 validation 小幅改善；
- 但 OOS 不改善，甚至略差；
- causal-fill sequential 也沒改善。

### 結論
HGB convergence blocker 已足以決策：更多 iteration 不能 rescue ML。  
狀態是：

```text
hgb_convergence_closed_but_no_strategy_gain
```

---

# D. Phase 04A：資料治理與 TXF-only protocol

## 14. 你整理 NEW1–NEW5 physical sources

### 你的問題
你提供去重後資料：

```text
58 unique CSV
22 sessions
32.58M raw rows
NEW1+NEW2 development
NEW3+NEW4 OOS
NEW5 fresh holdout
```

你問如何處理、之前換 label 是否正常。

### 我的回答
我說：

- 探索期換 label 正常；
- 現在要 freeze protocol。
- 資料角色：
  - NEW1+NEW2 = development
  - NEW3+NEW4 = used diagnostic OOS
  - NEW5 = final locked holdout
- label hierarchy：
  - raw diagnostic
  - excess H20 main signal
  - causal-fill sequential strategy layer

### 產生任務
04A dataset registry and protocol freeze。

---

## 15. 04A 結果是否嚴謹

### 我的回答
判定為 partial pass：

通過：

- SHA256 exact dedup
- NEW / NEW_ALL 排除
- NEW1 duplicate pair handled
- dataset role contract
- NEW5 no-PnL lock

不足：

- registry 欄位不完整
- NEW5 only sampled 20k
- product consistency insufficient
- info_time parser validation needed
- row count validation needed
- machine-readable checklist missing

### 產生任務
04A.1 metadata / product / parser patch。

---

## 16. 04A.1 結果

### 我的回答
04A.1 完成了 expanded registry、full NEW5 metadata、row count validation，但發現：

- raw files 是 mixed-product stream；
- product filter 必須 freeze；
- NEW5 20260518 TXF rows 很少；
- malformed info_time rows 需 loader-level drop。

### 結論
不是 clean freeze，狀態：

```text
blocked_by_product_mismatch_or_parser_issue
```

### 產生任務
04A.2 product filter and loader contract closure。

---

## 17. 04A.2 結果

### 我的回答
我判定通過。

固定：

- final product universe = TXF family only
- regex = `^TXF[A-Z0-9]*$`
- MXF/TMF diagnostic only unless normalized
- NEW5 20260128 usable
- NEW5 20260518 metadata-only
- malformed info_time drop and log

### 重要轉折
前面 broad-product 結果都降級為 exploratory / diagnostic，需要在 TXF-only 下重跑。

---

# E. Phase 04B：TXF-only final development rerun

## 18. 你要求 04B prompt

### 我的回答
我給出完整 coding agent prompt：

- build TXF-only action universe
- label decomposition
- non-ML baselines
- matched random
- causal-fill sequential
- simple ML
- ML diagnostics
- final candidate decision

### 禁令
No NEW5、No PHC、No TCN、No mixed product、No raw final target。

---

## 19. 04B 結果

### 我的回答
我判定：

- TXF-only universe 成功；
- label decomposition pass；
- old bid-only excess 下負；
- queue rule 穩定正；
- HGB action-level 在 TXF-only 下大幅打贏 queue rule。

但 HGB 還缺：

- matched random；
- causal-fill sequential；
- convergence closure。

### 產生任務
04B.1 ML candidate completion patch。

---

## 20. 04B.1 結果

### 我的回答
我判定：

> HGB 通過所有 development / diagnostic OOS gate，可以進 NEW5。

關鍵設定：

```text
HGB regressor
max_iter = 120
actual_n_iter = 98
threshold = 0.22419440373778343
selection_split = validation
```

通過：

- matched random
- causal-fill sequential
- convergence closure
- val / NEW3 / NEW4 beat queue

---

# F. Phase 04C：NEW5 one-shot final holdout

## 21. 你要求 04C prompt

### 我的回答
我給出 NEW5 one-shot prompt：

- load frozen contracts
- build NEW5 TXF-only action universe
- label decomposition audit
- frozen policy evaluation
- matched random
- causal-fill sequential
- claim safety table

### Hard rule
跑完 NEW5 後不准再調。

---

## 22. 04C 結果

### 我的回答
我判定 04C 通過。

NEW5 20260128 TXFB6：

| Policy | Action-level excess | Causal-fill |
|---|---:|---:|
| old bid-only | -11.459M | -553k |
| queue rule | +11.440M | +723k |
| HGB | +15.648M | +894k |

### 結論
Official final model = HGB action-value model。

### caveat
不是 true exchange replay，不是真實獲利保證，不證明 FPGA improves PnL。

---

# G. PHC-lite

## 23. 你問 PHC 是什麼、怎麼實作

### 我的回答
我解釋 PHC：

- recurrent temporal state
- within-timestep lateral / feedback interaction
- Multi-Transmission Loop
- full PHCSSM 包含 ALIF / STP / STDP / Dale 等生物機制

本專題先做 PHC-lite：

```text
input features
→ encoder
→ recurrent state update
→ multi-transmission loop
→ bid/ask action-value readout
```

### 禁令
PHC 不能使用 NEW5 作 selection；NEW5 只能 post-hoc。

---

## 24. 你問 PHC 實作是否符合

### 我的回答
一開始因缺 `models/phc_lite.py`，我說不能確認核心架構。

我要求你提供：

- model file
- training curve
- OOS eval
- matched random
- causal-fill
- seed stability
- final report

---

## 25. PHC-lite v1 結果

### 我的回答
我確認 PHC-lite v1 符合：

- encoder
- temporal recurrent state
- learned decay/gate
- MTL loop
- bid/ask readout

結果：

- beats queue / HGB on val / NEW3 / NEW4
- matched random pass
- causal-fill pass
- seed stability pass
- parameter_count = 1203

### 結論
PHC-lite v1 是 successful advanced sequence-model feasibility。

### caveat
不是 full PHCSSM，不是 Mamba，不是 fresh final holdout model。

---

## 26. PHC NEW5 post-hoc diagnostic

### 你的問題
PHC 沒碰 NEW5，為什麼不能用 NEW5？

### 我的回答
可以用，但只能 post-hoc diagnostic，因 NEW5 已被 04C 使用。

### 你提供結果後，我判定
PHC 在 PHC window universe 上：

| Policy | Action PnL | Causal-fill |
|---|---:|---:|
| queue | 361k | 40.9k |
| HGB | 487k | 59.7k |
| PHC | 583k | 68.7k |

PHC post-hoc 也贏。

### caveat
PHC universe 是 stride-32 windows，不是 dense event-level。

---

# H. Oracle headroom

## 27. Dense Oracle

### 你的問題
你問是否補上帝視角。

### 我的回答
強烈建議補：

- full action oracle
- matched-count oracle
- matched-count+side oracle
- causal-fill oracle

### 你提供結果後，我判定

Dense NEW5：

- full oracle = 84.52M
- HGB = 15.65M
- queue = 11.44M
- HGB matched-count capture ≈ 18.65%
- queue ≈ 13.77%
- sequential HGB capture ≈ 23%

### 解讀
HGB 比 queue 更接近 oracle，但仍有大量 headroom。

---

## 28. PHC-window Oracle

### 你的問題
Oracle 與 PHC 怎麼比。

### 我的回答
不能用 dense Oracle 直接比 PHC，需 PHC-window Oracle。

### 你提供結果後，我判定

PHC-window：

| Policy | PnL | capture |
|---|---:|---:|
| queue | 361k | 13.87% |
| HGB | 487k | 18.53% |
| PHC | 583k | 22.23% |

Sequential capture：

| Policy | capture |
|---|---:|
| queue | 11.4% |
| HGB | 16.5% |
| PHC | 19.0% |

### 解讀
PHC 更接近 same-universe oracle，但仍離 oracle 很遠。

---

# I. Dale’s Law ablation

## 29. 你問是否封存 PHC 並加 Dale / biological features

### 我的回答
建議：

1. 先封存 PHC-lite v1。
2. Dale / ALIF / STP 只能 controlled ablation。
3. 不要一次加滿。
4. Dale 不一定適合金融。

### PHC vs Mamba
我回答：

- PHC-lite 有 selective recurrent / SSM-like flavor。
- 但不是 Mamba。
- Mamba 是 selective SSM / selective scan。
- PHC-lite 是 gated recurrent state + MTL loop。

---

## 30. Dale ablation 結果

### 我的回答
Dale：

- implementation initially looked valid
- beats HGB / queue
- matched random pass
- causal-fill positive
- oracle capture > HGB/queue
- 但略輸 PHC-lite v1

### 補 code 後
我確認：

```python
W_eff = softplus(raw_weight) * dale_sign.view(1, -1)
message = linear(z, W_eff, bias)
```

方向正確，因 column 對應 presynaptic outgoing weights。

### 結論
Dale 是 valid negative / neutral ablation，不採用。

---

# J. STP-like dynamic gain

## 31. STP 初版 50 epoch

### 我的回答
STP 實作合理：

```python
gain_t = sigmoid(W_g encoded_t + U_g state_{t-1})
message = gain_t * silu(synapse(z))
```

結果：

- beats HGB / queue
- matched random pass
- causal-fill positive
- seed stable
- 但略輸 PHC-lite v1
- convergence 不完整

### 產生 patch
要求：

- common_05c2 audit
- extended epoch sanity
- gain behavior diagnostic
- session paired comparison
- final STP decision

---

## 32. STP patch 後結果

### 我的回答
更新判斷：

> STP 從 negative 變成 promising candidate。

80 epoch：

- action PnL 超過 v1
- oracle capture 超過 v1
- val / NEW3 causal-fill 超過 v1
- NEW4 causal-fill 小輸 v1
- 仍 hit max_epoch

### 結論
STP 尚未採用，需要 120 epoch closure。

---

# K. Oracle gap reduction

## 33. 你問 70% gap 是否正常

### 我的回答
我說：

> 距離 Oracle 70%+ 正常，因 Oracle 偷看未來。

不切實際追 80–90% capture。合理目標：

- action capture 22% → 24–27%
- sequential capture 19% → 21–24%

### 推薦方向
- Oracle error decomposition
- gap by regime
- ranking / pairwise head
- uncertainty gating
- streaming dense PHC
- STP 120 closure
- more real order-flow features

---

## 34. 你要求 coding agent prompt

### 我的回答
我給 Phase 05D prompt：

1. Oracle error decomposition
2. Oracle gap by market regime
3. pairwise / ranking action head
4. uncertainty / confidence gating
5. streaming dense PHC inference
6. STP 120-epoch closure
7. final oracle gap reduction summary

---

# L. HLS / FPGA / PYNQ-Z2

## 35. 你問何時適合上 HLS、PYNQ-Z2 可否

### 我的回答
我建議分三層：

1. **HLS queue rule baseline**
   - 現在就可以做。
   - 測 functional equivalence / latency / II / resource。

2. **HGB HLS**
   - 不優先，tree ensemble 對 FPGA 不漂亮。

3. **PHC-lite HLS**
   - 最有研究感，但先做：
     - streaming inference equivalence
     - fixed-point emulation
     - action agreement
     - no material PnL drift

### PYNQ-Z2
可做 prototype / demo，不可宣稱 production HFT。

---

# M. 報告與圖表

## 36. 你要求輸出 .md / HTML 報告

### 我的回答
我生成了：

- `TXF_HGB_PHC_Final_Evidence_Report.md`
- `TXF_HGB_PHC_Final_Evidence_Report.html`
- `TXF_HGB_PHC_Figure_Placement_Checklist.md`

內容包含：

- data governance
- label definitions
- feature definitions
- baselines
- HGB
- PHC
- Oracle
- Dale/STP
- weaknesses
- claim boundary
- figure placement

### 注意
若要插入真實圖表，需要你再提供對應圖檔或 CSV。

---

# N. 我們的協作模式

## 37. 你的習慣

你通常：

1. 貼結果 / code / report。
2. 問「這代表什麼」「有符合嗎」「要補什麼」「要看 code 嗎」。
3. 會嚴格挑戰：
   - 是否資料太少？
   - 是否偷看未來？
   - 是否訓練太爛？
   - 是否只是 validation overfit？
   - 是否可以報告？
4. 喜歡直接拿 PM / coding agent prompt。
5. 希望我不要講好聽話，要像嚴格教授審查。

## 38. 我的回覆模式

固定流程：

1. 先給硬判斷：
   - pass / partial / blocked / not adopted / candidate。
2. 拆數字：
   - action-level
   - matched random
   - causal-fill
   - oracle capture
   - seed stability
   - convergence。
3. 說能不能採用。
4. 說能講什麼、不能講什麼。
5. 給下一步 prompt。

## 39. 共同形成的標準

任何模型要採用，需過：

- data protocol
- label decomposition
- validation-only selection
- OOS diagnostic
- matched random
- causal-fill sequential
- convergence
- seed stability
- oracle capture
- session robustness
- claim boundary

---

# O. 避免踩坑清單

## 40. 資料坑
- 不要再把 NEW5 當 fresh holdout。
- 不要混 TXF / MXF / TMF 做 final pooled PnL。
- 不要把 NEW_ALL / NEW alias 當新資料。
- 不要 silent parse malformed `info_time`。
- 不要用 OOS 選模型。

## 41. Label 坑
- 不要用 raw PnL 當主 target。
- 不要把 Oracle 當策略。
- 不要把 toxic label 當 final objective。
- 不要忘記 raw = half-spread + excess。

## 42. 模型坑
- 不要只看 final PnL。
- 不要忽略 loss curve。
- 不要 hit max_epoch 還直接採用。
- 不要只用 validation 成功宣稱成功。
- 不要缺 matched random / causal-fill。
- 不要沒有 seed stability。

## 43. Oracle 坑
- dense Oracle 不可直接和 PHC window 比。
- Oracle 使用 future labels。
- Oracle 不能用來調 NEW5。
- capture ratio 要 same universe。

## 44. PHC / biological module 坑
- PHC-lite 不是 full PHCSSM。
- Dale-like 不是 full biological Dale。
- STP-like 不是 full biological STP。
- 不要一次加所有 biological features。
- 每個 ablation 都要和 frozen v1 比。

## 45. HLS / FPGA 坑
- 不要宣稱 FPGA improves PnL。
- PYNQ-Z2 只是 prototype。
- PHC HLS 前要 streaming + fixed-point equivalence。
- Python window inference 不是 hardware streaming inference。

---

# P. 目前最重要 status table

| Component | Status | Adopted? |
|---|---|---:|
| old bid-only | rejected artifact | No |
| queue rule | robust baseline | fallback yes |
| HGB | official 04C final model | Yes |
| PHC-lite v1 | best frozen sequence diagnostic | Yes as advanced |
| dense Oracle | headroom diagnostic | No |
| PHC-window Oracle | headroom diagnostic | No |
| Dale-only | valid negative / neutral | No |
| STP 50 epoch | undertrained | No |
| STP 80 epoch | promising candidate | Not yet |
| STP 120 epoch | pending | TBD |
| ranking head | proposed | TBD |
| uncertainty gating | proposed | TBD |
| streaming dense PHC | proposed | TBD |
| HLS queue rule | ready prototype | Yes |
| PHC HLS | later after equivalence | TBD |

---

# Q. 下一步建議順序

1. Freeze current artifacts。
2. STP 120 epoch convergence closure。
3. Oracle error decomposition。
4. Oracle gap by regime。
5. Ranking / pairwise action head。
6. Uncertainty gating。
7. Streaming dense PHC inference。
8. HLS queue rule baseline。
9. PHC fixed-point / streaming equivalence。
10. 若要 formal PHC final claim，收 NEW6 fresh holdout。

---

# R. 最後總結

這串對話的主線可以濃縮成：

```text
資料治理與 TXF-only protocol
→ raw bid-only artifact 被推翻
→ queue imbalance rule 成為強 baseline
→ HGB action-value model 通過 official NEW5 final holdout
→ PHC-lite v1 顯示 sequence modeling 明顯更強
→ Oracle headroom 證明 PHC 更接近 theoretical upper bound
→ Dale 是 valid negative ablation
→ STP-like gain 是 promising but not yet adopted candidate
→ 下一步應做 oracle gap reduction 與 streaming/HLS readiness
```

最大的優點是：現在的研究敘事不是「看哪個 PnL 高」，而是完整包含：

- data governance
- label correctness
- baseline
- matched random
- causal-fill
- convergence
- seed stability
- oracle headroom
- claim boundary
- HLS feasibility roadmap

這樣才是能對嚴格教授交代的版本。
