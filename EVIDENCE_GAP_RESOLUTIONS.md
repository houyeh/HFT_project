# 證據缺口裁決（Reviewer 覆核）

日期：2026-09-24
背景：Sonnet 完成階段 1–2 與 Section 4 後回報五項缺口。以下是逐項查證與裁決。
**本文件的裁決優先於 `REPORT_WRITING_PATTERN_FOR_SONNET.md` 附表中與之衝突的部分。**

---

## 裁決摘要

| # | Sonnet 回報 | 裁決 | 動作 |
|---|---|---|---|
| G1 | 參數量應為 1,731 不是 2,132 | **兩個都對，是兩個不同模型** | 全文分開標註，不得混用 |
| G2 | DeepLOB 只有 ~60k，1/67 無證據 | **Sonnet 正確，撤掉 1/67** | 只保留 1/28.1 |
| G3 | 無板上 cycle counter / ILA 延遲量測 | **Sonnet 正確，且這是最重要的一項** | 改寫 Section 8 的驗證等級聲明 |
| G4 | maker fee 15.0 缺交易所依據 | **正確，但 repo 自己已標為假設值，且有敏感度分析** | 照既有 caveat 寫，附敏感度 |
| G5 | 找不到 PROJECT_STORY_RETROSPECTIVE | **檔案存在，Sonnet 搜尋範圍錯誤** | 指路徑即可 |
| G6 | Section 2 文獻未納入證據包 | 預期內（外部文獻不在 repo） | 指向既有引用清單 |
| G7 | Section 5 PHC 逐項差異缺來源 | 來源存在但不在 n2ew_era | 指向構想書與結案報告 |

---

## G1｜參數量：1,731 與 2,132 是兩個不同模型（🔴 最重要）

**Sonnet 抓到了一個真實的問題，但結論只對一半。查證結果：**

| 數值 | 對應模型 | 輸入維度 | 用途 | 來源 |
|---:|---|---|---|---|
| **1,731** | `phc_stp_v2_cosine_lr_dense16` | **16 個 TXF 微結構特徵** | TXF 生產模型，**硬體 frontier 的對象** | `05f_phc_diagonal_ssm/PHC_DIAGONAL_SSM_ABLATION_REPORT.md:30`、`05e_phc_hidden8_ablation/PHC_HIDDEN8_ABLATION_REPORT.md:28` |
| **2,132** | `phc_fi2010_h50_seed0_raw_v3_fair512.pt` | **100×40（FI-2010 的 40 維 LOB）** | FI-2010 公開基準，**DeepLOB 比較的對象** | `08a/REPORT_STAGE1.md:120`、`08a/F1_DEFINITION_RECONCILIATION.md:80`、`08a/V3_COST_AND_FIRST_RESULT_REPORT.md:19` |

差 401 個參數，正是輸入維度從 16 變成 40 造成的。所有 2,132 的出處**無一例外都在 `08a_fi2010_public_benchmark/`**；所有 1,731 的出處都在 TXF 的 05e/05f。

> 這是我先前的權威數字表的錯誤 —— 我把「PHC-lite」當成單一模型列了 2,132，沒有區分兩條評估線。**Sonnet 的質疑是對的，但正確的修法不是把 2,132 換成 1,731，是把兩者分開。**

### 裁決（強制）

- **Section 5（模型）**：主述 TXF 生產模型 **1,731 params, 16 features, hidden 16**；並用一句話交代 FI-2010 版本因輸入維度不同而為 2,132。
- **Section 6、7、8（硬體 frontier / TCN 對照 / 上板）**：一律 **1,731**。硬體跑的是 TXF 模型。
- **Section 9（FI-2010 vs DeepLOB）**：一律 **2,132**。DeepLOB 也是在 FI-2010 上評估，用 1,731 去比是錯的。
- 報告中**第一次**出現參數量時，加一個明確的註腳說明兩者關係。不要讓讀者自己發現前後不一致。

---

## G2｜撤掉 1/67，只保留 1/28.1

查證：`142,435` 這個數字**在整個 n2ew_era 裡不存在**。它只出現在我先前寫的策略文件與 `demo_narrative_and_script_20260808.md`，來源是「公開 zcakhaa/DeepLOB TensorFlow/Keras 參考實作」——**那是第三方對別人論文的重新實作，不是論文本身的數字，也不是我們量的。**

### 裁決（強制）

- **刪除所有 1/67 與 142,435 的表述。**
- 只保留：**DeepLOB 60,000 params（論文 Table III 四捨五入值）vs PHC-lite FI-2010 版 2,132 → 約 1/28.1**。
- 若真的想提參考實作，只能寫成一句限定語：「a public reference implementation reports a larger count; we do not use it in our comparison because we could not verify it against the original paper.」**更建議整句不寫。**

1/28 已經夠強。為了一個查不到來源的數字把「1/28–1/67」寫成區間，是拿整份報告的可信度換一個沒必要的形容詞。

---

## G3｜板上沒有延遲量測 —— 這是 Sonnet 最重要的一次攔截

**這一項如果沒被抓到，會是全篇最嚴重的錯誤。**

事實是：**25.733 µs 是從布局繞線後的時序推導出來的**（2,316 cycles ÷ 90 MHz），**不是在矽片上量到的**。板子提供的是**功能與位元級等價的驗證**（pass/fail LED、4096 筆逐筆比對 0 決策翻轉），**不是獨立的延遲量測**。

### 裁決（強制）

Section 8 必須改寫成這個口徑：

> Latency is **derived from post-route static timing closure** (2,316 cycles at 90 MHz = 25.733 µs), not independently measured on silicon. The board-level validation establishes **functional and bit-level equivalence** between the deployed bitstream and the software reference (27 of 4,096 events differ by at most 1 LSB, producing **zero decision flips**), and confirms that the design operates at the closed timing. **An on-chip cycle counter or ILA-based latency measurement was not implemented; this is stated as a limitation.**

並在 Section 10（Limitations）補一條：「no independent on-silicon latency measurement」。

**不要為了補這個洞去改 RTL。** 加 cycle counter 要重新 P&R + 重燒 + 重新驗證，在申請時程內是不划算的風險；而且誠實聲明一個限制，比補一個匆忙的量測更能得分。

---

## G4｜maker fee：repo 自己已經標成假設值，而且有敏感度分析

查證結果比 Sonnet 說的樂觀。`n2ew_era/01_full_l2_action_universe_builder/FULL_L2_ACTION_UNIVERSE_REPORT.md:50` 已經明寫 `MAKER_FEE = 15.0` 是繼承自先前常數、**current source does not verify tax/exchange/broker schedule**；02b 的 AUDIT 也明確結論「未附來源的假設常數，建議在正式報告中明確標註為假設值」。

而且 `02b` 有現成的**費用敏感度分析**：`results/fee_sensitivity.csv`，公式為 `adjusted = base − (fee_multiplier − 1) × maker_fee × selected_count`，並算得出 breakeven multiplier。

### 裁決

照既有 caveat 寫，不要迴避也不要道歉：

> The cost model uses a maker fee of 15.0 (in the same units as the PnL labels) as an **inherited proxy constant**; the repository does not contain a decomposition into exchange fee, tax, and broker commission, and we do not claim it is exchange-certified. To bound its effect we report a fee-multiplier sensitivity sweep [source: 02b/results/fee_sensitivity.csv] and the break-even multiplier at which the selected policy's excess PnL reaches zero.

**把敏感度分析的結果實際放進去。** 一個標明為假設、但附了敏感度掃描的常數，是可接受的研究實踐；一個沒說明的魔術數字不是。

---

## G5｜PROJECT_STORY_RETROSPECTIVE 的位置

檔案存在：`C:\Users\holbo\00VS_projjects\00_HFT_project\PROJECT_STORY_RETROSPECTIVE.md`（19,867 bytes，2026-09-24）。
Sonnet 應該是只搜了 `n2ew_era/`。**它在 repo 根目錄，不在 n2ew_era 底下。**

注意：這份文件是**敘事骨架**，不是證據來源。Section 10 的否定結果清單要從各 phase 的原始 AUDIT 取數，不要引用 retrospective。

---

## G6｜Section 2 文獻

外部文獻本來就不在 repo，這不是缺口。完整引用清單在 `demo_narrative_and_script_20260808.md` 第 2 段，含五篇的全稱與「它對本文的作用」：

1. Hassaan, Yacoub, Said — *Gated recurrent unit accelerators for financial time series prediction on FPGA*, Eng. Appl. AI, vol. 162, 112534, 2025（KCU105, 50MHz, 0.773W, 13.6 GOPS）
2. Hassaan, Yacoub, Said — *FPGA-Accelerated ESN with Chaos Training*, MAKE, 7(4):160, 2025（83.5MHz, 0.677W）
3. Boutros, Grady, Abbas, Chow — *Build Fast, Trade Fast*, ReConFig 2017（Kintex US, 156MHz, <870ns 端到端）
4. Makinde — *T-KAN for High-Frequency LOB Forecasting*, arXiv:2601.02310, 2026（104,451 params）
5. Hedges — *The Inference-Compute Frontier...*, arXiv:2606.25986, 2026

⚠️ **每一篇的引用細節（頁碼、卷期、數字）在寫進報告前要自己開原文核對一次。** 這些是先前整理的二手紀錄，不是一手查證。FINN-GL 的引用在 `0000_competition_proposal/` 的構想書裡，同樣要核對。

---

## G7｜Section 5 的 PHC → PHC-lite 逐項對照

來源存在，但不在 `n2ew_era/`：

- `0000_competition_proposal/AMD Track-ZeroJitter-構想書_正式版.md`（及其他版本）
- `meeting/FINAL meeting/研究專題結案報告.docx` §3.1.1 有「對照表」
- 消融證據在 `05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md`

⚠️ 注意：結案報告 docx 是舊草稿，**裡面的延遲與 CPU 數字已過期**（83.16 µs 為主軸、把上板寫成未來工作）。**只取它的架構對照表，不要取任何數字。**

---

## 給 Sonnet 的補丁 prompt

```
【共用守則】（照舊貼上）

任務：依 EVIDENCE_GAP_RESOLUTIONS.md 的裁決，修補 EVIDENCE_PACK.md 與
REPORT_OUTLINE.md，並回頭檢查已完成的 Section 4。

必做項目：

1. 參數量分軌
   在 EVIDENCE_PACK 表 4 拆成兩列並加 `evaluation_track` 欄：
   - TXF production: 1,731 params, 16 features, hidden 16
     [source: 05f_phc_diagonal_ssm/PHC_DIAGONAL_SSM_ABLATION_REPORT.md,
              05e_phc_hidden8_ablation/PHC_HIDDEN8_ABLATION_REPORT.md]
   - FI-2010 benchmark: 2,132 params, 100x40 input
     [source: 08a/REPORT_STAGE1.md, 08a/F1_DEFINITION_RECONCILIATION.md]
   在 REPORT_OUTLINE 標註哪些節用哪一個（5/6/7/8 用 1,731；9 用 2,132）。

2. 刪除 142,435 與 1/67
   從 EVIDENCE_PACK 與 OUTLINE 移除。DeepLOB 參數量只保留 60,000
   （論文 Table III），比值只寫 ~1/28.1。

3. 重寫 Section 8 的驗證等級
   把 25.733 µs 的 tier 從任何帶有 "on-board/measured" 的字樣改為
   `routed (derived from post-route timing closure)`。
   板上證據的 tier 是 `on-board (functional + bit-level equivalence only)`。
   在 OUTLINE 的 Section 10 新增一條 limitation:
   "no independent on-silicon latency measurement (no cycle counter / ILA)".

4. 補 fee 敏感度
   讀 02b/results/fee_sensitivity.csv，把 fee multiplier 掃描結果與
   break-even multiplier 加進 EVIDENCE_PACK 表 2，
   並在 OUTLINE Section 4 加一小段處理這個假設。

5. 回頭稽核 Section 4
   確認 993 words 的正文裡沒有任何參數量、沒有誤用 tier、
   且 old_bid_only 案例的三個數字（1.43e9 / 90.3% / 99.87%）
   都有 source 標籤。若有動到，回報 diff。

6. 文獻核對清單
   把上述五篇 + FINN-GL 整理成 REFERENCES_TO_VERIFY.md，
   每篇標「已一手核對 / 待核對」，預設全部是「待核對」。
   不要自己填寫你沒有實際讀到的頁碼或數值。

完成後回報：修改了哪些檔案、參數量分軌後受影響的節、
以及 Section 4 是否需要修訂。
```

---

## 一句話評價

Sonnet 這一輪表現得很好 —— **G3 那一項如果沒被攔下來，會是全篇最嚴重的錯誤**（把推導值講成矽片量測值）。G1 的質疑方向也對，只是結論需要修正成「分軌」而不是「取代」。

繼續讓它用「先回報缺口、不自行填補」的模式跑，這個模式是對的。
