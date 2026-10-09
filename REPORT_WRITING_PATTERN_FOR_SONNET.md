# 成果報告撰寫 Pattern（給 Claude Sonnet 的四階段 prompt）

用途：申請研究所用的**英文技術報告**（8–12 頁，arXiv 風格）
時程：10 月中前要有能交給教授的草稿（推薦信素材）
基礎文件：`PROJECT_STORY_RETROSPECTIVE.md`（敘事骨架）

---

## 使用方式

**一次貼一個 prompt，做完一階段再進下一階段。**
不要把四個合成一個丟出去——那會產出看起來完整但數字錯誤的東西，而這份報告的全部價值就在數字是對的。

每個 prompt 開頭都先貼這段**共用守則**：

```
【共用守則 — 每次都適用】
1. 這是一份要給美國研究所招生委員與指導教授看的技術報告。讀者是 ECE /
   計算機架構背景，不是金融背景。
2. 🔴 絕對不要發明任何數字。所有數值必須來自 repo 內既有的 report/CSV，
   並在草稿中以 `[來源: <相對路徑>]` 標註。查不到就寫 `[MISSING: 需補]`，
   不要用「約」「大約」「估計」矇混。
3. 🔴 絕對不要把這個專案描述成 HFT 系統 / 高頻交易系統 / trading system。
   正確定位：a deterministic inference core, with limit-order-book quoting
   used as the driving benchmark。委託簿是標尺不是產品。
4. 🔴 不要美化。否定結果、失敗的嘗試、未解的根因，全部要留。
   這份報告的說服力來自誠實，不來自成績。
5. 區分驗證等級，每個數字都要標：estimate (csynth) / simulated (cosim) /
   routed (post-P&R) / on-board (實體板)。混用等級是最嚴重的錯誤。
6. 不要修改 repo 內任何既有檔案，只新增。
```

---

# 階段 1／證據蒐集

> 目標：產出一份 `EVIDENCE_PACK.md`，把散在 68 個 phase 裡的數字集中、標好來源。
> **這一階段不寫任何報告文字。**

```
【共用守則】（貼上面那段）

任務：掃描 n2ew_era/ 底下所有 phase 目錄，產出 EVIDENCE_PACK.md，
作為後續撰寫英文技術報告的唯一數據來源。

工作方式：
- 只讀既有的 .md report、AUDIT 檔、results/*.csv。
- 不要重跑任何實驗、不要重新訓練、不要跑 Vivado/HLS。
- NEW5 相關資料只能讀既有結果報告，不得重新評估。

請產出 EVIDENCE_PACK.md，包含以下九張表。每一列都要有 `source_path` 欄。

表 1｜資料集與切分
  每個 split (train/val/test/NEW3/NEW4/NEW5) 的 session 數、事件數、
  時間範圍、去重驗證結果。來源含 02e 的 NEW3 validity report、
  04a 的 registry freeze。

表 2｜非 ML 基準與代理環境失效證據
  old_bid_only 的 half-spread vs drift 分解（四個 split）、
  02e 的最終分類字串、以及 side_by_queue_imbalance 的 matched-random p 值。

表 3｜NEW5 一次性 holdout
  04c 的九項凍結契約逐項 expected/actual/pass，
  以及四個策略的 sequential_excess_pnl。

表 4｜模型與消融
  PHC-lite 參數量、hidden 維度、量化格式；
  05a 的四組 ablation（fixed_decay / memory_off / reverse_sequence /
  shuffle_timesteps）在 val/new3/new4 的 PnL delta。

表 5｜延遲–資源–價值 frontier（本報告的核心表）
  06F v8、06L unroll16、06U V1@90、06U V1@100、06U V2 hidden8、
  06U V3 diagonal，以及超標的 V3 unroll16。
  每列欄位：latency_us, cycles, freq_MHz, routed_WNS_ns, routed_TNS_ns,
  LUT%, FF%, DSP%, BRAM%, verification_tier, model_source,
  val/NEW3/NEW4 PnL, source_path。

表 6｜TCN 前饋對照組
  07a 的 TCN vs PHC PnL delta（3 seeds、matched-random p）、
  07c2/07c2b 的 unroll scaling（16,510 → 7,438 cycles）、
  07c2d 的 routed 結果（latency cycles, interval, WNS, LUT/FF/DSP/BRAM,
  1008-event cosim readout max diff 與 RMSE）。
  ⚠️ 07c2d 報告自己聲明 TCN streaming latency 與 PHC single-step
  不是同一個 latency 物件——這句話要原文抄進 EVIDENCE_PACK，
  之後撰寫時必須加註。

表 7｜上板驗證
  06s2 的 JTAG 燒錄紀錄、LED/HDMI 狀態、
  1-LSB 漂移統計（總筆數/漂移筆數/最大 LSB/方向分布/決策翻轉數/最小 margin）、
  S_DRAIN 實驗的負結果。

表 8｜FI-2010 公開基準
  08a 的 weighted-F1 與 macro-F1（PHC 3 seeds、hidden64、TCN small/scratch）、
  DeepLOB Table II 三個 horizon 的 accuracy/precision/recall/F1、
  以及 Accuracy==Recall 的驗證紀錄、weighted recall vs accuracy 的最大差值。

表 9｜延遲的金融意義
  06p/06q 的 resolution_limited 結論原文、
  06r 的 LOBSTER stale exposure（25.733/83.16/262.6 µs 三點）與資料集邊界聲明、
  06t 的 C(k) 每筆決策 PnL 衰減與其「hypothetical scale estimate」邊界聲明。

最後加兩節：

A｜🔴 缺口清單
  列出所有我要求但你在 repo 裡找不到的數字，標 [MISSING]。
  特別確認以下四項是否存在，不存在就明確說沒有：
  - 功耗（report_power 的輸出）
  - excess_pnl_H20 的單位定義（點數 / 跳動點 / 台幣）
  - 板上實測延遲（相對於 routed 估計）
  - maker fee 假設的推導依據

B｜⚠️ 互相矛盾的紀錄
  列出任何兩份報告對同一件事說法不一致的地方，附雙方路徑與原文。

完成後回報：EVIDENCE_PACK.md 的路徑、九張表各自的列數、
[MISSING] 項目數、以及矛盾項目數。
```

---

# 階段 2／骨架與資料綁定

> 目標：確定章節結構，並把階段 1 的每個數字指派到具體章節。
> **這一階段仍然不寫正文。**

```
【共用守則】（貼上）

任務：依照以下骨架，產出 REPORT_OUTLINE.md。
每一節都要列出：(a) 本節要主張什麼 (b) 用 EVIDENCE_PACK 的哪幾列支撐
(c) 需要哪張圖 (d) 預估頁數。

⚠️ 這一階段不要寫正文段落，只做結構與資料綁定。

報告骨架（英文，8–12 頁）：

Title (working):
  "What Does a Microsecond Cost? A Task-Level Design-Space Exploration of
   Small Recurrent Inference Cores on a Resource-Constrained FPGA"

1. Introduction (1 p)
   - 問題：固定大小的 FPGA 上，降低推論延遲要付出多少任務層級代價？
   - 明確聲明：this is NOT an HFT system; LOB quoting is the benchmark,
     not the product.
   - 三項貢獻條列。

2. Background and Related Work (1 p)
   - FPGA 上的金融/遞迴推論既有工作（GRU accelerators、Build Fast Trade Fast、
     FINN-GL、DeepLOB、T-KAN），每篇只寫「它對本文的作用」，不寫摘要。
   - 明確畫出差異線：既有工作停在軟體延遲量測或單一設計點；
     本文提供多點 routed frontier 並以任務價值為縱軸。

3. Methodology: The Feature-to-FPGA Pipeline (1.5 p)
   - 五層：causal features → model → fixed-point → manual HLS → P&R → board
   - 每層的固定驗證標準（這是可替換模型、不可替換流程的設計）
   - 驗證等級定義表：estimate / simulated / routed / on-board

4. Experimental Protocol and Its Enforcement (1.5 p)  ← 這節是本報告的差異化
   - 五路資料切分與 NEW5 的一次性鎖
   - 九項凍結契約驗證（放表）
   - matched-random permutation test
   - 🔴 重點案例：old_bid_only 在天真 markout 下 1.43e9 PnL / 90.3% 勝率，
     分解後 99.87% 來自 half-spread；在 NEW5 序列化模擬下為 −553,365。
     這個案例要當成「為什麼衡量方式決定一切」的核心敘事。

5. Model: PHC-lite (1 p)
   - 從 PHC (arXiv:2604.01295) 的簡化：保留什麼、移除什麼、每項的工程理由
   - 2,132 參數、16 維狀態、ap_fixed<24,12> / <40,16>
   - Ablation：時序順序與狀態記憶必要；learned decay 不必要

6. Latency–Resource–Value Frontier (2 p)  ← 核心結果
   - 表 5 全表
   - Figure 1：frontier（橫軸 latency µs log、縱軸 NEW3 excess PnL）
   - 三個發現：前 3.72× 免費；之後每級有標價；3.83µs 撞 DSP 452>220

7. Why Recurrence? An O(1)-State Argument (1.5 p)  ← 最像研究的一節
   - 誠實開頭：前饋 TCN 在 PnL 上打平或更好（三 split 全正，p<0.001）
   - 硬體對照表：BRAM 0% vs 23.57%、530 vs 7,439 cycles/event
   - 機制證據：unroll 1→4 只換來 2.22× 而非 4×（memory-port bound）
   - local array partition 的必要性（沒有它 cosim 268/1000 失敗）
   - ⚠️ 必須加註 latency 物件定義不同，只能比 interval 與資源

8. On-Board Validation (1 p)
   - JTAG 燒錄、25.733 µs 版本、LED/HDMI 狀態
   - 4096 筆逐位元比對：27 筆 1-LSB、0 筆決策翻轉、最小 margin 72 LSB
   - 根因未定，S_DRAIN 假設為負結果

9. Public Benchmark: FI-2010 (1 p)
   - weighted vs macro F1 的判定過程（Accuracy==Recall 恆等式）
   - 對照表；13.85 pp 差距 vs 約 1/28.1 參數量（**本節用 FI-2010 版 2,132**）
   - hidden64 容量零結果（落在 seed 分布內）

10. Negative Results and Limitations (1 p)  ← 不要刪，這節是加分的
    - 六個否定結果（見 PROJECT_STORY_RETROSPECTIVE 第十幕）
    - 🔴 方法論邊界：TXF 1 ms 時戳無法解析微秒 → 改用 LOBSTER ns 級資料
      量到 stale exposure 2.233% vs 7.199%，但明確聲明非 TXF
    - 缺口：無功耗數字（若已補則改寫）、無網路介面、非完整交易系統

11. Conclusion (0.5 p)

Appendix: 驗證等級對照表、完整資源表、重現步驟

完成後回報：REPORT_OUTLINE.md 路徑、總預估頁數、
以及哪幾節目前證據不足需要補。
```

---

# 階段 3／逐節撰寫

> **一次只寫一節。** 寫完檢查再寫下一節。
> 建議順序：4 → 6 → 7 → 10 → 5 → 8 → 9 → 3 → 2 → 1 → 11
> （先寫最有內容的，最後寫 Introduction——因為 Introduction 要總結你實際寫出來的東西，不是你想寫的東西。）

```
【共用守則】（貼上）

任務：撰寫 Section <N>：<節名> 的英文正文。

輸入：EVIDENCE_PACK.md、REPORT_OUTLINE.md 的對應節。

寫作要求：
1. 英文，學術但不浮誇。避免 "novel"、"significant"、"state-of-the-art"
   這類自我褒揚的形容詞——讓數字自己說話。
2. 每個數值後面加 `[source: <path>]`，最後統一轉腳註。
3. 每個數值標驗證等級（estimate / simulated / routed / on-board）。
4. 每段的第一句就是該段的主張，後面才是證據。
5. 圖表要有完整 caption：橫軸/縱軸/單位/資料來源/這張圖證明什麼。
6. 🔴 凡是我方主張比對手好的地方，同一段內要寫出限制條件。
   例：比 DeepLOB 參數少 → 同段要寫 F1 低 13.85 pp。
7. 長度控制在 outline 指定的頁數 ±20%。

🔴 禁止事項：
- 不得出現 HFT / high-frequency trading system 描述本作品
- 不得把 estimate 等級的數字寫得像量測值
- 不得省略否定結果
- 不得使用 EVIDENCE_PACK 裡沒有的數字
- 不得把 TCN 的 74.38 µs 與 PHC 的 5.30 µs 直接相除說「快 14 倍」

完成後回報：字數、使用到的 source_path 清單、
以及任何你認為證據不足以支撐的句子（標出來讓我決定刪或補）。
```

---

# 階段 4／稽核

> 目標：在拿給教授之前，自己先當一次 reviewer。

```
【共用守則】（貼上）

任務：對完成的英文報告全文做嚴格稽核，產出 REPORT_AUDIT.md。
你現在的角色是一位對這個領域熟悉、且傾向挑剔的審稿人。

逐項檢查並列出所有問題（每項附行號與建議修法）：

A｜數字正確性
   1. 每個數值是否都能在 EVIDENCE_PACK 找到對應？列出找不到的。
   2. 是否有同一個量在不同章節出現不同數值？
   3. 驗證等級標註是否正確？有沒有把 csynth estimate 講成量測值？

B｜過度宣稱
   4. 有沒有任何句子在讀者眼中會變成「這是一個 HFT 系統」？
   5. 有沒有比較是在不同定義/不同資料集/不同 latency 物件之間做的，
      而沒有加註？（特別檢查 TCN vs PHC、FI-2010 vs TXF、LOBSTER vs TXF）
   6. 有沒有把單 seed 結果講得像多 seed 結論？

C｜誠實性
   7. 十個否定結果/限制是否都還在？有沒有被稀釋成模糊措辭？
   8. 1-LSB 段落有沒有暗示根因已找到？
   9. 有沒有任何地方暗示這個系統可以實際部署交易？

D｜可讀性
   10. 一個 ECE 背景但不懂金融的讀者，能不能在不查資料的情況下
       看懂 Section 4 和 Section 6？列出需要補背景的地方。
   11. 每張圖的 caption 是否自足（不看正文也看得懂）？

E｜招生委員視角
   12. 前 200 個字有沒有講清楚「做了什麼、發現了什麼」？
   13. 有沒有明確可條列的三項貢獻？
   14. 全文有沒有一句話能被直接引用進推薦信？如果沒有，建議一句。

最後給一個總評：這份報告最強的一段是哪一段、最弱的是哪一段、
以及如果只能再改一個地方，改哪裡。
```

---

## 附：撰寫期間不可變動的權威數字

> 這些是這份報告的骨幹。Sonnet 若產出與此不符的數字，**以本表為準並要求它重查**。

**延遲 frontier（全部 routed）**
83.160 µs (8316 cyc @100MHz, WNS +0.020) → 25.733 µs (2316 @90MHz, **上板**) → 22.360 µs (2236 @100MHz, WNS +0.004) → 12.020 µs (1202 @100MHz, WNS +0.445) → **5.300 µs** (530 @100MHz, WNS +0.130, **DSP 89.09%**)。合計 **15.69×**。3.830 µs 版需 DSP 452 > 220，裝不下。

**對應 NEW3 樣本外 PnL**：05C4 dense 5,237,410／05E hidden8 5,032,255／05F diagonal 4,727,425。

**⚠️ 參數量分軌（2026-09-24 修正，見 `EVIDENCE_GAP_RESOLUTIONS.md` G1）**
- TXF 生產模型（Sections 5–8、硬體 frontier）：**1,731** params，16 features，hidden 16
- FI-2010 基準模型（Section 9、對 DeepLOB）：**2,132** params，輸入 100×40
兩者不可互換。

**PnL 單位（2026-09-24 查證）**：`POINT_VALUE = 200.0`（`01_full_l2_action_universe_builder/common_phase01.py:21`），即台指期每點新台幣 200 元；`pnl = Δpoints × 200 − MAKER_FEE`。**因此 `excess_pnl_H20` 的單位是新台幣（TWD）**，maker fee 15.0 亦為 TWD/口。所有 PnL 軸一律標 **TWD**。

**FI-2010**：PHC-lite 2,132 params，weighted-F1 **66.50 ± 1.15%**（3 seeds），macro-F1 63.97 ± 1.45%；DeepLOB weighted-F1 **80.35%**，**60,000** params（論文 Table III）。差 **13.85 pp**，參數比約 **1/28.1**。
🔴 **不得使用 142,435 或 1/67**——來源為第三方重新實作，無法一手查證。

**NEW5 一次性 holdout**：HGB **+893,555**／queue rule +722,745／no_quote 0／old_bid_only **−553,365**。九項凍結契約全數 pass。

**1-LSB**：4096 筆中 27 筆漂移，最大 1 LSB，方向混合（11×−1、21×+1），**決策翻轉 0 筆**，最小 margin 72 LSB。

**TCN 對照**：BRAM **0.00% vs 23.57%**；interval **530 vs 7,439 cycles/event**；unroll 1→4 為 16,510 → 7,438 cycles（**2.22× ≠ 4×**）；distilled TCN routed 74.38 µs @100MHz WNS +0.187。TCN 在 PnL 上 val +3,340／NEW3 +98,925／NEW4 +11,955（**優於 PHC**）。

**CPU 對照（單一口徑：06F N=4 round 1）**：mean 72.199 µs／p50 65.600／p95 126.300／**p99 262.600**／deadline 83.16／**miss rate 15.125%**。

**LOBSTER stale exposure**：2.233% @25.733 µs／3.712% @83.16 µs／**7.199% @262.6 µs**，差 4.965 pp。**必須標明：2012-06-21 NASDAQ 五檔樣本，非 TXF。**

**Ablation（val PnL delta）**：fixed_decay_0.9 **+470**／memory_off **−803,235**／reverse_sequence **−2,764,500**／shuffle_timesteps **−2,569,590**。

---

## 時程建議

| 期間 | 做什麼 |
|---|---|
| 9/24–9/30 | 階段 1（證據蒐集）+ 補功耗數字 + 查 PnL 單位 |
| 10/01–10/05 | 階段 2（骨架） |
| 10/06–10/18 | 階段 3（逐節撰寫，一天一節） |
| 10/19–10/22 | 階段 4（稽核）+ 修訂 |
| **10/23 前** | **把報告 + CV + SOP 草稿 + 三個希望被強調的具體事蹟，一起交給指導教授並正式請推薦信** |
| 10 月底–11 月 | 去識別化 repo 公開 + arXiv 上掛 + SOP 定稿 |

**推薦信的死線才是真死線。** 報告晚一週沒事，推薦信晚一週會出事。
