# Codex 任務包（2026-10-02）

兩個任務互不相依，可同時發。
**優先序：B 先發**（它會動到報告的圖與表，而 A 要等訓練）。

---

# 任務 A｜FI-2010 容量掃描（唯一新增實驗）

## 背景（貼給 Codex）

現行報告 9.4 節主張「加大模型容量無助於縮小與 DeepLOB 的差距」，但支撐證據只有：
hidden16 三個 seed（weighted-F1 66.50 ± 1.15）對上 **hidden64 單一 seed 0**（66.32）。

這個證據的方向會隨比法翻轉：
- hidden64 seed0（66.32）vs hidden16 **seed0**（65.18）→ **+1.14 pp，變好**
- hidden64 seed0（66.32）vs hidden16 **3-seed 平均**（66.50）→ **−0.18 pp，變差**

同一組數字、不同對照、相反符號 = 證據不足以支撐該節的標題。
本任務的目的是把「一次比較」升級為「一條容量曲線」。

## Prompt

```
任務：FI-2010 上的 PHC-lite 容量掃描，把 9.4 節的單點比較升級為多點曲線。

【階段 0 — 先停下來報成本，不要直接開跑】
在訓練任何東西之前，先回報：
  (a) 既有 v3_fair512 的訓練配方（epoch 上限、early-stop patience、batch size、
      LR schedule、loss 設定），從 02_train_phc_lite_stage1.py 與
      13_train_hidden64_capacity.py 讀出並列表
  (b) 既有可重用的 checkpoint 清單（哪些 hidden x seed 已經存在、路徑、參數量）
  (c) 單次訓練的實測或估計耗時，以及本次掃描需要新訓練幾個 run、總時數估計
  (d) 磁碟空間需求
回報後等我確認再進入階段 1。

【範圍與硬性禁止】
1. 只使用 FI-2010 公開資料。**嚴禁**讀取、載入或引用任何 TXF 資料、
   NEW1–NEW5、或 n2ew_era 底下 04b/04c/05c4/06* 的任何結果。
2. **不得覆寫或刪除任何既有 checkpoint、CSV、報告**。所有新產物用新檔名。
3. **不得修改既有訓練腳本**。要改就複製成新檔（例如 15_train_capacity_sweep.py）。
4. 不涉及 HLS、csynth、cosim、P&R、bitstream、板子。純 PyTorch。
5. 全部序列化執行，不要並行多個訓練把機器塞爆。

【階段 1 — 容量掃描】
變因只有一個：hidden 維度。其餘**全部必須與 v3_fair512 完全相同**。

  hidden ∈ {8, 16, 32, 64, 128}
  seeds  ∈ {0, 1, 2}
  horizon k = 50（注意：既有檔名裡的 h50 指的是 prediction horizon，不是 hidden
             維度，不要搞混；新檔名請用 cap{HIDDEN} 明確標示容量）

⚠️ 必須固定不變的三件事（這是整個實驗的效度所在）：
  - **batch size 固定**。V3_FAIR512_REPORT 的 status 明寫其結論是
    "after removing batch_size confound"，若掃描中 batch size 隨容量變動，
    整個結果作廢。
  - **early-stop / patience 規則固定**，否則容量差異會與訓練長度混淆。
  - **LR schedule 與 loss 設定固定**（raw loss，非 weighted）。

重用規則（省時間，但要明確標示）：
  - hidden16 的 seed 0/1/2 已存在於 models/phc_fi2010_h50_seed{0,1,2}_raw_v3_fair512.pt
    → 直接重用，不要重訓
  - hidden64 的 seed 0 已存在於 models/phc_fi2010_h50_seed0_raw_v3_hidden64_capacity.pt
    → 重用；只需補訓 seed 1、2
  - 但若上述 checkpoint 的訓練配方與本次掃描的固定配方**不一致**，
    必須回報並停下來問我，不要自行決定「差不多就好」

【階段 2 — 指標與一致性檢查】
每個 (hidden, seed) 都要輸出：
  trainable_parameters, test accuracy, test macro-F1, test weighted-F1,
  best_epoch, 實際訓練 epoch 數, 訓練 wall time

一致性閘（沿用 03_recompute_stage1_metrics.py 既有做法）：
  對每個 checkpoint 驗證 sklearn average="weighted" recall == accuracy_score，
  記錄最大絕對差。任何一個超過 1e-9 就回報並停下。

【階段 3 — 空間卷積前端的對照】
既有 14_train_spatial_frontend_probe.py 的結果（+288 參數、3-seed macro-F1 +3.67 pp）
需要一個**同容量**的對照，才能支撐「表徵優於容量」這個主張。
請找出 spatial conv 變體的總參數量，並在掃描曲線上指出
「用同樣多的額外參數去加大 hidden，可以換到多少」，作為控制變因比較。
若既有 probe 不是 3 seeds，請補到 3 seeds（同樣固定配方）。

【階段 4 — 產出】
1. results/capacity_sweep_v3.csv（長表，一列一個 run）
2. results/capacity_sweep_v3_summary.csv（每個 hidden 的 3-seed mean ± std）
3. figures/fig9_capacity_sweep.png + .svg + 產生腳本
   - 橫軸：trainable parameters（對數刻度）
   - 縱軸：test weighted-F1 (%)
   - 每個 hidden 畫 3-seed 的 mean ± std error bar
   - 水平虛線標 DeepLOB 80.35%（並在圖說註明其參數量 60,000、來源為論文 Table III）
   - 另標出 spatial conv 變體的點，用不同顏色
   - 軸標與單位齊全，圖說要自足
4. CAPACITY_SWEEP_REPORT.md，含：
   - 固定配方表（證明只有 hidden 在變）
   - 完整結果表
   - 一致性閘結果
   - **一句不帶修飾的結論**：曲線是平的，還是有斜率？
     若有斜率，給出斜率估計與 95% 區間；若是平的，給出「在此 seed 變異下，
     我們能排除多大的效應量」（即檢定力陳述）
   - ⚠️ 不要替我詮釋這對專題好不好。只報事實。

【完成後回報】
新訓練了幾個 run、總耗時、曲線形狀的一句話結論、
以及任何你認為會讓這個比較失效的疑慮。
```

---

# 任務 B｜PnL 量別稽核（最緊急，會動到圖與表）

## 背景（貼給 Codex）

已確認報告中混用了三個不同的 PnL 量。以 NEW3 的 PHC（05c4 cosine_lr seed2）為例：

| 數值 | 真正是什麼 |
|---:|---|
| **5,237,410** | `total_excess_pnl_H20` — **動作層級 markout 加總** |
| **611,095** | `causal_sequential_excess_pnl`（conservative_l2_stress, inventory_cap=1），成交 17,907 筆，平均每筆成交 34.126 |
| **488,870** | **佇列規則 baseline** 的序列化損益，**不是 PHC 的** |

已知的具體錯誤：
1. 報告第五章 5.4 表、第六章 frontier 表、以及 `figures/fig6_1_latency_value_frontier.png`
   的縱軸，把 **5,237,410 標示為「序列化超額損益」** —— 錯，它是 markout 加總。
2. `demo_narrative_and_script_20260808.md` 把 **488,870 當成本專案的成果**引用 ——
   那是佇列規則 baseline 的數字。

## Prompt

```
任務：PnL 量別全檔稽核與修正。這是純讀取與修正標籤的工作。

【硬性禁止】
1. **不得重新計算、重新訓練、重新模擬任何 PnL**。只從既有 CSV / report 讀取。
2. **不得修改 n2ew_era 底下任何既有實驗產物**。只修正報告層文件與圖。
3. 不得碰 NEW5 原始資料。

【要做的事】

Step 1｜建立量別字典
掃描 n2ew_era，找出所有 PnL 相關欄位名稱，建立一張字典表，每列包含：
  欄位名稱 / 口語稱呼 / 精確定義 / 單位 / 典型來源檔案 / 是 baseline 還是 model
至少必須涵蓋：
  total_excess_pnl_H20, causal_sequential_excess_pnl, action_pnl,
  phc_causal_fill_pnl, simple_rule_causal_fill_pnl, sequential_excess_pnl,
  avg_trade_excess_pnl, excess_pnl_H20
特別釐清：06u 的 "Signal Quality Sources" 表裡標為 val/NEW3/NEW4 action_pnl 的
  2,037,925 / 5,237,410 / 1,871,685 究竟是哪一個量——這組數字是 frontier 圖的縱軸，
  必須百分之百確定。

Step 2｜逐一盤點報告中的每個 PnL 數字
對下列檔案，列出每一個出現的 PnL 數值，標註它目前被寫成什麼、實際是什麼：
  - 專題成果報告_中文初稿_20261001.md
  - figures/make_fig6_1_frontier.py（含圖內文字與軸標）
  - demo_narrative_and_script_20260808.md
  - EVIDENCE_PACK.md、REPORT_OUTLINE.md、REPORT_SECTION_04.md
  - PROJECT_STORY_RETROSPECTIVE.md
  - TEAM_HANDOFF_AND_FINALS_PLAN_20260812.md
輸出 PNL_AUDIT.md，表格欄位：
  檔案 / 行號 / 數值 / 目前標示 / 正確量別 / 來源路徑 / 需不需要改 / 建議改法

Step 3｜核實這組序列化數字
確認以下是否全部屬於 PHC 05c4 cosine_lr seed2、而非 baseline：
  NEW3 causal_sequential_excess_pnl = 611,095
  fill_count = 17,907
  avg_trade_excess_pnl = 34.126
  相對佇列規則 baseline（488,870）的差 = 122,225（+25.0%）
並補齊 val 與 NEW4 的對應三元組（序列化損益 / 成交數 / 每筆平均）。

Step 4｜修正圖 6-1
修改 figures/make_fig6_1_frontier.py：
  - 縱軸標籤改為正確量別（依 Step 1 的結論，很可能是
    "Out-of-sample action-level excess PnL on NEW3 (H20 markout) [TWD]"）
  - 圖下方註腳補一句，說明本圖的 PnL 是 markout 加總，
    與序列化因果成交損益是不同的量
重新產生 png 與 svg。**不要改動任何數值，只改標籤與註腳。**

Step 5｜產出修正清單
PNL_AUDIT.md 末尾附一份可逐項勾選的修正清單，標明哪些是
  (a) 純標籤錯誤（改字即可）
  (b) 歸屬錯誤（把 baseline 的數字當成我們的）
  (c) 需要我判斷的（例如某個表到底該用哪個量）

【完成後回報】
發現幾處標籤錯誤、幾處歸屬錯誤、圖是否已重新產生、
以及任何你無法從既有檔案判定量別的數字（標 [UNRESOLVED]，不要猜）。
```

---

## 驗收重點（我會檢查的）

**任務 A**
- 固定配方表是否真的證明只有 hidden 在變（特別是 batch size）
- 重用的 checkpoint 配方是否與新訓練的一致
- 結論句有沒有夾帶詮釋（不該有）
- 若曲線是平的，有沒有給檢定力陳述

**任務 B**
- 06u 那組 action_pnl 的量別是否被確定下來（這是 frontier 圖的縱軸，不能模糊）
- 有沒有把「改不出來的」誠實標成 [UNRESOLVED] 而不是猜
- 圖只改標籤、數值未動
