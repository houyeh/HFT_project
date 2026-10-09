# Codex 任務包｜09A：ML 調節確定性規則（延伸實驗）

> 建立：2026-10-05｜性質：延伸額外嘗試，不改動既有報告結論
> 一句話：**快路徑跑簡單規則、慢路徑由 ML 每 Δ 時間調一次規則參數，這樣的 PnL 和「PHC 逐事件決策」比起來如何？**

---

## 背景（貼給 Codex）

現有結果（序列化因果成交口徑，conservative_l2_stress，inventory_cap=1，H20）：

| split | 佇列規則 θ=0.1 | PHC-STP v2（05c4，逐窗口） |
|---|---:|---:|
| val  | 103,260 | 171,835 |
| NEW3 | 303,700 | 611,095 |
| NEW4 |  72,415 | 129,160 |

07d 已證實：**單一靜態**淺層樹無法重現 PHC（251 葉、最弱切分只保留 33.3% 優勢）。
本任務測的是另一個假設：**隨時間變動的規則**（參數由 ML 低頻調整）能追回多少差距。

可直接重用（唯讀 import，不得修改）：
- `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py`
  - `load_sequences`、`queue_rule_decision`（q>0.1 → BID、q<-0.1 → ASK）、`fill_probabilities`、
    `stable_uniform`、`decision_pnl_from_payload`、`simulate_scheduled_by_event_index`、`matched_random_for_split`
- `n2ew_era/05c4_phc_stp_cosine_lr/common_05c4.py`、`results/phc_stp_v2_selected_policy.json`（門檻）
- `n2ew_era/07d_phc_decision_distillation/results/teacher_scores_{train,val,new3,new4}.npy`（凍結 PHC 分數，shape (N,2)=[bid,ask]）
- 07d 的 `causal_fill_metrics` 寫法（成交 salt 必須與 05c4/07d 相同，否則與既有數字不可比）

注意：payload 的決策點是 **每 32 事件一個窗口**（STRIDE=32），`timestamp_us`、`session_code` 皆已在 payload 內。

---

## Prompt

```
任務：09A ML 調節確定性規則。新目錄 n2ew_era/09a_ml_modulated_quote_rule/。

【硬性禁止】
1. NEW5 嚴禁讀取、載入、引用。任何 split 名稱為 new5 一律 raise。
2. 不得修改或覆寫任何既有腳本、checkpoint、CSV、npy、報告。只能 import 與讀取。
3. 不訓練任何神經網路。不重算 teacher scores（若 npy shape 對不上 → 停下回報，不要自己重算）。
4. 不涉及 HLS、csim、csynth、cosim、P&R、bitstream、板子。純 CPU Python。
5. 所有選擇（門檻、bucket 數、參數對應表、超參數）只能用 train 擬合、val 選擇。
   NEW3/NEW4 只報告，不得回頭用來選任何東西。
6. 序列化執行，不要並行。
7. 每個階段完成後 git commit，只 add 09a 目錄內的新檔，commit message 註明階段。

【階段 0 — 先停下來回報，不要直接跑實驗】
(a) 健全性重現：用既有函式重算下列 6 個數字，必須與上表完全一致（容差 0）：
    佇列規則 θ=0.1 與 PHC 逐窗口，在 val/NEW3/NEW4 的 sequential_excess_pnl。
    任何一個不一致 → 停止，回報差異與原因，不要進入階段 1。
(b) 時間解析度：每個 split 列出相鄰決策點的 timestamp 間距分佈
    （p10/p50/p90，單位 ms），以及每個 Δ 下每個 session 平均有幾個 block。
    若某個 Δ 小於 p50 間距（等同逐窗口），標記出來。
(c) 成本：單次 causal-fill 模擬耗時、本任務總共要跑幾次、總時數估計、記憶體峰值。
回報後等我確認再進入階段 1。

【階段 1 — 實驗組】

時間切塊：每個 session 內依 timestamp_us 切成長度 Δ 的 block。
因果規則：block k 使用的參數，只能由「block k 開始之前」可得的資訊決定
         （即 block k-1 及更早）。session 第一個 block 用預設參數 θ=0.1、雙邊、報價。
Δ ∈ {逐窗口(32 事件), 10 ms, 100 ms, 1 s, 10 s, 60 s}

慢路徑訊號（ML）：PHC teacher scores。兩種彙總，分開報告：
  - last（主設定，代表 ML 每 Δ 只跑一次）：block k-1 最後一個窗口的分數
  - mean（補充）：block k-1 所有窗口分數的平均
  由分數衍生兩個量：strength = max(bid, ask)，tilt = bid − ask

慢路徑訊號（非 ML 對照）：同樣機制，訊號改用 target_raw 的
  volatility_past_50 與 event_rate_past_50（取代 strength），
  imbalance 方向用 queue_imbalance 的 block 平均（取代 tilt）。
  目的：區分「增益來自 ML」還是「只是 regime 統計」。

參數對應：訊號切成 K 個分位數 bucket（K ∈ {2,3,5}，邊界在 train 上擬合）。
  每個 bucket 的參數，在 train 上以 total_excess_pnl_H20（markout，非路徑相依、快）挑選。
  K 與彙總方式在 val 上以 sequential_excess_pnl 選定。

組別：
  參考組
    R_fixed       佇列規則 θ=0.1（現行基準）
    R_static_best 佇列規則 θ ∈ {0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.5}，val 選一個固定 θ
                  ← 關鍵對照：ML 調節必須贏過「把門檻調好」才算有意義
    PHC_event     05c4 逐窗口決策（上界）
  調節組（每組 × 每個 Δ × {ML, 非ML}）
    G   開關：bucket → {報價, 不報價}；報價時用 R_static_best 的 θ
    T   門檻：bucket → θ ∈ 上述 θ 集合
    S   方向：bucket(tilt) → {雙邊, 只 BID, 只 ASK}；θ 用 R_static_best
    GTS 三者合併（bucket 為 strength × tilt 的組合，K ≤ 3 以免爆炸）

【階段 2 — 輸出】
results/leaderboard.csv：每列 = 組別 × Δ × 訊號來源 × split，欄位至少含
  sequential_excess_pnl, fill_count, avg_pnl_per_fill, max_drawdown,
  retention_vs_static = (arm − R_static_best) / (PHC_event − R_static_best),
  retention_vs_fixed  = (arm − R_fixed) / (PHC_event − R_fixed),
  matched_random_95p, beats_random_95p, sessions_won_vs_static（贏幾個 session / 共幾個）,
  param_updates_per_session, selected_K, selected_agg
figures/：
  (1) 主圖：x = Δ（log），y = retention_vs_static，線 = G/T/S/GTS，ML 實線、非 ML 虛線，
      分 NEW3/NEW4 兩個子圖
  (2) PnL 排行長條圖（NEW3、NEW4 各一）
REPORT_09A.md：先貼預先登記的判定（下方），再貼結果，列出全部組別（不得只列最佳）。

【預先登記的判定 — 跑之前就寫進 REPORT 開頭，跑完不得修改】
H1 ML 調節有增益：存在某組在 Δ ≥ 100 ms 時，val/NEW3/NEW4 三者皆 > R_static_best
H2 增益來自 ML：同組同 Δ 下，ML 版在三個 split 皆 > 非 ML 版
H3 慢路徑足夠：存在某組在 Δ ≥ 100 ms 時，NEW3 與 NEW4 的 retention_vs_static 皆 ≥ 0.5
任何假設不成立就標記 resolved_negative，照實報告。
證據等級一律標 dev_only（NEW5 已用盡，非新鮮保留集）。
```

---

## 我（使用者）跑完要看的三件事

1. **R_static_best 比 R_fixed 高多少？** 如果光調門檻就追回大半，ML 的故事要縮小。
2. **主圖曲線在哪個 Δ 開始掉？** 那個轉折點就是「ML 值得多快更新」的答案。
3. **ML 實線 vs 非 ML 虛線有沒有分開？** 沒分開 = PHC 在這裡只是昂貴的波動率指標。
