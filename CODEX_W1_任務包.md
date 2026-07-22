# Codex W1 任務包（2026-07-14 → 7/29 meeting 前）

> 發派順序即優先序。任務 0 先做完（定樁後才有「改動可追溯」的保障），其餘可並行。
> 任務 2、3 的 prompt 已存在於 `n2ew_era/NEW_TEAMMATE_BRIEFING_REPORT.md`，此處只列差異要求，發派時把原 prompt 原文複製給 Codex。

---

## 任務 0：Git 版本控制定樁（最高優先，先做）

```
任務：為整個 00_HFT_project repo 建立第一個 git baseline commit，堵掉「無任何 commit 歷史、
改動不可追溯」的審計缺口。

背景：repo 根目錄（00_HFT_project/）可能存在 .git 但沒有任何 commit（先前 git status 顯示
no commits yet 或 git log 失敗）。n2ew_era/ 的審計文件明確點名此問題（限制清單第16條）。

鐵則（絕對不可違反）：
1. 只做「新增」操作：git init（若需要）、建立 .gitignore、git add、git commit、git tag。
   禁止任何 rewrite/rebase/force/reset/清除既有 .git 的操作。若發現既有 .git 損壞，先回報
   狀況再處理，不要自行刪除重建。
2. commit 前必須先跑 `git status` 與 `git count-objects -vH` 檢查將被納入的檔案總量；
   若預估 repo 超過 2GB，先回報清單再決定，不要直接 commit。
3. 不得 commit：.venv/、__pycache__/、.Xil/、*.obj、HLS/Vivado 的中間建置產物
  （solution*/impl、solution*/sim、*.jou、*.log 的 Vivado 大檔、hls4ml_prj 的建置目錄）、
   data/ 下的原始大型資料檔（>50MB 的 csv/parquet/pt 逐一列出改用清單記錄）。
4. 必須 commit：所有 .md 報告、所有 .py 腳本、所有 results/*.csv（PnL 溯源鏈的核心，
   即使較大也要保留——單檔 >50MB 才例外並記錄）、csynth.rpt/timing 等報告文字檔、
   figures/*.png、HLS 原始碼（hls_src/）。
5. 對被排除的大檔，產生 EXCLUDED_FILES_MANIFEST.md：路徑、大小、SHA256、為何排除、
   如何重新生成（哪支腳本）。

步驟：
1. 盤點 repo 大小結構（du 分層），列出 >50MB 的檔案。
2. 撰寫 .gitignore（依上述規則，附註解說明每一類為什麼排除）。
3. git add -A && git commit，訊息格式：
   "baseline: first snapshot of 00_HFT_project (pre-departure freeze, 2026-07)"
4. git tag v0-baseline-20260714
5. 產出 GIT_BASELINE_REPORT.md：commit hash、納入檔案數、總大小、排除清單摘要、
   驗證指令（git log --stat 首屏截取）。

常見坑：
- Windows 路徑含中文檔名：先確認 git config core.quotepath false，避免 log 亂碼誤判。
- 不要順手「清理」任何現有檔案，包括看起來重複的舊版報告——快照的意義是保存現狀。
- results csv 若有數千個小檔，正常 commit，不要打包壓縮（會破壞逐檔溯源）。
```

---

## 任務 1：TXF 事件間隔分布分析（突破口 A，本週新實驗）

```
任務：量測 TXF L2 事件流的相鄰事件時間間隔（inter-arrival time）分布，回答
「83.16µs / 23.16µs 的推論延遲，在此市場會漏掉多少機會」。

背景：FPGA 推論核心單步延遲 83.16µs（06F v8）、已量測的優化版 23.16µs（06L unroll-16）。
外界質疑「83µs 對 HFT 太慢」。若 P(相鄰事件間隔 < 延遲) 極小，則此延遲在台指期
事件密度下實質零損失——這是把延遲預算建立在市場實測上的關鍵證據。

鐵則：
1. 禁止觸碰 NEW5。只用 NEW1–NEW4（train/val/NEW3/NEW4）。
2. 純描述性統計分析，不得順手改動任何模擬器、模型、門檻設定。
3. 第一步必須先驗證時間戳解析度：檢查原始資料 timestamp 欄位的最小非零差分。
   若解析度粗於 1ms，則「P(Δt<83µs)」根本無法量測——此時誠實回報解析度上限，
   改以「P(Δt < 資料解析度最小刻度)」呈現，並在報告開頭大字聲明此限制。
   絕對不要在解析度不足時硬給 µs 級機率數字。

分析內容：
A. 全事件流：相鄰 L2 事件 Δt 的分布（分 session 計算再彙總，不跨日算差分）。
   輸出：P50/P90/P99/P99.9、P(Δt<23.16µs)、P(Δt<83.16µs)、P(Δt<1ms)、P(Δt<10ms)。
B. quote-eligible 事件子集：同上統計（策略真正需要反應的時刻密度比全流稀疏，
   兩者都要給）。
C. 尖峰時段切片：每個 session 取事件率最高的 10% 時窗重算 A/B——延遲的代價
   集中在尖峰，平均值會低估風險。

輸出：
- AUDIT_event_interarrival_latency_budget.md（開頭 status + 時間戳解析度聲明）
- figures/event_interarrival_cdf.png：橫軸 Δt（對數刻度，單位標明 µs/ms）、
  縱軸累積機率 CDF；圖上垂直標線標出 23.16µs 與 83.16µs 兩個參考位置。
- 每個數字附來源 split 與計算腳本路徑。

常見坑：
- 跨 session（隔日開盤）的 Δt 是假間隔，必須排除。
- 時間戳若是交易所時間與接收時間兩欄，說明用了哪一欄、為什麼。
- 不要對 Δt=0（同戳事件批次）視而不見——回報同戳事件占比，這影響「事件級解析度」
  的解讀。
```

---

## 任務 2：seed0/seed1 csynth 實跑（跨種子硬體驗證）

**Prompt 原文**：複製 `NEW_TEAMMATE_BRIEFING_REPORT.md` §3.10 末尾的「給Codex的補做提示」整段。
**追加要求**：完成後在回報中附 `csynth.rpt` 的實際檔案路徑與 DSP/LUT/FF/BRAM/latency 五個數字的逐 seed 對照表；若 csynth 因環境（Vitis 授權/路徑）卡住，回報卡在哪一步的完整錯誤訊息，不要交敘事性總結。

## 任務 3：ML 逆轉根因診斷（03B6→04B1）

**Prompt 原文**：複製 `NEW_TEAMMATE_BRIEFING_REPORT.md` §3.4 的實驗 prompt 整段（含鐵則與 A/B/C 三個方向）。
**追加要求**：優先做方向 C（規則對照分解，計算量最小、鑑別力最強）；結論若 inconclusive 照實寫，7/29 meeting 需要的是「已執行的診斷」不是「漂亮的答案」。

---

## 任務 4：Latency–PnL 敏感度曲線（改寫版軟體目標 1）

```
任務：在既有 causal-fill 序列模擬器中注入不同的決策延遲，量測 PnL 對延遲的敏感度曲線，
找出（或證明不存在）PnL 開始劣化的臨界延遲。

背景：2026-06-26 已有 null result：CPU/FPGA 的延遲差異在現有 L2 資料解析度下，
成交機率差異率僅 0.091%，PnL 差異不可測。本任務不是推翻該結論，而是把它升級成
完整的定量曲線：從 0 到明顯有害之間，掃描延遲值，定位轉折點（若在資料解析度內存在）。

鐵則：
1. 禁止觸碰 NEW5。評估只在 val/NEW3/NEW4。
2. 不得重新訓練模型、不得重選門檻——三個凍結策略（simple rule / HGB / PHC seed2）
   原封不動，唯一變數是注入的延遲。
3. 延遲注入機制必須先寫成一段明確定義放在報告開頭：建議實作為「決策產生於事件 t，
   報價自事件 t 之後第一個時間戳 ≥ t+L 的事件起才進入成交判定」；若資料時間戳解析度
   不支援 µs 級 L（先跑任務 1 或直接檢查），L 的掃描下限就設在解析度允許的最小值，
   並在報告開頭聲明——不得在解析度以下編造差異。
4. 與 2026-06-26 null result 的關係必須在報告中明寫：本任務是其延伸，若曲線在整個
   可測範圍內平坦，結論就是「在此資料解析度與市場密度下，延遲低於 X 無經濟意義」，
   這是有效的邊界結果，照實寫。

掃描設計：
- L ∈ {0, 資料解析度最小刻度, 1ms, 5ms, 10ms, 50ms, 100ms, 500ms}（依解析度裁剪）
- 進階（時間允許）：固定均值 L、加入抖動（lognormal，σ 取 2 檔），比較
  「同均值、有/無抖動」的 PnL 差——這直接回應「確定性延遲的經濟價值」。
- 每個 (策略×L×split) 輸出 causal-fill sequential PnL 與成交次數。

輸出：
- AUDIT_latency_pnl_sensitivity.md（status + 延遲注入機制定義 + 解析度聲明開頭）
- figures/latency_pnl_curve.png：橫軸 注入延遲 L（對數刻度、標單位）、縱軸 causal-fill
  PnL（proxy 點值單位）、三條線（rule/HGB/PHC）、垂直參考線標 83.16µs 與 23.16µs
 （若低於解析度，於圖注說明參考線僅為視覺標記）。
- 結論段回答三選一：找到轉折點 X／全範圍平坦（邊界結果）／解析度不足以判定。

常見坑：
- 成交次數會隨 L 改變，PnL 下降可能只是「交易變少」而非「每筆變差」——同時回報
  per-fill 平均 PnL，把兩個效應拆開。
- 不要用「跳過 k 個事件」近似延遲——事件間隔不均勻，必須用時間戳。
- 若 PHC 在大 L 下狀態序列斷裂（隱藏狀態跨事件依賴），說明如何處理（延遲只影響
  報價生效時點，不改變模型看到的特徵流——模型仍逐事件推論）。
```

---

## 任務 5（W2 預告，W1 若有餘裕可先發）：23µs 版 routed P&R

```
任務：對 06L 已驗證的 unroll-16 組態（2,316 cycles / 23.16µs @100MHz）執行完整
Vivado implementation（P&R），取得 routed timing 證據，讓 23.16µs 從 csynth 估計
升級為與 83µs 版同級的實證數字。

鐵則：
1. 不得改動 06F v8 的任何凍結產物——在獨立工作目錄複製後修改。
2. 唯一允許的源碼差異：06L 的 unroll pragma 設定。clock 維持 10ns/100MHz，
   interface 設定與 06F 完全相同，否則資源/時序不可比。
3. 產出必須含 routed timing report 檔案路徑（WNS/TNS）、post-route utilization
  （LUT/FF/DSP/BRAM 四項，與 06F v8 並排成表）、以及 cosim 重跑確認 0 mismatch。
4. 若 timing fail（WNS<0）：不要調 clock 掩蓋，回報失敗的 critical path，並補跑
   90MHz 一版作為對照——「100MHz fail、90MHz pass」本身就是誠實且有用的結果。

回報格式：AUDIT_unroll16_pnr_timing.md，表格每格附 report 檔案路徑。
```

---

## 驗收清單（你本人 7/27 前逐項檢查）

- [ ] 任務 0：`git log` 有 baseline commit、tag 存在、EXCLUDED_FILES_MANIFEST.md 合理
- [ ] 任務 1：報告開頭有時間戳解析度聲明；CDF 圖兩條參考線；同戳事件占比有回報
- [ ] 任務 2：csynth.rpt 檔案真實存在（親自開檔確認，上次 Codex 交過「只有敘事沒有檔案」）
- [ ] 任務 3：AUDIT 檔 status 欄誠實（resolved/open/mixed）；沒有動到凍結模型
- [ ] 任務 4：延遲注入機制定義看得懂；per-fill 平均 PnL 有拆開；跟 6/26 null result 的關係有寫
- [ ] 所有新產出檔案已 commit（定樁後的第二個 commit：「W1 audit batch」）
