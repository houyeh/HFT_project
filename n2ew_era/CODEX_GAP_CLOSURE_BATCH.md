status: open
# Codex 缺口清單批次任務（一次發派版）

> 本檔彙整目前 `NEW_TEAMMATE_BRIEFING_REPORT.md` §1/§2.6/§3.4/§3.11/§4 列出的所有「需要實際跑程式才能補起來」的缺口，整理成可以一次性丟給 Codex 的獨立任務。每個任務都重複列出**鐵則**（即使你一次發派多個任務，每個任務的鐵則都要各自遵守，不能因為合併執行而省略）。任務之間的依賴關係見下表「順序建議」。

## 任務總表

| ID | 任務 | 是否阻擋PHC開NEW5 | 預期輸出 | 順序建議 |
|---|---|---|---|---|
| A | 03B6→04B1 ML輸贏逆轉根因診斷 | **是**（§1理由一） | `AUDIT_03b6_04b1_reversal_root_cause.md` | 優先，最重要 |
| B | Oracle執行假設敏感度掃描（inventory_cap × fill_model） | **是**（§1理由二） | `AUDIT_oracle_assumption_sensitivity_sweep.md` | 優先，可與A平行 |
| C | 同時滿足「出手次數匹配」+「序列模擬器約束」的最嚴格oracle | 否（補強§2.6，非開鎖必要條件，但建議在NEW5前做） | `AUDIT_matched_count_simulator_oracle.md` | B完成後做（依賴B的sweep結果決定用哪組假設） |
| D | 繪圖script標題/軸標籤cosmetic bug修正 | 否（純簡報風險，不影響NEW5決策） | 修正後的兩支script + 重新生成的 `AUDIT_chart_axis_semantics.md` | 可隨時獨立做，跟A/B/C完全無關 |
| E | PHC-lite跨種子（seed0/seed1）HLS資源穩定性複測 | 否（§3.10列為不阻擋因素） | `AUDIT_hls_cross_seed_resource_stability.md` | 最低優先，時間允許再做 |

**給Codex的開場提醒（建議貼在你發派任務時的訊息最前面）**：「以下是五個獨立任務，請依照ID順序執行，每個任務做完先回報結果再繼續下一個，不要一次性平行跑全部——尤其A跟B做完之後，C任務的設定要根據B的結果決定，不能憑空假設一組inventory_cap/fill_model。每個任務內部的『鐵則』都要遵守，違反鐵則的任何一條（尤其是碰NEW5）立刻停止並回報，不要自行決定『應該沒關係』。」

---

## 任務 A：03B6→04B1 ML輸贏逆轉根因診斷

任務：解釋並驗證「03B6 ML輸給簡單規則」與「04B1 TXF篩選後ML打贏」之間的根本機制（reversal root cause）

背景：
- 03B6（混合商品universe，TXF+其他商品）：HGB action-value模型在val/NEW3/NEW4上的表現劣於simple
  L2 rule（side_by_queue_imbalance_abs_0p1）。
- 04B/04B1（僅TXF商品重新開發）：相同類型的HGB模型（max_iter=120, threshold=0.224194）在
  val/NEW3/NEW4上轉為打贏simple rule，且beats_matched_random_95p=True、causal-fill sequential
  ml_beats_or_matches_queue=True。
- 目前沒有任何報告解釋「為什麼篩選掉非TXF商品後ML的相對排名會反轉」的具體機制。

鐵則（絕對不可違反）：
1. 禁止使用NEW5。本任務全程不得載入、讀取、或以任何方式觸碰NEW5資料。
2. 禁止用NEW3/NEW4做任何模型選擇、調參、門檻選擇——NEW3/NEW4只能用於「事後診斷分析」（產生
   統計量、畫圖），不能反過來改變03B6或04B1已經凍結的模型/門檻設定。
3. 不得重新訓練03B6或04B1已存在的模型；本任務是診斷分析，不是重新建模。
4. 所有分析只能用development（NEW1+NEW2 / train+val）與已存在的frozen NEW3/NEW4 diagnostic
   輸出做事後拆解，不能引入新的特徵或新的模型架構。

具體要做的事（任選其一或多個，依時間允許程度排序）：
A1. 特徵分布拆解：比較「混合商品universe」與「純TXF universe」在ML模型最重要的幾個特徵
    （用既有HGB feature_importances_或permutation importance，若03B6/04B1已存的模型物件還在，
    直接load出來看）上的分布差異，特別關注：queue_imbalance、deep_imbalance、spread_ticks、
    event_rate_past_50。輸出：每個特徵在「混合」vs「純TXF」資料下的分布統計表（mean/std/分位數），
    以及這些差異是否能解釋HGB決策邊界的改變。
A2. 預測誤差分解：對03B6與04B1兩個模型，分別在「同一份純TXF held-out資料」（不是NEW3/NEW4，
    用development內部切一份額外驗證子集，或如果03B6模型物件還留著，直接重新對純TXF development
    資料跑推論）上算prediction error，看03B6模型是否系統性地對「非TXF樣態」學到了會誤導純TXF
    判斷的pattern（例如：不同商品的tick size、價差結構不同，造成03B6學到的threshold在純TXF上失準）。
A3. 規則對照分解：simple rule（side_by_queue_imbalance_abs_0p1）在混合vs純TXF資料上的
    policy_pnl/win_rate本身是否也有變化？如果simple rule本身在兩種資料上表現穩定，但ML模型表現
    大幅變動，這會指向「ML對商品異質性更敏感」這個具體機制；反之如果兩者都變動，則可能是市場
    regime差異而非商品污染本身造成。

輸出格式要求：
- 產出一份AUDIT_03b6_04b1_reversal_root_cause.md，結構仿照現有n2ew_era/000_context/AUDIT_*.md
  系列（開頭status: resolved/open/mixed，結尾誠實結論段落）。
- 所有數字必須來自實際重新計算的程式輸出，不得估算或目視推測。
- 明確寫出「development-only」分析與「NEW3/NEW4事後診斷」分析的界線，每張表/每個數字都要標明
  用了哪個split。
- 如果分析完仍無法定論根因，誠實寫「diagnosed but inconclusive」，不要為了給出答案而過度推論。

常見坑（務必避開）：
- 不要因為要做「對照」就去新建一份「mixed universe development子集」再重新調參——這樣會等於
  重新做一次03B6，失去診斷意義，且可能誤觸碰NEW3/NEW4選模型的紅線。
- 不要把NEW3/NEW4當成驗證新假設的「測試集」——它們只能拿來看「同一個既定假設，在這兩個frozen
  split上是否也呈現同樣的pattern」，不能用來篩選哪個假設講得通。
- 若要load舊模型物件（pickle/joblib），先確認版本相容性，避免用新版sklearn載舊版模型導致
  attribute錯誤而誤判為「找不到特徵重要性」。

---

## 任務 B：Oracle執行假設敏感度掃描（inventory_cap × fill_model）

任務：檢驗 `04c1_new5_oracle_headroom_analysis` 系列的causal-fill sequential oracle結論，在不同的
inventory_cap與fill_model組合下是否穩健。

背景：
- 目前唯一存在的causal-fill sequential oracle只跑過一組設定：`fill_model=conservative_l2_stress`
  + `inventory_cap=1` + `H=20`。
- matched-count oracle（13.8%/18.6% capture ratio）跟dense oracle（21.5倍gap）都建立在這一組假設上，
  從未檢查換一組同樣合理的假設會不會讓結論（policy相對於oracle的差距大小、或policy之間的相對排名）
  顯著改變。

鐵則（絕對不可違反）：
1. 禁止使用NEW5。
2. 禁止用這次sweep的結果去重新選擇或調整任何已凍結的policy（HGB threshold、PHC checkpoint、
   simple rule門檻等一律不可更動）——本任務純粹是「同一批已凍結policy，在不同執行假設下重新跑
   評估」，不是「找一組讓結果更好看的假設」。
3. 只能用development（NEW1+NEW2）與NEW3/NEW4 diagnostic split，不可碰NEW5。
4. fill_model只能在既有三種（depth_haircut / queue_side_haircut / conservative_l2_stress）之間
   切換，不可發明新的fill model假設。

具體要做的事：
B1. 對既有的所有「已被報告引用」的policy（至少包含：side_by_queue_imbalance_abs_0p1、
    ml_hgb_regressor_shallow_convergence_selected、phc_stp_v2_cosine_lr seed2、dense oracle、
    matched-count oracle），在 inventory_cap ∈ {1, 2, 3} × fill_model ∈
    {depth_haircut, queue_side_haircut, conservative_l2_stress} 的3×3=9組合下（已有的cap=1+
    conservative_l2_stress這組直接重用既有結果，不必重跑），重跑causal-fill sequential評估，
    記錄每組合下的sequential_excess_pnl、fill_count、capture_ratio（against matched-count oracle
    重新算過的對應版本）。
B2. 畫出（或至少用表格呈現）capture_ratio隨inventory_cap/fill_model變化的趨勢，回答：
    「policy相對oracle的13.8%/18.6%這個數字，在合理的假設範圍內，最低/最高會落在哪個區間？」
    「不同policy之間的相對排名（例如ML是否一直贏queue rule）在9組假設下是否一致？」

輸出格式要求：
- 產出AUDIT_oracle_assumption_sensitivity_sweep.md，含一張9組合×各policy的完整數字表，以及
  一段誠實結論：明確回答「結論對假設選擇穩健」或「結論對假設選擇敏感，原本cap=1+
  conservative_l2_stress這組可能是比較樂觀/悲觀的特例」。
- 不得只挑好看的組合呈現，9組合全部要列出，即使其中某幾組讓現有policy看起來表現很差。

常見坑（務必避開）：
- inventory_cap調大後，policy的fill_count、forced_no_quote_count都會變，記得這些中間統計量
  也要記錄，不要只看最終PnL，否則無法判斷PnL變化是因為「執行假設變寬鬆」還是「policy本身行為
  改變」。
- 9組合計算量可能不小，若時間有限，至少要完成「固定fill_model=conservative_l2_stress、
  inventory_cap∈{1,2,3}」跟「固定inventory_cap=1、fill_model三種都跑」這兩條軸線，不要為了
  省時間直接跳過某個policy不測。

---

## 任務 C：同時滿足「出手次數匹配」+「序列模擬器約束」的最嚴格oracle

> 本任務建議在任務B完成後才執行，因為任務C需要從B的sweep結果中選一組（或多組）有代表性的
> inventory_cap/fill_model假設來建構這個oracle，不要憑空指定一組。

任務：建構並評估一個「同時遵守執行限制（成交機率模擬器+倉位上限）且出手次數與真實policy
完全匹配」的oracle，這是目前oracle headroom分析裡唯一還沒做出來的版本。

背景：
- 現有matched-count oracle只匹配了「出手次數」，沒有經過causal-fill序列模擬器（即假設這些
  挑選出來的動作100%都能成交）。
- 現有causal-fill sequential oracle經過序列模擬器，但出手次數跟真實policy不一樣（selection_rate
  =0.708，不是policy的真實出手次數）。
- 這兩者的「交集版本」——挑選時兼顧「未來最賺錢」且受「同樣的成交機率與倉位限制」約束，並讓
  最終實際成交次數逼近真實policy的出手次數——目前不存在。

鐵則（絕對不可違反）：
1. 禁止使用NEW5。
2. 這個oracle本身依然是「拿到未來標籤」的診斷工具，不是可交易策略——產出報告時必須明確聲明
   這一點，不可被誤用成「這是我們應該追求的新policy」。
3. 不可用NEW3/NEW4的結果回頭調整這個oracle的建構演算法（例如貪婪排序的細節、tie-breaking
   規則）——演算法設計只能基於development資料的探索，NEW3/NEW4只能用來報告最終數字。

具體要做的事：
C1. 設計一個序列化的貪婪選擇演算法：在development資料上，依時間順序逐個狀態檢查，用未來標籤
    排序所有「尚未用完的候選動作」的優先度，但每個動作是否真的計入oracle的PnL，仍要通過跟真實
    policy一樣的成交機率模型與inventory_cap檢查（用任務B選出的代表性假設組合）。目標是讓最終
    oracle的實際成交次數盡量逼近（容許小誤差，例如±2%）對應真實policy的fill_count，而不是
    quote_eligible次數。
C2. 在development上驗證這個演算法的行為合理（成交次數確實逼近目標、PnL確實是該假設下能達到的
    上界），再套到NEW3/NEW4 diagnostic split上報告最終capture_ratio。

輸出格式要求：
- 產出AUDIT_matched_count_simulator_oracle.md，明確列出這個oracle跟前兩種oracle的差異表格
  （三種oracle的「是否匹配出手次數」「是否經過序列模擬器」兩個維度的對照），以及新算出來的
  capture_ratio跟原本13.8%/18.6%的對比。
- 若新capture_ratio跟原本差距很大，要誠實解釋可能原因（例如：原本matched-count oracle因為沒
  受執行限制，挑的動作裡有一部分其實在真實成交機率模型下根本不會成交，因此真正的「同時匹配」
  oracle的PnL應該會比原本matched-count oracle更低，capture_ratio應該會比13.8%/18.6%更高）。

常見坑：
- 貪婪演算法如果用「先排序所有未來標籤再依序選」，要注意演算法本身不能用到未來才知道的「這個
  動作最後有沒有成功成交」的資訊去回頭調整排序順位（that would be另一種形式的未來資訊洩漏）；
  正確做法是「排序依據用標籤，但成交與否的判定完全依當下可得資訊，跟真實policy的判定邏輯共用
  同一套程式碼路徑」，不要為了方便另外寫一套簡化的成交判定。

---

## 任務 D：繪圖script標題/軸標籤cosmetic bug修正

任務：修正 n2ew_era/01_full_l2_action_universe_builder/03_generate_phase01_figures.py 與
n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py 兩支繪圖script中
title/xlabel/ylabel在多次savefig之間互相沿用舊文字的bug。

鐵則：
1. 不得更動任何資料計算邏輯、模型、或現有frozen結果csv——這純粹是畫圖script的文字標籤修正，
   不是分析邏輯修正。
2. 每一次產生新圖前，必須用 plt.figure() 或 fig, ax = plt.subplots() 建立全新的figure/axes物件，
   不可以重複使用同一個ax而不清除title/xlabel/ylabel。
3. 修正後，重新跑一次完整的圖檔生成流程，並重新跑 n2ew_era/000_context/ 下產生
   AUDIT_chart_axis_semantics.md 的稽核腳本（若無此腳本，依照該md檔的方法："from
   plt.savefig呼叫前後反查title/xlabel/ylabel原始碼"重新生成一份同格式的稽核表格），
   逐張確認inferred欄位降到最少、且title/xlabel/ylabel內容與檔名語意相符。
4. 輸出修正後的AUDIT_chart_axis_semantics.md供人工覆核，不要只回報「已修好」就結束。

---

## 任務 E：PHC-lite跨種子（seed0/seed1）HLS資源穩定性複測（低優先）

任務：對已通過csim/csynth/cosim/P&R全流程的`phc_stp_v2_cosine_lr` seed2 HLS實作，額外對seed0、
seed1的對應checkpoint重跑同一套HLS流程，確認資源使用量（DSP/LUT/FF/BRAM）與latency是否與seed2
版本一致或在合理誤差範圍內。

鐵則：
1. 不得更動06F既有seed2版本的任何已驗證通過的HLS專案設定（pragma、interface、clock constraint
   等）——seed0/seed1必須套用完全相同的設定，唯一差異應該只是權重數值不同。
2. 不需要重新跑P&R的timing closure驗證到跟seed2一樣完整的程度，但至少要跑csynth取得資源估計值，
   若時間允許再跑到cosim確認數值正確性。
3. 不可宣稱「跨種子穩定」除非真的跑出數字比較過——不能因為理論上权重不影響資源使用就跳過實測
   直接下結論。

輸出格式要求：
- 產出AUDIT_hls_cross_seed_resource_stability.md，至少包含seed0/seed1/seed2三者的csynth資源
  數字對照表，以及一段結論：資源使用量是否真的幾乎不受權重影響（理論上應該如此，因為HLS資源
  使用主要由運算圖結構決定，不是權重數值），若有意外差異要說明可能原因。

常見坑：
- 如果seed0/seed1的checkpoint不在標準路徑或檔名格式跟seed2不同，先確認路徑正確，不要因為load
  錯模型而產生誤導性的資源數字。
