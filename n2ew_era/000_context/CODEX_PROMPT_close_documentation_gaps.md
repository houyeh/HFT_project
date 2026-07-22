# Codex Prompt：補齊 N2EW_ERA_EVOLUTION_AND_INHERITANCE.md 的可查證缺口

> **2026-06-19晚間更新**：本檔案已從原始版本擴充。任務1-6維持原狀（目前查證為**尚未執行**——`AUDIT_*.md`/`GAP_CLOSURE_SUMMARY.md`等輸出檔案均不存在）。任務7已**撤回**（前提錯誤，已查證完成，見下方說明，不需要再做）。新增任務8/9/10，是針對`B_CLASS_RETROSPECTIVE_ANALYSIS.md`三個被Reviewer角度追問、目前文件裡只寫了「未來工作」「未明確說明」、但尚未真正動手查證或做實驗的問題。今晚請Codex依序處理：任務1-6（A類程式碼考古，原範圍不變）+ 任務8/9/10（新增，B類缺口的實證/查證任務，優先度高於1-6，因為這三個是已經被質疑過的論述缺口）。任務7不要做。

## 背景與目的

`n2ew_era/000_context/N2EW_ERA_EVOLUTION_AND_INHERITANCE.md` 是研究者用來在教授/reviewer面前備課的「底氣文件」，搭配`B_CLASS_RETROSPECTIVE_ANALYSIS.md`（事後建構論述）與`C_CLASS_HLS_STATUS_AND_VERIFY_STEPS.md`（HLS現況糾正）兩份補充文件。這幾份文件秉持一個嚴格原則：**任何報告正文沒寫清楚的地方都明確標注「未明確說明」，絕不替使用者編造**。

## 鐵則（請務必遵守，這是本專案最容易出包的地方）

1. **每一個你填補的答案，都必須附上明確出處**：檔案絕對路徑 + 行號（函式名稱也要）。沒有出處的答案不要寫進報告，寧可寫「查無此程式碼/此邏輯」。
2. **絕對不要修改任何已凍結的report（`*_REPORT.md`、`*_FREEZE_REPORT.md`等）或已凍結的config/checkpoint檔案**。你的所有發現請寫成**新的**審計報告檔案，不要動到既有凍結文件。
3. 如果你跑了腳本/實驗去驗證某件事，**必須附上你實際跑出來的數字/log**，不能只憑讀程式碼邏輯就下結論。**任務8/9/10都是要求你實際執行驗證或實驗，不是文獻式推理**——這點與任務1-6（純程式碼考古）不同，請特別注意。
4. 任何「我認為合理推測但程式碼/實驗裡沒有直接證據」的部分，請明確標注「推論」，不要寫成像是查到的事實。
5. 完成後，在每個任務的輸出報告開頭寫一行 `status: resolved` 或 `status: gap_confirmed_unresolvable`（後者代表你查了/跑了但確認真的沒有答案或答案是否定的，這跟"resolved"一樣有價值，請不要因為結果不如預期而硬掰成正面結論）。
6. **任務8/9/10如果實驗結果是負面的（例如PHC-lite的記憶機制其實沒有顯著優於固定窗口、或新評估方法其實也沒有真正被外部驗證過），請誠實回報負面結果**。這份文件的價值在於誠實，不在於把每個缺口都填成正面故事。負面結果一樣是`status: resolved`（代表問題已經有明確答案，答案是否定的）。

## 任務清單（原任務1-6，範圍不變，狀態：尚未執行）

### 任務1：圖表軸義還原（對應文件第8節）

以下圖檔目前報告正文沒有明確寫橫/縱軸定義，請找出**產生這些圖的繪圖程式碼**（搜尋`matplotlib`/`plt.savefig`/對應檔名字串），抓出`xlabel`/`ylabel`/`title`字串，或圖例(legend)文字：

- Phase 01: `action_count_waterfall.png`, `action_side_distribution.png`, `horizon_label_distribution.png`, `feature_distribution_snapshot.png`, `feature_distribution_by_split.png`, `old_residual_vs_new_universe.png`
- Phase 02: `validation_policy_pnl_heatmap.png`, `validation_family_pnl_heatmap.png`, `selected_policy_equity_curves.png`
- Phase 02f: `sequential_excess_equity_curves.png`
- Phase 03a: `bucket_lift_*.png`
- Phase 03b6: `hgb_val_policy_pnl_by_iter.png`
- Phase 05a: `phc_lite_loss_curve.png`, `phc_lite_val_policy_pnl_by_epoch.png`, `phc_lite_spearman_by_epoch.png`, `phc_lite_grad_norm_by_epoch.png`, `phc_lite_policy_comparison.png`, `phc_lite_session_pnl.png`, `phc_lite_causal_fill_comparison.png`, `phc_lite_oracle_action_confusion.png`, `phc_lite_score_bucket_lift.png`

如果繪圖程式碼裡確實沒有設`xlabel`/`ylabel`（很多quick-plot腳本會省略），請改用Python（PIL/matplotlib讀圖或直接open圖檔）目視確認圖上實際顯示的軸標籤文字，並回報你看到的文字。如果圖檔本身也沒有任何軸標籤文字，明確回報「圖檔本身無軸標籤，無法還原語意，僅能從變數命名推斷：xxx（推論）」。

輸出：`n2ew_era/000_context/AUDIT_chart_axis_semantics.md`，逐圖列出：圖檔名 / 來源腳本路徑+行號 / 還原出的xlabel / ylabel / title / 是否為推論。

### 任務2：PHC-lite的5個log1p/clip/zscore特徵精確定義

請在 `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/` 整個目錄搜尋資料前處理/feature engineering程式碼，找出PHC-lite用的16個特徵中，哪5個經過log1p/clip/zscore轉換，並抓出log1p底數、clip上下界（寫死常數或動態分位數）、zscore的mean/std計算範圍（全資料集/train split/rolling window）。

輸出：`n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_phc_lite_transform_features.md`，每個特徵附上函式名+行號+逐行公式。查無對應程式碼的特徵明確列出「查無此轉換邏輯」。

### 任務3：state_count vs target_valid_rows 換算關係

比對`04a2_product_filter_and_loader_contract_closure/PHASE04A2_PRODUCT_FILTER_AND_LOADER_CLOSURE_REPORT.md`的`target_valid_rows`欄位與`04b_txf_only_final_development_rerun/TXF_ONLY_FINAL_DEVELOPMENT_DECISION_REPORT.md`的`state_count`欄位（train=2,470,559、val=2,655,369、new3=5,656,822、new4=2,196,280）。找出各自程式碼路徑與filter邏輯差異，寫驗證腳本從原始資料重新計算，確認換算關係能否重現。

輸出：`n2ew_era/04b_txf_only_final_development_rerun/AUDIT_state_count_vs_target_valid_rows.md`，附實際跑出的驗證數字與誤差。

### 任務4：HGB三組max_iter全部停在n_iter=121的原因

在`03b6_hgb_convergence_closure/`找出iteration sweep與early-stopping audit程式碼，找出`tol`/`n_iter_no_change`/`validation_fraction`設定值，重跑三組max_iter（320/500/800）訓練，確認是否都停在n_iter=121，並用實際log/數字佐證原因。

輸出：`n2ew_era/03b6_hgb_convergence_closure/AUDIT_identical_n_iter_121.md`。

### 任務5：03a6特徵重建在session邊界的洩漏稽核

在`03a6_root_cause_feature_and_contract_closure/`找出`event_rate_past_50`/`spread_change_past`/`imbalance_change_past`等動態特徵在每個session第一筆事件的處理邏輯，寫審計腳本對全部NEW1-5資料集逐session抓出第一/第二/第三筆事件的這幾個特徵值，確認session內重置正確性與是否有跨session洩漏。

輸出：`n2ew_era/03a6_root_cause_feature_and_contract_closure/AUDIT_session_boundary_leakage.md`，附實際抓出的邊界事件數值表（每資料集至少3個session）與明確結論。

### 任務6：凍結split調參稽核機制（補強，不是真的能還原歷史）

對所有threshold/超參數相關設定檔檢查git歷史（`git log -p`），與NEW5解封時間做交叉比對，看是否有任何threshold/config在NEW5解封"之後"還有被修改的記錄。若git歷史不可用，明確回報「無法用git歷史重建調參時間序列，這條規則目前仍只能靠人工自律，沒有可稽核的技術證據」。

輸出：`n2ew_era/000_context/AUDIT_frozen_split_no_tuning_check.md`。

---

## 任務7（已撤回，不要做）

**狀態：gap_confirmed_resolved_already（2026-06-19查證）**。原任務假設「`phc_stp_v2_cosine_lr`從未跑過06系列HLS validation」不成立——`n2ew_era/06f_phc_lite_manual_hls/`已對此checkpoint完整跑過csim/csynth/cosim/Vivado P&R全流程且timing met。詳見`n2ew_era/000_context/C_CLASS_HLS_STATUS_AND_VERIFY_STEPS.md`。不需要、也不應該再交給Codex執行。

---

## 任務8（新增）：PHC-lite的可學習記憶機制是否真的優於固定窗口，而非過擬合/雜訊

**對應缺口**：`B_CLASS_RETROSPECTIVE_ANALYSIS.md` §B.2末段的Reviewer追問——「你怎麼知道PHC-lite真的學到了比固定窗口更好的記憶模式，而不是過擬合或學到雜訊？」目前文件只標注「未來工作」，使用者要求**現在就去找證據**，不能再用「未來工作」搪塞。

**鐵則重申**：這是要你**實際跑實驗**，不是讀程式碼推論。如果跑完實驗發現證據是負面的（PHC-lite沒有顯著優於固定窗口），要誠實寫出來，這仍然是`status: resolved`（問題已有答案）。

請依序做以下三件獨立但互補的驗證，三件都要做，不要只做一件就喊resolved：

### 8.1 學習到的decay/gain參數是否有結構性訊號（而非塌縮成trivial值）

讀取`05a_phc_lite_microstructure_sequence_feasibility/`（或05c系列，若05a的checkpoint不存在以實際存在的最早完整訓練checkpoint為準）已凍結的3個seed checkpoint，抓出`state_temporal`遞迴公式裡的`decay[i]`（16個channel，3個seed）：
- 計算每個seed的decay分布（min/max/mean/std），換算成等效記憶半衰期（`half_life = ln(0.5)/ln(decay)`，注意decay若是sigmoid輸出要先還原到(0,1)區間）。
- 跨3個seed比較：16個channel的decay值排序/分布是否在不同seed間保持一致的結構（例如某幾個channel穩定學到長記憶、某幾個穩定學到短記憶），還是每個seed學出完全不同、無規律的隨機排列？
- **如果decay全部塌縮在同一個值附近（例如全部接近1.0=幾乎不衰減，或全部接近0=幾乎無記憶）**，這是負面證據：代表模型可能沒有學到「依通道區分的動態記憶」，與B.2的論述（可學習記憶長度）矛盾，必須誠實回報。
- **如果跨seed的decay分布有結構一致性**（例如用Spearman相關係數比較3個seed的16維decay向量排序，相關係數顯著>0），這是正面證據：代表學到的記憶模式不是隨機初始化的雜訊殘留，是訓練收斂到的穩定結構。

### 8.2 固定窗口 ablation：凍結decay/gain，比較PnL

在`05a`（或最早可用版本）的訓練程式碼上做一個**最小修改**的ablation變體：把`decay[i]`從可學習參數改成固定常數（不參與梯度更新），固定值用以下兩種設定各跑一次：
(a) 用03a6既有的固定窗口統計量對應的等效衰減率（例如`event_rate_past_50`隱含窗口=50個事件，換算成等效decay常數）
(b) 一個合理的中間值（例如decay=0.9，对应約7個時間步半衰期，作為「合理但非學習出來」的基準）

其他超參數（學習率、epoch數、特徵輸入、loss function）**完全不變**，只凍結這一個機制，訓練同樣的3個seed，比較val/new3/new4的`total_excess_pnl_H20`與HGB/queue baseline。
- 若可學習decay版本（已凍結報告的05a數字）顯著優於兩個固定decay版本（同樣的train/val/new3/new4比較），這是B.2論述的直接正面證據。
- 若固定decay版本表現相近甚至更好，這代表可學習機制可能沒有實質貢獻，必須誠實寫出，不要為了支持B.2的論述而選擇性報告。

**重要**：這是訓練實驗，需要記錄完整訓練log（loss曲線、最佳epoch、3 seed的個別結果與平均），不能只報告最終一個數字。

### 8.3 時序打亂測試（sanity check：模型是否真的在用時序資訊）

用已凍結的05a PHC-lite checkpoint（不重新訓練，純推論），對驗證集做兩組推論：
(a) 原始時序輸入（baseline）
(b) 把每個session內的tick順序隨機打亂後再餵進模型（破壞真實時序結構，但保持same set of events）

比較(a)(b)兩組的`total_excess_pnl_H20`與`spearman`差異。
- 若打亂後PnL/spearman明顯下降（模型對時序順序敏感），這是模型確實在利用時序資訊的證據。
- 若打亂後表現幾乎不變，這代表模型可能沒有真正利用序列依賴關係（recurrent state可能只是學到了一個跟順序無關的aggregate statistic），這是負面證據，必須誠實回報，且這會直接削弱B.2「PHC-lite學到動態記憶」的論述。

輸出：`n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md`，包含8.1/8.2/8.3三節，每節附實際數字、log路徑、明確結論（正面/負面/混合證據），並在文件最後給出一段「對B_CLASS §B.2這句Reviewer追問的最終誠實回答」（不超過200字，直接可以複製貼到答辯稿）。

---

## 任務9（新增）：n2ew_era的新評估方法論本身，有沒有被獨立驗證過效果是好的

**對應缺口**：使用者追問——「所以我們可以說在N2EW_ERA[採用的這個]未[經外部]評估方法做出這樣的修正效果是好的嗎？」換句話說：`B_CLASS_RETROSPECTIVE_ANALYSIS.md` §C論證了new_era的「死胡同」根因是評估指標（session-final/quote-level markout）系統性高估了可執行edge，n2ew_era因此把causal-fill sequential PnL設為主要優化目標。**但這個「換評估方法」本身的修正效果，有沒有被任何獨立的、事前沒看過的資料驗證過是真的有效（而不是又一次自我參照的方法論信念）？**

這是一個比任務8更尖銳的問題：不是問「PHC-lite好不好」，是問「你賴以判斷一切的這個尺，本身被驗證過嗎」。

請依序查證：

### 9.1 列出n2ew_era裡所有「用causal-fill sequential PnL在train/val上選出模型/策略，然後在從未用於選擇的frozen split上驗證」的案例

逐phase搜尋（特別注意`02_short_horizon_sequential_proxy_baselines/03_frozen_test_new4_evaluation.py`、`04b_txf_only_final_development_rerun/`、`05a`~`05d`的NEW3/NEW4診斷、`05b`/`05b1`的NEW5事後診斷），列出一個表格：

| Phase | 選擇依據（train/val用的指標） | Frozen驗證集 | 該模型/策略在frozen集的causal-fill sequential PnL是否與train/val符號一致 | 出處 |

**這裡的鐵則**：`05b`/`05b1`明確自己聲明是「事後診斷（非新鮮測試）」，**不能算作獨立驗證**，必須在表格裡標注清楚，不能混進「通過的案例」裡灌水樣本數。真正算獨立驗證的，必須是模型/策略的選擇過程完全沒有用過該split的任何資訊（包括沒有用該split來做事後診斷後微調）。

### 9.2 計算「符號一致率」並誠實評估樣本量是否足夠下結論

根據9.1的表格，計算：在所有真正符合「未曾用於選擇」定義的frozen驗證案例中，train/val正向結果在frozen集上維持正向的比例是多少。
- 如果案例數**太少（例如<5個獨立案例）**，誠實回報「樣本量不足以下統計顯著的結論，這是方法論驗證本身的一個缺口」——**不要硬幫使用者編出一個看起來有說服力的百分比**。
- 如果案例數足夠，給出符號一致率，並與舊new_era方法論（session-final/quote-level markout）在phase 38之前的同類案例（如果能找到對應的「舊指標選出來、後來在phase38用新指標重算發現翻負」的明確配對）做對比——這是任務的核心：**新方法論的「自我修正」這件事本身，有沒有在獨立資料上展現出比舊方法論更可靠的預測能力，還是只是「目前還沒被同樣的方式打臉，但也沒被證明真的更準」？**

### 9.3 誠實結論

明確回答使用者的問題：「我們可以說n2ew_era的修正效果是好的嗎？」——可能的誠實答案類型（請依實際查證結果選擇，不要預設答案）：
(a) 「有限但方向正確的證據」：有N個獨立frozen案例，符號一致率為X%，優於舊方法論已知的失敗率，但N太小不足以做強結論
(b) 「目前缺乏真正獨立的驗證」：n2ew_era目前的frozen測試（NEW4/test）雖然形式上是held-out，但都還在同一個資料蒐集窗口/市場condition內，還沒有經過時間上更遠、condition更不同的真正outof-distribution測試
(c) 其他你查證後發現的更精確的答案

輸出：`n2ew_era/000_context/AUDIT_evaluation_methodology_external_validation.md`，含9.1表格、9.2計算過程、9.3誠實結論（給一段可直接用於答辯的200字以內摘要）。

---

## 任務10（新增）：Causal-fill sequential oracle的「可執行上限」推導與其合理性

**對應缺口**：`B_CLASS_RETROSPECTIVE_ANALYSIS.md` §D.2——使用者追問「這個可執行上限怎麼來的？為什麼這樣可以代表實際可執行上限？」目前文件只引用了數字（~3.82M-3.92M），沒有解釋推導邏輯本身是否嚴謹。

請找到計算causal-fill sequential oracle（`conservative_l2_stress` fill model, inventory_cap=1, horizon=20）的實際程式碼（應該在`04c1_new5_oracle_headroom_analysis/`或`38_sequential_proxy_inventory_backtest/`，請先grep`conservative_l2_stress`與`causal.fill.*oracle`等關鍵字確認確切位置），逐項查證：

### 10.1 求解方法是否真的是全局最優，還是貪婪近似

找出oracle action選擇的演算法實作（函式名稱+行號）。明確判斷：
- 這是一個對所有未來資訊已知、在inventory_cap=1與fill model約束下求解的**動態規劃/精確最優化**（理論上保證全局最優），還是一個**貪婪/啟發式近似**（例如逐步往前看一步就決策，不保證全局最優）？
- 如果是貪婪近似，**這個「上限」其實不是真正的數學上限，只是一個下界估計的下界**（因為更好的求解器可能找到更高的可執行PnL）——這點必須在文件裡明確揭露，現有措辭「代表實際可執行上限」可能需要改成「目前已知求解方法下的可執行PnL估計」。

### 10.2 約束假設的敏感度

`inventory_cap=1`與`conservative_l2_stress`兩個假設本身是怎麼決定的（查設定來源：是否有引用任何實際交易所限制、風控規則，還是研究者自行設定的便利假設）？如果程式碼裡能找到對這兩個假設做敏感度分析（例如inventory_cap=2/3時oracle PnL如何變化、換用不同stress程度的fill model時PnL如何變化），請列出數字。如果完全沒有敏感度分析，明確回報「目前只有單一假設組合下的數字，沒有敏感度分析，這是reviewer可能追問但目前答不出來的點」。

### 10.3 "代表實際可執行上限"這句話的精確性修正

根據10.1/10.2的查證結果，給出一個**修正後、更精確的措辑**，明確區分：
- 若求解是精確最優：「在`conservative_l2_stress`fill model與`inventory_cap=1`假設下的精確可執行上限」（仍要強調是conditional on這組假設，換一組假設數字會變）
- 若求解是近似：「在上述假設下、用[具體演算法名稱]求解得到的可執行PnL估計，不保證是該假設下的全局最優上限」

輸出：`n2ew_era/04c1_new5_oracle_headroom_analysis/AUDIT_causal_fill_oracle_derivation.md`，含10.1/10.2/10.3，並在文件最後給出可直接更新進`B_CLASS_RETROSPECTIVE_ANALYSIS.md` §D.2的修正段落文字（标注清楚這段要拿去取代原文件哪一段）。

---

## 明確不要做的事（B類缺口，請不要試圖"補完"）

以下是文件本身的固有缺口，已經過exhaustive搜尋確認查無依據，**請不要**為了把報告填得更完整而生出查無出處的答案：
- PHC的逐字全名展開、PHCSSM論文逐項公式對照（已在`B_CLASS_RETROSPECTIVE_ANALYSIS.md` §A解決，不需要重做）
- 「為什麼一開始決定先試PHC而非LSTM/GRU/Transformer」這個人為決策過程（已確認是opportunistic adoption，誠實事實，不要事後生出比較分析）
- TCN/HLS主線與quote-control/residual ML veto主線誰先誰後
- PROJECT_RESET.md的官方reset理由是否真的與TCN資源超限有因果連結

如果你在做任務8/9/10時**意外**翻到了能佐證或推翻上述任一點的新證據，請照樣附出處回報，但不要主動去找。

## 完成後

請在每個輸出報告檔案的第一行加上 `status: resolved` 或 `status: gap_confirmed_unresolvable`（任務8/9/10若結果為負面，仍寫`status: resolved`，並在開頭加一行`finding: negative`或`finding: mixed`/`finding: positive`，方便使用者快速辨識）。所有任務完成後，於`n2ew_era/000_context/`下新增`GAP_CLOSURE_SUMMARY.md`，逐任務（1-6, 8, 9, 10）列出一行結論（任務編號 / 是否resolved / finding方向（若適用） / 輸出檔案路徑），方便使用者快速核對與回填進`N2EW_ERA_EVOLUTION_AND_INHERITANCE.md`與`B_CLASS_RETROSPECTIVE_ANALYSIS.md`。
