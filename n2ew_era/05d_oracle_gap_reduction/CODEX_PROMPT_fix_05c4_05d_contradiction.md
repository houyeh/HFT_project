# Codex Agent 工作指令：修復 05c4 與 05d 之間的版本判定矛盾

## 背景（給 Codex，不要省略，避免它自己亂猜原因）

`n2ew_era/05c4_phc_stp_cosine_lr/PHC_STP_V2_COSINE_FINAL_DECISION_REPORT.md` 第3行明確寫著：

```
Final status: `stp_v2_adopted_as_best_sequence_model`.
```

該報告第89~98行的 Adoption Gates 表格顯示 `phc_stp_v2_cosine_lr` 這個模型（val/NEW3/NEW4 三個 split）在以下六項門檻**全部通過**：
Convergence closed、Seed stability、Matched random、OOS improvement、Causal-fill improvement、Oracle capture improvement。

但 `n2ew_era/05d_oracle_gap_reduction/07_oracle_gap_reduction_summary.py` 是「整個05系列最終誰該被採用」的彙整腳本，它跑出來的 `PHC_ORACLE_GAP_REDUCTION_SUMMARY_REPORT.md` 結論卻是 `keep_phc_lite_v1`（也就是完全沒提到05c4這個已經自稱「adopted」的模型）。

## 我已經親自讀過原始碼確認的根因（不是猜測，是逐行核對過的事實）

`07_oracle_gap_reduction_summary.py` 第123~152行的 `configs` list 硬編碼了只有四個模型：

```python
configs = [
    ("phc_stp_120epoch", ...),
    ("phc_rank_head", ...),
    ("phc_uncertainty_gating", ...),
    ("phc_streaming_dense", ...),
]
```

`phc_stp_v2_cosine_lr`（05c4 的模型）**完全沒有被加進這個 list**。所以第168~172行的 `adopted` mask（用字串 `"candidate_improves_v1|adopted_as_phc_v2"` 去比對 `adopted_status` 欄位）永遠不會看到05c4的資料，因為這個模型的資料根本沒被讀進 `summary` dataframe。最終 `recommendation` 因此預設落到 `keep_phc_lite_v1`。

這不是統計結論上的矛盾，是**pipeline遺漏（一個模型沒被加進彙整清單）**造成的工程bug。

## 你（Codex）要做的事

1. **先自己重新核對一次我上面的說法，不要直接相信我**：
   - 打開 `n2ew_era/05d_oracle_gap_reduction/07_oracle_gap_reduction_summary.py`，確認 `configs` list 目前的內容，逐一確認裡面有沒有任何一項對應到 05c4 / `phc_stp_v2_cosine_lr`。
   - 打開 `n2ew_era/05c4_phc_stp_cosine_lr/results/` 底下的檔案，列出實際存在的檔名（特別是 `phc_stp_v2_oos_eval.csv`、`phc_stp_v2_causal_fill_eval.csv`、`phc_stp_v2_window_oracle_capture.csv`、`phc_stp_v2_vs_v1_hgb_queue.csv` 這幾個我看到的檔案是否存在，檔案內欄位名稱是否跟 `model_rows()` 函式期望的欄位名稱一致）。
   - 確認 `infer_status()` 函式（第76~111行）對於一個新加入的 `phc_stp_v2_cosine_lr` 模型會怎麼分類——它目前的邏輯是用 `action_pnl >= v1_action` 和 `causal_fill_pnl >= v1_causal` 來判斷，這個邏輯本身是否能正確反映05c4報告裡『六項門檻全過』的結論，還是需要特殊處理（例如05c4用的是「6項gate全通過」而非單純PnL比較，要不要也把這個邏輯一併接進來，由你判斷並在交付時說明取捨）。

2. **修復方式（建議，但若你發現更好的做法可以調整，調整前先說明理由）**：
   - 在 `configs` list 中新增一筆 05c4 的設定，路徑對應到 `n2ew_era/05c4_phc_stp_cosine_lr/results/` 底下實際存在的檔案。
   - 確認 `model_rows()` 讀取這些檔案後產出的 `action_pnl`、`causal_fill_pnl`、`oracle_capture` 欄位數值，能否與 `PHC_STP_V2_COSINE_FINAL_DECISION_REPORT.md` 第84~87行表格中的數字（val: action_pnl=2.03792e+06, causal_fill_pnl=171835, oracle_capture=0.264287；new3/new4同理）完全吃合（容許浮點誤差），這是你交付前**必須**自行驗證的一步，不要只看程式碼能跑就算過。
   - 重新執行整個 `07_oracle_gap_reduction_summary.py`，產出新的 `PHC_ORACLE_GAP_REDUCTION_SUMMARY_REPORT.md` 與 `phc_oracle_gap_reduction_summary.csv`。

3. **絕對不要做的事**：
   - 不要為了讓「結論看起來合理」而手動修改 `infer_status()` 的判斷邏輯使其偏向某個你覺得「應該」的答案。判斷邏輯只能基於既有的、文件中已經定義好的gate規則調整，不能臆測。
   - 不要刪除或覆蓋 `phc_stp_120epoch`、`phc_rank_head`、`phc_uncertainty_gating`、`phc_streaming_dense` 這四個既有config，只能新增。
   - 不要動用 NEW5 資料做任何模型選擇（05d報告開頭明文寫「NEW5 is not used for model selection or threshold selection in this phase」，這條規則要繼續遵守）。
   - 如果重新跑出來的結果仍然不支持「採用05c4」，**不要為了符合我的預期而硬湊**，誠實回報實際數字與結論，即使結論變成「05c4依然不該被採用，但理由跟原報告寫的不同」。

4. **交付時請附上**：
   - 修改前後 `configs` list 的diff。
   - 重新計算出的05c4各split數值，與05c4原報告數值的逐項比對表（證明你的修復是對的，不是又一次巧合）。
   - 新的 `PHC_ORACLE_GAP_REDUCTION_SUMMARY_REPORT.md` 全文。
   - 一段你自己的結論：這次修復後，05c4 / `phc_stp_v2_cosine_lr` 究竟有沒有在05d的彙整邏輯下被判定為「adopted」？如果答案跟05c4報告本身的「stp_v2_adopted_as_best_sequence_model」不一致，請明確指出是哪個環節（門檻定義不同？資料口徑不同？）造成不一致，不要含糊帶過。

## 為什麼要交給你做（給你判斷優先順序的背景）

這個專案的治理規則是：任何「最終該凍結哪個模型版本」的結論都必須先過5層baseline比較與matched-random顯著性檢定，且禁止用NEW5做模型選擇。05c4自己內部的判定是乾淨的（六項gate都查得到具體數字），但因為05d的彙整腳本漏掉它，導致目前對外（包括給教授看的文件）只能誠實地說「PHC-lite v1是正式凍結版本，05c4理論上更好但尚未被正式納入決策」。這個bug修好之後，我才能在文件裡把這句話改成一個有依據的明確結論，而不是含糊的「矛盾待解決」。
