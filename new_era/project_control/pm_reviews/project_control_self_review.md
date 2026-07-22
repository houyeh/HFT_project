# Project Control Self-Review

## 1. Project Control 完整度
**完整度極高。** 已經建立了涵蓋實驗狀態、代理守則、簡報邏輯、數據來源與環境稽核共 15 份核心文件。新 Agent 只要讀取 `README_FOR_NEW_AGENT.md` 和 `00_PROJECT_STATE.md` 就能立刻掌握全貌。

## 2. 最重要的檔案
- `00_PROJECT_STATE.md` (統一數據源，防止 Agent 亂編數字)
- `04_FORBIDDEN_CLAIMS.md` (護欄，防止 Agent 過度吹噓)
- `05_CURRENT_SLIDE_STORY.md` (與指導教授 / 團隊報告的共識)
- `README_FOR_NEW_AGENT.md` (Agent 進入專案的第一個進入點)

## 3. 需人工再確認的數字
- 018 Queue-aware Validation 因缺乏真實 `aggressor_side` 欄位，後續所產生的「保守估計 Proxy 數字」必須由人工確認是否足夠合理（例如 fill probability haircut 是否設定得當）。
- EWMA vs Rolling Trend 在 017 報告中的取捨（EWMA NEW3 PnL = -132,325.0），如果團隊決定未來回歸 Rolling，需再次人工確認計算。

## 4. Allowed / Forbidden Claims 是否矛盾
**無矛盾。** 
- Allowed 允許說「0.55 在 NEW 上有 local edge（贏過 random+1std）」
- Forbidden 禁止說「0.55 是 production threshold」或「策略已經 production-ready」
這兩者完美契合研究現狀：有科學上的 signal，但工程與總體風控上不可靠。

## 5. 目前 Next Task 是否應為 018 Queue-aware Validation
**是的。** 
目前 016 (Binary Label) 證明沒贏過 random+1std，017 (Auto Gate) 被證明打不贏 Static Oracle。專案要能「止血」並給出最終報告，最需要的是 018 的 Queue-aware 壓力測試。如果 018 撐過 proxy 壓力測試，專題就能立足於「保守風控硬體」的地位。

## 6. 建議：先更新簡報還是先跑 018？
**建議同步進行，但以「更新簡報 (Week 13)」為優先。**
因為目前 `05_CURRENT_SLIDE_STORY.md` 的故事線已經非常完整，且能完美包裝 015、016、017 的發現（包括 valuable negative results）。先固定住這份誠實的簡報，再去跑 018 來當作 Future Work / MVP 結論，節奏最穩。

## 7. Repo Cleanup 哪些只能 Audit 不能刪
- **只能 Audit，嚴禁刪除**。
- `10_CLEANUP_PLAN.md` 已經將 `.pycache`、`.pytest_cache`、舊版重複資料夾（如 `015_models_playaround_result_expC_only/`）列入「待確認可封存/刪除」清單，但在未獲得明確 Human 授權前，絕對不更動現有結構。

## 8. 給 Codex 的下一步任務摘要
- **任務**：準備執行 018 Queue-aware sensitivity sweeps。
- **指示**：讀取 `018_queue_aware_validation/scripts/run_queue_sensitivity.py`，請注意沒有 `aggressor_side`。需輸出各 proxy 參數下的保守 net_pnl。禁止重訓模型，禁止覆蓋舊資料夾。

## 9. 給 Slide Agent 的下一步任務摘要
- **任務**：更新 Week 13 簡報。
- **指示**：嚴格依照 `05_CURRENT_SLIDE_STORY.md` 的大綱，將 `new_era/project_control/` 裡的數字對應到 HTML / PPTX 生成腳本，不可創造不存在的數字，尤其注意 Latency 不能寫成「3 ticks 後崩盤」。
