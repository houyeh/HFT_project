# **關鍵結論 2026_0506_2359**

- 目前 `NEW_ERA` 主線 TCN 的結果，我認為**暫時不能直接信任**。最大問題不是模型本身，而是資料時間軸與回測設計。`load_new_tw_data()` 只把 `HH:MM:SS:xxxxxx` 轉成「當日內 ns」，但它會把 `data/NEW` 底下多個 session / 多天檔案全部合併後只用 `timestamp` 排序；我確認 `data/NEW` 內實際有 `20260123Day`、`20260123Night`、`20260126Day`、`20260127Night` 等多個資料夾，因此不同日期的 08:45、13:30 會被混在一起。這會直接污染 rolling z-score、window、label、backtest chronology。參考 [01_feature_importance.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/01_feature_importance.py:19>)、[01_feature_importance.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/01_feature_importance.py:26>)、[01_feature_importance.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/01_feature_importance.py:75>)。

- `05_train_tcn_pytorch.py` 的 train/val 切分和 `09_mm_backtest.py` 的「尾段回測」**不是獨立 OOS**。`05` 是把全部 window 隨機打散後 `shuffle=True` 切 80/20；我重建後確認，`09` 使用的 TXFB6 尾端 114,091 筆樣本中，**91,135 筆，約 79.88% 曾經進過訓練集**。再加上 window 長度 100、`sample_stride=5`，相鄰樣本本來就高度重疊，等於 train / val / backtest 彼此都有很重的資訊洩漏。參考 [05_train_tcn_pytorch.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/05_train_tcn_pytorch.py:50>)、[datasets.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/src/training/datasets.py:136>)。

- `09_mm_backtest.py` 的 replay 邏輯本身也有偏誤。dataset 是 `sample_stride=5` 取樣，但回測把相鄰 sample 當成「next tick」來成交；也就是說 fill 判斷不是逐 quote，而是隔 5 個事件看一次。更嚴重的是 `tcn_alpha` 把 `my_bid = my_ask` / `my_ask = my_bid` 當成 aggressive，但實際上只有在**下一個 sample 還對你有利時才算成交**，這相當於給策略免費選擇權，不是真正的 taker execution。這很可能就是 current `09` 出現 `+520,145,216 TWD` 這種異常好看 PnL 的主因之一。參考 [datasets.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/src/training/datasets.py:149>)、[09_mm_backtest.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/09_mm_backtest.py:63>)、[09_mm_backtest.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/09_mm_backtest.py:80>)、[09_mm_backtest.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/09_mm_backtest.py:87>)。

- 兩個 `09` 的 accuracy / PnL 不能直接拿來互相比。`09_mm_backtest.py` 用的是 current NEW data 的 TXFB6 尾段；`09_mm_backtest_old_data.py` 用的是舊 replay `TXFJ5` 全資料、且舊 loader 會 `dedupe_quotes=True`，事件時鐘與 quote 密度都不同。參考 [09_mm_backtest.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/09_mm_backtest.py:129>)、[09_mm_backtest.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/09_mm_backtest.py:134>)、[09_mm_backtest_old_data.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/09_mm_backtest_old_data.py:140>)、[tw_replay.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/src/data/tw_replay.py:124>)。

- 目前分類指標也有誤導性。我實跑得到：
  `06_evaluate_tcn.py` validation accuracy = `45.76%`。
  `09_mm_backtest.py` 那段尾端資料 accuracy = `40.02%`，但 alpha PnL 卻是 `+520.1M TWD`。
  `09_mm_backtest_old_data.py` accuracy = `76.75%`，但 old replay 的 label 其實有 `93.5%` 都是 stationary，所以這個 accuracy **連永遠猜 stationary 都打不過**，PnL 也確實是負的。這表示現在的分類 loss 跟 MM PnL 目標明顯失配。

**目前 TCN 主線機制**

- feature decision 先從 `execution_aware` 12 個特徵出發，再用 RandomForest 在 100-tick window 的**最後一個 snapshot**做 importance ranking，不是 sequence-aware ranking。最後固定選 top 8：`micro_return`、`trade_intensity_short`、`queue_imbalance_change`、`deep_obi`、`net_cancel_pressure`、`queue_imbalance`、`weighted_mid_deviation`、`obi`。參考 [01_feature_importance.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/01_feature_importance.py:93>)、[01_feature_importance.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/01_feature_importance.py:113>)、[02_feature_engineering.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/02_feature_engineering.py:49>)。

- feature engineering 的 base 來自 `build_tw_feature_frame()`：dimensionless 5 個核心特徵 `normalized_spread / micro_return / obi / weighted_mid_deviation / deep_obi`，再加 execution-aware 特徵如 `queue_imbalance`、`queue_imbalance_change`、`aggressor_imbalance_interval`、`rolling_signed_flow_ratio`、`trade_intensity_short`、`net_cancel_pressure`、`cancel_pressure_short`。trade side 是靠 quote rule + mid rule + tick rule 推 aggressor sign，再 backward / searchsorted 對齊到 quote。參考 [tw_train_helpers.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/src/utils/tw_train_helpers.py:43>)、[dimensionless_features.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/src/features/dimensionless_features.py:88>)、[execution_features.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/src/features/execution_features.py:25>)、[execution_features.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/src/features/execution_features.py:84>)。

- z-score 是 rolling window `2000`、`min_periods=100`、`std=0 -> 1e-5`、`NaN -> 0`、最後 `clip[-10,10]`。這只套在 top 8 上。參考 [02_feature_engineering.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/02_feature_engineering.py:21>)。

- sample/label 生成是 `window_size=100`、`horizon=10`、`threshold_ticks=1.0`、`sample_stride=5`。label 規則是 `future_mid - current_mid > 1 tick => Up(2)`，`< -1 tick => Down(0)`，其餘 `Stationary(1)`。參考 [02_feature_engineering.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/02_feature_engineering.py:83>)、[datasets.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/src/training/datasets.py:115>)、[sliding_window.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/src/features/sliding_window.py:18>)。

- 主線模型是 `HLSFriendlyTCNClassifier`。輸入 `(B,8,100)`，3 個 block，dilation `1/2/4`，每個 block 兩層 causal `Conv1d(kernel=3)` + `ReLU` + `Dropout(0.1)`，第一個 block 有 residual `1x1 downsample`，最後 `AvgPool1d(100)` + `Linear(16->3)`。trainable params 我算是 `4,515`，code 中 receptive field 公式算出 `29 ticks`。參考 [tcn.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/src/models/tcn.py:301>)、[tcn.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/src/models/tcn.py:387>)、[tcn.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/src/models/tcn.py:443>)。

- 主線訓練在 `05_train_tcn_pytorch.py`：TXFB6 與 MXFB6 合併後總共 `1,489,205` 個 sample，先轉 `(B,8,100)`，再 `train_test_split(test_size=0.2, random_state=42, shuffle=True)`；class weights 是 inverse-frequency，我重建得到 `{0: 2.0781, 1: 0.4906, 2: 2.0808}`；loss 是 `CrossEntropyLoss(weight=...)`，optimizer 是 `Adam(lr=1e-3)`，batch size `1024`，epoch `20`，以最佳 `val_acc` 存成 `005_tcn_weights.pth`。參考 [05_train_tcn_pytorch.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/05_train_tcn_pytorch.py:45>)、[05_train_tcn_pytorch.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/05_train_tcn_pytorch.py:55>)、[05_train_tcn_pytorch.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/05_train_tcn_pytorch.py:69>)、[05_train_tcn_pytorch.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/05_train_tcn_pytorch.py:80>)、[05_train_tcn_pytorch.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/05_train_tcn_pytorch.py:83>)。

- `06_evaluate_tcn.py` 沿用**同一個 random split** 去評估，所以它不是真正 walk-forward OOS。參考 [06_evaluate_tcn.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/06_evaluate_tcn.py:43>)。

- 另外還有三條分支：`11_shrink_tick_size.py` 是 30-tick 版，`shuffle=False`；`12_qat_tcn.py` 是從頭訓練 QAT TCN，加入 `QuantConv1d + BN + QuantReLU`，也 `shuffle=False`；`13_fine_tune_qat.py` 則是把 FP32 權重手動映射到 QAT 後，再用 `lr=1e-4` fine-tune 10 epochs。這三條都存在，但目前 `09_mm_backtest.py` 用的仍然是主線 `005_tcn_weights.pth`。參考 [11_shrink_tick_size.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/11_shrink_tick_size.py:33>)、[12_qat_tcn.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/12_qat_tcn.py:153>)、[13_fine_tune_qat.py](</c:/Users/holbo/00VS_projjects/00_HFT_project/NEW_ERA/13_fine_tune_qat.py:223>)。

**下一步規劃**

- 先修資料軸，不先修這個，後面所有調參都沒有意義。做法是把 `load_new_tw_data()` 改成保留 `trading_date/session_id/file_order`，建立真正單調遞增的 composite timestamp，並且**每個 session 分開做 rolling z-score、window、label**，禁止跨日跨 session 串接。

- 把 train/val/test 改成**先按 session/day 切分，再做 window**。最低標準是 walk-forward；更好的做法是 `train=早期日子`、`val=中間日子`、`test=最後日子`。不要再用 window level 的 random shuffle。

- 重寫 backtest。maker fill 請至少回到 raw quote event；如果 inference 每 5 tick 更新一次，可以接受，但 fill simulation 不能每 5 個 sample 才跳一次。`tcn_alpha` 如果定義成 aggressive，就要**當下成交**，不能用「下一個 sample 還沒跑掉才算成交」的 free option 邏輯。

- 重新定義評估指標。除了 accuracy，至少同時看 `majority baseline`、`balanced accuracy`、`per-class recall`、`PnL / fill rate / inventory path / adverse selection / spread capture`。現在的 40.0% accuracy 卻有 +520M PnL、76.8% accuracy 卻大虧，已經明確顯示 objective misalignment。

- feature decision 建議再做一次，但要改成**訓練集內**完成，而且盡量 sequence-aware。現在 `01_feature_importance.py` 是用最後一個 snapshot 的 RF ranking，適合當 heuristic，不適合當 TCN 最終 feature 決策。

- 如果你要，我下一步可以直接幫你做兩件事中的一件：  
  1. 修 `load_new_tw_data()` 與 session-aware dataset split。  
  2. 修 `09_mm_backtest.py`，把 fill logic 改成真正逐 event replay。


# 2
主要改動：


新增 tcn_pipeline_utils.py (line 1)，集中管理 top 8 features、rolling z-score、session-aware split、class weights、MM fill simulation。

修正 01_feature_importance.py (line 19)：data/NEW 現在按資料夾建立 session_id，timestamp 會加 session offset，避免多天同時間混排。

修正 02_feature_engineering.py (line 52)：feature engineering、rolling z-score、window/label 都改成逐 session 建立，輸出的 npz key 會是 SYMBOL__SESSION_X/y。

修正 05_train_tcn_pytorch.py (line 35) 與 06_evaluate_tcn.py (line 32)：移除 random window split，改為 session-level train/val/test split。

重寫 09_mm_backtest.py (line 1)：只跑真正 held-out test sessions，回測 sample stride 改為 1，alpha 主動成交改為當下 best ask/bid 成交，不再等下一筆報價才「有利成交」。

同步修正 09_mm_backtest_old_data.py (line 1) 的成交邏輯，讓 old replay 與 current NEW data 用同一把 PnL 尺。

同步對齊 11_shrink_tick_size.py (line 20)、12_qat_tcn.py (line 127)、13_fine_tune_qat.py (line 36)，避免 QAT / 30-tick 路徑仍沿用舊的 random split。

# 3: 回答我就好，剛剛有測試過feature importance了嗎 要重跑嗎？直接按開始按鈕就可以重跑了嗎? 是要先測試各featrue 的importance再開始跑02 05 06的訓練嗎 01的做法(判斷一個訊號種不重要的詳細流程)、邏輯與意義是?有做的必要性嗎?現在的做法有沒有問題?

有，剛剛我已經看過 `01_feature_importance.py` 的流程，也讀了現成的 `001_feature_importance.csv`，但**沒有在你現在這版修正後重新完整重跑 `01`**。因為我剛剛改了 `01/02/05/06` 的資料切分與 session 邊界，**舊的 feature importance 結果已經不完全對應新的資料生成方式**，所以如果你要走新流程，`01` 建議重跑一次。

你可以直接按開始按鈕重跑，但順序上我建議是：

1. 先跑 `01_feature_importance.py`
2. 再跑 `02_feature_engineering.py`
3. 再跑 `05_train_tcn_pytorch.py`
4. 再跑 `06_evaluate_tcn.py`

原因很簡單：`01` 的結果會影響 `02` 要選哪些 top features。現在我們把資料改成 session-aware 之後，`01` 不只是「算重要性」，它還是在替後面的 `02` 提供 feature selection 的依據，所以最好先重跑。

`01` 的邏輯與意義是這樣：

- 它先把原始 LOB + trade 資料轉成一批微結構特徵。
- 然後用 sliding window 取出一段一段樣本。
- 但它不是拿整段序列去做 importance，而是只拿每個 window 的**最後一個 snapshot** 當作隨機森林的輸入。
- 最後用 RandomForest 的 Gini importance 去排出哪些 feature 對分類最有幫助。

它的用途不是為了直接訓練 TCN，而是做一個「先驗篩選」，幫你把特徵從一堆候選中縮到 8 個，降低噪音，也降低硬體成本。

有沒有必要做？  
有，但它是**工程上有用、統計上不夠嚴謹**的工具。

它的好處是：

- 可以快速縮小特徵空間
- 對 FPGA / HLS 很有幫助，因為 feature 少，後面整條管線都比較輕
- 可以讓你先看到哪些訊號真的有解釋力

但現在的做法也有幾個問題：

- 它用的是 RandomForest + 單一 snapshot，不是 TCN 的序列視角
- feature importance 不是因果，也不是穩定排序，只是「在這個資料切法下，這些欄位比較能分開類別」
- 如果資料切分不乾淨，importance 也會被污染
- 它選出來的 top 8 比較像是 heuristic，不應該被當成真理

所以我的判斷是：

- **`01` 要保留**
- **`01` 現在要重跑**
- 但不要把 `01` 當成最終裁判，它只是前處理與降維的工具
- 真正要驗證好不好，還是要看 `05/06` 的 session-aware 訓練與 `09` 的 out-of-sample 回測

如果你現在要最實際的建議，我會說：

- 先直接重跑 `01`
- 再跑 `02 -> 05 -> 06`
- 等這條主線跑完，再看 `09` 的 PnL 是否合理
- 如果 `01` 的 top 8 和現在差很多，才考慮改 feature set

目前這版 `01` 的做法不是不能用，但它以前最大的風險是資料切分不乾淨。那個問題我已經往 session-aware 方向修了，所以**現在重新跑 `01` 是有意義的**。

# 4 == Feature Importance Ranking ===                      Feature  Importance                          obi    0.272258              queue_imbalance    0.198800       weighted_mid_deviation    0.160398        trade_intensity_short    0.084955                     deep_obi    0.077876       queue_imbalance_change    0.070334            normalized_spread    0.057761        cancel_pressure_short    0.032954    rolling_signed_flow_ratio    0.018455                 micro_return    0.010821          net_cancel_pressure    0.010242 aggressor_imbalance_interval    0.005148  單純回答我，這跟目前的結果有不一樣嗎(忘記備份舊版的了  如果不一樣的話要怎麼做，如果要用新版的跑要改哪裡的那些程式?

有，不一樣，而且差很多。

你現在這版排序是：

- `obi`
- `queue_imbalance`
- `weighted_mid_deviation`
- `trade_intensity_short`
- `deep_obi`
- `queue_imbalance_change`
- `normalized_spread`
- `cancel_pressure_short`
- `rolling_signed_flow_ratio`
- `micro_return`
- `net_cancel_pressure`
- `aggressor_imbalance_interval`

跟你之前那份 `001_feature_importance.csv` 比，原本是 `micro_return` 壓倒性第一，現在變成 `obi` 和 `queue_imbalance` 這類 book imbalance 特徵領先，這是實質變化，不只是細微波動。

如果要用新版結果，主要要改的是：

1. `new_era/02_feature_engineering.py`
   - 把 `top_8_features` 改成新版排序前 8 名
   - 也就是你現在這份 importance 的前 8 個

2. `new_era/09_mm_backtest.py`
   - `top_8_features` 要同步成同一組
   - 不然訓練和回測用的特徵不一致

3. `new_era/09_mm_backtest_old_data.py`
   - 如果你要讓舊資料回測也對齊新版，就一樣改成同一組 top 8

4. `new_era/02_feature_engineering.py` 之後一定要重跑
   - 因為 `002_zscore_dataset.npz` 是用舊 top 8 做出來的
   - 改了 feature set 之後，資料集也要重建

最直接的做法是：先決定新版前 8，要不要真的換成它，然後我幫你把 `02/09/09_old` 三個地方一起同步。

# 5. 這是現在的結果，請幫我檢視為什麼反而虧錢了 並且現在幫我看這個模型與訓練方式在被動造勢的路上有沒有又用到重複的資料/時間沒對上/模型BUG等問題 並告訴我該如何修正目前的策略(主動攻擊不應該是最賺錢的嗎

反而虧錢的主因不是「TCN 完全沒用」，而是現在的 `TCN Alpha` 已經從以前偏樂觀的回測，變成比較真實的 taker 行為了。真實 taker 要付出 spread + taker fee，所以主動攻擊只有在訊號強度大到能覆蓋成本時才會賺。你這次的診斷結果顯示：訊號還沒有強到那個程度。

我剛剛又跑了一個 alpha trigger 診斷。重點數字是：

```text
test labels:
Down = 3,720
Stationary = 748,156
Up = 3,860
總樣本 = 755,736

Predicted Down precision = 2.24%
Predicted Up precision   = 1.69%

prob_up/down > 0.6:
active signals = 47,397
direction precision = 2.51%
stationary rate = 97.26%
avg signed future return = 0.169 ticks
median signed future return = 0 ticks
avg spread = 1.422 ticks
```

這裡已經很清楚了：模型「方向 recall」不低，但方向 precision 很低。它抓到一些真的 Up/Down，但同時對大量 stationary 樣本誤報方向。因為 test session 裡接近 99% 都是 stationary，所以整體 accuracy `81.36%` 其實主要來自猜對 stationary，不代表主動方向訊號很準。

**為什麼 Alpha 虧**
`TCN Alpha` 現在在 [tcn_pipeline_utils.py](/c:/Users/holbo/00VS_projjects/00_HFT_project/new_era/tcn_pipeline_utils.py:220) 的邏輯是：

- `prob_up > 0.6` 就用 best ask 主動買
- `prob_down > 0.6` 就用 best bid 主動賣
- 每次 taker 付 `50 TWD`
- 買在 ask / 賣在 bid，本身就先付掉 spread

以大台一點 `200 TWD` 算，單邊 taker fee 是 `0.25 tick`。如果平均 spread 約 `1.42 ticks`，你主動買在 ask 後，光從 mid-price 角度大概就先輸掉約 `0.71 tick + fee 0.25 tick`，還沒算出場成本。也就是說，訊號至少要提供接近 `1 tick` 以上的短期 edge 才有機會。

但現在 `prob > 0.6` 的平均 signed future return 只有 `0.169 ticks`。這個 edge 太小，所以 taker 越做越虧。這也解釋了為什麼：

```text
Baseline: +90,990
Gated:    +91,620
Alpha:    -1,873,575
```

不是主動攻擊理論上不該賺，而是**主動攻擊只有在訊號非常準、且期望報酬大於交易成本時才該賺**。以前 alpha 看起來最賺，是因為舊回測把 `my_bid = my_ask` 後還等下一筆報價確認成交，等於給了策略一個不真實的有利成交條件。現在修成當下 best ask/bid 成交後，成本就回來了。

**目前資料/時間/模型問題**
目前這版比之前乾淨很多，幾個大問題已經修掉：

- `09_mm_backtest.py` 現在只用 `20260127Night` 當 test session，沒有把同 session 混進 train。
- `02_feature_engineering.py` 現在逐 session 做 rolling z-score，不再跨日混算。
- `09_mm_backtest.py` 回測用 `sample_stride=1`，比訓練用 `stride=5` 更貼近逐 quote replay。
- Tensor 方向目前是對的：dataset 出 `(B,T,F)`，推論前 transpose 成 `(B,F,T)`，符合 `HLSFriendlyTCNClassifier`。

但還有幾個需要修的風險：

1. `09` 會用當前程式裡的 `TOP_8_FEATURES` 重建特徵；如果你訓練後又改了 feature order，模型會吃到錯順序的 feature。這是目前最大的「隱性模型 bug」風險。現在應該把 feature list / session split / window params 存進 npz 或 checkpoint metadata。

2. `05_train_tcn_pytorch.py` 用 class weight 強力放大 Up/Down，這會提高方向 recall，但也會製造大量 false positive。對 alpha taker 來說，false positive 非常貴，所以現在 loss function 和交易目標不一致。

3. label 是「未來 10 quote events 的 mid-price 是否超過 1 tick」，但 alpha taker 的真實 break-even 是「能不能覆蓋 spread + fee + 出場成本」。目前 label 門檻太低，和主動交易成本不匹配。

4. `gated` 有小幅改善，但只有 `+630 TWD`，代表模型目前比較適合當風控 filter，不適合直接拿來 aggressive trading。

**策略修正方向**
短期不要讓 `TCN Alpha` 直接 cross spread。現在最合理的路線是：

1. 保留 `baseline + gated`，先把 TCN 定位成「防禦型造市」：高機率下跌就撤 bid，高機率上漲就撤 ask。這條路和現在結果一致，至少沒有傷害 baseline。

2. 把 alpha 從 taker 改成 quote skew：例如 `prob_up` 高時不是直接買 ask，而是撤 ask、保留 bid，或只在 inventory 偏空時提高補回優先級。也就是讓模型調整掛單方向，不讓它主動付 spread。

3. 重新設計 alpha trigger：不要用 `prob_up > 0.6`，而是用 validation session 做 grid search，找 `prob_up/prob_down` 門檻是否真的能提升 PnL。從我剛剛的診斷看，就算 `threshold=0.9`，平均 signed future return 也只有 `0.386 ticks`，仍然不夠支付 taker 成本，所以單純提高門檻可能還是不夠。

4. 如果真的要做主動攻擊，label 要改成成本感知：例如 Up label 不是 `future_mid - current_mid > 1 tick`，而是 `future_mid - current_ask > taker_fee_ticks + exit_cost + margin`。Down 也同理用 `current_bid - future_mid`。這樣訓練目標才和 taker alpha 對齊。

5. 訓練上要降低 false positive：可以降低 class weight、改用 calibrated probability、或新增 expected return regression head。現在模型是「分類模型」，但交易需要的是「期望值模型」。

我會建議下一步先做兩個小修正，而不是直接重訓大模型：

1. 在 checkpoint / npz 存 `feature_cols`、`window_size`、`horizon`、`session_split`，並讓 `09` 檢查 feature order 是否與訓練一致。

2. 新增一個 `tcn_skew` 策略：只撤單與偏移掛單，不做 taker crossing，然後用 validation session 搜尋 gated/skew threshold，再固定門檻跑 test。