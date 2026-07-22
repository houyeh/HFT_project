# MODEL Review（教授用完整版）

此文件由 `meeting/w11/MODEL_filled_review.xlsx` 內容依 section 統整而成，並已把本次訓練/評估/回測輸出填入。

## 任務設定

### 專案目標
**本次答案**

事件級(逐筆)委託簿短期方向三分類；並以 HLS/FPGA 低延遲可部署為約束。

**為何重要**

定義研究的成功標準：不只看分類分數，也要能對應交易/部署需求。

**老師可能會問**

你是在做預測、做交易，還是做硬體？

**建議下一步**

報告時分清階段：目前重點是資料/模型管線驗證 + 訊號診斷 + 回測雛形；交易端仍在迭代。

**證據 / 來源**

new_era/01_feature_importance.py, 02_feature_engineering.py, 05_train_tcn_pytorch.py, 06_evaluate_tcn.py, 09_mm_backtest_fixed.py

### 任務型態
**本次答案**

三分類：Down(0) / Stationary(1) / Up(2)。

**為何重要**

決定 label、loss 與混淆矩陣解讀；在極度不平衡時 accuracy 會失真。

**老師可能會問**

為何不用回歸預測報酬？

**建議下一步**

三分類能直接對應交易決策(防禦撤單/偏置掛單/是否出手)，也較利於 FPGA 上的簡化決策。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py, new_era/06_evaluate_tcn.py

### 預測目標(label)
**本次答案**

以 WindowedLOBSequenceDataset 產生：window=100、horizon=10、threshold_ticks=1.0(以 mid-price tick 變動判斷)。

**為何重要**

監督學習的正確答案定義；必須可解釋且無洩漏。

**老師可能會問**

label 用到未來是否洩漏？

**建議下一步**

label 本來就需要未來(監督學習)；關鍵是 feature 不能包含任何由未來推導的欄位。建議做 label-shuffle sanity check。

**證據 / 來源**

src/training/datasets.py, src/features/sliding_window.py, new_era/tcn_pipeline_utils.py

### 輸入視窗(window)
**本次答案**

每筆樣本取過去 100 個事件/ tick 的 8 維特徵序列。

**為何重要**

控制模型可用的歷史長度，也影響延遲與資料量。

**老師可能會問**

為何是 100？做過比較嗎？

**建議下一步**

目前是 baseline；後續做 window_size=50/100/200 ablation 並以回測 PnL 驗證。

**證據 / 來源**

new_era/tcn_pipeline_utils.py (WINDOW_SIZE=100)

### 預測視野(horizon)
**本次答案**

horizon=10 (10 個事件後)。

**為何重要**

視野越長越難、越短越雜訊；需與策略可執行性對齊。

**老師可能會問**

10 事件是否太短/太吵？

**建議下一步**

比較 horizon=5/10/20，並用含成本回測檢驗可交易性而非只看 accuracy。

**證據 / 來源**

new_era/tcn_pipeline_utils.py (HORIZON=10)

### 門檻(threshold)
**本次答案**

threshold_ticks=1.0 (±1 tick)。

**為何重要**

定義 Up/Down 與盤整邊界；也決定 class imbalance 強度。

**老師可能會問**

±1 tick 是否小於 spread/手續費？

**建議下一步**

建議把 spread/費用納入：調整 threshold 或改用 dynamic threshold；並用 PR-AUC/回測 PnL 驗證。

**證據 / 來源**

new_era/tcn_pipeline_utils.py (THRESHOLD_TICKS=1.0)

## 資料集

### 原始資料來源
**本次答案**

`data/NEW` 內台指期逐筆委託簿 CSV；本次使用 TXFB6 與 MXFB6。

**為何重要**

老師會問標的/日期/盤別覆蓋度與資料品質。

**老師可能會問**

用哪些商品與哪些日期？

**建議下一步**

建議在報告中列出 train/val/test session 名單與對應 samples(本表已填)。

**證據 / 來源**

new_era/01_feature_importance.py, new_era/02_feature_engineering.py

### 資料型態 / 頻率
**本次答案**

事件級(非時間 bar) 委託簿快照 + 成交欄位。

**為何重要**

事件級資料對 leakage 與延遲要求更敏感；切割需嚴格按時間/Session。

**老師可能會問**

是時間 bar 嗎？

**建議下一步**

明確聲明是 event-level sliding window。

**證據 / 來源**

src/training/datasets.py, new_era/02_feature_engineering.py

### 委託簿深度
**本次答案**

5 檔 bid/ask 價量。

**為何重要**

深度決定可用的微結構訊息量。

**老師可能會問**

為何 5 檔？

**建議下一步**

後續可做 top-1/top-3/top-5 ablation。

**證據 / 來源**

new_era/01_feature_importance.py

### 成交資訊
**本次答案**

使用 last_trade_price/qty (qty>0) 並在 feature builder 中整合進特徵。

**為何重要**

成交流能補足單靠報價看不到的 aggressiveness / toxic flow。

**老師可能會問**

成交與委託簿如何對齊？

**建議下一步**

建議補對齊方式與統計(例如成交落在 LOB 時間戳的延遲分佈)。

**證據 / 來源**

src/utils/tw_train_helpers.py (build_tw_feature_frame)

### Session 處理
**本次答案**

按 session_id 分組做特徵與正規化；再做 session-level train/val/test split。

**為何重要**

避免 sample-level shuffle 造成相鄰視窗洩漏。

**老師可能會問**

視窗會跨日/跨盤嗎？

**建議下一步**

目前 per-session 建立 dataset；建議在投影片強調是 anti-leakage 設計。

**證據 / 來源**

new_era/tcn_pipeline_utils.py (split_records_by_session), new_era/02_feature_engineering.py

### 樣本數 / Split (本次)
**本次答案**

Train Data: X shape (886133, 8, 100), y shape (886133,)
Validation Data: X shape (213204, 8, 100), y shape (213204,)
Test Data: X shape (389652, 8, 100), y shape (389652,)
Session split: train=8, val=2, test=2

**為何重要**

樣本量與 split 決定可信度，也影響過擬合判斷。

**老師可能會問**

每個 split 有多少筆？

**建議下一步**

雖然 N 很大，但有效方向樣本比例極低；後續要改指標與策略用法。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py, new_era/06_evaluate_tcn.py

### Train/Val/Test 切法
**本次答案**

使用 split_records_by_session：每個 symbol 依 session_id 排序後切 train(70%)/val(15%)/test(剩餘)。

**為何重要**

時間序列必須用『往後留』方式做 out-of-sample。

**老師可能會問**

test 是否真的在時間上更後面？

**建議下一步**

本次 test session 為 20260127Night(較後)，符合 out-of-sample 方向；仍建議做 walk-forward。

**證據 / 來源**

new_era/tcn_pipeline_utils.py (split_records_by_session)

### Class 分佈 (本次)
**本次答案**

極端不平衡：Stationary 約 99%。
(可直接見 06_evaluate 的 support 與 09_mm_backtest_fixed 的診斷輸出)

**為何重要**

在此情境 accuracy 幾乎沒有決策價值；需看 Up/Down precision/PR-AUC。

**老師可能會問**

這樣的 accuracy 有意義嗎？

**建議下一步**

改用 macro-F1、balanced accuracy、PR-AUC；策略端只在高置信度時採用訊號，或先用於 maker skew/防禦撤單。

**證據 / 來源**

new_era/06_evaluate_tcn.py (classification_report), new_era/09_mm_backtest_fixed.py

### 缺值處理
**本次答案**

特徵表 per-session 先 dropna()；rolling z-score 的 NaN 以 0 補齊並 clip [-10,10]。

**為何重要**

避免 NaN 進模型，但會丟掉每個 session 開頭一段資料。

**老師可能會問**

每個 session 丟掉多少列？

**建議下一步**

建議在 02 輸出 dropna 前後筆數與比例，避免不小心偏移 label 分佈。

**證據 / 來源**

new_era/02_feature_engineering.py, new_era/tcn_pipeline_utils.py

### Outlier handling
**本次答案**

No explicit outlier handling found beyond filtering best bid/ask > 0 and ValueError skip.

**為何重要**

Outliers can dominate z-score/statistics or represent important market events.

**老師可能會問**

Did you remove bad ticks?

**建議下一步**

Add sanity checks: negative/zero prices, crossed book, abnormal spread, abnormal volume.

**證據 / 來源**

01 checks row length and bid/ask > 0

### Timestamp parsing
**本次答案**

parse_new_tw_timestamp converts time fields to nanoseconds; session offset added.

**為何重要**

Timing correctness is critical in HFT data.

**老師可能會問**

Are timestamp units correct?

**建議下一步**

Verify source timestamp format, especially if raw format is h:m:s:ms:subfield; current parser uses first 4 parts only.

**證據 / 來源**

01 parse_new_tw_timestamp

## 特徵工程

### 特徵產生
**本次答案**

使用 build_tw_feature_frame 從 LOB + Trades 產生微結構特徵，再挑 Top 8 做模型輸入。

**為何重要**

把原始 LOB 轉成可學習的經濟含義特徵；老師通常會追問每個特徵的直覺。

**老師可能會問**

輸入特徵到底是什麼？經濟意義是什麼？

**建議下一步**

建議報告時用一句話解釋每個 Top8：OBI/queue imbalance=買賣壓力；spread=摩擦；trade_intensity/cancel_pressure=毒性/流動性變化。

**證據 / 來源**

src/utils/tw_train_helpers.py (build_tw_feature_frame), new_era/tcn_pipeline_utils.py (TOP_8_FEATURES)

### 特徵選擇
**本次答案**

先用 RandomForest feature_importances_ 做粗篩，再固定使用 TOP_8_FEATURES=obi, queue_imbalance, weighted_mid_deviation, trade_intensity_short, deep_obi, queue_imbalance_change, normalized_spread, cancel_pressure_short。

**為何重要**

降低維度與硬體成本；但 RF 的重要度不等價於 TCN 的重要度，需要再驗證。

**老師可能會問**

為何是 Top 8？RF importance 可靠嗎？

**建議下一步**

把 Top4/8/12 做 ablation，並補 permutation importance / feature ablation（以 PR-AUC 與回測 PnL 為準）。

**證據 / 來源**

new_era/01_feature_importance.py, new_era/001_feature_importance.csv, new_era/tcn_pipeline_utils.py

### 特徵維度
**本次答案**

TCN 輸入維度=8（obi, queue_imbalance, weighted_mid_deviation, trade_intensity_short, deep_obi, queue_imbalance_change, normalized_spread, cancel_pressure_short），張量形狀為 N×8×100（Conv1d 的 channel-first）。

**為何重要**

維度越大越耗硬體/越容易過擬合；維度太小可能漏訊號。

**老師可能會問**

為什麼只用 8 維？

**建議下一步**

這是 HLS-friendly baseline；後續比較 Top8 vs Top12 的 PR-AUC / 回測改善幅度。

**證據 / 來源**

new_era/tcn_pipeline_utils.py, new_era/05_train_tcn_pytorch.py

### 正規化(rolling z-score)
**本次答案**

每個 session 內對 Top 8 做 rolling z-score：window=2000、min_periods=100；std=0 以 1e-5 避免除零；NaN→0；clip [-10,10]。

**為何重要**

穩定尺度、降低 regime shift 影響；rolling 計算本身要保持因果性。

**老師可能會問**

會不會用到未來資料造成洩漏？

**建議下一步**

pandas rolling 是因果(只用過去)；建議再用 label-shuffle sanity check 驗證沒有洩漏。

**證據 / 來源**

new_era/tcn_pipeline_utils.py (apply_rolling_zscore)

### 資料洩漏檢查(leakage)
**本次答案**

已做 session-level split 並在每個 session 內產生特徵；但尚未用 label-shuffle / forbidden-feature 清單做系統性驗證。

**為何重要**

教授通常會強力追問：是否有任何 feature 直接/間接用到未來 mid-price 或 label？

**老師可能會問**

你怎麼證明沒有洩漏？

**建議下一步**

新增：1) label shuffle 實驗 2) 列出禁止欄位清單(任何 future-derived 欄位) 3) 檢查 rolling 統計是否只用過去。

**證據 / 來源**

src/utils/tw_train_helpers.py, src/training/datasets.py, meeting/w11/MODEL_review_full.md

## Label 設計

### Label 規則(精確)
**本次答案**

以 decision time t 的 mid-price 與 t+horizon 比較：
(mid[t+h]-mid[t]) > +threshold => Up(2)
(mid[t+h]-mid[t]) < -threshold => Down(0)
其餘 => Stationary(1)
本次 threshold=1 tick, horizon=10。

**為何重要**

這是監督學習的正確答案，必須講清楚。

**老師可能會問**

端點(end-point) 還是區間 max/min？

**建議下一步**

目前是端點；若交易更重視『期間內觸發』可做 label variant 比較，並用回測驗證。

**證據 / 來源**

src/features/sliding_window.py (label_direction), src/training/datasets.py

### Label 類別定義
**本次答案**

0=Down, 1=Stationary, 2=Up。

**為何重要**

避免訓練/評估/回測的 class id 對不上。

**老師可能會問**

訓練與評估是否一致？

**建議下一步**

已一致；建議所有圖表/CSV 都固定同一順序 [Down,Stat,Up]。

**證據 / 來源**

new_era/06_evaluate_tcn.py

### Trading relevance of labels
**本次答案**

Current label detects ±1 tick within next 10 events, not directly profit after costs.

**為何重要**

A predictive label may still not be tradable.

**老師可能會問**

Does label map to actual order action?

**建議下一步**

Add backtest/signal evaluation after classification metrics.

**證據 / 來源**

Planned trading deployment

## 模型架構

### 模型類型
**本次答案**

HLSFriendlyTCNClassifier (TCN + global average pooling)。

**為何重要**

TCN 相對 RNN/Transformer 更平行、算子更簡單，更利於 HLS/FPGA。

**老師可能會問**

為何不用 LSTM/Transformer？

**建議下一步**

TCN 具備低延遲與可硬體化優勢，且參數量可極小化。

**證據 / 來源**

src/models/tcn.py, new_era/05_train_tcn_pytorch.py

### 輸入張量形狀
**本次答案**

X shape = (N, 8, 100)；其中 8=features channels，100=時間長度(window)。

**為何重要**

Conv1d 需要 channel-first；維度錯會導致模型學錯或 shape error。

**老師可能會問**

你確定沒有把 time/feature 軸搞反？

**建議下一步**

05 透過 concatenate_records(channel_first=True) 明確轉置到 (N, 8, 100)。

**證據 / 來源**

new_era/tcn_pipeline_utils.py (concatenate_records), new_era/05_train_tcn_pytorch.py

### 通道數/層數
**本次答案**

channel_list=[16,16,16]，共 3 個 TemporalBlock。

**為何重要**

控制容量與硬體資源。

**老師可能會問**

是否太小？

**建議下一步**

先做 baseline；後續用 ablation (8/16/32 channels) 觀察 PR-AUC 與回測改善幅度。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py, src/models/tcn.py

### Kernel 大小
**本次答案**

kernel_size=3。

**為何重要**

影響局部時序模式抽取與感受野。

**老師可能會問**

感受野夠嗎？

**建議下一步**

目前 RF=29 ticks；透過 global average pooling 彙總整段 100 ticks 的局部模式。若要更長依賴可加層/加 dilation。

**證據 / 來源**

src/models/tcn.py

### Dilation / 感受野(Receptive Field)
**本次答案**

dilation schedule=1,2,4 (3 層)。依程式公式：RF = 1 + 2*(k-1)*sum(dilation)=1+2*(3-1)*(1+2+4)=29 ticks。

**為何重要**

決定單一位置能看到多長的歷史。

**老師可能會問**

能不能看滿 100 ticks？

**建議下一步**

單一位置 RF=29；但模型最後做 global average pooling，等價於彙總整段 100 ticks 內的局部模式。若要捕捉更長期依賴可增加層數/ dilation。

**證據 / 來源**

src/models/tcn.py (receptive_field())

### Activation / 正規化
**本次答案**

Activation=ReLU；Normalization=無(BN/LN 未使用)；Dropout=0.1。

**為何重要**

影響訓練穩定與 HLS 算子複雜度。

**老師可能會問**

為何不加 BatchNorm？

**建議下一步**

優先維持 HLS-friendly、算子簡單；若需要再比較 BN/LN 的效益與硬體代價。

**證據 / 來源**

src/models/tcn.py

### 參數量
**本次答案**

4,515 個參數。

**為何重要**

容量/可部署性指標。

**老師可能會問**

能塞進 FPGA 嗎？

**建議下一步**

參數量極小；後續仍要補上 HLS 合成資源(LUT/DSP/BRAM)與 latency 數字。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py (Model Parameters)

## Model 架構

### Dropout
**本次答案**

dropout = 0.1.

**為何重要**

Regularizes model during training.

**老師可能會問**

Is dropout enough?

**建議下一步**

Reasonable low dropout for small model. Monitor train/val gap.

**證據 / 來源**

05 dropout=0.1

## 訓練設定

### Epoch 數
**本次答案**

epochs=20；Best Val Acc=0.8121。

**為何重要**

影響是否收斂；需要曲線證據。

**老師可能會問**

20 epochs 夠嗎？

**建議下一步**

目前未落盤曲線；建議輸出 training_history.csv 與 checkpoint/resume，再決定是否加 epochs / scheduler。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py

### Batch size
**本次答案**

train/val batch_size=1024。

**為何重要**

影響收斂、噪聲與吞吐量。

**老師可能會問**

為何 1024？

**建議下一步**

作為大資料的 baseline；後續比較 256/512/1024/2048 看 PR-AUC 與回測。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py

### Learning rate
**本次答案**

Adam lr=1e-3。

**為何重要**

影響收斂速度與穩定性。

**老師可能會問**

lr 是否太大/太小？

**建議下一步**

後續可加 scheduler 或測 3e-4；以 PR-AUC/回測 PnL 作比較基準。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py

### Optimizer
**本次答案**

Adam。

**為何重要**

基準優化器；但不含 weight decay。

**老師可能會問**

有沒有試 AdamW？

**建議下一步**

後續可測 AdamW(weight_decay=1e-4) 與 focal loss 的組合，看 minority precision 是否提升。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py

### Loss / 類別權重
**本次答案**

加權 CrossEntropyLoss；class weights=[67.37628936767578, 0.3366035223007202, 69.91187286376953]

**為何重要**

處理 class imbalance，使 Down/Up 對 loss 有足夠影響力。

**老師可能會問**

權重會不會讓訓練不穩？

**建議下一步**

需同時看 per-class precision/recall；後續可比較 focal loss、或改策略只用高置信度訊號。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py (Class Weights applied)

### 類別權重計算
**本次答案**

weights = total / (num_classes * count_c)（由 y_train 計算）。

**為何重要**

讓少數類別在 loss 的影響力不被淹沒。

**老師可能會問**

權重怎麼算的？

**建議下一步**

回報 y_train 的 class counts；並搭配 PR-AUC 觀察是否真的改善方向訊號。

**證據 / 來源**

new_era/tcn_pipeline_utils.py (compute_class_weight_tensor)

### LR scheduler
**本次答案**

目前未使用。

**為何重要**

可改善 plateau 時的收斂。

**老師可能會問**

為何沒有 scheduler？

**建議下一步**

後續可加 ReduceLROnPlateau 或 cosine，並以 PR-AUC/回測 PnL 作為選模指標。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py

### Early stopping
**本次答案**

未使用（僅保存 best val_acc 權重，不會提前停止）。

**為何重要**

避免過擬合與浪費訓練時間。

**老師可能會問**

怎麼避免 overfit？

**建議下一步**

加入 patience-based early stopping（建議監控 macro-F1 或 PR-AUC，而不是 val_acc）。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py

### Checkpoint / Resume
**本次答案**

目前只保存 best model state_dict 到 new_era/005_tcn_weights.pth；未保存 optimizer state，因此無法 resume。

**為何重要**

訓練更久/做更多實驗時必須可續訓且可重現。

**老師可能會問**

中斷後能續訓嗎？

**建議下一步**

改存 checkpoint(dict: model/optimizer/epoch/best/history)，並支援 resume。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py

### 隨機種子 / 重現性
**本次答案**

RF 有 random_state=42；PyTorch 訓練目前未固定 numpy/torch/cuda seeds。

**為何重要**

可重現性是研究可信度基本要求。

**老師可能會問**

你能 100% 重現嗎？

**建議下一步**

建議加入 seed 固定與 deterministic 設定，並把 seed 記錄到 experiment log。

**證據 / 來源**

new_era/01_feature_importance.py, new_era/05_train_tcn_pytorch.py

## 正則化

### Weight decay
**本次答案**

未使用（Adam 無 weight_decay/AdamW）。

**為何重要**

可抑制過擬合、改善泛化。

**老師可能會問**

有試過 AdamW 嗎？

**建議下一步**

後續測 AdamW + 合理 weight_decay；並用 PR-AUC/回測改善做比較。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py

### Gradient clipping
**本次答案**

未使用。

**為何重要**

可降低梯度爆炸、提高訓練穩定性。

**老師可能會問**

訓練會不會不穩？

**建議下一步**

若 loss 波動/爆炸，加入 clip_grad_norm_，並配合 scheduler。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py

## 評估

### 訓練/驗證曲線
**本次答案**

本次有印出每 epoch 的 train_loss/train_acc/val_loss/val_acc，但尚未落盤成 CSV/圖表。

**為何重要**

曲線能證明是否收斂，以及是否欠擬合/過擬合。

**老師可能會問**

能不能給我 loss curve？

**建議下一步**

在 05 增加 training_history.csv，並輸出 plot；同時記錄 best checkpoint 對應 epoch。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py

### 驗證集表現(Val)
**本次答案**

Loss=0.701812；Accuracy=81.21%

**為何重要**

用於選模；但在不平衡資料下要同時看 per-class 指標。

**老師可能會問**

accuracy 可信嗎？

**建議下一步**

同時報告 confusion matrix、macro avg、以及 Up/Down precision；策略端避免把低 precision 當成 taker 訊號。

**證據 / 來源**

new_era/06_evaluate_tcn.py；輸出檔：C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\eval_outputs

### 測試集表現(Test)
**本次答案**

Loss=0.713121；Accuracy=79.86%

**為何重要**

out-of-sample performance；只能用一次做最終報告。

**老師可能會問**

是不是調參調到 test？

**建議下一步**

固定 test 不回頭調參；後續採 walk-forward 評估更貼近部署。

**證據 / 來源**

new_era/06_evaluate_tcn.py；輸出檔：C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\eval_outputs

### 混淆矩陣(Val/Test)
**本次答案**

Val CM (Counts, [Down,Stat,Up]):
[[306, 886, 86], [12273, 172368, 25873], [39, 896, 477]]

Test CM (Counts):
[[326, 783, 63], [23724, 310235, 53253], [19, 646, 603]]

**為何重要**

看模型是否只會預測 Stationary，以及 Up/Down 錯誤型態。

**老師可能會問**

模型是不是只會預測盤整？

**建議下一步**

應以 normalized confusion matrix 觀察 class-wise recall；目前 Up/Down precision 很低，較適合用於 maker skew/防禦。

**證據 / 來源**

C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\eval_outputs (confusion_matrix_count/normalized.png)

### Precision/Recall/F1 (Val/Test)
**本次答案**

Val 摘要：Down precision=0.02 recall=0.24; Up precision=0.02 recall=0.34; Stationary precision=0.99 recall=0.82; macro avg f1=0.32
Test 摘要：Down precision=0.01 recall=0.28; Up precision=0.01 recall=0.48; Stationary precision=1.00 recall=0.80; macro avg f1=0.31

**為何重要**

在不平衡資料下，比 accuracy 更重要。

**老師可能會問**

Up/Down 的 precision 有多少？

**建議下一步**

目前 Up/Down precision 約 1~2%；需改用 PR-AUC、校準與更嚴格門檻，並讓策略端先採用 maker-only 的用法。

**證據 / 來源**

C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\eval_outputs (classification_report.csv)

### ROC-AUC / PR-AUC
**本次答案**

目前尚未輸出/保存 softmax 機率，因此尚未計算 ROC-AUC/PR-AUC（尤其 Up/Down 的 PR-AUC）。

**為何重要**

在極端不平衡下 PR-AUC 比 accuracy 更能反映方向訊號品質。

**老師可能會問**

你能量化『方向訊號』到底有多可靠嗎？

**建議下一步**

修改 06：輸出 softmax 機率（prob_down/prob_stat/prob_up），並計算 one-vs-rest PR-AUC；建議同時做 calibration。

**證據 / 來源**

new_era/06_evaluate_tcn.py（需擴充：輸出機率與 PR-AUC）

### Loss/曲線視覺化
**本次答案**

目前沒有把 train/val loss/acc 曲線輸出成圖或 CSV（只有 console print）。

**為何重要**

曲線能證明是否收斂，以及最佳 checkpoint 是否合理。

**老師可能會問**

你怎麼知道模型有收斂？

**建議下一步**

在 05 加入 training_history.csv，並輸出 loss/acc 曲線圖；同時記錄 best checkpoint 對應 epoch。

**證據 / 來源**

new_era/05_train_tcn_pytorch.py（需擴充：寫出 history）

## 穩健性

### Out-of-sample 測試
**本次答案**

已有：session-level test split（test sessions=20260127Night）。

**為何重要**

檢查是否能泛化到未見過的 session。

**老師可能會問**

test 是否真的是『未來』？

**建議下一步**

目前按 session_id 排序切分；後續用 walk-forward 更嚴格。

**證據 / 來源**

new_era/tcn_pipeline_utils.py, new_era/06_evaluate_tcn.py

### Walk-forward 測試
**本次答案**

尚未實作。

**為何重要**

比單次 split 更貼近真實部署（rolling）。

**老師可能會問**

不同市場狀態還有效嗎？

**建議下一步**

按 session 日期 rolling train→val→test，並在每窗做含成本造市回測。

**證據 / 來源**

（待新增腳本）

### 噪聲 / 擾動測試
**本次答案**

尚未實作。

**為何重要**

檢查模型是否脆弱依賴少數特徵或少數 regime。

**老師可能會問**

加噪後表現如何？

**建議下一步**

在 val/test 對特徵做 mask/noise，觀察 PR-AUC 與回測 PnL 退化。

**證據 / 來源**

（可擴充 new_era/06_evaluate_tcn.py）

## 合理性檢查

### Label shuffle 測試
**本次答案**

尚未實作。

**為何重要**

偵測 leakage：若 label 打亂仍高分，代表可能有洩漏。

**老師可能會問**

如何證明模型沒偷看未來？

**建議下一步**

新增 label-shuffle 訓練/評估並保存結果（應接近隨機）。

**證據 / 來源**

（待新增腳本）

### 特徵重要度(可解釋性)
**本次答案**

已產出 RF 的 feature importance（001_feature_importance.csv）。TCN 尚未做 permutation importance。

**為何重要**

教授會希望看到：模型用的訊號是否合理、是否可解釋。

**老師可能會問**

TCN 真的是靠這些特徵在預測嗎？

**建議下一步**

補 permutation importance / feature ablation（對 PR-AUC 與回測 PnL 的影響）。

**證據 / 來源**

new_era/001_feature_importance.csv, new_era/01_feature_importance.py

## 部署/交易

### 推論延遲(Inference latency)
**本次答案**

尚未量測 PyTorch batch=1 latency；亦尚未把此權重對應到 HLS 合成報告做 latency 對照。

**為何重要**

HFT/FPGA 的價值主張核心：低延遲。

**老師可能會問**

到底多快？

**建議下一步**

新增 benchmark：PyTorch CPU batch=1/批次吞吐；並以同權重做 hls4ml synth 報告（latency/resource）。

**證據 / 來源**

new_era/07_hls4ml_tcn_conversion.py, new_era/007_hls4ml_tcn_prj

### 模型大小 / 參數
**本次答案**

參數量 4,515；權重檔 new_era/005_tcn_weights.pth 約 25 KB。

**為何重要**

決定能否塞進 FPGA/BRAM 以及可達到的時脈。

**老師可能會問**

能塞進 PYNQ-Z2 嗎？

**建議下一步**

補上 HLS 資源(LUT/DSP/BRAM)與時脈/latency 以形成完整部署證據鏈。

**證據 / 來源**

new_era/005_tcn_weights.pth, new_era/05_train_tcn_pytorch.py

### 吞吐量(Throughput)
**本次答案**

尚未量測。

**為何重要**

需要能跟上 market data feed 的事件速度。

**老師可能會問**

是否能即時處理？

**建議下一步**

補 PyTorch/HLS samples/sec；並估算在真實行情下的峰值負載。

**證據 / 來源**

（待新增 benchmark）

### 交易成本 / 回測(含成本)
**本次答案**

已建立 session-aware 的 TCN 造市回測(固定版 maker-only 策略)。
Test session: TXFB6 20260127Night
Samples: 755,736
Accuracy: 0.8136
Confusion Matrix [Down,Stat,Up]: [[927, 2563, 230], [40425, 612276, 95455], [67, 2146, 1647]]
True dist: [3720, 748156, 3860]
Pred dist: [41419, 616985, 97332]
Spread: avg 2.144 ticks | median 2.000 ticks

Signal diagnostics:
prob>=0.55: active=67929, direction_precision=0.0237, stationary_rate=0.9742
prob>=0.60: active=47397, direction_precision=0.0251, stationary_rate=0.9726
prob>=0.70: active=19938, direction_precision=0.0293, stationary_rate=0.9685
prob>=0.80: active=5167, direction_precision=0.0379, stationary_rate=0.9599
prob>=0.90: active=360, direction_precision=0.0778, stationary_rate=0.9222

Strategy results:
Baseline passive maker: PnL=97,800 TWD, trades=454 (maker=454), ending_inv=-2, max_abs_inv=3
Best sweep (top): tcn_skew prob_threshold=0.70 edge_threshold=0.15, PnL=145,600 TWD (+47,800), trades=521, ending_inv=-1, max_abs_inv=3

**為何重要**

分類分數不等於可交易；必須在含成本的回測裡驗證訊號是否能轉成 PnL。

**老師可能會問**

扣掉成本還能賺嗎？該怎麼用這個訊號？

**建議下一步**

目前觀察：Up/Down precision 低，較適合用於 maker skew/防禦撤單，而非頻繁 taker 出手。下一步做 walk-forward 回測檢查穩健性。

**證據 / 來源**

new_era/09_mm_backtest_fixed.py, new_era/tcn_pipeline_utils.py (maker/taker fee + point value)

## 改善建議

### Main current weakness
**本次答案**

Evaluation/experiment tracking is incomplete: no saved training history, no complete test report table, no convergence evidence, no trading backtest.

**為何重要**

Identifies where teacher can advise.

**老師可能會問**

What is the biggest risk now?

**建議下一步**

Add experiment log, curves, normalized confusion matrix, class distribution, and ablations.

**證據 / 來源**

Current pipeline

### Next experiment
**本次答案**

Extend epochs with resume/checkpoint; save training_history; evaluate macro-F1/confusion matrix; run window/horizon/feature-count ablation.

**為何重要**

Creates systematic improvement path.

**老師可能會問**

What will you try next?

**建議下一步**

Prioritize: 1) longer training + curves, 2) class distribution, 3) normalized confusion matrix, 4) backtest.

**證據 / 來源**

Recommended

## 缺漏 / 待補強項目（從 Missing Info sheet 彙整）

- **High**：Up/Down 訊號品質指標（PR-AUC / calibration）：尚未完成
  - 重要性：在 Stationary 佔比 ~99% 時，比 accuracy 更能反映『可交易訊號』品質。
  - 補齊方式：修改 06：輸出 softmax 機率；計算 one-vs-rest PR-AUC；做 reliability diagram / ECE。
  - 記錄位置：Model Review: 評估區 + 附圖
- **High**：訓練曲線落盤（training_history.csv）：尚未完成
  - 重要性：需要向老師證明是否收斂/是否欠擬合或過擬合。
  - 補齊方式：修改 05：每 epoch 寫入 CSV；並把 best checkpoint 對應的 epoch 記錄下來。
  - 記錄位置：Model Review: 訓練設定/評估區
- **High**：Walk-forward（rolling）回測：尚未完成
  - 重要性：單次 split 不足以證明穩健性；walk-forward 更接近真實部署。
  - 補齊方式：以 session 日期 rolling train→val→test；每窗跑含成本 maker-only 回測。
  - 記錄位置：Model Review: 穩健性/部署交易區
- **Medium**：PyTorch / HLS 延遲與吞吐量 benchmark：尚未量測
  - 重要性：HFT/FPGA 的主張需要數字：batch=1 latency、samples/sec、HLS 資源與時脈。
  - 補齊方式：新增 benchmark 腳本；並把同一組權重對應到 hls4ml 專案的 synth 報告。
  - 記錄位置：Model Review: 部署/交易區
- **Medium**：隨機種子固定（torch/numpy/cuda）：尚未完成
  - 重要性：研究可重現性的基本要求。
  - 補齊方式：在 05/06 加入 seed 設定與 deterministic 配置；將 seed 寫入 log。
  - 記錄位置：Model Review: 訓練設定區
- **Low**：Label shuffle sanity check：尚未完成
  - 重要性：快速檢查是否存在 leakage：打亂 label 應使模型表現接近隨機。
  - 補齊方式：新增一個 label-shuffle 訓練/評估腳本並保存結果。
  - 記錄位置：Model Review: 合理性檢查區
