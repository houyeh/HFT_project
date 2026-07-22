from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import openpyxl


@dataclass(frozen=True)
class RowUpdate:
    row: int
    A: str | None = None
    B: str | None = None
    C: str | None = None
    D: str | None = None
    E: str | None = None
    F: str | None = None
    G: str | None = None


def _set_row(ws, upd: RowUpdate) -> None:
    if upd.A is not None:
        ws[f"A{upd.row}"].value = upd.A
    if upd.B is not None:
        ws[f"B{upd.row}"].value = upd.B
    if upd.C is not None:
        ws[f"C{upd.row}"].value = upd.C
    if upd.D is not None:
        ws[f"D{upd.row}"].value = upd.D
    if upd.E is not None:
        ws[f"E{upd.row}"].value = upd.E
    if upd.F is not None:
        ws[f"F{upd.row}"].value = upd.F
    if upd.G is not None:
        ws[f"G{upd.row}"].value = upd.G


def fill_workbook(
    *,
    in_path: Path,
    out_path: Path,
    # Metrics/output captured from the user's run logs (copied verbatim so the spreadsheet is reproducible).
    train_x_shape: str,
    val_x_shape: str,
    test_x_shape: str,
    class_weights: str,
    model_params: int,
    best_val_acc: float,
    val_loss: float,
    val_acc: float,
    test_loss: float,
    test_acc: float,
    val_cm: str,
    test_cm: str,
    val_report_summary: str,
    test_report_summary: str,
    eval_outputs_dir: str,
    # Backtest summary (from 09_mm_backtest_fixed.py run output)
    mm_session: str,
    mm_samples: int,
    mm_accuracy: float,
    mm_cm: str,
    mm_true_dist: str,
    mm_pred_dist: str,
    mm_spread: str,
    mm_threshold_diag: str,
    mm_strategy_results: str,
) -> None:
    wb = openpyxl.load_workbook(in_path)

    ws = wb["Model Review"]

    # Header
    _set_row(
        ws,
        RowUpdate(
            1,
            A="區塊",
            B="項目",
            C="依程式 / 模型與本次輸出填寫",
            D="定義 / 為何重要",
            E="老師可能會問",
            F="評估 / 建議下一步",
            G="證據 / 來源",
        ),
    )

    updates: list[RowUpdate] = []

    # 任務設定
    updates += [
        RowUpdate(
            2,
            A="任務設定",
            B="專案目標",
            C="事件級(逐筆)委託簿短期方向三分類；並以 HLS/FPGA 低延遲可部署為約束。",
            D="定義研究的成功標準：不只看分類分數，也要能對應交易/部署需求。",
            E="你是在做預測、做交易，還是做硬體？",
            F="報告時分清階段：目前重點是資料/模型管線驗證 + 訊號診斷 + 回測雛形；交易端仍在迭代。",
            G="new_era/01_feature_importance.py, 02_feature_engineering.py, 05_train_tcn_pytorch.py, 06_evaluate_tcn.py, 09_mm_backtest_fixed.py",
        ),
        RowUpdate(
            3,
            A="任務設定",
            B="任務型態",
            C="三分類：Down(0) / Stationary(1) / Up(2)。",
            D="決定 label、loss 與混淆矩陣解讀；在極度不平衡時 accuracy 會失真。",
            E="為何不用回歸預測報酬？",
            F="三分類能直接對應交易決策(防禦撤單/偏置掛單/是否出手)，也較利於 FPGA 上的簡化決策。",
            G="new_era/05_train_tcn_pytorch.py, new_era/06_evaluate_tcn.py",
        ),
        RowUpdate(
            4,
            A="任務設定",
            B="預測目標(label)",
            C="以 WindowedLOBSequenceDataset 產生：window=100、horizon=10、threshold_ticks=1.0(以 mid-price tick 變動判斷)。",
            D="監督學習的正確答案定義；必須可解釋且無洩漏。",
            E="label 用到未來是否洩漏？",
            F="label 本來就需要未來(監督學習)；關鍵是 feature 不能包含任何由未來推導的欄位。建議做 label-shuffle sanity check。",
            G="src/training/datasets.py, src/features/sliding_window.py, new_era/tcn_pipeline_utils.py",
        ),
        RowUpdate(
            5,
            A="任務設定",
            B="輸入視窗(window)",
            C="每筆樣本取過去 100 個事件/ tick 的 8 維特徵序列。",
            D="控制模型可用的歷史長度，也影響延遲與資料量。",
            E="為何是 100？做過比較嗎？",
            F="目前是 baseline；後續做 window_size=50/100/200 ablation 並以回測 PnL 驗證。",
            G="new_era/tcn_pipeline_utils.py (WINDOW_SIZE=100)",
        ),
        RowUpdate(
            6,
            A="任務設定",
            B="預測視野(horizon)",
            C="horizon=10 (10 個事件後)。",
            D="視野越長越難、越短越雜訊；需與策略可執行性對齊。",
            E="10 事件是否太短/太吵？",
            F="比較 horizon=5/10/20，並用含成本回測檢驗可交易性而非只看 accuracy。",
            G="new_era/tcn_pipeline_utils.py (HORIZON=10)",
        ),
        RowUpdate(
            7,
            A="任務設定",
            B="門檻(threshold)",
            C="threshold_ticks=1.0 (±1 tick)。",
            D="定義 Up/Down 與盤整邊界；也決定 class imbalance 強度。",
            E="±1 tick 是否小於 spread/手續費？",
            F="建議把 spread/費用納入：調整 threshold 或改用 dynamic threshold；並用 PR-AUC/回測 PnL 驗證。",
            G="new_era/tcn_pipeline_utils.py (THRESHOLD_TICKS=1.0)",
        ),
    ]

    # 資料集
    updates += [
        RowUpdate(
            8,
            A="資料集",
            B="原始資料來源",
            C="`data/NEW` 內台指期逐筆委託簿 CSV；本次使用 TXFB6 與 MXFB6。",
            D="老師會問標的/日期/盤別覆蓋度與資料品質。",
            E="用哪些商品與哪些日期？",
            F="建議在報告中列出 train/val/test session 名單與對應 samples(本表已填)。",
            G="new_era/01_feature_importance.py, new_era/02_feature_engineering.py",
        ),
        RowUpdate(
            9,
            A="資料集",
            B="資料型態 / 頻率",
            C="事件級(非時間 bar) 委託簿快照 + 成交欄位。",
            D="事件級資料對 leakage 與延遲要求更敏感；切割需嚴格按時間/Session。",
            E="是時間 bar 嗎？",
            F="明確聲明是 event-level sliding window。",
            G="src/training/datasets.py, new_era/02_feature_engineering.py",
        ),
        RowUpdate(
            10,
            A="資料集",
            B="委託簿深度",
            C="5 檔 bid/ask 價量。",
            D="深度決定可用的微結構訊息量。",
            E="為何 5 檔？",
            F="後續可做 top-1/top-3/top-5 ablation。",
            G="new_era/01_feature_importance.py",
        ),
        RowUpdate(
            11,
            A="資料集",
            B="成交資訊",
            C="使用 last_trade_price/qty (qty>0) 並在 feature builder 中整合進特徵。",
            D="成交流能補足單靠報價看不到的 aggressiveness / toxic flow。",
            E="成交與委託簿如何對齊？",
            F="建議補對齊方式與統計(例如成交落在 LOB 時間戳的延遲分佈)。",
            G="src/utils/tw_train_helpers.py (build_tw_feature_frame)",
        ),
        RowUpdate(
            12,
            A="資料集",
            B="Session 處理",
            C="按 session_id 分組做特徵與正規化；再做 session-level train/val/test split。",
            D="避免 sample-level shuffle 造成相鄰視窗洩漏。",
            E="視窗會跨日/跨盤嗎？",
            F="目前 per-session 建立 dataset；建議在投影片強調是 anti-leakage 設計。",
            G="new_era/tcn_pipeline_utils.py (split_records_by_session), new_era/02_feature_engineering.py",
        ),
        RowUpdate(
            13,
            A="資料集",
            B="樣本數 / Split (本次)",
            C=f"Train Data: {train_x_shape}\nValidation Data: {val_x_shape}\nTest Data: {test_x_shape}\nSession split: train=8, val=2, test=2",
            D="樣本量與 split 決定可信度，也影響過擬合判斷。",
            E="每個 split 有多少筆？",
            F="雖然 N 很大，但有效方向樣本比例極低；後續要改指標與策略用法。",
            G="new_era/05_train_tcn_pytorch.py, new_era/06_evaluate_tcn.py",
        ),
        RowUpdate(
            14,
            A="資料集",
            B="Train/Val/Test 切法",
            C="使用 split_records_by_session：每個 symbol 依 session_id 排序後切 train(70%)/val(15%)/test(剩餘)。",
            D="時間序列必須用『往後留』方式做 out-of-sample。",
            E="test 是否真的在時間上更後面？",
            F="本次 test session 為 20260127Night(較後)，符合 out-of-sample 方向；仍建議做 walk-forward。",
            G="new_era/tcn_pipeline_utils.py (split_records_by_session)",
        ),
        RowUpdate(
            15,
            A="資料集",
            B="Class 分佈 (本次)",
            C="極端不平衡：Stationary 約 99%。\n(可直接見 06_evaluate 的 support 與 09_mm_backtest_fixed 的診斷輸出)",
            D="在此情境 accuracy 幾乎沒有決策價值；需看 Up/Down precision/PR-AUC。",
            E="這樣的 accuracy 有意義嗎？",
            F="改用 macro-F1、balanced accuracy、PR-AUC；策略端只在高置信度時採用訊號，或先用於 maker skew/防禦撤單。",
            G="new_era/06_evaluate_tcn.py (classification_report), new_era/09_mm_backtest_fixed.py",
        ),
        RowUpdate(
            16,
            A="資料集",
            B="缺值處理",
            C="特徵表 per-session 先 dropna()；rolling z-score 的 NaN 以 0 補齊並 clip [-10,10]。",
            D="避免 NaN 進模型，但會丟掉每個 session 開頭一段資料。",
            E="每個 session 丟掉多少列？",
            F="建議在 02 輸出 dropna 前後筆數與比例，避免不小心偏移 label 分佈。",
            G="new_era/02_feature_engineering.py, new_era/tcn_pipeline_utils.py",
        ),
    ]

    # Label 設計
    updates += [
        RowUpdate(
            24,
            A="Label 設計",
            B="Label 規則(精確)",
            C="以 decision time t 的 mid-price 與 t+horizon 比較：\n(mid[t+h]-mid[t]) > +threshold => Up(2)\n(mid[t+h]-mid[t]) < -threshold => Down(0)\n其餘 => Stationary(1)\n本次 threshold=1 tick, horizon=10。",
            D="這是監督學習的正確答案，必須講清楚。",
            E="端點(end-point) 還是區間 max/min？",
            F="目前是端點；若交易更重視『期間內觸發』可做 label variant 比較，並用回測驗證。",
            G="src/features/sliding_window.py (label_direction), src/training/datasets.py",
        ),
        RowUpdate(
            25,
            A="Label 設計",
            B="Label 類別定義",
            C="0=Down, 1=Stationary, 2=Up。",
            D="避免訓練/評估/回測的 class id 對不上。",
            E="訓練與評估是否一致？",
            F="已一致；建議所有圖表/CSV 都固定同一順序 [Down,Stat,Up]。",
            G="new_era/06_evaluate_tcn.py",
        ),
    ]

    # 特徵工程（補上精確 Top8 名稱與 z-score 設定）
    top8 = (
        "obi, queue_imbalance, weighted_mid_deviation, trade_intensity_short, "
        "deep_obi, queue_imbalance_change, normalized_spread, cancel_pressure_short"
    )
    updates += [
        RowUpdate(
            19,
            A="特徵工程",
            B="特徵產生",
            C="使用 build_tw_feature_frame 從 LOB + Trades 產生微結構特徵，再挑 Top 8 做模型輸入。",
            D="把原始 LOB 轉成可學習的經濟含義特徵；老師通常會追問每個特徵的直覺。",
            E="輸入特徵到底是什麼？經濟意義是什麼？",
            F="建議報告時用一句話解釋每個 Top8：OBI/queue imbalance=買賣壓力；spread=摩擦；trade_intensity/cancel_pressure=毒性/流動性變化。",
            G="src/utils/tw_train_helpers.py (build_tw_feature_frame), new_era/tcn_pipeline_utils.py (TOP_8_FEATURES)",
        ),
        RowUpdate(
            20,
            A="特徵工程",
            B="特徵選擇",
            C=f"先用 RandomForest feature_importances_ 做粗篩，再固定使用 TOP_8_FEATURES={top8}。",
            D="降低維度與硬體成本；但 RF 的重要度不等價於 TCN 的重要度，需要再驗證。",
            E="為何是 Top 8？RF importance 可靠嗎？",
            F="把 Top4/8/12 做 ablation，並補 permutation importance / feature ablation（以 PR-AUC 與回測 PnL 為準）。",
            G="new_era/01_feature_importance.py, new_era/001_feature_importance.csv, new_era/tcn_pipeline_utils.py",
        ),
        RowUpdate(
            21,
            A="特徵工程",
            B="特徵維度",
            C=f"TCN 輸入維度=8（{top8}），張量形狀為 N×8×100（Conv1d 的 channel-first）。",
            D="維度越大越耗硬體/越容易過擬合；維度太小可能漏訊號。",
            E="為什麼只用 8 維？",
            F="這是 HLS-friendly baseline；後續比較 Top8 vs Top12 的 PR-AUC / 回測改善幅度。",
            G="new_era/tcn_pipeline_utils.py, new_era/05_train_tcn_pytorch.py",
        ),
        RowUpdate(
            22,
            A="特徵工程",
            B="正規化(rolling z-score)",
            C="每個 session 內對 Top 8 做 rolling z-score：window=2000、min_periods=100；std=0 以 1e-5 避免除零；NaN→0；clip [-10,10]。",
            D="穩定尺度、降低 regime shift 影響；rolling 計算本身要保持因果性。",
            E="會不會用到未來資料造成洩漏？",
            F="pandas rolling 是因果(只用過去)；建議再用 label-shuffle sanity check 驗證沒有洩漏。",
            G="new_era/tcn_pipeline_utils.py (apply_rolling_zscore)",
        ),
        RowUpdate(
            23,
            A="特徵工程",
            B="資料洩漏檢查(leakage)",
            C="已做 session-level split 並在每個 session 內產生特徵；但尚未用 label-shuffle / forbidden-feature 清單做系統性驗證。",
            D="教授通常會強力追問：是否有任何 feature 直接/間接用到未來 mid-price 或 label？",
            E="你怎麼證明沒有洩漏？",
            F="新增：1) label shuffle 實驗 2) 列出禁止欄位清單(任何 future-derived 欄位) 3) 檢查 rolling 統計是否只用過去。",
            G="src/utils/tw_train_helpers.py, src/training/datasets.py, meeting/w11/MODEL_review_full.md",
        ),
    ]

    # Model 架構
    updates += [
        RowUpdate(
            27,
            A="模型架構",
            B="模型類型",
            C="HLSFriendlyTCNClassifier (TCN + global average pooling)。",
            D="TCN 相對 RNN/Transformer 更平行、算子更簡單，更利於 HLS/FPGA。",
            E="為何不用 LSTM/Transformer？",
            F="TCN 具備低延遲與可硬體化優勢，且參數量可極小化。",
            G="src/models/tcn.py, new_era/05_train_tcn_pytorch.py",
        ),
        RowUpdate(
            28,
            A="模型架構",
            B="輸入張量形狀",
            C="X shape = (N, 8, 100)；其中 8=features channels，100=時間長度(window)。",
            D="Conv1d 需要 channel-first；維度錯會導致模型學錯或 shape error。",
            E="你確定沒有把 time/feature 軸搞反？",
            F="05 透過 concatenate_records(channel_first=True) 明確轉置到 (N, 8, 100)。",
            G="new_era/tcn_pipeline_utils.py (concatenate_records), new_era/05_train_tcn_pytorch.py",
        ),
        RowUpdate(
            29,
            A="模型架構",
            B="通道數/層數",
            C="channel_list=[16,16,16]，共 3 個 TemporalBlock。",
            D="控制容量與硬體資源。",
            E="是否太小？",
            F="先做 baseline；後續用 ablation (8/16/32 channels) 觀察 PR-AUC 與回測改善幅度。",
            G="new_era/05_train_tcn_pytorch.py, src/models/tcn.py",
        ),
        RowUpdate(
            30,
            A="模型架構",
            B="Kernel 大小",
            C="kernel_size=3。",
            D="影響局部時序模式抽取與感受野。",
            E="感受野夠嗎？",
            F="目前 RF=29 ticks；透過 global average pooling 彙總整段 100 ticks 的局部模式。若要更長依賴可加層/加 dilation。",
            G="src/models/tcn.py",
        ),
        RowUpdate(
            31,
            A="模型架構",
            B="Dilation / 感受野(Receptive Field)",
            C="dilation schedule=1,2,4 (3 層)。依程式公式：RF = 1 + 2*(k-1)*sum(dilation)=1+2*(3-1)*(1+2+4)=29 ticks。",
            D="決定單一位置能看到多長的歷史。",
            E="能不能看滿 100 ticks？",
            F="單一位置 RF=29；但模型最後做 global average pooling，等價於彙總整段 100 ticks 內的局部模式。若要捕捉更長期依賴可增加層數/ dilation。",
            G="src/models/tcn.py (receptive_field())",
        ),
        RowUpdate(
            33,
            A="模型架構",
            B="Activation / 正規化",
            C="Activation=ReLU；Normalization=無(BN/LN 未使用)；Dropout=0.1。",
            D="影響訓練穩定與 HLS 算子複雜度。",
            E="為何不加 BatchNorm？",
            F="優先維持 HLS-friendly、算子簡單；若需要再比較 BN/LN 的效益與硬體代價。",
            G="src/models/tcn.py",
        ),
        RowUpdate(
            34,
            A="模型架構",
            B="參數量",
            C=f"{model_params:,} 個參數。",
            D="容量/可部署性指標。",
            E="能塞進 FPGA 嗎？",
            F="參數量極小；後續仍要補上 HLS 合成資源(LUT/DSP/BRAM)與 latency 數字。",
            G="new_era/05_train_tcn_pytorch.py (Model Parameters)",
        ),
    ]

    # 訓練設定
    updates += [
        RowUpdate(
            35,
            A="訓練設定",
            B="Epoch 數",
            C=f"epochs=20；Best Val Acc={best_val_acc:.4f}。",
            D="影響是否收斂；需要曲線證據。",
            E="20 epochs 夠嗎？",
            F="目前未落盤曲線；建議輸出 training_history.csv 與 checkpoint/resume，再決定是否加 epochs / scheduler。",
            G="new_era/05_train_tcn_pytorch.py",
        ),
        RowUpdate(
            36,
            A="訓練設定",
            B="Batch size",
            C="train/val batch_size=1024。",
            D="影響收斂、噪聲與吞吐量。",
            E="為何 1024？",
            F="作為大資料的 baseline；後續比較 256/512/1024/2048 看 PR-AUC 與回測。",
            G="new_era/05_train_tcn_pytorch.py",
        ),
        RowUpdate(
            37,
            A="訓練設定",
            B="Learning rate",
            C="Adam lr=1e-3。",
            D="影響收斂速度與穩定性。",
            E="lr 是否太大/太小？",
            F="後續可加 scheduler 或測 3e-4；以 PR-AUC/回測 PnL 作比較基準。",
            G="new_era/05_train_tcn_pytorch.py",
        ),
        RowUpdate(
            38,
            A="訓練設定",
            B="Optimizer",
            C="Adam。",
            D="基準優化器；但不含 weight decay。",
            E="有沒有試 AdamW？",
            F="後續可測 AdamW(weight_decay=1e-4) 與 focal loss 的組合，看 minority precision 是否提升。",
            G="new_era/05_train_tcn_pytorch.py",
        ),
        RowUpdate(
            39,
            A="訓練設定",
            B="Loss / 類別權重",
            C=f"加權 CrossEntropyLoss；class weights={class_weights}",
            D="處理 class imbalance，使 Down/Up 對 loss 有足夠影響力。",
            E="權重會不會讓訓練不穩？",
            F="需同時看 per-class precision/recall；後續可比較 focal loss、或改策略只用高置信度訊號。",
            G="new_era/05_train_tcn_pytorch.py (Class Weights applied)",
        ),
        RowUpdate(
            40,
            A="訓練設定",
            B="類別權重計算",
            C="weights = total / (num_classes * count_c)（由 y_train 計算）。",
            D="讓少數類別在 loss 的影響力不被淹沒。",
            E="權重怎麼算的？",
            F="回報 y_train 的 class counts；並搭配 PR-AUC 觀察是否真的改善方向訊號。",
            G="new_era/tcn_pipeline_utils.py (compute_class_weight_tensor)",
        ),
        RowUpdate(
            41,
            A="訓練設定",
            B="LR scheduler",
            C="目前未使用。",
            D="可改善 plateau 時的收斂。",
            E="為何沒有 scheduler？",
            F="後續可加 ReduceLROnPlateau 或 cosine，並以 PR-AUC/回測 PnL 作為選模指標。",
            G="new_era/05_train_tcn_pytorch.py",
        ),
        RowUpdate(
            42,
            A="訓練設定",
            B="Early stopping",
            C="未使用（僅保存 best val_acc 權重，不會提前停止）。",
            D="避免過擬合與浪費訓練時間。",
            E="怎麼避免 overfit？",
            F="加入 patience-based early stopping（建議監控 macro-F1 或 PR-AUC，而不是 val_acc）。",
            G="new_era/05_train_tcn_pytorch.py",
        ),
        RowUpdate(
            43,
            A="訓練設定",
            B="Checkpoint / Resume",
            C="目前只保存 best model state_dict 到 new_era/005_tcn_weights.pth；未保存 optimizer state，因此無法 resume。",
            D="訓練更久/做更多實驗時必須可續訓且可重現。",
            E="中斷後能續訓嗎？",
            F="改存 checkpoint(dict: model/optimizer/epoch/best/history)，並支援 resume。",
            G="new_era/05_train_tcn_pytorch.py",
        ),
        RowUpdate(
            44,
            A="訓練設定",
            B="隨機種子 / 重現性",
            C="RF 有 random_state=42；PyTorch 訓練目前未固定 numpy/torch/cuda seeds。",
            D="可重現性是研究可信度基本要求。",
            E="你能 100% 重現嗎？",
            F="建議加入 seed 固定與 deterministic 設定，並把 seed 記錄到 experiment log。",
            G="new_era/01_feature_importance.py, new_era/05_train_tcn_pytorch.py",
        ),
    ]

    # 評估：把你最新 06 的輸出與檔案路徑寫進來
    updates += [
        RowUpdate(
            47,
            A="評估",
            B="訓練/驗證曲線",
            C="本次有印出每 epoch 的 train_loss/train_acc/val_loss/val_acc，但尚未落盤成 CSV/圖表。",
            D="曲線能證明是否收斂，以及是否欠擬合/過擬合。",
            E="能不能給我 loss curve？",
            F="在 05 增加 training_history.csv，並輸出 plot；同時記錄 best checkpoint 對應 epoch。",
            G="new_era/05_train_tcn_pytorch.py",
        ),
        RowUpdate(
            48,
            A="評估",
            B="驗證集表現(Val)",
            C=f"Loss={val_loss:.6f}；Accuracy={val_acc:.2%}",
            D="用於選模；但在不平衡資料下要同時看 per-class 指標。",
            E="accuracy 可信嗎？",
            F="同時報告 confusion matrix、macro avg、以及 Up/Down precision；策略端避免把低 precision 當成 taker 訊號。",
            G=f"new_era/06_evaluate_tcn.py；輸出檔：{eval_outputs_dir}",
        ),
        RowUpdate(
            49,
            A="評估",
            B="測試集表現(Test)",
            C=f"Loss={test_loss:.6f}；Accuracy={test_acc:.2%}",
            D="out-of-sample performance；只能用一次做最終報告。",
            E="是不是調參調到 test？",
            F="固定 test 不回頭調參；後續採 walk-forward 評估更貼近部署。",
            G=f"new_era/06_evaluate_tcn.py；輸出檔：{eval_outputs_dir}",
        ),
        RowUpdate(
            50,
            A="評估",
            B="混淆矩陣(Val/Test)",
            C=f"Val CM (Counts, [Down,Stat,Up]):\n{val_cm}\n\nTest CM (Counts):\n{test_cm}",
            D="看模型是否只會預測 Stationary，以及 Up/Down 錯誤型態。",
            E="模型是不是只會預測盤整？",
            F="應以 normalized confusion matrix 觀察 class-wise recall；目前 Up/Down precision 很低，較適合用於 maker skew/防禦。",
            G=f"{eval_outputs_dir} (confusion_matrix_count/normalized.png)",
        ),
        RowUpdate(
            51,
            A="評估",
            B="Precision/Recall/F1 (Val/Test)",
            C=f"Val 摘要：{val_report_summary}\nTest 摘要：{test_report_summary}",
            D="在不平衡資料下，比 accuracy 更重要。",
            E="Up/Down 的 precision 有多少？",
            F="目前 Up/Down precision 約 1~2%；需改用 PR-AUC、校準與更嚴格門檻，並讓策略端先採用 maker-only 的用法。",
            G=f"{eval_outputs_dir} (classification_report.csv)",
        ),
        RowUpdate(
            52,
            A="評估",
            B="ROC-AUC / PR-AUC",
            C="目前尚未輸出/保存 softmax 機率，因此尚未計算 ROC-AUC/PR-AUC（尤其 Up/Down 的 PR-AUC）。",
            D="在極端不平衡下 PR-AUC 比 accuracy 更能反映方向訊號品質。",
            E="你能量化『方向訊號』到底有多可靠嗎？",
            F="修改 06：輸出 softmax 機率（prob_down/prob_stat/prob_up），並計算 one-vs-rest PR-AUC；建議同時做 calibration。",
            G="new_era/06_evaluate_tcn.py（需擴充：輸出機率與 PR-AUC）",
        ),
        RowUpdate(
            53,
            A="評估",
            B="Loss/曲線視覺化",
            C="目前沒有把 train/val loss/acc 曲線輸出成圖或 CSV（只有 console print）。",
            D="曲線能證明是否收斂，以及最佳 checkpoint 是否合理。",
            E="你怎麼知道模型有收斂？",
            F="在 05 加入 training_history.csv，並輸出 loss/acc 曲線圖；同時記錄 best checkpoint 對應 epoch。",
            G="new_era/05_train_tcn_pytorch.py（需擴充：寫出 history）",
        ),
    ]

    # 部署/交易：回測（用你最新的 09_mm_backtest_fixed 輸出）
    updates += [
        RowUpdate(
            62,
            A="部署/交易",
            B="交易成本 / 回測(含成本)",
            C=(
                "已建立 session-aware 的 TCN 造市回測(固定版 maker-only 策略)。\n"
                f"Test session: {mm_session}\n"
                f"Samples: {mm_samples:,}\n"
                f"Accuracy: {mm_accuracy:.4f}\n"
                f"Confusion Matrix [Down,Stat,Up]: {mm_cm}\n"
                f"True dist: {mm_true_dist}\n"
                f"Pred dist: {mm_pred_dist}\n"
                f"Spread: {mm_spread}\n"
                "\nSignal diagnostics:\n"
                f"{mm_threshold_diag}\n"
                "\nStrategy results:\n"
                f"{mm_strategy_results}"
            ),
            D="分類分數不等於可交易；必須在含成本的回測裡驗證訊號是否能轉成 PnL。",
            E="扣掉成本還能賺嗎？該怎麼用這個訊號？",
            F="目前觀察：Up/Down precision 低，較適合用於 maker skew/防禦撤單，而非頻繁 taker 出手。下一步做 walk-forward 回測檢查穩健性。",
            G="new_era/09_mm_backtest_fixed.py, new_era/tcn_pipeline_utils.py (maker/taker fee + point value)",
        )
    ]

    # 正則化
    updates += [
        RowUpdate(
            45,
            A="正則化",
            B="Weight decay",
            C="未使用（Adam 無 weight_decay/AdamW）。",
            D="可抑制過擬合、改善泛化。",
            E="有試過 AdamW 嗎？",
            F="後續測 AdamW + 合理 weight_decay；並用 PR-AUC/回測改善做比較。",
            G="new_era/05_train_tcn_pytorch.py",
        ),
        RowUpdate(
            46,
            A="正則化",
            B="Gradient clipping",
            C="未使用。",
            D="可降低梯度爆炸、提高訓練穩定性。",
            E="訓練會不會不穩？",
            F="若 loss 波動/爆炸，加入 clip_grad_norm_，並配合 scheduler。",
            G="new_era/05_train_tcn_pytorch.py",
        ),
    ]

    # 穩健性 / 合理性檢查 / 部署交易欄位（把原表的 Missing placeholder 清掉）
    updates += [
        RowUpdate(
            54,
            A="穩健性",
            B="Out-of-sample 測試",
            C="已有：session-level test split（test sessions=20260127Night）。",
            D="檢查是否能泛化到未見過的 session。",
            E="test 是否真的是『未來』？",
            F="目前按 session_id 排序切分；後續用 walk-forward 更嚴格。",
            G="new_era/tcn_pipeline_utils.py, new_era/06_evaluate_tcn.py",
        ),
        RowUpdate(
            55,
            A="穩健性",
            B="Walk-forward 測試",
            C="尚未實作。",
            D="比單次 split 更貼近真實部署（rolling）。",
            E="不同市場狀態還有效嗎？",
            F="按 session 日期 rolling train→val→test，並在每窗做含成本造市回測。",
            G="（待新增腳本）",
        ),
        RowUpdate(
            56,
            A="穩健性",
            B="噪聲 / 擾動測試",
            C="尚未實作。",
            D="檢查模型是否脆弱依賴少數特徵或少數 regime。",
            E="加噪後表現如何？",
            F="在 val/test 對特徵做 mask/noise，觀察 PR-AUC 與回測 PnL 退化。",
            G="（可擴充 new_era/06_evaluate_tcn.py）",
        ),
        RowUpdate(
            57,
            A="合理性檢查",
            B="Label shuffle 測試",
            C="尚未實作。",
            D="偵測 leakage：若 label 打亂仍高分，代表可能有洩漏。",
            E="如何證明模型沒偷看未來？",
            F="新增 label-shuffle 訓練/評估並保存結果（應接近隨機）。",
            G="（待新增腳本）",
        ),
        RowUpdate(
            58,
            A="合理性檢查",
            B="特徵重要度(可解釋性)",
            C="已產出 RF 的 feature importance（001_feature_importance.csv）。TCN 尚未做 permutation importance。",
            D="教授會希望看到：模型用的訊號是否合理、是否可解釋。",
            E="TCN 真的是靠這些特徵在預測嗎？",
            F="補 permutation importance / feature ablation（對 PR-AUC 與回測 PnL 的影響）。",
            G="new_era/001_feature_importance.csv, new_era/01_feature_importance.py",
        ),
        RowUpdate(
            59,
            A="部署/交易",
            B="推論延遲(Inference latency)",
            C="尚未量測 PyTorch batch=1 latency；亦尚未把此權重對應到 HLS 合成報告做 latency 對照。",
            D="HFT/FPGA 的價值主張核心：低延遲。",
            E="到底多快？",
            F="新增 benchmark：PyTorch CPU batch=1/批次吞吐；並以同權重做 hls4ml synth 報告（latency/resource）。",
            G="new_era/07_hls4ml_tcn_conversion.py, new_era/007_hls4ml_tcn_prj",
        ),
        RowUpdate(
            60,
            A="部署/交易",
            B="模型大小 / 參數",
            C=f"參數量 {model_params:,}；權重檔 new_era/005_tcn_weights.pth 約 25 KB。",
            D="決定能否塞進 FPGA/BRAM 以及可達到的時脈。",
            E="能塞進 PYNQ-Z2 嗎？",
            F="補上 HLS 資源(LUT/DSP/BRAM)與時脈/latency 以形成完整部署證據鏈。",
            G="new_era/005_tcn_weights.pth, new_era/05_train_tcn_pytorch.py",
        ),
        RowUpdate(
            61,
            A="部署/交易",
            B="吞吐量(Throughput)",
            C="尚未量測。",
            D="需要能跟上 market data feed 的事件速度。",
            E="是否能即時處理？",
            F="補 PyTorch/HLS samples/sec；並估算在真實行情下的峰值負載。",
            G="（待新增 benchmark）",
        ),
    ]

    # Apply updates
    for upd in updates:
        _set_row(ws, upd)

    # Translate remaining section names in column A (keep already-Chinese ones)
    for r in range(2, ws.max_row + 1):
        a = ws[f"A{r}"].value
        if a == "Training 設定":
            ws[f"A{r}"].value = "訓練設定"
        elif a == "Regularization":
            ws[f"A{r}"].value = "正則化"
        elif a == "Evaluation":
            ws[f"A{r}"].value = "評估"
        elif a == "Robustness":
            ws[f"A{r}"].value = "穩健性"
        elif a == "Sanity check":
            ws[f"A{r}"].value = "合理性檢查"

    # Missing Info sheet: rewrite to Traditional Chinese and update with current status.
    ws_m = wb["Missing Info"]
    ws_m["A1"].value = "優先級"
    ws_m["B1"].value = "缺漏 / 不足資訊（目前狀態）"
    ws_m["C1"].value = "為何重要"
    ws_m["D1"].value = "如何取得 / 補齊"
    ws_m["E1"].value = "建議記錄位置"

    # Clear old rows (keep header)
    for r in range(2, 200):
        for c in range(1, 6):
            ws_m.cell(r, c).value = None

    missing_rows = [
        (
            "High",
            "Up/Down 訊號品質指標（PR-AUC / calibration）：尚未完成",
            "在 Stationary 佔比 ~99% 時，比 accuracy 更能反映『可交易訊號』品質。",
            "修改 06：輸出 softmax 機率；計算 one-vs-rest PR-AUC；做 reliability diagram / ECE。",
            "Model Review: 評估區 + 附圖",
        ),
        (
            "High",
            "訓練曲線落盤（training_history.csv）：尚未完成",
            "需要向老師證明是否收斂/是否欠擬合或過擬合。",
            "修改 05：每 epoch 寫入 CSV；並把 best checkpoint 對應的 epoch 記錄下來。",
            "Model Review: 訓練設定/評估區",
        ),
        (
            "High",
            "Walk-forward（rolling）回測：尚未完成",
            "單次 split 不足以證明穩健性；walk-forward 更接近真實部署。",
            "以 session 日期 rolling train→val→test；每窗跑含成本 maker-only 回測。",
            "Model Review: 穩健性/部署交易區",
        ),
        (
            "Medium",
            "PyTorch / HLS 延遲與吞吐量 benchmark：尚未量測",
            "HFT/FPGA 的主張需要數字：batch=1 latency、samples/sec、HLS 資源與時脈。",
            "新增 benchmark 腳本；並把同一組權重對應到 hls4ml 專案的 synth 報告。",
            "Model Review: 部署/交易區",
        ),
        (
            "Medium",
            "隨機種子固定（torch/numpy/cuda）：尚未完成",
            "研究可重現性的基本要求。",
            "在 05/06 加入 seed 設定與 deterministic 配置；將 seed 寫入 log。",
            "Model Review: 訓練設定區",
        ),
        (
            "Low",
            "Label shuffle sanity check：尚未完成",
            "快速檢查是否存在 leakage：打亂 label 應使模型表現接近隨機。",
            "新增一個 label-shuffle 訓練/評估腳本並保存結果。",
            "Model Review: 合理性檢查區",
        ),
    ]

    for i, row in enumerate(missing_rows, start=2):
        for j, val in enumerate(row, start=1):
            ws_m.cell(i, j).value = val

    # Pipeline Map header translation (if present)
    if "Pipeline Map" in wb.sheetnames:
        w = wb["Pipeline Map"]
        if w["A1"].value == "Step":
            w["A1"].value = "步驟"
            w["B1"].value = "腳本/檔案"
            w["C1"].value = "輸入"
            w["D1"].value = "輸出"
            w["E1"].value = "關鍵設定"
            w["F1"].value = "備註"

    if "Changes Made" in wb.sheetnames:
        w = wb["Changes Made"]
        if w["A1"].value == "Change":
            w["A1"].value = "變更"
            w["B1"].value = "原因/說明"
            w["C1"].value = "證據"
            w["D1"].value = "狀態"

    # Save: try overwrite original first (often locked if Excel is open); fallback to out_path.
    try:
        wb.save(in_path)
        saved_path = in_path
    except PermissionError:
        wb.save(out_path)
        saved_path = out_path

    # Also emit a professor-facing Markdown summary organized by section.
    md_path = saved_path.with_suffix("").with_suffix(".md")
    write_model_review_md(ws, wb["Missing Info"], md_path)


def _iter_rows(ws) -> Iterable[tuple[str, str, str, str, str, str]]:
    # Returns (section, item, filled, definition, teacher_ask, action, evidence)
    for r in range(2, ws.max_row + 1):
        section = ws[f"A{r}"].value
        item = ws[f"B{r}"].value
        filled = ws[f"C{r}"].value
        definition = ws[f"D{r}"].value
        teacher_ask = ws[f"E{r}"].value
        action = ws[f"F{r}"].value
        evidence = ws[f"G{r}"].value
        if any(v not in (None, "") for v in (section, item, filled, definition, teacher_ask, action, evidence)):
            yield (
                str(section or "").strip(),
                str(item or "").strip(),
                str(filled or "").strip(),
                str(definition or "").strip(),
                str(teacher_ask or "").strip(),
                str(action or "").strip(),
                str(evidence or "").strip(),
            )


def write_model_review_md(ws_review, ws_missing, md_path: Path) -> None:
    rows = list(_iter_rows(ws_review))
    by_section: dict[str, list[tuple[str, str, str, str, str, str]]] = {}
    for section, item, filled, definition, teacher_ask, action, evidence in rows:
        by_section.setdefault(section or "（未分類）", []).append(
            (item, filled, definition, teacher_ask, action, evidence)
        )

    lines: list[str] = []
    lines.append("# MODEL Review（教授用完整版）")
    lines.append("")
    lines.append("此文件由 `meeting/w11/MODEL_filled_review.xlsx` 內容依 section 統整而成，並已把本次訓練/評估/回測輸出填入。")
    lines.append("")

    for section in [s for s in by_section.keys() if s]:
        lines.append(f"## {section}")
        lines.append("")
        for item, filled, definition, teacher_ask, action, evidence in by_section[section]:
            lines.append(f"### {item}")
            if filled:
                lines.append("**本次答案**")
                lines.append("")
                lines.append(filled)
                lines.append("")
            if definition:
                lines.append("**為何重要**")
                lines.append("")
                lines.append(definition)
                lines.append("")
            if teacher_ask:
                lines.append("**老師可能會問**")
                lines.append("")
                lines.append(teacher_ask)
                lines.append("")
            if action:
                lines.append("**建議下一步**")
                lines.append("")
                lines.append(action)
                lines.append("")
            if evidence:
                lines.append("**證據 / 來源**")
                lines.append("")
                lines.append(evidence)
                lines.append("")

    # Missing Info
    lines.append("## 缺漏 / 待補強項目（從 Missing Info sheet 彙整）")
    lines.append("")
    for r in range(2, ws_missing.max_row + 1):
        pri = ws_missing.cell(r, 1).value
        item = ws_missing.cell(r, 2).value
        why = ws_missing.cell(r, 3).value
        how = ws_missing.cell(r, 4).value
        where = ws_missing.cell(r, 5).value
        if not any(v not in (None, "") for v in (pri, item, why, how, where)):
            continue
        lines.append(f"- **{pri}**：{item}")
        if why:
            lines.append(f"  - 重要性：{why}")
        if how:
            lines.append(f"  - 補齊方式：{how}")
        if where:
            lines.append(f"  - 記錄位置：{where}")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    in_path = root / "meeting" / "w11" / "MODEL_filled_review.xlsx"
    out_path = root / "meeting" / "w11" / "MODEL_filled_review.zhTW.filled.xlsx"

    fill_workbook(
        in_path=in_path,
        out_path=out_path,
        train_x_shape="X shape (886133, 8, 100), y shape (886133,)",
        val_x_shape="X shape (213204, 8, 100), y shape (213204,)",
        test_x_shape="X shape (389652, 8, 100), y shape (389652,)",
        class_weights="[67.37628936767578, 0.3366035223007202, 69.91187286376953]",
        model_params=4515,
        best_val_acc=0.8121,
        val_loss=0.701812,
        val_acc=0.8121,
        test_loss=0.713121,
        test_acc=0.7986,
        val_cm="[[306, 886, 86], [12273, 172368, 25873], [39, 896, 477]]",
        test_cm="[[326, 783, 63], [23724, 310235, 53253], [19, 646, 603]]",
        val_report_summary="Down precision=0.02 recall=0.24; Up precision=0.02 recall=0.34; Stationary precision=0.99 recall=0.82; macro avg f1=0.32",
        test_report_summary="Down precision=0.01 recall=0.28; Up precision=0.01 recall=0.48; Stationary precision=1.00 recall=0.80; macro avg f1=0.31",
        eval_outputs_dir=str((root / "new_era" / "eval_outputs").resolve()),
        mm_session="TXFB6 20260127Night",
        mm_samples=755736,
        mm_accuracy=0.8136,
        mm_cm="[[927, 2563, 230], [40425, 612276, 95455], [67, 2146, 1647]]",
        mm_true_dist="[3720, 748156, 3860]",
        mm_pred_dist="[41419, 616985, 97332]",
        mm_spread="avg 2.144 ticks | median 2.000 ticks",
        mm_threshold_diag=(
            "prob>=0.55: active=67929, direction_precision=0.0237, stationary_rate=0.9742\n"
            "prob>=0.60: active=47397, direction_precision=0.0251, stationary_rate=0.9726\n"
            "prob>=0.70: active=19938, direction_precision=0.0293, stationary_rate=0.9685\n"
            "prob>=0.80: active=5167, direction_precision=0.0379, stationary_rate=0.9599\n"
            "prob>=0.90: active=360, direction_precision=0.0778, stationary_rate=0.9222"
        ),
        mm_strategy_results=(
            "Baseline passive maker: PnL=97,800 TWD, trades=454 (maker=454), ending_inv=-2, max_abs_inv=3\n"
            "Best sweep (top): tcn_skew prob_threshold=0.70 edge_threshold=0.15, PnL=145,600 TWD (+47,800), trades=521, ending_inv=-1, max_abs_inv=3"
        ),
    )


if __name__ == "__main__":
    main()
