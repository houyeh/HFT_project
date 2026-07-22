# Week 6: CME 資料接軌、直接驗證與主線收斂

## 1. 本週目標
這週的工作有兩個主題：

1. 把 `CME` 資料真正接到現有研究 pipeline。
2. 用最直接的方式驗證：`TW replay` 上訓練好的 TCN，能不能直接搬到 `CME` 使用。

這週做完之後，我們也同步做了一個重要決策：

> `CME` 先停在「資料接軌完成、直接驗證失敗」這個清楚狀態，不再讓不穩定的支線實驗拖慢主線。

---

## 2. 實際完成的工作

### 2.1 解壓並整理 CME 原始資料
- 原始壓縮檔：`CME_data.zip`
- 解壓後位置：`data/raw/CME/`
- 目前確認可用的資料夾：
  - `ES_MES`
  - `CL_MCL`
  - `__ES_MES`

這一步代表資料面已經不再卡住，我們手上已經有可以實際拿來做 adapter 與驗證的 `CME Logic20` 原始檔。

### 2.2 完成 CME Logic20 adapter
新增：
- [src/data/cme_logic20.py](/C:/Users/holbo/00VS_projjects/00_HFT_project/src/data/cme_logic20.py)

這個 adapter 做的事情是：
- 讀入 `Logic20_SetProduct` CSV
- 將雙商品快照拆成單商品事件流
- 映射成專案現有的標準 LOB schema
- 補出可供 feature pipeline 使用的欄位

核心輸出欄位包含：
- `bid_px_1`, `bid_vol_1`
- `ask_px_1`, `ask_vol_1`
- `bid_px_2`, `bid_vol_2`
- `ask_px_2`, `ask_vol_2`
- `trade_price`
- `trade_volume`

### 2.3 對齊現有 feature pipeline
目前 `CME Logic20` 實際可用的是前兩檔快照，不是完整 5 檔 MBP。  
為了讓既有 feature pipeline 先走通，這週採用的是「保守對齊」策略：

- 第 `3` 到 `5` 檔價格，先依 tick size 向外展開
- 第 `3` 到 `5` 檔量，先補 `0`

這樣做的目的不是假裝它就是完整 5 檔，而是先確保：
- pipeline 可重用
- feature 順序不變
- 後續若拿到更完整 CME 5 檔資料，可以直接替換 adapter

### 2.4 完成 frozen model 直接驗證腳本
新增：
- [scripts/02_train_eval/evaluate_cme_frozen_model.py](/C:/Users/holbo/00VS_projjects/00_HFT_project/scripts/02_train_eval/evaluate_cme_frozen_model.py)

這支腳本會：
- 讀取目前已凍結的 `TW replay` TCN 權重
- 直接在 `CME` 資料上做 inference
- 輸出 classification metrics、gating 後 metrics 與 signal backtest

這一步的意義是：

> 先回答「跨市場直接搬用」行不行，再決定要不要投入更多時間做 `CME` 重訓。

---

## 3. 本週採用的模型與特徵設定

### 3.1 Sample 定義
目前專案不是用固定秒數 K 棒，而是：

> event-level / tick-level snapshot

也就是每一筆有效 quote / book event 都是一個基本事件單位。

### 3.2 輸入視窗與標籤設定
本週 `CME direct validation` 主要沿用目前主線模型設定：
- `window_size = 100`
- `horizon = 10`
- `threshold_ticks = 1.0`
- `sample_stride = 1000`

意思是：
- 模型每次看最近 `100` 個事件
- 預測未來 `10` 個事件後的方向
- 如果未來價格變動不到 `1 tick`，就標成 `Stationary`

### 3.3 Feature set
本週使用的是 `execution_aware` 共 `12` 維特徵：

1. `normalized_spread`
2. `micro_return`
3. `obi`
4. `weighted_mid_deviation`
5. `deep_obi`
6. `queue_imbalance`
7. `queue_imbalance_change`
8. `aggressor_imbalance_interval`
9. `rolling_signed_flow_ratio`
10. `trade_intensity_short`
11. `net_cancel_pressure`
12. `cancel_pressure_short`

這組特徵的設計目的，是同時保留：
- 價格與價差資訊
- 委託簿不平衡資訊
- 成交與撤單流資訊

---

## 4. CME 直接驗證結果

### 4.1 主要驗證結果
主要參考結果：
- [summary.json](/C:/Users/holbo/00VS_projjects/00_HFT_project/artifacts/02_model_runs/cme_eval_transfer/20260407_161938/summary.json)

設定：
- symbols：`ESM6`, `CLK6`
- feature set：`execution_aware`
- sample stride：`1000`

結果：
- `gated accuracy = 0.8202`
- `gated macro_f1 = 0.3868`
- `gated signal net pnl = -36,316.5 ticks`

### 4.2 如何解讀
這組結果代表：
- `adapter` 是通的
- `TW` 上凍結的 TCN 可以直接在 `CME` 上完整跑通 pipeline
- 分類指標看起來不算差
- 但實際交易結果仍是大幅負值

也就是說：

> 現在不能因為 accuracy 還可以，就說模型已經可以跨市場通用。

### 4.3 為什麼 accuracy 不差，但 PnL 很差
目前最合理的解釋是：
- 模型仍然抓到很多 `Stationary`
- 因此分類表面上不錯
- 但真正和交易有關的 `directional edge` 並沒有被學到足夠好

這正是高頻模型常見的陷阱：

> 分類能跑，不代表交易能賺。

---

## 5. 這週的關鍵決策：先把 CME 收斂回直接驗證失敗

這週做完後，我們決定把 `CME` 的 repo 狀態與敘事收斂為：

1. `CME` 資料已接上
2. 已完成 direct validation
3. direct validation 效果不好
4. 下週不再硬用共享模型，而是改成每個商品各自訓練

這個決策的目的，是避免下面兩種情況：
- 支線實驗太多，報告講不清楚
- repo 裡保留過多中間版本，讓真正主線失焦

---

## 6. 本週同步做的整理

### 6.1 保留的 CME 核心內容
- [src/data/cme_logic20.py](/C:/Users/holbo/00VS_projjects/00_HFT_project/src/data/cme_logic20.py)
- [scripts/02_train_eval/evaluate_cme_frozen_model.py](/C:/Users/holbo/00VS_projjects/00_HFT_project/scripts/02_train_eval/evaluate_cme_frozen_model.py)
- [artifacts/02_model_runs/cme_eval_smoke](/C:/Users/holbo/00VS_projjects/00_HFT_project/artifacts/02_model_runs/cme_eval_smoke)
- [artifacts/02_model_runs/cme_eval_transfer](/C:/Users/holbo/00VS_projjects/00_HFT_project/artifacts/02_model_runs/cme_eval_transfer)

### 6.2 移除的支線
已整理掉不再作為主線敘事的支線腳本與 artifact，包括：
- `CME calibration` 支線
- `CME walk-forward` 支線
- 對應的中間 artifact 與 smoke test

這樣做之後，現在 repo 中的 `CME` 位置就很清楚：

> 它是已接軌、已驗證，但尚未成功的次線任務。

---

## 7. 下週規劃

`CME` 下週的明確任務是：

1. `ES` 一顆模型
2. `CL` 一顆模型
3. `NQ` 一顆模型

目前這包資料先確認到 `ES / CL` 系列，因此會先做：
- `ES` 專屬模型
- `CL` 專屬模型

`NQ` 會沿用同一個訓練流程設計，資料到位就直接接上。

---

## 8. 本週一句話總結
Week 6 的真正成果不是把 `CME` 做成正報酬，而是把問題定位清楚：

> `CME` 資料已經接上，直接驗證也做完了，但現有共享模型的跨市場交易效果不好，因此接下來要改走分商品模型，而不是繼續硬撐共享 transfer。
