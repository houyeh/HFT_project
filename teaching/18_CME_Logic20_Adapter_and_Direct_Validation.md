# 18. CME Logic20 Adapter 與直接驗證

## 這份文件在講什麼
這一章只做一件事：把 `CME` 資料接進我們現有的 TW pipeline，然後直接拿現有模型去驗證，看它能不能跨市場工作。

這個階段刻意不做太多延伸，因為我們要先回答最基本的問題：

> 台股 replay 上訓練好的模型，丟到 CME 上，有沒有基本通用性？

---

## 1. 資料來源與格式

目前 CME 資料解壓在：
- [data/raw/CME](/C:/Users/holbo/00VS_projjects/00_HFT_project/data/raw/CME)

這批資料不是理想中的完整 `5 檔 / 10 檔 MBP`，而是 `Logic20` 風格的快照資料，所以我們需要先做 adapter。

對應程式：
- [cme_logic20.py](/C:/Users/holbo/00VS_projjects/00_HFT_project/src/data/cme_logic20.py)

這個 adapter 的工作是：
- 讀出 CME 檔案
- 把欄位轉成我們現有 pipeline 可接受的標準 LOB schema
- 整理出 quote frame 與 trade frame

---

## 2. 驗證方法

對應腳本：
- [evaluate_cme_frozen_model.py](/C:/Users/holbo/00VS_projjects/00_HFT_project/scripts/02_train_eval/evaluate_cme_frozen_model.py)

流程如下：

1. 使用 CME adapter 讀出 `ESM6`、`CLK6`
2. 依照 TW 線的 feature pipeline 建出 `execution_aware` 特徵
3. 使用同一個 `window=100 / horizon=10 / threshold=1 tick`
4. 不重新訓練，直接把 frozen TW TCN 丟上去
5. 評估分類結果與 signal backtest

這個設計的目的很清楚：
- 不是要追求最好成績
- 而是要先測「現有模型能不能直接搬去 CME」

---

## 3. 直接驗證結果

主要 artifact：
- [summary.json](/C:/Users/holbo/00VS_projjects/00_HFT_project/artifacts/02_model_runs/cme_eval_transfer/20260407_161938/summary.json)

關鍵數字：
- `gated accuracy = 0.8202`
- `gated macro_f1 = 0.3868`
- `gated signal net pnl = -36316.5 ticks`

---

## 4. 該怎麼解讀

### 為什麼 accuracy 看起來不低，但 PnL 很差？
這是高頻資料常見的狀況。

原因是：
- 大部分樣本其實是 `Stationary`
- 模型只要很會判斷「大多時候不太動」，accuracy 就可能不低
- 但交易真正需要的是：
  - 在少數會動的時候判對方向
  - 且方向 edge 要足夠大，能覆蓋成本

所以這個結果說明的是：
- 模型有一些分類能力
- 但還沒有形成可交易的跨市場 edge

---

## 5. 為什麼這一步很重要
很多時候研究容易一開始就直接做 fine-tune、calibration、walk-forward，但這樣會很快把問題搞複雜。

先做 direct validation 的價值是：
- 可以先檢查 adapter 有沒有接通
- 可以先看現有 feature 是否有基本通用性
- 可以先知道問題是「資料接不上」還是「模型 edge 不夠」

在這個專案裡，答案很清楚：

> 問題不是 CME 接不起來，而是現有共享模型在 CME 上沒有直接形成可交易 edge。

---

## 6. 下一步
因此本專案接下來的 CME 路線不再是：
- 直接硬搬 TW 模型

而是：
- `ES` 一顆模型
- `CL` 一顆模型
- `NQ` 一顆模型

也就是從「共享模型 direct transfer」改成「每個商品各自訓練專屬模型」。

---

## 7. 小結
這一章最重要的結論只有一句：

> CME adapter 已完成，現有模型已驗證過，但效果不好，因此下一步應該改做分商品專屬模型，而不是再假設共享模型可以直接通用。
