# 1. 模型探索與選型定調 (Model Experimentation & Identity)

## 1.1 Dataset, Parameters, Features & Labels (設定與實驗變數)

在過去幾週，我們針對 TCN 模型進行了多維度的變數嘗試，確保我們的實驗設計符合最嚴謹的學術與業界標準。以下是最終確立並採用的核心參數：

* **資料集 (Dataset)**：
  * 使用台指期 (TXF) LOB 高頻數據。
  * **切分方法 (Data Splitting)**：嚴格依照 Session 時間順序進行切分 (Train/Val/Test)，絕不使用 Random Shuffle，確保**無未來數據洩漏 (Zero Look-ahead Bias)**。
* **特徵與視窗 (Features & Parameters)**：
  * **特徵數量 (Feature Count)**：8 個精煉的高頻微觀特徵 (Micro-structure features)。包含價差、掛單量不平衡等。
  * **觀察視窗 (Window Size)**：100 Ticks (確保模型能捕捉到足夠深度的微觀結構變化)。
  * **預測視距 (Horizon)**：5 Ticks。
* **標籤定義 (Label Evolution)**：
  * 經歷多次迭代後，我們選用 `smoothed_mid_move_3class`，將未來 5 步的價格變化分為三類：**Down (下跌)**, **Stationary (盤整/微幅震盪)**, **Up (上漲)**。

## 1.2 訓練結果與學習曲線 (Training Results)

下圖展示了模型訓練過程中的 Epoch-Loss 與 Macro F1 曲線：

![Training History Plot](train_history_plot.png)

* **專業解讀**：
  * **Loss 曲線**顯示 Training 與 Validation 穩定收斂，未見嚴重的 Overfitting。
  * **Macro F1** 同步上升並在 Validation 上找到最佳點 (Min Val Loss / Max Val F1 標示處)，這證明我們的 TCN 確實能從高頻雜訊中穩定抽取有效特徵。

## 1.3 分類指標深度剖析 (Deep Dive into Classification Metrics)

要決定模型「該如何被使用」，我們必須檢視最重要的 Confusion Matrix (混淆矩陣) 及其 Percentage 分布：

![Confusion Matrix Percent](confusion_matrix_percent_test.png)

* **觀察現象**：
  * 由於高頻市場絕大多數時間處於盤整，**Stationary 類別佔據了壓倒性的多數**。
  * 導致的結果是：模型預測 Down/Up 的 **Precision 極低** (約 2.8%)。當模型喊出 Down 時，有高達 90% 以上的情況其實是 Stationary。
  * 然而，模型的 **Recall 相對優異** (約 65%)。這代表市場「真正發生」大跌或大漲時，模型有高達六成的機率能成功捕捉到。

## 1.4 最終選用決策：化缺點為防護罩 (The Final Verdict)

基於上述「低 Precision, 高 Recall」的特性，我們做出了極具戰略意義的架構轉換：

* **不作方向性預測器 (Not a Directional Signal)**：如果把這個模型當作主動買賣的訊號，過低的 Precision 會帶來大量的假訊號與交易手續費，保證虧損。
* **定位為「高敏感度撤單過濾器 (High-Sensitivity Cancel Filter)」**：
  * 對於 Maker-only (被動造市) 策略來說，我們不怕錯過獲利 (Opportunity Cost)，我們最怕的是「被趨勢碾壓而吃下的毒性訂單 (Adverse Selection / Toxic Fills)」。
  * **寧可錯殺，絕不放過**：由於模型的高 Recall，我們將其部署為防禦機制 (Risk Filter)。只要模型判斷有一絲危險 (Probability > Threshold)，我們就立刻撤單 (Pull Quote)。
  * 這完美發揮了該模型捕捉微觀結構破壞的能力，成為保護造市策略核心獲利的最強防盾。

**(下一章節，我們將介紹此 Cancel Filter 如何結合宏觀的 Regime Guard，形成完整的防護架構。)**
