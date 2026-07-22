# TCN 模型 Review（繁中版）與報告講稿（2026-05-07）

以下內容是把 `MODEL_filled_review` 的重點改寫成「能直接報告」的列點與講稿。建議用投影片每頁 1 個 section，講稿照著念即可，教授要追問時也有數字與程式位置能回指。

---

## Slide 1：研究目標（我現在到底在做什麼）

**列點**
- 研究目標：用事件級(逐筆)委託簿資料做短期方向三分類（Down / Stationary / Up），並以可部署到 HLS/FPGA 的低延遲模型為約束。
- 現階段定位：先把資料管線、label、特徵工程、模型訓練與評估做成可重現；再把訊號導入「被動造市」回測驗證是否有 edge。

**講稿**
> 我這個專題分成兩條線同步推進：第一條是把事件級委託簿的方向預測模型做到可重現、可解釋；第二條是把訊號接到造市策略做含成本回測。也就是說，我現在不是只追求分類準確率，而是把「資料正確性、延遲與可部署性」當成同等重要的約束，最後用回測結果驗證訊號是否真的能變成 PnL。

---

## Slide 2：資料來源與切割（避免 time leakage）

**列點**
- 原始資料：`data/NEW`，標的包含 `TXFB6` 與 `MXFB6`，事件級委託簿快照與成交欄位。
- 切割方式：以 session 為單位切 train/val/test，並對每個商品依 `session_id` 排序後做切分（避免相鄰 tick 隨機打散造成洩漏）。
- 本次 split（以 `002_zscore_dataset.npz` 匯總）：
- Train：886,133 samples（8 sessions：20260123Day/Night、20260126Day/Night × 2 symbols）
- Val：213,204 samples（2 sessions：20260127Day × 2 symbols）
- Test：389,652 samples（2 sessions：20260127Night × 2 symbols）

**講稿**
> 我特別強調切割方式：不是隨機 shuffle 的 sample-level split，而是 session-level split，並且每個商品都依 session 的時間順序切開。原因是事件級資料相鄰樣本高度重疊，如果隨機打散，模型很容易拿到未來的資訊而得到假高分。這裡 train/val/test 的 session 名單跟 sample 數量都有在輸出與資料檔裡可以對照。

---

## Slide 3：Label 定義（監督學習的「正確答案」）

**列點**
- window_size=100、horizon=10、threshold_ticks=1.0。
- 使用端點(end-point) label：比較 decision time `t` 的 mid-price(以 tick 單位) 與 `t+horizon` 的 mid-price。
- 若 `mid[t+h]-mid[t] > +1 tick` ⇒ Up(2)
- 若 `mid[t+h]-mid[t] < -1 tick` ⇒ Down(0)
- 否則 ⇒ Stationary(1)

**講稿**
> 我目前的 label 是最乾淨的端點差分定義：看 10 個事件後 mid-price 是否超過 ±1 tick。這個設計的好處是定義清楚、容易重現；缺點是它不一定等價於「在這 10 個事件內曾經觸發過一次可交易的波動」。所以我後面計畫會把 label 變體（例如 horizon 內 max/min 或 dynamic threshold）加入 ablation，並用回測的 PnL 來判斷哪種 label 更貼近交易。

---

## Slide 4：特徵工程（8 維 + rolling Z-score）

**列點**
- 特徵產生：`build_tw_feature_frame` 從 LOB + trades 產出微結構特徵。
- 最終輸入採 Top 8 features，並做 rolling z-score（per-session，window=2000，min_periods=100，clip [-10,10]，NaN→0）。
- Top 8（順序即 `TOP_8_FEATURES`）：
- `obi`
- `queue_imbalance`
- `weighted_mid_deviation`
- `trade_intensity_short`
- `deep_obi`
- `queue_imbalance_change`
- `normalized_spread`
- `cancel_pressure_short`

**講稿**
> 我把 feature 維度固定在 8 維，是為了兩個目的：第一是可解釋性，第二是可部署性。這 8 個特徵都能對應到微結構直覺：例如 OBI/queue_imbalance 反映買賣盤壓力，spread 反映成交摩擦，trade_intensity/cancel_pressure 反映市場毒性或流動性變化。正規化採用每個 session 自己的 rolling z-score，避免跨天 regime shift 造成尺度漂移；同時 rolling 本身是因果的，只用過去資料計算。

---

## Slide 5：模型架構（HLS-Friendly TCN）

**列點**
- 模型：`HLSFriendlyTCNClassifier`（3 個 TemporalBlock，channel_list=[16,16,16]，kernel_size=3，dropout=0.1）。
- Activation：ReLU；Normalization：無（未使用 BN/LN）。
- 參數量：4,515。
- Dilation：1, 2, 4。
- 感受野(Receptive Field)：
- 依程式公式 `RF = 1 + 2*(k-1)*sum(dilation)`，k=3、sum=7 ⇒ RF=29 ticks。
- 輸入張量：N×8×100（channel-first for Conv1d）。

**講稿**
> 我選 TCN 是因為它比 RNN/Transformer 更適合低延遲與硬體化：計算可平行、算子簡單。這個版本特別採用 HLS-friendly 寫法，避免一些不利於匯出的操作。模型非常小，只有 4,515 個參數；這有利於 FPGA，但也意味著能力可能不足，所以我後面會用「策略回測改善幅度」來判斷是否需要加大模型。需要注意的是，雖然輸入序列長度是 100，但單一位置的有效感受野是 29；最後透過 global average pooling 把整段 100 內的局部模式做彙總。

---

## Slide 6：訓練設定（本次實驗）

**列點**
- device：CPU（本次輸出）。
- epochs=20、batch_size=1024。
- optimizer：Adam(lr=1e-3)。
- loss：加權 CrossEntropyLoss（class weights 由 y_train 反比計算）。
- 本次 class weights：`[67.3763, 0.3366, 69.9119]`（Down / Stationary / Up）。
- 未使用：LR scheduler、weight decay、gradient clipping、early stopping、checkpoint/resume、random seed 固定（torch/numpy）。

**講稿**
> 這版訓練設定是刻意保守且易重現的 baseline：Adam 1e-3、batch 1024、20 epochs。因為資料極度不平衡，我有用 class weight 去把 Down/Up 的 loss 權重拉大。另一方面，我也承認目前還缺了完整的實驗工程化：例如把每個 epoch 的 loss/acc 曲線落盤、加入 checkpoint/resume、固定 seed、以及 scheduler。這些會是下一步優先補齊，因為教授會希望看到「收斂證據」而不是只看單次輸出。

---

## Slide 7：分類結果（為什麼 80% accuracy 不能代表可交易）

**列點**
- Val：Loss=0.701812、Acc=81.21%。
- Test：Loss=0.713121、Acc=79.86%。
- Val confusion matrix（Actual×Pred [Down,Stat,Up]）：
- `[[306, 886, 86], [12273, 172368, 25873], [39, 896, 477]]`
- Test confusion matrix：
- `[[326, 783, 63], [23724, 310235, 53253], [19, 646, 603]]`
- 重大觀察：Up/Down precision 只有 ~1–2%（Stationary 約 99% 主導）。

**講稿**
> 這裡是整個專題最關鍵的診斷點：整體 accuracy 接近 80%，但原因不是模型很會抓方向，而是因為資料中 Stationary 佔了約 99%。當我真的把 Up/Down 當成交易訊號使用時，會因為 precision 太低而大量誤觸發。也因此，我後面策略端會先把訊號用在「被動造市的偏置與防禦」而不是「頻繁主動吃單」，同時評估指標也要從 Accuracy 轉向 PR-AUC、macro-F1、校準等更貼近可交易訊號的指標。

---

## Slide 8：造市回測（目前可工作的方向：maker-only skew）

**列點**
- 已有 session-aware 回測雛形（含成本）：maker fee=15 TWD、taker fee=50 TWD、point value=200 TWD/點。
- `TXFB6 20260127Night`：
- baseline passive maker：PnL=97,800（454 trades，maker-only）
- `tcn_skew`（prob>=0.70）PnL=145,600（+47,800）
- 結論：訊號比較適合做「偏置報價/撤單防禦」；若用來頻繁做 taker alpha，會被低 precision 與成本放大成虧損。

**講稿**
> 我把模型訊號先接到「被動造市」的策略上，避免把低 precision 的 Up/Down 直接當成吃單訊號。初步結果顯示：用 TCN 只做 maker skew（調整掛單偏向）可以提升 PnL。相反地，若做主動攻擊(taker)會因為方向 precision 太低而虧損。這也呼應前面分類評估的觀察：要讓主動攻擊變成最賺的，前提是 Up/Down precision 必須顯著提高，或者我們必須用更嚴格的門檻讓出手機率下降。

---

## Slide 9：下一步規劃（請教授給建議的地方）

**列點**
- 評估指標工程化：
- 06 輸出 softmax 機率，計算 Up/Down PR-AUC、做 calibration/reliability。
- 訓練工程化：
- 05 落盤 training_history.csv（train/val loss/acc），加 checkpoint/resume，固定 seed。
- 資料/label 實驗：
- window_size(50/100/200)、horizon(5/10/20)、Top4/8/12 features、label(端點 vs max/min) 做 ablation。
- 回測方法學：
- 做 walk-forward（按 session 日期 rolling）並只評估 maker-only skew/gated 的策略，避免「看過 test」的調參。
- 部署數字：
- PyTorch CPU batch=1 latency/throughput + hls4ml latency/resource（同一組權重綁定）。

**講稿**
> 我希望教授幫我確認「哪一條線最該先做」：我目前的排序是先把評估指標跟回測一致化，因為它直接決定我該把模型改善目標放在 precision、校準，還是模型容量。第二步是把訓練/評估工程化補齊，讓每次實驗都可重現。第三步才是大量做 ablation 與擴資料。最後再把延遲與資源數字補上，才能把 FPGA 的價值主張講完整。

