📊 一、資料與市場微結構
Q1. LOB 快照 vs 逐筆成交的對齊問題 你的資料有 quote（委託簿）與 trade（成交）兩種流。它們的時間戳記頻率不同。你是用什麼原則對齊它們的？如果一個 trade tick 在兩個 quote 快照之間到達，你的特徵引用的是哪一個 quote 的狀態？這個選擇對 cancel_pressure、queue_imbalance 這類特徵的數值準確性有什麼影響？

Q2. tick_size 標準化的必要性 你說用 tick_size 將特徵標準化讓不同商品可比較。請解釋：為什麼直接用原始價格差（raw spread in dollars）在跨商品訓練時會出問題？tick_size normalization 本質上是在做什麼假設？這個假設在 CL（原油）vs ES（S&P 500 期貨）之間成立嗎？

Q3. 時間切分的 look-ahead bias 你用時間序列的 train/val/test 切分（70/15/15 或類似）。請問：sliding_window.py 在建立每一個 (X, y) 樣本時，y 的 horizon 是往未來幾個 tick？在切分邊界附近，val set 的第一個樣本的 y 標籤是否可能涉及到在 train set 時間範圍內才有的未來資訊？你是如何處理這個邊界問題的？

Q4. 市場微結構假設 你的整個模型建立在「LOB 的狀態可以預測未來短期價格方向」這個假設上。這個假設的理論基礎是什麼（請用 Kyle's lambda、Glosten-Milgrom、Adverse Selection 等概念回答）？在什麼市場條件下，這個假設會完全失效？

🧠 二、特徵工程
Q5. OBI（Order Book Imbalance）的數學定義 你的特徵中有 obi、deep_obi、near_depth_obi。請寫出 OBI 的精確數學公式。deep_obi 跟 obi 的差別是什麼？它們在預測能力上各自捕捉了什麼不同的市場資訊？

Q6. cancel_pressure 的構造 cancel_pressure_short 這個特徵是怎麼計算的？它的分子跟分母分別是什麼？為什麼「撤單壓力」在 HFT 場景下是有意義的預測訊號——背後的市場行為邏輯是什麼？

Q7. rolling_signed_flow_ratio 的洩漏風險 rolling_signed_flow_ratio 使用了一個 rolling window 的累積成交量。這個 rolling 的計算是否在每個時間點只使用「到當下為止」的資料？你用的是 .rolling() + .shift(1) 還是直接 .rolling()？如果後者，第一個 window 的第一筆資料是否有 lookahead？

Q8. 特徵的 stationarity 統計學上，時間序列模型需要輸入特徵是平穩的（stationary）。你的特徵如 micro_return、weighted_mid_deviation 等是否是平穩的？如果一段時間的市場進入趨勢行情，這些特徵的分佈是否會大幅漂移？你有沒有在 train/test 之間做過特徵分佈的 drift analysis？

🏗️ 三、模型架構
Q9. TCN 的感受野計算 你的 TCN 設定是：kernel_size=3、dilation=[1, 2, 4]（推測）、sequence_length=100。請計算這個 TCN 的實際感受野（receptive field）是多少個 time step？這個感受野能覆蓋你 sequence_length=100 的輸入嗎？如果感受野不夠，模型在序列末端的預測會有什麼問題？

Q10. Residual Connection 的必要性 TCN block 中的 residual connection（跳接）做的是什麼？在你的設定（input channels 12 → hidden channels 16）下，input 跟 output 的 channel 數不一樣，這個 residual connection 是如何處理維度不匹配的問題的？你的程式碼裡有沒有 1×1 convolution 做維度對齊？

Q11. Causal Padding 的正確性 TCN 必須使用 causal padding 才能確保不洩漏未來資訊（每個時間步的輸出只能看到它自己和過去的輸入）。請解釋你的 causal padding 實作：對於 kernel_size=3、dilation=2 的 conv1d，padding 應該是多少？你是 pad 在左邊還是兩邊？如果 pad 在兩邊，就已經洩漏未來了。

Q12. 三分類的輸出層設計 模型輸出是 Down/Stationary/Up 三類。你用的是 softmax 輸出。請解釋：交叉熵損失函數（Cross-Entropy Loss）的數學形式是什麼？它跟 KL-divergence 的關係是什麼？在類別不平衡時，Cross-Entropy 的梯度會如何被多數類別（Stationary）所主導？

⚖️ 四、訓練與評估
Q13. Class Weighting 的正確計算方式 你用 class weighting 來處理類別不平衡。請問這個 weight 是怎麼算的？是用 1/frequency、total_samples / (n_classes * class_count) 還是別的方式？這些不同的計算方式在梯度更新上有什麼具體差異？

Q14. Accuracy vs F1 在不平衡資料上的問題 你同時報告 Accuracy 和 Macro-F1。在 Stationary 類別佔 70% 的情況下，一個「永遠預測 Stationary」的 dummy classifier 的 Accuracy 會是多少？它的 Macro-F1 會是多少？為什麼在這個任務中 Macro-F1 比 Accuracy 更有意義？

Q15. Early Stopping 的 patience 設計 你的 patience=2。這意味著只要連續 2 個 epoch val loss 沒有改善就停止。在你的資料量與 batch_size=256 下，一個 epoch 包含多少個 gradient updates？patience=2 是太保守（模型訓練不足）還是太激進（可能在 val loss 短暫震盪時就停了）？你怎麼判斷？

Q16. 回測的 Sharpe Ratio 計算方式 你說 "Sharpe-like"。請問分母用的是什麼：所有交易 PnL 的標準差、每日 PnL 的標準差、還是固定時間窗的 PnL 標準差？你有沒有年化（annualize）？在 HFT 場景下，每秒可能有數十個訊號，你的 Sharpe 計算周期是什麼？如果不一致，跨商品的 Sharpe 比較是否有意義？

🚦 五、Gating 與回測邏輯
Q17. Probability Gating 的信心閾值意義 你的 gating 用 prob_threshold 過濾低信心預測。請問：softmax 輸出的機率值是否已校準（calibrated）？也就是說，softmax 輸出 0.8 的機率是否真的意味著這個預測有 80% 的歷史勝率？如果沒有校準，設定這個門檻的理論依據是什麼？

Q18. Signal Backtest 的向量化假設 Signal Backtest 假設每次訊號都可以以「當下 mid-price」成交。這個假設在市場微結構上意味著什麼？你忽略了哪些真實交易成本（bid-ask spread、slippage、queue position、market impact）？這些忽略會讓你的 backtest PnL 比真實好多少？

Q19. Event-Driven Backtest 的持有時間假設 Event-Driven 模擬中，你設定了最長持有時間（max holding time）。請解釋：如果一個 Up 訊號進場後，價格並沒有上漲，你在 max holding time 到期後是以什麼價格出場的？是 mid-price、best bid 還是市價？這個出場假設對 PnL 的影響大嗎？

Q20. Cooldown 的設計動機 你在 event-driven backtest 中有 cooldown 機制。這是為了防止什麼？它是在模擬真實的什麼限制（是技術上的限制、是策略上的限制、還是監管要求）？cooldown 的長度是怎麼決定的——是 grid search 得來的，還是基於某個市場微結構原則？

🔢 六、量化（Quantization）
Q21. FP32 → INT8 的量化誤差來源 你將模型從 FP32 量化到 INT8。INT8 有 256 個離散值，FP32 有 ~2^32 個。量化誤差（quantization error）的大小取決於什麼？你用的是 symmetric 還是 asymmetric 量化？zero-point 是怎麼設定的？

Q22. Per-layer vs Per-tensor 量化 量化可以按整個 tensor 用一組 scale（per-tensor），也可以按每個 channel 用不同的 scale（per-channel）。你選的是哪一種？在 TCN 的 depthwise conv 層，per-channel 量化通常表現更好——你有比較過嗎？兩者的精度差距有多大？

Q23. Bit-Accurate Verification 的意義 你建立了 Python → C++ → HLS 的 golden test vector 驗證鏈。請解釋：「bit-accurate」的意思是什麼？為什麼在 INT8 定點數下，C++ 的結果必須跟 Python 完全一致到每一個 bit？如果不一致，代表什麼？哪些操作最容易造成 C++ 和 Python 之間的位元差異？

⚙️ 七、HLS 與 FPGA 硬體
Q24. HLS 的 pipeline pragma 原理 在 Vitis HLS 中，#pragma HLS PIPELINE II=1 的意思是什麼？Initiation Interval（II）=1 意味著每個 clock cycle 可以開始一個新的 inference。對你的 TCN kernel 而言，達到 II=1 需要什麼條件？如果資料依賴（data dependency）存在於 loop 的不同 iteration 之間，II 會是多少？

Q25. FPGA 上的 latency 估算方法 你估算 FPGA latency 約 9.2 μs。請說明這個數字怎麼算出來的：是 MAC 數量 / 操作頻率，還是 HLS synthesis 報告的 clock cycle 數？這個估算有沒有包含 I/O 搬移時間（DMA transfer latency）？如果沒有，板上量測的實際 latency 可能會差多少？

Q26. hls4ml 轉換失敗的根本原因 hls4ml 在轉換時卡在 permute 和 pad 操作。請解釋：為什麼這兩個操作對 hls4ml 是挑戰？permute（維度轉置）在硬體上意味著什麼樣的資料搬移？你說「問題不在模型大小，而在 graph 的工具可轉換性」——這句話的更深一層的含義是什麼？是不是說 hls4ml 只支援特定的 computation graph 拓樸？

Q27. 手刻 HLS C++ vs hls4ml 的取捨 你選擇放棄 hls4ml，改用手刻 fixed-shape C++/HLS。這個決策的代價是什麼？手刻的好處（更細緻的 pragma 控制）和壞處（維護成本、與 Python model 的同步風險）分別是什麼？如果未來模型架構改變（例如 channel 數從 16 改成 32），你的手刻 HLS code 需要改哪些地方？

🌐 八、跨市場泛化
Q28. CME 遷移的 domain shift 問題 你的模型先在台指（TW）資料訓練，再嘗試遷移到 CME（CL、ES）。這兩個市場的 tick_size、流動性、平均 bid-ask spread、市場參與者組成都不同。這種跨市場的 domain shift 在機器學習上是什麼問題？你是用 fine-tuning 還是 zero-shot 評估？如果是 zero-shot，你預期哪些特徵在 CME 上仍然有效，哪些會失效？

Q29. 幣安虛擬倉回測的特殊風險 你手癢做了幣安（Binance）的模型並在虛擬倉回測。加密貨幣市場的 LOB 微結構跟傳統期貨市場有哪些根本差異（例如：沒有 uptick rule、24/7、無漲跌幅限制、wash trading 的存在）？這些差異是否意味著你為台指或 CME 設計的特徵，在幣安數據上使用時已經違反了某些底層假設？

🔬 九、整體系統設計
Q30. 整個 pipeline 的 overfitting 來源清單 在你的整個系統中（從資料 → 特徵 → 訓練 → gating → calibration → backtest），請列出至少 5 個可能造成 overfitting 或 backtest 過度樂觀（overly optimistic） 的地方，並解釋每一個的機制。特別是：你的 calibration 是在 val set 上搜尋參數，然後在 test set 上報告結果——這在統計上是合法的，但它的前提假設是什麼？如果 val 和 test 的市場 regime 不同，這個 calibration 有用嗎？

教授的最後叮嚀：這 30 個問題不是要你背答案，而是要你能夠從第一原理（first principles）重建答案。如果你能流利地回答其中 20 個，你對這個專案的掌握程度就已經遠超過大多數做出「跑得動的 code」的人。剩下那 10 個，就是你目前的知識邊界——而知道自己的知識邊界，才是真正做研究的開始。
