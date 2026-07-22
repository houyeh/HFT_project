# hls4ml 設計空間探索 (Design Space Exploration, DSE) 指南

恭喜你！根據你專案根目錄的 `guide.md`，我們目前已經正式完成了 **Week 10 (TCN 架構重構與訓練)**，並且正準備進入最能展現硬體工程師價值的階段：**Week 11 (設計空間探索 DSE)**。

在這個階段，你的目標不再是「把模型訓練得更準」，而是要在「準確率 (Accuracy)」、「推論延遲 (Latency)」與「硬體資源消耗 (DSP/BRAM/LUT)」這三個維度中，找到完美的平衡點，並畫出神級的 Trade-off (取捨) 圖表來展示給教授看！

以下是你在 `07_hls4ml_tcn_conversion.py` 中可以開始調整的三大核心參數：

---

## 1. 資源重用率 (Reuse Factor)
這是控制「速度 vs. 面積」的最重要參數。
在神經網路中，每一層都有大量的矩陣乘法。你要決定 FPGA 是「同時派出 100 個人來算 (超快但超耗資源)」，還是「派出 1 個人連續算 100 次 (省資源但超慢)」。

- **設定位置**：
  ```python
  # 在 07_hls4ml_tcn_conversion.py 中
  for layer in config['LayerName'].keys():
      config['LayerName'][layer]['ReuseFactor'] = 1
  ```
- **如何調整**：
  - `ReuseFactor = 1`：(我們目前的設定) 全面平行展開！這是追求極致低延遲 (Ultra-low latency) 的 HFT 首選，但可能會把 PYNQ-Z2 的 DSP 資源撐爆。
  - `ReuseFactor = 4` 或 `8` 或 `16`：神經元共用乘法器。如果你的模型太大塞不進晶片，把這個數字調高，硬體資源消耗會成比例下降，但延遲時間 (Latency) 會成比例上升。

## 2. 數值精確度 (Precision)
決定權重和特徵要用多少個 bits 存儲。
- **設定位置**：
  ```python
  config = hls4ml.utils.config_from_pytorch_model(
      ..., 
      default_precision='ap_fixed<16,6>'
  )
  ```
- **如何調整**：
  - `ap_fixed<16,6>`：(16-bit 總寬，其中 6-bit 是整數)。這是 HFT 業界標準的安全牌，準確率幾乎等同 Python 原生模型。
  - `ap_fixed<12,4>`：(12-bit 總寬)。稍微縮減，省資源但幾乎不掉準確度。
  - `ap_fixed<8,2>`：(傳說中的 INT8 量化)。極致省記憶體 (BRAM)，但整數部分只有 2-bit，容易溢位 (Overflow)。這也就是為什麼我們 Week 11 規劃提到「可能需要 INT8 Trade-off 圖表」。

## 3. 合成策略 (Strategy)
告訴 Vivado 編譯器你的最終目標是什麼。
- **設定位置**：
  ```python
  config = hls4ml.utils.config_from_pytorch_model(
      ...,
      Strategy='Latency'  # 或是 'Resource'
  )
  ```
- **如何調整**：
  - `Latency`：讓編譯器無所不用其極地把迴圈拆開、減少時鐘週期。適合高頻交易。
  - `Resource`：讓編譯器盡量折疊迴圈、重用記憶體與乘法器。

---

## 4. 序列長度 (Sequence Length) - TCN 的面積殺手
對於影像辨識 (CNN)，面積取決於解析度；但對於高頻交易 (TCN)，面積取決於**「你看了多久的歷史資料」**。
在 Dataflow 架構下，各層之間是用 FIFO (緩衝區) 連接的。如果輸入長度是 100，硬體就會瘋狂合成深度為 100 的 FIFO，這就是為什麼我們的 LUT 會暴增到 400%！
- **如何調整**：
  退回 Python 端，將資料預處理的 `seq_length` 從 100 降為 30。
- **Trade-off**：硬體的 BRAM 與 LUT 消耗量將呈線性下降 (約剩原本的 1/3)，但代價是模型看到的歷史變短了，軟體回測的 PnL 可能會下降。這是非常值得放進報告的 Trade-off！

## 5. 真正的 INT8 量化 (Quantization-Aware Training & BatchNorm)
你不能只在 HLS 腳本裡把精度改成 `ap_fixed<8,2>`，因為神經網路的累加值會輕易超過 2，導致「溢位 (Overflow)」，算出完全錯誤的答案 (Test failed)。
要實現真正的 INT8 加速，你必須：
1. **加入 BatchNorm1d**：在 PyTorch 模型的每一個 `Conv1d` 後面加上 Batch Normalization，把數值強行壓平到 `[-1, 1]` 之間。
2. **導入 Brevitas 套件**：在訓練階段就讓模型習慣 8-bit 的雜訊 (這稱為 QAT)。

---

## 你的實戰作業 (Week 11 DSE 任務)
為了產出你研究所或競賽所需的「Killer Charts」，請依照以下順序 (Ablation Study) 進行：

### 階段一：壓制 LUT 與 BRAM (軟硬雙修)
1. **序列截斷版**：退回 Python，改用 30 Ticks 訓練，驗證 PnL 是否存活，並轉出硬體觀察 LUT 是否降至 100% 以下。
2. **HLS 資源優化版**：在 30 Ticks 基礎上，開啟 `Strategy='Resource'` 與 `ReuseFactor=64`，觀察 DSP 是否進一步壓縮。

### 階段二：挑戰極致面積 (大魔王)
1. **INT8 QAT 版**：引入 Brevitas 與 BatchNorm，重新訓練後設定 `Precision=<8,2>`。

將以上專案在 Vivado 裡跑完 C Synthesis (參考 004 指南)，然後紀錄它們的：
- **Latency (clock cycles)**
- **DSP48E (%)**
- **BRAM_18K (%)**
- **LUT (%)**
- 並搭配我們之前 `06_evaluate_tcn.py` 算出的 Accuracy，畫成折線圖。這張圖將會是你整個專案最值錢的產出！
