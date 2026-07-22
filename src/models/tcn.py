"""
硬體友善型 TCN 模型架構 (Hardware-Friendly TCN)
================================================
Week 3 核心交付物。

架構總覽：
  Input (B, T, F)              ← 3D 張量 from Week 1-2
      ↓ permute → (B, F, T)   ← PyTorch Conv1d 要求 channel-first
      ↓
  TemporalConvNet              ← 堆疊多層 TemporalBlock
      │  Block 0: dilation=1   ← 看 1 步以內
      │  Block 1: dilation=2   ← 看 3 步以內 (跳著看)
      │  Block 2: dilation=4   ← 看 7 步以內 (跳更遠)
      ↓
  Global Average Pooling       ← 把整個時間軸壓成 1 個向量
      ↓
  Fully Connected → Softmax    ← 3 分類 (Down / Stationary / Up)

設計原則：
  1. 因果性 (Causal): 只看過去，絕不偷看未來
  2. 膨脹 (Dilated): 用少量參數看到長距離歷史
  3. 殘差連接 (Residual): 穩定梯度，避免退化
  4. 極簡 (Minimal): 16 channels × 3 layers，方便後續 FPGA 部署
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Module note: this file contains two parallel model definitions:
# 1. the training-time TCN used in PyTorch experiments
# 2. the HLS-friendly TCN variant whose operator choices are easier to export
#    into fixed-shape C++ and Vitis HLS.


class CausalConv1d(nn.Module):
    """
    因果卷積 (Causal Convolution)
    
    普通的 Conv1d 會在輸入兩側做 padding，這會讓模型「偷看」未來的資料。
    因果卷積只在「左邊」(過去方向) 做 padding，確保時間點 T 的輸出
    只依賴 T 以及 T 之前的資料。
    
    Parameters
    ----------
    in_channels : int
        輸入通道數
    out_channels : int
        輸出通道數
    kernel_size : int
        卷積核大小
    dilation : int
        膨脹係數 (控制跳幾步看)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        # 計算需要在左邊 pad 多少：(kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0,  #  causal padding
            dilation=dilation,
        )
    
    def forward(self, x):
        """
        x: (Batch, Channels, Time)
        """
        # 只在時間軸的左邊 pad，右邊不 pad → 因果！
        x = F.pad(x, (self.padding, 0)) # (padding_left, padding_right)
        return self.conv(x)


class TemporalBlock(nn.Module):
    """
    時間卷積塊 (Temporal Block)
    
    一個 TemporalBlock 包含：
      CausalConv1d → ReLU → Dropout → CausalConv1d → ReLU → Dropout
      + 殘差連接 (Residual Connection)
    
    殘差連接的意義：
      如果這一層學不到什麼有用的東西，至少讓輸入直接「跳」到輸出，
      不會因為多加了一層而讓模型變差。這是深度學習中防止「退化問題」的經典技巧。
    
    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    dilation : int
    dropout : float
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        
        # 兩層因果卷積 (經典 TCN 設計)
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 殘差連接：如果 in_channels != out_channels，需要一個 1×1 卷積來對齊維度
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) 
            if in_channels != out_channels 
            else nn.Identity()
        )
    
    def forward(self, x):
        """
        x: (Batch, Channels, Time)
        return: (Batch, out_channels, Time)
        """
        # 主路徑
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 殘差連接：把輸入 x 加到輸出上
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    """
    時間卷積網路骨幹 (TCN Backbone)
    
    堆疊多個 TemporalBlock，每一層的 dilation 倍增：
      Layer 0: dilation = 1  (看 1 步)
      Layer 1: dilation = 2  (跳 1 步看)
      Layer 2: dilation = 4  (跳 3 步看)
      ...
    
    這樣用 3 層就能看到 2^3 × (kernel_size - 1) + 1 筆歷史資料，
    比 LSTM 更有效率！
    
    Parameters
    ----------
    num_inputs : int
        輸入特徵數 (= F)
    channel_list : list[int]
        每層的輸出通道數, e.g. [16, 16, 16]
    kernel_size : int
    dropout : float
    """
    
    def __init__(self, num_inputs, channel_list, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        num_levels = len(channel_list)
        
        for i in range(num_levels):
            dilation = 2 ** i  # 指數級膨脹: 1, 2, 4, 8, ...
            in_ch = num_inputs if i == 0 else channel_list[i - 1]
            out_ch = channel_list[i]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        x: (Batch, num_inputs, Time)
        return: (Batch, channel_list[-1], Time)
        """
        return self.network(x)


class TCNClassifier(nn.Module):
    """
    完整的 TCN 分類器
    
    將 TCN backbone 的輸出透過 Global Average Pooling 壓縮時間軸，
    再用一個全連接層做最終的 3 分類預測。
    
    為什麼用 Global Average Pooling 而不是只取最後一個時間點？
      1. 更穩健：平均能抵消雜訊
      2. 更適合 FPGA：平均計算比全連接層消耗更少資源
    
    Parameters
    ----------
    num_inputs : int
        輸入特徵數 (= 5)
    num_classes : int
        分類數 (= 3: Down, Stationary, Up)
    channel_list : list[int]
        TCN 每層通道數
    kernel_size : int
    dropout : float
    """
    
    def __init__(self, num_inputs, num_classes, channel_list, 
                 kernel_size=3, dropout=0.1):
        super().__init__()
        
        self.tcn = TemporalConvNet(num_inputs, channel_list, kernel_size, dropout)
        self.fc = nn.Linear(channel_list[-1], num_classes)
    
    def forward(self, x):
        """
        x: (Batch, Sequence, Features) — 來自 Week 1-2 的 3D 張量
        return: (Batch, num_classes) — 3 分類 logits
        """
        # Step 1: 轉換維度 (Batch, Time, Features) → (Batch, Features, Time)，因為 Conv1d 要求 channel-first
        x = x.permute(0, 2, 1)
        
        # Step 2: 通過 TCN backbone
        x = self.tcn(x)  # (Batch, Channels, Time)
        
        # Step 3: Global Average Pooling — 沿時間軸取平均
        x = x.mean(dim=2)  # (Batch, Channels)
        
        # Step 4: 全連接層 → logits
        x = self.fc(x)  # (Batch, num_classes)
        
        return x
    
    def predict_proba(self, x):
        """
        輸出 softmax 機率分布。
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def count_parameters(self):
        """
        計算模型的可訓練參數量。
        FPGA 部署時，參數量直接影響 BRAM 消耗。
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def receptive_field(self):
        """
        計算模型的感受野 (Receptive Field)。
        
        公式：RF = 1 + 2 * (kernel_size - 1) * sum(dilation_i for each layer)
        其中每層有兩個因果卷積，且 dilation_i = 2^i
        """
        num_levels = len(self.tcn.network)
        kernel_size = self.tcn.network[0].conv1.conv.kernel_size[0]
        
        # 每個 TemporalBlock 有 2 個因果卷積
        total_dilation = sum(2 ** i for i in range(num_levels))
        rf = 1 + 2 * (kernel_size - 1) * total_dilation
        return rf


# ---------------------------------------------------------------------------
#  HLS 友善版 (HLS-Friendly Variant)
#  - 不使用 F.pad：改用 torch.cat([zeros, x]) 做 causal padding
#  - 不使用 permute：輸入直接接受 channel-first (B, C, T)
#  - GAP 使用 nn.AdaptiveAvgPool1d(1) 而非 .mean(dim=2)
#  - 權重格式與原版完全相容，可直接遷移
# ---------------------------------------------------------------------------


class CausalConv1dHLS(nn.Module):
    """
    HLS 友善版因果卷積。

    與 CausalConv1d 數學行為完全相同，但避免使用 F.pad()。
    改為在前向傳播中手動 concat 一個全零 tensor 到時間軸左邊，
    這在 torch.jit.trace / ONNX export / hls4ml 中都更容易被正確解析。

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    dilation : int
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.causal_pad_len = (kernel_size - 1) * dilation
        self.pad = nn.ConstantPad1d((self.causal_pad_len, 0), 0.0)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x):
        """
        x: (Batch, Channels, Time)
        """
        if self.causal_pad_len > 0:
            x = self.pad(x)
        return self.conv(x)


class TemporalBlockHLS(nn.Module):
    """
    HLS 友善版時間卷積塊。

    結構與 TemporalBlock 完全相同（CausalConv → ReLU → Dropout × 2 + Residual），
    但使用 CausalConv1dHLS 取代 CausalConv1d。

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    dilation : int
    dropout : float
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()

        self.conv1 = CausalConv1dHLS(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1dHLS(out_channels, out_channels, kernel_size, dilation)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        """
        x: (Batch, Channels, Time)
        return: (Batch, out_channels, Time)
        """
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TemporalConvNetHLS(nn.Module):
    """
    HLS 友善版 TCN Backbone。

    與 TemporalConvNet 使用相同的 dilation schedule (1, 2, 4, ...)，
    但內部使用 TemporalBlockHLS。

    Parameters
    ----------
    num_inputs : int
    channel_list : list[int]
    kernel_size : int
    dropout : float
    """

    def __init__(self, num_inputs, channel_list, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        num_levels = len(channel_list)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else channel_list[i - 1]
            out_ch = channel_list[i]
            layers.append(
                TemporalBlockHLS(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (Batch, num_inputs, Time)
        return: (Batch, channel_list[-1], Time)
        """
        return self.network(x)


class HLSFriendlyTCNClassifier(nn.Module):
    """
    HLS 友善版完整 TCN 分類器。

    與 TCNClassifier 的差異：
      1. 不做 permute — 輸入必須是 channel-first (Batch, Features, Time)
      2. CausalConv 不用 F.pad — 改用 torch.cat
      3. GAP 使用 nn.AdaptiveAvgPool1d(1) — 而非 .mean(dim=2)

    這些改動不影響數學行為，權重可以直接從 TCNClassifier 遷移。

    Parameters
    ----------
    num_inputs : int
    num_classes : int
    channel_list : list[int]
    kernel_size : int
    dropout : float
    """

    def __init__(self, num_inputs, num_classes, channel_list, kernel_size=3, dropout=0.1, seq_length=100):
        super().__init__()

        self.tcn = TemporalConvNetHLS(num_inputs, channel_list, kernel_size, dropout)
        self.gap = nn.AvgPool1d(kernel_size=seq_length)
        self.fc = nn.Linear(channel_list[-1], num_classes)

    def forward(self, x):
        """
        x: (Batch, Features, Time) — 注意：channel-first，不做 permute
        return: (Batch, num_classes) — 3 分類 logits
        """
        # Step 1: 通過 HLS 友善版 TCN backbone
        x = self.tcn(x)  # (Batch, Channels, Time)

        # Step 2: Global Average Pooling — 使用 AvgPool1d(100)
        x = self.gap(x).squeeze(2)  # 移除最後的時間維度 -> (Batch, Channels)

        # Step 3: 全連接層 → logits
        x = self.fc(x)  # (Batch, num_classes)

        return x

    def predict_proba(self, x):
        """
        輸出 softmax 機率分布。
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def count_parameters(self):
        """
        計算模型的可訓練參數量。
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def receptive_field(self):
        """
        計算模型的感受野。
        """
        num_levels = len(self.tcn.network)
        kernel_size = self.tcn.network[0].conv1.conv.kernel_size[0]
        total_dilation = sum(2 ** i for i in range(num_levels))
        rf = 1 + 2 * (kernel_size - 1) * total_dilation
        return rf


def migrate_weights_to_hls(source_model, target_model):
    """
    將 TCNClassifier 的權重遷移到 HLSFriendlyTCNClassifier。

    兩者的 parameter name 和 shape 完全相同（因為 HLS 版只改了
    forward 中的運算方式，沒有改結構），所以可以直接 load_state_dict。

    Parameters
    ----------
    source_model : TCNClassifier
        已訓練的原版模型
    target_model : HLSFriendlyTCNClassifier
        要接收權重的 HLS 友善版模型

    Returns
    -------
    HLSFriendlyTCNClassifier
        載入權重後的模型
    """
    target_model.load_state_dict(source_model.state_dict())
    return target_model


# ---------------------------------------------------------------------------
#  Factory functions
# ---------------------------------------------------------------------------


def build_default_model(num_inputs=None):
    """
    使用 constants.py 中的預設超參數建立 TCN 模型（原版）。
    """
    from src.utils.constants import (
        NUM_INPUT_FEATURES, NUM_CLASSES, 
        TCN_CHANNELS, TCN_KERNEL_SIZE, TCN_DROPOUT
    )

    if num_inputs is None:
        num_inputs = NUM_INPUT_FEATURES
    
    model = TCNClassifier(
        num_inputs=num_inputs,
        num_classes=NUM_CLASSES,
        channel_list=TCN_CHANNELS,
        kernel_size=TCN_KERNEL_SIZE,
        dropout=TCN_DROPOUT,
    )
    return model


def build_default_model_hls(num_inputs=None):
    """
    使用 constants.py 中的預設超參數建立 HLS 友善版 TCN 模型。
    """
    from src.utils.constants import (
        NUM_INPUT_FEATURES, NUM_CLASSES,
        TCN_CHANNELS, TCN_KERNEL_SIZE, TCN_DROPOUT
    )

    if num_inputs is None:
        num_inputs = NUM_INPUT_FEATURES

    model = HLSFriendlyTCNClassifier(
        num_inputs=num_inputs,
        num_classes=NUM_CLASSES,
        channel_list=TCN_CHANNELS,
        kernel_size=TCN_KERNEL_SIZE,
        dropout=TCN_DROPOUT,
    )
    return model
