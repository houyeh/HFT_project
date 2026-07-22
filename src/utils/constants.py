"""
常數定義模組 (Constants)
存放專案中所有需要統一的常數，如 Tick Size、Tick Value 等。
"""

# ===== 各商品的 Tick Size (最小跳動點) =====
# Module note: these values are the canonical assumptions shared by the
# training, backtest, and hardware-export paths.
TICK_SIZES = {
    "ES": 0.25,   # E-mini S&P 500
    "NQ": 0.25,   # E-mini NASDAQ 100
    "CL": 0.01,   # Crude Oil (WTI)
}

# ===== 各商品的 Tick Value (每跳動一點的美元收益) =====
TICK_VALUES = {
    "ES": 12.50,  # 1 Tick = $12.50
    "NQ": 5.00,   # 1 Tick = $5.00
    "CL": 10.00,  # 1 Tick = $10.00
}

# ===== 特徵工程相關預設參數 =====
DEFAULT_WINDOW_SIZE = 100       # 滑動視窗長度 (Ticks)
DEFAULT_HORIZON = 10            # 預測距離 N-ticks ahead
DEFAULT_THRESHOLD = 0.5         # 漲跌分類的 Tick 閾值

# ===== LOB 深度 =====
LOB_DEPTH = 5  # 我們使用 5 檔深度

# ===== TCN 模型超參數 (Week 3) =====
NUM_INPUT_FEATURES = 5          # 輸入特徵數 (normalized_spread, micro_return, obi, weighted_mid, deep_obi)
NUM_CLASSES = 3                 # 分類數 (0=Down, 1=Stationary, 2=Up)
TCN_CHANNELS = [16, 16, 16]    # 每層的輸出通道數 (FPGA 友善: 保持小且一致)
TCN_KERNEL_SIZE = 3             # 卷積核大小
TCN_DROPOUT = 0.1               # Dropout 機率
