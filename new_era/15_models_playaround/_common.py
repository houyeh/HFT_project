from __future__ import annotations

"""
共用設定（建議只改這裡）
====================

你如果要做 ablation / label 切換，最常改的地方都集中在這支檔案：

- label_type（方向三分類 / smoothed / toxicity / fill-conditional / active）
- feature_count（6/7/8/9）
- window_size / horizon / threshold_ticks
- adverse_threshold_ticks（toxic/adverse_fill 用）
- 訓練超參數（epochs/batch_size/lr）
- 回測費用與庫存上限（maker fee / taker fee / max inventory）

跑出來的成果會寫到：
`new_era/15_models_playaround/015_models_playaround_result/<experiment_id>/...`
"""

import sys
from pathlib import Path

def _ensure_project_root_on_path() -> Path:
    # new_era/15_models_playaround/_common.py -> parents[2] = project root
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


# 讓你不管從哪個目錄執行（包含 `new_era/15_models_playaround/`）都能正常 import `new_era.*`
_ensure_project_root_on_path()


from new_era.models_playaround_15.config import (  # noqa: E402
    BacktestConfig,
    DataConfig,
    ExperimentConfig,
    FeatureSet,
    LabelConfig,
    ModelConfig,
    SplitConfig,
    TrainingConfig,
    WindowConfig,
)
from new_era.models_playaround_15.paths import build_experiment_paths, experiment_id  # noqa: E402


# =============================================================================
# 常用實驗參數（最常改）
# =============================================================================

# 你要切換 label 就改這裡：
# LABEL_TYPE = "mid_move_3class"
LABEL_TYPE = "smoothed_mid_move_3class"
# LABEL_TYPE = "toxic_bid_binary"
# LABEL_TYPE = "toxic_ask_binary"
# LABEL_TYPE = "adverse_bid_fill_binary"
# LABEL_TYPE = "adverse_ask_fill_binary"
# LABEL_TYPE = "active_buy_profitable_binary"
# LABEL_TYPE = "active_sell_profitable_binary"

# 你要做 feature ablation 就改這裡（6/7/8/9）：
FEATURE_COUNT = 8

# 你要改 label horizon / threshold 就改這裡：
WINDOW_SIZE = 100
HORIZON = 5
THRESHOLD_TICKS = 0.5

# toxic / adverse_fill 會用到
ADVERSE_THRESHOLD_TICKS = 1.0

# active label 會用到（這裡是成本假設）
TAKER_FEE_TWD = 50.0
POINT_VALUE_TWD_PER_POINT = 200.0
EXIT_COST_TICKS = 0.0
MARGIN_TICKS = 0.0

# Dataset / z-score
DATA_SYMBOLS = ("TXFB6",)
ZSCORE_WINDOW = 2000
SAMPLE_STRIDE_TRAIN = 5

# Train
EPOCHS = 30
BATCH_SIZE = 1024
LR = 1e-3
SEED = 42
DEVICE = "auto"  # "cuda" / "cpu" / "auto"

# Backtest（maker-only）
BT_SYMBOL = "TXFB6"
MAKER_FEE_TWD = 15.0
BT_TAKER_FEE_TWD = 50.0
MAX_INVENTORY = 3
SAMPLE_STRIDE_BACKTEST = 1


def build_cfg() -> ExperimentConfig:
    project_root = _ensure_project_root_on_path()

    cfg = ExperimentConfig(
        data=DataConfig(
            data_dir=project_root / "data" / "NEW_ALL",
            symbols=DATA_SYMBOLS,
            zscore_window=ZSCORE_WINDOW,
        ),
        window=WindowConfig(
            window_size=WINDOW_SIZE,
            horizon=HORIZON,
            threshold_ticks=THRESHOLD_TICKS,
            sample_stride_train=SAMPLE_STRIDE_TRAIN,
        ),
        split=SplitConfig(train_session_frac=0.70, val_session_frac=0.15),
        features=FeatureSet(feature_count=int(FEATURE_COUNT)),
        labels=LabelConfig(
            label_type=LABEL_TYPE,  # 你要切 label 改這裡（或改上面的 LABEL_TYPE 常數）
            adverse_threshold_ticks=float(ADVERSE_THRESHOLD_TICKS),
            taker_fee_twd=float(TAKER_FEE_TWD),
            point_value_twd_per_point=float(POINT_VALUE_TWD_PER_POINT),
            exit_cost_ticks=float(EXIT_COST_TICKS),
            margin_ticks=float(MARGIN_TICKS),
        ),
        model=ModelConfig(channel_list=(16, 16, 16), kernel_size=3, dropout=0.1),
        train=TrainingConfig(
            epochs=int(EPOCHS),
            batch_size=int(BATCH_SIZE),
            lr=float(LR),
            seed=int(SEED),
            device=str(DEVICE),
        ),
        backtest=BacktestConfig(
            symbol=str(BT_SYMBOL),
            point_value_twd_per_point=float(POINT_VALUE_TWD_PER_POINT),
            maker_fee_twd=float(MAKER_FEE_TWD),
            taker_fee_twd=float(BT_TAKER_FEE_TWD),
            max_inventory=int(MAX_INVENTORY),
            sample_stride=int(SAMPLE_STRIDE_BACKTEST),
            allow_active_alpha=False,
        ),
    )
    return cfg


def build_paths(cfg: ExperimentConfig):
    paths = build_experiment_paths(cfg)
    paths.ensure_dirs()
    return paths


def print_experiment_banner(cfg: ExperimentConfig) -> None:
    print("=" * 80)
    print("015 models playaround")
    print("- experiment_id:", experiment_id(cfg))
    print("- label_type:", cfg.labels.label_type)
    print("- feature_count:", cfg.features.feature_count)
    print("- window/horizon:", cfg.window.window_size, cfg.window.horizon)
    print("- threshold_ticks:", cfg.window.threshold_ticks)
    if cfg.labels.label_type not in ("mid_move_3class", "smoothed_mid_move_3class"):
        print("- adverse_threshold_ticks:", cfg.labels.adverse_threshold_ticks)
    print("=" * 80)
