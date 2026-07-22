from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "01_full_l2_action_universe_builder"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

SOURCE_QUOTE_ELIGIBLE = PROJECT_ROOT / "new_era" / "16_binary_label_v2" / "outputs" / "quote_eligible_events.parquet"
SOURCE_FILTER_REPORT = PROJECT_ROOT / "new_era" / "16_binary_label_v2" / "outputs" / "quote_eligible_filter_report.csv"
SOURCE_NEW4_DIR = PROJECT_ROOT / "data" / "NEW4"
OLD_RESIDUAL_UNIVERSE = PROJECT_ROOT / "new_era" / "28_residual_quote_ev_veto" / "data" / "baseline_universe_quote_ev.parquet"

FULL_DATASET_DIR = DATA_DIR / "full_l2_action_universe.parquet"
SPLIT_DATASET_TEMPLATE = "full_l2_action_universe_{split}.parquet"

HORIZONS = [1, 5, 10, 20, 50, 100]
POINT_VALUE = 200.0
MAKER_FEE = 15.0
SEVERE_NEGATIVE_PNL = -5000.0
BOUNDARY_FIRST_TICKS = 100
BOUNDARY_LAST_TICKS = 10
NEW4_SPLIT = "new4"
FEATURE_DRIFT_COLUMNS = [
    "spread_ticks",
    "queue_imbalance",
    "deep_imbalance",
    "event_rate_past_50",
    "volatility_past_50",
    "top_depth",
]

ACTION_SIDES = ["bid", "ask", "no_quote"]

IDENTITY_COLUMNS = ["split", "session_id", "symbol", "event_index", "action_side"]

BOOK_SOURCE_COLUMNS = [
    "dataset",
    "symbol",
    "session_id",
    "split",
    "local_row_id",
    "timestamp",
    "tick_size",
    "mid_price",
    "spread_ticks",
    "quote_eligible",
]

L2_SOURCE_COLUMNS = []
for level in range(1, 6):
    L2_SOURCE_COLUMNS.extend([f"bid_px_{level}", f"bid_vol_{level}", f"ask_px_{level}", f"ask_vol_{level}"])

OPTIONAL_SOURCE_COLUMNS = [
    "queue_imbalance",
    "deep_obi",
    "queue_imbalance_change",
    "normalized_spread",
    "trade_intensity_short",
    "cancel_pressure_short",
]

SOURCE_COLUMNS = BOOK_SOURCE_COLUMNS + L2_SOURCE_COLUMNS + OPTIONAL_SOURCE_COLUMNS

CAUSAL_FEATURE_COLUMNS = [
    "best_bid",
    "best_ask",
    "spread_ticks",
    "queue_imbalance",
    "deep_imbalance",
    "top_depth",
    "depth_ratio",
    "event_rate_past_50",
    "inter_arrival_ms",
    "volatility_past_50",
    "abs_return_past_50",
    "imbalance_change_past",
    "spread_change_past",
    "hawkes_excite",
    "normalized_spread",
]

for level in range(1, 6):
    CAUSAL_FEATURE_COLUMNS.extend([f"bid_px_{level}", f"bid_sz_{level}", f"ask_px_{level}", f"ask_sz_{level}"])

ACTION_COLUMNS = ["quote_price", "action_slot"]

FUTURE_LABEL_COLUMNS = []
for horizon in HORIZONS:
    FUTURE_LABEL_COLUMNS.extend(
        [
            f"future_mid_H{horizon}",
            f"markout_ticks_H{horizon}",
            f"pnl_H{horizon}",
            f"positive_pnl_H{horizon}",
            f"negative_pnl_H{horizon}",
            f"severe_negative_pnl_H{horizon}",
        ]
    )


def ensure_dirs() -> None:
    for path in [ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR, FULL_DATASET_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def split_dataset_dir(split: str) -> Path:
    return DATA_DIR / SPLIT_DATASET_TEMPLATE.format(split=split)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataset_paths() -> list[Path]:
    return sorted(FULL_DATASET_DIR.glob("split=*/session_id=*/*.parquet"))
