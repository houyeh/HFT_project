# Model Selection Process

## Phase 1: Label Type Sweep (8 label types x fc in {6,7,8,9}, H=10, thr=1)
- 8 label types tested: smoothed_mid_move_3class, mid_move_3class, adverse_bid_fill_binary, adverse_ask_fill_binary, toxic_bid_binary, toxic_ask_binary, active_sell_profitable_binary, active_buy_profitable_binary
- Only `smoothed_mid_move_3class` and `mid_move_3class` showed positive mean pnl_improvement.
- Selected: `smoothed_mid_move_3class` (mean pnl_improvement = 17,979 TWD).
- Rejected all binary labels: adverse/toxic/active did not improve PnL over baseline.

## Phase 2: Feature Count and Horizon Selection
- Confirmed fc=8, window=100.
- H=10 thr=1 gave highest raw pnl_improvement (+79,635) under original backtest framework.
- Switched to H=5 thr=0.5 for cancel-filter diagnostic framework (simulate_with_metrics).

## Phase 3: expA/B/C Snapshot Iteration (H=5, thr=0.5, fc=8)
- expB_w10 had lowest val_loss (0.511) but zscore_window=10 is too short for OOS stability.
- expC_snapshot selected: val_loss=0.781 with zscore_window=2000 (stable, consistent).
- expC is the base model for all downstream diagnostics.

**Data sources:**
- `experiment_summary.csv`: C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\15_models_playaround\015_models_playaround_result\experiment_summary.csv
- `backtest_summary_sorted.csv`: C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\15_models_playaround\015_models_playaround_result\backtest_summary_sorted.csv
