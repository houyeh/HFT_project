from __future__ import annotations

"""
04 - Backtest（第四步）
=====================

正確流程（研究紀律）：
1) 在 validation sessions 上 sweep prob_threshold / edge_threshold
2) 選出 val PnL after fees 最佳的組合
3) 固定該組合，在 test sessions 上只跑一次

輸出（backtest_outputs/）：
- `val_sweep_results.csv`
- `selected_strategy_config.json`
- `test_final_results.csv`
- `pnl_curve_val_selected.png`
- `pnl_curve_test_final.png`
- `inventory_curve_test_final.png`
- `baseline_vs_model_pnl_bar.png`
- `threshold_sweep_heatmap_val.png`（3-class 才會有）

你要改的地方：
- `_common.py` 的 maker/taker fee、max inventory、以及 label/feature/window 參數
"""

from _common import build_cfg, build_paths, print_experiment_banner
from new_era.models_playaround_15.backtest_val_sweep import backtest_val_sweep_then_test


def main() -> None:
    cfg = build_cfg()
    print_experiment_banner(cfg)
    paths = build_paths(cfg)
    out = backtest_val_sweep_then_test(cfg, paths)
    print(f"[04] wrote -> {paths.backtest_dir}")
    tf = out["test_final"]
    print(f"[04] test baseline={tf['baseline_pnl']:.2f} model={tf['model_pnl']:.2f} (+{tf['improvement_vs_baseline']:.2f})")


if __name__ == "__main__":
    main()
