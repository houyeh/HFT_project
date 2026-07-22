# NEW2 Top-3 Playaround Backtest (Fixed Config)

Data: `data/NEW2` (TXFB6, 3 sessions)

This folder contains out-of-sample backtests on NEW2 using the **top-3 experiments from 015 playaround** (ranked by `pnl_improvement_flatten` on the original NEW dataset), with **fixed strategy parameters** taken from each experiment's `backtest_outputs/selected_strategy_config.json` (no re-sweep on NEW2).

## Summary

See: `top3_playaround_on_new2_summary.csv`

Columns:
- `baseline_pnl_after_fees_flatten`, `model_pnl_after_fees_flatten`, `improvement_flatten`
- `baseline_trades`, `model_trades`
- `baseline_max_drawdown`, `model_max_drawdown`

## Per experiment outputs

Each experiment has its own folder:

- `<experiment_id>/baseline_new2.csv`
- `<experiment_id>/model_new2.csv`
- `<experiment_id>/pnl_curve_new2.png`
- `<experiment_id>/inventory_curve_new2.png`
- `<experiment_id>/summary_new2.json`

