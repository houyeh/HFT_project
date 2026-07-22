# Threshold Selection Report

Thresholds were frozen before test evaluation.

- Selection split: `val`
- `test_used`: `False`
- Selection mode: `target_cancel_rate`
- Target validation cancel rate per side: `0.1`
- Bid threshold: `0.015858395025134087`
- Ask threshold: `0.02090054750442505`

Validation evidence:

- Bid val cancel rate: `0.100000`
- Ask val cancel rate: `0.100000`
- Combined val proxy cancel value: `3040.50` ticks

Test evidence, evaluated after freezing:

- Test evaluation count: `1`
- Test net PnL: `132590.00`

Code references:

- `05_select_thresholds.py:274-275` writes validation-only metadata.
- `06_backtest_binary_cancel.py:295` rejects threshold files that are not validation-only.
- `06_backtest_binary_cancel.py:383` documents blind application to test.