# Reproducibility Manifest

## Exact Pipeline

- Experiment root: `new_era/16_binary_label_v2`
- Data source: `data/NEW_ALL`
- Symbol: `TXFB6`
- Label fill window: `1` tick
- Label horizon: `10` ticks
- Adverse threshold: `0.5` ticks
- Training run: `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\16_binary_label_v2\outputs\model_weighted_bce`
- Model: `HLSFriendlyTCNClassifier`, channels `[16, 16, 16]`, kernel `3`, dropout `0.1`
- Sampling mode: `weighted_bce`
- Epochs: `20`
- Batch size: `1024`
- Learning rate: `0.001`
- Seed: `42`
- Bid pos_weight: `122.209570`
- Ask pos_weight: `123.339737`
- Threshold selection split: `val`
- Test used during threshold selection: `False`
- Frozen bid threshold: `0.015858395025134087`
- Frozen ask threshold: `0.02090054750442505`
- Final test net PnL: `132590.00`

## One-Block Reproduction Command

```bash
python new_era/16_binary_label_v2/01_reproduce_016.py
python new_era/16_binary_label_v2/02_preprocess_quote_eligible.py
python new_era/16_binary_label_v2/03_make_binary_labels.py
python new_era/16_binary_label_v2/04_train_binary_model.py --sampling_mode weighted_bce
python new_era/16_binary_label_v2/05_select_thresholds.py
python new_era/16_binary_label_v2/06_backtest_binary_cancel.py
python new_era/16_binary_label_v2/08_generate_audits.py
python new_era/16_binary_label_v2/09_filter_waterfall.py
python new_era/16_binary_label_v2/10_robust_random_baseline.py --trials 5000
python new_era/16_binary_label_v2/11_model_metrics.py
python new_era/16_binary_label_v2/12_pnl_decomposition.py
python new_era/16_binary_label_v2/13_feature_ablation.py
```