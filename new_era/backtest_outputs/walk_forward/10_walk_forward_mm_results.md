# Walk-Forward Market Making Results

- Model weights: c:\Users\holbo\00VS_projjects\00_HFT_project\new_era\005_tcn_weights.pth
- Features: TOP_8_FEATURES=['obi', 'queue_imbalance', 'weighted_mid_deviation', 'trade_intensity_short', 'deep_obi', 'queue_imbalance_change', 'normalized_spread', 'cancel_pressure_short']
- Z-score: per-session rolling window=2000, min_periods=100, clip [-10,10]

## TXFB6

- Fold 1: test=20260126Night N=563,484 acc=0.8459 baseline_pnl=-47,205 best=tcn_skew pnl=-72,030 (+-24,825) (pthr=0.55, ethr=0.05)
- Fold 2: test=20260127Day N=410,587 acc=0.8213 baseline_pnl=-27,205 best=tcn_skew pnl=-24,205 (+3,000) (pthr=0.90, ethr=0.05)
- Fold 3: test=20260127Night N=755,736 acc=0.8136 baseline_pnl=90,990 best=tcn_skew pnl=132,320 (+41,330) (pthr=0.55, ethr=0.30)
