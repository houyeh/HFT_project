# Data Sources Audit

- NEW_ALL: Main train/val/test dataset for robust testing. (Symbol: TXFB6)
- NEW3: Stress-test dataset representing unfavorable regimes. (Symbol: TXFA6)
- session list: 9 sessions for NEW_ALL (e.g. `20260121Day` to `20260127Night`). Multiple sessions for NEW3 (e.g. `20260113` to `20260330`).
- train / val / test split: Session-ordered chronological split (70/15/15) to prevent forward data leakage.
- trade tape availability: YES
- LOB depth availability: YES
- aggressor_side availability: NO

**CRITICAL LIMITATION**: 
Because `aggressor_side` is missing from the dataset, true exchange queue reconstruction/replay is fundamentally impossible. We can only use queue-aware sensitivity proxies (e.g. `price_level_removed`) for fill estimation.
