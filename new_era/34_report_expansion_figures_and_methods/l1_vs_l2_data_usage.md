# L1 vs L2 Data Usage

The deployed baseline uses only L1-level information:

- best bid price
- best ask price
- spread in ticks
- quote side

The rejected ML extensions used richer causal features derived from L2 or event history, such as queue imbalance, depth ratios, volatility, and event rate. Those features may still be useful for future research, but they did not pass the incremental value criteria in the residual-veto setting.

This distinction matters for hardware design: the baseline maps to simple comparators, while ML/L2 policies require feature extraction, score calibration, and additional audit burden.

Figure: `figures/l1_l2_book_diagram.png`.
