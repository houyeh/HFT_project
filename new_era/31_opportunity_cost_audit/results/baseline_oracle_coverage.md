# Baseline Coverage vs Oracle

Oracle-positive quote means `quote_ev_moderate > 0`. The oracle is non-tradable and uses future information; this table measures opportunity cost only.

- `train`: baseline PnL `-1586875`, captured oracle-positive PnL `1961215`, missed oracle-positive PnL `4437350`, avoided negative EV `2383540`.
- `val`: baseline PnL `1218095`, captured oracle-positive PnL `1273450`, missed oracle-positive PnL `19215`, avoided negative EV `1315070`.
- `test`: baseline PnL `2312590`, captured oracle-positive PnL `2326930`, missed oracle-positive PnL `3160`, avoided negative EV `3440970`.
- `new4`: baseline PnL `6348770`, captured oracle-positive PnL `6881315`, missed oracle-positive PnL `373845`, avoided negative EV `7286565`.

Interpretation: high missed oracle-positive PnL means the bid-only spread baseline is conservative. Whether that opportunity is tradable depends on causal selector performance, not the oracle itself.
