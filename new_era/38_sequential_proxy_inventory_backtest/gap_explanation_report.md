# Gap Explanation Report

This report decomposes why `quote_ev_moderate` is positive while the H10 sequential proxy is negative.

## Aggregate Bridge

| Component | PnL units |
| --- | ---: |
| Sequential H10 proxy PnL | -145,995 |
| Add: kept terminal markout vs all H10 markout | 8,901,200 |
| Add: fee reversal for haircut-removed fills | 8,505 |
| Add: adverse penalty bridge | -102,350 |
| Result: quote_ev_moderate | 8,661,360 |

## Required Answer

Is the original positive quote_ev mainly coming from session-final directional drift rather than short-horizon market-making edge? Yes. The bridge is dominated by replacing short-horizon H10 markout with session-final terminal markout on fills kept by the moderate haircut.

The terminal-vs-H10 bridge is `8,901,200` against a total H10-to-quote-EV difference of `8,807,355`, a share of `101.07%`.

Interpretation boundary: this does not prove the controller has no use. It proves the large positive quote-level EV is not the same object as short-horizon executable market-making PnL.

## Outputs

- `results/quote_vs_sequential_gap_detail.csv`
- `results/quote_vs_sequential_gap_by_session.csv`
- `results/quote_vs_sequential_gap_by_bucket.csv`
- `figures/gap_waterfall.png`
