# Horizon Break-Even Report

This is a diagnostic evaluation of the frozen `bid_only_spread_ge_1` baseline. No horizon is selected for deployment and no strategy parameter is tuned.

- First positive test horizon: `50`.
- First positive NEW4 horizon: `session_final`.

## Multi-Horizon Summary

| split   | horizon       | fill_count   | sequential_proxy_pnl   | max_drawdown   | max_abs_inventory   | average_holding_events   | win_rate   | average_trade_pnl   |
|:--------|:--------------|:-------------|:-----------------------|:---------------|:--------------------|:-------------------------|:-----------|:--------------------|
| NEW4    | 1             | 714          | -163,610.00            | 59,480.00      | 1                   | 1.0                      | 0.00%      | -229.15             |
| NEW4    | 5             | 714          | -125,910.00            | 51,335.00      | 3                   | 5.0                      | 16.95%     | -176.34             |
| NEW4    | 10            | 714          | -123,110.00            | 61,235.00      | 4                   | 10.0                     | 25.63%     | -172.42             |
| NEW4    | 20            | 714          | -78,410.00             | 71,735.00      | 6                   | 20.0                     | 39.22%     | -109.82             |
| NEW4    | 50            | 714          | -76,910.00             | 119,250.00     | 10                  | 50.0                     | 43.98%     | -107.72             |
| NEW4    | 100           | 714          | -104,610.00            | 160,615.00     | 11                  | 100.0                    | 41.88%     | -146.51             |
| NEW4    | 200           | 714          | -119,810.00            | 185,985.00     | 14                  | 200.0                    | 44.12%     | -167.80             |
| NEW4    | session_final | 714          | 13,061,790.00          | 6,316,555.00   | 273                 | 367001.7450980392        | 71.01%     | 18,293.82           |
| test    | 1             | 419          | -65,385.00             | 38,070.00      | 1                   | 1.0                      | 0.00%      | -156.05             |
| test    | 5             | 419          | -39,185.00             | 24,985.00      | 1                   | 5.0                      | 23.15%     | -93.52              |
| test    | 10            | 419          | -22,885.00             | 22,495.00      | 2                   | 10.0                     | 38.42%     | -54.62              |
| test    | 20            | 419          | -9,685.00              | 16,580.00      | 2                   | 20.0                     | 49.64%     | -23.11              |
| test    | 50            | 419          | 1,315.00               | 15,890.00      | 2                   | 50.0                     | 55.61%     | 3.14                |
| test    | 100           | 419          | 8,315.00               | 14,395.00      | 2                   | 100.0                    | 52.98%     | 19.84               |
| test    | 200           | 419          | 19,315.00              | 10,545.00      | 3                   | 200.0                    | 54.89%     | 46.10               |
| test    | session_final | 419          | 5,275,115.00           | 1,928,820.00   | 299                 | 302513.13603818615       | 92.84%     | 12,589.77           |

## Interpretation

Short horizons should be read as a closer proxy for market-making execution. The `session_final` row is a long-horizon directional markout diagnostic, not an executable short-horizon market-making claim.
At least one split only turns positive at `session_final`, which means the original positive quote-level EV is not well supported as a short-horizon market-making edge.
