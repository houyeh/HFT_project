# Cancel Signal Definition

For `016_v2_binary`, a cancel signal is a side-specific binary quote action derived from the validation-frozen toxic-score threshold.

- `pull_bid[t] = bid_prob_toxic[t] >= bid_threshold`
- `pull_ask[t] = ask_prob_toxic[t] >= ask_threshold`
- Frozen thresholds: bid `0.015858395025134087`, ask `0.02090054750442505`
- Threshold source: validation only, target 10% cancel rate per side
- Execution meaning: when `pull_bid` is true, the passive bid quote is cancelled for that event; when `pull_ask` is true, the passive ask quote is cancelled for that event.

These are quote-risk proxy signals, not exchange-confirmed cancels or queue-position-aware fill outcomes.
