# Formal Binary Label Definition

Let `t` denote a quote-eligible event index within a single session.

- Current best bid: `b_t`.
- Current best ask: `a_t`.
- Current mid: `m_t = (a_t + b_t) / 2`.
- Tick size: `q`.
- Fill horizon: `tau_fill = 1` raw quote tick.
- Adverse horizon: `H = 10` raw quote ticks.
- Adverse threshold: `theta = 0.5` ticks.

Bid fill proxy:

`F_bid(t) = 1{min_trade_price[t+1, t+tau_fill] <= b_t OR min_ask[t+1, t+tau_fill] <= b_t}`

Ask fill proxy:

`F_ask(t) = 1{max_trade_price[t+1, t+tau_fill] >= a_t OR max_bid[t+1, t+tau_fill] >= a_t}`

Future mid return in ticks:

`r_H(t) = (m_{t+H} - m_t) / q`

Labels:

`bid_toxic(t) = 1{F_bid(t)=1 and r_H(t) <= -theta}`

`ask_toxic(t) = 1{F_ask(t)=1 and r_H(t) >= theta}`

Interpretation: this is a side-specific maker quote-cancel risk target, not an active alpha label and not true fill truth.