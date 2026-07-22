# Baseline Method Explanation

The final non-ML controller is `bid_only_spread_ge_1`.

Decision logic:

- Start from quote-eligible rows with a valid L1 book: positive best bid, positive best ask, and non-negative spread.
- Construct two passive candidate actions per event: one bid quote and one ask quote.
- Keep a candidate only when `side == bid` and `spread_ticks >= 1`.
- The rule does not read future returns, quote EV, toxicity labels, oracle labels, or model scores.

Important interpretation:

- In the current quote-eligible universe, all candidate spreads are at least one tick, so `spread_ticks >= 1` is saturated.
- The controller is therefore best described as a simple bid-side regime filter, not a fine-grained alpha model.
- The evidence is simulator-derived proxy robustness, not proof of real-world profitability.

Figure: `figures/baseline_flowchart.png`.
