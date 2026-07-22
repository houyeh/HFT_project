# HLS Non-ML Controller Spec

Inputs: side bit, spread_ticks, spread_bucket.
Selected validation policy: `bid_only_spread_ge_1`.
Output: quote/keep when the side-aware spread rule or side x spread bucket LUT returns true; otherwise block quote.
No ML score, floating-point model, online retraining, or future information is required.
