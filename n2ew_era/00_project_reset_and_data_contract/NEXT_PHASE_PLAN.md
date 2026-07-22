# Next Phase Plan

## Roadmap

1. `01_full_l2_action_universe_builder`
   - Construct full bid/ask candidate actions from quote-eligible rows.
   - Preserve causal feature boundaries.
   - Attach short-horizon sequential proxy labels.

2. `02_short_horizon_sequential_proxy_baselines`
   - Evaluate old baseline and simple rules at H = 1, 5, 10, 20, 50, 100.
   - Report session, side, spread, event-rate, and drawdown diagnostics.

3. `03_l2_imbalance_rule_search`
   - Perform validation-only search over L2 depth, imbalance, spread, and event-rate regimes.
   - Compare against no quote, old baseline, and random matched controls.

4. `04_full_action_ml_ev_models`
   - Train ML on the full action universe, not only the residual baseline-selected universe.
   - Use only causal L2 features.
   - Evaluate frozen test and OOS performance at short horizons.

5. `05_hls_quote_gate_prototype`
   - Convert the best validation-approved simple policy or compact ML policy into an HLS-friendly quote-control primitive.
   - Use fixed thresholds, small LUTs, or simple arithmetic.
   - Avoid claims of real-world profitability or FPGA-driven PnL improvement.

## Recommended Phase 01 Start

Begin with the full L2 action universe builder and data contract checks before any policy search or ML training.
