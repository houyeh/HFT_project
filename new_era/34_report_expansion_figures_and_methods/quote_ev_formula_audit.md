# Quote EV Formula Audit

`quote_ev_moderate` is a simulator-derived proxy label. It is not a true exchange queue replay label, and it does not use real queue position, order acknowledgements, or exchange fill IDs.

## Code Locations

- Historical train action EV generation: `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\26_robust_baseline_side_aware\baseline_utils.py` around line 75 (`actions_from_frame`).
- Historical val/test and NEW4 stress formula: `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\27_non_ml_final_baseline_audit\common_027.py` around line 185 (`quote_value_for_stress`).
- Residual ML dataset construction: `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\28_residual_quote_ev_veto\common_028.py` around line 194 (`baseline_universe`).
- Candidate action/fill-proxy trigger: `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\18_queue_aware\queue_stress_core.py` around line 60 (`build_action_table`).

## Horizon and Fill Proxy

- Horizon `H = 10` ticks.
- Historical labels store `label_horizon_ticks = 10` and `future_mid_return_H_ticks` in `new_era/16_binary_label_v2/outputs/binary_labels_v2.parquet`.
- NEW4 construction also uses `build_new4_frame(horizon_ticks=10)`.
- Bid fill-proxy candidate: next row ask price crosses the current quote, `ask[t+1] <= bid[t]`.
- Ask fill-proxy candidate: next row bid price crosses the current quote, `bid[t+1] >= ask[t]`.

## Quote Price Assumption

- Bid action quote price is the current best bid, `price = bid_px_1[t]`.
- Ask action quote price is the current best ask, `price = ask_px_1[t]`.
- The final mark-to-market price is the session final mid price, `final_mid = last(mid_price)` within the same session.

## Base Formula

For an action with side-specific future markout:

```text
side_markout = future_mid_return_H_ticks                for bid
side_markout = -future_mid_return_H_ticks               for ask
adverse_ticks = max(0, -side_markout)
terminal_ticks = final_mid - quote_price                for bid
terminal_ticks = quote_price - final_mid                for ask
eligible = neutral_u <= neutral_fill_haircut
quote_ev = terminal_ticks * POINT_VALUE - MAKER_FEE - max(toxic_multiplier - 1, 0) * adverse_ticks * POINT_VALUE
quote_ev_stressed = quote_ev if eligible else 0
```

Constants used by the current proxy implementation:

- `POINT_VALUE = 200.0`.
- `MAKER_FEE = 15.0`.
- `neutral_u` is a deterministic stable random mask keyed by event/action and side; it is fixed across stress settings for comparability.

## Stress Parameters

| Stress level | neutral_fill_haircut | toxic_multiplier | cancel_delay_ticks |
| --- | ---: | ---: | ---: |
| base | 1.00 | 1.00 | 0 |
| mild | 0.75 | 1.25 | 0 |
| moderate | 0.50 | 1.50 | 0 |

The residual ML label uses the moderate setting: `quote_ev_moderate = quote_ev_stressed` under `neutral_fill_haircut=0.5`, `toxic_multiplier=1.5`, and `delay=0`.

## Explicit Limitation

This formula is a proxy quote-EV label. It is useful for controlled falsification and stress tests, but it should not be described as realized exchange PnL or true queue-aware fill toxicity.
