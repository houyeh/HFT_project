# Proxy Quote PnL Definition

This document intentionally uses the term `quote-level proxy PnL` rather than realized trading PnL.
The current artifacts evaluate the value of candidate quote opportunities using future markout, fees, and stress assumptions.
They do not constitute an order-level exchange replay.

## 1. What Is One Unit of PnL?

- Prices are futures index points or ticks depending on the source column; the implementation converts price/tick movement into currency-like PnL with `POINT_VALUE = 200.0`.
- Each filled quote pays `MAKER_FEE = 15.0`.
- Therefore one tick or point of favorable markout contributes `200.0` PnL units before fees and stress penalties.
- Reported `quote_value_*`, `quote_ev_moderate`, and policy `selected_quote_pnl` fields are net of the maker fee when the fill is kept by the proxy.

## 2. Exact Quote-Level Formula

Let:

- `side` be `bid` or `ask`.
- `quote_price` be `bid_px_1[t]` for bid quotes and `ask_px_1[t]` for ask quotes.
- `final_mid` be the last mid price in the session.
- `future_mid_return_H_ticks = (mid[t+H] - mid[t]) / tick_size`.
- `POINT_VALUE = 200.0` and `MAKER_FEE = 15.0`.

Side-specific H-horizon adverse ticks:

```text
side_markout_H = future_mid_return_H_ticks       for bid
side_markout_H = -future_mid_return_H_ticks      for ask
adverse_ticks = max(0, -side_markout_H)
```

Session-final terminal markout:

```text
bid_terminal_ticks = final_mid - bid_quote_price
ask_terminal_ticks = ask_quote_price - final_mid
```

Stress-adjusted quote-level proxy value:

```text
extra_adverse_multiplier = max(toxic_multiplier - 1.0, 0.0)
gross_value = terminal_ticks * POINT_VALUE
fee = MAKER_FEE
adverse_penalty = extra_adverse_multiplier * adverse_ticks * POINT_VALUE
kept_by_fill_haircut = neutral_u <= neutral_fill_haircut
quote_value_stressed = gross_value - fee - adverse_penalty  if kept_by_fill_haircut else 0
```

For bid quotes:

```text
quote_value_bid = (final_mid - bid_price) * POINT_VALUE
                  - MAKER_FEE
                  - max(toxic_multiplier - 1.0, 0.0) * max(0, -future_mid_return_H_ticks) * POINT_VALUE
```

For ask quotes:

```text
quote_value_ask = (ask_price - final_mid) * POINT_VALUE
                  - MAKER_FEE
                  - max(toxic_multiplier - 1.0, 0.0) * max(0, future_mid_return_H_ticks) * POINT_VALUE
```

If `neutral_u > neutral_fill_haircut`, the stressed quote value is set to `0.0`, representing a fill removed by the proxy queue haircut.

## 3. Future Horizon H

- The adverse-selection horizon is `H = 10` events/ticks in the current pipeline.
- Historical labels store this through `label_horizon_ticks = 10` and `future_mid_return_H_ticks` in `new_era/16_binary_label_v2/outputs/binary_labels_v2.parquet`.
- NEW4 generation uses `build_new4_frame(horizon_ticks=10)`.
- Important nuance: `H` is used to compute the adverse penalty and fill toxicity direction. The main `quote_value_moderate` terminal markout uses `final_mid`, not an explicit exit order at `t+H`.

## 4. Fill Proxy Trigger Condition

Candidate quote actions are only created when the next row crosses the resting quote proxy:

- Bid fill proxy: `ask_px_1[t+1] <= bid_px_1[t]`.
- Ask fill proxy: `bid_px_1[t+1] >= ask_px_1[t]`.

This is a proxy for a passive quote being reached. It is not exchange aggressor-side truth and not a queue-position replay.

## 5. Fees and Stress Multipliers

Current stress settings:

| Stress | neutral_fill_haircut | toxic_multiplier |
| --- | ---: | ---: |
| base | 1.00 | 1.00 |
| mild | 0.75 | 1.25 |
| moderate | 0.50 | 1.50 |

- Fee is subtractive: `- MAKER_FEE` for every kept filled quote.
- `toxic_multiplier` acts only through the extra adverse penalty `max(toxic_multiplier - 1, 0) * adverse_ticks * POINT_VALUE`.
- In some reports the term `adverse_multiplier` refers to this extra adverse penalty mechanism; the implementing code names it `toxic_multiplier`.
- `neutral_fill_haircut` is a deterministic random-mask keep rule, not a learned model.

## 6. Does the Simulator Track Inventory?

No, it is quote-level markout evaluation.

The existing `quote_ev_moderate` pipeline values each candidate quote independently. It does not track persistent inventory, order acknowledgements, partial fills, queue position, or explicit exit orders. Phase 038 adds a separate sequential proxy inventory backtest for interpretability, but that is still a proxy and not true exchange replay.

## 7. Source Files

| source_file                                                                                          | function_or_artifact              |   line | role                                                                                                                           |
|:-----------------------------------------------------------------------------------------------------|:----------------------------------|-------:|:-------------------------------------------------------------------------------------------------------------------------------|
| C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\18_queue_aware\queue_stress_core.py             | POINT_VALUE / MAKER_FEE constants |     24 | Defines one PnL unit conversion and per-fill maker fee: POINT_VALUE=200.0, MAKER_FEE=15.0.                                     |
| C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\18_queue_aware\queue_stress_core.py             | build_action_table                |     60 | Creates candidate quote-fill proxy actions: bid fills when next ask <= current bid; ask fills when next bid >= current ask.    |
| C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\26_robust_baseline_side_aware\baseline_utils.py | actions_from_frame                |     75 | Computes historical train quote_value_moderate from final-mid markout, maker fee, fill haircut, and adverse H-horizon penalty. |
| C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\27_non_ml_final_baseline_audit\common_027.py    | STRESS_SPECS                      |     42 | Defines base/mild/moderate neutral_fill_haircut and toxic_multiplier settings.                                                 |
| C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\27_non_ml_final_baseline_audit\common_027.py    | build_new4_frame                  |    128 | Computes NEW4 future_mid_return_H_ticks using H=10 by default.                                                                 |
| C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\27_non_ml_final_baseline_audit\common_027.py    | quote_value_for_stress            |    185 | Computes quote_value_base/mild/moderate for val/test/NEW4 actions.                                                             |
| C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\27_non_ml_final_baseline_audit\common_027.py    | load_actions_for_split            |    200 | Loads val/test action tables and attaches stress quote values when missing.                                                    |
| C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\28_residual_quote_ev_veto\common_028.py         | baseline_universe                 |    194 | Copies quote_value_moderate into quote_ev_moderate and defines negative_ev_moderate.                                           |
| C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\31_opportunity_cost_audit\common_031.py         | build_full_action_universe        |     73 | Combines train/val/test/NEW4 actions and exposes quote_ev_moderate in the full action universe.                                |
