# Queue-Aware Stress Explanation

Queue-aware stress is implemented as a proxy stress framework.

- Fill haircut reduces realized fills to simulate worse queue priority.
- Good-fill haircut preferentially removes favorable fills for pessimistic analysis.
- Toxic multiplier increases the penalty of adverse fills.
- Cancel delay shifts the action timing to mimic slower control.

Exact proxy formula from `new_era/18_queue_aware/queue_stress_core.py`:

- Effective cancel mask: if `delay <= 0`, `pull_eff[t] = pull[t]`; otherwise `pull_eff[t] = pull[t - delay]` and the first `delay` rows cannot be cancelled by that signal.
- A candidate fill is kept only if `neutral_u <= neutral_fill_haircut`.
- For favorable fills, an additional pessimistic filter applies: keep only if `good_u <= good_fill_haircut`.
- Favorable definition: bid fill is favorable when `future_mid_return_H_ticks > 0`; ask fill is favorable when `future_mid_return_H_ticks < 0`.
- `extra_mult = max(toxic_multiplier - 1, 0)`.
- Bid adverse ticks: `max(0, -future_mid_return_H_ticks)`.
- Ask adverse ticks: `max(0, future_mid_return_H_ticks)`.
- `toxic_penalty_ticks = extra_mult * adverse_ticks`.
- Bid fill cash update: `cash -= bid_price * POINT_VALUE + toxic_penalty_ticks * POINT_VALUE`; `fees += MAKER_FEE`; `inventory += 1`.
- Ask fill cash update: `cash += ask_price * POINT_VALUE - toxic_penalty_ticks * POINT_VALUE`; `fees += MAKER_FEE`; `inventory -= 1`.
- Session final equity: `cash - fees + inventory * final_mid * POINT_VALUE`.

Equivalent single-fill final mark-to-market contribution:

- Bid: `(final_mid - bid_price) * POINT_VALUE - toxic_penalty_ticks * POINT_VALUE - MAKER_FEE`.
- Ask: `(ask_price - final_mid) * POINT_VALUE - toxic_penalty_ticks * POINT_VALUE - MAKER_FEE`.

These stressors are useful for conservative sensitivity analysis, but they are not true queue replay. The data lacks real order IDs, acknowledgements, queue position, and exchange-level fill priority.
