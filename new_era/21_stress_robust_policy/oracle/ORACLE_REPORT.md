# Oracle Upper Bound Report

Oracle rows use future information and are not tradable policies.

- Perfect-oracle moderate test feasibility: `positive feasible`.
- perfect_toxic_cancel moderate test PnL: `-87355`.
- perfect_quote_pnl_filter moderate test PnL: `217660`.

## Validation-Selected Diagnostic Policies

- Best validation score threshold: `best_val_score_r0.4`.
- Frozen test moderate PnL: `-68495`.
- Best validation spread+score grid: `best_val_spread_score_s2_r0.4`.
- Frozen test moderate PnL: `-83715`.

If perfect oracles are positive but validation-selected score policies are not, the bottleneck is signal/policy quality rather than mathematical infeasibility under the locked stress assumptions.