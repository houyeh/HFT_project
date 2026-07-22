# ML Comeback Future Work

The conservative baseline has two pain points:

- It is blunt: in this universe it is effectively `bid_all`, not fine-grained quote selection.
- It may miss positive-EV opportunities outside the bid-only region.

Future ML routes should avoid repeating the residual-veto trap:

- Full-action EV model: predict stress-adjusted quote EV across bid and ask actions, not only inside the baseline universe.
- Dynamic regime classifier: learn when the bid-side regime is active rather than hard-coding it globally.
- Rolling calibration: use past-only quantile thresholds instead of static absolute scores.
- Sequence model research: use the exported RTX training pack for causal TCNs, while keeping validation-only policy selection.

Any comeback must still beat strong non-ML baselines, matched random controls, and fresh OOS/walk-forward tests.

Figure: `figures/ml_future_work_decision_tree.png`.
