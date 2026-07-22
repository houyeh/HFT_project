# Root-Cause Blocker: HGB Convergence Inconclusive

Phase 03B.5 found that the selected shallow HGB family reached `max_iter=80` without sklearn early stopping terminating first. This does not prove the Phase 03B result is invalid, but it prevents a clean claim that training convergence is fully closed.

Required closure before model-expansion work:

- Run a targeted train/validation-only HGB iteration-budget audit.
- Keep test, NEW3, and NEW4 frozen.
- Do not use this blocker to tune thresholds on OOS.
- Continue to forbid raw absolute price and future label features.

Evidence files:

- `results/training_diagnostics.csv`
- `results/hgb_staged_scores.csv`
