# Strict Timing Rebuild Audit

This audit separates three questions that were conflated in Phase 032:

- A: current-row book decision with current proxy EV. This is retained only as a legacy timing reference.
- B: previous-row book decision with current proxy EV. Decision features are available before the entry row.
- C: previous-row book decision with one-tick delayed-entry proxy EV. This is the strictest timing rebuild in this audit.

No setup uses `quote_ev`, `future_*`, toxicity labels, oracle labels, or PnL columns inside the decision rule. Those fields are used only after selection for evaluation.

Decision source no-future-token pass: `True`.
Session-boundary lag reset pass across groups: `True`.
