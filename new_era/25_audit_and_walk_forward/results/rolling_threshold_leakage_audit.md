# Rolling Threshold Leakage Audit

- Selected policy: `side_specific__016_v2__w50__bid0.50__ask0.10`.
- Required window: `50`.
- Source contains explicit `shift(1)`: `True`.
- Threshold formula audited: `threshold_t = quantile(score[t-N:t-1], q)` within `session_id|side` groups.
- Recomputed threshold max absolute difference: `9.02056207508e-17`.
- Recomputed selected-mask mismatches: `0`.
- First row of each session/side group has no threshold and is not selected: `True`.

Conclusion: No lookahead leakage detected in the frozen 024 action file.
