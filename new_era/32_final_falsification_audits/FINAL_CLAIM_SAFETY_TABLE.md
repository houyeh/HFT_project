# Corrected Final Claim Safety Table

This file overwrites the earlier Phase 032 interpretation with the Phase 033 corrected interpretation.

The key correction is that the exact matched random quote baseline is saturated and non-informative: `bid_only_spread_ge_1` selects the entire exact candidate universe `side=bid and spread>=1`.

## Audit Summary

- `new4` current-row baseline PnL `6,348,770`, selected `714`.
- `test` current-row baseline PnL `2,312,590`, selected `419`.
- Phase 032 lagged-book timing-artifact delay `0` `new4` PnL `6,348,770`, gap vs delay0 `0`; this is not delayed-entry EV.
- Phase 032 lagged-book timing-artifact delay `1` `new4` PnL `6,348,770`, gap vs delay0 `0`; this is not delayed-entry EV.
- Phase 032 lagged-book timing-artifact delay `2` `new4` PnL `6,348,770`, gap vs delay0 `0`; this is not delayed-entry EV.
- Phase 032 lagged-book timing-artifact delay `5` `new4` PnL `6,348,770`, gap vs delay0 `0`; this is not delayed-entry EV.
- Phase 032 lagged-book timing-artifact delay `10` `new4` PnL `6,348,770`, gap vs delay0 `0`; this is not delayed-entry EV.
- Phase 032 lagged-book timing-artifact delay `0` `test` PnL `2,312,590`, gap vs delay0 `0`; this is not delayed-entry EV.
- Phase 032 lagged-book timing-artifact delay `1` `test` PnL `2,312,590`, gap vs delay0 `0`; this is not delayed-entry EV.
- Phase 032 lagged-book timing-artifact delay `2` `test` PnL `2,312,590`, gap vs delay0 `0`; this is not delayed-entry EV.
- Phase 032 lagged-book timing-artifact delay `5` `test` PnL `2,312,590`, gap vs delay0 `0`; this is not delayed-entry EV.
- Phase 032 lagged-book timing-artifact delay `10` `test` PnL `2,312,590`, gap vs delay0 `0`; this is not delayed-entry EV.
- Exact matched random `new4` p-value `1.000`, zero-variance exact universe `True`; non-informative saturation diagnostic only.
- Exact matched random `test` p-value `1.000`, zero-variance exact universe `True`; non-informative saturation diagnostic only.
- Worst stress-grid PnL `new4`: `903,050`.
- Worst stress-grid PnL `test`: `524,200`.
- Worst LOSO `test` drops `20260127Night`: remaining PnL `996,450`, dropped session contribution `1,316,140`.
- Worst LOSO `new4` drops `20260102`: remaining PnL `2,575,440`, dropped session contribution `3,773,330`.

## Claim Table

- `HLS-friendly deterministic comparator logic`: `ALLOWED`. The controller is HLS-friendly comparator logic under this proxy dataset. Evidence: Decision is side==bid and spread_ticks>=1; no floating point model or dynamic state needed.
- `No future columns in decision logic`: `ALLOWED`. Decision logic is audited as not using quote_ev/future columns. Evidence: future-token audit pass=True; current-row mismatch pass=True; boundary pass=True.
- `External OOS proxy robustness`: `ALLOWED_WITH_PROXY_CAVEAT`. The rule shows simulator-derived proxy robustness, with duplicate-data caveats. Evidence: Frozen test and NEW4 remain positive under the base proxy; Phase 033 extends checks to all rebuilt unused OOS and flags NEW/NEW1 duplication risk.
- `Phase 032 delay table as true execution-delay proxy`: `FORBIDDEN_AS_WRITTEN`. Treat the Phase 032 delay table as lagged-book timing-artifact stress only. Evidence: Phase 032 delay grid only lagged decision features and did not recalculate delayed-entry EV; Phase 033 supplies corrected execution_delay_proxy_grid.csv.
- `Exact-matched random quote-selection edge`: `FORBIDDEN`. The exact-matched random quote baseline is non-informative due to saturation. Evidence: Exact matched random saturated=True; selected count equals candidate universe count for side=bid and spread>=1.
- `Broader regime-vs-random diagnostic`: `ALLOWED_DIAGNOSTIC`. Use broader all-valid-action random controls only as diagnostic evidence of side/regime behavior. Evidence: Corrected Phase 033 broader random baselines available=True.
- `NEW and NEW1 as independent OOS`: `FORBIDDEN_IF_DEDUP_FLAGS_DUPLICATE`. Do not sum NEW and NEW1 as independent OOS if dedup flags likely duplication. Evidence: Corrected Phase 033 dedup audit available=True; it must be checked before summing OOS PnL.
- `Real-world profitability`: `FORBIDDEN`. Do not claim real-world profitability. Evidence: Data lacks true queue priority, aggressor side, and exchange order-level replay.
- `FPGA PnL necessity`: `FORBIDDEN`. Frame FPGA only as a risk-control/latency candidate. Evidence: Latency sensitivity is not proof that FPGA deployment is necessary for production PnL.
- `True queue priority replay`: `FORBIDDEN`. Do not claim true exchange queue priority replay. Evidence: The evaluation uses simulator-derived quote EV, not order-id queue replay.

## Corrected Phase 033 Artifacts

- Broader random controls available: `True` at `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\33_corrected_final_falsification_audits\results\regime_vs_random_baselines.csv`.
- Corrected delayed-entry proxy grid available: `True` at `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\33_corrected_final_falsification_audits\results\execution_delay_proxy_grid.csv`.
- NEW/NEW1 dedup audit available: `True` at `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\33_corrected_final_falsification_audits\results\dedup_independence_audit.csv`.

## Final Boundary

Allowed claims: HLS-friendly deterministic logic and proxy OOS robustness with duplicate-data caveats.

Forbidden claims: real-world profitability, FPGA PnL necessity, true queue priority replay, true execution-delay proof from the old Phase 032 delay table, and beating exact-matched random quote selection.
