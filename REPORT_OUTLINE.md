# Report Outline

Working title: **What Does a Microsecond Cost? A Task-Level Design-Space Exploration of Small Recurrent Inference Cores on a Resource-Constrained FPGA**

Status: `outline_only_evidence_bound_no_body_prose`

Evidence authority: `EVIDENCE_PACK.md`. Every quantitative statement below is either bound to a named row in that pack or marked `[EVIDENCE GAP]`. Verification tiers must remain visible in the final report.

## Page Budget

| component | estimated pages |
|---|---:|
| Main text, Sections 1-11 | 10.5 |
| Appendix | 1.0 |
| **Total** | **11.5** |

## 1. Introduction

### (a) Claims

- Frame the research question: on a fixed, low-cost FPGA, what task-level value and resource cost are paid for each reduction in inference latency?
- State the scope boundary prominently: **this is not an HFT system; limit-order-book quoting is the benchmark task, not the product.**
- Contribution 1: a multi-point, routed latency-resource-value frontier rather than a single implementation point.
- Contribution 2: a replaceable-model but fixed-verification feature-to-board pipeline with explicit evidence tiers.
- Contribution 3: an empirical recurrence-versus-feedforward analysis that separates task quality, persistent-state cost, and non-comparable latency objects.

### (b) Evidence bindings

- `EVIDENCE_PACK` Table 5, all 8 rows: routed and non-routed frontier.
- Table 3, frozen contract rows: evaluation governance.
- Tables 6 and 9: recurrence/feedforward and financial-meaning boundaries.

### (c) Figure/table needs

- One compact problem diagram: fixed FPGA budget -> latency/resource choices -> task-value consequences.
- No quantitative hero chart here; reserve the frontier for Section 6.

### (d) Estimated length

- 0.75 page.

## 2. Background and Related Work

### (a) Claims

- Position prior work by its role in this report, not by paper-summary chronology:
  - GRU accelerators: recurrent hardware precedent.
  - Build Fast Trade Fast: finance/FPGA systems context.
  - FINN-GL: quantized FPGA inference/toolflow context.
  - DeepLOB: public LOB accuracy benchmark.
  - T-KAN: alternative temporal modeling context.
  - FastBiNLOB/Hedges arXiv:2606.25986: closest latency-and-LOB comparison.
- Draw the distinction line: this report contributes several routed design points with task value on the vertical axis, rather than only software timing or one hardware point.
- Required FastBiNLOB distinction: its reported latency is a single CPU timing object; this work uses routed P&R evidence plus physical-board programming/functional evidence.

### (b) Evidence bindings

- Table 8, DeepLOB Table II rows and metric-definition check.
- Table 5, routed frontier rows.
- Table 7, JTAG/board-validation rows.
- Bibliographic verification status is tracked in `REFERENCES_TO_VERIFY.md`; no numerical literature claim may enter the report until its row is marked `已一手核對`.

### (c) Figure/table needs

- Related-work positioning matrix: work x software timing / synthesis / routed P&R / board / task-value frontier.
- All non-DeepLOB cells remain citation placeholders until the bibliography is independently verified.

### (d) Estimated length

- 0.75 page.

## 3. Methodology: The Feature-to-FPGA Pipeline

### (a) Claims

- Present the reusable pipeline as six linked stages: causal features -> model -> fixed-point conversion -> manual HLS -> routed P&R -> board wrapper/programming.
- Define fixed acceptance gates at each stage so model candidates can change while validation rules do not.
- Define evidence tiers explicitly: `estimate_only`, software/CSim, RTL cosim, csynth, routed P&R failed/met, and on-board functional observation.
- Keep numerical fidelity, timing closure, task value, and board behavior as separate claims.

### (b) Evidence bindings

- Table 1: causal split provenance.
- Table 4, HLS numeric-format row.
- Table 5, `verification_tier` field for every design.
- Table 7, comparator/XSim, programming, and board-status rows.
- Table 9, proxy and timestamp-resolution boundaries.

### (c) Figure/table needs

- Figure: feature-to-FPGA pipeline with one verification gate below each stage.
- Table: evidence-tier definitions and claims permitted at each tier.

### (d) Estimated length

- 1.0 page.

## 4. Experimental Protocol and Its Enforcement

### (a) Claims

- Explain the development/diagnostic/one-shot partition contract and why NEW5 is not a tuning split.
- Show all nine frozen-contract checks as an enforcement artifact, not merely a stated intention.
- Define matched-random testing and distinguish “beats random” from “beats another learned model.”
- Core case study: naive quote-level markout can produce an apparently spectacular result dominated by spread capture, yet the same old-bid policy loses under the locked sequential NEW5 simulator.
- Verified naive case: NEW3 old_bid_only H20 total PnL `1,432,466,070`, positive rate `90.344977%`, and half-spread share of positive components `99.867925%`.
- Locked one-shot contrast: NEW5 sequential excess PnL `-553,365` for old_bid_only.
- Fee-model boundary: maker fee `15.0` is an inherited proxy constant rather than an exchange-certified decomposition; bind the H20 `1x/2x/3x/5x` sweep and break-even multipliers from Table 2 to show sensitivity rather than treating the assumption as exact.

### (b) Evidence bindings

- Table 1, all split rows, with registry scope versus model/frozen scope retained.
- Table 3, nine frozen-contract rows and four sequential causal-fill rows.
- Table 2, NEW3 old_bid_only decomposition row: exact source for `1.43e9`, `90.3%`, and `99.87%`.
- Table 2, four H20 fee-sensitivity rows: old_bid_only and side_by_queue_imbalance on val/NEW4, including `1x/2x/3x/5x` adjusted PnL and break-even multipliers.
- Table 2, side_by_queue_imbalance matched-random row.
- Table 6 note: pooled TCN-vs-PHC sign tests are not matched-random tests.

### (c) Figure/table needs

- Table: nine expected/actual/pass contract checks.
- Figure: paired bars for naive NEW3 old_bid_only decomposition versus locked NEW5 sequential result; annotate that splits and simulators differ, so this is a methodology contrast rather than a direct paired return comparison.
- Small flow diagram: development selection -> frozen NEW3/NEW4 diagnostics -> one-shot NEW5.
- Compact fee-sensitivity table or line inset: horizontal axis fee multiplier (`1x`, `2x`, `3x`, `5x`), vertical axis adjusted H20 PnL proxy units, with the analytical break-even multiplier annotated for each displayed policy/split.

### (d) Estimated length

- 1.25 pages.

## 5. Model: PHC-lite / PHC-STP Deployment Candidate

### (a) Claims

- Describe which PHC-inspired mechanisms were retained and removed, with an engineering reason for each.
- Use the `TXF production` evaluation track: deployed 05C4 PHC-STP v2 dense16 has `1,731` parameters, 16 features, and hidden dimension 16.
- Add the first-use track note: the separate `FI-2010 benchmark` model has `2,132` parameters because its input is a 100x40 window; it is used only in Section 9.
- Numeric implementation: input/weights `ap_fixed<24,12>`, accumulators `ap_fixed<40,16>`.
- Ablation claim: causal ordering and retained state matter strongly; the exact learned decay is not necessary because fixed decay 0.9 nearly matches it.

### (b) Evidence bindings

- Table 4, `TXF production` PHC-STP row and HLS numeric-format row; the `FI-2010 benchmark` row is used only for the one-sentence track distinction.
- Table 4, all fixed-decay, memory-off, reverse-sequence, and shuffled-timestep rows.
- Architecture-only source leads: `0000_competition_proposal/AMD Track-ZeroJitter-構想書_正式版.md` and `meeting/FINAL meeting/研究專題結案報告.docx` Section 3.1.1. The old document's latency and CPU numbers are not admissible.

### (c) Figure/table needs

- Architecture block diagram with retained/removed mechanisms visually separated.
- Ablation delta plot over val/NEW3/NEW4.
- Small identity note preventing reuse of the FI-2010 `2,132` count.

### (d) Estimated length

- 0.75 page.

## 6. Latency-Resource-Value Frontier

### (a) Claims

- Present all eight Table 5 points, including failed and over-budget points; do not draw a frontier from successful points alone.
- Parameter track: all PHC-STP production-model frontier claims in this section use `TXF production` = `1,731` parameters; reduced/diagonal ablations must retain their own model-source labels.
- Finding 1: the production-signal path improves from `83.160 us` to `22.360 us`, a `3.72x` latency reduction with unchanged software action-level H20 markout sum. Call this **task-value-free**, not resource-free: routed DSP rises from 21.36% to 68.64%.
- Finding 2: further reductions have explicit model/value costs: hidden8 reaches `12.020 us`; diagonal reaches `5.300 us`; both carry lower val/NEW3/NEW4 action-level H20 markout sums.
- Finding 3: `3.830 us` is not deployable on Zynq-7020 because csynth DSP is `205.45%` (`452/220`).
- Keep verification labels visible: routed P&R met, cosim failure, or csynth-only.

### (b) Evidence bindings

- Table 5, all 8 rows.
- Table 5 power note: 06F `0.357 W` and 06L `0.564 W`, both Medium-confidence routed reports; incomplete coverage prevents a full power frontier.
- Table 9 only for the later interpretation of whether microseconds have measurable economic meaning; do not merge proxy evidence into hardware measurements.

### (c) Figure/table needs

- **Figure 1:** log-scale latency in microseconds versus NEW3 action-level H20 markout sum (`total_excess_pnl_H20`); marker shape/color encodes verification tier and resource feasibility. The caption must state that this is not causal-fill sequential PnL.
- Full frontier table with cycles, frequency, WNS/TNS, LUT/FF/DSP/BRAM percentages, model source, and three-split action-level H20 markout sums.
- Optional inset: latency versus DSP utilization to show the budget collision.

### (d) Estimated length

- 1.75 pages.

## 7. Why Recurrence? An O(1)-State Argument

### (a) Claims

- Open with the statistical boundary, exactly:
  - distilled TCN causal-fill deltas versus PHC are directionally positive on all three splits: val `+3,340`, NEW3 `+98,925`, NEW4 `+11,955`;
  - each selected TCN policy beats its matched-random control at empirical `p<0.001`;
  - the pooled TCN-minus-PHC exact sign tests use only three split observations and give `p=0.25` for distilled causal-fill/action rows and `p=1.00` for scratch action PnL, so they do not establish TCN superiority at 5%.
- Defensible inference: the current diagnostics cannot show that recurrence is required for task quality, but recurrence offers compact persistent state and lower memory demand.
- Parameter track: the recurrent production reference in this section is the `TXF production` 1,731-parameter model, not the 2,132-parameter FI-2010 classifier.
- Hardware contrast: PHC diagonal point uses 0% BRAM and 530 RTL latency cycles; routed streaming TCN uses 23.57% BRAM and 7,439 interval cycles.
- Mechanism evidence: TCN unroll 1 -> 4 changes latency from 16,510 to 7,438 cycles (`2.22x`, not `4x`), consistent with a memory-port-bound schedule.
- Local partition is functionally necessary in the tested diagonal partial-unroll path: removing it produced `268/1,000` RTL cosim mismatches.
- Mandatory comparability limit: TCN streaming-step latency/interval objects must not be ranked directly against PHC recurrent single-step latency or naive whole-window recompute latency.

### (b) Evidence bindings

- Table 6, distilled val/NEW3/NEW4 rows and pooled-sign-test note.
- Table 6, unroll-scaling and 07C2D routed rows.
- Table 5, V3 diagonal no-local-partition row (`268/1,000`) and valid local-partition V3 row.
- Table 4, memory/order ablation rows.
- Required Table 6 quotation:
  > Latency/interval are still the HLS streaming-step latency and interval objects from csynth. They should not be ranked directly against PHC recurrent single-step latency or naive TCN whole-window recompute latency.

### (c) Figure/table needs

- Hardware comparison table: state representation, BRAM, latency/interval object, cycles, and verification tier.
- Figure: TCN unroll factor versus cycles, with ideal 4x reference and observed 2.22x.
- Callout: no-local-partition `268/1,000` failure versus local-partition `0/1,000` pass.

### (d) Estimated length

- 1.25 pages.

## 8. On-Board Validation

### (a) Claims

- Parameter track: this section uses the `TXF production` 1,731-parameter model.
- Latency tier: `routed (derived from post-route timing closure)`.
- Board-evidence tier: `on-board (functional + bit-level equivalence only)`.
- Required wording:
  > Latency is **derived from post-route static timing closure** (2,316 cycles at 90 MHz = 25.733 µs), not independently measured on silicon. The board-level validation establishes **functional and bit-level equivalence** between the deployed bitstream and the software reference (27 of 4,096 events differ by at most 1 LSB, producing **zero decision flips**), and confirms that the design operates at the closed timing. **An on-chip cycle counter or ILA-based latency measurement was not implemented; this is stated as a limitation.**
- The stable demo path is the solid-color HDMI bitstream, with LED-only fallback.
- Replay result: 4,096 rows, 27 drift rows, max 1 LSB, 32 channel deltas split 11 negative/21 positive, and zero decision flips.
- Margin wording must be exact: minimum policy margin **among drift rows** is 72 LSB; minimum across all rows is 1 LSB.
- The extra S_DRAIN cycle produced an identical drift set, a negative result for that narrow settle-cycle hypothesis; root cause remains inconclusive.

### (b) Evidence bindings

- Table 7, all 7 rows.
- Missing item M1: no board-measured latency and no `.ltx` capture.
- Contradictions C2/C3: preflight guide and stale P&R CSV must not override later JTAG records.

### (c) Figure/table needs

- Board-validation ladder with tier labels: XSim positive/negative -> `routed (derived from post-route timing closure)` -> JTAG programming -> `on-board (functional + bit-level equivalence only)`.
- Drift histogram/table: 0/1-LSB magnitude, sign distribution, decision flips, and both margin definitions.
- Photo/screenshot placeholder only if an existing board image is separately provenance-checked.

### (d) Estimated length

- 0.75 page.

## 9. Public Benchmark: FI-2010

### (a) Claims

- Explain why weighted F1, not macro F1, is the directly comparable metric for DeepLOB Table II; Accuracy==weighted Recall is the verification clue.
- Parameter track: this section uses `FI-2010 benchmark` = `2,132` parameters and a 100x40 input window; it does not use the TXF production count.
- At horizon 50, PHC v3 hidden16 obtains `66.50%` weighted F1 versus DeepLOB `80.35%`, a `13.85` percentage-point gap.
- Capacity result: PHC hidden64 (`23,876` parameters) remains within the hidden16 seed distribution in accuracy/F1 terms; added capacity does not close the gap.
- Efficiency claim must be scoped: the FI-2010 PHC v3 has `2,132` parameters versus the DeepLOB paper Table III rounded count of `60,000`, or approximately `1/28.1`; the same paragraph must retain the `13.85` percentage-point weighted-F1 deficit at horizon 50.
- Include TCN small and TCN scratch as capacity-matched and higher-capacity feedforward controls.

### (b) Evidence bindings

- Table 8, all 9 rows.
- Table 8, PHC hidden16, hidden64, TCN small, TCN scratch, and DeepLOB horizon 10/20/50 rows.
- Table 8, Accuracy==Recall and sklearn recompute checks.

### (c) Figure/table needs

- Comparison table: weighted F1, macro F1 where available, parameter count, horizon, and evidence type.
- Scatter: parameter count versus weighted F1, with published DeepLOB visually distinguished from local runs.

### (d) Estimated length

- 0.75 page.

## 10. Negative Results and Limitations

### (a) Claims

- Preserve negative results as design evidence. Six evidence-backed cases for the main text:
  1. naive old-bid markout is dominated by half-spread and reverses under locked sequential evaluation;
  2. learned decay is not necessary even though memory/order are important;
  3. no-local-partition partial unroll fails RTL cosim (`268/1,000`);
  4. the fastest diagonal unroll16 point exceeds DSP budget (`452/220`);
  5. S_DRAIN extra-cycle experiment does not remove the 1-LSB drift;
  6. TXF's 1 ms timestamp resolution cannot resolve 23.16 versus 83.16 us economic effects.
- Method boundary: LOBSTER provides nanosecond-scale stale exposure (`2.233353%` at 25.733 us versus `7.198588%` at corrected CPU N=4 p99 262.6 us), but it is a five-ticker, single-day 2012 NASDAQ sample, not TXF and not a PnL/fill measurement.
- 06T C(k) is an event-index paired-label diagnostic and hypothetical scale estimate, not causal latency-to-PnL evidence.
- Limitations: no network interface; not a complete trading system; **no independent on-silicon latency measurement (no cycle counter / ILA)**; power exists only for two routed points and has Medium confidence; maker-fee derivation is not exchange-certified, although Table 2 bounds the inherited constant with a multiplier sweep.
- Do **not** say “no power numbers”: routed power exists for 06F and 06L, but coverage is incomplete.

### (b) Evidence bindings

- Table 2 and Table 3 old-bid rows.
- Table 4 fixed-decay and memory/order rows.
- Table 5 no-local-partition and over-budget rows; Table 5 power note.
- Table 7 S_DRAIN row.
- Table 9 all resolution, LOBSTER, and C(k) rows plus required boundary quotations.
- Missing items M1 and M2.
- Contradiction C4: only corrected 262.6 us / 7.198588% is official; 646.318/6299.416 us remains stress/noise provenance.
- `PROJECT_STORY_RETROSPECTIVE.md` is a root-level narrative index, not a quantitative source; each negative-result number remains bound to the phase AUDIT/results rows above.

### (c) Figure/table needs

- Negative-results table: hypothesis, observed result, engineering consequence, evidence tier.
- Resolution-boundary graphic: TXF 1 ms granularity versus FPGA/CPU microsecond references, alongside the separate LOBSTER stale-exposure points.

### (d) Estimated length

- 1.0 page.

## 11. Conclusion

### (a) Claims

- Answer the title question conditionally: the first production-model speedup preserves the action-level H20 markout task-value metric but consumes more hardware; sub-12-us points require model simplification and measurable loss on that metric; the 3.83-us point exceeds the device DSP budget.
- Reiterate that routed determinism and board functional evidence are established, while board-measured latency and a complete trading-system claim are not.
- Close with the transferable methodological contribution: task value, numerical fidelity, resources, timing closure, and board behavior must be reported as separate axes.

### (b) Evidence bindings

- Table 5, all frontier rows.
- Table 7, programming/board evidence and M1 boundary.
- Tables 6 and 9, recurrence and financial-meaning boundaries.

### (c) Figure/table needs

- No new figure; refer back to Figure 1 and the validation-tier table.

### (d) Estimated length

- 0.5 page.

## Appendix

### (a) Claims

- Provide traceability rather than new interpretation.
- Include full verification-tier definitions, complete resource table, source paths, and minimal reproduction commands already documented by the source phases.

### (b) Evidence bindings

- `EVIDENCE_PACK` row-count manifest and all source paths.
- Table 5 complete resource rows.
- Table 7 bitstream identities.
- Missing/contradiction registers M1-M2 and C1-C5.

### (c) Figure/table needs

- Full resource/latency table.
- Evidence-tier legend.
- Reproduction-path index.

### (d) Estimated length

- 1.0 page, outside the 10.5-page main text.

## Evidence-Insufficiency Register for Drafting

| id | affected section | unsupported or incomplete item | drafting action | status/source result |
|---|---|---|---|---|
| O1 | 2 | Bibliographic claims for GRU accelerators, Build Fast Trade Fast, FINN-GL, T-KAN, and FastBiNLOB | Verify primary papers before drafting related-work prose; track status in `REFERENCES_TO_VERIFY.md`. | `[EVIDENCE GAP]` until individually marked `已一手核對` |
| O2 | 5 | Exact PHC-paper-to-PHC-lite retained/removed mechanism mapping | Extract only the architecture comparison from the located proposal/final-report sources; reject their obsolete latency/CPU statements. | `[SOURCE LOCATED; EXTRACTION PENDING]` |
| O5 | 8/10 | Physical board-measured latency | Keep the latency claim at `routed (derived from post-route timing closure)`; board evidence is `on-board (functional + bit-level equivalence only)`. | `[MISSING]` M1 |
| O6 | 4/10 | Exchange/broker basis for maker fee 15.0 | Label it an inherited proxy assumption and report the Table 2 sensitivity sweep and break-even multipliers. | `[MISSING DERIVATION; SENSITIVITY AVAILABLE]` M2 |

## Required Special-Evidence Checks

| requested fact | result | authoritative binding | drafting permission |
|---|---|---|---|
| old_bid_only naive markout: approximately `1.43e9` PnL, `90.3%` positive rate, `99.87%` half-spread share | **Verified** for NEW3 H20: `1,432,466,070`, `90.344977%`, `99.867925%` | `EVIDENCE_PACK` Table 2, old_bid_only NEW3 row; source `n2ew_era/02e_final_non_ml_feasibility_decision/results/old_bid_only_pnl_decomposition.csv` | May be used with split, horizon, and naive-markout qualifier |
| local array partition removed: RTL cosim `268/1,000` failed | **Verified** for 06U V3 diagonal unroll8 without local partition | `EVIDENCE_PACK` Table 5, no-local-partition row; source `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` | May be used only as the tested partial-unroll/local-partition result, not as a universal theorem |
