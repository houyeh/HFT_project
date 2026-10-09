status: diagnosed_impact_bounded_root_cause_inconclusive

# 06S2 Wrapper 1-LSB Readout Drift Root Cause And Impact

## Scope Boundary

- Scope is the existing 06S2 PYNQ-Z2 demo wrapper replay path only:
  `n2ew_era/06s2_pynq_z2_wrapper_bringup/rtl/pynq_z2_phc_demo_wrapper.sv`.
- XSim used the same 06S2 sequential HLS RTL path already used by wrapper bring-up:
  `hls_seed2_90_replay_component/hls_component/hls_component_seq/hls/sim/verilog`.
- Replay size was the full 4096-event positive wrapper replay.
- This audit did not modify HLS core arithmetic, HLS pragmas, model weights, BRAM `.mem/.coe`, bitstream, P&R outputs, or any NEW5 data.
- Debug changes are compile-time gated:
  `PHC_WRAPPER_DUMP_ALL_READOUT` writes simulation-only CSV rows; `PHC_DRAIN_EXTRA_CYCLE` adds one experimental wrapper drain cycle only when explicitly defined.

## Instrumentation Added

`pynq_z2_phc_demo_wrapper.sv` now has an ifdef-only full readout dump at `S_CHECK`. Each row records:

- vector index
- actual captured readout0/readout1
- expected BRAM readout0/readout1
- signed LSB deltas and absolute LSB diffs
- exact-drift flag before applying the existing 1-LSB tolerance

The helper script `scripts/run_xsim_demo_wrapper_direct.ps1` now supports separate work/log suffixes plus `-DumpAllReadout` and `-DrainExtraCycle`. The analysis script is:

`03_analyze_wrapper_readout_drift.py`

It decodes signed Q24.12 values and applies the existing 05A/05C4 policy rule:

```python
decision[(bid >= ask) & (bid > threshold)] = DECISION_BID
decision[(ask > bid) & (ask > threshold)] = DECISION_ASK
```

with the selected 05C4 threshold `0.0`.

## XSim Runs

| run | compile-time experiment | XSim status | row count | drift rows | max abs diff | log |
|---|---:|---:|---:|---:|---:|---|
| baseline | `PHC_WRAPPER_DUMP_ALL_READOUT` | pass | 4096 | 27 | 1 LSB | `logs/xsim_direct_positive_dump_baseline_xsim.log` |
| drain2 | `PHC_WRAPPER_DUMP_ALL_READOUT` + `PHC_DRAIN_EXTRA_CYCLE` | pass | 4096 | 27 | 1 LSB | `logs/xsim_direct_positive_dump_drain2_xsim.log` |

Both runs passed the existing positive wrapper gate with `READOUT_TOL_LSB=1`:

```text
DEMO_WRAPPER_DONE current_index=4095 pass=1 fail=0 first_mismatch_index=4294967295
MISMATCH_STATS count=0 max_readout_lsb_diff=1 readout_tol_lsb=1
POSITIVE_TEST_PASS
```

## Drift Summary

The old `BRAM_VECTOR_PACKING.md` note said exact zero-LSB comparison exposed 27 rows. The full dump re-confirmed that number exactly:

| run | rows | drift rows | nonzero channel deltas | delta -1 | delta +1 | delta other | direction | decision flips |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| baseline | 4096 | 27 | 32 | 11 | 21 | 0 | mixed | 0 |
| drain2 | 4096 | 27 | 32 | 11 | 21 | 0 | mixed | 0 |

Important: the drift direction is mixed, not one-sided. That argues against a fixed directional rounding/truncation bias. It does not support saying "all actual values are consistently one LSB high" or "all are consistently one LSB low".

## Per-Row Drift List

Full CSVs:

- `results/wrapper_readout_dump_baseline.csv`
- `results/wrapper_1lsb_drift_rows_baseline.csv`
- `results/wrapper_readout_dump_drain2.csv`
- `results/wrapper_1lsb_drift_rows_drain2.csv`
- `results/wrapper_1lsb_drift_summary.csv`

Baseline and drain2 have identical drift indices and identical actual/expected values after excluding the `run` label.

| idx | bid actual/expected | d_bid | ask actual/expected | d_ask | decision | margin LSB |
|---:|---|---:|---|---:|---|---:|
| 3223 | `0xfffbf2` / `0xfffbf2` | 0 | `0x00017c` / `0x00017d` | -1 | ask -> ask | 381 |
| 3224 | `0xfffbda` / `0xfffbda` | 0 | `0x0001ac` / `0x0001ad` | -1 | ask -> ask | 429 |
| 3225 | `0xfffb8d` / `0xfffb8d` | 0 | `0x0001fe` / `0x0001ff` | -1 | ask -> ask | 511 |
| 3244 | `0xfff8b3` / `0xfff8b3` | 0 | `0x000488` / `0x000487` | +1 | ask -> ask | 1159 |
| 3245 | `0xfff6d1` / `0xfff6d1` | 0 | `0x000667` / `0x000666` | +1 | ask -> ask | 1638 |
| 3247 | `0xfff3c0` / `0xfff3c0` | 0 | `0x0009b2` / `0x0009b1` | +1 | ask -> ask | 2481 |
| 3250 | `0xfff7f6` / `0xfff7f7` | -1 | `0x00058e` / `0x00058d` | +1 | ask -> ask | 1421 |
| 3254 | `0xfffb40` / `0xfffb3f` | +1 | `0x000227` / `0x000227` | 0 | ask -> ask | 551 |
| 3256 | `0x000298` / `0x000298` | 0 | `0xfffaf1` / `0xfffaf0` | +1 | bid -> bid | 664 |
| 3258 | `0xffff5e` / `0xffff5e` | 0 | `0xfffdcc` / `0xfffdcb` | +1 | no_quote -> no_quote | 565 |
| 3259 | `0xffff83` / `0xffff83` | 0 | `0xfffd8c` / `0xfffd8b` | +1 | no_quote -> no_quote | 629 |
| 3261 | `0x0001c4` / `0x0001c4` | 0 | `0xfffb8e` / `0xfffb8d` | +1 | bid -> bid | 452 |
| 3264 | `0x000249` / `0x000248` | +1 | `0xfffad2` / `0xfffad2` | 0 | bid -> bid | 584 |
| 3265 | `0x000537` / `0x000537` | 0 | `0xfff7f9` / `0xfff7f8` | +1 | bid -> bid | 1335 |
| 3268 | `0x000049` / `0x000048` | +1 | `0xfffd14` / `0xfffd14` | 0 | bid -> bid | 72 |
| 3269 | `0xfffde0` / `0xfffde0` | 0 | `0xffff8b` / `0xffff8c` | -1 | no_quote -> no_quote | 544 |
| 3271 | `0x000114` / `0x000113` | +1 | `0xfffc40` / `0xfffc41` | -1 | bid -> bid | 275 |
| 3272 | `0x000073` / `0x000072` | +1 | `0xfffcd9` / `0xfffcd9` | 0 | bid -> bid | 114 |
| 3274 | `0x000380` / `0x00037f` | +1 | `0xfff9f9` / `0xfff9f9` | 0 | bid -> bid | 895 |
| 3275 | `0xffff87` / `0xffff86` | +1 | `0xfffdf0` / `0xfffdf1` | -1 | no_quote -> no_quote | 527 |
| 3277 | `0xfffaa1` / `0xfffaa1` | 0 | `0x0002dd` / `0x0002de` | -1 | ask -> ask | 734 |
| 3278 | `0xfffbae` / `0xfffbad` | +1 | `0x0001ad` / `0x0001ad` | 0 | ask -> ask | 429 |
| 3281 | `0xfffab1` / `0xfffab0` | +1 | `0x000294` / `0x000294` | 0 | ask -> ask | 660 |
| 3283 | `0xfffdd7` / `0xfffdd7` | 0 | `0xffff5d` / `0xffff5e` | -1 | no_quote -> no_quote | 553 |
| 3288 | `0x00006f` / `0x00006e` | +1 | `0xfffcc2` / `0xfffcc2` | 0 | bid -> bid | 110 |
| 3289 | `0x0004bb` / `0x0004ba` | +1 | `0xfff8ac` / `0xfff8ad` | -1 | bid -> bid | 1210 |
| 3295 | `0x00027c` / `0x00027b` | +1 | `0xfffb55` / `0xfffb56` | -1 | bid -> bid | 635 |

## Decision And Scoreboard Impact

Using the existing 05A/05C4 decision rule with threshold `0.0`, the 27 drift rows produce:

- decision flips among drift rows: 0 / 27
- decision mix among drift rows: 11 bid, 11 ask, 5 no_quote
- minimum policy margin among drift rows: 72 LSB, equal to `72 / 4096 = 0.017578125` Q24.12

Across all 4096 rows, the closest non-drift policy boundary margin is 1 LSB, but the drift rows themselves are not near enough to threshold/tie boundaries to flip decisions.

The PYNQ demo scoreboard is also unaffected by this readout drift because the existing demo scoreboard CSVs use `stale_top_change_events` from event timing/book-change counts. The scoreboard files do not consume wrapper readout or policy decisions:

- `results/demo_replay_summary.csv`
- `results/demo_replay_staleness_timeseries.csv`

Therefore this 1-LSB readout drift has no observed impact on the current demo scoreboard and no impact on the existing 05C4-style threshold decision for these 4096 replay rows.

## S_DRAIN Extra-Cycle Experiment

The suspected code path was `S_DRAIN`, whose comment already states that HLS `ap_memory` outputs may complete their final write on the completion edge. I tested a compile-time-only extra cycle:

- baseline: `S_DRAIN` waits 1 cycle before `S_CHECK`
- drain2: `S_DRAIN` waits 2 cycles before `S_CHECK`

Result: the extra cycle did not reduce or change the drift set.

| comparison | result |
|---|---|
| baseline drift rows | 27 |
| drain2 drift rows | 27 |
| baseline-only indices | none |
| drain2-only indices | none |
| actual/expected/delta rows excluding `run` label | identical |

This is a negative result for the narrow "one more S_DRAIN settle cycle fixes it" hypothesis.

## Honest Conclusion

The audit confirms the exact 27-row, max-1-LSB wrapper readout drift and bounds its practical impact to zero decision flips for the replayed 4096 rows. It also shows the drift is mixed-direction, so it is not behaving like a fixed AP_TRN rounding-mode or BRAM packing sign bias.

The simplest state-feedback settle-timing repair, adding one more `S_DRAIN` cycle before compare/state feedback, did not change the drift at all. That means the broader wrapper/RTL-feedback boundary remains the most plausible locality, but this run does not prove the root cause is merely "S_DRAIN was one cycle too short." The current evidence is best described as: HLS core CSim/Cosim and BRAM packing remain unlikely culprits; wrapper RTL-feedback recurrence/timing remains the suspect area; the single extra-drain-cycle experiment is inconclusive-to-negative for that specific timing fix.
