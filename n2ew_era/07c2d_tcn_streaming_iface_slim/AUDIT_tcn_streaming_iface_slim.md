status: resolved_iface_slim_cosim_memory_fix

# 07C2D TCN Streaming Top Interface Slimming Audit

## Scope Boundary

- Work directory: `n2ew_era/07c2d_tcn_streaming_iface_slim/`.
- Source baseline: copied from `n2ew_era/07c2c_tcn_streaming_cosim_diagnostic/hls_src/`.
- No change to `n2ew_era/07c2_tcn_streaming_closure/`, `n2ew_era/07c2b_tcn_streaming_speed/`, or `n2ew_era/07c2c_tcn_streaming_cosim_diagnostic/`.
- No model retraining, no weight changes, no NEW5 access, no unroll/BRAM setting changes.
- Only the top-level HLS interface was slimmed. The shared-history `shared_u4` architecture and `TCN_STAGE2_CHANNEL_UNROLL=4` were kept.
- After verification, three unused copied legacy files with the old diagnostic-output signature were removed from the 07C2D copy only: `hls_src/tcn_stage2_resource_testbench.cpp`, `hls_src/tcn_stage2_resource_testbench_partial.cpp`, and `hls_src/tcn_stage2_streaming_impl.hpp`. They were not referenced by the 07C2D manifests or HLS configs used for the runs below.

## Interface Change

The 07C2C top-level `tcn_streaming_step` exposed the final readout plus five block-level diagnostic outputs (`block0_out` through `block4_out`) and their valid flags. 07C2D removes those diagnostic outputs from the synthesizable top interface.

07C2D C++ top arguments are now only:

| argument | role |
| --- | --- |
| `x[TCN_CHANNELS]` | input feature vector |
| `reset` | reset stream state |
| `readout[TCN_READOUT_DIM]` | final readout |
| `readout_valid[1]` | final readout validity |

The block outputs are still computed internally as local arrays and still feed downstream blocks and readout pooling. A CSim-only helper, guarded by `#ifndef __SYNTHESIS__`, was added so the debug CSim can compare `block0` to `block4` against 07C2C without exposing those arrays as RTL top ports.

Static check after cleanup: all remaining 07C2D testbenches call `tcn_streaming_step(in_vec, reset, readout, readout_valid)` only; no remaining HLS source file declares `port=blockN_*` top interfaces.

## CSim Equivalence Gate

CSim debug summaries were compared against the 07C2C summaries for the event scales that had 07C2C reference outputs. All numeric cells and text rows matched exactly.

| variant | event scale | compared cells | rows equal | max numeric delta |
| --- | ---: | ---: | --- | ---: |
| distilled_shared_u4 | 174 | 34 | true | 0.0 |
| scratch_shared_u4 | 174 | 34 | true | 0.0 |
| distilled_shared_u4 | 252 | 34 | true | 0.0 |
| scratch_shared_u4 | 252 | 34 | true | 0.0 |

Source: `results/iface_slim_csim_vs_07c2c_summary_compare.csv`.

This is the guardrail that the slim interface did not change the block/readout computation. 1008-event CSim was also run with the readout-only testbench; it passed for both variants, but 07C2C did not have a completed 1008-event reference summary for exact side-by-side comparison.

## Harness Complexity

The harness complexity collapse matches the 07C2C hypothesis: removing diagnostic top ports removes the `ap_ovld` explosion and the output-feedback handling.

| design/run | autowrap bytes | TV markers | interface rows | `ap_ovld` rows | output-feedback messages | SVR active mentions | UVM agent mentions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 07C2C shared_u4 diagnostic top | 116,334 | 386 | 279 | 240 | 80 | 185 | 185 |
| 07C2D slim top | 39,664 | 46 | 29 | 0 | 0 | 20 | 20 |

Reduction summary:

| metric | old/new ratio | reduction |
| --- | ---: | ---: |
| autowrap bytes | 2.93x smaller | 65.9% |
| TV markers | 8.39x smaller | 88.1% |
| interface rows | 9.62x smaller | 89.6% |
| `ap_ovld` rows | removed | 100.0% |
| output-feedback messages | removed | 100.0% |

Sources: `n2ew_era/07c2c_tcn_streaming_cosim_diagnostic/results/cosim_harness_complexity_counts.csv` and `results/iface_slim_cosim_metrics.csv`.

## Cosim Results

All cosim runs below used explicit timeouts and completed before the timeout. `wall_seconds` is the XSIM kernel elapsed time parsed from the cosim log, not the full Vitis command wall time including compile/elaboration overhead.

| scale | variant | timeout | RTL transactions | readout rows | outcome | xsim peak memory KB | wall seconds | seconds/tx |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 174-event reduced | distilled_shared_u4 | 1h | 175/175 | 0 | PASS | 98,060 | 262 | 1.50 |
| 174-event reduced | scratch_shared_u4 | 1h | 175/175 | 0 | PASS | 98,072 | 259 | 1.48 |
| 252-event one window | distilled_shared_u4 | 2h | 253/253 | 1 | PASS | 111,368 | 390 | 1.54 |
| 252-event one window | scratch_shared_u4 | 2h | 253/253 | 1 | PASS | 111,300 | 237 | 0.94 |
| 1008-event original scale | distilled_shared_u4 | 2h | 1012/1012 | 4 | PASS | 234,812 | 954 | 0.94 |
| 1008-event original scale | scratch_shared_u4 | 2h | 1012/1012 | 4 | PASS | 234,784 | 1000 | 0.99 |

Source: `results/iface_slim_cosim_metrics.csv`.

Readout-only post-check summaries:

| scale | variant | readout max abs diff | readout RMSE | joint sign agreement | valid alignment errors |
| --- | --- | ---: | ---: | ---: | ---: |
| 252 | distilled_shared_u4 | 0.324146 | 0.249628 | 1 | 0 |
| 252 | scratch_shared_u4 | 0.0883064 | 0.0669563 | 1 | 0 |
| 1008 | distilled_shared_u4 | 0.420075 | 0.308342 | 1 | 0 |
| 1008 | scratch_shared_u4 | 0.149958 | 0.076498 | 1 | 0 |

The 174-event scale has zero final readout rows because it stops before the final readout warm-up completes. It is still useful for interface-memory comparison, but the 252/1008 runs are the readout-level evidence.

## Old vs New Outcome

| case | old 07C2B/07C2C outcome | new 07C2D slim outcome |
| --- | --- | --- |
| 174-event distilled | PASS, 175/175, 17,229,916 KB | PASS, 175/175, 98,060 KB |
| 174-event scratch | PASS, 175/175, 18,961,868 KB | PASS, 175/175, 98,072 KB |
| 252-event distilled | FAIL/OOM at 187/253 after 47m23s | PASS, 253/253, 111,368 KB |
| 252-event scratch | not run after distilled OOM in 07C2C | PASS, 253/253, 111,300 KB |
| 1008-event distilled | timeout, 177/1012 after 7200s budget in 07C2B | PASS, 1012/1012, 234,812 KB |
| 1008-event scratch | timeout, 132/1012 after 7200s budget in 07C2B | PASS, 1012/1012, 234,784 KB |

The 174-event memory drop is 175.7x for distilled and 193.3x for scratch. The full 1008-event cosim now completes at roughly 0.94 to 0.99 xsim seconds per transaction, versus the previous 07C2B timeout trajectory of about 40.7 to 54.5 seconds per observed transaction.

## Verification Tier

Current 07C2D evidence tier:

`cosim_1008event_pass_readout_only_iface_slim`

This is stronger than the 07C2C tier `cosim_174event_pass_block4_50valid`, because 07C2D completed the original 1008-event transaction count for both scratch and distilled variants and includes final readout post-check rows. It is still not a new P&R result; no P&R rerun was requested or performed in 07C2D.

## Artifact Index

| artifact | path |
| --- | --- |
| CSim equivalence summary | `results/iface_slim_csim_vs_07c2c_summary_compare.csv` |
| Cosim metrics | `results/iface_slim_cosim_metrics.csv` |
| 174-event distilled log | `artifacts/distilled_shared_u4_event174_cosim/hls_run_cosim.log` |
| 174-event scratch log | `artifacts/scratch_shared_u4_event174_cosim/hls_run_cosim.log` |
| 252-event distilled log | `artifacts/distilled_shared_u4_1window_cosim/hls_run_cosim.log` |
| 252-event scratch log | `artifacts/scratch_shared_u4_1window_cosim/hls_run_cosim.log` |
| 1008-event distilled log | `artifacts/distilled_shared_u4_4window_cosim/hls_run_cosim.log` |
| 1008-event scratch log | `artifacts/scratch_shared_u4_4window_cosim/hls_run_cosim.log` |

## Honest Conclusion

The interface-slim experiment resolves the cosim memory question. The 07C2C diagnosis was correct: the dominant problem was the top-level diagnostic interface and generated cosim harness, not the RTL cycle count alone. After removing the block diagnostic outputs from the synthesizable top interface, `ap_ovld` rows fell from 240 to 0, interface rows fell from 279 to 29, autowrap size fell from 116,334 bytes to 39,664 bytes, and xsim memory dropped from 17-19 GB at 174 events to about 98 MB. The same slim interface then completed 252-event and 1008-event RTL cosim for both shared_u4 variants.

Practical next step: keep diagnostic block outputs out of the HLS top interface. If block-level debug is needed, use CSim-only helpers or a separate diagnostic-only component, not the deployable cosim/P&R top.
