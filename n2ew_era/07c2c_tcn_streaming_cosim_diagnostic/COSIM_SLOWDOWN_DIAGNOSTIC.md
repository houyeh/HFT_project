status: resolved_reduced_cosim_evidence_full_window_open

This report lives in `n2ew_era/07c2c_tcn_streaming_cosim_diagnostic/`.

**Scope Boundary**

This task only used the 07C2B shared-history `u4` TCN artifacts as the starting point. I did not modify model weights, did not retrain, did not touch NEW5, did not try `TCN_STAGE2_CHANNEL_UNROLL=8`, and did not modify `n2ew_era/07c2_tcn_streaming_closure/` or earlier 07C directories. The only older-file edit was the requested wording correction in `n2ew_era/07c2b_tcn_streaming_speed/AUDIT_tcn_streaming_speed.md`.

The 07C2C HLS source copy is only a verification copy. The added partial testbench accepts an optional event limit so the same shared_u4 RTL can be tested at 252 events and then at a smaller 174-event scale.

**Executive Finding**

The TCN cosim slowdown is not explained by RTL latency cycles alone. PHC 06U V3 advances one transaction by about 531 cycles and finishes 1000/1000 xsim transactions in 14 seconds. TCN shared_u4 advances steady transactions by about 5,933 to 7,386 cycles, only about 11x to 14x more simulated time, but observed wall time is about 10 seconds per transaction for the 174-event completed runs and 41 to 55 seconds per observed transaction in the 1008-event timeout runs.

The dominant evidence points to the HLS-generated cosim harness and interface topology: TCN exposes many scalar `ap_ovld` diagnostic ports, triggers 80 `output feeding back to input case` messages, creates 185 SVR/UVM agent-related log mentions, and its autowrap file has 386 TV marker occurrences versus 18 for PHC. Completed 174-event TCN cosim also reaches 17 to 19 million KB of xsimkernel-reported simulation memory, while PHC stays at 33,840 KB.

The 252-event one-window cosim still did not complete: distilled reached 187/253 transactions and failed with xsim out-of-memory after 47m23s. I then cut to 174 events. Both scratch and distilled passed the reduced RTL cosim at 175/175 transactions. This is valid reduced-scale evidence only: `cosim_174event_pass_block4_50valid`, not a 252-event full-window pass and not deployable-level evidence.

**Transaction Structure**

| run | top-call / transaction structure | expected transactions | observed result |
| --- | --- | ---: | --- |
| PHC 06U V3 | one CSV vector calls `phc_diag_step` once | 1000 | PASS, 1000/1000 |
| 07C2B TCN scratch shared_u4 | 4 windows; each window has 1 reset call + 252 event calls | 1012 | timeout, 132/1012 after 7200s budget |
| 07C2B TCN distilled shared_u4 | 4 windows; each window has 1 reset call + 252 event calls | 1012 | timeout, 177/1012 after 7200s budget |
| 07C2C TCN distilled shared_u4, 1 window | 1 reset call + 252 event calls | 253 | FAIL, OOM at 187/253 after 47m23s |
| 07C2C TCN distilled shared_u4, event_limit=174 | 1 reset call + 174 event calls | 175 | PASS, 175/175 |
| 07C2C TCN scratch shared_u4, event_limit=174 | 1 reset call + 174 event calls | 175 | PASS, 175/175 |

Source: `results/xsim_transaction_time_stats.csv`, parsed from Vitis/xsim logs.

**xsim Timing Distribution**

| run | outcome | tx observed / total | p50 delta cycles | p90 delta cycles | max delta cycles | wall sec / observed tx | xsimkernel peak memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PHC 06U V3 | pass | 1000 / 1000 | 531 | 531 | 534 | 0.014 | 33,840 KB |
| TCN 07C2B scratch u4 | timeout | 132 / 1012 | 5,933 | 6,663 | 7,386 | 54.55 | N/A, external timeout |
| TCN 07C2B distilled u4 | timeout | 177 / 1012 | 5,933 | 7,386 | 7,386 | 40.68 | N/A, external timeout |
| TCN 07C2C distilled 252 | OOM fail | 187 / 253 | 6,663 | 7,386 | 7,386 | 15.20 | OOM before final memory report |
| TCN 07C2C distilled 174 | pass | 175 / 175 | 5,933 | 7,386 | 7,386 | 10.47 | 17,229,916 KB |
| TCN 07C2C scratch 174 | pass | 175 / 175 | 5,933 | 7,386 | 7,386 | 9.88 | 18,961,868 KB |

Interpretation: the simulated RTL time gap is about one order of magnitude, but the wall-clock gap is hundreds to thousands of times. That mismatch makes a pure datapath-cycle explanation insufficient.

**Harness Complexity Evidence**

| design | autowrap cpp bytes | TV marker occurrences | interface port rows | `ap_ovld` port rows | output-feedback messages | SVR active mentions | UVM agent mentions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PHC 06U V3 | 32,705 | 18 | 34 | 0 | 0 | 0 | 0 |
| TCN shared_u4 | 116,334 | 386 | 279 | 240 | 80 | 185 | 185 |

Source: `results/cosim_harness_complexity_counts.csv`.

The TCN top function returns not only final readout but also per-block diagnostic arrays (`block0_out` through `block4_out`) and validity flags. In csynth, those become 279 RTL interface port rows, including 240 `ap_ovld` rows. This is consistent with the 80 repeated HLS messages:

`This is output feeding back to input case. Cosim wouldn't go into SVUVM flow.`

That message count comes from the actual 07C2C HLS log, not from source-code inspection alone.

**xsim Mode And Waveform Check**

Both PHC and TCN run through Vitis HLS Verilog C/RTL cosim using xsim. The xsimkernel lines show `-simmode gui` and `-wdb ...` for both designs, so GUI/WDB mode by itself does not explain the TCN-only slowdown.

For TCN, `tcn_streaming_step.tcl` contains `#log_wave -r /`, followed by `run all` and `quit`; the `log_wave` line is commented. The parser found no explicit `log_wave` command in the HLS cosim logs. Therefore there is no log evidence that an accidental full waveform dump is the primary bottleneck.

The HLS cosim logs also did not contain `ram_2p` or `BIND_STORAGE` strings, so I cannot pin the slowdown on a named BRAM port model from log evidence. The shared BRAM history is still part of the design state, but the directly observed bottleneck is the generated scalar-port/readwrite-output cosim topology plus xsim memory growth.

**Reduced Cosim Result**

The 252-event one-window test was attempted first, as requested. Distilled passed CSim and csynth, then RTL cosim failed:

| variant | event scale | CSim summary | RTL cosim |
| --- | ---: | --- | --- |
| distilled shared_u4 | 252 events + reset = 253 tx | 1 readout row, joint sign agreement 1, alignment errors 0 | FAIL, OOM at 187/253 |
| scratch shared_u4 | 252 events + reset = 253 tx | 1 readout row, joint sign agreement 1, alignment errors 0 | not run after distilled OOM; moved to smaller scale |

The distilled 252-event HLS log records `Out of memory on request for a fresh 8388608 bytes`, `Total memory consumed so far :1724767044 bytes`, `Simulation engine not responding`, and `C/RTL co-simulation finished: FAIL`. The generated cosim report marks Verilog status as `Fail`.

The smaller 174-event scale then completed for both variants:

| variant | event scale | RTL cosim | block4 valid rows | block4 max abs diff | block4 RMSE | readout rows | alignment errors |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| distilled shared_u4 | 174 events + reset = 175 tx | PASS | 50 | 0.037681575625 | 0.0058845000499 | 0 | 0 |
| scratch shared_u4 | 174 events + reset = 175 tx | PASS | 50 | 0.013252259375 | 0.00258053249258 | 0 | 0 |

This proves the RTL cosim can complete at a reduced scale covering block4 warm-up plus 50 valid block4 outputs. It does not prove the final readout path over a full 252-event window, because readout rows are 0 at event_limit=174.

**Pipeline Wording Correction**

I corrected `n2ew_era/07c2b_tcn_streaming_speed/AUDIT_tcn_streaming_speed.md` to say the top-level `#pragma HLS PIPELINE` attempt was ignored because it conflicted with inner `PIPELINE off` pragmas. The correct conclusion is: pipeline throughput effect remains untested. It should not be read as evidence that pipeline is ineffective for this architecture.

**Artifacts**

| artifact | path |
| --- | --- |
| xsim transaction summary | `results/xsim_transaction_time_stats.csv` |
| xsim per-transaction deltas | `results/xsim_transaction_deltas.csv` |
| harness complexity counts | `results/cosim_harness_complexity_counts.csv` |
| distilled 252 OOM log bundle | `artifacts/distilled_shared_u4_event252_oom/` |
| distilled 174 PASS log bundle | `artifacts/distilled_shared_u4_event174_pass/` |
| scratch 174 PASS log bundle | `artifacts/scratch_shared_u4_event174_pass/` |
| distilled 174 TB summary | `results/stage2_cosim174_distilled_shared_u4_e174_tb_summary.csv` |
| scratch 174 TB summary | `results/stage2_cosim174_scratch_shared_u4_e174_tb_summary.csv` |

**Honest Conclusion**

07C2C resolves the immediate diagnostic question: the original 1008-event TCN cosim scale is practically infeasible under the current xsim/HLS autowrap setup, and simply increasing timeout is not a good closure path. The best-supported root cause is not only the 74.38us/event RTL latency, but the much larger generated cosim interface/harness: many scalar `ap_ovld` ports, output-feedback/readwrite handling, large SVR/UVM topology, and severe xsim memory growth.

The reduced 174-event cosim passes for both scratch and distilled shared_u4, but the full 252-event one-window readout-level cosim remains open, and the 1008-event original requirement remains open. Current evidence tier is `cosim_174event_pass_block4_50valid`, not deployable closure.
