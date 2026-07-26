status: mixed

# AUDIT: 06L unroll-16 routed P&R timing

Updated: 2026-07-26 Asia/Taipei. This update incorporates the 2026-07-25
100/95 MHz routed attempts plus the 90 MHz fallback routed run from this pass.

## Scope and Guardrails

This audit promotes the 06L unroll-16 PHC-lite latency claim only as far as
the routed evidence supports.

- Frozen 06F v8 artifacts were not modified.
- PHC-lite HLS C source and generated RTL were not edited for the P&R closure
  attempts. The 90 MHz workspace is a copied implementation workspace with the
  clock relaxed to 11.111 ns; the rerun also copies the original HLS `.dat`
  ROM initialization files so the direct Vivado flow has the same RTL data
  inputs as the 100/95 MHz direct runs.
- Interface, top module, target part, and unroll-16 source configuration remain
  aligned with 06L/06F comparability requirements. The implementation attempts
  change only Vivado implementation strategy/directives or clock period.
- No NEW5 data was loaded or used.

## Verified Before P&R

| Item | 06F v8 frozen baseline | 06L unroll-16 copied workspace |
|---|---|---|
| HLS latency | 8316 cycles / 83.160 us<br>`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt` | 2316 cycles / 23.160 us at 100 MHz<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/variants/factor_16/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt` |
| HLS interval | 8317 cycles<br>`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt` | 2317 cycles<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/variants/factor_16/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt` |
| HLS estimated clock | 7.147 ns<br>`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt` | 8.223 ns<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/variants/factor_16/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt` |
| RTL cosim | Verilog PASS, existing 06F evidence<br>`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/reports/hls_cosim.rpt` | Verilog PASS, 2316-cycle latency, `total=25600 mismatches=0`<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/reports/hls_cosim.rpt`<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/logs/hls_run_cosim.log` |

The 2316-cycle / 23.160 us number is therefore valid as a csynth plus RTL
cosim result. It is not a routed 100 MHz timing-met result.

## Routed Timing Attempts

All WNS/TNS/failing-endpoint numbers below come from routed timing summary
reports generated after `route_design`. The direct Vivado scripts use
`report_timing_summary -routable_nets -report_unconstrained` for the routed
summary.

| Design / attempt | Clock | Implementation setting | WNS | TNS | Failing endpoints | Latency interpretation | Routed evidence |
|---|---:|---|---:|---:|---:|---|---|
| 06F v8 frozen baseline | 100.000 MHz / 10.000 ns | Vitis/Vivado packaged 06F v8 implementation | +0.020 ns<br>`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/impl/verilog/project.runs/impl_1/bd_0_wrapper_timing_summary_routed.rpt` | 0.000 ns<br>`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/impl/verilog/project.runs/impl_1/bd_0_wrapper_timing_summary_routed.rpt` | 0<br>`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/impl/verilog/project.runs/impl_1/bd_0_wrapper_timing_summary_routed.rpt` | 8316 cycles / 83.160 us, routed timing met | Timing: `n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/impl/verilog/project.runs/impl_1/bd_0_wrapper_timing_summary_routed.rpt`<br>Util: `n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/reports/hls_impl_pnr.rpt` |
| 06L unroll-16 100 MHz baseline | 100.000 MHz / 10.000 ns | Direct Vivado non-project flow from HLS RTL; default `place_design`, `phys_opt_design`, `route_design` | -0.235 ns<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports/phc_step_timing_summary_routed.rpt` | -13.737 ns<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports/phc_step_timing_summary_routed.rpt` | 144<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports/phc_step_timing_summary_routed.rpt` | 2316 cycles / 23.160 us remains csynth/cosim only; routed timing failed | Timing: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports/phc_step_timing_summary_routed.rpt`<br>Util: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports/phc_step_utilization_routed.rpt` |
| 06L unroll-16 100 MHz aggressive physical optimization | 100.000 MHz / 10.000 ns | `place_design -directive ExtraNetDelay_high`, `phys_opt_design -directive AggressiveExplore`, `route_design -directive AggressiveExplore` | -0.159 ns<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_aggressive_explore_100mhz/phc_step_timing_summary_routed.rpt` | -13.956 ns<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_aggressive_explore_100mhz/phc_step_timing_summary_routed.rpt` | 96<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_aggressive_explore_100mhz/phc_step_timing_summary_routed.rpt` | Best 100 MHz routed attempt, but still timing failed | Timing: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_aggressive_explore_100mhz/phc_step_timing_summary_routed.rpt`<br>Util: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_aggressive_explore_100mhz/phc_step_utilization_routed.rpt` |
| 06L unroll-16 100 MHz fanout/replication attempt | 100.000 MHz / 10.000 ns | Attempted `REGISTER_DUPLICATION` on `ap_CS_fsm_reg[9]`; Vivado reported the property does not exist. `MAX_FANOUT=16` plus forced replication created `ap_CS_fsm_reg[9]_replica_9`, but timing worsened. | -0.383 ns<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_fanout_dup_100mhz/phc_step_timing_summary_routed.rpt` | -53.903 ns<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_fanout_dup_100mhz/phc_step_timing_summary_routed.rpt` | 240<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_fanout_dup_100mhz/phc_step_timing_summary_routed.rpt` | 2316 cycles / 23.160 us is not routed timing met | Timing: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_fanout_dup_100mhz/phc_step_timing_summary_routed.rpt`<br>Util: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_fanout_dup_100mhz/phc_step_utilization_routed.rpt`<br>Log: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/results/vivado_impl_strategy_fanout_dup_100mhz_retry2.log` |
| 06L unroll-16 95 MHz baseline | 95.003 MHz / 10.526 ns | Direct Vivado non-project flow, relaxed clock only | -0.033 ns<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_95mhz/phc_step_timing_summary_routed.rpt` | -1.578 ns<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_95mhz/phc_step_timing_summary_routed.rpt` | 48<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_95mhz/phc_step_timing_summary_routed.rpt` | 2316 cycles / 24.378 us at this period, but not timing met | Timing: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_95mhz/phc_step_timing_summary_routed.rpt`<br>Util: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_95mhz/phc_step_utilization_routed.rpt` |
| 06L unroll-16 90 MHz baseline | 90.001 MHz / 11.111 ns | Separate 90 MHz copied workspace; direct Vivado non-project flow, relaxed clock only. Clean rerun includes 106 copied `.dat` ROM init files and has no `$readmem` missing-file warning. | +0.151 ns<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_90mhz/phc_step_timing_summary_routed.rpt` | 0.000 ns<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_90mhz/phc_step_timing_summary_routed.rpt` | 0<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_90mhz/phc_step_timing_summary_routed.rpt` | 2316 cycles / 25.733 us, routed timing met | Timing: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_90mhz/phc_step_timing_summary_routed.rpt`<br>Util: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_90mhz/phc_step_utilization_routed.rpt`<br>Route: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_90mhz/phc_step_route_status.rpt` |

## Routed Utilization

06F values are from Vitis HLS packaged P&R resource summary. 06L values are
from Vivado routed utilization reports in the direct out-of-context flow; those
reports label BRAM as `Block RAM Tile`.

| Design / attempt | LUT | FF | DSP | BRAM |
|---|---:|---:|---:|---:|
| 06F v8 frozen baseline | 7555<br>`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/reports/hls_impl_pnr.rpt` | 8217<br>`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/reports/hls_impl_pnr.rpt` | 47<br>`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/reports/hls_impl_pnr.rpt` | 6<br>`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/reports/hls_impl_pnr.rpt` |
| 06L 100 MHz baseline | 14818<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports/phc_step_utilization_routed.rpt` | 19809<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports/phc_step_utilization_routed.rpt` | 125<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports/phc_step_utilization_routed.rpt` | 0 Block RAM Tile<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports/phc_step_utilization_routed.rpt` |
| 06L 100 MHz aggressive physical optimization | 14870<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_aggressive_explore_100mhz/phc_step_utilization_routed.rpt` | 19836<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_aggressive_explore_100mhz/phc_step_utilization_routed.rpt` | 125<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_aggressive_explore_100mhz/phc_step_utilization_routed.rpt` | 0 Block RAM Tile<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_aggressive_explore_100mhz/phc_step_utilization_routed.rpt` |
| 06L 100 MHz fanout/replication attempt | 14826<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_fanout_dup_100mhz/phc_step_utilization_routed.rpt` | 19834<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_fanout_dup_100mhz/phc_step_utilization_routed.rpt` | 125<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_fanout_dup_100mhz/phc_step_utilization_routed.rpt` | 0 Block RAM Tile<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_fanout_dup_100mhz/phc_step_utilization_routed.rpt` |
| 06L 95 MHz baseline | 14822<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_95mhz/phc_step_utilization_routed.rpt` | 19801<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_95mhz/phc_step_utilization_routed.rpt` | 125<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_95mhz/phc_step_utilization_routed.rpt` | 0 Block RAM Tile<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_95mhz/phc_step_utilization_routed.rpt` |
| 06L 90 MHz baseline | 14826<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_90mhz/phc_step_utilization_routed.rpt` | 19805<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_90mhz/phc_step_utilization_routed.rpt` | 125<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_90mhz/phc_step_utilization_routed.rpt` | 0 Block RAM Tile<br>`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_90mhz/phc_step_utilization_routed.rpt` |

## Critical Path Diagnosis From Failed 100 MHz Baseline

The 100 MHz baseline failure was dominated by one repeated high-fanout path
shape, not by 144 independent unrelated bottlenecks:

- Source:
  `grp_phc_step_Outline_VITIS_LOOP_127_11_fu_2060/ap_CS_fsm_reg[9]`
- Destination family:
  `mul_24s_40s_64_2_1_U289/buff0_reg__0/PCIN[x]`
- Data path delay: 8.751 ns
- Logic delay / route delay: 5.033 ns / 3.718 ns
- Logic levels: 5 (`DSP48E1=1`, `LUT5=2`, `LUT6=2`)
- Evidence:
  `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports/phc_step_timing_paths_routed.rpt`

The implementation-only fanout attempt did replicate the FSM net, but did not
solve timing:

- `REGISTER_DUPLICATION` could not be applied because Vivado reported the
  property does not exist.
- `MAX_FANOUT=16` was accepted, and physical optimization created
  `ap_CS_fsm_reg[9]_replica_9`.
- Routed timing worsened to WNS=-0.383 ns / TNS=-53.903 ns.
- Evidence log:
  `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/results/vivado_impl_strategy_fanout_dup_100mhz_retry2.log`

## Scripts and Logs

| Run | Script / log evidence |
|---|---|
| 100 MHz baseline direct P&R | Script: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/run_direct_phc_step_pnr.tcl`<br>Log: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/results/vivado_direct_phc_step_pnr_unroll16_100mhz_tcllibpath.log` |
| 100 MHz aggressive physical optimization | Script: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/run_direct_phc_step_impl_strategy_from_synth.tcl`<br>Log: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/results/vivado_impl_strategy_aggressive_explore_100mhz.log` |
| 100 MHz fanout/replication attempt | Script: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/run_direct_phc_step_impl_strategy_from_synth.tcl`<br>Log: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/results/vivado_impl_strategy_fanout_dup_100mhz_retry2.log` |
| 95 MHz baseline | Script: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/hls_component/hls_component/hls/impl/verilog/run_direct_phc_step_pnr_custom_clock.tcl`<br>Log: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_100mhz/results/vivado_direct_phc_step_pnr_95mhz_retry.log` |
| 90 MHz baseline | Script: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/hls_component/hls_component/hls/impl/verilog/run_direct_phc_step_pnr_custom_clock.tcl`<br>Clean log: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/results/vivado_direct_phc_step_pnr_90mhz_with_dat.log`<br>Exit code: `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/results/vivado_direct_phc_step_pnr_90mhz_with_dat.exitcode.txt` |

## Honest Conclusion

The unroll-16 design has a real routed timing-met result, but not at 100 MHz.
The strongest supported claim is now:

`2316 cycles / 25.733 us at 90.001 MHz, routed timing met with WNS=+0.151 ns,
TNS=0.000 ns, and 0 failing endpoints.`

The 23.160 us number remains a csynth plus RTL cosim result at 100 MHz. It
must not be described as post-route timing-met evidence, because all three
100 MHz routed attempts still fail timing. The 95 MHz baseline is a useful
near miss (WNS=-0.033 ns), but it also must not be claimed as timing met.

Compared with the frozen 06F v8 routed baseline of 8316 cycles / 83.160 us at
100 MHz, the 90 MHz unroll-16 routed result is still materially lower latency:
25.733 us versus 83.160 us. The cost is higher routed resource use, especially
DSPs (125 versus 47) and LUT/FF usage (about 14.8k/19.8k versus 7.6k/8.2k).
