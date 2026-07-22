# Cosim Latency Confirmation

The C/RTL co-simulation report at `n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/sim/report/phc_step_cosim.rpt` reports Verilog `Pass` with latency min/avg/max = `8316 / 8316 / 8316` clock cycles and interval min/avg/max = `8317 / 8317 / 8317` clock cycles.

This matches the csynth-derived latency used in the technical report: `8316` cycles, which is `83.16 us` at a 100 MHz clock. The cosim log also reports `total=25600 mismatches=0`, `max_abs_diff=0.00725275`, and `mean_abs_diff=0.000372553`.

Therefore Section 4.4 may describe the 83.16 us figure as a C/RTL co-simulation-confirmed cycle-accurate latency for the generated Verilog under the tested HLS configuration, not merely a scheduler estimate from C synthesis.

Claim boundary: this confirms RTL simulation latency for the proxy HLS design and testbench. It does not prove board-level timing closure, end-to-end exchange latency, real-world profitability, or FPGA-caused PnL improvement.
