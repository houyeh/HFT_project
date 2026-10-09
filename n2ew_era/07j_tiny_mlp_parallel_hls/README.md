# Tiny MLP Parallel HLS

This phase preserves the 07I Q28.14/Q40.24 arithmetic while exchanging FPGA
resources for lower latency. The first linear-chain candidate exposed a
25.506 ns critical path. The current candidate computes all eight hidden
neurons concurrently and uses explicit balanced adder trees for the 16-input
and 8-input dot products. CSim must remain bit-exact; synthesis and routed P&R
decide whether the candidate fits timing and the 220-DSP budget.

Final result: 25,600-vector CSim and Verilog cosim are bit-exact. The balanced
tree core has 18-cycle latency and 19-cycle interval. Vivado 2025.2 OOC routing
at 100 MHz closes timing with WNS +0.331 ns and TNS 0, giving a cycle-derived
core latency of 0.180 us. See `AUDIT_tiny_mlp_parallel_routed.md` for the
task-value trade-off and verification boundary.
