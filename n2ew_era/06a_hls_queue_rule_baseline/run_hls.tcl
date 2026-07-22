# run_hls.tcl — Vitis HLS synthesis script for queue_rule_action
# Target: PYNQ-Z2 / Zynq-7020  (xc7z020clg400-1)
# Usage: vitis_hls -f run_hls.tcl   (run from 06a directory)

# ── 1. Create project ──────────────────────────────────────────────────────────
open_project queue_rule_proj
set_top queue_rule_action

# ── 2. Add source & testbench files ───────────────────────────────────────────
add_files queue_rule_hls.cpp
add_files queue_rule_hls.hpp
add_files -tb queue_rule_testbench.cpp

# ── 3. Create solution (target clock 10 ns = 100 MHz) ─────────────────────────
open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

# ── 4. C Simulation (functional check) ────────────────────────────────────────
# Pass input/output CSV paths as arguments to the testbench
set_directive_top -name queue_rule_action "queue_rule_hls.cpp/queue_rule_action"
csim_design -argv "vectors/queue_rule_test_vectors.csv results/csim_out.csv"

# ── 5. C Synthesis ────────────────────────────────────────────────────────────
csynth_design

# ── 6. RTL Co-simulation (optional, slower) ───────────────────────────────────
# cosim_design -argv "vectors/test_vectors.csv results/cosim_out.csv"

# ── 7. Export IP (optional) ───────────────────────────────────────────────────
# export_design -format ip_catalog -output ./queue_rule_ip

exit
