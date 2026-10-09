# Tiny MLP Parallel Core: PYNQ-Z2 Board Validation

This phase wraps the 07J fully parallel Tiny MLP without modifying its HLS
source or generated RTL. It replays the first 4,096 frozen 07J test vectors and
compares all three 40-bit logits plus the 2-bit action exactly.

Verification order:

1. `python 00_generate_board_vectors.py`
2. Vivado batch: `-source tcl/run_xsim.tcl`
3. Vivado batch: `-source tcl/run_board_pnr.tcl`
4. With the PYNQ-Z2 connected: `-source tcl/program_and_read.tcl`

LED mapping: LD0 done, LD1 exact pass, LD2 fail, LD3 heartbeat. The reported
core latency is cycle-derived from the routed 100 MHz clock. The physical-board
run completed all 4,096 rows exactly; see `AUDIT_tiny_mlp_parallel_on_board.md`.
