# V3 Diagonal Stage B Wrapper Integration Report

Status: `stage_b_pass_interface_and_handshake`.

Stage B validates the generated RTL interface, wrapper FSM handshake, and both V3 output write ports. Numerical equivalence is intentionally deferred until Stage C generates model-matched V3 replay expectations; the existing `bram/` expected readouts belong to the 05C4 dense model.

## Stage B report block

```text
[B] rtl_export final file count: 44 .v + 21 .dat
[B] port differences relative to original wrapper:
    - HLS module phc_step -> phc_diag_step
    - state_prev port 1 is enabled and connected
    - readout port 1 address/ce/we/data is added and captured
    - state_next port 1 address/ce/we/data is added and captured
[B] readout/state_next dual-port handling:
    - each write port independently applies ce && we
    - each port writes its own address/data into the shared capture array
    - sticky activity flags expose whether port 1 wrote during replay
[B] clock: 125 MHz input -> MMCM DIVCLK=5, MULT=40, CLKOUT0_DIVIDE=10 -> 100 MHz
[B] XSim: PASS (direct xvlog/xelab/xsim flow)
[B] handshake: current_index=7, start_seen=1, ready_seen=1, done_seen=1
[B] XSim port1 write activity: readout=1, state_next=1
[B] per-inference write coverage: readout=2'b11, state_next=16'hFFFF
[B] same-cycle same-address collision: none
[B] coverage_error=0, collision_error=0
```

## Interface changes

The input side required no structural adapter:

- `x_t` remains a dual-read-port `ap_memory` interface.
- `state_prev` remains a dual-read-port `ap_memory` interface; `PHC_HAS_STATE_PREV_PORT1` is defined in the V3 wrapper.

The V3 HLS core exposes two write ports for each output array, whereas the previous wrapper captured only port 0. The V3 wrapper adds:

```text
readout_address1, readout_ce1, readout_we1, readout_d1
state_next_address1, state_next_ce1, state_next_we1, state_next_d1
```

Both ports are active in RTL simulation. Port 1 is neither left open nor tied off.

Activity alone does not prove complete output coverage, so the wrapper also clears
per-inference write masks at `S_START`, marks each address written by either port,
and checks the masks after the `ap_done` write edge has settled. It separately
latches any same-cycle, same-address dual-port write collision. The eight-vector
Stage B run covered both readout addresses and all 16 state addresses with no
collision.

## Clock generation

The new clock module is `rtl/pynq_z2_clock_100mhz.sv`. It uses the same 125 MHz PYNQ-Z2 `sysclk` source as the existing 90 MHz module:

```text
VCO  = 125 MHz / 5 * 40 = 1000 MHz
Fout = 1000 MHz / 10 = 100 MHz
```

The new board-level top is `rtl/pynq_z2_phc_demo_board_top_v3diag.sv`; the existing board top and 90 MHz clock module were not modified.

## XSim evidence

Vivado `create_project` remains blocked by the known user Tcl Store problem. Stage B therefore reused the established direct simulator flow, which invokes `xvlog`, `xelab`, and `xsim` without IP packaging or Vivado project creation.

The required pass markers are:

```text
V3DIAG_STAGE_B_HANDSHAKE current_index=7 start_seen=1 ready_seen=1 done_seen=1
V3DIAG_STAGE_B_PORT1 readout_write_seen=1 state_next_write_seen=1
V3DIAG_STAGE_B_COVERAGE coverage_error=0 collision_error=0
V3DIAG_STAGE_B_PASS
```

Evidence files:

- `logs/xsim_v3diag_stage_b_xvlog.log`
- `logs/xsim_v3diag_stage_b_xelab.log`
- `logs/xsim_v3diag_stage_b_xsim.log`
- `scripts/run_xsim_v3diag_stage_b_direct.ps1`
- `sim/tb_pynq_z2_phc_demo_wrapper_v3diag_stage_b.sv`

The simulator warning that the memory file has more words than the Stage B testbench's eight-row memory is expected: the interface-only test intentionally consumes the first 8 rows of the unchanged 4,096-row input file.

## New files

- `rtl/pynq_z2_phc_demo_wrapper_v3diag.sv`
- `rtl/pynq_z2_phc_demo_board_top_v3diag.sv`
- `rtl/pynq_z2_clock_100mhz.sv`
- `sim/tb_pynq_z2_phc_demo_wrapper_v3diag_stage_b.sv`
- `tcl/run_xsim_demo_wrapper_v3diag_stage_b.tcl`
- `scripts/run_xsim_v3diag_stage_b_direct.ps1`

No original wrapper, board top, clock module, HLS source, model checkpoint, or existing BRAM file was modified.
