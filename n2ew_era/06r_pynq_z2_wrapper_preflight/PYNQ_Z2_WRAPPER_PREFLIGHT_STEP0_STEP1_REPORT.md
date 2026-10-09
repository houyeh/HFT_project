# PYNQ-Z2 Wrapper Preflight: Step 0 and Step 1

Status: `step0_step1_complete_waiting_for_confirmation`

This report covers only the requested pre-wrapper audit gates:

- Step 0: compare the routed RTL top-level port lists for the 06F frozen v8 baseline and the 06L unroll16 routed candidate.
- Step 1: inspect the 06F frozen v8 HLS RTL interface contract and state-feedback requirements.

No wrapper RTL was written. No XSim, P&R, JTAG, or board work was run.

## Source Artifacts

06F frozen v8 RTL:

`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/impl/verilog/phc_step.v`

06F interface report:

`n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/syn/report/phc_step_csynth.rpt`

06L unroll16 90 MHz routed RTL:

`n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/hls_component/hls_component/hls/impl/verilog/phc_step.v`

06L unroll16 source provenance:

`n2ew_era/06l_phc_lite_pipeline_budget_sweep/variants/factor_16/hls_src/phc_step_hls.cpp`

Audit script:

`n2ew_era/06r_pynq_z2_wrapper_preflight/00_rtl_interface_audit.py`

Generated audit tables:

- `n2ew_era/06r_pynq_z2_wrapper_preflight/results/06f_v8_frozen_100mhz_port_list.csv`
- `n2ew_era/06r_pynq_z2_wrapper_preflight/results/06l_unroll16_90mhz_port_list.csv`
- `n2ew_era/06r_pynq_z2_wrapper_preflight/results/rtl_port_diff.csv`
- `n2ew_era/06r_pynq_z2_wrapper_preflight/results/rtl_core_generation_info.csv`

## Step 0: Top-Level Port Comparison

Result: `not_identical`

The two routed RTL modules share the same logical HLS top function name, handshake signals, output memory interface, and first read port for each input array. They do not expose an identical block-level Verilog port list.

### Common Ports

These ports match in name, direction, and width:

| Port | Direction | Width |
|---|---:|---:|
| ap_clk | input | 1 |
| ap_rst | input | 1 |
| ap_start | input | 1 |
| ap_done | output | 1 |
| ap_idle | output | 1 |
| ap_ready | output | 1 |
| x_t_address0 | output | 4 |
| x_t_ce0 | output | 1 |
| x_t_q0 | input | 24 |
| state_prev_address0 | output | 4 |
| state_prev_ce0 | output | 1 |
| state_prev_q0 | input | 24 |
| readout_address0 | output | 1 |
| readout_ce0 | output | 1 |
| readout_we0 | output | 1 |
| readout_d0 | output | 24 |
| state_next_address0 | output | 4 |
| state_next_ce0 | output | 1 |
| state_next_we0 | output | 1 |
| state_next_d0 | output | 24 |

### 06L-Only Extra Ports

The 06L unroll16 routed RTL has a second read port for both input arrays:

| Port | Direction | Width | Present in 06F? |
|---|---:|---:|---:|
| x_t_address1 | output | 4 | no |
| x_t_ce1 | output | 1 | no |
| x_t_q1 | input | 24 | no |
| state_prev_address1 | output | 4 | no |
| state_prev_ce1 | output | 1 | no |
| state_prev_q1 | input | 24 | no |

The likely cause is the factor-16 unroll in the 06L source path, which increases read parallelism for `x_t` and `state_prev` and causes HLS to expose dual read memory ports. The 06F frozen baseline exposes only one read port per input array.

### Wrapper Sharing Impact

A single literal wrapper instantiation cannot be shared between the two routed RTL files unchanged, because the 06L module has six additional ports.

A shared wrapper architecture is still feasible if the wrapper is parameterized or split into small variant adapters:

- 06F adapter: one read port for `x_t` and one read port for `state_prev`.
- 06L adapter: two read ports for `x_t` and two read ports for `state_prev`.
- Common outer controller: same `ap_ctrl_hs` sequencing, state feedback policy, readout checking, BRAM-vector player, and pass/fail latch.

Alternative: regenerate one of the HLS candidates with explicit interface pragmas to force a common memory interface. This would need a new HLS synthesis and verification pass, so it should be treated as a design decision, not assumed.

## Step 1: 06F Frozen V8 Interface Contract

### Control Interface

The 06F csynth interface table reports `ap_ctrl_hs` for:

| Port | Direction | Width | Protocol |
|---|---:|---:|---:|
| ap_clk | input | 1 | ap_ctrl_hs |
| ap_rst | input | 1 | ap_ctrl_hs |
| ap_start | input | 1 | ap_ctrl_hs |
| ap_done | output | 1 | ap_ctrl_hs |
| ap_idle | output | 1 | ap_ctrl_hs |
| ap_ready | output | 1 | ap_ctrl_hs |

Evidence: `phc_step_csynth.rpt`, interface table lines 502-507.

Expected wrapper sequencing:

1. Present stable `x_t` and `state_prev` memory responses.
2. Assert `ap_start` when the core is idle/ready.
3. Wait for `ap_done`.
4. Latch `readout` and `state_next`.
5. Commit `state_next` as the next invocation's `state_prev`.

### Data Interfaces

All 06F data arrays use `ap_memory`, not `ap_none`, `ap_vld`, or `ap_hs`.

| Logical Array | Port | Direction | Width | Protocol |
|---|---|---:|---:|---:|
| x_t | x_t_address0 | output | 4 | ap_memory |
| x_t | x_t_ce0 | output | 1 | ap_memory |
| x_t | x_t_q0 | input | 24 | ap_memory |
| state_prev | state_prev_address0 | output | 4 | ap_memory |
| state_prev | state_prev_ce0 | output | 1 | ap_memory |
| state_prev | state_prev_q0 | input | 24 | ap_memory |
| readout | readout_address0 | output | 1 | ap_memory |
| readout | readout_ce0 | output | 1 | ap_memory |
| readout | readout_we0 | output | 1 | ap_memory |
| readout | readout_d0 | output | 24 | ap_memory |
| state_next | state_next_address0 | output | 4 | ap_memory |
| state_next | state_next_ce0 | output | 1 | ap_memory |
| state_next | state_next_we0 | output | 1 | ap_memory |
| state_next | state_next_d0 | output | 24 | ap_memory |

Evidence: `phc_step_csynth.rpt`, interface table lines 508-521.

The 24-bit data width matches the manual HLS fixed-point contract `ap_fixed<24,12>`.

### Reset Behavior

The routed 06F RTL exposes `ap_rst` as a 1-bit input. The FSM reset branch uses:

`if (ap_rst == 1'b1) begin ap_CS_fsm <= ap_ST_fsm_state1; end`

This appears inside clocked `always @(posedge ap_clk)` logic, so the generated control reset is active-high and synchronous in the inspected RTL.

Evidence: `phc_step.v`, port declaration lines 294-313 and reset logic beginning around line 2728.

### State Feedback

The HLS core is a stateless step function at the RTL boundary:

```cpp
void phc_step(
    const phc_fixed_t x_t[16],
    const phc_fixed_t state_prev[16],
    phc_fixed_t readout[2],
    phc_fixed_t state_next[16]
);
```

It does not internally preserve recurrent state across top-level invocations. The board wrapper must hold the 16-word recurrent state externally.

Required wrapper policy:

- On reset or session boundary, initialize the state buffer to all zeros.
- For each input event, serve that buffer as `state_prev`.
- After `ap_done`, copy the written `state_next` buffer into the state buffer before launching the next event.
- Do not overwrite the current state buffer while the core is still reading `state_prev`.

This matches the Python test-vector generator:

```python
state = torch.zeros(n, 16, dtype=torch.float32)
...
state_prev = state.clone()
x_cat = torch.cat([x_t, state], dim=-1)
out_step = step_model(x_cat)
readout_t, state = step_model.split_output(out_step)
```

Evidence: `n2ew_era/06f_phc_lite_manual_hls/01_generate_test_vectors.py`, lines 69 and 77-80.

## Pending Confirmation Before Wrapper Work

The wrapper can proceed in one of two directions:

1. `variant_adapter_wrapper`: one common outer controller plus variant-specific memory adapters for 06F single-read and 06L dual-read input ports.
2. `force_common_hls_interface`: regenerate one candidate with explicit HLS interface constraints so both routed RTL modules expose the same block-level port list.

No wrapper should be written until this choice is confirmed.

## Not Yet Performed

Per the Step 1 stop point, these items are intentionally not done in this report:

- PYNQ-Z2 clock-source selection or reference-manual finalization.
- MMCM/PLL parameter design.
- BRAM vector packing or `.coe` generation.
- XSim positive/negative full-system simulation.
- Any P&R, bitstream, JTAG, or physical board action.

