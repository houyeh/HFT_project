# V3 Diagonal Stage D Board-Level P&R Report

Status: `stage_d_pass_100mhz_baseline_routed_bitstream_generated`.

The V3 diagonal replay wrapper met board-level timing on the first 100 MHz
baseline attempt. No aggressive, fanout, low-jitter-clock, or 90 MHz fallback
attempt was run because the prescribed decision tree stops at the first routed
design with non-negative WNS and zero TNS.

## Stage D report block

```text
[D] attempt sequence:
    100 MHz / current clock / baseline: WNS=+0.148 ns, TNS=0.000 ns,
                                        0 setup failing endpoints, met
    100 MHz / aggressive: not run (baseline met timing)
    100 MHz / fanout: not run (baseline met timing)
    100 MHz / low-jitter clock variants: not run (baseline met timing)
    90 MHz fallback: not run (baseline met timing)
[D] adopted: 100 MHz baseline, WNS=+0.148 ns, TNS=0.000 ns
[D] hold timing: WHS=+0.014 ns, THS=0.000 ns, 0 failing endpoints
[D] derived latency: 530 RTL-cosim cycles / 100 MHz = 5.300 us
[D] board-level routed resources:
    LUT=12,285 / 53,200 (23.09%)
    FF=13,772 / 106,400 (12.94%)
    DSP=196 / 220 (89.09%)
    BRAM_18K=57 / 280 (20.36%; 28 RAMB36E1 + 1 RAMB18E1)
[D] instantiated replay BRAM: input+expected-readout=yes, expected-state=no
[D] clock: 125 MHz input, DIVCLK=5, MULT=40, CLKOUT0_DIVIDE=10,
           VCO=1000 MHz, output=100 MHz
[D] bitstream: results_v3_diag/pnr_100mhz_baseline/
               phc_v3diag_demo_100mhz_baseline.bit
[D] bitstream SHA-256:
    A4661EB5D94481596040E6586C12AB5E1F11246F7A3A698CA703BE408BE61085
```

## Required pre-P&R simulation gate

The board build was allowed to start only after full 4,096-row positive and
negative XSim runs passed. The positive run used the built-in one-LSB gate and
also dumped every readout for an exact comparison. Behavioral RTL was exactly
equal to the V3 sequential CSim reference in this simulation:

```text
V3DIAG_WRAPPER_DONE current_index=4095 pass=1 fail=0
V3DIAG_MISMATCH_STATS count=0 max_readout_lsb_diff=0 readout_tol_lsb=1
V3DIAG_COVERAGE port1_readout=1 port1_state=1 coverage_error=0 collision_error=0
V3DIAG_POSITIVE_TEST_PASS
```

The negative run changed the expected readout by 16 LSB at index 17 and the
comparator caught exactly that injected error:

```text
V3DIAG_WRAPPER_DONE current_index=4095 pass=0 fail=1 first_mismatch_index=17
V3DIAG_MISMATCH_STATS count=1 max_readout_lsb_diff=16 readout_tol_lsb=1
V3DIAG_COVERAGE port1_readout=1 port1_state=1 coverage_error=0 collision_error=0
V3DIAG_NEGATIVE_TEST_PASS
```

Evidence:

- `logs/xsim_v3diag_full_positive_xsim.log`
- `logs/xsim_v3diag_full_negative_xsim.log`
- `results_v3_diag/v3diag_wrapper_readout_dump_xsim.csv`
- `results_v3_diag/XSIM_POSITIVE_AND_NEGATIVE_PASS.marker`

## Routed signoff evidence

Vivado 2025.2 reported that all user timing constraints were met. The generated
reports are:

- `results_v3_diag/pnr_100mhz_baseline/v3diag_board_timing_summary_routed.rpt`
- `results_v3_diag/pnr_100mhz_baseline/v3diag_board_timing_paths_routed.rpt`
- `results_v3_diag/pnr_100mhz_baseline/v3diag_board_utilization_routed.rpt`
- `results_v3_diag/pnr_100mhz_baseline/v3diag_board_route_status.rpt`
- `results_v3_diag/pnr_100mhz_baseline/v3diag_board_drc_routed.rpt`
- `results_v3_diag/pnr_100mhz_baseline/v3diag_board_power_routed.rpt`
- `results_v3_diag/pnr_100mhz_baseline/v3diag_board_routed.dcp`
- `results_v3_diag/pnr_100mhz_baseline/vivado_stage_d_baseline.log`

The worst setup path is internal to the V3 HLS core, from
`ap_CS_fsm_reg[3]` to a DSP48E1 `PCIN` input. Its 8.211 ns data path comprises
4.740 ns logic (57.728%) and 3.471 ns routing (42.272%) against a 10.000 ns
requirement. This is the same general FSM-to-DSP path class already identified
in earlier unrolled designs, but it closes here with 0.148 ns slack.

The vectorless power estimate is 0.533 W total on-chip power (0.419 W dynamic,
0.114 W device static), with `Medium` confidence. Vivado warned that inferred
reset switching activity may make this estimate inaccurate; it is not an
on-board power measurement and is not used as a measured result.

## Scope and verification tier

Verification tier is `routed`: the 5.300 us value is derived from the 530-cycle
RTL co-simulation latency and the 100 MHz routed timing closure. It is not an
independent on-silicon latency measurement.

The generated bitstream does not instantiate the optional ILA core. The board
top preserves `MARK_DEBUG` signals and the wrapper exposes readout and handshake
signals, but `PHC_ENABLE_ILA` was not defined for this build. Therefore this
bitstream supports the LED/comparator replay gate but cannot by itself provide
an ILA capture of per-event readout values. Stage E has not programmed the board
or run the required on-board 4,096-event comparison.
