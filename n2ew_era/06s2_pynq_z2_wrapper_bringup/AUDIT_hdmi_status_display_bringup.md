# PYNQ-Z2 HDMI Status Display Bring-up

status: `routed_pnr_met_jtag_programmed_visual_observation_pending`

## Scope Boundary

This audit adds a new HDMI demo display path inside `n2ew_era/06s2_pynq_z2_wrapper_bringup/` only.

- Did not modify the verified LED-only board top: `rtl/pynq_z2_phc_demo_board_top.sv`.
- Did not modify the PHC HLS core, replay vectors, BRAM payload, model weights, or NEW1-NEW5 data.
- Added a sibling top: `rtl/pynq_z2_phc_demo_board_top_hdmi.sv`.
- LED behavior is preserved in the HDMI top: `led[0]=done`, `led[1]=pass`, `led[2]=fail`, `led[3]=heartbeat`.
- HDMI is intentionally simple: one full-screen solid color, no text, no numbers, no per-event rendering.

References used:

- Digilent `rgb2dvi` IP: https://github.com/Digilent/vivado-library/tree/master/ip/rgb2dvi
- PYNQ-Z2 video reference direction: https://github.com/xflorentw/Pynq-Z2-Video

## Implementation Summary

Added a local copy of the Digilent `rgb2dvi` RTL under `third_party/digilent_rgb2dvi/src/`, then wrapped it with `rtl/rgb2dvi_fixed.vhd` using external pixel and serializer clocks.

The HDMI timing block is `640x480` style:

| item | value |
|---|---:|
| Pixel clock | 25.0 MHz |
| Serializer clock | 125.0 MHz |
| Active video | 640 x 480 |
| Total timing | 800 x 525 |
| Sync polarity | negative hsync / negative vsync |

Note: the first fixed-color P&R attempt used 25.2/126 MHz with `CLKFBOUT_MULT_F=40.320`; Vivado bitgen rejected that MMCM setting because the fractional multiplier was not on a legal 7-series granularity. The working version uses legal 25.0/125.0 MHz clocks from `CLKFBOUT_MULT_F=40.000`.

## Stage 1: HDMI-Only Fixed Color

Top: `rtl/pynq_z2_hdmi_fixed_color_top.sv`

Purpose: verify the clock/timing/rgb2dvi/TMDS path without PHC logic.

| metric | value |
|---|---:|
| Verification tier | `routed_pnr_met_bitgen_jtag_programmed_visual_pending` |
| Bitstream | `results/hdmi_fixed_color.bit` |
| Bitstream size | 4,045,704 bytes |
| Setup WNS | +35.709 ns |
| Setup TNS | 0.000 ns |
| Setup failing endpoints | 0 |
| Hold WHS | +0.040 ns |
| LUT | 38 / 53,200 = 0.07% |
| FF | 73 / 106,400 = 0.07% |
| DSP | 0 / 220 = 0.00% |
| Block RAM Tile | 0 / 140 = 0.00% |
| IOB | 15 / 125 = 12.00% |
| OSERDES | 8 |
| MMCM | 1 / 4 = 25.00% |

JTAG programming succeeded on `xc7z020_1` with `PYNQ_Z2_HDMI_PROGRAM_STATUS: programmed`. I did not visually observe the attached HDMI monitor, so display color is not claimed as visually confirmed.

## Stage 2: PHC Demo HDMI Status Top

Top: `rtl/pynq_z2_phc_demo_board_top_hdmi.sv`

Color mapping:

| demo state | HDMI color |
|---|---|
| reset / not started | blue |
| replay running | yellow |
| replay done and comparator pass | green |
| replay done and comparator fail | red |

PHC replay status is generated in the 90 MHz replay clock domain and synchronized into the 25 MHz pixel domain. The final P&R Tcl applies an asynchronous clock-group constraint between `clk90_raw` and `clk_pixel_raw`; before this fix, Vivado incorrectly timed the status synchronizer as a related-clock path and reported a false inter-clock violation.

Final routed result:

| metric | value |
|---|---:|
| Verification tier | `routed_pnr_met_bitgen_jtag_programmed_visual_pending` |
| Bitstream | `results/phc_demo_wrapper_hdmi.bit` |
| Bitstream size | 4,045,707 bytes |
| Setup WNS | +0.273 ns |
| Setup TNS | 0.000 ns |
| Setup failing endpoints | 0 |
| Hold WHS | +0.008 ns |
| Hold THS | 0.000 ns |
| Hold failing endpoints | 0 |
| PHC clock | 90 MHz |
| HDMI pixel clock | 25 MHz |
| HDMI serializer clock | 125 MHz |
| LUT | 15,747 / 53,200 = 29.60% |
| FF | 20,354 / 106,400 = 19.13% |
| DSP | 131 / 220 = 59.55% |
| Block RAM Tile | 28.5 / 140 = 20.36% |
| IOB | 15 / 125 = 12.00% |
| OSERDES | 8 |
| OBUFDS | 4 |
| BUFG | 5 / 32 = 15.63% |
| MMCM | 2 / 4 = 50.00% |

JTAG programming succeeded on `xc7z020_1`; final board state after this task is programmed with `results/phc_demo_wrapper_hdmi.bit`.

## Delta vs LED-Only Top

LED-only baseline: `rtl/pynq_z2_phc_demo_board_top.sv`, `results/phc_demo_wrapper.bit`.

| metric | LED-only | PHC + HDMI | delta |
|---|---:|---:|---:|
| Setup WNS | +0.198 ns | +0.273 ns | +0.075 ns |
| Setup TNS | 0.000 ns | 0.000 ns | 0.000 ns |
| Failing endpoints | 0 | 0 | 0 |
| LUT | 15,693 | 15,747 | +54 |
| FF | 20,285 | 20,354 | +69 |
| DSP | 131 | 131 | 0 |
| Block RAM Tile | 28.5 | 28.5 | 0 |
| IOB | 7 | 15 | +8 |
| BUFG | 2 | 5 | +3 |
| MMCM | 1 | 2 | +1 |

Interpretation: the HDMI status display adds the expected TMDS I/O, one HDMI MMCM, and a small amount of timing/color logic. It does not change PHC DSP or BRAM use.

## Hardware Evidence

Hardware detection:

```text
HW_DEVICES_BEGIN
arm_dap_0
xc7z020_1
HW_DEVICES_END
PYNQ_Z2_DETECTION_STATUS: hw_device_present
```

Fixed-color HDMI bitstream programming:

```text
PYNQ_Z2_HDMI_PROGRAM_DEVICE: xc7z020_1
PYNQ_Z2_HDMI_PROGRAM_BITSTREAM: .../results/hdmi_fixed_color.bit
PYNQ_Z2_HDMI_PROGRAM_STATUS: programmed
```

Final PHC HDMI status bitstream programming:

```text
PYNQ_Z2_HDMI_PROGRAM_DEVICE: xc7z020_1
PYNQ_Z2_HDMI_PROGRAM_BITSTREAM: .../results/phc_demo_wrapper_hdmi.bit
PYNQ_Z2_HDMI_PROGRAM_STATUS: programmed
```

## Repro Commands

Fixed-color HDMI-only P&R:

```powershell
& 'C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat' -mode batch -source n2ew_era\06s2_pynq_z2_wrapper_bringup\tcl\run_pnr_hdmi_variant.tcl -tclargs fixed
```

PHC HDMI status P&R:

```powershell
$env:PHC_CORE_RTL_DIR='C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06s2_pynq_z2_wrapper_bringup\hls_seed2_90_replay_component\hls_component\hls_component_seq\hls\sim\verilog'
& 'C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat' -mode batch -source n2ew_era\06s2_pynq_z2_wrapper_bringup\tcl\run_pnr_hdmi_variant.tcl -tclargs phc
```

Program final HDMI demo bitstream:

```powershell
n2ew_era\06s2_pynq_z2_wrapper_bringup\scripts\program_pynq_z2_hdmi_demo.cmd
```

Program fixed-color bitstream instead:

```powershell
$env:PHC_06S2_PROGRAM_BIT_FILE='C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06s2_pynq_z2_wrapper_bringup\results\hdmi_fixed_color.bit'
n2ew_era\06s2_pynq_z2_wrapper_bringup\scripts\program_pynq_z2_hdmi_demo.cmd
Remove-Item Env:\PHC_06S2_PROGRAM_BIT_FILE
```

## Claim Boundary

- This verifies RTL integration, routed timing closure, bitstream generation, hardware detection, and JTAG programming.
- This does not claim human visual HDMI confirmation, because I cannot see the attached display from the tool output.
- This does not alter or supersede the existing LED-only demo path; the original top and bitstream remain available as failover.
- This does not make any PnL or model-performance claim.

## Honest Conclusion

The HDMI upgrade is integrated as a sibling board top and reaches routed P&R with timing met. Both the fixed-color HDMI-only bitstream and the final PHC status-color bitstream programmed successfully over JTAG, and the final board state is the PHC+HDMI bitstream. The only remaining gap is physical monitor observation: the evidence supports `routed_pnr_met_bitgen_jtag_programmed_visual_pending`, not full visual demo acceptance.
