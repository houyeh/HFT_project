# HDMI Numeric Display Bring-up

status: `visual_failed_final_abandoned`

## One-line Conclusion

The final done-edge numeric HDMI attempt met routed timing (`WNS=+0.100ns`) and was programmed, but the live screen still flickered/turned black after briefly showing the expected final packet (`4095`, `-0.012`, `-0.136`, decision `0`). After three attempts, the HDMI numeric-display branch is formally abandoned for demo use; the board was restored to the verified solid-color HDMI fallback bitstream.

## Scope Boundary

- Did not modify the verified LED-only top: `rtl/pynq_z2_phc_demo_board_top.sv`.
- Did not modify the verified solid-color HDMI top: `rtl/pynq_z2_phc_demo_board_top_hdmi.sv`.
- Did not overwrite the fallback bitstreams:
  - `results/phc_demo_wrapper.bit`
  - `results/phc_demo_wrapper_hdmi.bit`
- Did not modify PHC HLS core RTL, replay vectors, BRAM payload, model weights, threshold, checkpoint, or NEW1-NEW5 data.
- Added only sibling/new-path artifacts for this high-risk attempt.

## Added Artifacts

| type | path |
|---|---|
| seven-segment numeric overlay | `rtl/pynq_z2_hdmi_numeric_overlay.sv` |
| HDMI-only numeric smoke-test top | `rtl/pynq_z2_hdmi_numeric_test_top.sv` |
| PHC HDMI numeric sibling top | `rtl/pynq_z2_phc_demo_board_top_hdmi_numeric.sv` |
| isolated numeric P&R flow | `tcl/run_pnr_hdmi_numeric_variant.tcl` |
| numeric programming helper | `scripts/program_pynq_z2_hdmi_numeric_demo.cmd` |
| numeric smoke-test bitstream | `results/hdmi_numeric_test.bit` |
| PHC numeric bitstream | `results/phc_demo_wrapper_hdmi_numeric.bit` |

The repaired numeric top is intended to display:

- Row 1: event index, clamped at 9999.
- Row 2: bid readout from the wrapper-exposed PHC readout register, signed decimal.
- Row 3: ask readout from the wrapper-exposed PHC readout register, signed decimal.
- Row 4: decision code from the existing threshold-0 bid/ask/no_quote rule (`0=no_quote`, `1=bid`, `2=ask`).

The background status color is unchanged from the solid-color HDMI version: reset blue, running yellow, pass green, fail red.

## Follow-up Diagnostic and Repair Attempt

Date: 2026-08-09

User-observed symptoms from the first numeric bitstream:

1. The screen showed content resembling `E xxxx B xxxx A xxx d x`, making the numeric fields look like hexadecimal output with letters mixed into decimal values.
2. The screen briefly displayed plausible content, then flickered/turned black.

### Symptom 0: LED Heartbeat / Core Health Check

The requested first physical check was whether LED heartbeat continued while HDMI flickered. I cannot directly see the board LEDs from the tool output. No new evidence was collected that the PHC replay FSM or LED heartbeat stopped. The investigation therefore treats core failure as unconfirmed and focuses on the two directly auditable display-path issues below.

### Symptom 1: Hex-like Letters in Numeric Fields

Root-cause status: `confirmed_not_bcd_failure_confirmed_label_design_issue`

Evidence:

- The RTL did instantiate `phc_numeric_binary_to_bcd` for event index and fractional score digits.
- A standalone xsim testbench was added at `sim/tb_numeric_bcd.sv`.
- Behavioral simulation command:

```powershell
& 'C:\AMDDesignTools\2025.2\Vivado\bin\xvlog.bat' --sv ..\rtl\pynq_z2_hdmi_numeric_overlay.sv ..\sim\tb_numeric_bcd.sv
& 'C:\AMDDesignTools\2025.2\Vivado\bin\xelab.bat' tb_numeric_bcd -s tb_numeric_bcd
& 'C:\AMDDesignTools\2025.2\Vivado\bin\xsim.bat' tb_numeric_bcd -runall
```

Simulation output:

```text
PASS event value=0 bcd=0x0000
PASS event value=9 bcd=0x0009
PASS event value=10 bcd=0x0010
PASS event value=42 bcd=0x0042
PASS event value=99 bcd=0x0099
PASS event value=4095 bcd=0x4095
PASS frac value=0 bcd=0x000
PASS frac value=9 bcd=0x009
PASS frac value=10 bcd=0x010
PASS frac value=42 bcd=0x042
PASS frac value=99 bcd=0x099
PASS frac value=999 bcd=0x999
TB_NUMERIC_BCD_PASS failures=0
```

Conclusion: the BCD converter itself did not produce non-decimal nibbles. The confusing letters were primarily caused by the first renderer intentionally using seven-segment letter labels `E`, `B`, `A`, and `D`; those labels looked like hexadecimal characters mixed with the numeric fields.

Repair made:

- Removed `E/B/A/D` prefixes from the renderer.
- Changed the on-screen layout to four decimal-only rows:
  - Row 1: event index, four decimal digits.
  - Row 2: bid score, signed decimal.
  - Row 3: ask score, signed decimal.
  - Row 4: decision code, decimal only: `0=no_quote`, `1=bid`, `2=ask`.

### Symptom 2: Flicker / Black Screen

Root-cause status: `cdc_risk_confirmed_timing_fix_failed`

Evidence:

- The previous numeric top directly double-flopped multi-bit buses from the 90 MHz replay domain into the 25 MHz pixel domain:
  - `current_index`
  - `ila_readout0`
  - `ila_readout1`
  - decision code
- Direct double-flop synchronization is acceptable for a single control bit, but unsafe for coherent multi-bit payloads. It can create torn values and corrupted display input.
- The previous numeric bitstream had positive but smaller timing margin than the solid-color HDMI top:

| variant | WNS | TNS | failing endpoints |
|---|---:|---:|---:|
| solid-color HDMI fallback | +0.273 ns | 0.000 ns | 0 |
| first numeric HDMI | +0.213 ns | 0.000 ns | 0 |

Repair attempted:

- Replaced direct multi-bit double-flop CDC with a source-domain packet latch plus toggle synchronization.
- The 90 MHz domain now updates a whole display packet and toggles `display_toggle_clk90`.
- The 25 MHz pixel domain double-flops only the toggle bit, then captures the entire stable packet after the toggle arrives.
- Reduced display update frequency from every 16 events to every 256 events, while still forcing an update on the final event.

Post-route result after this repair:

| metric | repaired numeric result |
|---|---:|
| Setup WNS | -0.040 ns |
| Setup TNS | -1.912 ns |
| Setup failing endpoints | 48 |
| Hold WHS | +0.009 ns |
| Hold THS | 0.000 ns |
| Hold failing endpoints | 0 |
| Timing status | `failed` |

The failing setup path is in the PHC 90 MHz region, not the HDMI pixel timing path:

```text
Slack (VIOLATED): -0.040ns
Source: u_demo_wrapper/u_phc_step/grp_phc_step_Outline_VITIS_LOOP_127_14_fu_2250/ap_CS_fsm_reg[10]/C
```

Conclusion: the unsafe CDC issue is confirmed and was repaired at the RTL level, but the repaired numeric variant no longer meets routed timing. Per the safety rule, this repaired bitstream was not programmed or visually accepted.

### Repaired Numeric Resource Delta vs Verified Solid-color HDMI

This is the resource delta of the repaired-but-timing-failed numeric build.

| metric | solid-color HDMI | repaired numeric HDMI | delta |
|---|---:|---:|---:|
| Setup WNS | +0.273 ns | -0.040 ns | -0.313 ns |
| Setup TNS | 0.000 ns | -1.912 ns | -1.912 ns |
| Setup failing endpoints | 0 | 48 | +48 |
| LUT | 15,747 | 16,743 | +996 |
| FF | 20,354 | 20,524 | +170 |
| DSP | 131 | 133 | +2 |
| Block RAM Tile | 28.5 | 28.5 | 0 |
| IOB | 15 | 15 | 0 |
| BUFG | 5 | 5 | 0 |
| MMCM | 2 | 2 | 0 |

Because timing failed, this repaired numeric build is diagnostic only and must not be used for demo.

Safety note: `scripts/program_pynq_z2_hdmi_numeric_demo.cmd` now checks `results/phc_demo_wrapper_hdmi_numeric_timing_summary_routed.rpt` and refuses to program the numeric bitstream unless that report contains `All user specified timing constraints are met.` This keeps the timing-failed numeric artifact from being accidentally used on stage.

Safety helper check:

```text
ERROR: Numeric HDMI timing report does not show timing met.
       Refusing to program ".../results/phc_demo_wrapper_hdmi_numeric.bit".
       Use scripts\program_pynq_z2_hdmi_demo.cmd for the verified solid-color HDMI fallback.
```

## Final Attempt: Done-edge Final Packet Latch

Date: 2026-08-09

Objective: shrink the numeric HDMI behavior to the lowest-risk useful form. Instead of scrolling/updating during the 4096-event replay, the top now latches exactly one final display packet when wrapper `done` first rises in the 90 MHz replay domain, then holds that packet until reset.

Implementation:

- Removed `DISPLAY_UPDATE_MASK`, per-event/per-`ila_core_done` capture, and toggle-on-update behavior from `rtl/pynq_z2_phc_demo_board_top_hdmi_numeric.sv`.
- Added `done_clk90_d` edge detection and `packet_ready_clk90`.
- On `done_rise_clk90`, latches wrapper-level exposed signals only:
  - `current_index`
  - `ila_readout0`
  - `ila_readout1`
  - derived decision code
- Pixel domain double-flops only the one-bit `packet_ready_clk90`, then captures the already-stable whole packet.
- No PHC HLS RTL, replay BRAM payload, model weights, threshold, or fallback top module was modified.

Timing result:

| variant | setup WNS | setup TNS | setup failing endpoints | hold WHS | timing status |
|---|---:|---:|---:|---:|---|
| solid-color HDMI fallback | +0.273 ns | 0.000 ns | 0 | +0.008 ns | met |
| second numeric CDC-safe repair | -0.040 ns | -1.912 ns | 48 | +0.009 ns | failed |
| third numeric done-edge latch | +0.100 ns | 0.000 ns | 0 | +0.007 ns | met |

The third attempt did reduce pressure relative to the second attempt and cleared timing:

```text
All user specified timing constraints are met.
```

Worst setup path was still in the PHC 90 MHz region, not the video overlay:

```text
Slack (MET): 0.100ns
Source: u_demo_wrapper/u_phc_step/grp_phc_step_Outline_VITIS_LOOP_127_14_fu_2250/ap_CS_fsm_reg[5]/C
```

Resource delta versus the verified solid-color HDMI fallback:

| metric | solid-color HDMI | third numeric done-edge | delta |
|---|---:|---:|---:|
| LUT | 15,747 | 16,736 | +989 |
| FF | 20,354 | 20,523 | +169 |
| DSP | 131 | 133 | +2 |
| Block RAM Tile | 28.5 | 28.5 | 0 |
| IOB | 15 | 15 | 0 |
| BUFG | 5 | 5 | 0 |
| MMCM | 2 | 2 | 0 |

JTAG programming succeeded because the numeric helper's timing guard passed:

```text
PYNQ_Z2_HDMI_PROGRAM_BITSTREAM: .../results/phc_demo_wrapper_hdmi_numeric.bit
PYNQ_Z2_HDMI_PROGRAM_STATUS: programmed
```

User visual observation:

```text
flicker/black
4095
-0.012
-0.136
0
```

Interpretation of the four displayed rows:

| row | meaning |
|---|---|
| `4095` | final replay event index, confirming the packet was captured at the end of the 4096-event replay |
| `-0.012` | final bid readout score, signed decimal rendering of `ila_readout0` |
| `-0.136` | final ask readout score, signed decimal rendering of `ila_readout1` |
| `0` | decision code: no quote, because both final bid/ask scores are non-positive |

The value display itself therefore reached the intended final-packet state, but the HDMI screen still did not remain visually reliable.

Auto-rerun hypothesis:

- Current RTL does not intentionally rerun after `done`.
- `started` stays high after the first auto-start pulse, and `pynq_z2_phc_demo_wrapper` holds in `S_DONE` until reset.
- A true rerun would require a reset-path event such as `btn_rst`, MMCM lock loss, or board/system reset.
- Changing to manual button start would improve demo ergonomics by keeping reset/idle blue until the user presses start, but it would require a fourth RTL/P&R round and would not directly address the observed HDMI black-screen symptom. Per the final-attempt rule, no fourth round is attempted.

Additional overlay-only simulation check:

- A frame-count testbench was added at `sim/tb_numeric_overlay_frame_count.sv`.
- It instantiated only `pynq_z2_hdmi_numeric_overlay` with the observed final packet class and a green background.
- Output over one active 640x480 frame:

```text
OVERLAY_FRAME_COUNT white=10656 black=3975 bg=292569 other=0 total=307200
OVERLAY_FRAME_COUNT_PASS
```

Interpretation: the glyph/shadow renderer is not painting the whole screen black. The live black-screen symptom is therefore more consistent with HDMI link/reset/encoder instability or PHC-integration interaction than with a simple overlay fill bug.

If a future fourth attempt is explicitly authorized, the next diagnostic should not be another full PHC numeric build first. The lower-risk sequence is:

1. Program and visually test the existing fixed-value numeric smoke bitstream (`results/hdmi_numeric_test.bit`). This isolates numeric overlay + rgb2dvi from the PHC replay wrapper.
2. If the fixed-value smoke bitstream also flickers, redesign the overlay as a registered pixel pipeline: precompute decimal character codes once, use a tiny glyph ROM/lookup in the pixel domain, register RGB one cycle before `rgb2dvi`, and remove double-dabble/multiply logic from the per-pixel path.
3. If the fixed-value smoke bitstream is stable, keep the overlay architecture mostly intact but fix PHC integration: use manual button start, hold in idle until HDMI lock has been stable for many frames, expose `hdmi_locked`/`video_reset` on debug LEDs, and only start replay after the monitor has already locked.
4. In either case, the fallback solid-color HDMI bitstream remains the demo default until a new sibling bitstream passes timing and visual hold.

Final decision: this third attempt is `visual_failed_final_abandoned`. Numeric HDMI is not accepted for demo use.

## Stage 1: HDMI-only Numeric Smoke Test

Top: `rtl/pynq_z2_hdmi_numeric_test_top.sv`

This excluded the PHC replay wrapper and used fixed test values: event `1234`, bid `-0.392`, ask `0.076`, decision code `2`.

| metric | value |
|---|---:|
| Verification tier | `routed_pnr_met_bitgen_no_board_visual` |
| Bitstream | `results/hdmi_numeric_test.bit` |
| Bitstream size | 4,045,705 bytes |
| Setup WNS | +27.184 ns |
| Setup TNS | 0.000 ns |
| Setup failing endpoints | 0 |
| Hold WHS | +0.040 ns |
| Hold THS | 0.000 ns |
| Hold failing endpoints | 0 |
| LUT | 625 / 53,200 = 1.17% |
| FF | 81 / 106,400 = 0.08% |
| DSP | 2 / 220 = 0.91% |
| Block RAM Tile | 0 / 140 = 0.00% |
| IOB | 15 / 125 = 12.00% |
| OSERDES | 8 |
| OBUFDS | 4 |
| BUFG | 3 / 32 = 9.38% |
| MMCM | 1 / 4 = 25.00% |

No separate monitor observation was claimed for this smoke-test bitstream.

## Stage 2: PHC HDMI Numeric Variant

Top: `rtl/pynq_z2_phc_demo_board_top_hdmi_numeric.sv`

Implementation details:

- Instantiates the unchanged `pynq_z2_phc_demo_wrapper`.
- Reads already exposed wrapper signals: `current_index`, `ila_readout0`, `ila_readout1`, `done`, `pass`, and `fail`.
- Adds display-only capture registers; these do not feed back into the PHC core or comparator.
- Synchronizes status/index/readout/decision snapshots from the 90 MHz replay clock domain into the 25 MHz pixel clock domain using double-flop CDC, following the solid-color HDMI status approach.
- The first programmed version used `DISPLAY_UPDATE_MASK=15`, so the visual fields updated every 16 events plus the final event. The repaired-but-timing-failed version changed this to `255`, i.e. every 256 events plus the final event.

P&R/bitgen result:

| metric | value |
|---|---:|
| Verification tier before board observation | `routed_pnr_met_bitgen_jtag_programmed` |
| Bitstream | `results/phc_demo_wrapper_hdmi_numeric.bit` |
| Bitstream size | 4,045,715 bytes |
| Setup WNS | +0.213 ns |
| Setup TNS | 0.000 ns |
| Setup failing endpoints | 0 |
| Hold WHS | +0.016 ns |
| Hold THS | 0.000 ns |
| Hold failing endpoints | 0 |
| LUT | 16,565 / 53,200 = 31.14% |
| FF | 20,638 / 106,400 = 19.40% |
| DSP | 133 / 220 = 60.45% |
| Block RAM Tile | 28.5 / 140 = 20.36% |
| IOB | 15 / 125 = 12.00% |
| OSERDES | 8 |
| OBUFDS | 4 |
| BUFG | 5 / 32 = 15.63% |
| MMCM | 2 / 4 = 50.00% |

JTAG programming succeeded:

```text
PYNQ_Z2_HDMI_PROGRAM_DEVICE: xc7z020_1
PYNQ_Z2_HDMI_PROGRAM_BITSTREAM: .../results/phc_demo_wrapper_hdmi_numeric.bit
PYNQ_Z2_HDMI_PROGRAM_STATUS: programmed
```

However, user-side visual feedback was negative: the screen could not hold a useful persistent display and only flashed with simple information. Therefore this variant is not accepted for demo use.

## Resource Delta vs Verified Solid-color HDMI

Baseline: `rtl/pynq_z2_phc_demo_board_top_hdmi.sv`, `results/phc_demo_wrapper_hdmi.bit`.

| metric | solid-color HDMI | numeric HDMI | delta |
|---|---:|---:|---:|
| Setup WNS | +0.273 ns | +0.213 ns | -0.060 ns |
| Setup TNS | 0.000 ns | 0.000 ns | 0.000 ns |
| Setup failing endpoints | 0 | 0 | 0 |
| Hold WHS | +0.008 ns | +0.016 ns | +0.008 ns |
| Hold THS | 0.000 ns | 0.000 ns | 0.000 ns |
| Hold failing endpoints | 0 | 0 | 0 |
| LUT | 15,747 | 16,565 | +818 |
| FF | 20,354 | 20,638 | +284 |
| DSP | 131 | 133 | +2 |
| Block RAM Tile | 28.5 | 28.5 | 0 |
| IOB | 15 | 15 | 0 |
| OSERDES | 8 | 8 | 0 |
| OBUFDS | 4 | 4 | 0 |
| BUFG | 5 | 5 | 0 |
| MMCM | 2 | 2 | 0 |
| Bitstream bytes | 4,045,707 | 4,045,715 | +8 |

Interpretation: timing still met, but the overlay adds about 818 LUT, 284 FF, and 2 DSP. The 2 DSP come from the decimal conversion arithmetic in the display path, not from a model change.

## Visual Bring-up Result

Final status is `visual_failed_final_abandoned`, not `visual_confirmed_working`.

Observed across the three attempts:

- Attempt 1: timing met and JTAG programming succeeded, but the screen was flickery/non-persistent and the `E/B/A/D` row labels looked like hexadecimal characters.
- Attempt 2: BCD/label issue was fixed and multi-bit CDC was repaired, but post-route timing failed (`WNS=-0.040ns`), so the bitstream was not programmed.
- Attempt 3: final-at-done packet latch met routed timing (`WNS=+0.100ns`) and was programmed, but the live screen still flickered/turned black after briefly showing the correct final packet.

The third attempt did prove the final-packet display path was reading sensible wrapper-level values:

| displayed row | interpretation |
|---|---|
| `4095` | final replay event index |
| `-0.012` | final bid score |
| `-0.136` | final ask score |
| `0` | no-quote decision code |

However, it still failed the demo requirement: the HDMI numeric display did not remain visually stable.

Decision:

- Do not use `results/phc_demo_wrapper_hdmi_numeric.bit` for the live demo.
- 已嘗試三輪，正式放棄HDMI數字顯示，demo採用純色HDMI+筆電視窗方案。
- Keep the stable solid-color HDMI plus laptop-side known-result visualization as the recommended presentation path.

## Fallback Restored

After the failed visual checks, the board was reprogrammed with the verified solid-color HDMI bitstream:

```text
PYNQ_Z2_HDMI_PROGRAM_DEVICE: xc7z020_1
PYNQ_Z2_HDMI_PROGRAM_BITSTREAM: .../results/phc_demo_wrapper_hdmi.bit
PYNQ_Z2_HDMI_PROGRAM_STATUS: programmed
```

Current recommended bitstream for HDMI demo:

```text
n2ew_era\06s2_pynq_z2_wrapper_bringup\results\phc_demo_wrapper_hdmi.bit
```

LED-only fallback bitstream remains available:

```text
n2ew_era\06s2_pynq_z2_wrapper_bringup\results\phc_demo_wrapper.bit
```

## One-line Fallback Commands

Restore verified solid-color HDMI:

```powershell
n2ew_era\06s2_pynq_z2_wrapper_bringup\scripts\program_pynq_z2_hdmi_demo.cmd
```

Restore verified LED-only demo:

```powershell
n2ew_era\06s2_pynq_z2_wrapper_bringup\scripts\program_pynq_z2_demo.cmd
```

Use the HDMI programming Tcl with an explicit bitstream override:

```powershell
$env:PHC_06S2_PROGRAM_BIT_FILE='C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06s2_pynq_z2_wrapper_bringup\results\phc_demo_wrapper_hdmi.bit'; n2ew_era\06s2_pynq_z2_wrapper_bringup\scripts\program_pynq_z2_hdmi_demo.cmd; Remove-Item Env:\PHC_06S2_PROGRAM_BIT_FILE
```

Use the same override path for LED-only:

```powershell
$env:PHC_06S2_PROGRAM_BIT_FILE='C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06s2_pynq_z2_wrapper_bringup\results\phc_demo_wrapper.bit'; n2ew_era\06s2_pynq_z2_wrapper_bringup\scripts\program_pynq_z2_hdmi_demo.cmd; Remove-Item Env:\PHC_06S2_PROGRAM_BIT_FILE
```

## Repro Commands

Numeric smoke-test P&R:

```powershell
& 'C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat' -mode batch -source n2ew_era\06s2_pynq_z2_wrapper_bringup\tcl\run_pnr_hdmi_numeric_variant.tcl -tclargs numeric_test
```

PHC numeric P&R:

```powershell
$env:PHC_CORE_RTL_DIR='C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06s2_pynq_z2_wrapper_bringup\hls_seed2_90_replay_component\hls_component\hls_component_seq\hls\sim\verilog'
& 'C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat' -mode batch -source n2ew_era\06s2_pynq_z2_wrapper_bringup\tcl\run_pnr_hdmi_numeric_variant.tcl -tclargs phc_numeric
```

Program numeric experimental bitstream, only if a future numeric timing report is clean:

```powershell
n2ew_era\06s2_pynq_z2_wrapper_bringup\scripts\program_pynq_z2_hdmi_numeric_demo.cmd
```

Current repaired numeric report is not clean, so this command intentionally refuses to program.

## Honest Conclusion

The high-risk numeric HDMI branch reached a narrow technical milestone in the third attempt: routed timing met, JTAG programming succeeded, and the screen briefly showed the correct final packet. It still failed the practical demo criterion because the display flickered/turned black instead of holding a stable numeric page. Per the final-attempt rule, there will be no fourth RTL/P&R round. For the demo, use `results/phc_demo_wrapper_hdmi.bit` for the board display and the laptop-side known-result viewer for inference numbers. The original LED-only bitstream also remains untouched.
