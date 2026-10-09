# HDMI Live Upload Feasibility

status: `live_upload_not_feasible_with_current_bitstream_use_prebaked_replay`

Date: 2026-08-08

## Scope

This was a time-boxed attempt to determine whether the demo laptop can upload a new replay burst to the already-programmed PYNQ-Z2, without rebuilding or reprogramming the main demo bitstream, and then read actual FPGA readout values back to the laptop.

No RTL, HLS core, checkpoint, replay data, HDMI bitstream, LED-only bitstream, or NEW1-NEW5 data was modified for this feasibility check.

## Short Conclusion

The current LED/HDMI demo bitstreams do not expose a host-writable replay memory or a host-readable result path. Vivado Hardware Manager reports that the programmed `xc7z020_1` design has no supported soft debug cores, and the enumerated hardware objects contain no `hw_ila`, `hw_vio`, `hw_axi`, or `hw_probe` objects.

Therefore, JTAG-based live BRAM write/readback is not available for the current bitstream. UART live upload is also not available without a new experimental RTL wrapper and a new bitstream, because the current board top exposes only clock, buttons, LEDs, and HDMI pins.

Recommended demo path: keep using the existing prebaked replay bitstream plus LED/HDMI display, and run the laptop CPU live replay demo side-by-side.

## Evidence

Read-only hardware probe script:

```text
n2ew_era/06s2_pynq_z2_wrapper_bringup/tcl/probe_live_upload_hw_objects.tcl
```

One-click probe wrapper:

```text
n2ew_era/06s2_pynq_z2_wrapper_bringup/scripts/probe_live_upload_hw_objects.cmd
```

Probe log:

```text
n2ew_era/06s2_pynq_z2_wrapper_bringup/logs/probe_live_upload_hw_objects.log
```

Key lines from the probe:

```text
PYNQ_Z2_DETECTION_STATUS hw_device_present
CURRENT_HW_DEVICE xc7z020_1
INFO: [Labtools 27-1434] Device xc7z020 (JTAG device index = 1) is programmed with a design that has no supported soft debug core(s) in it.
OBJECT_KIND hw_ilas COUNT 0
OBJECT_KIND hw_vios COUNT 0
OBJECT_KIND hw_axis COUNT 0
OBJECT_KIND hw_probes COUNT 0
```

There is also no `.ltx` debug-probe file under the phase folder for the current demo bitstreams, which is consistent with the Hardware Manager result.

## Route Assessment

| route | result | reason |
|---|---|---|
| JTAG-based BRAM write with current bitstream | Not feasible | The replay memory is an internal `$readmemh`-initialized RTL array, not an AXI BRAM mapped into a Vivado `hw_axi`/`hw_mem` address space. Hardware Manager exposes no writable memory object. |
| JTAG readback of actual readout values with current bitstream | Not feasible | `MARK_DEBUG` attributes in RTL are not enough by themselves. The programmed bitstream has no ILA/VIO/debug probes visible to hw_server. |
| UART live upload with current bitstream | Not feasible | The current top-level ports do not include UART RX/TX, and the wrapper has no serial protocol, replay-memory write port, or result-readback buffer. |
| UART or JTAG-to-AXI in a future experimental bitstream | Feasible as a follow-up | Requires adding a communication endpoint, dual-port replay memory or AXI-lite registers, result capture buffer, XSim, P&R, and a new bitstream. This is outside the no-reprogramming goal. |

## Why `updatemem` Is Not The Same Thing

Vivado can sometimes update BRAM initialization contents inside a bitstream and then generate a patched bitstream. That still requires generating and programming a new bitstream. It does not provide live replay upload into an already running FPGA design, so it does not satisfy the demo goal for this task.

## Fallback Demo Procedure

Use the already verified prebaked replay flow.

1. Connect PYNQ-Z2 power and USB/JTAG.
2. From Command Prompt:

```cmd
cd /d C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06s2_pynq_z2_wrapper_bringup
scripts\check_pynq_z2_hw.cmd
```

Expected detection:

```text
PYNQ_Z2_DETECTION_STATUS: hw_device_present
```

3. Program the HDMI status demo if an HDMI monitor is available:

```cmd
scripts\program_pynq_z2_hdmi_demo.cmd
```

4. If HDMI is not needed or misbehaves, program the LED-only fallback:

```cmd
scripts\program_pynq_z2_demo.cmd
```

Expected programming success:

```text
PYNQ_Z2_PROGRAM_STATUS: programmed
```

Expected LED behavior for the prebaked replay:

| LED | expected meaning |
|---|---|
| LD0 | replay done, latched high after completion |
| LD1 | pass |
| LD2 | fail, should remain off |
| LD3 | heartbeat blinking |

5. Run the CPU live replay visualization on the laptop side:

```cmd
scripts\run_cpu_live_replay_demo.cmd
```

This is the recommended way to show "same replay data, CPU latency/jitter vs fixed routed FPGA latency" without risking the stable FPGA demo path.

## If Live Upload Is Needed Later

The lowest-risk future design is not to poke inferred BRAM directly. Instead, build a separate experimental bitstream with one explicit host interface:

- JTAG-to-AXI Debug Bridge + AXI BRAM Controller or AXI-lite replay registers.
- Replay memory changed to dual-port RAM: one port for wrapper playback, one port for host writes.
- Result capture RAM/registers for readback.
- A control register block: reset, load count, start, done, pass/fail, current index, first mismatch, readout snapshot.
- XSim positive/negative tests before P&R.

That would be a new bitstream and should remain separate from the current prebaked LED/HDMI demo bitstream.

## Final Decision

`live_upload_not_feasible_with_current_bitstream_use_prebaked_replay`

The existing "prebaked replay + LED/HDMI status + laptop CPU live replay" demo remains the safe path for presentation.
