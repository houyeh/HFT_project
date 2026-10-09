# FPGA Known-Result Replay Viewer

status: `fpga_known_result_viewer_ready`

## Scope Boundary

- This is a laptop-side software visualization only.
- No RTL, bitstream, threshold, checkpoint, or P&R artifact was modified.
- `03_cpu_live_replay_demo.py` was not modified; this viewer is an independent script/window.
- The viewer does not read FPGA/JTAG/internal state. It displays the FPGA replay results already prepared before programming.
- The script reads only existing 06S2 demo replay artifacts and writes derived viewer CSV/JSON/PNG outputs.

## Numeric Source Lock

The displayed bid/ask scores are decoded from the same BRAM readout artifact used by the board wrapper comparator:

- BRAM readout file: `n2ew_era\06s2_pynq_z2_wrapper_bringup\bram\demo_replay_x_readout_q24_12.mem`
- BRAM packing statement: `n2ew_era\06s2_pynq_z2_wrapper_bringup\bram\BRAM_VECTOR_PACKING.md`
- BRAM update manifest: `n2ew_era\06s2_pynq_z2_wrapper_bringup\results\bram_expected_from_hls_csim_manifest.json`
- Source HLS DUT output: `n2ew_era\06s2_pynq_z2_wrapper_bringup\results\demo_replay_seed2_90_hls_csim_sequential_out.csv`
- Event timestamp/source metadata: `n2ew_era\06s2_pynq_z2_wrapper_bringup\results\demo_replay_input_features.csv`

`BRAM_VECTOR_PACKING.md` states that each `demo_replay_x_readout_q24_12.mem` line stores `x_t_0 ... x_t_15 expected_readout_0 expected_readout_1`, and that the expected readout fields are from `demo_replay_seed2_90_hls_csim_sequential_out.csv`. The viewer verifies this before plotting by decoding the final two Q24.12 words from every BRAM row and comparing them against the HLS sequential CSim DUT output at LSB precision.

## Verification

| check | result |
|---|---:|
| rows checked | 4096 |
| source-vs-BRAM readout LSB mismatches | 0 |
| source-vs-BRAM decision match rate | 100.000% |
| max source Q24.12 grid residual | 2.048e-07 LSB |
| bid decisions | 1430 |
| ask decisions | 1701 |
| no_quote decisions | 965 |
| fixed FPGA latency model | 25.733 us/event |
| modeled 4096-event FPGA playback span | 105.377 ms |

Decision logic matches the existing 06S2 wrapper drift audit convention: threshold is `0` LSB; bid wins when `bid >= ask` and `bid > 0`; ask wins when `ask > bid` and `ask > 0`; otherwise the decision is `no_quote`.

## Viewer

Script:

```cmd
python n2ew_era\06s2_pynq_z2_wrapper_bringup\04_fpga_known_result_replay_viewer.py
```

Headless screenshot/update run:

```cmd
python n2ew_era\06s2_pynq_z2_wrapper_bringup\04_fpga_known_result_replay_viewer.py --no-show --save-screenshot n2ew_era\06s2_pynq_z2_wrapper_bringup\figures\fpga_known_result_replay_viewer.png
```

The live display uses the fixed FPGA latency model for the replay clock and x-axis. The default interactive playback is slowed down for human viewing with `--playback-slowdown 250`; this affects only the laptop animation pace, not the reported FPGA elapsed-time model.

## Screen Description

The window has four regions:

- A rolling bid/ask/no_quote decision strip.
- Bid and ask Q24.12 readout score traces with the zero threshold line.
- A cumulative decision distribution bar chart.
- Current-row details: row index, event timestamp, modeled FPGA elapsed time, bid/ask score and LSB value, current decision, and cumulative source-match rate.

Screenshot:

`n2ew_era\06s2_pynq_z2_wrapper_bringup\figures\fpga_known_result_replay_viewer.png`

## Outputs

- Derived row CSV: `n2ew_era\06s2_pynq_z2_wrapper_bringup\results\fpga_known_result_replay_viewer_rows.csv`
- Summary JSON: `n2ew_era\06s2_pynq_z2_wrapper_bringup\results\fpga_known_result_replay_viewer_summary.json`
- Screenshot PNG: `n2ew_era\06s2_pynq_z2_wrapper_bringup\figures\fpga_known_result_replay_viewer.png`

## Interpretation Boundary

This window shows the results that the fixed FPGA demo replay is expected to produce from the already programmed replay artifacts. It is not a live readback path from the FPGA, and it must not be described as JTAG readout, internal-state probing, or a hardware telemetry stream. It is a synchronized presentation aid for the known 4096-event replay.
