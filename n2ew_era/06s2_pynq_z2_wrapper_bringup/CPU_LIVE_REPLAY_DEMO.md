# CPU Live Replay Latency Demo

status: `cpu_live_replay_demo_ready`

## Scope

- Laptop-side software demo only; no RTL, bitstream, threshold, or checkpoint was modified.
- The script uses the existing 4096-event replay burst prepared for the FPGA demo.
- The CPU model is the PyTorch float32 step model migrated from 05C4 seed2, matching the HLS deployment lineage.
- NEW5 is not read; the script consumes only the prepared 06S2 replay CSV and manifest.

## Inputs

- Replay CSV: `n2ew_era\06s2_pynq_z2_wrapper_bringup\results\demo_replay_input_features.csv`
- Manifest: `n2ew_era\06s2_pynq_z2_wrapper_bringup\results\replay_vector_manifest.json`
- Step checkpoint: `n2ew_era\06e_phc_stp_step_hls4ml\models\phc_stp_step_hls_seed2.pt`
- FPGA reference latency: `25.733 us`
- CPU execution mode: `trace`
- Torch threads: `8`; interop threads: `8`
- Warmup iterations before timing: `100`
- Timed region: `TorchScript/eager forward(x_cat) only; x_cat copy and output slicing are outside the measured region.`

## One-Click Launch

From `cmd.exe` or a double-clickable terminal:

```cmd
cd /d C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06s2_pynq_z2_wrapper_bringup
scripts\run_cpu_live_replay_demo.cmd
```

Headless smoke test, useful if the plotting window gets stuck:

```cmd
scripts\run_cpu_live_replay_demo.cmd --no-plot --no-hold
```

## Expected Demo Screen

- Default terminal mode: a live recent-latency strip plus a running histogram. `.` means CPU <= FPGA latency, `+` means <= 2x FPGA, `*` and `!` mark slower tail events.
- Optional matplotlib mode: use `scripts\run_cpu_live_replay_demo.cmd --plot-mode matplotlib` for a red CPU scatter plot and running histogram with a green FPGA reference line.
- Running stats show `mean`, `median`, `p99`, `jitter = p99 - median`, and deadline-miss rate vs the FPGA reference.
- In matplotlib mode, the final plot stays open after the 4096-event replay. Close the window to exit.

## Final Summary

| run | mode | threads | timed region | mean_us | median_us | p99_us | jitter_us | min_us | max_us |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| previous unoptimized Python | eager | 1 | eager model forward plus Python split_output | 191.191 | 110.300 | 1662.475 | 1552.175 | 76.700 | 42285.400 |
| optimized Python demo | trace | 8 | forward only | 60.251 | 41.800 | 378.050 | 336.250 | 37.000 | 4070.400 |
| known LibTorch C++ baseline | TorchScript C++ | 8 | module.forward only | 77.450 | 51-57.800 | not rerun here | not rerun here | not rerun here | not rerun here |
| Python trace parity check | trace | 8 | fixed zero input, forward only | 95.242 | 50.400 | 754.305 | 703.905 | 30.800 | 58898.300 |
| controlled C++ retest | TorchScript C++ | 8 | fixed zero input, module.forward only | 72.207 | 48.900 | 543.900 | 495.000 | 31.300 | 2592.600 |
| controlled Python trace retest | trace | 8 | fixed zero input, forward only | 66.696 | 43.300 | 457.007 | 413.707 | 28.900 | 2410.000 |

Optimized Python deadline miss rate vs `25.733 us`: `100.0000%`.

The optimized Python demo now uses the same broad CPU opponent class as the report baseline: warmup, 8 Torch threads, and a traced TorchScript-style forward path. It remains a Python-hosted live visualizer, not the C++ LibTorch executable itself.
The optimized Python mean is close to the already reported optimized CPU reference, while preserving live replay visualization.

## Methodology Parity Check vs LibTorch C++ Baseline

Known C++ baseline source: `n2ew_era/06f_phc_lite_manual_hls/cpp_libtorch_benchmark.cpp`, with measured provenance in `n2ew_era/06f_phc_lite_manual_hls/results/_tmp_libtorch_probe.json` (`mean=77.4484us`, `median=57.8us`, `p99=361.1us`) and a later rerun in `cpu_baseline_timing_libtorch_raw_rerun2.json` (`mean=64.0703us`, `median=43.8us`, `p99=434.1us`).

| item | C++ LibTorch baseline | Python live replay demo | Python trace parity check |
|---|---|---|---|
| Input | fixed `torch::zeros({1,32})` | 4096 real replay events plus recurrent state | fixed `torch.zeros(1,32)` |
| Timed region | `module.forward(inputs)` only | `timed_model(x_cat)` only; tensor copy and slicing excluded | `timed_model(x_cat)` only |
| Warmup | 100 | 100 | 100 |
| Iterations | 10000 | 4096 replay events | 10000 |
| Threads | `torch::set_num_threads(8)`, interop 8 | `torch.set_num_threads(8)`, interop 8 | same Python setting |
| Runtime | C++ LibTorch executable | Python host with TorchScript trace | Python host with TorchScript trace |

The fixed-zero-input Python trace parity check aligns the input and iteration protocol more closely with `cpp_libtorch_benchmark.cpp`; the first parity run measured `mean=95.242us`, `median=50.400us`, and `p99=754.305us`. A later controlled same-session retest measured C++ at `mean=72.207us`, `median=48.900us`, `p99=543.900us` and Python trace at `mean=66.696us`, `median=43.300us`, `p99=457.007us`.

Conclusion: do not say in the demo that the Python live visualizer is faster than the C++ LibTorch baseline. The honest line is: `the Python live demo is in the same order of magnitude as the optimized C++ CPU baseline, but it remains a Python-hosted visualization path; the FPGA reference is deterministic and already routed.`

## Root Cause Determination

- TorchScript-vs-eager checkpoint equivalence: `pass`, max_abs_diff `0` across five fixed inputs.
- The older C++ baseline files were produced on different dates (`2026-06-26` and `2026-06-29`) and do not contain CPU-load snapshots, so their background-load state cannot be reconstructed.
- The controlled retest recorded CPU load rather than assuming idle. It was not single-digit idle: `33.584%` before C++, `35.478%` between C++ and Python, and `27.452%` after Python. A follow-up idle probe stayed between `23.173%` and `38.184%`.
- The controlled C++ retest used the same TorchScript checkpoint and same fixed-zero-input forward-only protocol as `cpp_libtorch_benchmark.cpp`; it measured `mean=72.207us`, `median=48.900us`, `p99=543.900us`.
- The controlled Python trace retest measured `mean=66.696us`, `median=43.300us`, `p99=457.007us` under the same fixed-zero-input protocol.

Final determination: model/checkpoint mismatch is excluded. Background host load is the most plausible explanation for the C++/Python spread and p99 instability, but it remains partly inferential because the older runs did not record load and this retest could not reach a single-digit idle CPU state. For demo/QA, cite the controlled C++ retest (`mean=72.207us`, `median=48.900us`, `p99=543.900us`) as the current optimized CPU single-stream reference, and describe Python live replay as comparable, not faster.

## Demo Interpretation

Use this as a latency/jitter visualization, not a PnL claim. The intended visual message is that the CPU per-event latency moves around while the FPGA reference is a fixed routed latency line.

The CPU latency is measured live on the laptop for each replay event. The FPGA line is the routed hardware reference latency and is not live-synchronized to the board in this software-only demo.

If the plot window stops responding, close it and rerun the launcher. Use the headless command above to confirm the model/data path is healthy before the live presentation.

## Outputs

- `n2ew_era\06s2_pynq_z2_wrapper_bringup\results\cpu_live_replay_latency.csv`
- `n2ew_era\06s2_pynq_z2_wrapper_bringup\results\cpu_live_replay_latency_summary.json`
- `figures/cpu_live_replay_latency.png` when plotting is enabled.
