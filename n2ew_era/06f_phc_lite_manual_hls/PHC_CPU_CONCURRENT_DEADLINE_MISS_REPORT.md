# PHC-lite CPU Concurrent Deadline Miss Report

status: mixed

## Scope

This report measures CPU-side LibTorch inference latency under concurrent streams. It does not use NEW5, does not touch PnL / fill / market replay logic, and does not modify or resynthesize the PHC-lite HLS IP / RTL. The FPGA value `83.16 us` is used only as the fixed latency deadline threshold.

## Completed Verification

### Original Benchmark Location

Original source:

- `n2ew_era/06f_phc_lite_manual_hls/cpp_libtorch_benchmark.cpp`
- Prior result: `n2ew_era/06f_phc_lite_manual_hls/results/cpu_baseline_timing_libtorch.json`

Original execution model confirmed from source:

- Single process, single caller loop, one TorchScript module instance.
- `torch::set_num_threads(8)` and `torch::set_num_interop_threads(8)`.
- `torch::NoGradGuard` enabled.
- Input is `torch::zeros({1, 32}, float32 CPU)`.
- Warmup = 100 calls; timed loop = 10,000 serial calls.
- No thread pinning / affinity control found in the benchmark source.

Relevant source lines:

- Thread settings: `cpp_libtorch_benchmark.cpp:129-130`
- Model/input setup: `cpp_libtorch_benchmark.cpp:132-136`
- Warmup/timing loop: `cpp_libtorch_benchmark.cpp:138-155`

### Concurrent Benchmark Added

New source:

- `n2ew_era/06f_phc_lite_manual_hls/cpp_libtorch_concurrent_benchmark.cpp`

Build output:

- `n2ew_era/06f_phc_lite_manual_hls/results/libtorch_build/cpp_libtorch_concurrent_benchmark.exe`

Design choices:

- N concurrent worker threads.
- Each worker loads its own TorchScript module instance from the same `.pt` file, to avoid shared-module locking accidentally serializing the test.
- Each worker uses the same input shape and forward path as the original benchmark.
- Each worker warms up 100 calls before timing.
- A shared atomic start flag releases all streams into the timed loop together.
- Each stream performs 10,000 timed calls per round.
- Two independent rounds are recorded separately for each N.
- Deadline miss is `latency_us > 83.16`.

Relevant source lines:

- Deadline and default iteration settings: `cpp_libtorch_concurrent_benchmark.cpp:19-23`
- Deadline miss calculation: `cpp_libtorch_concurrent_benchmark.cpp:85-92`
- Worker/module/timing loop: `cpp_libtorch_concurrent_benchmark.cpp:107-139`
- Atomic concurrent start: `cpp_libtorch_concurrent_benchmark.cpp:150-153`
- Thread settings: `cpp_libtorch_concurrent_benchmark.cpp:193-194`

Build environment:

- Compiler: Visual Studio 2022 MSVC x64 developer environment.
- PyTorch C++ headers/libs from `.venv/Lib/site-packages/torch`.
- Runtime DLL path required: `.venv/Lib/site-packages/torch/lib`.

## Machine / Background Load

CPU:

- `Intel(R) Core(TM) Ultra 7 258V`
- 8 cores, 8 logical processors
- `PROCESSOR_IDENTIFIER = Intel64 Family 6 Model 189 Stepping 1, GenuineIntel`

This matches the earlier benchmark metadata (`hardware_concurrency=8`, same `PROCESSOR_IDENTIFIER`). However, the machine was not fully idle during this run.

Background load snapshots:

| snapshot | total CPU counter samples (%) | top observed process deltas |
|---|---:|---|
| before run | 40.34, 38.91, 41.69, 41.53, 41.37 | Codex 1.44%, Code 1.15%, PowerShell/Taskmgr 0.58% each |
| after run | 42.20, 37.31, 38.49, 41.65, 45.41 | Code 1.26%, Codex 0.78%, Taskmgr 0.68% |

The total CPU counter stayed around 37-45% even though the top per-process deltas were small. Therefore these results are valid empirical measurements for this run, but not a clean idle-machine isolation study.

Raw environment files:

- `results/cpu_concurrent_benchmark_environment_before.json`
- `results/cpu_concurrent_benchmark_environment_after.json`

## N=1 Sanity Check

Prior reported LibTorch baseline:

| source | mean us | median/p50 us | p99 us | min us |
|---|---:|---:|---:|---:|
| prior `cpu_baseline_timing_libtorch.json` | 74.5444 | 51.0 | 636.7 | 43.8 |

Immediate rerun of the original single-stream executable in this task:

| source | mean us | median us | p99 us | min us | deadline miss rate |
|---|---:|---:|---:|---:|---:|
| `cpu_baseline_timing_libtorch_probe_20260629.json` / `.csv` | 77.4484 | 57.8 | 361.1 | 52.6 | 15.12% |

Concurrent harness N=1:

| N | round | samples | mean us | p50 us | p95 us | p99 us | min us | max us | miss count | miss rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 10,000 | 63.499160 | 45.7 | 135.9 | 338.8 | 28.9 | 3,455.0 | 1,166 | 11.6600% |
| 1 | 2 | 10,000 | 64.346260 | 49.3 | 150.0 | 306.9 | 29.9 | 1,486.7 | 1,254 | 12.5400% |

The N=1 concurrent harness does not exactly reproduce the earliest `74.5444 / 51.0 / 636.7 / 43.8 us` baseline. This is not a new inconsistency: `results/cpu_baseline_timing_libtorch_raw_drift_report.md` already documented same-protocol raw reruns with lower p99 values than the original summary. The current N=1 numbers should therefore be treated as the in-run control for the concurrency experiment, not as the original raw distribution.

## Concurrent Results

Deadline threshold: `83.16 us`.

Each row is one independent 10,000-iteration-per-stream round. Samples per round are therefore `N * 10,000`.

| N | round | samples | mean us | p50 us | p95 us | p99 us | min us | max us | miss count | miss rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 10,000 | 63.499160 | 45.7 | 135.9 | 338.8 | 28.9 | 3,455.0 | 1,166 | 11.6600% |
| 1 | 2 | 10,000 | 64.346260 | 49.3 | 150.0 | 306.9 | 29.9 | 1,486.7 | 1,254 | 12.5400% |
| 2 | 1 | 20,000 | 60.700305 | 46.3 | 129.5 | 262.2 | 31.5 | 1,087.7 | 2,466 | 12.3300% |
| 2 | 2 | 20,000 | 67.094420 | 50.9 | 142.6 | 284.8 | 38.6 | 2,957.4 | 2,825 | 14.1250% |
| 4 | 1 | 40,000 | 72.198630 | 65.6 | 126.3 | 262.6 | 33.8 | 7,293.3 | 6,050 | 15.1250% |
| 4 | 2 | 40,000 | 70.616475 | 56.9 | 115.2 | 247.7 | 41.3 | 3,647.0 | 6,055 | 15.1375% |
| 8 | 1 | 80,000 | 79.867154 | 63.8 | 96.2 | 202.6 | 36.1 | 32,380.6 | 6,705 | 8.3812% |
| 8 | 2 | 80,000 | 90.140400 | 69.9 | 91.6 | 220.2 | 53.6 | 26,408.2 | 7,500 | 9.3750% |

Machine-readable outputs:

- `results/cpu_baseline_timing_libtorch_probe_20260629.json`
- `results/cpu_baseline_timing_libtorch_probe_20260629.csv`
- `results/cpu_concurrent_benchmark_summary.csv`
- `results/cpu_concurrent_benchmark_N1.json`
- `results/cpu_concurrent_benchmark_N2.json`
- `results/cpu_concurrent_benchmark_N4.json`
- `results/cpu_concurrent_benchmark_N8.json`
- Raw CSVs: `results/cpu_concurrent_benchmark_N{1,2,4,8}.csv`
- Run logs: `results/cpu_concurrent_benchmark_N{1,2,4,8}.log`

## Interpretation

Verified observations:

- CPU inference latency crosses the `83.16 us` FPGA deadline in every measured CPU condition.
- N=4 produced the highest repeatable deadline miss rate in this run: `15.1250%` and `15.1375%`.
- N=8 did not produce the highest miss rate, but it did produce the largest rare-tail stalls: max `32,380.6 us` and `26,408.2 us`.
- The miss-rate trend is not monotonic: N=8 miss rate (`8.3812%`, `9.3750%`) is lower than N=4, even though N=8 mean latency and max tail are higher.

Therefore, this run supports a narrow latency claim:

> Under the measured LibTorch CPU conditions, a nonzero fraction of software inference calls missed the `83.16 us` fixed-latency FPGA deadline, while the FPGA reference latency is fixed by the existing RTL/cosim result.

This run does not support a stronger claim that CPU deadline miss rate monotonically worsens with concurrency. It also does not support any PnL, fill, or economic impact claim.

## N=16 Recommendation

Recommendation: run N=16 only as a separately labeled oversubscription stress test after explicit approval.

Reason:

- The 8-core / 8-logical-processor matched test did not show monotonic miss-rate growth.
- N=8 already showed rare long stalls above 26 ms, so N=16 may be useful for scheduler-pressure characterization.
- Because N=16 exceeds the confirmed logical CPU count, it should not be presented as the main multi-target matched-core result. It should be labeled as oversubscription / stress, not as the primary realistic concurrency case.

## Honest Conclusion

This task is diagnosed but mixed. The CPU benchmark and concurrent harness were located, extended, built, and executed. The deadline miss metric was computed from raw measured latencies without pooling the two rounds. The result confirms nonzero CPU deadline misses relative to `83.16 us`, but it does not confirm the expected monotonic degradation of miss rate as concurrency rises. The cleanest defensible statement is about CPU latency variability and deadline crossings, not about average speed superiority, PnL, or guaranteed concurrency-driven miss-rate growth.
