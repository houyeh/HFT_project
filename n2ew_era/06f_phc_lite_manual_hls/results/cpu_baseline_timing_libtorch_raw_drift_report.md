# LibTorch CPU Raw Latency Capture Drift Report

This check reran the original LibTorch CPU benchmark protocol after patching `cpp_libtorch_benchmark.cpp` to persist every measured call latency:

- model: PHC-lite step model TorchScript, seed=2 checkpoint
- batch size: 1
- warmup iterations: 100
- timing iterations: 10,000
- `torch::NoGradGuard`: enabled
- `torch::set_num_threads(8)` and `torch::set_num_interop_threads(8)`
- NEW5 usage: none

The patched benchmark writes per-call raw samples with schema `iteration, latency_us`.

## Files

| file | purpose |
|---|---|
| `cpu_baseline_timing_libtorch.json` | original LibTorch summary used in prior reporting |
| `cpu_baseline_timing_libtorch_raw.json` | first rerun summary with raw CSV enabled |
| `cpu_baseline_timing_libtorch_raw.csv` | first rerun raw 10,000 latency samples |
| `cpu_baseline_timing_libtorch_raw_rerun2.json` | second rerun summary with raw CSV enabled |
| `cpu_baseline_timing_libtorch_raw_rerun2.csv` | second rerun raw 10,000 latency samples |

## Summary Drift Versus Original

| rerun | mean_us | median_us | p99_us | min_us | mean_delta_vs_original | median_delta_vs_original | p99_delta_vs_original | min_delta_vs_original |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw rerun 1 | 63.9025 | 45.4 | 386.4 | 37.3 | -14.28% | -10.98% | -39.31% | -14.84% |
| raw rerun 2 | 64.0703 | 43.8 | 434.1 | 34.3 | -14.05% | -14.12% | -31.82% | -21.69% |

Original LibTorch summary: mean=74.5444us, median=51.0us, p99=636.7us, min=43.8us.

## Decision

The two raw reruns are internally similar, but they do **not** reproduce the original summary, especially in the tail. The p99 latency is 31.82% to 39.31% lower than the original report.

Therefore the raw files are valid empirical samples for the current rerun environment, but they should not be silently treated as the raw distribution behind the earlier reported `74.5444 / 51.0 / 636.7 / 43.8us` summary. Any latency-aware fill/PnL analysis using these raw files must explicitly state that the CPU latency distribution was recaptured and differs from the earlier summary.
