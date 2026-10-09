# N=4 CPU Benchmark Reconciliation

status: `reconciled_use_06f_n4_round1_for_demo_mouthpiece`

## Decision

Use `n2ew_era/06f_phc_lite_manual_hls/results/cpu_concurrent_benchmark_N4.csv` round 1 as the demo and paper mouthpiece for the N=4 concurrent CPU latency distribution:

`mean=72.198630us`, `p50=65.600us`, `p95=126.300us`, `p99=262.600us`, deadline `83.16us`, deadline miss rate `15.1250%`.

The 06S N=4 rows in `n2ew_era/06s_unroll16_verification_closure/results/final_latency_comparison.csv` remain valid provenance, but should be cited only as a noisy/stress closure run, not as the standard demo CPU baseline.

## Source Comparison

| item | 06F N=4 concurrent benchmark | 06S N=4 closure benchmark |
|---|---|---|
| Primary file | `n2ew_era/06f_phc_lite_manual_hls/results/cpu_concurrent_benchmark_N4.json` | `n2ew_era/06s_unroll16_verification_closure/results/final_latency_comparison.csv` |
| Raw samples | `n2ew_era/06f_phc_lite_manual_hls/results/cpu_concurrent_benchmark_N4.csv` | `n2ew_era/06s_unroll16_verification_closure/results/cpu/cpu_concurrent_N4_06s_raw.csv` |
| Executable | `cpp_libtorch_concurrent_benchmark.exe` | same executable via `cpu_benchmark_manifest_06s.json` |
| TorchScript checkpoint | `phc_stp_step_hls_seed2_torchscript.pt` | same checkpoint via `cpu_benchmark_manifest_06s.json` |
| Concurrency | 4 streams | 4 streams |
| Batch size | 1 | 1 |
| Warmup | 100 per stream | 100 per stream |
| Timed samples | 40,000 per round | 40,000 per round |
| Torch threads / interop | 8 / 8 | 8 / 8 |
| Host CPU | Intel Core Ultra 7 258V, 8 cores/logical processors | same CPU |
| Recorded host load | not captured in 06F JSON | before `44.57%`, after `94.77%` |
| Intended use | stable demo empirical latency distribution | 06S verification-closure stress context |

## Numeric Difference

| source | round | mean_us | p50_us | p95_us | p99_us | max_us | miss_rate_vs_83p16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 06F N=4 | 1 | 72.198630 | 65.600 | 126.300 | 262.600 | 7293.300 | 15.1250% |
| 06F N=4 | 2 | 70.616475 | 56.900 | 115.200 | 247.700 | 3647.000 | 15.1375% |
| 06S N=4 | 1 | 110.187375 | 83.400 | n/a | 646.318 | 18518.200 | 50.3775% |
| 06S N=4 | 2 | 242.064900 | 87.800 | n/a | 6299.416 | 81096.200 | 59.8125% |

## Diagnosis

The two measurements are not different because of the model, checkpoint, concurrency level, warmup count, or Torch thread settings; those match. The most likely cause of the p99 jump is host run context and background load during the 06S closure run. The 06S CPU snapshot already shows a high-load environment (`44.57%` before the CPU benchmark and `94.77%` after), with long-lived Python, Codex, Claude, iCloud, VS Code, and OneDrive processes present.

The strongest evidence that 06S was not a clean CPU baseline is that its N=1 rows are already heavily contaminated: p99 is `21019.382us` and `20191.726us`, far worse than both 06F N=1 and the 06F N=4 rows. Therefore the 06S N=4 p99 values should not silently replace the cleaner 06F N=4 distribution in demo material.

## Citation Rule

- Demo / oral presentation / staleness replay sampling: cite the 06F N=4 round 1 mouthpiece above.
- Robustness or stress discussion: cite the 06S closure rows explicitly as `06S noisy/stress closure N=4`, with both rounds shown.
- Do not mix 06F and 06S N=4 numbers in the same paragraph without naming the source phase.

No benchmark CSV was deleted or overwritten for this reconciliation.
