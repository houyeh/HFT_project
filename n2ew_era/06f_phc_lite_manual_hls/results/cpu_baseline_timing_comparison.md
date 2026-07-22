# CPU Baseline Timing Comparison

Scope: batch_size=1 PHC step inference on CPU, warmup=100, timing=10,000 iterations, no NEW5 access.

| backend        | model                                     |   batch_size |   warmup_iters |   timing_iters |   mean_us |   median_us |   p99_us |   min_us | torch_no_grad   |   torch_num_threads |
|:---------------|:------------------------------------------|-------------:|---------------:|---------------:|----------:|------------:|---------:|---------:|:----------------|--------------------:|
| python_pytorch | PHC-lite step model (PyTorch float32)     |            1 |            100 |          10000 |   117.507 |        84.2 |    778.8 |     61.2 | True            |                   8 |
| libtorch_cpp   | PHC-lite step model (TorchScript float32) |            1 |            100 |          10000 |    74.544 |        51   |    636.7 |     43.8 | True            |                   8 |

Environment:
- Python/PyTorch baseline source: `n2ew_era/06f_phc_lite_manual_hls/results/cpu_baseline_timing.json`.
- LibTorch C++ baseline source: `n2ew_era/06f_phc_lite_manual_hls/results/cpu_baseline_timing_libtorch.json`.
- TorchScript export manifest: `n2ew_era/06f_phc_lite_manual_hls/results/torchscript_export_manifest.json`.
- CPU: `Intel64 Family 6 Model 189 Stepping 1, GenuineIntel`; hardware_concurrency=8.
- Thread setting: C++ `torch::set_num_threads(8)`, `torch::set_num_interop_threads(8)`; Python PyTorch reported 8 threads in this environment.

Notes:
- The C++ path uses TorchScript + `torch::NoGradGuard`; it removes Python call overhead but still executes the float32 PyTorch operator graph, not hand-written scalar C++ math.
- Both Python and LibTorch show min latency lower than median. This should be treated as timing jitter/cache/scheduler behavior rather than the representative latency. Median and p99 are safer for prose.
- This benchmark is a CPU software baseline only; it does not validate HLS or FPGA PnL claims.
