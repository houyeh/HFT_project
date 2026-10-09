# Capacity-Matched TCN vs PHC v3 Comparison

status: capacity_matched_tcn_small_complete_phc_hidden64_seed0_measured

## Scope

This report covers only FI-2010 `k=50`, `seed=0`. It does not use TXF NEW1-NEW5 and does not modify existing PHC v2/v3/fair512/speedprobe result files. Weighted-F1 rows are recomputed from existing checkpoints only; no model is retrained.

Seed1 was not launched in this turn. The seed0 hidden64 run reached natural early stopping, but it also consumed 10:10 wall-clock time and included one large epoch-duration anomaly; launching another hidden64 seed immediately would conflict with the resource-yield boundary for concurrent hardware/demo work.

## Direction A: Shrink TCN

Integer channel width alone cannot hit the +/-10% band: channels=7 gives 1,851 params (-13.18%), while channels=8 gives 2,355 params (+10.46%). The selected strict-band variant uses channels=8 and disables only input/readout bias, yielding 2344 params (+9.94%).

The TCN topology is unchanged: 5 residual causal-conv blocks, kernel=3, dilations `[1, 2, 4, 8, 16]`, receptive field 125, final-timestep readout. The `tcn_small` run uses raw cross entropy.

## Direction B: Expand PHC

PHC hidden64 now has a software-measured training result: best_epoch=27, final_epoch=37, val macro-F1=60.07%, val weighted-F1=60.40%, test macro-F1=64.43%, test weighted-F1=66.32%. The run used hidden_dim=64, batch_size=512, torch_threads=2, patience=10.

The hidden64 capacity run has 23,876 trainable parameters, which is 1/2.51 of the DeepLOB paper-rounded 60,000 parameter count and 1/5.97 of the 142,435-parameter public reference implementation count recorded in `EFFICIENCY_RESOURCE_TRADEOFF_REPORT.md`. The intended claim is therefore not that fewer parameters alone win; it is that PHC capacity/performance should be interpreted under the low-latency deployment feasibility constraint.

### Hidden64 Interruption Diagnosis

The previous hidden64 formal run left no persisted stdout/stderr log, checkpoint, or metrics CSV in this phase; only `results/parameter_count_v3_hidden64_probe.csv` and `results/capacity_phc_hidden64_speedprobe.csv` were present. The strongest codebase-local resource clue is `results/v3_fair512_resource_preflight.csv`: available memory was 2133 MB and the sampled process list included background OneDrive plus Anaconda Python processes. That does not prove OOM, but it makes the old exit unsuitable as evidence of a PHC hidden64 code bug.

For the new run, `results/hidden64_current_resource_preflight.csv` recorded 16834 MB available before launch. A `vitis_hls.exe` process was present, so the run kept `batch_size=512` for v2 comparability but limited PyTorch to 2 threads to reduce contention. The new run reached epoch-level logging under batch512, so immediate batch-size memory failure and pre-epoch code failure are ruled out for this configuration. Heartbeat anomaly summary: seed 0 epoch 10: 23844.4s (threshold 1065.7s).

## Main Comparison

| model | architecture | trainable_parameters | loss_mode | convergence_status | best_epoch | final_epoch | val macro-F1 | val weighted-F1 | test macro-F1 | test weighted-F1 | notes |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PHC v3 hidden16 fair512 | recurrent PHC-lite | 2,132 | raw | early_stop_closed | 22 | 32 | 57.22% | 57.79% | 62.31% | 65.18% | Existing fair512 seed0 run; weighted metrics from `stage1_metrics_v3_fair512_weighted_recompute.csv`. |
| PHC v3 hidden64 capacity | recurrent PHC-lite | 23,876 | raw | early_stop_closed | 27 | 37 | 60.07% | 60.40% | 64.43% | 66.32% | Hidden_dim=64, batch_size=512 capacity run; weighted metrics from `stage1_metrics_v3_hidden64_capacity_weighted_recompute.csv`. |
| TCN small | feed-forward causal TCN | 2,344 | raw_cross_entropy | early_stop_closed | 28 | 38 | 55.26% | 55.90% | 61.23% | 64.02% | channels=8, input/readout bias disabled to stay within +10% of PHC hidden16 parameters. |
| TCN scratch width16 | feed-forward causal TCN | 8,547 | class_weighted_cross_entropy | early_stop_closed | 41 | 51 | 58.12% | 58.60% | 65.20% | 67.60% | Previously completed wider TCN reference. |

## Interpretation

With parameter count aligned, the previous TCN advantage shrinks and slightly reverses under both F1 definitions. TCN small reaches 61.23% test macro-F1 and 64.02% test weighted-F1 versus PHC v3 hidden16 fair512 at 62.31% macro-F1 and 65.18% weighted-F1. Because the current fair512 CSV reports natural early stopping, this is a cleaner capacity-controlled single-seed result than the earlier partial report. It shows that the width16 TCN edge was materially affected by capacity: the 8,547-param TCN reaches 65.20% macro-F1 / 67.60% weighted-F1, while the 2,344-param TCN small falls to 61.23% / 64.02%.

The honest current answer is: after parameter alignment, the architecture gap mostly collapses and slightly reverses within this one k=50/seed0 slice. TCN small does not beat the converged PHC hidden16 result under macro-F1 or weighted-F1; the wider TCN does. Hidden64 is now framed as a deployability-constrained capacity probe: it reaches 64.43% macro-F1 / 66.32% weighted-F1, but its 23,876 trainable parameters remain materially below the published DeepLOB parameter counts used elsewhere in this phase and materially above the PHC hidden16 / TCN small capacity-matched pair.

## Files

- Summary CSV: `results/capacity_matched_comparison.csv`
- TCN channel selection: `results/tcn_small_channel_selection.csv`
- TCN small metrics: `results/tcn_fi2010_metrics_small_raw.csv`
- PHC hidden64 synthetic speedprobe: `results/capacity_phc_hidden64_speedprobe.csv`
- PHC hidden64 parameter count: `results/parameter_count_v3_hidden64_probe.csv`
- PHC hidden64 capacity helper: `13_train_hidden64_capacity.py`
- PHC hidden64 capacity stdout/stderr logs: `results/train_v3_hidden64_capacity_seed0.log`, `results/train_v3_hidden64_capacity_seed0.err.log`
- PHC hidden64 current preflight: `results/hidden64_current_resource_preflight.csv`
- PHC hidden64 metrics, if completed: `results/stage1_metrics_v3_hidden64_capacity.csv`
- PHC hidden64 weighted recompute metrics: `results/stage1_metrics_v3_hidden64_capacity_weighted_recompute.csv`
- PHC hidden64 weighted recompute summary: `results/stage1_summary_v3_hidden64_capacity_weighted_recompute.csv`
- PHC hidden64 heartbeat, if completed: `results/training_heartbeat_v3_hidden64_capacity.csv`

## Claim Boundary

- This is a single-horizon, single-seed control, not a 3-seed benchmark.
- `TCN small` is converged by early stopping; PHC hidden16 fair512 convergence status is read from the current CSV.
- PHC hidden64 is reported as a performance point only if `results/stage1_metrics_v3_hidden64_capacity.csv` exists; otherwise only parameter count and speed/cost probes are reported.
- No HLS, P&R, board work, distillation, or NEW1-NEW5 data are involved.
