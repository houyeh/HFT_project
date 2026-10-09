# FI-2010 GPU Capacity Sweep Upload Manifest

Status: `prepared_not_uploaded_not_submitted`

This is the proposed minimum upload set for the approved **Step 0 probe only**. Every listed input belongs to the FI-2010 public benchmark. No TXF data, NEW1-NEW5 data, or outputs from phases 04b/04c/05c4/06 are included.

## Files to upload

| relative path | purpose | approximate size |
|---|---|---:|
| `fi2010_phc_lite.py` | v3 recurrent model definition | 5.6 KB |
| `02_train_phc_lite_stage1.py` | authoritative fair512 training/evaluation functions; uploaded read-only | 33 KB |
| `15_gpu_capacity_run.py` | isolated CUDA runner, environment capture, wall-time watchdog, weighted-recall identity gate | 13,657 B |
| `16_compare_probe_trajectory.py` | field-by-field GPU-vs-CPU comparison for epochs 1-3 | <4 KB |
| `slurm/step0_gpu_probe.sbatch` | one-GPU, hidden16/seed0, exactly three epochs | 3 KB |
| `requirements_gpu_sweep.txt` | package-name checklist; versions are recorded at runtime rather than silently installed | <1 KB |
| `cpu_hidden16_seed0_first3_reference.csv` | frozen CPU fair512 trajectory used by the post-probe comparison | <2 KB |
| `GPU_SWEEP_UPLOAD_SHA256.txt` | transfer-integrity hashes for every uploaded script and cache | <2 KB |
| `data/cache/train_cf7.npz` | FI-2010 Setup-2 train cache | 42,034,264 B |
| `data/cache/test_cf7.npz` | FI-2010 day-8 cache | 9,154,384 B |
| `data/cache/test_cf8.npz` | FI-2010 day-9 cache | 8,608,894 B |
| `data/cache/test_cf9.npz` | FI-2010 day-10 cache | 5,270,119 B |

Operational upload payload including the checksum file: **65,125,948 bytes**. The packaged audit/manifest/command documentation adds only a few kilobytes.

## Explicit exclusions

- All existing CPU checkpoints and result tables: retained locally as provenance, not needed by the three-epoch probe.
- The original 791 MB FI-2010 archive: the four verified caches are sufficient.
- All `n2ew_era/04b*`, `04c*`, `05c4*`, and `06*` files.
- All TXF data and all NEW1-NEW5 data or results.
- HLS, Vivado, bitstream, and board files.

## Probe contract

- Grid point: hidden16, seed0, horizon50.
- Hard stop: exactly 3 epochs in the Python runner.
- Scheduler stop: `#SBATCH --time=00:20:00`.
- Process watchdog: 900 seconds.
- Fixed recipe: batch512, raw cross-entropy, Adam, LR 1e-3, no scheduler, patience10, weight decay0, gradient clip5.
- CUDA absence is fatal; the runner never falls back to CPU.
- Outputs are isolated under `results/gpu_probe/` and `models/gpu_probe/`.

After the job finishes, obtain authoritative accounting with:

```bash
sacct -j <JOB_ID> -X -P \
  --format=JobID,JobName,State,ElapsedRaw,AllocTRES,Billing,TotalCPU,MaxRSS
```

The report will quote both measured GPU-hours (`elapsed * allocated GPU count`) and the cluster-provided `Billing`/SU field. The conversion from GPU-hours to SU is cluster policy and will not be guessed.

## Step 1 files intentionally deferred

The nine-run launcher, aggregator, capacity figure, and final report generator are not part of this upload yet. Step 1 requires approval after the measured probe cost.

The conflict has been resolved in favor of SU efficiency because the CPU fair512 per-epoch trajectory is available. After Step 0, the GPU and CPU first-three-epoch trajectories will be compared field by field. If the user accepts that comparison, Step 1 will launch all nine runs concurrently in one GPU allocation. The final hidden16 three-seed comparison will be a report item, not a mid-run stop gate. No Step 1 launcher is prepared or submitted before that approval.
