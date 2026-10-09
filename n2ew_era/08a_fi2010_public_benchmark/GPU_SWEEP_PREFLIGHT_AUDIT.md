# FI-2010 GPU Sweep Preflight Audit

Status: `step0_ready_not_submitted`

## Cache provenance

Result: **PASS using the permitted path-and-shape method.** The historical CPU fair512 run did not record SHA-256 hashes of the generated NPZ cache files themselves, so this audit does not claim a historical NPZ-hash match.

Evidence:

- `02_train_phc_lite_stage1.py` defines `CACHE = ROOT / "data" / "cache"` and `load_cache()` reads exactly `train_cf7.npz`, `test_cf7.npz`, `test_cf8.npz`, and `test_cf9.npz`.
- The CPU fair512 logs identify phase `08a_stage1_v3_fair512` and were written on 2026-08-04 through 2026-08-05.
- All four cache files were created and last modified on 2026-07-28. Their modification times therefore predate the CPU run and show no later rewrite.
- Current NPZ shapes exactly match `results/file_manifest.csv`:

| cache | feature shape | label shape | current SHA-256 |
|---|---:|---:|---|
| `train_cf7.npz` | `254750x40` | `254750x5` | `4e4717de95df70ee55d82d4f822db01c1277ef24e1a150020e1fd1500cf166f2` |
| `test_cf7.npz` | `55478x40` | `55478x5` | `23ea6280181ef3cf2616c35d7494e48f57c1d9a71b91e4161870687ded30e631` |
| `test_cf8.npz` | `52172x40` | `52172x5` | `25e67ccf09b1807f1dd3d0e4bf66aebac0d089fde5b337f7a09216bb9915e548` |
| `test_cf9.npz` | `31937x40` | `31937x5` | `4aeeff9b5428c3c7b71a1650136a00f57eb5b9357a5fa4bab7628a079591251b` |

`results/file_manifest.csv` stores SHA-256 for the extracted source text files, not for the NPZ products. Those source hashes were deliberately not substituted for NPZ hashes here.

## CPU per-epoch trajectory availability

Result: **AVAILABLE.** `results/training_curves_v3_fair512.csv` contains 177 rows: seed0 has epochs 1-32, seed1 has epochs 1-89, and seed2 has epochs 1-56. Each row includes train loss, validation loss, validation accuracy, validation macro precision, validation macro recall, and validation macro F1.

The hidden16/seed0 CPU epochs 1-3 were copied without recomputation into `cpu_hidden16_seed0_first3_reference.csv`. The Step 0 job writes the analogous GPU values plus per-epoch wall time to `results/gpu_probe/cap16_seed0/probe_epoch_metrics.csv`, then writes a field-by-field delta table to `cpu_gpu_first3_trajectory_comparison.csv`.

Under the user's conflict ruling, this evidence selects the **single-wave nine-process Step 1 design** if the Step 0 trajectory is accepted. The final hidden16 three-seed device comparison remains a report item rather than a mid-run stop gate. Step 1 is still not prepared or submitted pending explicit approval after Step 0.
