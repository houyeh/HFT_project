# 08A Stage 1 v2 Interruption Diagnostic

status: `diagnosed_timeout_during_weighted_training`

## Evidence

- `results/training_curves_v2.csv` contained 11 rows, all for `horizon=10`, `seed=0`, `loss_mode=raw`.
- `results/stage1_metrics_v2.csv` contained exactly one completed metrics row: `horizon=10`, `seed=0`, `loss_mode=raw`.
- `models/phc_fi2010_h10_seed0_weighted_v2.pt` existed with checkpoint metadata `phase=08a_stage1_v2`, `horizon=10`, `seed=0`, `loss_mode=weighted`, `best_epoch=2`, `max_epochs=60`, `patience=10`, and `new5_used=False`.
- No dedicated stdout/stderr training log was written under the phase directory during the interrupted run.
- The previous execution was terminated by the runner wall-clock timeout after roughly 7204 seconds.

## Diagnosis

The scan did not stop because the weighted metrics writer silently failed. It was interrupted while `train_one()` was still inside the `horizon=10`, `seed=0`, `loss_mode=weighted` training call. The weighted best checkpoint was saved during training, but the function never reached its final validation/test evaluation and therefore never appended a row to `stage1_metrics_v2.csv`.

The deeper cause was runtime: the original full-window recurrent path recomputed all 100 timesteps for every sliding window on a CPU-only PyTorch environment. That made the requested `max_epochs=60`, `patience=10`, 5-horizon by 3-seed scan too slow for the interactive execution window.

## Fix

`02_train_phc_lite_stage1.py` now supports `--fast-exact-final-step`. This is a mathematical shortcut, not a model change: the current FI-2010 PHC-lite classifier has fixed zero `decay` and zero `alpha_scalar` buffers, so the final logits are independent of the first 99 timesteps and depend only on the last event in each 100-event window. The script verifies this before training by comparing full-window logits against the final-step shortcut and raises if the maximum absolute difference exceeds `--fast-equiv-atol`.

The v2 run should therefore be restarted from scratch with the same Stage 1 protocol but with `--fast-exact-final-step`, producing complete `_v2` outputs. Stage 2 HLS remains blocked until this complete v2 report is reviewed.
