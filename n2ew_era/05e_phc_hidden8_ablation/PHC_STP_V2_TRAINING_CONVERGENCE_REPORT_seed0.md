# PHC Hidden8 Ablation Training Convergence Report seed 0

This is a from-scratch clean training run. No 05C2/05C3/05C4/05D weights or optimizer state are loaded.
This is a controlled single-variable ablation versus 05C4: `HIDDEN_DIM=8` instead of `HIDDEN_DIM=16`. All other 05C4 training hyperparameters are retained.
- Best epoch: `171`.
- Best validation policy PnL: `1,929,985.00`.
- Early stop triggered: `True`.
- Hit max epoch 200: `False`.
- Convergence closed by strict rule (`early_stop=True and best_epoch<185`): `True`.
- LR scheduler: `CosineAnnealingLR`.
- LR at best epoch: `0.00006395`.
- LR at final epoch: `0.00001609`.
- Grad norm at best epoch: `0.062403`.
- Grad norm at final epoch: `0.077941`.
- Optimizer restarted: `False`.
- Phase: `05e`.

NEW3/NEW4/NEW5 were not used for epoch selection.
