# PHC Diagonal SSM Training Convergence Report seed 2

This is a from-scratch clean training run. No 05C2/05C3/05C4/05D/05E weights or optimizer state are loaded.
This is a controlled recurrence-structure ablation versus 05C4: hidden_dim remains 16, but the dense recurrent core is replaced by a diagonal per-channel state transition. All other 05C4 training hyperparameters are retained.
- Best epoch: `157`.
- Best validation policy PnL: `1,771,525.00`.
- Early stop triggered: `True`.
- Hit max epoch 200: `False`.
- Convergence closed by strict rule (`early_stop=True and best_epoch<185`): `True`.
- LR scheduler: `CosineAnnealingLR`.
- LR at best epoch: `0.00012360`.
- LR at final epoch: `0.00004476`.
- Grad norm at best epoch: `0.075613`.
- Grad norm at final epoch: `0.070357`.
- Optimizer restarted: `False`.
- Phase: `05f`.

NEW3/NEW4/NEW5 were not used for epoch selection.
