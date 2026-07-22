# PHC Diagonal SSM Training Convergence Report seed 1

This is a from-scratch clean training run. No 05C2/05C3/05C4/05D/05E weights or optimizer state are loaded.
This is a controlled recurrence-structure ablation versus 05C4: hidden_dim remains 16, but the dense recurrent core is replaced by a diagonal per-channel state transition. All other 05C4 training hyperparameters are retained.
- Best epoch: `184`.
- Best validation policy PnL: `1,676,175.00`.
- Early stop triggered: `False`.
- Hit max epoch 200: `True`.
- Convergence closed by strict rule (`early_stop=True and best_epoch<185`): `False`.
- LR scheduler: `CosineAnnealingLR`.
- LR at best epoch: `0.00002754`.
- LR at final epoch: `0.00001006`.
- Grad norm at best epoch: `0.066784`.
- Grad norm at final epoch: `0.061183`.
- Optimizer restarted: `False`.
- Phase: `05f`.

NEW3/NEW4/NEW5 were not used for epoch selection.
