# PHC STP v2 Cosine LR Training Convergence Report seed 0

This is a from-scratch clean training run. No 05C2/05C3/05D weights or optimizer state are loaded.
The only protocol change versus 05C3 is `CosineAnnealingLR` with `eta_min=1e-5`.
- Best epoch: `167`.
- Best validation policy PnL: `1,990,575.00`.
- Early stop triggered: `True`.
- Hit max epoch 200: `False`.
- Convergence closed by strict rule (`early_stop=True and best_epoch<185`): `True`.
- LR scheduler: `CosineAnnealingLR`.
- LR at best epoch: `0.00007893`.
- LR at final epoch: `0.00002192`.
- Grad norm at best epoch: `0.095012`.
- Grad norm at final epoch: `0.083618`.
- Optimizer restarted: `False`.
- Phase: `05c4`.

NEW3/NEW4/NEW5 were not used for epoch selection.
