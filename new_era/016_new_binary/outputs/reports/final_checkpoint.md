# Final Checkpoint

## 1. Label distribution
- bid toxic rate train/val/test: 0.035863 / 0.032294 / 0.027704
- ask toxic rate train/val/test: 0.030777 / 0.026889 / 0.026641

## 2. Best bid model
- val loss=1.003543, val PR-AUC=0.126072, test PR-AUC=0.072409, best threshold=0.60

## 3. Best ask model
- val loss=0.983688, val PR-AUC=0.108589, test PR-AUC=0.080254, best threshold=0.70

## 4. Best NEW test trading result
- net_pnl=266045.00, random mean+1std=272035.83, smoothed baseline net_pnl=84980.00, recommendation=not_ready

## 5. NEW3 result
- net_pnl=229385.00
- vs prior smoothed NEW3 model (-551705): +781090
- vs prior regime-gated NEW3 best (72520): +156865
- note: keep framework-alignment caution when comparing with external hybrid reports.

## 6. Final recommendation
- not_ready
