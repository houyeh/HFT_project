# Training Reliability Report

## bid model
- Did bid model converge? YES
- Does PR-AUC exceed positive base rate? YES (test_pr_auc=0.072409, base_rate=0.027704)
- Is there overfitting risk? NO
- Are probabilities degenerate? NO
- Is model usable for backtest? YES
- Best epoch=28, min_val_loss=1.003543, best_threshold=0.60, best_threshold_f1=0.134432

## ask model
- Did ask model converge? YES
- Does PR-AUC exceed positive base rate? YES (test_pr_auc=0.080254, base_rate=0.026641)
- Is there overfitting risk? NO
- Are probabilities degenerate? NO
- Is model usable for backtest? YES
- Best epoch=30, min_val_loss=0.983688, best_threshold=0.70, best_threshold_f1=0.152965

Notes:
- This report evaluates pos_weight configuration only (no unweighted run in this pass).