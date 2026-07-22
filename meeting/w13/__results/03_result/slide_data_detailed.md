# Training and Classification — Verified Numbers

## Training Convergence
Source: `lt=.../train_outputs/train_history.csv`

| Epoch | train_loss | val_loss | val_acc |
|-------|-----------|----------|---------|
| 1  | 1.0797 | 1.0176 | 0.3246 |
| 10 | 0.8843 | 0.8127 | 0.5268 |
| 20 | 0.8642 | 0.7924 | 0.5232 |
| 25 | 0.8606 | 0.7903 | 0.6482 |
| 28 | 0.8604 | 0.7846 | 0.6489 |
| **30** | **0.8561** | **0.7814** | **0.5803** |

- Best epoch by val_loss: **30**, min val_loss = **0.7814**
- train_loss monotonically decreasing → no divergence
- val_loss also generally decreasing (epoch 30 is best) → no overfitting

## Classification Report (Test: 20260127Day + 20260127Night)
Source: `lt=.../eval_outputs/classification_report_test.csv`
Test samples: 233,268

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Down  | 0.028 | 0.694 | 0.054 | 2,762  |
| Stationary | 0.987 | 0.452 | 0.620 | 227,711 |
| Up    | 0.029 | 0.642 | 0.056 | 2,795  |
| macro avg | 0.348 | 0.596 | 0.243 | 233,268 |
| weighted avg | 0.964 | 0.457 | 0.606 | 233,268 |

## Cancel Filter Interpretation
- Down/Up precision ≈ 2.8%: NOT usable as directional alpha
- Down/Up recall ≈ 67%: captures majority of adverse moves → useful as cancel trigger
- Strategy: use as high-sensitivity cancel filter, not directional entry

## Sources
- `lt=.../train_outputs/train_history.csv`
- `lt=.../eval_outputs/classification_report_test.csv`
- `lt=.../eval_outputs/confusion_matrix_percent_test.png`
