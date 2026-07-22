# 訓練與分類 — 驗證數據

## 訓練收斂情況
來源檔案: `lt=.../train_outputs/train_history.csv`

| Epoch | train_loss | val_loss | val_acc |
|-------|-----------|----------|---------|
| 1  | 1.0797 | 1.0176 | 0.3246 |
| 10 | 0.8843 | 0.8127 | 0.5268 |
| 20 | 0.8642 | 0.7924 | 0.5232 |
| 25 | 0.8606 | 0.7903 | 0.6482 |
| 28 | 0.8604 | 0.7846 | 0.6489 |
| **30** | **0.8561** | **0.7814** | **0.5803** |

- 由 val_loss 選出的最佳 epoch: **30**, 最低 val_loss = **0.7814**
- train_loss 單調遞減 → 無發散
- val_loss 也整體呈現遞減 (epoch 30 為最佳) → 無過度擬合

## 分類報告 (Test: 20260127Day + 20260127Night)
來源檔案: `lt=.../eval_outputs/classification_report_test.csv`
測試樣本數: 233,268

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Down  | 0.028 | 0.694 | 0.054 | 2,762  |
| Stationary | 0.987 | 0.452 | 0.620 | 227,711 |
| Up    | 0.029 | 0.642 | 0.056 | 2,795  |
| macro avg | 0.348 | 0.596 | 0.243 | 233,268 |
| weighted avg | 0.964 | 0.457 | 0.606 | 233,268 |

## 撤單過濾器解讀
- Down/Up 精準度 (Precision) ≈ 2.8%: 不可用作主動方向性訊號
- Down/Up 召回率 (Recall) ≈ 67%: 捕捉了多數的不利價格移動 → 適合作為撤單觸發器
- 策略定位：作為高敏感度的撤單過濾器，而非方向性進場點

## 資料來源
- `lt=.../train_outputs/train_history.csv`
- `lt=.../eval_outputs/classification_report_test.csv`
- `lt=.../eval_outputs/confusion_matrix_percent_test.png`
