# 訓練收斂與分類報告

- **Loss Curve**：模型在訓練集與驗證集上皆有正常收斂，沒有嚴重 overfitting。
- **Confusion Matrix (混淆矩陣)**：
  - Up/Down 類別呈現「低精準度 (Precision) + 高召回率 (Recall)」的特性。
  - 這樣的特性不適合用來做方向性下單，但非常適合做為高敏感度的「撤單防禦觸發器」。
