import os
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
sys.path.insert(0, str(project_root))

from src.models.tcn import HLSFriendlyTCNClassifier
from new_era.tcn_pipeline_utils import (
    concatenate_records,
    load_sequence_records,
    split_records_by_session,
)


def run_inference(model, x, device, batch_size=2048):
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch_X = torch.FloatTensor(x[i:i+batch_size]).to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            if i % (batch_size * 20) == 0:
                print(f"Processed {i}/{len(x)} samples...")
    return np.array(all_preds)


def print_report(split_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\n" + "="*50)
    print(f"TCN {split_name} Accuracy: {acc*100:.2f}%")
    print("="*50)
    print("\nConfusion Matrix:")
    print("               Predicted Down | Predicted Stat | Predicted Up")
    print(f"Actual Down  | {cm[0][0]:14d} | {cm[0][1]:14d} | {cm[0][2]:12d}")
    print(f"Actual Stat  | {cm[1][0]:14d} | {cm[1][1]:14d} | {cm[1][2]:12d}")
    print(f"Actual Up    | {cm[2][0]:14d} | {cm[2][1]:14d} | {cm[2][2]:12d}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Down (0)', 'Stationary (1)', 'Up (2)'], zero_division=0))

def main():
    """
    主程式：
    1. 載入先前訓練好的 TCN 模型權重與 Z-score 特徵資料。
    2. 確保使用與訓練階段完全相同的隨機種子 (random_state=42) 來切割出真正的 Validation Set。
    3. 對 Validation Set 進行批次推論 (Inference)。
    4. 計算準確率 (Accuracy) 並印出混淆矩陣 (Confusion Matrix) 與分類報告，評估模型在未見過資料上的預測能力。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    npz_path = project_root / "new_era" / "002_zscore_dataset.npz"
    weights_path = project_root / "new_era" / "005_tcn_weights.pth"
    
    if not npz_path.exists() or not weights_path.exists():
        print("Error: Dataset or weights file not found. Make sure training has started/finished.")
        sys.exit(1)
        
    print(f"Loading high-purity Z-score dataset from {npz_path}...")
    records = load_sequence_records(npz_path)
    _, val_records, test_records = split_records_by_session(records)

    if not val_records:
        print("Error: no validation sessions found. Re-run 02_feature_engineering.py first.")
        sys.exit(1)

    X_val, y_val = concatenate_records(val_records, channel_first=True)
    print(f"Validation Data Shape: X {X_val.shape}, y {y_val.shape}")
    
    # 初始化 TCN 模型結構，必須和訓練時完全一致
    model = HLSFriendlyTCNClassifier(
        num_inputs=8, 
        num_classes=3, 
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1
    ).to(device)
    
    print(f"Loading weights from {weights_path}...")
    # 載入模型權重
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval() # 設定為評估模式，關閉 Dropout 等干擾
    
    print("Running validation predictions...")
    val_preds = run_inference(model, X_val, device)
    print_report("Validation", y_val, val_preds)

    if test_records:
        X_test, y_test = concatenate_records(test_records, channel_first=True)
        print(f"Test Data Shape: X {X_test.shape}, y {y_test.shape}")
        print("Running test predictions...")
        test_preds = run_inference(model, X_test, device)
        print_report("Test", y_test, test_preds)

if __name__ == "__main__":
    # 程式進入點
    main()
