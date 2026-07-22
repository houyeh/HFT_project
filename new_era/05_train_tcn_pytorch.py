import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
sys.path.insert(0, str(project_root))

from src.models.tcn import HLSFriendlyTCNClassifier
from new_era.tcn_pipeline_utils import (
    build_loader,
    compute_class_weight_tensor,
    concatenate_records,
    load_sequence_records,
    split_records_by_session,
)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    return val_loss / max(len(val_loader), 1), correct / max(total, 1)

def main():
    """
    主程式：
    1. 讀取 Z-score 正規化後的 .npz 特徵檔案。
    2. 針對 PyTorch 卷積層的需求，進行張量維度轉換 (Transpose)。
    3. 計算類別權重 (Class Weights) 以對抗市場資料的極度不平衡。
    4. 建立 DataLoader 以批次方式輸入資料。
    5. 初始化專為 HLS (硬體合成) 設計的 TCN (時間卷積網路) 模型。
    6. 使用 Adam 優化器與 CrossEntropyLoss 進行模型訓練，並儲存最佳權重。
    """
    # 確認是否有 GPU 可用，否則退回 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    npz_path = project_root / "new_era" / "002_zscore_dataset.npz"
    if not npz_path.exists():
        print(f"Error: {npz_path} not found.")
        sys.exit(1)
        
    print(f"Loading high-purity Z-score dataset from {npz_path}...")
    records = load_sequence_records(npz_path)
    train_records, val_records, test_records = split_records_by_session(records)

    if not val_records:
        print("Error: no validation sessions were available. Re-run 02_feature_engineering.py with multiple sessions.")
        sys.exit(1)

    print(f"Session split: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")
    for split_name, split_records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        sessions = ", ".join(record.key for record in split_records)
        print(f"  {split_name}: {sessions}")

    # PyTorch Conv1d 使用 (Batch, Features, Time)，因此載入後統一轉為 channel-first。
    X_train, y_train = concatenate_records(train_records, channel_first=True)
    X_val, y_val = concatenate_records(val_records, channel_first=True)
    print(f"Train Data: X shape {X_train.shape}, y shape {y_train.shape}")
    print(f"Validation Data: X shape {X_val.shape}, y shape {y_val.shape}")
    
    # 計算 Class Weights 處理不平衡 (Up/Down vs Stationary)
    weights_tensor = compute_class_weight_tensor(y_train, num_classes=3, device=device)
    print(f"Class Weights applied: {weights_tensor.detach().cpu().tolist()}")
    
    # 建立 PyTorch Dataset 與 DataLoader 以支援批次化 (Batch) 訓練
    train_loader = build_loader(X_train, y_train, batch_size=1024, shuffle=True)
    val_loader = build_loader(X_val, y_val, batch_size=1024, shuffle=False)
    
    # 建立 TCN 模型
    # 輸入特徵 8 個，分類 3 個 (上漲/下跌/盤整)
    # TCN 通道數設為 [16, 16, 16]，保持網路極度輕量化以確保 FPGA 能塞得下並高速運算
    model = HLSFriendlyTCNClassifier(
        num_inputs=8, 
        num_classes=3, 
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1
    ).to(device)
    
    print(f"Model Parameters: {model.count_parameters()}")
    
    # 定義損失函數 (帶入權重以加重少數類別的懲罰) 與優化器
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20 ############ 0507: 改成25 看收斂的狀況?
    best_val_acc = 0.0
    model_save_path = project_root / "new_era" / "005_tcn_weights.pth"
    
    print("\nStarting PyTorch TCN Training...")
    # 開始 Epoch 訓練迴圈
    for epoch in range(epochs):
        model.train() # 切換為訓練模式 (啟動 Dropout 等)
        train_loss = 0.0
        correct = 0
        total = 0
        
        # 遍歷所有訓練 Batch
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()         # 梯度歸零
            outputs = model(batch_X)      # 前向傳播 (Forward pass)
            loss = criterion(outputs, batch_y) # 計算損失
            loss.backward()               # 反向傳播 (Backward pass)
            optimizer.step()              # 更新權重
            
            train_loss += loss.item()
            _, predicted = outputs.max(1) # 取機率最大的類別當作預測結果
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
        train_acc = correct / total
        
        # Validation 驗證階段
        avg_val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 如果驗證準確率有提升，就儲存模型權重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            
    print(f"\nTraining completed. Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"PyTorch TCN model saved to: {model_save_path}")

if __name__ == "__main__":
    # 程式進入點
    main()
