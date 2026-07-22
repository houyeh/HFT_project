import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat

project_root = Path(r'c:\Users\holbo\00VS_projjects\00_HFT_project')
sys.path.insert(0, str(project_root))
from new_era.tcn_pipeline_utils import (
    compute_class_weight_tensor,
    concatenate_records,
    load_sequence_records,
    split_records_by_session,
)

# =========================================================================
# QAT TCN Architecture Definition
# =========================================================================
class QATTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        
        # Conv 1
        self.conv1 = qnn.QuantConv1d(
            in_channels, out_channels, kernel_size, 
            dilation=dilation, padding=0,
            weight_quant=Int8WeightPerTensorFloat,
            return_quant_tensor=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels) # VERY IMPORTANT FOR QAT
        self.relu1 = qnn.QuantReLU(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.dropout1 = nn.Dropout(dropout)

        # Conv 2
        self.conv2 = qnn.QuantConv1d(
            out_channels, out_channels, kernel_size, 
            dilation=dilation, padding=0,
            weight_quant=Int8WeightPerTensorFloat,
            return_quant_tensor=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = qnn.QuantReLU(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.dropout2 = nn.Dropout(dropout)

        # Downsample for residual connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = qnn.QuantConv1d(
                in_channels, out_channels, 1,
                weight_quant=Int8WeightPerTensorFloat,
                return_quant_tensor=False
            )
        
        # Final activation after residual addition
        self.relu_out = qnn.QuantReLU(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)

        # Causal Padding Layers
        self.causal_pad1 = nn.ConstantPad1d((self.padding, 0), 0.0)
        self.causal_pad2 = nn.ConstantPad1d((self.padding, 0), 0.0)

    def forward(self, x):
        # Path 1: Convolution
        out = self.causal_pad1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.causal_pad2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Path 2: Residual (Identity)
        res = x if self.downsample is None else self.downsample(x)

        # Add and activate
        out = self.relu_out(out + res)
        return out


class QATTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, channel_list, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(len(channel_list)):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else channel_list[i - 1]
            out_ch = channel_list[i]
            layers.append(QATTemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class QATTCNClassifier(nn.Module):
    def __init__(self, num_inputs, num_classes, channel_list, kernel_size=3, dropout=0.1, seq_length=100):
        super().__init__()
        self.tcn = QATTemporalConvNet(num_inputs, channel_list, kernel_size, dropout)
        self.gap = nn.AvgPool1d(kernel_size=seq_length)
        
        # The fully connected layer is also quantized
        self.fc = qnn.QuantLinear(
            channel_list[-1], num_classes,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=False
        )

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x).squeeze(2)
        x = self.fc(x)
        return x

def main():
    print("======================================================")
    print(" Phase 1: QAT Model Software Training (100 Ticks) ")
    print("======================================================")
    
    # 1. Load Data
    npz_path = project_root / 'new_era' / '002_zscore_dataset.npz'
    records = load_sequence_records(npz_path)
    train_records, val_records, test_records = split_records_by_session(records)
    if not val_records:
        raise RuntimeError("No validation sessions found. Re-run 02_feature_engineering.py first.")

    print(f"Session split: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    SEQ_LEN = 100
    X_train, y_train = concatenate_records(train_records, channel_first=True)
    X_val, y_val = concatenate_records(val_records, channel_first=True)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # 2. Setup QAT Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QATTCNClassifier(
        num_inputs=8, 
        num_classes=3, 
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
        seq_length=SEQ_LEN
    ).to(device)

    # 3. Train Model
    BATCH_SIZE = 1024
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    class_weights = compute_class_weight_tensor(y_train, num_classes=3, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training QAT Model with {SEQ_LEN} ticks on {device}...")
    # Training for 20 epochs as a fast demonstration. For production, run 50+
    EPOCHS = 20
    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS} complete.")

    # 4. Save Weights
    weights_path = project_root / 'new_era' / '012_qat_tcn_weights.pth'
    torch.save(model.state_dict(), weights_path)
    print(f"QAT Model trained and saved to {weights_path}")

    # 5. Quick Accuracy Check
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_val_t.to(device)), dim=1).cpu().numpy()
        acc = (preds == y_val).mean()
        print(f"INT8 QAT TCN Accuracy on Validation Set: {acc:.4f}")

    print("\nNext Step: Run backtest script using the QAT model weights to verify software PnL!")

if __name__ == "__main__":
    main()
