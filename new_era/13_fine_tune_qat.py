import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
sys.path.insert(0, str(project_root))

from src.models.tcn import HLSFriendlyTCNClassifier
from new_era.tcn_pipeline_utils import (
    concatenate_records,
    load_sequence_records,
    split_records_by_session,
)


def load_qat_module():
    qat_path = project_root / "new_era" / "12_qat_tcn.py"
    spec = importlib.util.spec_from_file_location("qat_tcn_module", qat_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load QAT module from {qat_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_session_split_dataset():
    npz_path = project_root / "new_era" / "002_zscore_dataset.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    records = load_sequence_records(npz_path)
    train_records, val_records, test_records = split_records_by_session(records)
    if not val_records:
        raise ValueError("No validation sessions found. Re-run 02_feature_engineering.py first.")

    x_train, y_train = concatenate_records(train_records, channel_first=True)
    x_val, y_val = concatenate_records(val_records, channel_first=True)
    return x_train, x_val, y_train, y_val, test_records


def compute_class_weights(y_train, num_classes):
    counts = np.bincount(y_train, minlength=num_classes)
    total = counts.sum()
    weights = np.zeros(num_classes, dtype=np.float32)

    for idx, count in enumerate(counts):
        weights[idx] = 0.0 if count == 0 else total / (num_classes * count)

    return torch.tensor(weights, dtype=torch.float32)


def reset_batchnorm_layers(module):
    if isinstance(module, nn.BatchNorm1d):
        module.reset_running_stats()
        if module.affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def copy_module_params(src_module, dst_module, label, copied_labels):
    if src_module.weight.shape != dst_module.weight.shape:
        raise ValueError(
            f"Shape mismatch for {label}: "
            f"{tuple(src_module.weight.shape)} vs {tuple(dst_module.weight.shape)}"
        )

    dst_module.weight.copy_(src_module.weight.detach())
    if src_module.bias is not None and dst_module.bias is not None:
        if src_module.bias.shape != dst_module.bias.shape:
            raise ValueError(
                f"Bias shape mismatch for {label}: "
                f"{tuple(src_module.bias.shape)} vs {tuple(dst_module.bias.shape)}"
            )
        dst_module.bias.copy_(src_module.bias.detach())

    copied_labels.append(label)


def manually_map_fp32_to_qat(fp32_model, qat_model):
    copied_labels = []

    if len(fp32_model.tcn.network) != len(qat_model.tcn.network):
        raise ValueError("FP32 and QAT TCN depth do not match.")

    with torch.no_grad():
        for block_idx, (fp32_block, qat_block) in enumerate(
            zip(fp32_model.tcn.network, qat_model.tcn.network)
        ):
            copy_module_params(
                fp32_block.conv1.conv,
                qat_block.conv1,
                f"tcn.block{block_idx}.conv1",
                copied_labels,
            )
            copy_module_params(
                fp32_block.conv2.conv,
                qat_block.conv2,
                f"tcn.block{block_idx}.conv2",
                copied_labels,
            )

            if fp32_block.downsample is not None or qat_block.downsample is not None:
                if fp32_block.downsample is None or qat_block.downsample is None:
                    raise ValueError(
                        f"Residual projection mismatch in block {block_idx}."
                    )
                copy_module_params(
                    fp32_block.downsample,
                    qat_block.downsample,
                    f"tcn.block{block_idx}.downsample",
                    copied_labels,
                )

        copy_module_params(fp32_model.fc, qat_model.fc, "fc", copied_labels)

    return copied_labels


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_y.size(0)
            total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total_samples += batch_y.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def main():
    print("======================================================")
    print(" Phase 2: Fine-tune QAT from Profitable FP32 Weights ")
    print("======================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    seq_len = 100
    batch_size = 1024
    epochs = 10
    learning_rate = 1e-4

    fp32_weights_path = project_root / "new_era" / "005_tcn_weights.pth"
    output_path = project_root / "new_era" / "013_qat_finetuned_weights.pth"

    qat_module = load_qat_module()
    QATTCNClassifier = qat_module.QATTCNClassifier

    print("Loading dataset...")
    x_train, x_val, y_train, y_val, test_records = load_session_split_dataset()
    print(f"Dataset loaded: train X {x_train.shape}, val X {x_val.shape}, held-out test sessions={len(test_records)}")

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_weights = compute_class_weights(y_train, num_classes=3).to(device)
    print(f"Class weights: {class_weights.detach().cpu().tolist()}")

    print("Building FP32 source model...")
    fp32_model = HLSFriendlyTCNClassifier(
        num_inputs=8,
        num_classes=3,
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
        seq_length=seq_len,
    ).to(device)
    fp32_model.load_state_dict(load_state_dict(fp32_weights_path, device))
    fp32_model.eval()

    print("Building QAT target model...")
    qat_model = QATTCNClassifier(
        num_inputs=8,
        num_classes=3,
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
        seq_length=seq_len,
    ).to(device)

    print("Manually mapping FP32 weights into QAT layers...")
    copied_labels = manually_map_fp32_to_qat(fp32_model, qat_model)
    print("Mapped layers:")
    for label in copied_labels:
        print(f"  - {label}")

    print("Resetting BatchNorm layers to standard initialization...")
    qat_model.apply(reset_batchnorm_layers)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(qat_model.parameters(), lr=learning_rate)

    best_val_acc = -1.0

    print(f"Starting QAT fine-tuning for {epochs} epochs with lr={learning_rate}...")
    for epoch in range(epochs):
        qat_model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = qat_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_y.size(0)
            train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            train_samples += batch_y.size(0)

        train_loss = train_loss_sum / max(train_samples, 1)
        train_acc = train_correct / max(train_samples, 1)
        val_loss, val_acc = evaluate(qat_model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(qat_model.state_dict(), output_path)
            print(f"  -> Saved improved QAT weights to {output_path}")

    print("======================================================")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved fine-tuned QAT weights: {output_path}")
    print("======================================================")


if __name__ == "__main__":
    main()
