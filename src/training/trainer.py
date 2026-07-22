"""Training, validation, and prediction helpers for the TCN experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.training.metrics import classification_metrics


@dataclass
class TrainerConfig:
    epochs: int = 6
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 2
    device: str = "cpu"


def select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _move_batch(batch: dict, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    sample_weight = batch.get("sample_weight", torch.ones_like(y, dtype=torch.float32)).to(device)
    return x, y, sample_weight


def _run_train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    # 一個 epoch = 全部 train_loader 的 batch 各跑一次 forward/backward/update
    model.train()
    total_loss = 0.0
    total_weight = 0.0

    for batch in dataloader:
        x, y, sample_weight = _move_batch(batch, device)
        # optimizer 負責更新權重；每個 batch 前先清空梯度
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        # 這裡的 criterion 是 CrossEntropyLoss(reduction="none")，
        # 先得到每筆樣本 loss，再乘 sample_weight 做加權平均。
        per_sample_loss = criterion(logits, y)
        loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()

        total_loss += float((per_sample_loss * sample_weight).sum().item())
        total_weight += float(sample_weight.sum().item())

    return total_loss / max(total_weight, 1.0)


def evaluate_model(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: str,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            x, y, sample_weight = _move_batch(batch, device)
            logits = model(x)
            per_sample_loss = criterion(logits, y)

            predictions = logits.argmax(dim=1)
            total_loss += float((per_sample_loss * sample_weight).sum().item())
            total_weight += float(sample_weight.sum().item())
            y_true.append(y.cpu().numpy())
            y_pred.append(predictions.cpu().numpy())

    y_true_np = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    y_pred_np = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)
    metrics = classification_metrics(y_true_np, y_pred_np)
    metrics["loss"] = total_loss / max(total_weight, 1.0)
    return metrics


def predict_model(model: nn.Module, dataloader, device: str) -> pd.DataFrame:
    model.eval()
    records = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            truths = batch["y"].cpu().numpy()
            timestamps = batch["timestamp"].cpu().numpy()
            future_returns = batch["future_return_ticks"].cpu().numpy()
            spreads = batch["spread_ticks"].cpu().numpy()
            tick_sizes = batch["tick_size"].cpu().numpy()
            rolling_vols = batch["rolling_volatility"].cpu().numpy()
            current_mid = batch["current_mid"].cpu().numpy()
            future_mid = batch["future_mid"].cpu().numpy()
            bid_px_1 = batch["bid_px_1"].cpu().numpy()
            ask_px_1 = batch["ask_px_1"].cpu().numpy()
            bid_vol_1 = batch["bid_vol_1"].cpu().numpy()
            ask_vol_1 = batch["ask_vol_1"].cpu().numpy()
            queue_imbalance = batch["queue_imbalance"].cpu().numpy()
            queue_imbalance_change = batch["queue_imbalance_change"].cpu().numpy()
            trade_intensity_short = batch["trade_intensity_short"].cpu().numpy()
            aggressor_imbalance_interval = batch["aggressor_imbalance_interval"].cpu().numpy()
            bid_cancel_proxy = batch["bid_cancel_proxy"].cpu().numpy()
            ask_cancel_proxy = batch["ask_cancel_proxy"].cpu().numpy()
            net_cancel_pressure = batch["net_cancel_pressure"].cpu().numpy()
            cancel_pressure_short = batch["cancel_pressure_short"].cpu().numpy()
            symbols = batch["symbol"]

            for idx in range(len(predictions)):
                records.append(
                    {
                        "symbol": symbols[idx],
                        "timestamp": int(timestamps[idx]),
                        "y_true": int(truths[idx]),
                        "y_pred": int(predictions[idx]),
                        "prob_down": float(probabilities[idx, 0]),
                        "prob_stationary": float(probabilities[idx, 1]),
                        "prob_up": float(probabilities[idx, 2]),
                        "future_return_ticks": float(future_returns[idx]),
                        "spread_ticks": float(spreads[idx]),
                        "tick_size": float(tick_sizes[idx]),
                        "rolling_volatility": float(rolling_vols[idx]),
                        "current_mid": float(current_mid[idx]),
                        "future_mid": float(future_mid[idx]),
                        "bid_px_1": float(bid_px_1[idx]),
                        "ask_px_1": float(ask_px_1[idx]),
                        "bid_vol_1": float(bid_vol_1[idx]),
                        "ask_vol_1": float(ask_vol_1[idx]),
                        "queue_imbalance": float(queue_imbalance[idx]),
                        "queue_imbalance_change": float(queue_imbalance_change[idx]),
                        "trade_intensity_short": float(trade_intensity_short[idx]),
                        "aggressor_imbalance_interval": float(aggressor_imbalance_interval[idx]),
                        "bid_cancel_proxy": float(bid_cancel_proxy[idx]),
                        "ask_cancel_proxy": float(ask_cancel_proxy[idx]),
                        "net_cancel_pressure": float(net_cancel_pressure[idx]),
                        "cancel_pressure_short": float(cancel_pressure_short[idx]),
                    }
                )

    return pd.DataFrame(records)


def train_with_early_stopping(
    model: nn.Module,
    train_loader,
    val_loader,
    config: TrainerConfig,
    output_dir: str | Path,
    class_weights: torch.Tensor | None = None,
) -> tuple[nn.Module, pd.DataFrame]:
    device = config.device
    model.to(device)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    # loss: CrossEntropy 比較 logits 與真實類別 y 的距離（分類誤差）
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    # optimizer: Adam 根據 loss 的梯度調整 model.parameters()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    # epoch 迴圈：每一圈都會完整掃過一次訓練資料
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training finished without a best model state.")

    model.load_state_dict(best_state)
    torch.save(best_state, output_path / "best_model.pt")
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_path / "history.csv", index=False)
    pd.Series(asdict(config)).to_json(output_path / "train_config.json", indent=2)
    return model, history_df
