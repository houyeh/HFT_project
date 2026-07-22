import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

out_dir = Path("c:/Users/holbo/00VS_projjects/00_HFT_project/meeting/w13/01_docs")
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = "c:/Users/holbo/00VS_projjects/00_HFT_project/new_era/15_models_playaround/015_models_playaround_result/lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot/train_outputs/train_history.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2.5)
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=2.5)
    axes[0].set_title("Training & Validation Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].legend()
    
    # Mark min val loss
    min_val_loss = df["val_loss"].min()
    min_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
    axes[0].scatter(min_epoch, min_val_loss, color="red", zorder=5, s=100)
    axes[0].annotate(f"Min Val Loss: {min_val_loss:.4f}", (min_epoch, min_val_loss), 
                     xytext=(min_epoch, min_val_loss + 0.05),
                     fontsize=12, arrowprops=dict(facecolor='red', shrink=0.05))

    # Plot 2: Macro F1
    axes[1].plot(df["epoch"], df["train_macro_f1"], label="Train Macro F1", linewidth=2.5, color='green')
    axes[1].plot(df["epoch"], df["val_macro_f1"], label="Val Macro F1", linewidth=2.5, color='orange')
    axes[1].set_title("Training & Validation Macro F1", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1 Score")
    axes[1].legend()

    # Mark max val F1
    max_val_f1 = df["val_macro_f1"].max()
    max_epoch = df.loc[df["val_macro_f1"].idxmax(), "epoch"]
    axes[1].scatter(max_epoch, max_val_f1, color="red", zorder=5, s=100)
    axes[1].annotate(f"Max Val F1: {max_val_f1:.4f}", (max_epoch, max_val_f1), 
                     xytext=(max_epoch, max_val_f1 - 0.02),
                     fontsize=12, arrowprops=dict(facecolor='red', shrink=0.05))

    plt.tight_layout()
    plt.savefig(out_dir / "train_history_plot.png", dpi=300)
    print("Saved train_history_plot.png")
