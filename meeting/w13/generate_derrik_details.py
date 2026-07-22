import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DERRIK_ROOT = Path("c:/Users/holbo/00VS_projjects/00_HFT_project/meeting/w13/___Derrik_results")
MODEL_DIR = Path("c:/Users/holbo/00VS_projjects/00_HFT_project/new_era/15_models_playaround/015_models_playaround_result/lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot")

def split_ordered_session_ids(session_ids, train_frac, val_frac):
    ordered = sorted(session_ids)
    total = len(ordered)
    train_end = max(1, int(total * train_frac))
    val_end = max(train_end + 1, int(total * (train_frac + val_frac)))
    val_end = min(val_end, total - 1)
    return ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]

def do_02():
    manifest = json.loads((MODEL_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
    cfg = json.loads((MODEL_DIR / "experiment_config.json").read_text(encoding="utf-8"))
    
    sessions = list(manifest.get("sessions", []))
    class_dist = manifest.get("class_distribution_by_session", {})
    split_cfg = cfg.get("split", {})
    train_frac = float(split_cfg.get("train_session_frac", 0.7))
    val_frac = float(split_cfg.get("val_session_frac", 0.15))
    train_sessions, val_sessions, test_sessions = split_ordered_session_ids(sessions, train_frac, val_frac)
    
    def count(ss):
        total = 0
        for s in ss:
            d = class_dist.get(s, {})
            total += int(d.get("0", 0)) + int(d.get("1", 0)) + int(d.get("2", 0))
        return total

    lines = ["# 資料集詳細使用清單 (Dataset Details)", ""]
    lines.append(f"**資料來源目錄**: `{cfg.get('data', {}).get('data_dir', 'MISSING')}`")
    lines.append(f"**商品代號 (Symbol)**: `{cfg.get('data', {}).get('symbols', ['MISSING'])[0]}`")
    lines.append("")
    
    lines.append(f"## Train Split ({train_frac*100}%) - {count(train_sessions):,} 筆資料")
    lines.append("使用的實際檔案（Sessions）：")
    for s in train_sessions:
        lines.append(f"- {s}.csv (筆數: {sum(int(v) for v in class_dist.get(s, {}).values()):,})")
    
    lines.append("")
    lines.append(f"## Validation Split ({val_frac*100}%) - {count(val_sessions):,} 筆資料")
    lines.append("使用的實際檔案（Sessions）：")
    for s in val_sessions:
        lines.append(f"- {s}.csv (筆數: {sum(int(v) for v in class_dist.get(s, {}).values()):,})")
        
    lines.append("")
    lines.append(f"## Test Split ({(1-train_frac-val_frac)*100:.0f}%) - {count(test_sessions):,} 筆資料")
    lines.append("使用的實際檔案（Sessions）：")
    for s in test_sessions:
        lines.append(f"- {s}.csv (筆數: {sum(int(v) for v in class_dist.get(s, {}).values()):,})")

    out_file = DERRIK_ROOT / "02_result" / "data_source_details.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")
    
def do_03():
    train_history = DERRIK_ROOT / "03_result" / "train_history.csv"
    if train_history.exists():
        df = pd.read_csv(train_history)
        if "val_loss" in df.columns:
            best_idx = df["val_loss"].idxmin()
            best_row = df.loc[best_idx]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(df["epoch"], df["train_loss"], label="Train Loss", color="tab:blue")
            ax.plot(df["epoch"], df["val_loss"], label="Val Loss", color="tab:orange")
            
            # Annotation
            ax.scatter([best_row["epoch"]], [best_row["val_loss"]], color="red", zorder=3, s=100)
            
            # Label
            ax.annotate(
                f"Best Val Loss: {best_row['val_loss']:.4f}\n(Epoch {int(best_row['epoch'])})",
                (best_row["epoch"], best_row["val_loss"]),
                xytext=(10, 40), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='red', lw=2),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", lw=2)
            )
            
            ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(DERRIK_ROOT / "03_result" / "training_loss_plot_annotated.png", dpi=150)
            plt.close(fig)

def do_10():
    df_path = DERRIK_ROOT / "10_result" / "cancel_delay_threshold_055.csv"
    if df_path.exists():
        df = pd.read_csv(df_path)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["delay_events"], df["net_pnl"], marker="o", color="tab:blue")
        for i, row in df.iterrows():
            ax.annotate(
                f"{row['net_pnl']:.0f}",
                (row["delay_events"], row["net_pnl"]),
                xytext=(0, 8), textcoords="offset points", ha='center', fontsize=9
            )
        ax.set_title("Cancel Delay vs Net PnL (Threshold=0.55)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Cancel Delay (ticks)", fontsize=12)
        ax.set_ylabel("Net PnL (TWD)", fontsize=12)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(DERRIK_ROOT / "10_result" / "cancel_delay_net_pnl_annotated.png", dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    do_02()
    do_03()
    do_10()
    print("Done generating detailed data and annotated charts.")
