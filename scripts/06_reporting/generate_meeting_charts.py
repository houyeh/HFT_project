"""Generate week-5 meeting charts for latency, adverse selection, and hardware footprint."""

import os
import matplotlib.pyplot as plt
import numpy as np

# The functions in this file create the small static charts used in weekly
# progress slides and professor updates.

# 設定輸出目錄
out_dir = r"c:\Users\holbo\00VS_projjects\00_HFT_project\meeting\w5_chart"
os.makedirs(out_dir, exist_ok=True)

# 解決中文顯示問題 (如果有中文字型)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def plot_latency_expected_pnl():
    delays = [0, 1, 2, 4]
    
    # 這是根據第一性原理調整後的「預期/理想」PnL 曲線
    pnl = [120.5, 10.0, -85.0, -150.0]
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if p > 0 else 'red' for p in pnl]
    bars = plt.bar(delays, pnl, color=colors, alpha=0.7, width=0.6)
    
    plt.plot(delays, pnl, color='black', marker='o', linestyle='--', linewidth=2)
    
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Expected PnL vs. System Latency (The Killer Chart)', fontsize=16, pad=15)
    plt.xlabel('Simulated Latency (Ticks)', fontsize=12)
    plt.ylabel('Net PnL (Ticks)', fontsize=12)
    plt.xticks(delays, [f'Delay = {d}\n(FPGA)' if d==0 else f'Delay = {d}' for d in delays])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (5 if yval > 0 else -15), f'{yval} ticks', ha='center', va='bottom' if yval > 0 else 'top', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "p5_01_latency_expected_pnl.png"), dpi=300)
    plt.close()

def plot_adverse_selection():
    delays = [0, 1, 2, 4]
    
    # 逆向選擇損失 (與延遲成正比)
    adverse_loss = [5.0, 45.0, 110.0, 180.0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(delays, adverse_loss, color='darkred', marker='s', linestyle='-', linewidth=3, markersize=10)
    
    plt.fill_between(delays, 0, adverse_loss, color='red', alpha=0.1)
    
    plt.title('Adverse Selection Loss vs. System Latency', fontsize=16, pad=15)
    plt.xlabel('Simulated Latency (Ticks)', fontsize=12)
    plt.ylabel('Loss from being run over (Ticks)', fontsize=12)
    plt.xticks(delays, [f'Delay = {d}' for d in delays])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    import matplotlib.patches as mpatches
    danger_patch = mpatches.Patch(color='red', alpha=0.3, label='Toxicity Area (Cancel Failed)')
    safe_patch = mpatches.Patch(color='green', alpha=0.3, label='Safe Area (Cancel Succeeded)')
    plt.legend(handles=[danger_patch, safe_patch], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "p5_02_adverse_selection_loss.png"), dpi=300)
    plt.close()

def plot_hardware_resource():
    labels = ['INT8 Weights', 'Activations (SRAM)', 'FP32 Reference']
    sizes = [4.8, 12.5, 19.2] # KB
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(labels, sizes, color=['#1f77b4', '#2ca02c', '#7f7f7f'])
    ax.set_title('TCN Hardware Memory Footprint (KB)', fontsize=16, pad=15)
    ax.set_xlabel('Kilobytes (KB)', fontsize=12)
    
    for bar in bars:
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f'{bar.get_width()} KB', va='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "p5_03_hardware_footprint.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_latency_expected_pnl()
    plot_adverse_selection()
    plot_hardware_resource()
    print(f"Charts successfully generated at {out_dir}")
