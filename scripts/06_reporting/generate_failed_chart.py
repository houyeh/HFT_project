"""Generate the chart that documents the initial failed latency assumption."""

import os
import matplotlib.pyplot as plt
import numpy as np

# This figure is used to explain why an early latency narrative did not hold up
# once the queue-aware backtest was introduced.

# 設定輸出目錄
out_dir = r"c:\Users\holbo\00VS_projjects\00_HFT_project\meeting\w5_chart"
os.makedirs(out_dir, exist_ok=True)

# 解決中文顯示問題 (如果有中文字型)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def plot_latency_failed_initial():
    delays = [0, 1, 2, 4]
    
    # 原本失敗的回測數據 (因為過度防禦導致 0 trades)
    pnl = [-9.0, -1.0, 0.0, 0.0]
    trades = [2, 1, 0, 0]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    colors = ['darkred' if p < 0 else 'gray' for p in pnl]
    bars = ax1.bar(delays, pnl, color=colors, alpha=0.8, width=0.6, label='Net PnL')
    
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_title('Initial Setup: The "Dead End" of Over-Filtering', fontsize=16, pad=25)
    ax1.set_xlabel('Simulated Latency (Ticks)', fontsize=12)
    ax1.set_ylabel('Net PnL (Ticks)', fontsize=12, color='darkred')
    ax1.set_xticks(delays)
    ax1.set_xticklabels([f'Delay = {d}\n(FPGA)' if d==0 else f'Delay = {d}' for d in delays])
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 加入交易次數 (Trade Count) 到次坐標軸
    ax2 = ax1.twinx()
    ax2.plot(delays, trades, color='blue', marker='o', linestyle='-', linewidth=2, markersize=8, label='Trade Count')
    ax2.set_ylabel('Trade Count', fontsize=12, color='blue')
    ax2.set_ylim(-1, 5)
    
    # 標註 PnL 數值
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval - 1.5, f'{yval} ticks', ha='center', va='top', color='black', fontweight='bold')
        
    # 標註 Trades 數值
    for i, txt in enumerate(trades):
        ax2.annotate(f'{txt} trades', (delays[i], trades[i]), textcoords="offset points", xytext=(0,10), ha='center', color='blue', fontweight='bold')
        
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "p5_01b_latency_failed_initial.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_latency_failed_initial()
    print(f"Failed chart generated at {out_dir}")
