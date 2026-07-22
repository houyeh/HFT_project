"""
生成 Slide 9/10/11：Software/Hardware Pipeline 流程方塊圖
生成 Slide 14：HLS csynth 資源超限對比圖
生成 Slide 16：CME per-symbol PnL 對比圖
生成 Slide 11：Latency 層次對比圖
輸出：../figures/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Microsoft JhengHei", "Noto Sans TC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

DARK_BG   = "#FFFFFF"
PANEL_BG  = "#F8F9FA"  # 非常淡的灰色背景
ACCENT1   = "#0056b3"  # 深藍
ACCENT2   = "#166534"  # 深綠
ACCENT3   = "#991b1b"  # 深紅
ACCENT4   = "#854d0e"  # 深黃/棕
TEXT_COL  = "#000000"
GRID_COL  = "#DEE2E6"

# ─────────────────────────────────────────────────────
# Fig 1: Pipeline Diagram（Slide 9）
# ─────────────────────────────────────────────────────
def draw_pipeline():
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    def box(ax, x, y, w, h, label, sublabel="", color=ACCENT1, fontsize=9):
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.03", linewidth=1.5,
            edgecolor=color, facecolor=color + "22")
        ax.add_patch(rect)
        ax.text(x, y + 0.02 if sublabel else y, label,
                ha="center", va="center", color=color,
                fontsize=fontsize, fontweight="bold")
        if sublabel:
            ax.text(x, y - 0.07, sublabel, ha="center", va="center",
                    color=TEXT_COL, fontsize=7, alpha=0.8)

    def arrow(ax, x1, y1, x2, y2, color="#666666"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 2.8)

    # TW 主線（top row）
    nodes_tw = [
        (0.8, 2.3, "TW Replay\nCSV", "replay1_TW.csv", ACCENT2),
        (2.3, 2.3, "LOB 重建", "tw_replay.py", ACCENT2),
        (3.8, 2.3, "特徵工程\n(12 維)", "execution_features.py", ACCENT2),
        (5.3, 2.3, "Sliding Window\n(100×12)", "stride=10", ACCENT2),
        (6.8, 2.3, "TCN 訓練", "3 blocks, ch=16", ACCENT2),
        (8.3, 2.3, "Frozen Model", "frozen_tcn.pt", ACCENT4),
    ]
    for i, (x, y, lbl, sub, col) in enumerate(nodes_tw):
        box(ax, x, y, 1.2, 0.4, lbl, sub, col, fontsize=8)
        if i > 0:
            arrow(ax, nodes_tw[i-1][0]+0.6, y, x-0.6, y)

    # Hardware 主線（bottom）
    nodes_hw = [
        (8.3, 0.7, "INT8 量化匯出", ".wts + headers", ACCENT3),
        (6.8, 0.7, "C++ Fixed Core", "HLS-friendly", ACCENT3),
        (5.3, 0.7, "HLS Testbench", "golden logits", ACCENT3),
        (3.8, 0.7, "Vitis HLS\ncsim [PASS]", "csynth [超限]", ACCENT3),
        (2.3, 0.7, "Bit-Accurate\n驗證 [PASS]", "Python↔C++ logits", ACCENT1),
    ]
    for i, (x, y, lbl, sub, col) in enumerate(nodes_hw):
        box(ax, x, y, 1.2, 0.4, lbl, sub, col, fontsize=8)
        if i > 0:
            arrow(ax, nodes_hw[i-1][0]-0.6, y, x+0.6, y)

    # Vertical arrow: Frozen Model → INT8
    arrow(ax, 8.3, 2.1, 8.3, 0.9, ACCENT4)

    # CME 次線（middle）
    box(ax, 0.8, 1.5, 1.2, 0.35, "CME Logic20\nCSV", "ESM6, CLK6", "#9b59b6", 8)
    box(ax, 2.3, 1.5, 1.2, 0.35, "CME Adapter", "L3-L5 補齊", "#9b59b6", 8)
    box(ax, 3.8, 1.5, 1.2, 0.35, "Direct Validation\n❌ PnL<0", "TW frozen model", "#9b59b6", 8)
    arrow(ax, 0.8+0.6, 1.5, 2.3-0.6, 1.5)
    arrow(ax, 2.3+0.6, 1.5, 3.8-0.6, 1.5)
    # CME shares the feature engineering
    ax.annotate("", xy=(3.8, 2.1), xytext=(3.8, 1.68),
                arrowprops=dict(arrowstyle="->", color="#9b59b6", lw=1.2, linestyle="dashed"))
    ax.annotate("", xy=(8.3-0.6, 1.5), xytext=(6.8+0.6, 2.3),
                arrowprops=dict(arrowstyle="->", color="#9b59b6", lw=1.2, linestyle="dashed"))

    # Labels
    ax.text(0.1, 2.6, "▶ TW 主線", color=ACCENT2, fontsize=10, fontweight="bold")
    ax.text(0.1, 1.8, "▶ CME 次線", color="#9b59b6", fontsize=10, fontweight="bold")
    ax.text(0.1, 1.1, "▶ 硬體主線", color=ACCENT3, fontsize=10, fontweight="bold")

    ax.set_title("完整 Pipeline 方塊圖（Week 6–8）", color=TEXT_COL, fontsize=13, fontweight="bold", pad=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_pipeline.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print("Saved: fig_pipeline.png")

draw_pipeline()

# ─────────────────────────────────────────────────────
# Fig 2: HLS 資源超限對比圖（Slide 14）
# ─────────────────────────────────────────────────────
def draw_hls_resources():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)

    # Resource utilization bar chart
    ax1 = axes[0]
    ax1.set_facecolor(PANEL_BG)
    resources = ["BRAM", "DSP", "FF", "LUT"]
    utilization = [9, 194, 105, 410]
    colors = [ACCENT2 if u <= 100 else ACCENT3 for u in utilization]

    bars = ax1.barh(resources, utilization, color=colors, height=0.5, edgecolor=DARK_BG, linewidth=0.5)
    ax1.axvline(x=100, color=ACCENT4, linewidth=2, linestyle="--", label="100% limit")
    ax1.set_xlabel("Utilization (%)", color=TEXT_COL)
    ax1.tick_params(colors=TEXT_COL, labelsize=12)
    ax1.spines[:].set_color(GRID_COL)
    ax1.set_facecolor(PANEL_BG)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_color(TEXT_COL)
    ax1.set_xlim(0, 450)
    for bar, val in zip(bars, utilization):
        ax1.text(val + 5, bar.get_y() + bar.get_height()/2,
                 f"{val}%", va="center", color=TEXT_COL, fontsize=11, fontweight="bold")
    ax1.set_title("xc7z020 資源使用率（csynth 未優化）", color=TEXT_COL, fontsize=12, pad=8)
    ax1.legend(facecolor=PANEL_BG, labelcolor=TEXT_COL, fontsize=10)

    # Latency comparison（Slide 11 data embedded here too）
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    labels = ["PyTorch CPU\n(p50)", "Eng. Est.\n(FPGA)", "csynth\n未優化", "目標"]
    values_us = [446.5, 9.217, 932000, 50]  # all in µs
    bar_colors = [ACCENT1, ACCENT4, ACCENT3, ACCENT2]

    y_pos = np.arange(len(labels))
    bars2 = ax2.barh(y_pos, values_us, color=bar_colors, height=0.5, edgecolor="#000000", linewidth=0.5)
    ax2.set_xscale("log")
    ax2.axvline(x=50, color=ACCENT2, linewidth=2, linestyle="--", label="50µs 目標")
    ax2.set_xlabel("延遲（µs，對數軸）", color=TEXT_COL)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, color=TEXT_COL, fontsize=10)
    ax2.tick_params(colors=TEXT_COL)
    ax2.spines[:].set_color(GRID_COL)
    for label in ax2.get_xticklabels():
        label.set_color(TEXT_COL)

    for bar, val in zip(bars2, values_us):
        unit = "ms" if val >= 1000 else "µs"
        disp = val / 1000 if val >= 1000 else val
        ax2.text(val * 1.1, bar.get_y() + bar.get_height()/2,
                 f"{disp:.2f} {unit}", va="center", color=TEXT_COL, fontsize=10)

    ax2.set_title("延遲估算三層次對比（對數尺度）", color=TEXT_COL, fontsize=12, pad=8)
    ax2.legend(facecolor=PANEL_BG, labelcolor=TEXT_COL, fontsize=10)

    fig.suptitle("Vitis HLS 合成結果 與 延遲估算", color=TEXT_COL, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_hls_resources_and_latency.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print("Saved: fig_hls_resources_and_latency.png")

draw_hls_resources()

# ─────────────────────────────────────────────────────
# Fig 3: CME per-symbol PnL + confusion matrix（Slide 16/17）
# ─────────────────────────────────────────────────────
def draw_cme_results():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)

    # Subplot 1: PnL by symbol
    ax1 = axes[0]
    ax1.set_facecolor(PANEL_BG)
    symbols = ["ESM6", "CLK6", "Total"]
    pnl = [-1181.5, -35135.0, -36316.5]
    bar_colors = [ACCENT3 if p < 0 else ACCENT2 for p in pnl]
    bars = ax1.bar(symbols, pnl, color=bar_colors, edgecolor="#000000", linewidth=0.5, width=0.5)
    ax1.axhline(0, color="#000000", linewidth=1)
    ax1.set_ylabel("Net PnL (ticks)", color=TEXT_COL)
    ax1.set_title("Gated Signal PnL by Symbol", color=TEXT_COL, fontsize=11, pad=8)
    for bar, val in zip(bars, pnl):
        ax1.text(bar.get_x() + bar.get_width()/2, val - 500,
                 f"{val:,.0f}", ha="center", va="top", color=TEXT_COL, fontsize=10, fontweight="bold")
    ax1.tick_params(colors=TEXT_COL)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_color(TEXT_COL)
    ax1.spines[:].set_color(GRID_COL)

    # Subplot 2: Gated confusion matrix
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    cm = np.array([[370, 512, 591],
                   [5903, 54259, 4140],
                   [392, 557, 551]])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    im = ax2.imshow(cm_pct, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    labels_cls = ["Down", "Stationary", "Up"]
    ax2.set_xticklabels(["Pred\nDown", "Pred\nStation.", "Pred\nUp"], color="#000000", fontsize=9)
    ax2.set_yticklabels(["True\nDown", "True\nStation.", "True\nUp"], color="#000000", fontsize=9)
    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            pct = cm_pct[i, j]
            txt_color = "white" if pct > 50 else "black"
            ax2.text(j, i, f"{val:,}\n({pct:.0f}%)", ha="center", va="center",
                     color=txt_color, fontsize=8, fontweight="bold")
    ax2.set_title("Gated Confusion Matrix\n(ESM6 + CLK6)", color=TEXT_COL, fontsize=11, pad=8)
    ax2.tick_params(colors=TEXT_COL)

    # Subplot 3: Macro-F1 comparison
    ax3 = axes[2]
    ax3.set_facecolor(PANEL_BG)
    categories = ["隨機基準\n(Random)", "CME Gated\n(Transfer)", "目標\n(Symbol-Specific)"]
    f1_values = [0.333, 0.387, "> 0.5"]
    f1_nums = [0.333, 0.387, 0.5]
    bar_colors2 = ["#EEEEEE", ACCENT3, ACCENT4]
    bars3 = ax3.bar(categories, f1_nums, color=bar_colors2, edgecolor="#000000", linewidth=0.5, width=0.5)
    ax3.set_ylabel("Macro-F1", color=TEXT_COL)
    ax3.set_title("Macro-F1 對比", color=TEXT_COL, fontsize=11, pad=8)
    ax3.set_ylim(0, 0.7)
    ax3.axhline(0.5, color=ACCENT2, linewidth=1.5, linestyle="--", label="Edge 門檻 (0.5)")
    for bar, val, label in zip(bars3, f1_nums, [f"{v:.3f}" for v in f1_nums]):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 label if val < 0.45 else "> 0.5", ha="center",
                 color=TEXT_COL, fontsize=11, fontweight="bold")
    ax3.legend(facecolor=PANEL_BG, labelcolor=TEXT_COL, fontsize=9)
    ax3.tick_params(colors=TEXT_COL)
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_color(TEXT_COL)
    ax3.spines[:].set_color(GRID_COL)

    fig.suptitle("CME Direct Validation 結果：TW Frozen Model → CME（ESM6 + CLK6）",
                 color=TEXT_COL, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_cme_results.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print("Saved: fig_cme_results.png")

draw_cme_results()

# ─────────────────────────────────────────────────────
# Fig 4: Gating 機制示意圖（Slide 9/10）
# ─────────────────────────────────────────────────────
def draw_gating_diagram():
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    def box(x, y, w, h, main, sub="", color=ACCENT1):
        r = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.08", linewidth=1.5,
            edgecolor=color, facecolor=color + "22")
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2 + (0.1 if sub else 0), main,
                ha="center", va="center", color=color, fontsize=10, fontweight="bold")
        if sub:
            ax.text(x + w/2, y + h/2 - 0.25, sub, ha="center", va="center",
                    color=TEXT_COL, fontsize=8, alpha=0.85)

    def arrow(x1, x2, y, color="#666666"):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    # Model output
    box(0.2, 1.5, 1.6, 1.2, "TCN 輸出", "prob_up / prob_down\nprob_stationary", ACCENT1)
    arrow(1.8, 2.5, 2.1)

    # Gate 1
    box(2.5, 1.7, 1.8, 0.8, "① 機率門檻", "P_dir ≥ 0.5 + λ·spread\n+ λ·vol", ACCENT4)
    arrow(4.3, 5.0, 2.1)

    # Gate 2
    box(5.0, 1.7, 1.8, 0.8, "② Margin 門檻", "P_dir − P_stationary\n≥ 0", ACCENT4)
    arrow(6.8, 7.5, 2.1)

    # Gate 3
    box(7.5, 1.7, 1.8, 0.8, "③ Regime Guards", "Vol / Intensity\n/ Queue / Cancel", ACCENT3)
    arrow(9.3, 9.7, 2.1)

    # Output: Trade or No Trade
    ax.text(9.8, 2.5, "Trade", color=ACCENT2, fontsize=12, fontweight="bold", va="center")
    ax.text(9.8, 1.7, "No Trade", color=ACCENT3, fontsize=11, va="center")

    # CME annotation
    ax.text(5.0, 0.5, "CME 驗證結論：套用 Gating 後幾乎全部訊號被過濾，交易 edge 不足",
            color="#4A148C", fontsize=10, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5", edgecolor="#9b59b6"))

    ax.set_title("Gating 訊號過濾流程：從模型輸出到可交易訊號", color=TEXT_COL, fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_gating_diagram.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print("Saved: fig_gating_diagram.png")

draw_gating_diagram()

# ─────────────────────────────────────────────────────
# Fig 5: TCN 架構示意圖（Slide 7）
# ─────────────────────────────────────────────────────
def draw_tcn_arch():
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.5)

    def block(x, y, w, h, lbl, sub="", col=ACCENT1):
        r = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.05", linewidth=1.5, edgecolor=col, facecolor=col + "22")
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2 + (0.1 if sub else 0), lbl,
                ha="center", va="center", color=col, fontsize=9, fontweight="bold")
        if sub:
            ax.text(x + w/2, y + h/2 - 0.25, sub, ha="center", va="center",
                    color=TEXT_COL, fontsize=7.5)

    def arr(x1, x2, y, col="#666666"):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))

    # Input
    block(0.1, 1.5, 1.2, 1.2, "Input\n(100×12)", "100 events\n12 features", ACCENT2)

    # 3 Residual Blocks
    block_colors = [ACCENT1, "#a371f7", "#f78166"]
    for i, (dil, col) in enumerate(zip([1, 2, 4], block_colors)):
        bx = 2.0 + i * 3.5
        block(bx, 2.2, 3.0, 0.75, f"CausalConv1d", f"k=3, dil={dil}, ch=16", col)
        block(bx, 1.35, 3.0, 0.75, "CausalConv1d", f"k=3, dil={dil}, ch=16", col)
        block(bx, 0.5, 3.0, 0.75, f"Residual Block {i+1}", f"dilation={dil}", col)
        ax.text(bx + 1.5, 3.15, f"RF contrib: +{2*(3-1)*dil}", ha="center",
                color=col, fontsize=8, style="italic")
        if i > 0:
            arr(bx - 0.5, bx, 2.575, col)

    arr(1.3, 2.0, 2.575)

    # Global Avg Pool
    block(12.5, 1.7, 1.2, 0.85, "GlobalAvg\nPool", "seq→scalar", ACCENT4)
    arr(11.5, 12.5, 2.1)

    # FC + Softmax
    block(14.0, 1.7, 0.9, 0.85, "FC+\nSoftmax", "3 classes", ACCENT4)
    arr(13.7, 14.0, 2.1)

    # Receptive Field annotation
    ax.text(6.5, 4.1, "總感受野 (RF) = 1 + 2×(3-1)×(1+2+4) = 29 events",
            ha="center", color=ACCENT4, fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1000", edgecolor=ACCENT4))

    # Stats
    stats_text = "params: 4,771  |  MACs: 460,848  |  INT8 weights: 4.8 KB"
    ax.text(7.5, 0.05, stats_text, ha="center", color="#888ea8", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=PANEL_BG, edgecolor=GRID_COL))

    ax.set_title("TCN Classifier 架構（3 ResidualBlocks, dilation=1/2/4）", color=TEXT_COL, fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_tcn_arch.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print("Saved: fig_tcn_arch.png")

draw_tcn_arch()

print("\n✅ All figures generated successfully.")
