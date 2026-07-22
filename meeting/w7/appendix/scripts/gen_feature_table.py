"""
生成 Slide 4/5：Feature 定義表格圖表
輸出：../figures/fig_feature_table.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# 12 個特徵的定義表
features = [
    # (名稱, 公式, 意義, 類別)
    ("normalized_spread",     "(Ask₁ − Bid₁) / tick",                      "買賣價差（成本代理）",          "基礎微結構"),
    ("micro_return",          "(mid_t − mid_{t−1}) / tick",                 "事件級中間價變動（動能）",      "基礎微結構"),
    ("obi",                   "(ΣV_bid − ΣV_ask) / (ΣV_bid + ΣV_ask) L1-5","全深度委託簿失衡",             "基礎微結構"),
    ("weighted_mid_deviation","(VWAP_mid − mid) / tick",                    "量加權中間價偏離",              "基礎微結構"),
    ("deep_obi",              "同 OBI 但只用 L2-5",                          "深層委託簿失衡（背後力量）",    "基礎微結構"),
    ("queue_imbalance",       "(V_bid1 − V_ask1) / (V_bid1 + V_ask1)",      "最佳一檔即時壓力",             "執行感知"),
    ("queue_imbalance_change","queue_imbalance_t − queue_imbalance_{t−1}",   "委託壓力方向變化",             "執行感知"),
    ("aggressor_imbalance",   "Σ(sign_i·V_i) / ΣV_i  (區間成交)",           "淨主動方向 (interval OFI)",    "執行感知"),
    ("signed_flow_ratio",     "滾動 N 筆淨成交流向比率",                     "近期 Order Flow Imbalance",    "執行感知"),
    ("trade_intensity_short", "近期 N quote 事件內成交數量",                  "短期市場活躍度",               "執行感知"),
    ("net_cancel_pressure",   "最佳一檔掛單量減少量代理值",                   "撤單壓力（流動性消失）",        "執行感知"),
    ("cancel_pressure_short", "更短視窗的撤單壓力滾動值",                     "即時流動性不穩定訊號",          "執行感知"),
]

category_colors = {
    "基礎微結構": "#0056b3",  # 深藍
    "執行感知":  "#d97706",  # 深橘
}

fig, ax = plt.subplots(figsize=(18, 7))
ax.axis("off")
fig.patch.set_facecolor("#FFFFFF")

col_widths = [0.15, 0.22, 0.28, 0.35]
col_labels = ["名稱", "公式（簡化版）", "物理意義", "類別"]
col_x = [0.01, 0.16, 0.38, 0.66]

header_y = 0.96
row_height = 0.073

# Header
for i, (label, x) in enumerate(zip(col_labels, col_x)):
    ax.text(x, header_y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="#000000",
            va="top", ha="left")

# Separator line
ax.axhline(y=header_y - 0.035, color="#CCCCCC", linewidth=1, xmin=0, xmax=1)

# Rows
for row_idx, (name, formula, meaning, category) in enumerate(features):
    y = header_y - 0.05 - row_idx * row_height
    bg_col = "#F8F9FA" if row_idx % 2 == 0 else "#FFFFFF"
    rect = mpatches.FancyBboxPatch((0, y - row_height + 0.005), 1, row_height - 0.004,
                                    boxstyle="round,pad=0",
                                    linewidth=0.5, edgecolor="#E0E0E0", facecolor=bg_col,
                                    transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)

    cat_color = category_colors.get(category, "#555555")
    ax.text(col_x[0], y - 0.005, f"  {name}", transform=ax.transAxes,
            fontsize=8.5, color="#000000", va="top", ha="left", fontfamily="monospace")
    ax.text(col_x[1], y - 0.005, formula, transform=ax.transAxes,
            fontsize=8, color="#333333", va="top", ha="left")
    ax.text(col_x[2], y - 0.005, meaning, transform=ax.transAxes,
            fontsize=8.5, color="#333333", va="top", ha="left")
    ax.text(col_x[3] + 0.01, y - 0.005, category, transform=ax.transAxes,
            fontsize=9, color=cat_color, va="top", ha="left", fontweight="bold")

# Legend
patches = [mpatches.Patch(color=c, label=l) for l, c in category_colors.items()]
ax.legend(handles=patches, loc="lower right", framealpha=0.9,
          labelcolor="#000000", facecolor="#FFFFFF", edgecolor="#CCCCCC", fontsize=10)

ax.set_title("12 個輸入特徵定義總表（execution-aware feature set）",
             fontsize=14, fontweight="bold", color="#000000", pad=12)

plt.tight_layout()
out_path = OUT_DIR / "fig_feature_table.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
